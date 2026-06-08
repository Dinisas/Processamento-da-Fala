#!/usr/bin/env python
"""QLoRA fine-tuning on CoQA for the Lab-3 conversational-QA pipeline.

Run this OFFLINE on a GPU box (e.g. an RTX 5090 / A100) — NOT inside the Colab
notebook. It produces a LoRA adapter in ./coqa_lora that the notebook loads on top
of the same 4-bit base model.

Why fine-tune on CoQA (and not TriviaQA)? CoQA answers are grounded in the provided
passage, so SFT teaches the model to emit short, passage-grounded spans — a real,
low-risk gain. Closed-book TriviaQA, by contrast, is knowledge-bound: SFT there mostly
teaches answer *format* and can increase hallucination (Gekhman et al., EMNLP 2024).

The prompt template here MUST stay identical to the notebook's inference template
(CONV_SYSTEM + build_coqa_prompt), otherwise the adapter and inference diverge.

Example:
    python finetune_coqa_qlora.py --model_id Qwen/Qwen2.5-7B-Instruct \
        --out coqa_lora --max_stories 3000 --epochs 1

    # 12B on a 24GB+ GPU (gated — run `huggingface-cli login` first):
    python finetune_coqa_qlora.py --model_id google/gemma-3-12b-it --out coqa_lora_gemma

Tested with: transformers>=4.44, peft>=0.12, trl>=0.13, bitsandbytes>=0.43, accelerate.
"""
import argparse

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

# ── Prompt template — keep identical to the notebook (mp-cell3) ──────────────
CONV_SYSTEM = (
    "You are a helpful conversational QA assistant. "
    "Answer questions based only on the provided passage. "
    "Keep answers brief: one short sentence or a few words only."
)


def build_coqa_prompt(passage, history, question):
    """Passage + conversation history + current question as a single user message."""
    hist_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
    body = f"Passage:\n{passage[:1200]}"
    if hist_text:
        body += f"\n\nConversation so far:\n{hist_text}"
    body += f"\n\nCurrent question: {question}\nAnswer:"
    return body


def build_examples(split, max_stories):
    """One training example per CoQA turn, with gold conversation history (teacher forcing)."""
    ds = load_dataset("stanfordnlp/coqa", split=split)
    rows = []
    for i, story in enumerate(ds):
        if max_stories and i >= max_stories:
            break
        passage = story["story"]
        questions = story["questions"]
        answers = story["answers"]["input_text"]
        history = []
        for q, a in zip(questions, answers):
            rows.append({
                "system": CONV_SYSTEM,
                "user": build_coqa_prompt(passage, history, q),
                "assistant": str(a),
            })
            history.append((q, a))
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Base model — MUST match BIG_MODEL_ID in the notebook.")
    p.add_argument("--out", default="coqa_lora", help="Adapter output dir.")
    p.add_argument("--max_stories", type=int, default=3000,
                   help="Cap CoQA stories for speed (0 = all ~7.2k).")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "QLoRA needs a CUDA GPU."

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    rows = build_examples("train", args.max_stories)
    print(f"Built {len(rows)} CoQA turn-examples from {args.max_stories or 'all'} stories.")
    ds = Dataset.from_list(rows)

    def add_text(example):
        msgs = [{"role": "system", "content": example["system"]},
                {"role": "user",   "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]}]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    ds = ds.map(add_text)

    from trl import SFTConfig, SFTTrainer
    cfg = SFTConfig(
        output_dir=args.out, per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, bf16=True, logging_steps=20, save_strategy="epoch",
        lr_scheduler_type="cosine", warmup_ratio=0.03, optim="paged_adamw_8bit",
        max_seq_length=args.max_seq_len, packing=False, dataset_text_field="text",
        report_to="none",
    )
    # processing_class (trl>=0.13) with a fallback to the older tokenizer= kwarg.
    try:
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                             peft_config=lora, processing_class=tok)
    except TypeError:
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                             peft_config=lora, tokenizer=tok)

    trainer.train()
    trainer.save_model(args.out)     # saves the LoRA adapter
    tok.save_pretrained(args.out)
    print(f"\nSaved CoQA LoRA adapter to ./{args.out}")
    print("Copy that folder next to the notebook (or mount it) and run the adapter-load cell.")


if __name__ == "__main__":
    main()
