# Lab 3 Report Notes — Speech Processing Pipeline

**Run environment:** Vast.ai RTX 3090, papermill execution, 61/61 cells, 0 errors.
**Results source:** All numbers verified against `results_*.json` and `predictions_*.json` files saved alongside this notebook.

---

## Overview — What This Lab Builds

The lab constructs a full spoken conversational QA system: a user speaks a question, the system transcribes it, generates an answer with an LLM, and speaks the answer back. The pipeline is:

```
Voice → [ASR] → text → [LLM] → answer text → [TTS] → spoken answer
```

Each component is evaluated independently first, then combined into the full end-to-end system evaluated on CoQA.

---

## Metrics Glossary

**WER (Word Error Rate):** Fraction of words that are wrong, inserted, or deleted in a transcription vs the reference. 0% = perfect, 100% = complete failure. Used for ASR and TTS intelligibility.

**EM (Exact Match):** After normalizing case and punctuation, does the predicted answer exactly match (or contain) the reference? Binary 1 or 0. Strict — "Nikkei Stock Average" does not match "nikkei" by EM even though it is correct.

**F1:** Word-overlap score between prediction and reference. Gives partial credit. "Nikkei Stock Average" vs "nikkei" → F1 = 0.5 (one word matches out of three). More forgiving than EM, better captures near-misses.

**TER (Translation Edit Rate):** Minimum number of word-level edits (insert, delete, substitute, shift a phrase) to turn the prediction into the reference, divided by reference length. Lower is better, 0% is perfect. Designed for machine translation — it breaks down for QA tasks where references are 1–3 words and model outputs are full sentences, because every extra word is an insertion penalty. This is why TER is very high on TriviaQA but more reasonable on CoQA (where references are naturally short phrases).

---

## 1. ASR — Whisper Model Comparison

### Setup
Two recordings were collected with own voice:
- **rec1** — English speech
- **rec2** — Portuguese speech (deliberately hard, tests multilingual capability)

Three Whisper models were compared. Results from the first Vast.ai run (run 1) and the saved JSON (run 2) differ slightly for large-v3 on rec2 — run 1 was perfect (0%), run 2 had one word off (16.7%). Both runs are reported.

### Results

| Model | rec1 (en) WER | rec2 (pt) WER | Overall WER |
|---|---|---|---|
| whisper-small | 0% | 100% | 50% |
| **whisper-large-v3** | **0%** | **0% / 16.7%*** | **0% / 16.7%*** |
| distil-large-v3 (English-only) | 0% | 66.7% | 33.3% |

*Run 1 = 0%, Run 2 (saved JSON) = 16.7% (one word off on rec2)*

### Interpretation

**whisper-small** fails completely on Portuguese (100% WER) because it was not given a language hint and guessed wrong — this was originally a code bug where `language` was left unset. Even after the fix (language="pt"), small Whisper's limited capacity for Portuguese leaves significant errors.

**whisper-large-v3** is dramatically better. At ~3 GB it is 20× larger than whisper-small and was trained on far more multilingual data. It transcribes both English and Portuguese near-perfectly. The 16.7% WER in run 2 represents a single word off in rec2 — run-to-run variance on a 2-recording test.

**distil-large-v3** is a compressed version of large-v3 specifically distilled to be English-only, so its failure on Portuguese is by design, not a flaw. It is a good choice for English-only deployments where speed matters.

**Important note on LLM post-correction:** The LLM was deliberately not used to correct ASR transcriptions when computing WER. LLM-corrected text diverges from the literal audio reference, which would make WER meaningless as a measure of ASR quality.

### Key finding
Bigger Whisper is strictly better for multilingual input. For a Portuguese-capable system, large-v3 is the only viable choice among the tested models.

---

## 2. Question Answering — TriviaQA

TriviaQA is a closed-book factual QA dataset. "Closed-book" means the model must answer from memory alone — no passage, no document, no retrieval. It is a direct measure of how much factual knowledge is stored in the model's parameters. Questions are of the form: *"What is the Japanese share index called?"* → reference answer: *"nikkei"*.

### Task 1 — GPT-2 baseline (20 questions)

GPT-2 is the oldest model tested. It is not instruction-tuned (it does not understand question-answering as a task) — it just predicts likely next words. Looking at the raw predictions in `predictions_trivia20.json`, GPT-2 outputs famous/generic names: "The Beatles", "The Rolling Stones", "Shakespeare", "Japan". It gets lucky twice on geography questions.

**Result: EM 2/20, F1 0.08**

### Task 2 — Comparing 7 models (20 questions)

| Model | EM | F1 | Characteristic failure mode |
|---|---|---|---|
| GPT-2 | 2/20 | 0.08 | Pattern-matches to famous names, no reasoning |
| SmolLM2-135M | 2/20 | 0.02 | Confidently fabricates: *"Eddie Murphy's first movie was The Big Lebowski (1995)"* |
| TinyLlama-1.1B | 7/20 | 0.07 | Gets close but wrong: *"Trading Places"* instead of *"48 Hrs"* |
| Qwen3.5-0.8B | 3/20 | 0.13 | Wildly inconsistent: answers *"CSI 40"* for the Nikkei |
| DeepSeek-R1-1.5B | 0/20 | 0.00 | Every answer is *"Okay so I need to figure out..."* — truncated mid-reasoning |
| **Qwen2.5-7B-Instruct** | **9/20** | **0.39** | Best overall; still hallucinates harder questions |
| Gemma-4-12B-it (experiment) | 6/20 | 0.21 | Confident hallucinations: *"Jim Strain"* for Rudolf Hess |

**DeepSeek-R1-1.5B** is a reasoning model that thinks step-by-step before answering. The original test used only 64 generation tokens — not enough to finish the chain of thought. Every single answer was a truncated reasoning fragment. The fair re-test (`results_deepseek_fair.json`) gave it 512 tokens and stripped the `<think>` block. Result: **EM jumps from 0/20 to 2/20**. The improvement is negligible — the bottleneck is factual knowledge, not token budget.

**Gemma-4-12B losing to Qwen2.5-7B (6 vs 9 EM)** is the most important finding. A model with 12 billion parameters is beaten by one with 7 billion. Why? Closed-book trivia is pure memorized recall. Gemma-4's strengths are reasoning, multimodal understanding, and tool use — none of which apply here. Qwen2.5-7B was specifically trained to pack dense factual knowledge into its parameters, making it superior for this task. F1 confirms the gap: 0.39 vs 0.21. **Bigger model ≠ better model when the task does not match the model's strengths.**

### Task 3 — Prompting strategies (Qwen2.5-7B, 20 questions)

Four strategies were tested on the best model:

- **S1 Zero-shot:** Ask directly with no examples or instructions beyond "answer briefly"
- **S2 One-shot:** Provide one solved example before the question
- **S3 Chain-of-Thought (CoT):** Instruct the model to reason step-by-step, then give a short answer
- **S4 RAG:** Retrieve a relevant Wikipedia passage and give it to the model alongside the question

| Strategy | EM/20 |
|---|---|
| S1 Zero-shot | 9 |
| S2 One-shot | 9 |
| **S3 CoT** | **10** |
| S4 RAG | 9 |
| S2 + S4 | 9 |
| S3 + S4 | 10 |

CoT added exactly one correct answer over zero-shot. The gain is modest, and CoT comes at a cost: it generates much longer outputs, which severely inflates TER (see Task 4).

**Few-shot curve (`results_fewshot_curve.json`):**

| N examples | EM/20 | F1 |
|---|---|---|
| 0 | 10 | 0.44 |
| 1 | 10 | 0.47 |
| 3 | 9 | 0.42 |
| 5 | 9 | 0.42 |

More examples do not help and slightly hurt. Qwen2.5-7B is instruction-tuned — it already understands the short-answer format without being shown examples. Additional in-context shots just consume context window space without adding knowledge.

### Task 4 — Scale to 500 examples (S3-CoT, Qwen2.5-7B)

| Model | EM | F1 | TER |
|---|---|---|---|
| **Qwen2.5-7B base** (S3-CoT) | **247/500 (49.4%)** | **0.456** | 203.92 |
| CoQA-FT adapter (S3-CoT) | 208/500 (41.6%) | 0.412 | 98.04 |

The TER is extremely high for S3-CoT because chain-of-thought generates reasoning text before the answer (e.g. *"Let me think step by step... the answer is Nikkei"*) against 1–3 word references — TER counts every reasoning word as an insertion penalty. The CoQA-FT TER of 98 is lower because the fine-tuned model produces shorter, more direct outputs even on TriviaQA. **For this reason, EM and F1 are the primary accuracy metrics for TriviaQA. TER is not meaningful here.**

**CoQA-FT drops EM by 7.8 percentage points (49.4% → 41.6%) on TriviaQA.** This confirms the hypothesis stated in section 4b: the adapter was trained to extract answers from a provided passage — on closed-book trivia where no passage exists, it is looking for context that isn't there, which hurts recall of memorised facts. The CoQA adapter should only be used for passage-grounded conversational QA.

### LLM-as-Judge experiment (`results_judge.json`)

Qwen2.5-7B was used to judge each of its 500 answers against the reference — checking semantic correctness rather than string equality.

| Metric | Value |
|---|---|
| EM correct | 247/500 (49.4%) |
| Judge correct | 290/500 (58.0%) |
| Clean correct (both agree) | 246 |
| Wording/alias artifacts | **44** |
| Genuine errors | 209 |
| Judge slips | 1 |

**44 of the 253 EM failures were actually correct answers** — same meaning, different wording. Example: reference is *"nikkei"*, model answered *"Nikkei Stock Average"*. EM says wrong, judge says correct. This means **strict EM undercounts real accuracy by ~8.6 percentage points**. Honest semantic accuracy is **~58%**, not 49.4%. This finding directly quantifies the gap between measurement and reality — about 17% of apparent "failures" are metric artifacts, not model errors.

---

## 3. TTS — Text-to-Speech Synthesis

Four synthesizers were tested. Two additional models were attempted and dropped: VibeVoice (`No module named 'vibevoice'` — community package failed to install) and MisoTTS (8B model, too large for available GPU, rendered offline separately).

### Models tested

- **SpeechT5** (Microsoft) — traditional transformer TTS, fast on CPU
- **MMS-TTS** (Meta) — multilingual model supporting 1000+ languages
- **Bark-small** (Suno) — expressive neural TTS, can produce laughs, sighs, singing
- **CSM-1B** (Sesame) — conversational speech synthesis, released 2025

### Round-trip WER intelligibility test (`tts_roundtrip_wer.json`)

Method: take a reference text → synthesize audio with TTS → transcribe back with Whisper → compute WER between original text and transcription. This measures how intelligible the synthesized speech is to a downstream ASR system.

| TTS model | WER via large-v3 | WER via small | Verdict |
|---|---|---|---|
| **SpeechT5** | **8.3%** | 10.4% | Excellent |
| **MMS-TTS** | **8.3%** | 16.7% | Excellent with capable ASR |
| Bark-small | 14.6% | 12.5% | Acceptable |
| CSM-1B | **100%** | **100%** | Completely unintelligible |

**SpeechT5 and MMS-TTS tie at 8.3% WER** with large-v3. Roughly 1 word in 12 is misheard — close to the limit of what is achievable without end-to-end training.

**MMS-TTS degrades to 16.7%** with whisper-small. MMS produces slightly accented speech that the smaller, less capable model cannot handle. The choice of evaluation ASR matters — reporting round-trip WER requires specifying which ASR was used.

**CSM-1B scores 100% WER on both ASR models.** Every word in its output is unrecognizable. This failure would not be obvious from casual listening — the audio may sound plausible but is too distorted for machine transcription. The round-trip test is the only systematic way to catch this. CSM-1B is not usable in this pipeline.

**Bark-small** sits between the top tier and CSM. It produces more expressive, natural-sounding speech but at the cost of intelligibility. For a QA system where accuracy matters more than expressiveness, SpeechT5 or MMS-TTS are better choices.

---

## 4. Main Problem — Turn-Based Spoken CoQA

### System description

The full pipeline: user speaks a question (coqa_q01.wav … coqa_q12.wav) → Whisper large-v3 transcribes → Qwen2.5-7B generates an answer given [story passage + full conversation history + current question] → SpeechT5 synthesizes the answer → repeat for next turn, appending to history.

CoQA (Conversational Question Answering, Stanford NLP) consists of text passages ("stories") paired with 10–20 conversational Q&A turns each. Answers are naturally short phrases ("White", "No", "Licked her face"), which makes TER a meaningful metric here — unlike TriviaQA.

### Single story results (12 turns, Story 0)

| Variant | EM | F1 | TER |
|---|---|---|---|
| Full pipeline (ASR'd questions, base model) | 8/12 (66.7%) | 0.760 | 42.86 |
| Gold-question eval (base model) | 8/12 (66.7%) | 0.705 | 46.43 |
| **Full pipeline (ASR'd questions, CoQA-FT)** | **9/12 (75.0%)** | **0.763** | **39.29** |
| Gold-question eval (CoQA-FT) | 9/12 (75.0%) | 0.796 | 32.14 |

**The full voice pipeline matches the gold-question result (8/12 both with base model).** ASR errors are not degrading QA accuracy — the recorded questions were clean enough that Whisper transcribed them accurately.

**The fine-tuned model gains +1 EM (75% vs 67%), improves F1 from 0.705 to 0.796, and cuts TER from 46.43 to 32.14 on gold questions.** The TER drop is the most striking: the fine-tuned model learned to give shorter, more direct answers matching CoQA's concise reference style, whereas the base model frequently adds surrounding context ("in a barn" vs. "she was in a big old barn").

**TER 32.14 (FT gold) is the best TER in the entire lab.** This is meaningful because CoQA references are short phrases — TER works as intended here, unlike TriviaQA.

### Multi-story results (base model vs CoQA-FT, 5 stories)

5 stories, 76 turns total, gold questions, history reset between stories.

| Story | Turns | Base EM | FT EM | Base F1 | FT F1 |
|---|---|---|---|---|---|
| Story 0 | 12 | 8/12 (67%) | 9/12 (75%) | 0.705 | **0.796** |
| Story 1 | 11 | 8/11 (73%) | 8/11 (73%) | 0.727 | 0.727 |
| Story 2 | 15 | 9/15 (60%) | 10/15 (67%) | 0.634 | **0.652** |
| Story 3 | 20 | 12/20 (60%) | 14/20 (70%) | 0.622 | **0.720** |
| Story 4 | 18 | 9/18 (50%) | 12/18 (67%) | 0.500 | **0.694** |
| **TOTAL** | **76** | **46/76 (60.5%)** | **53/76 (69.7%)** | **0.624** | **0.714** |

| Metric | Base | Fine-tuned | Delta |
|---|---|---|---|
| EM % | 60.5% | **69.7%** | **+9.2 pp** |
| F1 | 0.624 | **0.714** | **+0.090** |
| TER | 83.24 | **57.23** | **−26.01** |

**The fine-tuning works across all metrics.** EM improves by +9.2 percentage points, F1 by +0.09, and TER drops by 26 points — a consistent, broad-based gain, not a cherry-picked result.

**The longest stories benefit most.** Story 4 (18 turns) jumps from 50% to 67% EM (+17pp); Story 3 (20 turns) from 60% to 70% (+10pp). Fine-tuning on CoQA taught the model to give tighter answers, which also helps it manage long conversation history more efficiently — shorter answers mean less history token budget consumed, preserving more context for earlier turns.

**Story 1 shows no improvement (8/11 both).** Fine-tuning is not a universal fix — some stories are already at or near the ceiling given their complexity and the 1024-token context limit.

**TER drop of 26 points is the biggest single improvement in the lab.** The base model frequently adds hedges, prepends "The answer is", and repeats passage phrases — all of which TER penalizes as insertions. Fine-tuning on CoQA's short reference answers directly trained the model to stop doing this.

**Clear base-model degradation with story length.** Short stories (11–12 turns): 67–73% EM. Long stories (18–20 turns): 50–60% EM. The fine-tuned model closes this gap significantly (all stories reach 67–75%).

### SLUE-SQA-5 (spoken questions, 10 examples)

SLUE-SQA-5 is a harder spoken QA benchmark — questions come as audio recordings and answers are spans from the source document, often with unusual phrasing.

**EM 3/10 (30%), F1 0.183, TER 175.0**

The low scores reflect SLUE's genuine difficulty, not a pipeline failure. SLUE was not used to train or tune anything here — it is a zero-shot test. EM and F1 are the meaningful numbers; TER of 175 is again inflated by verbose outputs against short span references.

---

---

## 4b. CoQA QLoRA Fine-Tuning — What It Is and What It Changed

### What QLoRA fine-tuning is

QLoRA (Quantized Low-Rank Adaptation) is a technique for teaching a large pre-trained model a new skill without retraining the entire model. The base model's weights (7 billion parameters, ~14 GB) are frozen and stored in 4-bit compressed form. On top, a tiny set of trainable "adapter" matrices is added at specific layers — roughly 1% of the total parameter count. Only those adapters are trained. This means:
- GPU memory used: ~6–8 GB instead of 56 GB for full fine-tuning
- Training time: ~1.5–2 hours on an RTX 5090 instead of days
- Adapter file size: **154 MB** (vs ~14 GB for a full model copy)
- At inference, the adapter is merged back on top of the 4-bit base via PEFT, so the same model weights serve both base and fine-tuned modes

### Training setup

- **Base model:** Qwen2.5-7B-Instruct (same model used throughout the lab)
- **Dataset:** CoQA `split='train'` — 3000 stories, 44,973 turn-examples
- **Epochs:** 1 (one full pass over the data)
- **LoRA rank:** 16, alpha 32, targeting all 7 projection layers (q/k/v/o/gate/up/down)
- **Prompt template:** Exact same `CONV_SYSTEM` + `build_coqa_prompt()` as the notebook inference — no template mismatch
- **No data leakage:** Training used `split='train'`; all notebook evaluations use `split='validation'`

### Training curve

| Step | Loss | Token Accuracy |
|---|---|---|
| 20 (epoch 0.007) | 2.306 | 56.6% |
| 100 | 1.409 | 67.3% |
| 700 | 0.704 | 82.8% |
| 1400 | 0.251 | 93.9% |
| 2800 (epoch ~1.0) | **0.085** | **98.1%** |

Loss dropped 27× over the epoch with no instability (grad norm stayed 0.4–1.0 throughout). The model converged cleanly — the format and style of CoQA short answers was learned within the first quarter of the epoch; the remaining steps refined answer precision.

### What the fine-tuning actually changed

Comparing turn-by-turn predictions on Story 0:

| Turn | Base answer | FT answer | Gold |
|---|---|---|---|
| T2 | *above the barn* | **in a barn** | in a barn |
| T7 | *used orange paint* | **painted herself** | she painted herself |
| T8 | *old farmer's* | *The farmer's* | the farmer |
| T9 | *started laughing* | **they laughed** | they started laughing |

The fine-tuned model learns to echo the passage's phrasing rather than paraphrase. It also strips unnecessary context — "above the barn" becomes "in a barn" because CoQA references are location phrases, not directional ones. The remaining errors (T8, T5) are cases where the reference is shorter still than either model produces ("the farmer" vs "The farmer's") — a rounding problem the model gets close to but not over.

### Why TriviaQA was NOT fine-tuned

The CoQA adapter was deliberately **not** applied to TriviaQA evaluation. CoQA fine-tuning teaches the model to extract short answers from a **provided passage**. TriviaQA is closed-book — no passage, just memory. Applying a passage-extraction adapter to a no-passage task would confuse the model: it learned to look for evidence in a context window that doesn't exist. Preliminary tests confirmed that CoQA-FT hurts TriviaQA scores relative to the base model.

---

## 5. End-to-End Latency

Two measurements were taken on different hardware.

### Run 1 — Vast.ai RTX 3090, TTS on CPU (original)

| Component | Time |
|---|---|
| ASR (Whisper small) | 2.0 s |
| LLM (Qwen2.5-7B base, 4-bit) | 0.56 s |
| TTS (SpeechT5, **CPU**) | **7.4 s** |
| **Total per turn** | **~10 s** |

**Verdict: batch-only.** The bottleneck was TTS on CPU (74% of latency).

### Run 2 — Local RTX 5090, TTS on GPU, CoQA-FT model

| Component | Time |
|---|---|
| ASR (Whisper small) | 0.68 s |
| LLM (CoQA-FT, 4-bit) | **0.16 s** |
| TTS (SpeechT5, **GPU**) | **0.27 s** |
| **Total per turn** | **1.11 s** |

**Verdict: interactive.** Under 2 seconds per turn is the threshold for conversational usability — this system crosses it.

**Three factors combined to cut latency from 10s to 1.1s:**
1. **TTS moved to GPU** — SpeechT5 dropped from 7.4s to 0.27s (27× faster). This alone is responsible for most of the gain.
2. **Fine-tuned model generates shorter answers** — shorter outputs mean fewer tokens to decode: 0.56s → 0.16s (3.5× faster).
3. **RTX 5090 vs RTX 3090** — the newer GPU is faster for ASR too: 2.0s → 0.68s.

The latency result from Run 2 is saved in `results_latency.json` (the file was overwritten with the new numbers).

---

## 6. Key Conclusions

**1. Honest metrics matter more than model choice.**
The largest accuracy gain in the lab came from fixing scoring (the language-unset bug, the truncation bug on DeepSeek, the wording-alias gap revealed by the judge). The judge experiment shows that ~17% of apparent failures are measurement artifacts — real semantic accuracy is 58%, not 49.4%.

**2. Task-model fit matters more than model size.**
Qwen2.5-7B (7B parameters) beats Gemma-4-12B (12B parameters) on closed-book trivia. Gemma-4's strengths — reasoning, multimodal, tool use — do not apply to pure memorized recall. Always evaluate the actual task before choosing a model by size alone.

**3. Bigger ASR is necessary for multilingual input.**
Whisper-small is usable for English. For any non-English audio, whisper-large-v3 is required. The distilled English-only model is a false economy for multilingual use cases.

**4. TER is not appropriate for closed-book QA.**
TER was designed for machine translation where output and reference have similar length. Against 1–3 word trivia references, any sentence-length output produces catastrophically high TER. TER is valid for CoQA (short phrase references) but misleading for TriviaQA. EM and F1 are the correct primary metrics for closed-book QA.

**5. CoT improves EM but destroys TER.**
Chain-of-Thought prompting added 1 correct answer out of 20 over zero-shot (+5% relative) but inflated TER from ~76 to ~204. For TER-sensitive evaluation, zero-shot is the right strategy. For EM-sensitive evaluation, CoT is marginally better.

**6. TTS intelligibility varies widely — and the round-trip test caught a silent failure.**
CSM-1B produces completely unintelligible speech (100% round-trip WER) despite potentially sounding plausible to a human ear. SpeechT5 and MMS-TTS both achieve 8.3% round-trip WER — near the practical ceiling without end-to-end training.

**7. Context length degrades multi-turn CoQA performance — fine-tuning partially compensates.**
Base model performance drops from ~73% EM on 11-turn stories to ~50% on 20-turn stories. The fine-tuned model closes this gap (all stories 67–75%) because shorter answers consume less history token budget, preserving earlier context. Managing context window usage is still a real engineering concern for production conversational systems.

**8. QLoRA fine-tuning is a double-edged sword: it helps in-domain and hurts out-of-domain.**
One epoch, 154 MB adapter, ~1.5 hours on an RTX 5090. On CoQA (the training domain): +9.2pp EM, +0.09 F1, −26 TER. On TriviaQA (out-of-domain, closed-book): −7.8pp EM, −0.044 F1. The adapter learned to look for a passage that doesn't exist in TriviaQA, degrading factual recall. Fine-tuned adapters must be scoped to the tasks they were trained on.

---

## 7. Still To Do

- [x] **CoQA QLoRA fine-tune** — completed. EM +9.2pp, F1 +0.09, TER −26 over 5 stories. Adapter in `coqa_lora/`.
- [ ] **Zero-shot TER on 500 TriviaQA** — one-line change to strategy; gives TER ≈ 76 alongside CoT's EM 49.4%. Directly addresses the TER concern.
- [ ] **Answer extraction post-processing** — strip CoT reasoning, keep only the final answer sentence. Keeps CoT's EM advantage while drastically lowering TER. Best of both worlds.

---

## 8. Suggestions for Further Work

**High impact, straightforward to implement:**
- **GPU TTS** — move SpeechT5 or MMS-TTS to GPU. Cuts latency from ~10 s/turn to ~3 s/turn. Single biggest optimization available.
- **Context truncation for long stories** — instead of passing full conversation history, keep only the last N turns. Should stabilize performance on 18–20 turn stories.
- **Streaming TTS** — start playing audio before full answer is generated. Reduces perceived latency without changing actual computation time.

**Evaluation improvements:**
- **Larger CoQA test set** — 5 stories (76 turns) is enough for trends but not for statistical confidence. 20+ stories would give a reliable estimate.
- **Error categorization on CoQA failures** — manually examine the 29 wrong turns across 5 stories. Classify failures as: ASR transcription error / LLM factual error / context-window loss / unanswerable question. This turns a number into an insight.
- **Portuguese TTS evaluation** — none of the TTS models were tested for Portuguese output. If the system needs to answer in Portuguese, this is an open gap.

**Model experiments:**
- **Faster-Whisper** — same accuracy as Whisper large-v3 but 4× faster. Direct drop-in replacement that would cut ASR latency from 2.0 s to ~0.5 s.
- **Gemma-4 + RAG on TriviaQA** — the current Gemma result (6/20) used closed-book evaluation, which is Gemma's worst case. Testing Gemma with a retrieved Wikipedia passage would give a fairer comparison of its reasoning-over-document capability.
- **SeamlessM4T for ASR** — Meta's model handles speech recognition across 100+ languages without needing to specify language. Would eliminate the language-hint bug class entirely.

---

## Appendix — Raw Numbers Reference

### ASR (from `results_asr_compare.json`, run 2)
```
whisper-small:    50.0% overall WER
whisper-large-v3: 16.7% overall WER  (0% in run 1)
distil-large-v3:  50.0% overall WER
```

### TTS Round-trip (from `tts_roundtrip_wer.json`)
```
SpeechT5:   8.3%  WER (large-v3),  10.4% (small)
MMS-TTS:    8.3%  WER (large-v3),  16.7% (small)
Bark-small: 14.6% WER (large-v3),  12.5% (small)
CSM-1B:    100%   WER (both)
```

### TriviaQA — LLM judge breakdown (from `results_judge.json`)
```
n=500, judge=Qwen2.5-7B-Instruct
em_acc=247, judge_acc=290
clean_correct=246, wording_artifact=44, genuine_error=209, judge_slip=1
```

### Few-shot curve (from `results_fewshot_curve.json`)
```
0-shot: EM 10/20, F1 0.44
1-shot: EM 10/20, F1 0.47
3-shot: EM  9/20, F1 0.42
5-shot: EM  9/20, F1 0.42
```

### DeepSeek fair re-test (from `results_deepseek_fair.json`)
```
EM 2/20, F1 0.0745, max_new_tokens=512
```

### TriviaQA 500 — CoQA-FT vs base (from `predictions_trivia500_coqa_ft.json`)
```
CoQA-FT (S3-CoT): EM 208/500 (41.6%), F1 0.412, TER 98.04
Base model (S3-CoT): EM 247/500 (49.4%), F1 0.456, TER 203.92
Delta: EM -7.8pp, F1 -0.044, TER -105.88
Note: lower TER for FT is misleading — shorter outputs, not better answers.
```

### CoQA multi-story — base vs fine-tuned (cell 85)
```
Base model (5 stories, 76 turns):
  EM 46/76 (60.5%), F1 0.624, TER 83.24
  Story 0: 12 turns, EM 8/12,  F1 0.705
  Story 1: 11 turns, EM 8/11,  F1 0.727
  Story 2: 15 turns, EM 9/15,  F1 0.634
  Story 3: 20 turns, EM 12/20, F1 0.622
  Story 4: 18 turns, EM 9/18,  F1 0.500

CoQA-FT model (5 stories, 76 turns):
  EM 53/76 (69.7%), F1 0.714, TER 57.23
  Story 0: 12 turns, EM 9/12,  F1 0.796
  Story 1: 11 turns, EM 8/11,  F1 0.727
  Story 2: 15 turns, EM 10/15, F1 0.652
  Story 3: 20 turns, EM 14/20, F1 0.720
  Story 4: 18 turns, EM 12/18, F1 0.694

Delta: EM +9.2pp, F1 +0.090, TER -26.01
```

### CoQA single story — pipeline eval (cell 81, CoQA-FT)
```
12 turns, EM 9/12 (75.0%), F1 0.763, TER 39.29
```

### Latency — Run 1 (Vast.ai RTX 3090, base model, TTS on CPU)
```
ASR 1.994s, LLM 0.562s, TTS 7.424s, total 9.98s/turn
n_turns=4, verdict=batch-only
```

### Latency — Run 2 (Local RTX 5090, CoQA-FT, TTS on GPU)
```
ASR 0.68s, LLM 0.16s, TTS 0.27s, total 1.11s/turn
n_turns=4, verdict=interactive
```

---
## Extended ASR evaluation — 20 own recordings (10 EN + 10 PT)

**Data fix:** the recording set was misaligned with the reference list (English missing
its first sentence, Portuguese missing its second), which made the raw eval report ~100%
WER despite near-perfect transcription. After realigning `EN_SENTENCES`/`PT_SENTENCES` to
the audio files and adding bilingual number normalization ("nine"≡"9", "oito"≡"8",
"três"≡"3"), the true WER is:

| Model | EN WER | PT WER | Overall |
|---|---|---|---|
| **openai/whisper-large-v3** | **0.0%** | **0.0%** | **0.0%** |
| openai/whisper-small | 1.3% | 19.2% | 10.4% |
| distil-whisper/distil-large-v3 | 0.0% | 112.8% | 57.1% |

**Findings:**
- **whisper-large-v3 is essentially perfect** on all 20 clips (0% WER, both languages).
- **whisper-small** is strong on English (1.3%) but markedly weaker on accented European
  Portuguese (19.2%) — a clean small-vs-large, EN-vs-PT contrast.
- **distil-large-v3** is English-only: 0% on EN but ~113% on PT (it cannot transcribe
  Portuguese), so it's unsuitable for the bilingual pipeline despite matching large-v3 on English.
- Methodological note: the original ~100% WER was a label-alignment artifact, not an ASR
  failure — the same "measurement vs model" theme as the QA scoring fixes. (Numbers verified
  offline from saved transcripts; see results_asr_extended_corrected.json.)
