# Lab 3 report notes (Interspeech 2026 template)

Final numbers from the **full Vast.ai 3090 run** (papermill, 61/61 cells, **0 errors**).
Headline: once scoring was made honest, a 4-bit Qwen2.5-7B gives ~49% EM on TriviaQA and
67% on CoQA; a larger Whisper fixes the Portuguese clip; and a newer 12B (Gemma-4) does
**not** beat the 7B on closed-book recall.

---

## 1. ASR (Whisper) — model comparison
Per-recording explicit language (rec1=en, rec2=pt). rec2 is a deliberately hard Portuguese clip.

| Model | rec1 (en) | rec2 (pt) | overall |
|---|---|---|---|
| whisper-small | 0.0% | 100.0% | 50.0% |
| **whisper-large-v3** | **0.0%** | **0.0%** | **0.0%** |
| distil-large-v3 (English-only) | 0.0% | 66.7% | 33.3% |

**Finding:** the original 50% was an unset-language bug on rec1 (now 0%); the *remaining* failure
was whisper-small mis-hearing Portuguese — **large-v3 transcribes it perfectly (0% overall)**.
distil-large-v3 is English-only so it fails the Portuguese clip. LLM post-correction deliberately
not used for WER (it diverges from the literal reference).

## 2. TriviaQA QA
**Task 2 (20 ex)** — EM / mean token-F1 (SQuAD-normalized + alias + token-containment matcher):

| Model | EM | F1 |
|---|---|---|
| GPT-2 | 2/20 | 0.08 |
| SmolLM2-135M | 2/20 | 0.02 |
| TinyLlama-1.1B | 7/20 | 0.07 |
| Qwen3.5-0.8B | 3/20 | 0.13 |
| DeepSeek-R1-1.5B | 0/20 | 0.00 |
| **Qwen2.5-7B-Instruct** | **9/20** | **0.39** |
| Gemma-4-12B-it (experiment) | 6/20 | 0.21 |

> **Gemma-4-12B lost to Qwen-7B** on closed-book recall (6 vs 9). Inspecting its answers, the
> misses are genuine factual hallucinations (e.g. "Foreigner" for Richard Marx, "Jim Strain" for
> Rudolf Hess), not formatting. Task-fit explanation: closed-book trivia is pure memorized recall,
> where Gemma-4's tool-use/reasoning/multimodal strengths don't apply and Qwen2.5's dense knowledge
> shines. F1 is the honest discriminator (0.39 vs ≤0.21). API frontier models intentionally excluded.

**Task 3 — strategies (Qwen2.5-7B-Instruct), EM/20:**
S1-ZeroShot 9 · S2-OneShot 9 · **S3-CoT 10** · S4-RAG 9 · S2+S4 9 · S3+S4 10. **Best: S3-CoT (10/20)** —
chain-of-thought added one over zero-shot. The Task-2→Task-3 API regression is gone.

**Task 4 — 500 examples (Qwen2.5-7B-Instruct, S3-CoT):**
- **Exact-match 247/500 = 49.4%**, **Token-F1 0.456**, TER **203.92**.
- ⚠️ TER is high because the auto-selected best strategy (S3-CoT) is **verbose**, and TER penalizes
  length vs the 1-word references. Report **EM/F1 as the accuracy metric**; the zero-shot run gives a
  clean TER ≈ 76 (CoT trades TER for a small EM gain). Lead with EM 49.4% / F1 0.456.

## 3. TTS
SpeechT5 / Bark-small / MMS-TTS / **CSM-1B** all ran (4 working synthesizers; Bark & CSM are
modern/expressive). **VibeVoice failed** (`No module named 'vibevoice'` — the community-package
install didn't register the module). MisoTTS not attempted (offline-only). The working four are a
sufficient TTS comparison.

## 4. Main problem — turn-based spoken CoQA (one story, 12 turns)
Whisper(en) → Qwen2.5-7B (passage+history) → TTS, on our own recordings.

| Variant | EM | F1 | TER |
|---|---|---|---|
| Base 7B — full pipeline (ASR'd Q) | **8/12 (66.7%)** | **0.760** | **42.86** |
| Base 7B — gold-question eval | 8/12 | 0.705 | 53.57 |
| CoQA-FT (QLoRA) | pending | pending | pending |

Clean grounded answers (e.g. `White`/`white`, `No`/`no`, `Licked her face`/`licked her face`).

## 5. SLUE-SQA-5 (spoken QA, 10 ex)
EM **3/10**, F1 **0.183**, TER **175.0** — SLUE is genuinely hard (spoken questions, odd span refs);
EM/F1 are the meaningful numbers.

## Key takeaways (discussion)
1. **Honest metrics first** — biggest gains came from fixing scoring/extraction, not models.
2. **Right tool per task** — Qwen-7B (dense knowledge) > Gemma-4-12B on closed-book recall; bigger ≠ better here.
3. **Bigger ASR where it counts** — large-v3 fixes the Portuguese clip (0% WER) that small couldn't.
4. **Metric ≠ goal** — CoT raises EM but inflates TER (verbosity); TER is length-sensitive.

## Still to do
- [ ] CoQA QLoRA fine-tune (base-vs-FT row) — run finetune_coqa_qlora.py, load coqa_lora/.
- [ ] (optional) re-report TER-500 with zero-shot for a clean ~76 number alongside the CoT EM.
- [ ] (optional) Gemma + RAG / larger sample — fair test of its tool-use strength (closed-book is its worst case).
- [x] ASR model comparison (large-v3 = 0% overall) — DONE.
- [x] VibeVoice — community install failed; dropped. MisoTTS — offline-only, dropped.

---
## Additional experiments (run 2, Vast 3090) — saved as results_*.json

### LLM-as-judge (Qwen-7B, judged WITH reference) over the 500 TriviaQA answers
- EM 247/500 (49.4%) · **Judge accuracy 290/500 (58.0%)**
- Cross-tab: clean-correct 246 · **wording/alias artifact 44** · genuine error 209 · judge-slip 1
- Of 253 EM-misses, **44 are actually correct (wording/alias)**, 209 genuinely wrong → **~17% of "failures" are metric artifacts**. Honest semantic accuracy ≈ **58%** vs strict EM 49.4% (+8.6 pts). This *quantifies* the measurement-vs-model thesis.

### Few-shot curve (Qwen-7B, 20 ex) — EM / F1
- 0-shot 10/20 (0.44) · 1-shot 10/20 (0.47) · 3-shot 9/20 (0.42) · 5-shot 9/20 (0.42)
- **In-context examples do NOT help** this instruct model on TriviaQA (0–1 best; more shots slightly hurt) — it already knows the short-answer format.

### DeepSeek-R1-1.5B fair re-test (512 tokens, `<think>` stripped)
- EM **2/20** (was 0/20 only due to 64-token truncation). **Reasoning room barely helps (0→2)** — closed-book trivia is a *knowledge* gap, not a token-budget one.

### TTS round-trip WER (text → TTS → Whisper) — intelligibility
| Synthesizer | via large-v3 | via small |
|---|---|---|
| **SpeechT5** | **8.3%** | 10.4% |
| **MMS-TTS** | **8.3%** | 16.7% |
| Bark-small | 14.6% | 12.5% |
| CSM-1B | 100% | 100% (unintelligible — round-trip caught it) |

### End-to-end latency (per turn)
- ASR 2.0 s · **LLM 0.56 s** · **TTS 7.4 s** · total **≈10 s/turn** → batch-only. TTS (CPU) is the bottleneck; the 4-bit 7B LLM is fast.

### CoQA over 5 stories (76 turns, gold Q, history reset per story)
- **EM 47/76 (61.8%)**, F1 0.64, TER 82.66 (per-story: 8/12, 8/11, 9/15, 12/20, 10/18) — more robust than the single-story 66.7%.

(ASR this run: large-v3 overall **16.7%** — rec2 one word off vs 0% last run; distil 50%, English-only fails Portuguese.)
