#!/usr/bin/env python
"""Local ASR eval on the asr_recordings/ clips (Apple MPS).

NOTE: the English recordings are shifted +1 vs the notebook's EN_SENTENCES list
(en_00.wav actually says the en_01 sentence; ref en_00 "who is the CEO" has no audio).
PT is aligned. So for EN we score BOTH as-labeled and shifted(+1) to expose the
mismatch and show the true WER once aligned. Saves transcripts + WER to JSON.
"""
import json, re, string
from pathlib import Path
import torch, soundfile as sf, librosa
from transformers import AutoProcessor, WhisperForConditionalGeneration
from evaluate import load as load_metric

NB = "Speech_Processing_26_27_Lab_3.ipynb"
REC = Path("asr_recordings"); SR = 16000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = ["openai/whisper-small", "openai/whisper-large-v3", "distil-whisper/distil-large-v3"]
_p = str.maketrans("", "", string.punctuation)
norm = lambda s: " ".join(s.lower().translate(_p).split())

def refs(name):
    src = "".join("".join(c["source"]) for c in json.load(open(NB))["cells"])
    blk = re.search(name + r"\s*=\s*\[(.*?)\]", src, re.S).group(1)
    return re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', blk)

EN, PT = refs("EN_SENTENCES"), refs("PT_SENTENCES")
en_w = [s for s, _ in EN if (REC / f"{s}.wav").exists()]
pt_w = [s for s, _ in PT if (REC / f"{s}.wav").exists()]
en_ref = {s: r for s, r in EN}; pt_ref = {s: r for s, r in PT}
en_txt = [r for _, r in EN]   # ordered EN reference texts (for the +1 shift)
audio = {s: (lambda a, sr: librosa.resample(a, orig_sr=sr, target_sr=SR) if sr != SR else a)(*sf.read(str(REC / f"{s}.wav")))
         for s in en_w + pt_w}
print(f"Device {DEVICE} | EN {len(en_w)}  PT {len(pt_w)}\n")

wer = load_metric("wer"); out = {}
for mid in MODELS:
    print(f"=== {mid} ===", flush=True)
    proc = AutoProcessor.from_pretrained(mid)
    mdl = WhisperForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float32).to(DEVICE).eval()
    tr = {}
    def transcribe(slug, lang):
        f = proc(audio=audio[slug].astype("float32"), sampling_rate=SR, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = mdl.generate(**f, language=lang, task="transcribe")
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    for s in en_w:
        tr[s] = transcribe(s, "en")
    for s in pt_w:
        tr[s] = transcribe(s, "en" if "distil" in mid else "pt")
    # EN as-labeled (wav slug -> same slug ref)
    en_lab = wer.compute(predictions=[norm(tr[s]) for s in en_w], references=[norm(en_ref[s]) for s in en_w])
    # EN shifted +1 (wav en_0i -> en_txt[i+1])
    en_shift = wer.compute(predictions=[norm(tr[en_w[i]]) for i in range(len(en_w))],
                           references=[norm(en_txt[i + 1]) for i in range(len(en_w))])
    pt = wer.compute(predictions=[norm(tr[s]) for s in pt_w], references=[norm(pt_ref[s]) for s in pt_w])
    out[mid] = {"en_as_labeled": round(en_lab, 4), "en_shift+1": round(en_shift, 4),
                "pt": round(pt, 4), "transcripts": tr}
    print(f"  EN as-labeled {en_lab*100:5.1f}%  |  EN shift+1 {en_shift*100:5.1f}%  |  PT {pt*100:5.1f}%\n")
    del mdl
    if DEVICE == "mps":
        torch.mps.empty_cache()

print(f"{'Model':<34}{'EN(lab)':>9}{'EN(+1)':>9}{'PT':>8}")
print("-" * 60)
for m, r in out.items():
    print(f"{m:<34}{r['en_as_labeled']*100:>8.1f}%{r['en_shift+1']*100:>8.1f}%{r['pt']*100:>7.1f}%")
json.dump(out, open("results_asr_local.json", "w"), ensure_ascii=False, indent=1)
print("\nsaved results_asr_local.json  (EN(+1) = corrected alignment = true WER)")
