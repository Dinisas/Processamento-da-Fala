#!/usr/bin/env python
"""(Bonus) Render a few answer clips with MisoTTS, OFFLINE on a GPU box.

MisoTTS (MisoLabs/MisoTTS, ~8B, float32 ≈ 20 GB) does not fit a Colab T4, so we
render samples on the RTX 5090 and load the resulting WAVs in the notebook's bonus
cell. MisoTTS is brand new (June 2026) with no published benchmarks — treat this as
an experimental comparison, not a load-bearing part of the pipeline.

NOTE: the MisoTTS inference API is new and may change. This follows the documented
usage (github.com/MisoLabsAI/MisoTTS); adjust the import/call if the repo differs.

Setup:
    git clone https://github.com/MisoLabsAI/MisoTTS && cd MisoTTS
    pip install -e .            # or: uv sync
Then from this folder:
    python gen_misotts_samples.py
"""
import os
import numpy as np
import soundfile as sf

OUT_DIR = "misotts_samples"
SAMPLE_ANSWERS = [
    ("q1_white", "White."),
    ("q2_barn", "In a barn near a farmhouse."),
    ("q3_kilimanjaro", "Mount Kilimanjaro."),
    ("q4_richard_marx", "Richard Marx."),
]


def load_generator():
    """Load MisoTTS. Tries the documented helper, then a couple of fallbacks."""
    try:
        from generator import load_miso_8b          # documented entry point
        return load_miso_8b(device="cuda")
    except Exception as e:
        print(f"load_miso_8b failed ({e}); trying transformers pipeline...")
        from transformers import pipeline
        return pipeline("text-to-speech", model="MisoLabs/MisoTTS",
                        trust_remote_code=True, device=0)


def synth(gen, text):
    """Call whichever API the loaded object exposes; return (audio, sr)."""
    if hasattr(gen, "generate"):                     # native MisoTTS generator
        audio = gen.generate(text=text, speaker=0, context=[])
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        sr = getattr(gen, "sample_rate", 24000)
        return audio, sr
    out = gen(text)                                  # transformers pipeline
    return np.asarray(out["audio"], dtype=np.float32).squeeze(), out.get("sampling_rate", 24000)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gen = load_generator()
    for name, text in SAMPLE_ANSWERS:
        audio, sr = synth(gen, text)
        path = os.path.join(OUT_DIR, f"{name}.wav")
        sf.write(path, audio, sr)
        print(f"wrote {path}  ({len(audio) / sr:.1f}s @ {sr} Hz)")
    print(f"\nDone. Copy ./{OUT_DIR} next to the notebook for the bonus cell.")


if __name__ == "__main__":
    main()
