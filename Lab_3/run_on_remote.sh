#!/usr/bin/env bash
# Bootstrap for a rented GPU box (e.g. Vast.ai 3090, ~50 GB disk).
# Usage (on the box, inside the Lab_3 folder you scp'd over):
#   bash run_on_remote.sh                 # fine-tune CoQA QLoRA (default)
#   RUN_NOTEBOOK=1 bash run_on_remote.sh  # also execute the notebook headless (disk-heavy!)
# Pass extra args straight to the trainer, e.g.:
#   bash run_on_remote.sh --max_stories 2000 --epochs 1
set -euo pipefail

# Keep the HF model cache on this volume and watch disk (the 7B download is ~15 GB).
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

echo "=== GPU ===";  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
echo "=== Disk ==="; df -h "$PWD" | tail -1
FREE_GB=$(df -PBG "$PWD" | awk 'NR==2{gsub("G","",$4); print $4}')
echo "Free: ${FREE_GB} GB   (HF_HOME=$HF_HOME)"
[ "${FREE_GB:-0}" -lt 25 ] && echo "WARNING: <25 GB free — the 7B base alone needs ~15 GB. Trim caches first."

echo "=== Installing deps (torch/CUDA assumed present on the image) ==="
pip install -q -U "transformers>=4.44" "peft>=0.12" "trl>=0.13" "bitsandbytes>=0.43" \
                  accelerate "datasets<3.0.0" sentencepiece hf_transfer

echo "=== QLoRA fine-tune on CoQA ==="
python finetune_coqa_qlora.py --model_id Qwen/Qwen2.5-7B-Instruct --out coqa_lora "$@"

if [ "${RUN_NOTEBOOK:-0}" = "1" ]; then
  echo "=== Executing notebook headless (downloads the full model zoo — needs lots of disk) ==="
  pip install -q -U jupyter nbconvert evaluate jiwer librosa soundfile wikipedia-api
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=3600 Speech_Processing_26_27_Lab_3.ipynb
fi

echo
echo "DONE. Adapter is in ./coqa_lora — pull it back to your Mac with:"
echo "  scp -P 17070 -r root@91.150.160.38:~/Lab_3/coqa_lora ./Lab_3/"
du -sh coqa_lora 2>/dev/null || true
