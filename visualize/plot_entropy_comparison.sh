#!/usr/bin/env bash
# Quantitative attention-sharpness comparison: TALE (full) vs w/o local loss.
# Usage: bash visualize/plot_entropy_comparison.sh [GPU_ID] [N_SAMPLES]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU="${1:-0}"
N="${2:-500}"
CKPT_FULL="${PROJECT_ROOT}/checkpoints/best/vit_small_bestZeroShotAll_ckpt.pth"
CKPT_ABL="${PROJECT_ROOT}/checkpoints/去掉局部对比/vit_small_bestZeroShotAll_ckpt.pth"
OUTPUT="${PROJECT_ROOT}/visualize/fig-attention-entropy.png"

for f in "${CKPT_FULL}" "${CKPT_ABL}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[error] checkpoint not found: ${f}" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=true

python "${SCRIPT_DIR}/plot_entropy_comparison.py" \
  --ckpt "${CKPT_FULL}" \
  --ckpt-ablation "${CKPT_ABL}" \
  --dataset ptbxl-form \
  --max-samples "${N}" \
  --output "${OUTPUT}"

echo "[done] ${OUTPUT}"
