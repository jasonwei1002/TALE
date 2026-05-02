#!/usr/bin/env bash
# Plot the local-alignment attention figure with TALE (full) vs.
# TALE (w/o local loss) side by side.
#
# Usage:
#   bash visualize/plot_attention_map.sh [GPU_ID]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU="${1:-0}"
CKPT_FULL="${PROJECT_ROOT}/checkpoints/best/vit_small_bestZeroShotAll_ckpt.pth"
CKPT_ABL="${PROJECT_ROOT}/checkpoints/去掉局部对比/vit_small_bestZeroShotAll_ckpt.pth"
OUTPUT="${PROJECT_ROOT}/visualize/fig-local-alignment-compare.pdf"

for f in "${CKPT_FULL}" "${CKPT_ABL}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[error] checkpoint not found: ${f}" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=true

python "${SCRIPT_DIR}/plot_attention_map.py" \
  --ckpt "${CKPT_FULL}" \
  --ckpt-ablation "${CKPT_ABL}" \
  --output "${OUTPUT}"

echo "[done] saved figure to ${OUTPUT}"
