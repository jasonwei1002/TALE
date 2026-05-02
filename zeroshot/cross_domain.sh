#!/usr/bin/env bash
# Run MERL Table 2 zero-shot row across all six (source -> target) cells.
#
# Usage:
#   bash cross_domain.sh [CKPT_PATH] [GPU_ID]
#   CKPT_PATH defaults to ../checkpoints/best/vit_small_bestZeroShotAll_ckpt.pth
#   GPU_ID    defaults to 0
#
# Example:
#   bash cross_domain.sh ../checkpoints/sweep1/run_a/vit_small_bestZeroShotAll_ckpt.pth 1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CKPT="${1:-${SCRIPT_DIR}/../checkpoints/best/vit_small_bestZeroShotAll_ckpt.pth}"
GPU="${2:-0}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=true

echo "[run] checkpoint = ${CKPT}"
echo "[run] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"

# Override the default checkpoint path inside test_cross_domain.py at runtime
# without editing the file: pass via env var, the script can pick it up if set.
TALE_DOMAIN_CKPT="${CKPT}" python test_cross_domain.py
