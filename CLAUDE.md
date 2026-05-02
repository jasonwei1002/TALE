# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TALE is an ECG–text contrastive pre-training framework built on top of [MERL](https://github.com/cheliu-computation/MERL-ICML2024) and [ECG-JEPA](https://arxiv.org/abs/2410.13867). It pre-trains a 12-lead ECG encoder against radiology-style reports using a CLIP-style multimodal objective and several proposed ablations:

- **JEPA initialization** for `vit_small` (loads `target_encoder.*` weights from `pretrain-ckpt/best_chkpt.pt`).
- **Jaccard soft-negative CLIP loss** (`clip_loss_jaccard`) — converts in-batch negatives that share labels into multi-positives, with a soft-negative band controlled by `jaccard_t` and `soft_neg_scale`.
- **GLoRIA-style local contrastive loss** (`gloria_local_loss`) over per-token ECG and per-token text embeddings.
- **UMA loss** from MERL (kept as baseline, never ablated).

The repo contains **three coordinated training stages** plus shared evaluation utilities:

1. `ECG-JEPA/` — standalone JEPA pre-training of the ViT encoder on raw ECG (no text). Produces `best_chkpt.pt` consumed by the next stage.
2. `pretrain/` — DDP CLIP-style ECG↔report pre-training (entry: `pretrain/main.py`, config: `pretrain/config.py`).
3. `finetune/` — supervised fine-tuning / linear probing of the pre-trained encoder on PTB-XL, ICBEB (CPSC2018), and Chapman (CSN) (entry: `finetune/main_single.py`).
4. `zeroshot/` — CKEPE-prompt zero-shot evaluation (entry: `zeroshot/test_zeroshot.py`).

## Common Commands

### JEPA pre-training (stage 1)
```bash
# Dump MIMIC-IV-ECG to a single .npy (only once)
cd ECG-JEPA && python -m scripts.dump_data --data-dir "/path/to/mimic-iv-ecg" --verbose

# Pre-train the ViT-S encoder
cd ECG-JEPA && bash launch.sh
# or:
python -m pretrain --data "mimic-iv-ecg=/path/to/train.npy" --out "pretrain-output-dir" --config "ViTS_mimic" --amp "bfloat16"
```
Configs live in [ECG-JEPA/configs/pretrain/](ECG-JEPA/configs/pretrain/) (`ViTS_mimic.yaml`, `ViTS_all.yaml`, `ViTXS_*`, `ViTB_all.yaml`).

### CLIP-style pre-training (stage 2)
```bash
# Preprocess MIMIC reports/records into train.npy + train.csv
cd pretrain && jupyter nbconvert --execute preprocess.ipynb   # or run preprocess.py

# Distributed run (4 GPUs)
cd pretrain && bash launch.sh
# launch.sh contents: torchrun --nproc_per_node=4 main.py
```
The trainer reads everything from [pretrain/config.py](pretrain/config.py); there is **no CLI override**. To ablate components, toggle these dict keys:

- `network.use_jepa_init` — disable JEPA encoder warm-start.
- `trainer.use_jaccard_mask` — disable Jaccard soft-negative loss (falls back to standard `clip_loss`).
- `trainer.use_local_loss` — disable GLoRIA local loss.
- `trainer.use_uma_loss` — kept `True` by default (MERL baseline component).

Sweep runs can override the checkpoint dir via `SWEEP_CKPT_FOLDER=...` environment variable (see [utils/utils_trainer.py](utils/utils_trainer.py#L100)).

### Fine-tuning / linear probing (stage 3)
```bash
cd finetune/sub_script && bash run_all_linear.sh
```
This calls `sub_<dataset>.sh` which runs [finetune/main_single.py](finetune/main_single.py) at three label ratios (1/10/100%) for each dataset.

For a single run:
```bash
cd finetune && python main_single.py \
  --dataset ptbxl-super \      # ptbxl-super | ptbxl-sub | ptbxl-form | ptbxl-rhythm | icbeb | chapman
  --ratio 100 \                # 1 / 10 / 100 — fraction of training data
  --backbone vit_small \       # resnet18/34/50/101 | vit_small | ecgfm_{small,base,large}
  --pretrain_path /path/to/<name>_bestZeroShotAll_encoder.pth \
  --name linear \              # 'linear' freezes backbone; otherwise full fine-tune
  --checkpoint-dir ./ckpt_dir
```
Important: pass the `*_encoder.pth` checkpoint, not the full `*_ckpt.pth` (the encoder file matches the backbone state dict directly).

### Zero-shot evaluation
```bash
cd zeroshot && bash zeroshot.sh   # = python test_zeroshot.py
```
Edit the `ckpt` path inside [zeroshot/test_zeroshot.py](zeroshot/test_zeroshot.py) before running. The script also produces a t-SNE plot on Chapman.

## Architecture

### Top-level model: `ECGCLIP` ([utils/utils_builder.py](utils/utils_builder.py))

A two-tower CLIP variant:

- **ECG tower** dispatches on `network.ecg_model` string substring:
  - `'resnet'` → `ResNet{18,34,50,101}` from [utils/resnet1d.py](utils/resnet1d.py) + `AttentionPool2d` head.
  - `'vit'` → `vit_small` from [utils/vit1d.py](utils/vit1d.py) + MLP `proj_e`.
  - `'ecgfm'` → `ECGFMModel` from [utils/ecgfm.py](utils/ecgfm.py) (suffix `_small/_base/_large`).
  - For `vit_small` with `use_jepa_init=True`, the encoder loads `target_encoder.*` keys from `../pretrain-ckpt/best_chkpt.pt` (path is relative to where you launch `main.py`, typically `pretrain/`).
- **Text tower** is a HuggingFace `AutoModel` (default: `ncbi/MedCPT-Query-Encoder`), with optional layer freezing via `network.free_layers` (12 = freeze all). Pooling is masked mean-pooling, then a 2-layer MLP `proj_t`.
- **Forward** returns dict with `ecg_emb` (two dropout views for UMA), `proj_ecg_emb`, `proj_text_emb`, and (when `return_text_tokens=True`) per-token `ecg_token_emb`, `text_token_emb`, `text_token_mask` for the local loss.

### Trainer ([utils/utils_trainer.py](utils/utils_trainer.py))

DDP-only training loop. Key behaviors:

- All-gathers `proj_ecg_emb`, `proj_text_emb` (and optionally `ecg_emb1/2`, `labels`) across ranks before computing CLIP losses on the global batch.
- Total loss = `cma_loss + uma_loss + local_loss_weight * local_loss`. Each term is gated by its `use_*` switch.
- Cosine LR schedule with warmup (`CosineAnnealingWarmupRestarts` from [utils/scheduler.py](utils/scheduler.py)); 10% of total steps are warmup.
- Optionally splits text-side params (`lm_model` + `proj_t`) into a separate param group with `lr_text` to slow embedding-space drift.
- Validation runs both contrastive validation loss and zero-shot AUROC across all val sets in `config.zeroshot.val_sets`. Best checkpoint is selected by `avg_auc` and saved as both full model (`*_ckpt.pth`) and encoder-only (`*_encoder.pth`). Early stop: `patience=5` on no improvement.
- Checkpoint folder defaults to `../checkpoints/<UTC+8 timestamp>/`; override via `SWEEP_CKPT_FOLDER`.

### Loss design ([utils/utils_loss.py](utils/utils_loss.py))

- `clip_loss` — standard symmetric InfoNCE with `temperature=0.07`.
- `clip_loss_jaccard(x, y, labels, jaccard_t, soft_neg_scale)` — multi-positive InfoNCE. Builds a Jaccard similarity matrix over multi-hot `labels`; pairs with Jaccard ≥ `jaccard_t` are positives, those in `(0, jaccard_t)` are soft negatives weighted by `soft_neg_scale`. When `jaccard_t <= 0`, falls back to plain `clip_loss`.
- `gloria_local_loss` — per-image attention over text tokens (GLoRIA-style), masked by `text_token_mask` to skip CLS/SEP/PAD.
- `local_contrastive_loss` — alternative sentence-level local loss (currently not wired into the trainer).

### Datasets

- Pre-training: [utils/utils_dataset.py](utils/utils_dataset.py) — `MIMIC_E_T_Dataset` reads ECG from `train.npy`/`val.npy` (already preprocessed; values divided by 1000), report from `total_report` column, and label vector from columns 6 onwards (used by Jaccard loss).
- Fine-tune / zero-shot: [finetune/finetune_dataset.py](finetune/finetune_dataset.py) — `ECGDataset` handles PTB-XL (4 subtasks: super/sub/form/rhythm), ICBEB, and Chapman from raw WFDB or `.mat` files. The same module re-exports `getdataset` for zero-shot via `from finetune_dataset import getdataset as get_zero_dataset`.

### Path conventions

- `PROJECT_ROOT = pretrain/config.py.parent.parent` and `DATASETS_ROOT = PROJECT_ROOT.parent / 'datasets'` — the code expects `datasets/` to live as a **sibling of the repo**, with subdirs `datasets/pretrain/` and `datasets/finetune/`. Adjust in [pretrain/config.py](pretrain/config.py) and [zeroshot/config.py](zeroshot/config.py) if your layout differs.
- Several scripts contain hardcoded absolute paths from the original author (`/public/home/hs_mmcd_5/...`); update them before running on a different machine — especially in the `finetune/sub_script/**/*.sh` scripts.
- `sys.path.append("../utils")` is used throughout to make the `utils/` modules importable; entry points must be launched from their containing directory (e.g., `cd pretrain && torchrun ... main.py`).

### Logging

Pre-training and fine-tuning log to **SwanLab** (project `MERL_pretrain`); ensure `swanlab` is installed and authenticated. There is no W&B integration.

## Gotchas

- **`pretrain/launch.sh` has a typo** (`main.pycd your_path/...`) — the actual command is `torchrun --nproc_per_node=4 main.py` and the rest is leftover from `zeroshot.sh`.
- **`use_jepa_init=True` requires `pretrain-ckpt/best_chkpt.pt`** to exist, located at `../pretrain-ckpt/best_chkpt.pt` relative to the `pretrain/` working directory. If you only want CLIP pre-training without JEPA warm-start, set this to `False`.
- **DDP is mandatory for `pretrain/main.py`** — it calls `dist.init_process_group("nccl")` and `dist.all_gather` unconditionally; running with `python main.py` will fail.
- **`finetune.py` and `pretrain.py` at the ECG-JEPA top level** are the JEPA stage; do not confuse them with `finetune/main_single.py` and `pretrain/main.py` at the repo root.
- Python deps for the JEPA stage are pinned in [ECG-JEPA/requirements.txt](ECG-JEPA/requirements.txt) (`torch==2.7.0` etc.); the rest of the repo additionally needs `transformers`, `wfdb`, `swanlab`, `pytz`, `tqdm`, `scikit-learn`, `matplotlib`.
