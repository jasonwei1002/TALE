"""Replicate MERL Table 2 zero-shot row using a TALE checkpoint.

Runs all six cross-domain cells:
    PTBXL-Super -> CPSC2018, CSN
    CPSC2018    -> PTBXL-Super, CSN
    CSN         -> PTBXL-Super, CPSC2018

Implementation follows MERL's official protocol: each target dataset is
zero-shot-evaluated once with its own labels + CKEPE prompts; per-cell AUROC
is the average over the overlap target classes for that (source, target).
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(_THIS_DIR.parent / "utils"))

import utils_builder  # noqa: E402
from config import config  # noqa: E402
from cross_domain_map import CROSS_DOMAIN_OVERLAPS  # noqa: E402
from cross_domain_eval import cross_domain_zeroshot_eval  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "true"

torch.manual_seed(42)
random.seed(0)
np.random.seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Build model ----
network_cfg = dict(config["network"])
network_cfg["ecg_model"] = "vit_small"
# Skip JEPA init at construction time; the trained weights are loaded right after.
network_cfg["use_jepa_init"] = False
model = utils_builder.ECGCLIP(network_cfg)

_default_ckpt = _THIS_DIR.parent / "checkpoints/best/vit_small_bestZeroShotAll_ckpt.pth"
ckpt_path = Path(os.environ.get("TALE_DOMAIN_CKPT", str(_default_ckpt))).resolve()
print(f"Loading checkpoint: {ckpt_path}")
state = torch.load(str(ckpt_path), map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"  [warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
if unexpected:
    print(f"  [warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

model = model.to(DEVICE)
model = torch.nn.DataParallel(model)

args_zeroshot_eval = config["zeroshot"]

# ---- Run all six cells ----
zs_cache: dict = {}
all_results = {}
for (src, tgt), overlap in CROSS_DOMAIN_OVERLAPS.items():
    res = cross_domain_zeroshot_eval(
        model=model,
        source_set=src,
        target_set=tgt,
        overlap=overlap,
        args_zeroshot_eval=args_zeroshot_eval,
        device=DEVICE,
        cache=zs_cache,
    )
    all_results[(src, tgt)] = res

# ---- Final summary table ----
print("\n========== Cross-domain zero-shot AUROC ==========")
header = f"{'Source':<22s}{'Target':<22s}{'AUROC':>8s}  {'#cls':>5s}"
print(header)
print("-" * len(header))
for (src, tgt), res in all_results.items():
    n = len(res["per_class"])
    auc = res["avg_auc"]
    auc_str = f"{auc:8.4f}" if not np.isnan(auc) else "    nan "
    print(f"{src:<22s}{tgt:<22s}{auc_str}  {n:5d}")
