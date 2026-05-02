"""Cross-domain zero-shot evaluation following MERL's official protocol.

MERL official guidance for domain transfer:
    "No need to re-implement any new experiments — just compute the metric
     across overlapping categories."

For one (source -> target) cell of Table 2:

1. Run the standard zero-shot pipeline on ``target_set`` using that target
   dataset's own label set and CKEPE prompts (i.e., reuse ``zeroshot_eval``).
2. Average AUROC only over the target labels in the (source, target) overlap
   list; classes outside the overlap are excluded from the average.

Each ``target_set`` is full-evaluated at most once thanks to the optional
``cache`` argument, so all six cells cost three full zero-shot passes
(ptbxl_super_class, icbeb, chapman).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(_THIS_DIR.parent / "utils"))
sys.path.append(str(_THIS_DIR.parent / "finetune"))

from zeroshot_val import zeroshot_eval  # noqa: E402


def _run_full_zeroshot(model, target_set: str, args_zeroshot_eval: dict, device: str) -> Dict[str, float]:
    """Run standard zero-shot on target_set; return ``{label_name: auroc}``."""
    _, _, _, _, _, AUROCs, res_dict = zeroshot_eval(
        model=model,
        set_name=target_set,
        device=device,
        args_zeroshot_eval=args_zeroshot_eval,
    )
    label_names = [
        k[len("AUROC_"):]
        for k in res_dict
        if k.startswith("AUROC_") and k != "AUROC_avg"
    ]
    return dict(zip(label_names, AUROCs))


def cross_domain_zeroshot_eval(
    model,
    source_set: str,
    target_set: str,
    overlap: List[str],
    args_zeroshot_eval: dict,
    device: str = "cuda",
    cache: Optional[dict] = None,
) -> dict:
    """Compute one (source -> target) AUROC over overlap target classes.

    Args:
        model: ECGCLIP wrapped in DataParallel/DDP (same as ``zeroshot_eval``).
        source_set: only used for logging; prompts come from ``target_set``.
        target_set: key in ``args_zeroshot_eval['test_sets']``.
        overlap: list of target label names to average over.
        cache: optional dict for caching full per-target results across cells.

    Returns:
        ``{"avg_auc": float, "per_class": {label: auc}, "missing": [...]}.``
    """
    if cache is not None and target_set in cache:
        per_label_auc = cache[target_set]
    else:
        per_label_auc = _run_full_zeroshot(model, target_set, args_zeroshot_eval, device)
        if cache is not None:
            cache[target_set] = per_label_auc

    selected: Dict[str, float] = {}
    missing: List[str] = []
    for name in overlap:
        if name in per_label_auc:
            selected[name] = per_label_auc[name]
        else:
            missing.append(name)

    avg = float(np.mean(list(selected.values()))) if selected else float("nan")
    print(
        f"[{source_set} -> {target_set}] "
        f"avg AUROC over {len(selected)}/{len(overlap)} overlap classes = {avg:.4f}"
    )
    for k, v in selected.items():
        print(f"   {k:>6s}: {v:.4f}")
    if missing:
        print(f"   missing target columns: {missing}")
    return {"avg_auc": avg, "per_class": selected, "missing": missing}
