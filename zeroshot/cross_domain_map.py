"""Cross-domain category overlap for MERL Table 2.

MERL paper publishes only the three forward mapping tables:
    Table 11 — PTBXL-Super -> CPSC2018
    Table 12 — PTBXL-Super -> CSN
    Table 13 — CPSC2018    -> CSN

Under MERL's official protocol ("no need to re-implement, just average AUROC
over overlapping categories"), each cell of Table 2 only needs the set of
*target* labels that are covered by some source-domain category. We can read
both directions out of the same forward table:

* Forward direction (source on the LHS, target on the RHS):
    overlap target labels = union of all RHS lists.
* Reverse direction (source on the RHS, target on the LHS):
    overlap target labels = LHS keys that have a non-empty mapping.

So we never write a separate "reverse" table; we always read from the
forward dict, picking the appropriate side.
"""
from __future__ import annotations

from typing import Dict, List, Optional


# ---- Forward tables (MERL Tables 11/12/13) ----

PTBXL_SUPER_TO_CPSC2018: Dict[str, Optional[List[str]]] = {
    "HYP":  None,
    "NORM": ["NORM"],
    "CD":   ["1AVB", "CRBBB", "CLBBB"],
    "MI":   None,
    "STTC": ["STE", "STD"],
}

PTBXL_SUPER_TO_CSN: Dict[str, Optional[List[str]]] = {
    "HYP":  ["RVH", "LVH"],
    "NORM": ["SR"],
    "CD":   ["2AVB", "2AVB1", "1AVB", "AVB", "LBBB", "RBBB", "STDD"],
    "MI":   ["MI"],
    "STTC": ["STTC", "STE", "TWO", "STTU", "QTIE", "TWC"],
}

CPSC2018_TO_CSN: Dict[str, Optional[List[str]]] = {
    "AFIB":  ["AFIB"],
    "VPC":   ["VPB"],
    "NORM":  ["SR"],
    "1AVB":  ["1AVB"],
    "CRBBB": ["RBBB"],
    "STE":   ["STE"],
    "PAC":   ["APB"],
    "CLBBB": ["LBBB"],
    "STD":   ["STE", "STTC", "STTU", "STDD"],
}


def _overlap_rhs(forward: Dict[str, Optional[List[str]]]) -> List[str]:
    """Forward direction: union of all listed target labels (RHS)."""
    out: List[str] = []
    for v in forward.values():
        if not v:
            continue
        for x in v:
            if x not in out:
                out.append(x)
    return out


def _overlap_lhs(forward: Dict[str, Optional[List[str]]]) -> List[str]:
    """Reverse direction: LHS source-domain keys with a non-empty mapping."""
    return [k for k, v in forward.items() if v]


# ``(source_set, target_set) -> overlap target labels``.
# Keys must match ``args_zeroshot_eval['test_sets']`` keys.
CROSS_DOMAIN_OVERLAPS: Dict[tuple, List[str]] = {
    # forward 3
    ("ptbxl_super_class", "icbeb"):    _overlap_rhs(PTBXL_SUPER_TO_CPSC2018),
    ("ptbxl_super_class", "chapman"):  _overlap_rhs(PTBXL_SUPER_TO_CSN),
    ("icbeb",   "chapman"):            _overlap_rhs(CPSC2018_TO_CSN),
    # reverse 3 (read the LHS of the same forward table)
    ("icbeb",   "ptbxl_super_class"):  _overlap_lhs(PTBXL_SUPER_TO_CPSC2018),
    ("chapman", "ptbxl_super_class"):  _overlap_lhs(PTBXL_SUPER_TO_CSN),
    ("chapman", "icbeb"):              _overlap_lhs(CPSC2018_TO_CSN),
}
