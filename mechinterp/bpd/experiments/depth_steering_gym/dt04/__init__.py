"""BPD-DT04: Prospective competent depth-transport and protected causal steering study.

Additive package leaving all archived DT01, DT02-A, DT03 and UI files untouched.
"""
from __future__ import annotations

STUDY_ID = "BPD-DT04"
SMOKE_ROOT = 2603999
DEV_ROOTS = tuple(range(2604000, 2604059))  # 59 independent development + sham-calibration roots
SEALED_CONFIRM_ROOTS = tuple(range(2604100, 2604356))  # Strictly sealed; never generated or executed
FIT_DEPTHS = (2, 4, 6)
HELDOUT_DEPTHS = (8, 12)
SAME_CASE_EVAL_DEPTHS = (6, 8, 12)
