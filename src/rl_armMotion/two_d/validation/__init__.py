"""Validation harnesses for trained policies.

This subpackage contains scientific-validation harnesses for verifying that a
trained policy reproduces emergent behaviours observed in human and primate
motor control. The benchmarks follow Fischer et al. (2021), Sci. Rep. 11:14445.

Currently implemented:
    - fitts_law: Movement time vs. log2(2D/W) regression (Fitts, 1954).
"""

from rl_armMotion.two_d.validation.fitts_law import (
    FittsLawCondition,
    FittsLawResult,
    FittsLawTrial,
    FittsLawValidator,
)

__all__ = [
    "FittsLawCondition",
    "FittsLawResult",
    "FittsLawTrial",
    "FittsLawValidator",
]
