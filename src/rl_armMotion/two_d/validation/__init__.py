"""Validation harnesses for trained policies.

This subpackage contains scientific-validation harnesses for verifying that a
trained policy reproduces emergent behaviours observed in human and primate
motor control. The benchmarks follow Fischer et al. (2021), Sci. Rep. 11:14445.

Currently implemented:
    - fitts_law: Movement time vs. log2(2D/W) regression (Fitts, 1954).
    - power_law: Two-thirds Power Law regression of log V against log C
      (Lacquaniti, Terzuolo, & Viviani, 1983).
"""

from rl_armMotion.two_d.validation.fitts_law import (
    FittsLawCondition,
    FittsLawResult,
    FittsLawTrial,
    FittsLawValidator,
)
from rl_armMotion.two_d.validation.power_law import (
    PowerLawResult,
    PowerLawTrial,
    PowerLawValidator,
)

__all__ = [
    "FittsLawCondition",
    "FittsLawResult",
    "FittsLawTrial",
    "FittsLawValidator",
    "PowerLawResult",
    "PowerLawTrial",
    "PowerLawValidator",
]
