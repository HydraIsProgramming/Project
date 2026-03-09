"""3D package for RL Arm Motion.

Subpackages are intentionally left lazy to avoid importing optional training
dependencies when launching visualization-only tools.
"""

__all__ = [
    "config",
    "environments",
    "gui",
    "models",
    "training",
    "utils",
]
