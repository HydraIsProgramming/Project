"""
RL Arm Motion - Reinforcement Learning for robotic arm motion control

A Python package for training and deploying reinforcement learning agents
for controlling robotic arm motion.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# NOTE:
# Keep top-level package imports lazy so lightweight modules (for example,
# interactive GUIs) can run without optional training dependencies such as
# stable_baselines3.

__all__ = [
    "models",
    "environments",
    "data",
    "utils",
    "two_d",
    "three_d",
]
