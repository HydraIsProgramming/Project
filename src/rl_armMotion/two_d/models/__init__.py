"""RL Arm Motion - Models module

This module contains RL model implementations and algorithms for training robotic arm control policies.

Components:
- trainers: BaseTrainer and RLTrainer classes for Stable-Baselines3 integration
- agents: Algorithm-specific wrappers (PPO, SAC, A3C, etc.)
- callbacks: Training callbacks for monitoring and early stopping
- metrics: Evaluation metrics and performance tracking
"""

from rl_armMotion.two_d.models.trainers import BaseTrainer, RLTrainer
from rl_armMotion.two_d.models.callbacks import GUICallback, PPOMetricsCallback

__all__ = [
    "BaseTrainer",
    "RLTrainer",
    "GUICallback",
    "PPOMetricsCallback",
]
