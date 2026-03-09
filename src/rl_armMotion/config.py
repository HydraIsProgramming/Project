"""Configuration settings for RL Arm Motion project"""

import os
from pathlib import Path

from rl_armMotion.two_d.config import ArmConfiguration

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "default_learning_rate": 0.001,
    "default_batch_size": 32,
    "default_episodes": 1000,
}

# Environment configuration
ENV_CONFIG = {
    "observation_space_dim": 2,  # 2-DOF arm (shoulder + elbow)
    "action_space_dim": 2,
    "max_episode_steps": 1000,
}

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "MODEL_CONFIG",
    "ENV_CONFIG",
    "ArmConfiguration",
]
