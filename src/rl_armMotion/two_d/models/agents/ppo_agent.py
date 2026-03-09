"""PPO (Proximal Policy Optimization) Agent wrapper"""

from typing import Any, Dict, Optional

from stable_baselines3 import PPO


class PPOAgent:
    """PPO agent wrapper with preset hyperparameters."""

    ALGORITHM = "PPO"

    # Default hyperparameters tuned for 2-DOF arm task
    DEFAULT_HYPERPARAMS = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
    }

    @staticmethod
    def get_hyperparams() -> Dict[str, Any]:
        """Get PPO default hyperparameters."""
        return PPOAgent.DEFAULT_HYPERPARAMS.copy()

    @staticmethod
    def create_model(env: Any, hyperparams: Optional[Dict] = None) -> PPO:
        """
        Create PPO model.

        Args:
            env: Gymnasium environment
            hyperparams: Hyperparameter dict (uses defaults if None)

        Returns:
            PPO model instance
        """
        if hyperparams is None:
            hyperparams = PPOAgent.DEFAULT_HYPERPARAMS.copy()

        return PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            **hyperparams,
        )

    @staticmethod
    def get_description() -> str:
        """Get algorithm description."""
        return (
            "PPO (Proximal Policy Optimization): A policy gradient method that uses"
            " clipped objective to ensure stable training. Good balance of performance"
            " and stability for continuous control tasks."
        )


__all__ = ["PPOAgent"]
