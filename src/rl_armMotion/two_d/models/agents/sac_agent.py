"""SAC (Soft Actor-Critic) Agent wrapper"""

from typing import Any, Dict, Optional

from stable_baselines3 import SAC


class SACAgent:
    """SAC agent wrapper with preset hyperparameters."""

    ALGORITHM = "SAC"

    # Default hyperparameters tuned for 2-DOF arm task
    # SAC is sample-efficient and good for continuous control
    DEFAULT_HYPERPARAMS = {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_update_interval": 1,
    }

    @staticmethod
    def get_hyperparams() -> Dict[str, Any]:
        """Get SAC default hyperparameters."""
        return SACAgent.DEFAULT_HYPERPARAMS.copy()

    @staticmethod
    def create_model(env: Any, hyperparams: Optional[Dict] = None) -> SAC:
        """
        Create SAC model.

        Args:
            env: Gymnasium environment
            hyperparams: Hyperparameter dict (uses defaults if None)

        Returns:
            SAC model instance
        """
        if hyperparams is None:
            hyperparams = SACAgent.DEFAULT_HYPERPARAMS.copy()

        return SAC(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            **hyperparams,
        )

    @staticmethod
    def get_description() -> str:
        """Get algorithm description."""
        return (
            "SAC (Soft Actor-Critic): A state-of-the-art off-policy algorithm that"
            " combines entropy regularization with reinforcement learning. Highly"
            " sample-efficient and stable for continuous control tasks."
        )


__all__ = ["SACAgent"]
