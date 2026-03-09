"""A3C (Asynchronous Advantage Actor-Critic) Agent wrapper."""

from typing import Any, Dict, Optional

from stable_baselines3 import A2C

try:
    from stable_baselines3 import A3C  # type: ignore[attr-defined]
except Exception:
    A3C = None


class A3CAgent:
    """A3C agent wrapper with preset hyperparameters."""

    ALGORITHM = "A3C"

    # Default hyperparameters tuned for 2-DOF arm task
    # A3C is good for parallel training with multiple workers
    DEFAULT_HYPERPARAMS = {
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
    }

    @staticmethod
    def get_hyperparams() -> Dict[str, Any]:
        """Get A3C default hyperparameters."""
        return A3CAgent.DEFAULT_HYPERPARAMS.copy()

    @staticmethod
    def create_model(env: Any, hyperparams: Optional[Dict] = None) -> Any:
        """
        Create A3C model.

        Args:
            env: Gymnasium environment
            hyperparams: Hyperparameter dict (uses defaults if None)

        Returns:
            A3C model instance when available, otherwise A2C fallback
        """
        if hyperparams is None:
            hyperparams = A3CAgent.DEFAULT_HYPERPARAMS.copy()

        AlgorithmClass = A3C if A3C is not None else A2C
        return AlgorithmClass(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            **hyperparams,
        )

    @staticmethod
    def get_description() -> str:
        """Get algorithm description."""
        return (
            "A3C (Asynchronous Advantage Actor-Critic): parallel policy-gradient style."
            " If native A3C is unavailable in your Stable-Baselines3 version, this"
            " wrapper falls back to A2C with compatible defaults."
        )


__all__ = ["A3CAgent"]
