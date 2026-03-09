"""Training callbacks for monitoring and metrics tracking

Provides callbacks for real-time training monitoring, best model checkpointing,
and data collection for GUI visualization.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GUICallback(BaseCallback):
    """
    Callback for tracking training metrics for GUI display.

    Accumulates episode rewards, losses, and other metrics that can be
    polled by GUI for real-time visualization.
    """

    def __init__(
        self,
        check_freq: int = 100,
        verbose: int = 0,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Initialize callback.

        Args:
            check_freq: Check frequency (steps between updates)
            verbose: Verbosity level
            progress_callback: Optional callback for progress updates
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.progress_callback = progress_callback

        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.timesteps: List[int] = []
        self.best_reward = -float("inf")
        self.best_timestep = 0

        # Running metrics
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Track episode completion (detect done flag)
        # Note: This is called every step; actual episode tracking
        # happens through wrapper's ep info callback

        if self.num_timesteps % self.check_freq == 0:
            # Periodic callback
            if self.progress_callback:
                self.progress_callback(self.num_timesteps)

        return True  # Continue training

    def _on_training_end(self) -> None:
        """Called when training ends."""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current training metrics.

        Returns:
            Dictionary with training statistics
        """
        if len(self.episode_rewards) == 0:
            return {
                "episodes": 0,
                "timesteps": self.num_timesteps,
                "mean_reward": 0.0,
                "best_reward": self.best_reward,
                "episode_rewards": [],
            }

        return {
            "episodes": len(self.episode_rewards),
            "timesteps": self.num_timesteps,
            "mean_reward": float(np.mean(self.episode_rewards[-100:]))  # Last 100 episodes
            if len(self.episode_rewards) > 0
            else 0.0,
            "best_reward": self.best_reward,
            "episode_rewards": self.episode_rewards.copy(),
            "episode_lengths": self.episode_lengths.copy(),
        }

    def add_episode(self, reward: float, length: int) -> None:
        """
        Add episode data (called by external tracking).

        Args:
            reward: Episode total reward
            length: Episode length in steps
        """
        self.episode_rewards.append(float(reward))
        self.episode_lengths.append(int(length))
        self.timesteps.append(self.num_timesteps)

        if reward > self.best_reward:
            self.best_reward = float(reward)
            self.best_timestep = self.num_timesteps


class MetricsTracker:
    """Tracks and accumulates training metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_infos: List[Dict[str, Any]] = []
        self.current_reward = 0.0
        self.current_length = 0
        self.best_reward = -float("inf")

    def reset_episode(self) -> None:
        """Reset current episode tracking."""
        self.current_reward = 0.0
        self.current_length = 0

    def step(self, reward: float, length: int = 1) -> None:
        """
        Record a step.

        Args:
            reward: Step reward
            length: Steps incremented (default 1)
        """
        self.current_reward += float(reward)
        self.current_length += int(length)

    def end_episode(self, info: Optional[Dict[str, Any]] = None) -> None:
        """
        End current episode and record stats.

        Args:
            info: Optional info dict to store
        """
        self.episode_rewards.append(self.current_reward)
        self.episode_lengths.append(self.current_length)

        if info:
            self.episode_infos.append(info)

        if self.current_reward > self.best_reward:
            self.best_reward = self.current_reward

        self.reset_episode()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get accumulated statistics.

        Returns:
            Dictionary with aggregated metrics
        """
        if len(self.episode_rewards) == 0:
            return {
                "episodes": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "best_reward": self.best_reward,
                "mean_length": 0.0,
            }

        return {
            "episodes": len(self.episode_rewards),
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "best_reward": float(self.best_reward),
            "mean_length": float(np.mean(self.episode_lengths)),
            "total_steps": int(np.sum(self.episode_lengths)),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_infos.clear()
        self.reset_episode()
        self.best_reward = -float("inf")


class PPOMetricsCallback(GUICallback):
    """
    Enhanced callback for PPO-specific metrics extraction.

    Extracts policy loss, value loss, and entropy for visualization.
    """

    def __init__(
        self,
        check_freq: int = 100,
        verbose: int = 0,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Initialize PPO metrics callback.

        Args:
            check_freq: Steps between updates
            verbose: Verbosity level
            progress_callback: Optional progress callback
        """
        super().__init__(
            check_freq=check_freq,
            verbose=verbose,
            progress_callback=progress_callback,
        )
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropies: List[float] = []

    def _on_step(self) -> bool:
        """Called after each environment step."""
        super()._on_step()

        # Extract PPO metrics periodically
        if self.num_timesteps % self.check_freq == 0:
            self._extract_ppo_metrics()

        return True  # Continue training

    def _extract_ppo_metrics(self) -> None:
        """Extract PPO-specific metrics from model."""
        try:
            model = self.model
            if hasattr(model, "logger") and model.logger:
                # Get named values from logger
                logger_data = getattr(model.logger, "name_to_value", {})

                # Extract PPO metrics
                policy_loss = float(logger_data.get("train/policy_loss", 0.0))
                value_loss = float(logger_data.get("train/value_loss", 0.0))

                # Policy entropy
                entropy = 0.0
                if hasattr(model, "policy"):
                    try:
                        # For continuous policies, estimate entropy from log_std
                        if hasattr(model.policy, "log_std"):
                            entropy = float(model.policy.log_std.mean().item())
                    except Exception:
                        entropy = 0.0

                # Record metrics
                self.policy_losses.append(policy_loss)
                self.value_losses.append(value_loss)
                self.entropies.append(entropy)

        except Exception:
            # Silently ignore metric extraction errors
            pass

    def get_ppo_metrics(self) -> Dict[str, Any]:
        """
        Get current PPO metrics.

        Returns:
            Dictionary with PPO-specific metrics
        """
        return {
            "policy_losses": self.policy_losses.copy(),
            "value_losses": self.value_losses.copy(),
            "entropies": self.entropies.copy(),
            "current_policy_loss": self.policy_losses[-1] if self.policy_losses else 0.0,
            "current_value_loss": self.value_losses[-1] if self.value_losses else 0.0,
            "current_entropy": self.entropies[-1] if self.entropies else 0.0,
        }


__all__ = ["GUICallback", "MetricsTracker", "PPOMetricsCallback"]
