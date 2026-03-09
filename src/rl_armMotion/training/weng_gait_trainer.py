"""
Training wrapper for the Weng gait environment.

This module provides a lightweight training and evaluation interface around
`stable_baselines3` that is specialised for the gait environment defined in
`weng_gait_env.py`.  It exposes hooks for curriculum learning, penalty
ramping and evaluation of a trained policy against a suite of metrics such
as success rate, time to target, pose error, smoothness and effort.  The
wrapper does not implement the GUI streaming facilities of the 2D
trainers; instead it is focused on research and development workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from rl_armMotion.environments.weng_gait_env import WengGaitEnv, PenaltySchedule


@dataclass
class TrainingResult:
    """Container for training and evaluation results."""

    mean_reward: float
    std_reward: float
    success_rate: float
    mean_time_to_target: float
    mean_pose_error: float
    mean_smoothness: float
    mean_effort: float
    safety_violations: int
    model: Any


def _default_hyperparams(algorithm: str) -> Dict[str, Any]:
    """Return sensible default hyperparameters for PPO and SAC."""
    alg = algorithm.upper()
    if alg == "PPO":
        return {
            "learning_rate": 3e-4,
            "gamma": 0.995,
            "gae_lambda": 0.98,
            "ent_coef": 0.005,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 10,
            "clip_range": 0.2,
        }
    if alg == "SAC":
        return {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
        }
    raise ValueError(f"Unsupported algorithm: {algorithm}")


class PenaltyRampCallback(BaseCallback):
    """Callback that linearly increases penalty multipliers during training."""

    def __init__(self, env: WengGaitEnv, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # Update penalty schedule based on progress_remaining provided by SB3
        progress_remaining: float = self.model._current_progress_remaining  # type: ignore[attr-defined]
        # Convert progress_remaining (1→0) to fraction completed (0→1)
        fraction_completed = 1.0 - float(progress_remaining)
        self.env.penalty_schedule.update(fraction_completed)
        return True


class WengGaitTrainer:
    """Simple trainer for the Weng gait environment using SB3 algorithms."""

    SUPPORTED_ALGORITHMS = ("PPO", "SAC")

    def __init__(
        self,
        env: Optional[WengGaitEnv] = None,
        *,
        total_timesteps: int = 200_000,
        algorithm: str = "PPO",
        hyperparams: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise the trainer.

        Args
        ----
        env: WengGaitEnv, optional
            The environment to train in.  When ``None`` a default
            ``WengGaitEnv`` is created.
        total_timesteps: int
            Number of environment timesteps to train for.
        algorithm: str
            Either "PPO" or "SAC".  Case-insensitive.
        hyperparams: dict, optional
            Hyperparameters to pass to the SB3 model.  If omitted then
            reasonable defaults are supplied.
        seed: int, optional
            Random seed for both the environment and the SB3 model.
        """
        alg = algorithm.upper()
        if alg not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm {algorithm}; choose from {self.SUPPORTED_ALGORITHMS}")
        self.total_timesteps = int(total_timesteps)
        self.env: WengGaitEnv = env or WengGaitEnv()
        # Set seed on environment if provided
        if seed is not None:
            self.env.reset(seed=seed)
        self.algorithm: str = alg
        self.hyperparams: Dict[str, Any] = hyperparams or _default_hyperparams(alg)
        # Create SB3 model
        if alg == "PPO":
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                verbose=0,
                seed=seed,
                **self.hyperparams,
            )
        else:
            # SAC: uses continuous actions, identical interface
            self.model = SAC(
                policy="MlpPolicy",
                env=self.env,
                verbose=0,
                seed=seed,
                **self.hyperparams,
            )

    def train(self) -> None:
        """Train the policy for the configured number of timesteps."""
        # Use penalty ramp callback to gradually increase penalties
        callback = PenaltyRampCallback(self.env)
        self.model.learn(total_timesteps=self.total_timesteps, callback=callback)

    def evaluate(self, num_episodes: int = 20) -> TrainingResult:
        """Evaluate the trained policy across several episodes and compute metrics.

        The evaluation runs in the deterministic mode of the policy and
        collects metrics described in the project description.  Metrics are
        averaged across episodes to produce summary statistics.

        Args
        ----
        num_episodes: int
            Number of episodes to run for evaluation.

        Returns
        -------
        TrainingResult
            Dataclass containing mean reward, success rate and other metrics.
        """
        # Wrap environment in DummyVecEnv for vectorised evaluation (SB3 util)
        eval_env = DummyVecEnv([lambda: self.env])
        # Evaluate policy using SB3 helper; returns mean and std of returns
        mean_reward, std_reward = evaluate_policy(
            self.model, eval_env, n_eval_episodes=num_episodes, deterministic=True, return_episode_rewards=False
        )
        # Custom metrics
        successes: List[bool] = []
        times_to_target: List[int] = []
        pose_errors: List[float] = []
        smoothness_vals: List[float] = []
        effort_vals: List[float] = []
        safety_violations: int = 0

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            step_count = 0
            smoothness_acc = 0.0
            effort_acc = 0.0
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                new_obs, reward, terminated, truncated, info = self.env.step(action)
                # Track smoothness and effort from the info dictionary if available
                if isinstance(info, dict):
                    smoothness_acc += info.get("smoothness_penalty", 0.0)
                    effort_acc += info.get("effort_penalty", 0.0)
                # Check termination reasons
                if terminated and not truncated:
                    # Terminated due to fall or joint limit
                    safety_violations += 1
                # Update step and obs
                step_count += 1
                obs = new_obs
                done = bool(terminated or truncated)
            # At the end of episode compute metrics
            # Success if end effector is within tolerance of target
            dist = info.get("distance_to_target", np.inf) if isinstance(info, dict) else np.inf
            is_success = bool(dist < self.env.goal_tolerance)
            successes.append(is_success)
            times_to_target.append(step_count)
            pose_errors.append(float(dist))
            smoothness_vals.append(float(smoothness_acc))
            effort_vals.append(float(effort_acc))
        # Compute aggregated statistics
        success_rate = float(np.mean(successes)) if successes else 0.0
        mean_time = float(np.mean(times_to_target)) if times_to_target else 0.0
        mean_pose_err = float(np.mean(pose_errors)) if pose_errors else 0.0
        mean_smoothness = float(np.mean(smoothness_vals)) if smoothness_vals else 0.0
        mean_effort = float(np.mean(effort_vals)) if effort_vals else 0.0
        return TrainingResult(
            mean_reward=float(mean_reward),
            std_reward=float(std_reward),
            success_rate=success_rate,
            mean_time_to_target=mean_time,
            mean_pose_error=mean_pose_err,
            mean_smoothness=mean_smoothness,
            mean_effort=mean_effort,
            safety_violations=safety_violations,
            model=self.model,
        )