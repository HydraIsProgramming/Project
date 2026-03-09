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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
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
    """Callback that linearly increases penalty multipliers and advances the curriculum.

    This callback updates the penalty schedule based on the fraction of
    training completed and automatically progresses through the
    stabilise→step→walk curriculum.  Curriculum thresholds are
    configurable via the ``curriculum_thresholds`` argument.
    """

    def __init__(self, env: WengGaitEnv, curriculum_thresholds: Tuple[float, float] = (0.33, 0.66), verbose: int = 0) -> None:
        super().__init__(verbose)
        self.env = env
        # Two threshold fractions at which to transition from stage 1→2 and 2→3
        # Values must be in increasing order between 0 and 1.
        if not (0.0 <= curriculum_thresholds[0] <= curriculum_thresholds[1] <= 1.0):
            raise ValueError("curriculum_thresholds must be monotonic between 0 and 1")
        self.threshold_stage2, self.threshold_stage3 = curriculum_thresholds

    def _on_step(self) -> bool:
        # Update penalty schedule based on progress_remaining provided by SB3
        progress_remaining: float = self.model._current_progress_remaining  # type: ignore[attr-defined]
        # Convert progress_remaining (1→0) to fraction completed (0→1)
        fraction_completed = 1.0 - float(progress_remaining)
        self.env.penalty_schedule.update(fraction_completed)
        # Automated curriculum progression: advance to stage 2 and 3 at thresholds
        try:
            # Use explicit API on environment when available
            if fraction_completed >= self.threshold_stage2 and self.env.curriculum_stage < 2:
                self.env.set_curriculum_stage(2)  # type: ignore[attr-defined]
            if fraction_completed >= self.threshold_stage3 and self.env.curriculum_stage < 3:
                self.env.set_curriculum_stage(3)  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback: directly set stage attribute if set_curriculum_stage not defined
            if fraction_completed >= self.threshold_stage2 and self.env.curriculum_stage < 2:
                self.env.curriculum_stage = 2
            if fraction_completed >= self.threshold_stage3 and self.env.curriculum_stage < 3:
                self.env.curriculum_stage = 3
        return True


class LoggingCallback(BaseCallback):
    """Callback that records per‑episode metrics during training.

    The callback accumulates reward and metric sums for each episode and
    stores them in a list of dictionaries.  At the end of training the
    collected data can be written to a CSV file when ``log_path`` is
    provided.  The logged metrics include episode reward, episode
    length, accumulated smoothness and effort penalties, joint and
    stability penalties, the curriculum stage, and a simple success flag.
    """

    def __init__(self, env: WengGaitEnv, log_path: Optional[str] = None, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.env = env
        self.log_path = log_path
        # Internal accumulators
        self._current_reward = 0.0
        self._current_length = 0
        self._current_smoothness = 0.0
        self._current_effort = 0.0
        self._current_joint = 0.0
        self._current_stability = 0.0
        self._episodes_data: List[Dict[str, Any]] = []
        self._episode_index = 0

    def _on_step(self) -> bool:
        # Extract per‑step info; assume single environment (vectorised envs wrap this callback separately)
        reward = float(self.locals.get("rewards", [0.0])[0])  # type: ignore[index]
        info = {}
        if "infos" in self.locals and isinstance(self.locals["infos"], (list, tuple)) and self.locals["infos"]:
            info = self.locals["infos"][0] or {}
        done_flags = self.locals.get("dones", [False])  # type: ignore[index]
        done = bool(done_flags[0])
        # Accumulate metrics
        self._current_reward += reward
        self._current_length += 1
        # Sum penalties; use .get() with default 0.0 when absent
        if isinstance(info, dict):
            self._current_smoothness += float(info.get("smoothness_penalty", 0.0))
            self._current_effort += float(info.get("effort_penalty", 0.0))
            self._current_joint += float(info.get("joint_limit_penalty", 0.0))
            self._current_stability += float(info.get("stability_penalty", 0.0))
        # When episode terminates or truncates, record metrics and reset accumulators
        if done:
            # Determine success: if the agent remained within goal tolerance long enough
            # We use the environment's internal success counter relative to hold requirement
            try:
                success = 1 if self.env._success_counter >= self.env._hold_required else 0  # type: ignore[attr-defined]
            except Exception:
                success = 0
            episode_record = {
                "episode": self._episode_index,
                "stage": self.env.curriculum_stage,
                "reward": self._current_reward,
                "length": self._current_length,
                "smoothness_sum": self._current_smoothness,
                "effort_sum": self._current_effort,
                "joint_sum": self._current_joint,
                "stability_sum": self._current_stability,
                "success": success,
            }
            self._episodes_data.append(episode_record)
            # Reset accumulators for next episode
            self._episode_index += 1
            self._current_reward = 0.0
            self._current_length = 0
            self._current_smoothness = 0.0
            self._current_effort = 0.0
            self._current_joint = 0.0
            self._current_stability = 0.0
        return True

    def _on_training_end(self) -> None:
        # Write logs to CSV when a path is provided
        if self.log_path and self._episodes_data:
            import csv
            from pathlib import Path
            # Ensure directory exists
            log_file = Path(self.log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._episodes_data[0].keys()))
                writer.writeheader()
                writer.writerows(self._episodes_data)
        # Expose collected data for users (e.g., after training)
        self.training_log = self._episodes_data


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
        log_path: Optional[str] = None,
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
        # Path where training logs should be stored; None disables file output
        self.log_path: Optional[str] = log_path
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

        # Placeholder for logging callback; will be initialised in train()
        self.logging_callback: Optional[LoggingCallback] = None

    def train(self) -> None:
        """Train the policy for the configured number of timesteps."""
        # Create callbacks: penalty ramp (with curriculum progression) and logging
        ramp_callback = PenaltyRampCallback(self.env)
        # Instantiate logging callback with given log path
        self.logging_callback = LoggingCallback(self.env, log_path=self.log_path)
        callback_list = CallbackList([ramp_callback, self.logging_callback])
        self.model.learn(total_timesteps=self.total_timesteps, callback=callback_list)

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


# ----------------------------------------------------------------------
# Plotting utilities
def plot_training_progress(log_file: str, metrics: Optional[List[str]] = None, save_dir: Optional[str] = None) -> None:
    """Plot training metrics from a CSV log file.

    This helper reads a CSV file produced by :class:`LoggingCallback` and
    generates a simple line plot for each requested metric.  Each plot
    displays the metric value versus episode index.  When ``save_dir`` is
    provided, figures are saved as PNG files with the metric name in
    the filename; otherwise figures are shown interactively.

    Parameters
    ----------
    log_file: str
        Path to the CSV file containing logged training metrics.
    metrics: list[str], optional
        Names of the metrics to plot.  By default the function plots
        ``reward``, ``smoothness_sum``, ``effort_sum``, ``joint_sum`` and
        ``stability_sum``.  Metric names must match the column names in
        ``log_file``.
    save_dir: str, optional
        Directory in which to save the generated figures.  When omitted
        or ``None``, the figures are displayed interactively.
    """
    import csv
    import matplotlib.pyplot as plt  # type: ignore
    from pathlib import Path

    # Read the CSV log into a list of dicts
    with open(log_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        logs = list(reader)
    if not logs:
        raise ValueError(f"No data found in log file {log_file}")

    # Default metrics to plot
    if metrics is None:
        metrics = ["reward", "smoothness_sum", "effort_sum", "joint_sum", "stability_sum"]
    # Convert numeric fields to float lists
    for metric in metrics:
        if metric not in logs[0]:
            raise ValueError(f"Metric '{metric}' not found in log file")
        values = [float(row[metric]) for row in logs]
        episodes = list(range(len(values)))
        plt.figure()
        plt.plot(episodes, values)
        plt.xlabel("Episode")
        plt.ylabel(metric.replace("_", " "))
        plt.title(f"Training {metric.replace('_', ' ')}")
        plt.grid(True)
        if save_dir:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{metric}.png"
            plt.savefig(out_path)
        else:
            plt.show()

