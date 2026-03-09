"""Training wrappers with GUI-friendly metrics streaming for 3D training."""

from __future__ import annotations

import queue
import threading
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from rl_armMotion.three_d.environments import ArmTaskEnv3D
from rl_armMotion.two_d.models.callbacks import GUICallback
from rl_armMotion.two_d.models.trainers import RLTrainer


class RLTrainerWithMetrics3D:
    """Wrapper around RLTrainer that streams 3D metrics for GUI visualization."""

    SUPPORTED_ALGORITHMS = ("PPO", "SAC", "A2C")

    def __init__(
        self,
        env: Optional[ArmTaskEnv3D] = None,
        total_timesteps: int = 100000,
        algorithm: str = "PPO",
        hyperparams: Optional[Dict[str, Any]] = None,
        preload_model_path: Optional[str] = None,
        metrics_queue: Optional[queue.Queue] = None,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        check_freq: int = 100,
    ):
        if env is None:
            env = ArmTaskEnv3D()

        self.algorithm = algorithm.upper()
        if self.algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Algorithm {self.algorithm} not supported for 3D GUI trainer. "
                f"Choose from {list(self.SUPPORTED_ALGORITHMS)}"
            )

        self.env = env
        self.total_timesteps = int(total_timesteps)
        self.preload_model_path = preload_model_path
        self.metrics_queue = metrics_queue
        self.metrics_callback = metrics_callback
        self.check_freq = int(check_freq)
        self._external_should_stop = should_stop

        self.trainer = RLTrainer(
            env=env,
            algorithm=self.algorithm,
            hyperparams=hyperparams,
        )

        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.episode_goals_reached = deque(maxlen=1000)
        self.episode_goal_distances = deque(maxlen=1000)

        self.policy_losses = deque(maxlen=10000)
        self.value_losses = deque(maxlen=10000)
        self.entropies = deque(maxlen=10000)

        self.latest_joint_angles = np.deg2rad(np.asarray(self.env.config.initial_angles_deg, dtype=float))
        self.latest_end_effector_position = np.zeros(3, dtype=float)
        self.latest_shoulder_position = np.asarray(self.env.shoulder_base_position, dtype=float)
        self.goal_position = np.asarray(self.env.goal_position, dtype=float)
        self.goal_direction = str(self.env.goal_direction)
        self.latest_goal_distance = float("inf")
        self.latest_height_error = 0.0
        self.latest_signed_axis_error = 0.0
        self.latest_lateral_error = 0.0
        self.latest_orientation_error = 0.0
        self.latest_hold_counter = 0
        self.latest_hold_steps_required = int(getattr(self.env, "hold_steps_required", 1))
        self.latest_hold_progress = 0.0
        self.latest_in_goal_region = False
        self.latest_gradient_norm = 0.0

        self.best_reward = -float("inf")
        self.best_distance = float("inf")
        self.timesteps_trained = 0
        self.episodes_completed = 0
        self.start_time: Optional[datetime] = None
        self.training_active = False
        self.stop_requested = False

        self.metrics_lock = threading.RLock()

    def request_stop(self) -> None:
        self.stop_requested = True

    def should_stop(self) -> bool:
        if self.stop_requested:
            return True
        if self._external_should_stop is None:
            return False
        try:
            return bool(self._external_should_stop())
        except Exception:
            return False

    def train(self) -> Dict[str, Any]:
        self.start_time = datetime.now()
        self.training_active = True
        self.stop_requested = False

        if self.preload_model_path:
            self.trainer.load(self.preload_model_path)

        gui_callback = TrainingGUICallback3D(
            trainer=self,
            check_freq=self.check_freq,
        )

        try:
            result = self.trainer.train(
                total_timesteps=self.total_timesteps,
                callback=gui_callback,
            )
            return {
                **result,
                **self._get_training_summary(),
                "stopped_early": bool(self.stop_requested),
            }
        except Exception as exc:
            self._safe_queue_put({"error": str(exc), "type": "training_error"})
            raise
        finally:
            self.training_active = False

    def get_current_metrics(self) -> Dict[str, Any]:
        with self.metrics_lock:
            episode_rewards_list = list(self.episode_rewards)
            episode_goals = list(self.episode_goals_reached)
            episode_distances = list(self.episode_goal_distances)

            mean_reward = float(np.mean(episode_rewards_list[-100:])) if episode_rewards_list else 0.0
            mean_distance = float(np.mean(episode_distances[-100:])) if episode_distances else 0.0
            success_rate = float(np.mean(episode_goals[-100:]) * 100.0) if episode_goals else 0.0

            policy_loss = float(self.policy_losses[-1]) if self.policy_losses else 0.0
            value_loss = float(self.value_losses[-1]) if self.value_losses else 0.0
            entropy = float(self.entropies[-1]) if self.entropies else 0.0

            elapsed_time = 0.0
            if self.start_time is not None:
                elapsed_time = (datetime.now() - self.start_time).total_seconds()

            return {
                "algorithm": self.algorithm,
                "timesteps": self.timesteps_trained,
                "episodes": self.episodes_completed,
                "episode_rewards": episode_rewards_list,
                "episode_lengths": list(self.episode_lengths),
                "mean_reward": mean_reward,
                "best_reward": float(self.best_reward),
                "success_rate": success_rate,
                "avg_goal_distance": mean_distance,
                "best_distance": float(self.best_distance),
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "policy_losses": list(self.policy_losses),
                "value_losses": list(self.value_losses),
                "entropies": list(self.entropies),
                "training_time_sec": elapsed_time,
                "joint_angles": self.latest_joint_angles.tolist(),
                "end_effector_position": self.latest_end_effector_position.tolist(),
                "shoulder_position": self.latest_shoulder_position.tolist(),
                "goal_position": self.goal_position.tolist(),
                "goal_direction": self.goal_direction,
                "goal_distance": float(self.latest_goal_distance),
                "height_error": float(self.latest_height_error),
                "signed_axis_error": float(self.latest_signed_axis_error),
                "lateral_error": float(self.latest_lateral_error),
                "orientation_error": float(self.latest_orientation_error),
                "hold_counter": int(self.latest_hold_counter),
                "hold_steps_required": int(self.latest_hold_steps_required),
                "hold_progress": float(self.latest_hold_progress),
                "in_goal_region": bool(self.latest_in_goal_region),
                "gradient_norm": float(self.latest_gradient_norm),
            }

    def _push_periodic_metrics(self) -> None:
        metrics = self.get_current_metrics()
        metrics["type"] = "metrics_update"
        self._emit_metrics(metrics)

    def _add_episode_result(
        self,
        reward: float,
        length: int,
        goal_reached: bool,
        goal_distance: float,
    ) -> None:
        with self.metrics_lock:
            self.episode_rewards.append(float(reward))
            self.episode_lengths.append(int(length))
            self.episode_goals_reached.append(float(goal_reached))
            self.episode_goal_distances.append(float(goal_distance))
            self.episodes_completed += 1

            if reward > self.best_reward:
                self.best_reward = float(reward)
            if goal_distance < self.best_distance:
                self.best_distance = float(goal_distance)

        metrics = self.get_current_metrics()
        metrics["type"] = "episode_completed"
        self._emit_metrics(metrics)

    def _add_algorithm_metrics(
        self,
        policy_loss: float,
        value_loss: float,
        entropy: float,
    ) -> None:
        with self.metrics_lock:
            self.policy_losses.append(float(policy_loss))
            self.value_losses.append(float(value_loss))
            self.entropies.append(float(entropy))

    def _set_timesteps(self, timesteps: int) -> None:
        with self.metrics_lock:
            self.timesteps_trained = int(timesteps)

    def _set_latest_pose(self, info: Dict[str, Any]) -> None:
        with self.metrics_lock:
            joint_angles = info.get("joint_angles")
            if joint_angles is not None:
                self.latest_joint_angles = np.asarray(joint_angles, dtype=float)

            end_effector = info.get("end_effector_position")
            if end_effector is not None:
                self.latest_end_effector_position = np.asarray(end_effector, dtype=float)

            shoulder_pos = info.get("shoulder_position")
            if shoulder_pos is not None:
                self.latest_shoulder_position = np.asarray(shoulder_pos, dtype=float)

            goal_position = info.get("goal_position")
            if goal_position is not None:
                self.goal_position = np.asarray(goal_position, dtype=float)

            goal_direction = info.get("goal_direction")
            if goal_direction is not None:
                self.goal_direction = str(goal_direction)

            goal_distance = info.get("goal_distance")
            if goal_distance is not None:
                self.latest_goal_distance = float(goal_distance)

            height_error = info.get("height_error")
            if height_error is not None:
                self.latest_height_error = float(height_error)

            signed_axis_error = info.get("signed_axis_error")
            if signed_axis_error is not None:
                self.latest_signed_axis_error = float(signed_axis_error)

            lateral_error = info.get("lateral_error")
            if lateral_error is not None:
                self.latest_lateral_error = float(lateral_error)

            orientation_error = info.get("orientation_error")
            if orientation_error is not None:
                self.latest_orientation_error = float(orientation_error)

            hold_counter = info.get("hold_counter")
            if hold_counter is not None:
                self.latest_hold_counter = int(hold_counter)

            hold_steps_required = info.get("hold_steps_required")
            if hold_steps_required is not None:
                self.latest_hold_steps_required = int(hold_steps_required)

            hold_progress = info.get("hold_progress")
            if hold_progress is not None:
                self.latest_hold_progress = float(hold_progress)

            in_goal_region = info.get("in_goal_region")
            if in_goal_region is not None:
                self.latest_in_goal_region = bool(in_goal_region)

            gradient_norm = info.get("gradient_norm")
            if gradient_norm is not None:
                self.latest_gradient_norm = float(gradient_norm)

    def _get_training_summary(self) -> Dict[str, Any]:
        return {
            "final_metrics": self.get_current_metrics(),
            "total_episodes": self.episodes_completed,
            "algorithm": self.algorithm,
        }

    def evaluate_policy(
        self,
        num_episodes: int = 5,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        return self.trainer.evaluate(
            env=self.env,
            num_episodes=num_episodes,
            deterministic=deterministic,
        )

    def save_model_and_results(self, save_dir: str) -> Dict[str, str]:
        import csv
        import json
        import os
        from pathlib import Path

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        model_path = os.path.join(save_dir, f"{self.algorithm.lower()}_model")
        self.trainer.save(model_path)

        history_path = os.path.join(save_dir, "training_history.csv")
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "reward",
                "length",
                "goal_reached",
                "goal_distance",
            ])
            for i, (reward, length, goal, distance) in enumerate(
                zip(
                    self.episode_rewards,
                    self.episode_lengths,
                    self.episode_goals_reached,
                    self.episode_goal_distances,
                )
            ):
                writer.writerow([i, reward, length, goal, distance])

        stats_path = os.path.join(save_dir, "training_stats.json")
        final_metrics = self.get_current_metrics()
        stats = {
            "algorithm": self.algorithm,
            "total_timesteps": final_metrics["timesteps"],
            "total_episodes": final_metrics["episodes"],
            "mean_reward": final_metrics["mean_reward"],
            "best_reward": final_metrics["best_reward"],
            "success_rate": final_metrics["success_rate"],
            "hold_progress": final_metrics.get("hold_progress", 0.0),
            "gradient_norm": final_metrics.get("gradient_norm", 0.0),
            "training_time_sec": final_metrics["training_time_sec"],
            "stopped_early": bool(self.stop_requested),
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        return {
            "model": model_path,
            "history": history_path,
            "stats": stats_path,
        }

    def _safe_queue_put(self, message: Dict[str, Any]) -> None:
        if self.metrics_queue is None:
            return
        try:
            self.metrics_queue.put_nowait(message)
        except queue.Full:
            pass

    def _emit_metrics(self, metrics: Dict[str, Any]) -> None:
        self._safe_queue_put(metrics)
        if self.metrics_callback is not None:
            try:
                self.metrics_callback(metrics)
            except Exception:
                pass


class TrainingGUICallback3D(GUICallback):
    """Callback that extracts 3D episode and optimizer metrics for GUI updates."""

    def __init__(
        self,
        trainer: RLTrainerWithMetrics3D,
        check_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(check_freq=check_freq, verbose=verbose)
        self.trainer = trainer
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return [value]

    def _on_step(self) -> bool:
        self.trainer._set_timesteps(self.num_timesteps)

        if self.trainer.should_stop():
            return False

        infos = self._ensure_list(self.locals.get("infos"))
        rewards = self._ensure_list(self.locals.get("rewards"))
        dones = [bool(v) for v in self._ensure_list(self.locals.get("dones"))]

        env_count = max(len(infos), len(rewards), len(dones), 1)
        if len(self._episode_rewards) != env_count:
            self._episode_rewards = [0.0] * env_count
            self._episode_lengths = [0] * env_count

        for idx in range(env_count):
            reward = float(rewards[idx]) if idx < len(rewards) else 0.0
            self._episode_rewards[idx] += reward
            self._episode_lengths[idx] += 1

            info = infos[idx] if idx < len(infos) and isinstance(infos[idx], dict) else {}
            if info:
                self.trainer._set_latest_pose(info)

            done = dones[idx] if idx < len(dones) else False
            if not done:
                continue

            episode_reward = self._episode_rewards[idx]
            episode_length = self._episode_lengths[idx]

            episode_info = info.get("episode")
            if isinstance(episode_info, dict):
                episode_reward = float(episode_info.get("r", episode_reward))
                episode_length = int(episode_info.get("l", episode_length))

            self.trainer._add_episode_result(
                reward=episode_reward,
                length=episode_length,
                goal_reached=bool(info.get("goal_reached", False)),
                goal_distance=float(info.get("goal_distance", 0.0)),
            )

            self._episode_rewards[idx] = 0.0
            self._episode_lengths[idx] = 0

        if self.num_timesteps % self.check_freq == 0:
            self._extract_algorithm_metrics()
            self.trainer._push_periodic_metrics()

        return True

    def _extract_algorithm_metrics(self) -> None:
        model = self.model
        logger = getattr(model, "logger", None)
        if logger is None:
            return

        logger_data = getattr(logger, "name_to_value", {})
        if not isinstance(logger_data, dict):
            return

        policy_loss = self._get_first_present(logger_data, [
            "train/policy_loss",
            "train/actor_loss",
            "train/loss",
        ])
        value_loss = self._get_first_present(logger_data, [
            "train/value_loss",
            "train/critic_loss",
            "train/entropy_loss",
        ])
        entropy = self._get_first_present(logger_data, [
            "train/entropy_loss",
            "train/ent_coef_loss",
        ])

        if entropy == 0.0 and hasattr(model, "policy"):
            try:
                if hasattr(model.policy, "log_std"):
                    entropy = float(model.policy.log_std.mean().item())
            except Exception:
                entropy = 0.0

        self.trainer._add_algorithm_metrics(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
        )

    @staticmethod
    def _get_first_present(data: Dict[str, Any], keys: List[str]) -> float:
        for key in keys:
            if key in data:
                try:
                    return float(data[key])
                except Exception:
                    continue
        return 0.0


__all__ = [
    "RLTrainerWithMetrics3D",
    "TrainingGUICallback3D",
]
