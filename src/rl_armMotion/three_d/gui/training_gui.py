"""3D training GUI for RL algorithms with real-time visualization and metrics."""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from rl_armMotion.three_d.config import ArmConfiguration3D
from rl_armMotion.three_d.environments import ArmTaskEnv3D
from rl_armMotion.three_d.training import RLTrainerWithMetrics3D
from rl_armMotion.three_d.utils import ArmKinematics3D


class TrainingGUI3D:
    """GUI for training 3D RL policies with live metrics."""

    ALGORITHMS = ["PPO", "SAC", "A2C"]
    GOAL_DIRECTIONS = ["EAST", "WEST", "NORTH"]

    def __init__(
        self,
        total_timesteps: int = 100000,
        save_dir: str = "./trained_models_3d",
        algorithm: str = "PPO",
    ):
        self.total_timesteps = int(total_timesteps)
        self.save_dir = save_dir
        self.selected_algorithm = algorithm.upper()
        if self.selected_algorithm not in self.ALGORITHMS:
            self.selected_algorithm = "PPO"

        probe_env = ArmTaskEnv3D(goal_direction="EAST")
        self.selected_arm_config = ArmConfiguration3D.from_dict(probe_env.config.to_dict())
        self.default_shoulder = np.asarray(probe_env.shoulder_base_position, dtype=float)
        self.default_goal_position = np.asarray(probe_env.goal_position, dtype=float)
        self.default_goal_direction = str(probe_env.goal_direction)
        self.default_goal_height = float(probe_env.goal_height)
        self.default_observation_dim = int(probe_env.observation_space.shape[0])
        self.default_action_dim = int(probe_env.action_space.shape[0])
        probe_env.close()

        self.root = tk.Tk()
        self.root.title("RL Training Dashboard - 3D Arm Motion")
        self.root.geometry("1560x880")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.metrics_queue = queue.Queue(maxsize=200)
        self.training_thread: Optional[threading.Thread] = None
        self.trainer: Optional[RLTrainerWithMetrics3D] = None
        self.training_active = False

        self.episode_numbers = deque(maxlen=1000)
        self.episode_rewards = deque(maxlen=1000)
        self.moving_average = deque(maxlen=1000)
        self.policy_losses = deque(maxlen=10000)
        self.value_losses = deque(maxlen=10000)
        self.entropies = deque(maxlen=10000)

        self.fig_rewards: Optional[Figure] = None
        self.fig_losses: Optional[Figure] = None
        self.fig_entropy: Optional[Figure] = None
        self.fig_arm: Optional[Figure] = None

        self.ax_rewards = None
        self.ax_losses_policy = None
        self.ax_losses_value = None
        self.ax_entropy = None
        self.ax_arm = None

        self.canvas_rewards = None
        self.canvas_losses = None
        self.canvas_entropy = None
        self.canvas_arm = None

        self.status_text = None
        self.metrics_text = None
        self.training_button = None
        self.stop_button = None
        self.save_button = None
        self.algorithm_combo = None
        self.timesteps_entry = None
        self.goal_direction_combo = None
        self.open_config_button = None
        self.config_name_entry = None

        self.episode_counter = 0
        self.start_time: Optional[datetime] = None

        self.algorithm_var = tk.StringVar(value=self.selected_algorithm)
        self.timesteps_var = tk.StringVar(value=f"{self.total_timesteps}")
        self.goal_direction_var = tk.StringVar(value="EAST")
        self.selected_config_name_var = tk.StringVar(value=str(self.selected_arm_config.name))

        self.selected_config_path: Optional[str] = None
        self.selected_goal_direction = "EAST"

        self.create_window()

    def create_window(self) -> None:
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.LabelFrame(main_frame, text="Training Curves", padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.fig_rewards = Figure(figsize=(6, 2.8), dpi=100)
        self.ax_rewards = self.fig_rewards.add_subplot(111)
        self.ax_rewards.set_xlabel("Episode")
        self.ax_rewards.set_ylabel("Reward")
        self.ax_rewards.set_title("Episode Rewards & Moving Average")
        self.ax_rewards.grid(True, alpha=0.3)
        self.canvas_rewards = FigureCanvasTkAgg(self.fig_rewards, master=left_frame)
        self.canvas_rewards.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_losses = Figure(figsize=(6, 2.8), dpi=100)
        self.ax_losses_policy = self.fig_losses.add_subplot(121)
        self.ax_losses_policy.set_xlabel("Step")
        self.ax_losses_policy.set_ylabel("Loss")
        self.ax_losses_policy.set_title("Policy/Actor Loss")
        self.ax_losses_policy.grid(True, alpha=0.3)

        self.ax_losses_value = self.fig_losses.add_subplot(122)
        self.ax_losses_value.set_xlabel("Step")
        self.ax_losses_value.set_ylabel("Loss")
        self.ax_losses_value.set_title("Value/Critic Loss")
        self.ax_losses_value.grid(True, alpha=0.3)

        self.canvas_losses = FigureCanvasTkAgg(self.fig_losses, master=left_frame)
        self.canvas_losses.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_entropy = Figure(figsize=(6, 2.8), dpi=100)
        self.ax_entropy = self.fig_entropy.add_subplot(111)
        self.ax_entropy.set_xlabel("Step")
        self.ax_entropy.set_ylabel("Entropy")
        self.ax_entropy.set_title("Policy Entropy")
        self.ax_entropy.grid(True, alpha=0.3)

        self.canvas_entropy = FigureCanvasTkAgg(self.fig_entropy, master=left_frame)
        self.canvas_entropy.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        center_frame = ttk.LabelFrame(main_frame, text="Arm Visualization", padding=5)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        model_frame = ttk.Frame(center_frame)
        model_frame.pack(fill=tk.X, padx=2, pady=2)

        self.open_config_button = ttk.Button(
            model_frame,
            text="Open Saved Arm Config",
            command=self._on_open_saved_arm_config,
        )
        self.open_config_button.pack(side=tk.LEFT, padx=(0, 6))

        self.config_name_entry = ttk.Entry(
            model_frame,
            textvariable=self.selected_config_name_var,
            state="readonly",
        )
        self.config_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.fig_arm = Figure(figsize=(5.8, 8), dpi=100)
        self.ax_arm = self.fig_arm.add_subplot(111, projection="3d")
        self.canvas_arm = FigureCanvasTkAgg(self.fig_arm, master=center_frame)
        self.canvas_arm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_arm_pose()

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)

        setup_frame = ttk.LabelFrame(right_frame, text="Training Setup", padding=5)
        setup_frame.pack(fill=tk.X, pady=5)

        ttk.Label(setup_frame, text="Algorithm:").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.algorithm_combo = ttk.Combobox(
            setup_frame,
            textvariable=self.algorithm_var,
            values=self.ALGORITHMS,
            state="readonly",
            width=12,
        )
        self.algorithm_combo.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        ttk.Label(setup_frame, text="Timesteps:").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.timesteps_entry = ttk.Entry(setup_frame, textvariable=self.timesteps_var, width=14)
        self.timesteps_entry.grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        ttk.Label(setup_frame, text="Goal Direction:").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        self.goal_direction_combo = ttk.Combobox(
            setup_frame,
            textvariable=self.goal_direction_var,
            values=self.GOAL_DIRECTIONS,
            state="readonly",
            width=12,
        )
        self.goal_direction_combo.grid(row=2, column=1, sticky="ew", padx=2, pady=2)
        self.goal_direction_combo.bind("<<ComboboxSelected>>", self._on_goal_direction_changed)

        ttk.Label(
            setup_frame,
            text=f"Obs/Action Dim: {self.default_observation_dim} / {self.default_action_dim}",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=2, pady=(4, 2))
        setup_frame.columnconfigure(1, weight=1)

        metrics_frame = ttk.LabelFrame(right_frame, text="Training Metrics", padding=5)
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.metrics_text = tk.Text(
            metrics_frame,
            width=38,
            height=20,
            font=("Courier", 9),
            state=tk.DISABLED,
        )
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        status_frame = ttk.LabelFrame(right_frame, text="Status", padding=5)
        status_frame.pack(fill=tk.X, pady=5)

        self.status_text = tk.Label(
            status_frame,
            text="Ready to train",
            font=("Courier", 10),
            fg="green",
        )
        self.status_text.pack(fill=tk.X)

        control_frame = ttk.LabelFrame(right_frame, text="Controls", padding=5)
        control_frame.pack(fill=tk.X, pady=5)

        self.training_button = ttk.Button(
            control_frame,
            text="Start Training",
            command=self._on_start_training,
        )
        self.training_button.pack(fill=tk.X, pady=2)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Training",
            command=self._on_stop_training,
            state=tk.DISABLED,
        )
        self.stop_button.pack(fill=tk.X, pady=2)

        self.save_button = ttk.Button(
            control_frame,
            text="Save Model & Results",
            command=self._on_save_model,
            state=tk.DISABLED,
        )
        self.save_button.pack(fill=tk.X, pady=2)

        self._on_goal_direction_changed()
        self._schedule_metrics_check()

    def _set_setup_controls_enabled(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        entry_state = "normal" if enabled else "disabled"
        self.algorithm_combo.config(state=state)
        self.timesteps_entry.config(state=entry_state)
        self.goal_direction_combo.config(state=state)

    def _update_default_goal_target(self, direction: str) -> None:
        direction = str(direction).strip().upper()
        shoulder = np.asarray(self.selected_arm_config.shoulder_position, dtype=float)
        max_reach = float(np.sum(self.selected_arm_config.link_lengths[:2]))

        if direction == "EAST":
            goal = shoulder + np.array([max_reach, 0.0, 0.0], dtype=float)
        elif direction == "WEST":
            goal = shoulder + np.array([-max_reach, 0.0, 0.0], dtype=float)
        elif direction == "NORTH":
            goal = shoulder + np.array([0.0, max_reach, 0.0], dtype=float)
        else:
            goal = shoulder + np.array([max_reach, 0.0, 0.0], dtype=float)
            direction = "EAST"

        self.default_shoulder = shoulder
        self.default_goal_position = goal
        self.default_goal_height = float(goal[1])
        self.default_goal_direction = direction

    def _on_goal_direction_changed(self, _event: Optional[tk.Event] = None) -> None:
        direction = self.goal_direction_var.get().strip().upper()
        self._update_default_goal_target(direction)
        initial_angles = np.deg2rad(np.asarray(self.selected_arm_config.initial_angles_deg, dtype=float))
        preview_payload = {
            "joint_angles": initial_angles,
            "shoulder_position": self.default_shoulder,
            "goal_position": self.default_goal_position,
            "goal_direction": direction,
        }
        self._draw_arm_pose(preview_payload)

    def _on_open_saved_arm_config(self) -> None:
        default_dir = Path("arm_configuration")
        initial_dir = str(default_dir if default_dir.exists() else Path.cwd())
        selected = filedialog.askopenfilename(
            title="Open saved arm configuration",
            initialdir=initial_dir,
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return

        try:
            config = ArmConfiguration3D.from_json(selected)
            if int(config.dof) != 4:
                messagebox.showerror(
                    "Invalid Configuration",
                    "3D training GUI supports only 4-DOF spherical-shoulder configurations",
                )
                return

            self.selected_arm_config = config
            self.selected_config_path = selected
            self.selected_config_name_var.set(str(config.name))
            self._on_goal_direction_changed()
            self.status_text.config(text=f"Loaded arm config: {config.name}", fg="green")
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load arm configuration:\n{exc}")

    def _on_start_training(self) -> None:
        if self.training_active:
            messagebox.showwarning("Warning", "Training already in progress")
            return

        try:
            timesteps = int(str(self.timesteps_var.get()).replace(",", "").strip())
            if timesteps <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Timesteps must be a positive integer")
            return

        algorithm = self.algorithm_var.get().strip().upper()
        if algorithm not in self.ALGORITHMS:
            messagebox.showerror("Invalid Input", f"Algorithm must be one of: {', '.join(self.ALGORITHMS)}")
            return

        goal_direction = self.goal_direction_var.get().strip().upper()
        if goal_direction not in self.GOAL_DIRECTIONS:
            messagebox.showerror("Invalid Input", f"Goal direction must be one of: {', '.join(self.GOAL_DIRECTIONS)}")
            return

        self.total_timesteps = timesteps
        self.selected_algorithm = algorithm
        self.selected_goal_direction = goal_direction
        self.training_active = True
        self.episode_counter = 0
        self.start_time = datetime.now()

        self.training_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self._set_setup_controls_enabled(False)
        self.status_text.config(text=f"Training {algorithm} ({goal_direction}) in progress...", fg="blue")

        self.episode_numbers.clear()
        self.episode_rewards.clear()
        self.moving_average.clear()
        self.policy_losses.clear()
        self.value_losses.clear()
        self.entropies.clear()

        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()

    def _on_stop_training(self) -> None:
        if not self.training_active:
            return

        self.training_active = False
        if self.trainer is not None:
            self.trainer.request_stop()

        self.status_text.config(text="Stopping...", fg="orange")
        self.stop_button.config(state=tk.DISABLED)

    def _on_save_model(self) -> None:
        if self.trainer is None:
            messagebox.showerror("Error", "No trained model available")
            return

        save_dir = filedialog.askdirectory(
            title="Select directory to save model",
            initialdir=self.save_dir,
        )
        if not save_dir:
            return

        try:
            self.status_text.config(text="Saving model...", fg="blue")
            self.root.update()

            save_paths = self.trainer.save_model_and_results(save_dir)
            self._save_plots(save_dir)

            self.status_text.config(text=f"Model saved to {save_dir}", fg="green")

            messagebox.showinfo(
                "Success",
                f"Model saved to {save_dir}\n\n"
                f"Files:\n"
                f"- Model: {save_paths['model']}\n"
                f"- History: {save_paths['history']}\n"
                f"- Stats: {save_paths['stats']}",
            )

        except Exception as exc:
            self.status_text.config(text="Save failed", fg="red")
            messagebox.showerror("Error", f"Failed to save model:\n{str(exc)}")

    def _training_loop(self) -> None:
        try:
            env_config = ArmConfiguration3D.from_dict(self.selected_arm_config.to_dict())
            training_env = ArmTaskEnv3D(
                goal_direction=self.selected_goal_direction,
                config=env_config,
            )
            self.trainer = RLTrainerWithMetrics3D(
                env=training_env,
                total_timesteps=self.total_timesteps,
                algorithm=self.selected_algorithm,
                metrics_queue=self.metrics_queue,
                should_stop=lambda: not self.training_active,
                check_freq=100,
            )

            result = self.trainer.train()
            stopped_early = bool(result.get("stopped_early", False)) or not self.training_active
            if stopped_early:
                self.metrics_queue.put({"type": "training_stopped", "result": result})
            else:
                self.metrics_queue.put({"type": "training_complete", "result": result})

        except Exception as exc:
            self.metrics_queue.put({"type": "training_error", "error": str(exc)})
        finally:
            self.training_active = False

    def _schedule_metrics_check(self) -> None:
        try:
            while True:
                metrics = self.metrics_queue.get_nowait()
                self._process_metrics(metrics)
        except queue.Empty:
            pass

        self.root.after(100, self._schedule_metrics_check)

    def _process_metrics(self, metrics: Dict[str, Any]) -> None:
        msg_type = metrics.get("type", "metrics")

        if msg_type in {"episode_completed", "metrics_update"}:
            self._update_from_episode_data(metrics)
        elif msg_type == "training_complete":
            self._on_training_complete(metrics.get("result", {}))
        elif msg_type == "training_stopped":
            self._on_training_stopped(metrics.get("result", {}))
        elif msg_type == "training_error":
            self._on_training_error(metrics.get("error", "Unknown error"))

    def _update_from_episode_data(self, metrics: Dict[str, Any]) -> None:
        episode_rewards = metrics.get("episode_rewards", []) or []
        policy_losses = metrics.get("policy_losses", []) or []
        value_losses = metrics.get("value_losses", []) or []
        entropies = metrics.get("entropies", []) or []

        self.episode_rewards.clear()
        self.episode_rewards.extend(list(episode_rewards)[-1000:])

        self.episode_numbers.clear()
        self.episode_numbers.extend(range(1, len(self.episode_rewards) + 1))

        self.moving_average.clear()
        reward_list = list(self.episode_rewards)
        for idx in range(len(reward_list)):
            start = max(0, idx - 99)
            self.moving_average.append(float(np.mean(reward_list[start : idx + 1])))

        self.policy_losses.clear()
        self.policy_losses.extend(list(policy_losses)[-10000:])

        self.value_losses.clear()
        self.value_losses.extend(list(value_losses)[-10000:])

        self.entropies.clear()
        self.entropies.extend(list(entropies)[-10000:])

        self.episode_counter = int(metrics.get("episodes", len(self.episode_rewards)))

        self._update_plots()
        self._draw_arm_pose(metrics)
        self._update_metrics_display(metrics)

    def _update_plots(self) -> None:
        try:
            self.ax_rewards.clear()
            if self.episode_rewards:
                ep_list = list(self.episode_numbers)
                reward_list = list(self.episode_rewards)
                ma_list = list(self.moving_average)

                self.ax_rewards.plot(ep_list, reward_list, alpha=0.35, label="Episode Reward")
                self.ax_rewards.plot(ep_list, ma_list, linewidth=2, label="100-ep Moving Avg")
                self.ax_rewards.legend(loc="best")

            self.ax_rewards.set_xlabel("Episode")
            self.ax_rewards.set_ylabel("Reward")
            self.ax_rewards.set_title("Episode Rewards & Moving Average")
            self.ax_rewards.grid(True, alpha=0.3)
            self.canvas_rewards.draw_idle()
        except Exception:
            pass

        try:
            self.ax_losses_policy.clear()
            if self.policy_losses:
                steps = list(range(len(self.policy_losses)))
                self.ax_losses_policy.plot(steps, list(self.policy_losses), linewidth=1)

            self.ax_losses_policy.set_xlabel("Step")
            self.ax_losses_policy.set_ylabel("Loss")
            self.ax_losses_policy.set_title("Policy/Actor Loss")
            self.ax_losses_policy.grid(True, alpha=0.3)

            self.ax_losses_value.clear()
            if self.value_losses:
                steps = list(range(len(self.value_losses)))
                self.ax_losses_value.plot(steps, list(self.value_losses), linewidth=1)

            self.ax_losses_value.set_xlabel("Step")
            self.ax_losses_value.set_ylabel("Loss")
            self.ax_losses_value.set_title("Value/Critic Loss")
            self.ax_losses_value.grid(True, alpha=0.3)

            self.canvas_losses.draw_idle()
        except Exception:
            pass

        try:
            self.ax_entropy.clear()
            if self.entropies:
                steps = list(range(len(self.entropies)))
                self.ax_entropy.plot(steps, list(self.entropies), linewidth=1, color="green")

            self.ax_entropy.set_xlabel("Step")
            self.ax_entropy.set_ylabel("Entropy")
            self.ax_entropy.set_title("Policy Entropy")
            self.ax_entropy.grid(True, alpha=0.3)
            self.canvas_entropy.draw_idle()
        except Exception:
            pass

    def _draw_arm_pose(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        try:
            payload = metrics or {}

            default_angles = np.deg2rad(np.asarray(self.selected_arm_config.initial_angles_deg, dtype=float))
            joint_angles = np.asarray(payload.get("joint_angles", default_angles), dtype=float)
            if joint_angles.shape[0] != int(self.selected_arm_config.dof):
                joint_angles = default_angles

            shoulder = np.asarray(payload.get("shoulder_position", self.default_shoulder), dtype=float)
            goal_direction = str(payload.get("goal_direction", self.goal_direction_var.get().strip().upper()))
            goal_position = np.asarray(payload.get("goal_position", self.default_goal_position), dtype=float)

            positions = ArmKinematics3D.forward_kinematics(joint_angles, self.selected_arm_config)
            shoulder_fk = np.asarray(positions[1], dtype=float)
            shift = shoulder - shoulder_fk
            positions = positions + shift

            ee_from_info = payload.get("end_effector_position")
            ee = np.asarray(ee_from_info, dtype=float) if ee_from_info is not None else np.asarray(positions[-1], dtype=float)

            self.ax_arm.clear()

            arm = positions[1:]
            self.ax_arm.plot(arm[:, 0], arm[:, 1], arm[:, 2], "b-o", linewidth=2, label="Arm links")
            self.ax_arm.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c="black", s=26, label="Origin")
            self.ax_arm.scatter(shoulder[0], shoulder[1], shoulder[2], c="green", s=90, label="Shoulder")
            self.ax_arm.scatter(ee[0], ee[1], ee[2], c="red", s=120, marker="*", label="End-effector")

            self.ax_arm.scatter(
                goal_position[0],
                goal_position[1],
                goal_position[2],
                c="orange",
                s=96,
                marker="X",
                label=f"Goal ({goal_direction})",
            )
            self.ax_arm.plot(
                [shoulder[0], goal_position[0]],
                [shoulder[1], goal_position[1]],
                [shoulder[2], goal_position[2]],
                color="orange",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )

            reach = float(np.sum(self.selected_arm_config.link_lengths[:2])) + 0.4
            mins = np.minimum.reduce(
                [
                    shoulder - reach,
                    goal_position - 0.1,
                    np.array([-0.2, -0.2, -0.2], dtype=float),
                ]
            )
            maxs = np.maximum.reduce(
                [
                    shoulder + reach,
                    goal_position + 0.1,
                    np.array([0.2, 0.2, 0.2], dtype=float),
                ]
            )
            self.ax_arm.set_xlim(mins[0], maxs[0])
            self.ax_arm.set_ylim(mins[1], maxs[1])
            self.ax_arm.set_zlim(mins[2], maxs[2])
            self.ax_arm.set_box_aspect((1, 1, 1))
            try:
                self.ax_arm.view_init(elev=18, azim=-58, vertical_axis="y")
            except TypeError:
                self.ax_arm.view_init(elev=18, azim=-58)

            self.ax_arm.set_title(f"Policy Execution ({goal_direction})")
            self.ax_arm.set_xlabel("X (m)")
            self.ax_arm.set_ylabel("Y (m)")
            self.ax_arm.set_zlabel("Z (m)")
            self.ax_arm.grid(True, alpha=0.3)
            self.ax_arm.legend(loc="upper left", fontsize=8)
            self.canvas_arm.draw_idle()
        except Exception:
            pass

    def _update_metrics_display(self, metrics: Dict[str, Any]) -> None:
        try:
            elapsed = 0.0
            if self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()

            goal_pos = np.asarray(metrics.get("goal_position", self.default_goal_position), dtype=float)
            ee_pos = np.asarray(metrics.get("end_effector_position", np.zeros(3)), dtype=float)
            goal_pos_text = f"[{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]"
            ee_pos_text = f"[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]"

            text = (
                f"Algorithm:         {metrics.get('algorithm', self.selected_algorithm)}\n"
                f"Episode:           {self.episode_counter}\n"
                f"Timesteps:         {int(metrics.get('timesteps', 0)):,}\n"
                f"Training Time:     {elapsed:.1f}s\n"
                f"\n"
                f"Rewards:\n"
                f"  Mean (100ep):    {metrics.get('mean_reward', 0):.2f}\n"
                f"  Best:            {metrics.get('best_reward', 0):.2f}\n"
                f"\n"
                f"Task Metrics:\n"
                f"  Success Rate:    {metrics.get('success_rate', 0):.1f}%\n"
                f"  Avg Distance:    {metrics.get('avg_goal_distance', 0):.3f}m\n"
                f"  Best Distance:   {metrics.get('best_distance', 0):.3f}m\n"
                f"  Goal Direction:  {metrics.get('goal_direction', self.goal_direction_var.get())}\n"
                f"  Goal Position:   {goal_pos_text}\n"
                f"  End-effector:    {ee_pos_text}\n"
                f"  Height Error:    {metrics.get('height_error', 0):.3f}m\n"
                f"  Axis Error:      {metrics.get('signed_axis_error', 0):.3f}m\n"
                f"  Lateral Error:   {metrics.get('lateral_error', 0):.3f}m\n"
                f"  Orient Error:    {metrics.get('orientation_error', 0):.3f}rad\n"
                f"  Hold Progress:   {metrics.get('hold_counter', 0)}/{metrics.get('hold_steps_required', 0)}"
                f" ({metrics.get('hold_progress', 0) * 100:.1f}%)\n"
                f"  In Goal Region:  {'Yes' if metrics.get('in_goal_region', False) else 'No'}\n"
                f"  Gradient Norm:   {metrics.get('gradient_norm', 0):.3f}\n"
                f"\n"
                f"Optimizer Metrics:\n"
                f"  Policy/Actor:    {metrics.get('policy_loss', 0):.4f}\n"
                f"  Value/Critic:    {metrics.get('value_loss', 0):.4f}\n"
                f"  Entropy:         {metrics.get('entropy', 0):.4f}\n"
            )

            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(1.0, text)
            self.metrics_text.config(state=tk.DISABLED)
        except Exception:
            pass

    def _on_training_complete(self, result: Dict[str, Any]) -> None:
        self.training_active = False
        self.training_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self._set_setup_controls_enabled(True)
        self.status_text.config(text="Training completed successfully", fg="green")

        messagebox.showinfo(
            "Training Complete",
            f"Algorithm: {self.selected_algorithm}\n"
            f"Episodes: {self.episode_counter}\n"
            f"Best reward: {result.get('best_reward', 0):.2f}",
        )

    def _on_training_stopped(self, _result: Optional[Dict[str, Any]] = None) -> None:
        self.training_active = False
        self.training_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL if self.trainer is not None else tk.DISABLED)
        self._set_setup_controls_enabled(True)
        self.status_text.config(text="Training stopped", fg="orange")

    def _on_training_error(self, error: str) -> None:
        self.training_active = False
        self.training_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self._set_setup_controls_enabled(True)
        self.status_text.config(text="Training failed", fg="red")

        messagebox.showerror(
            "Training Error",
            f"Training failed with error:\n{error}",
        )

    def _save_plots(self, save_dir: str) -> None:
        try:
            rewards_path = Path(save_dir) / "rewards.png"
            self.fig_rewards.savefig(rewards_path, dpi=150, bbox_inches="tight")

            losses_path = Path(save_dir) / "losses.png"
            self.fig_losses.savefig(losses_path, dpi=150, bbox_inches="tight")

            entropy_path = Path(save_dir) / "entropy.png"
            self.fig_entropy.savefig(entropy_path, dpi=150, bbox_inches="tight")

            arm_path = Path(save_dir) / "arm_pose_3d.png"
            self.fig_arm.savefig(arm_path, dpi=150, bbox_inches="tight")
        except Exception:
            pass

    def _on_close(self) -> None:
        if self.training_active:
            if not messagebox.askyesno("Quit", "Training is still running. Stop and quit?"):
                return
            self._on_stop_training()

        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL Arm Motion 3D Training GUI")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--algorithm", type=str, default="PPO", help="Algorithm: PPO/SAC/A2C")
    parser.add_argument("--save-dir", type=str, default="./trained_models_3d", help="Default save directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gui = TrainingGUI3D(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        algorithm=args.algorithm,
    )
    gui.run()


if __name__ == "__main__":
    main()
