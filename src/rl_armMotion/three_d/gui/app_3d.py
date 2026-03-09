"""Interactive 3D arm controller GUI with spherical shoulder constraints."""

from __future__ import annotations

import pickle
import sys
from collections import deque
from pathlib import Path
from time import time

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, ttk

# Ensure src is importable when this file is launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl_armMotion.three_d.config import ArmConfiguration3D
from rl_armMotion.three_d.utils import ArmController3D, ArmKinematics3D, MotionRecorder3D


class ArmControllerGUI3D:
    """Tkinter GUI for the 3D arm model."""

    def __init__(self, config: ArmConfiguration3D | None = None):
        self.config = config or ArmConfiguration3D.get_default()
        # Shoulder anchor is fixed at world origin for global-axis rotations.
        self.config.shoulder_position = [0.0, 0.0, 0.0]
        self.controller = ArmController3D(self.config)
        self.recorder = MotionRecorder3D()

        self.running = True
        self.recording = False
        self.playing_back = False
        self.simulation_active = False
        self.playback_frames = []
        self.playback_index = 0
        self.selected_joint = 0

        self.show_trajectory = True
        self.trajectory_points = []

        self.last_frame_time = time()
        self.frame_count = 0
        self.fps_counter = 0.0

        self.prev_angles = self.controller.angles.copy()
        self.prev_velocities = np.zeros(self.config.dof, dtype=float)
        self.prev_update_time = time()

        self.latest_shoulder_torque = 0.0
        self.latest_elbow_torque = 0.0

        self.torque_time_hist = deque(maxlen=2000)
        self.shoulder_torque_hist = deque(maxlen=2000)
        self.elbow_torque_hist = deque(maxlen=2000)
        self.torque_plot_time = 0.0

        self.ee_kin_time_hist = deque(maxlen=2000)
        self.ee_vx_hist = deque(maxlen=2000)
        self.ee_vy_hist = deque(maxlen=2000)
        self.ee_vz_hist = deque(maxlen=2000)
        self.ee_ax_hist = deque(maxlen=2000)
        self.ee_ay_hist = deque(maxlen=2000)
        self.ee_az_hist = deque(maxlen=2000)
        self._ee_prev_pos = None
        self._ee_prev_vel = np.zeros(3, dtype=float)
        self._ee_prev_time = None
        self._ee_plot_time = 0.0

        self.root = tk.Tk()
        self.root.title("RL Arm Motion - 3D Interactive Controller")
        self.root.geometry("1400x920")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.fig = None
        self.ax = None
        self.canvas_agg = None
        self.torque_fig = None
        self.torque_ax = None
        self.torque_canvas = None
        self.velacc_fig = None
        self.velacc_ax_vel = None
        self.velacc_ax_acc = None
        self.velacc_canvas = None
        self._view_elev = 18.0
        self._view_azim = -58.0

        self.slider_vars = {}
        self.angle_labels = {}
        self.metrics_text = None
        self.sim_model_label = None
        self.config_details_text = None
        self.sim_toggle_button = None
        self.record_button = None
        self.playback_button = None
        self.sim_model_path = ""
        self.sim_algorithm = ""
        self.sim_model_metadata = {}

    def _force_shoulder_origin(self):
        """Keep shoulder anchor fixed at world origin."""
        self.config.shoulder_position = [0.0, 0.0, 0.0]

    @staticmethod
    def _normalize_model_base(filepath: str) -> str:
        """Normalize selected model path to Stable-Baselines3 base path."""
        path = Path(filepath)
        if path.suffix.lower() == ".zip":
            return str(path.with_suffix(""))
        return str(path)

    def _read_model_metadata(self, model_base_path: str):
        """Load optional metadata saved alongside the trained model."""
        metadata_path = Path(f"{model_base_path}_metadata.pkl")
        if not metadata_path.exists():
            return {}
        try:
            with metadata_path.open("rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _get_algorithm_from_metadata(self, model_base_path: str) -> str:
        """Infer algorithm from metadata first; fallback to filename patterns."""
        metadata = self._read_model_metadata(model_base_path)
        algo = str(metadata.get("algorithm", "")).strip().upper() if isinstance(metadata, dict) else ""
        if algo in {"PPO", "SAC", "A2C", "DQN", "A3C"}:
            return algo

        name = Path(model_base_path).name.lower()
        if "ppo" in name:
            return "PPO"
        if "sac" in name:
            return "SAC"
        if "a2c" in name:
            return "A2C"
        if "dqn" in name:
            return "DQN"
        if "a3c" in name:
            return "A3C"
        return "UNKNOWN"

    def _create_properties_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="ARM PROPERTIES", padding=10)

        ttk.Label(frame, text="Link Lengths (m):", font=("Arial", 10, "bold")).pack(fill="x", pady=(0, 4))
        for i in range(2):
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            row.columnconfigure(1, weight=5)
            row.columnconfigure(3, weight=2)

            ttk.Label(row, text=f"Link {i + 1}:", width=10).grid(row=0, column=0, sticky="w")
            var = tk.DoubleVar(value=float(self.config.link_lengths[i]))
            self.slider_vars[f"link_len_{i}"] = var
            ttk.Scale(
                row,
                from_=0.1,
                to=2.0,
                variable=var,
                orient="horizontal",
                command=lambda val, idx=i: self._on_link_length_change(idx, val),
            ).grid(row=0, column=1, sticky="ew", padx=5)

            val_label = ttk.Label(row, text=f"{var.get():.2f}", width=7)
            val_label.grid(row=0, column=2, sticky="e")
            self.slider_vars[f"link_len_{i}_label"] = val_label

            entry_var = tk.StringVar(value=f"{var.get():.2f}")
            self.slider_vars[f"link_len_{i}_entry"] = entry_var
            entry = ttk.Entry(row, textvariable=entry_var, width=7)
            entry.grid(row=0, column=3, sticky="ew", padx=(4, 0))
            entry.bind("<Return>", lambda _e, idx=i: self._on_link_length_entry(idx))
            entry.bind("<FocusOut>", lambda _e, idx=i: self._on_link_length_entry(idx))

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(frame, text="Link Masses (kg):", font=("Arial", 10, "bold")).pack(fill="x", pady=(0, 4))
        for i in range(2):
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            row.columnconfigure(1, weight=5)
            row.columnconfigure(3, weight=2)

            ttk.Label(row, text=f"Mass {i + 1}:", width=10).grid(row=0, column=0, sticky="w")
            var = tk.DoubleVar(value=float(self.config.masses[i]))
            self.slider_vars[f"mass_{i}"] = var
            ttk.Scale(
                row,
                from_=0.1,
                to=10.0,
                variable=var,
                orient="horizontal",
                command=lambda val, idx=i: self._on_mass_change(idx, val),
            ).grid(row=0, column=1, sticky="ew", padx=5)

            val_label = ttk.Label(row, text=f"{var.get():.2f}", width=7)
            val_label.grid(row=0, column=2, sticky="e")
            self.slider_vars[f"mass_{i}_label"] = val_label

            entry_var = tk.StringVar(value=f"{var.get():.2f}")
            self.slider_vars[f"mass_{i}_entry"] = entry_var
            entry = ttk.Entry(row, textvariable=entry_var, width=7)
            entry.grid(row=0, column=3, sticky="ew", padx=(4, 0))
            entry.bind("<Return>", lambda _e, idx=i: self._on_mass_entry(idx))
            entry.bind("<FocusOut>", lambda _e, idx=i: self._on_mass_entry(idx))

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(frame, text="Global Damping:", font=("Arial", 10, "bold")).pack(fill="x", pady=(0, 4))
        damp_row = ttk.Frame(frame)
        damp_row.pack(fill="x")

        damp_var = tk.DoubleVar(value=float(self.config.damping))
        self.slider_vars["damping"] = damp_var
        ttk.Scale(
            damp_row,
            from_=0.0,
            to=1.0,
            variable=damp_var,
            orient="horizontal",
            command=self._on_damping_change,
        ).pack(side="left", fill="x", expand=True, padx=5)
        damp_label = ttk.Label(damp_row, text=f"{damp_var.get():.2f}", width=7)
        damp_label.pack(side="left")
        self.slider_vars["damping_label"] = damp_label

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=8)

        # Match 2D app: Reset / Save / Load in a single row for arm properties.
        button_row = ttk.Frame(frame)
        button_row.pack(fill="x", pady=4)
        for col in range(3):
            button_row.columnconfigure(col, weight=1, uniform="props_btns")

        ttk.Button(button_row, text="Reset", command=self._on_reset_defaults, width=10).grid(
            row=0, column=0, sticky="ew", padx=2
        )
        ttk.Button(button_row, text="Save", command=self._on_save_config, width=10).grid(
            row=0, column=1, sticky="ew", padx=2
        )
        ttk.Button(button_row, text="Load", command=self._on_load_config, width=10).grid(
            row=0, column=2, sticky="ew", padx=2
        )

        return frame

    def _create_control_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="JOINT CONTROL", padding=10)

        ttk.Label(
            frame,
            text=(
                "Spherical Shoulder Limits (deg): X 0..120, Y -90..90, Z -90..120\n"
                "Edit Min/Max limits below each joint. Arrow keys: Left/Right selects joint, Up/Down changes selected joint"
            ),
            justify="left",
            font=("Arial", 9),
        ).pack(fill="x", pady=(0, 8))
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=5)

        mins_deg = np.asarray(self.config.joint_limits_deg_min, dtype=float)
        maxs_deg = np.asarray(self.config.joint_limits_deg_max, dtype=float)
        for i in range(self.config.dof):
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)

            ttk.Label(row, text=self.config.joint_names[i], width=6).pack(side="left")
            ttk.Button(row, text="-", width=3, command=lambda idx=i: self._increment_joint(idx, -2.0)).pack(
                side="left", padx=2
            )

            angle_deg = np.rad2deg(self.controller.angles[i])
            val = ttk.Label(row, text=f"{angle_deg:7.2f} deg", width=12, font=("Courier", 9))
            val.pack(side="left", padx=4)
            self.angle_labels[i] = val

            ttk.Button(row, text="+", width=3, command=lambda idx=i: self._increment_joint(idx, 2.0)).pack(
                side="left", padx=2
            )
            ttk.Label(row, text="Min", width=4).pack(side="left", padx=(6, 1))
            min_var = tk.StringVar(value=f"{mins_deg[i]:.1f}")
            self.slider_vars[f"joint_min_{i}_entry"] = min_var
            min_entry = ttk.Entry(row, textvariable=min_var, width=6)
            min_entry.pack(side="left", padx=(0, 4))
            min_entry.bind("<Return>", lambda _e, idx=i: self._on_joint_limit_entry(idx))
            min_entry.bind("<FocusOut>", lambda _e, idx=i: self._on_joint_limit_entry(idx))

            ttk.Label(row, text="Max", width=4).pack(side="left", padx=(2, 1))
            max_var = tk.StringVar(value=f"{maxs_deg[i]:.1f}")
            self.slider_vars[f"joint_max_{i}_entry"] = max_var
            max_entry = ttk.Entry(row, textvariable=max_var, width=6)
            max_entry.pack(side="left")
            max_entry.bind("<Return>", lambda _e, idx=i: self._on_joint_limit_entry(idx))
            max_entry.bind("<FocusOut>", lambda _e, idx=i: self._on_joint_limit_entry(idx))

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=8)

        actions_row = ttk.Frame(frame)
        actions_row.pack(fill="x", pady=3)
        for col in range(4):
            actions_row.columnconfigure(col, weight=1, uniform="ctrl_btn")

        self.record_button = ttk.Button(actions_row, text="Record", command=self._on_record, width=10)
        self.record_button.grid(row=0, column=0, sticky="ew", padx=1)
        ttk.Button(actions_row, text="Clear", command=self._on_clear_recording, width=10).grid(
            row=0, column=1, sticky="ew", padx=1
        )
        self.playback_button = ttk.Button(actions_row, text="Playback", command=self._on_playback, width=10)
        self.playback_button.grid(row=0, column=2, sticky="ew", padx=1)
        ttk.Button(actions_row, text="Reset Arm", command=self._on_reset_arm, width=10).grid(
            row=0, column=3, sticky="ew", padx=1
        )

        return frame

    def _create_model_selection_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="ARM MODEL SELECTION", padding=6)

        control_row = ttk.Frame(frame)
        control_row.pack(fill="x", pady=(0, 6))
        ttk.Button(control_row, text="Load Model", command=self._on_select_model).pack(
            side="left", padx=(0, 4), fill="x"
        )
        self.sim_toggle_button = ttk.Button(
            control_row, text="Run Simulation", command=self._on_toggle_simulation, state="disabled"
        )
        self.sim_toggle_button.pack(side="left", padx=(0, 4), fill="x")

        self.sim_model_label = ttk.Label(
            control_row, text="Model: not selected", anchor="w", font=("Courier", 8)
        )
        self.sim_model_label.pack(side="left", fill="x", expand=True, padx=(6, 0))

        details = ttk.LabelFrame(frame, text="Model Details", padding=4)
        details.pack(fill="x")
        self.config_details_text = tk.Text(
            details, height=4, wrap="word", state="disabled", font=("Courier", 8)
        )
        self.config_details_text.pack(fill="x")
        self._update_config_details()
        return frame

    def _create_visualization_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="ARM VISUALIZATION", padding=5)
        self.fig = Figure(figsize=(7.2, 6.8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0.03, right=0.98, bottom=0.05, top=0.95)
        self.ax.set_proj_type("persp")
        # Keep the world frame fixed for this application: Y is vertical.
        self._apply_fixed_camera_view()

        # Disable mouse-driven axis/camera movement.
        try:
            self.ax.mouse_init(rotate_btn=None, zoom_btn=None)
        except Exception:
            pass

        self.canvas_agg = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_agg.draw()
        canvas_widget = self.canvas_agg.get_tk_widget()
        for seq in (
            "<Button-1>",
            "<ButtonRelease-1>",
            "<B1-Motion>",
            "<Button-2>",
            "<ButtonRelease-2>",
            "<B2-Motion>",
            "<Button-3>",
            "<ButtonRelease-3>",
            "<B3-Motion>",
            "<MouseWheel>",
            "<Button-4>",
            "<Button-5>",
        ):
            canvas_widget.bind(seq, lambda _e: "break")
        canvas_widget.pack(fill="both", expand=True)
        return frame

    def _apply_fixed_camera_view(self):
        """Apply a fixed world camera with Y as vertical axis."""
        try:
            self.ax.view_init(elev=self._view_elev, azim=self._view_azim, vertical_axis="y")
        except TypeError:
            # Fallback for older Matplotlib without vertical_axis kwarg.
            self.ax.view_init(elev=self._view_elev, azim=self._view_azim)

    def _create_metrics_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="METRICS & STATUS", padding=10)
        self.metrics_text = tk.Text(
            frame, height=20, width=34, font=("Courier", 9), bg="black", fg="white", state="disabled"
        )
        self.metrics_text.pack(fill="both", expand=True)
        return frame

    def _create_torque_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="TORQUE vs TIME", padding=5)
        self.torque_fig = Figure(figsize=(5.6, 2.8), dpi=100)
        self.torque_ax = self.torque_fig.add_subplot(111)
        self.torque_fig.subplots_adjust(left=0.16, right=0.97, bottom=0.22, top=0.88)
        self.torque_ax.set_title("Shoulder/Elbow Torque")
        self.torque_ax.set_xlabel("Time (s)")
        self.torque_ax.set_ylabel("Torque (Nm)")
        self.torque_ax.grid(True, alpha=0.3)
        self.torque_canvas = FigureCanvasTkAgg(self.torque_fig, master=frame)
        self.torque_canvas.draw()
        self.torque_canvas.get_tk_widget().pack(fill="both", expand=True)
        return frame

    def _create_velocity_acceleration_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="VELOCITY & ACCELERATION PLOT", padding=5)
        self.velacc_fig = Figure(figsize=(4.2, 3.1), dpi=100)
        self.velacc_ax_vel = self.velacc_fig.add_subplot(211)
        self.velacc_ax_acc = self.velacc_fig.add_subplot(212, sharex=self.velacc_ax_vel)
        self.velacc_fig.subplots_adjust(left=0.16, right=0.97, bottom=0.14, top=0.92, hspace=0.38)

        self.velacc_ax_vel.set_title("End-Effector Velocity")
        self.velacc_ax_vel.set_ylabel("m/s")
        self.velacc_ax_vel.grid(True, alpha=0.3)

        self.velacc_ax_acc.set_title("End-Effector Acceleration")
        self.velacc_ax_acc.set_xlabel("Time (s)")
        self.velacc_ax_acc.set_ylabel("m/s^2")
        self.velacc_ax_acc.grid(True, alpha=0.3)

        self.velacc_canvas = FigureCanvasTkAgg(self.velacc_fig, master=frame)
        self.velacc_canvas.draw()
        self.velacc_canvas.get_tk_widget().pack(fill="both", expand=True)
        return frame

    def _sync_ui_to_config(self):
        for i in range(2):
            val = float(np.clip(self.config.link_lengths[i], 0.1, 2.0))
            self.config.link_lengths[i] = val
            self.slider_vars[f"link_len_{i}"].set(val)
            self.slider_vars[f"link_len_{i}_label"].config(text=f"{val:.2f}")
            self.slider_vars[f"link_len_{i}_entry"].set(f"{val:.2f}")

        for i in range(2):
            val = float(np.clip(self.config.masses[i], 0.1, 10.0))
            self.config.masses[i] = val
            self.slider_vars[f"mass_{i}"].set(val)
            self.slider_vars[f"mass_{i}_label"].config(text=f"{val:.2f}")
            self.slider_vars[f"mass_{i}_entry"].set(f"{val:.2f}")

        damp = float(np.clip(self.config.damping, 0.0, 1.0))
        self.config.damping = damp
        self.slider_vars["damping"].set(damp)
        self.slider_vars["damping_label"].config(text=f"{damp:.2f}")

        for i in range(self.config.dof):
            self.angle_labels[i].config(text=f"{np.rad2deg(self.controller.angles[i]):7.2f} deg")
            min_key = f"joint_min_{i}_entry"
            max_key = f"joint_max_{i}_entry"
            if min_key in self.slider_vars:
                self.slider_vars[min_key].set(f"{float(self.config.joint_limits_deg_min[i]):.1f}")
            if max_key in self.slider_vars:
                self.slider_vars[max_key].set(f"{float(self.config.joint_limits_deg_max[i]):.1f}")

        if self.sim_model_label is not None:
            if self.sim_model_path:
                self.sim_model_label.config(
                    text=f"Model: {Path(self.sim_model_path).name} ({self.sim_algorithm or 'UNKNOWN'})"
                )
            else:
                self.sim_model_label.config(text="Model: not selected")
        if self.sim_toggle_button is not None:
            self.sim_toggle_button.config(state="normal")
        self._update_config_details()

    def _update_config_details(self):
        if self.config_details_text is None:
            return
        metadata = self.sim_model_metadata if isinstance(self.sim_model_metadata, dict) else {}
        episodes = "-"
        best_reward = "-"
        timestamp = "-"
        policy_name = "-"
        if metadata:
            history = metadata.get("training_history", {})
            if isinstance(history, dict):
                rewards = history.get("episode_rewards", [])
                if isinstance(rewards, list):
                    episodes = len(rewards)
            best_reward = metadata.get("best_reward", "-")
            timestamp = metadata.get("timestamp", "-")
            policy_name = str(metadata.get("policy", "MlpPolicy"))

        if self.sim_model_path:
            model_block = (
                f"Policy Model: {Path(self.sim_model_path).name}.zip\n"
                f"Algorithm: {self.sim_algorithm or 'UNKNOWN'}\n"
                f"Policy: {policy_name}\n"
                f"Episodes: {episodes}\n"
                f"Best Reward: {best_reward}\n"
                f"Saved: {timestamp}"
            )
        else:
            model_block = "Policy Model: not selected"

        details = (
            f"Name: {self.config.name}\n"
            f"J1x: {self.config.joint_limits_deg_min[0]:.1f}..{self.config.joint_limits_deg_max[0]:.1f} deg\n"
            f"J1y: {self.config.joint_limits_deg_min[1]:.1f}..{self.config.joint_limits_deg_max[1]:.1f} deg\n"
            f"J1z: {self.config.joint_limits_deg_min[2]:.1f}..{self.config.joint_limits_deg_max[2]:.1f} deg\n"
            f"J2: {self.config.joint_limits_deg_min[3]:.1f}..{self.config.joint_limits_deg_max[3]:.1f} deg\n"
            "Shoulder anchor: (0.0, 0.0, 0.0) [fixed]\n"
            "----------------------------------------\n"
            f"{model_block}"
        )
        self.config_details_text.config(state="normal")
        self.config_details_text.delete("1.0", tk.END)
        self.config_details_text.insert("1.0", details)
        self.config_details_text.config(state="disabled")

    def _on_joint_limit_entry(self, joint_id: int):
        """Update editable joint limits from min/max entry boxes."""
        min_key = f"joint_min_{joint_id}_entry"
        max_key = f"joint_max_{joint_id}_entry"
        cur_min = float(self.config.joint_limits_deg_min[joint_id])
        cur_max = float(self.config.joint_limits_deg_max[joint_id])

        try:
            new_min = float(str(self.slider_vars[min_key].get()).strip())
        except (ValueError, KeyError):
            new_min = cur_min
        try:
            new_max = float(str(self.slider_vars[max_key].get()).strip())
        except (ValueError, KeyError):
            new_max = cur_max

        new_min = float(np.clip(new_min, -360.0, 360.0))
        new_max = float(np.clip(new_max, -360.0, 360.0))
        if new_min > new_max:
            mid = 0.5 * (new_min + new_max)
            new_min = mid
            new_max = mid

        self.config.joint_limits_deg_min[joint_id] = new_min
        self.config.joint_limits_deg_max[joint_id] = new_max

        self.slider_vars[min_key].set(f"{new_min:.1f}")
        self.slider_vars[max_key].set(f"{new_max:.1f}")

        self.controller.angles = self.config.clamp_angles_rad(self.controller.angles)
        self._compute_positions()
        for i in range(self.config.dof):
            self.angle_labels[i].config(text=f"{np.rad2deg(self.controller.angles[i]):7.2f} deg")
        self._update_config_details()

    def _compute_positions(self):
        self.controller.positions = ArmKinematics3D.forward_kinematics(self.controller.angles, self.config)

    def _on_link_length_change(self, index: int, value):
        self.config.link_lengths[index] = float(value)
        self.slider_vars[f"link_len_{index}_label"].config(text=f"{float(value):.2f}")
        self.slider_vars[f"link_len_{index}_entry"].set(f"{float(value):.2f}")
        self._compute_positions()

    def _on_mass_change(self, index: int, value):
        self.config.masses[index] = float(value)
        self.slider_vars[f"mass_{index}_label"].config(text=f"{float(value):.2f}")
        self.slider_vars[f"mass_{index}_entry"].set(f"{float(value):.2f}")

    def _on_link_length_entry(self, index: int):
        key = f"link_len_{index}_entry"
        current = float(self.config.link_lengths[index])
        try:
            value = float(self.slider_vars[key].get().strip())
        except ValueError:
            value = current
        value = float(np.clip(value, 0.1, 2.0))
        self.slider_vars[f"link_len_{index}"].set(value)
        self.slider_vars[key].set(f"{value:.2f}")

    def _on_mass_entry(self, index: int):
        key = f"mass_{index}_entry"
        current = float(self.config.masses[index])
        try:
            value = float(self.slider_vars[key].get().strip())
        except ValueError:
            value = current
        value = float(np.clip(value, 0.1, 10.0))
        self.slider_vars[f"mass_{index}"].set(value)
        self.slider_vars[key].set(f"{value:.2f}")

    def _on_damping_change(self, value):
        self.config.damping = float(value)
        self.slider_vars["damping_label"].config(text=f"{float(value):.2f}")

    def _increment_joint(self, joint_id: int, delta_deg: float):
        if self.playing_back:
            self.playing_back = False
            self.playback_button.config(text="Playback")
        self.controller.increment_joint(joint_id, np.deg2rad(delta_deg))
        self.angle_labels[joint_id].config(text=f"{np.rad2deg(self.controller.angles[joint_id]):7.2f} deg")

    def _on_record(self):
        self.recording = not self.recording
        if self.recording:
            self.recorder.start_recording()
            self.record_button.config(text="Stop REC")
        else:
            self.recorder.stop_recording()
            self.record_button.config(text="Record")

    def _on_clear_recording(self):
        self.recorder.clear_frames()
        self.trajectory_points = []

    def _on_playback(self):
        if self.recorder.get_num_frames() == 0:
            messagebox.showwarning("No Recording", "No recorded frames available for playback.")
            return
        if self.simulation_active:
            self._stop_simulation()
        self.playing_back = not self.playing_back
        if self.playing_back:
            self.playback_frames = self.recorder.get_frames()
            self.playback_index = 0
            self.playback_button.config(text="Stop Playback")
        else:
            self.playback_button.config(text="Playback")

    def _on_toggle_simulation(self):
        if self.simulation_active:
            self._stop_simulation()
        else:
            self._start_simulation()

    def _start_simulation(self):
        if self.playing_back:
            self.playing_back = False
            self.playback_button.config(text="Playback")
        self.simulation_active = True
        self._reset_histories()
        self.prev_angles = self.controller.angles.copy()
        self.prev_velocities = np.zeros(self.config.dof, dtype=float)
        self.prev_update_time = time()
        if self.sim_toggle_button is not None:
            self.sim_toggle_button.config(text="Stop Simulation")

    def _stop_simulation(self):
        self.simulation_active = False
        self._reset_histories()
        if self.sim_toggle_button is not None:
            self.sim_toggle_button.config(text="Run Simulation")

    def _on_reset_arm(self):
        if self.simulation_active:
            self._stop_simulation()
        self.playing_back = False
        self.playback_button.config(text="Playback")
        self.controller.set_home_position()
        self.prev_angles = self.controller.angles.copy()
        self.prev_velocities = np.zeros(self.config.dof, dtype=float)
        self._reset_histories()
        for i in range(self.config.dof):
            self.angle_labels[i].config(text=f"{np.rad2deg(self.controller.angles[i]):7.2f} deg")

    def _on_reset_defaults(self):
        if self.simulation_active:
            self._stop_simulation()
        self.config = ArmConfiguration3D.get_default()
        self._force_shoulder_origin()
        self.controller = ArmController3D(self.config)
        self.prev_angles = self.controller.angles.copy()
        self.prev_velocities = np.zeros(self.config.dof, dtype=float)
        self._reset_histories()
        self.sim_model_path = ""
        self.sim_algorithm = ""
        self.sim_model_metadata = {}
        self._sync_ui_to_config()

    def _on_save_config(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            self.config.to_json(filepath)
            messagebox.showinfo("Saved", f"3D arm configuration saved:\n{filepath}")
        except Exception as exc:
            messagebox.showerror("Save Error", f"Failed to save configuration:\n{exc}")

    def _on_select_model(self):
        """Load a trained policy model (.zip) for simulation controls."""
        filepath = filedialog.askopenfilename(
            title="Select trained model (.zip)",
            filetypes=[("Model files", "*.zip"), ("All files", "*.*")],
        )
        if not filepath:
            return

        model_base = self._normalize_model_base(filepath)
        model_zip = Path(f"{model_base}.zip")
        if not model_zip.exists():
            messagebox.showerror("Load Error", "Selected model artifact not found.")
            return

        self.sim_model_path = model_base
        self.sim_algorithm = self._get_algorithm_from_metadata(model_base)
        self.sim_model_metadata = self._read_model_metadata(model_base)

        if self.sim_model_label is not None:
            self.sim_model_label.config(
                text=f"Model: {Path(model_base).name} ({self.sim_algorithm or 'UNKNOWN'})"
            )
        if self.sim_toggle_button is not None:
            self.sim_toggle_button.config(state="normal")
        self._update_config_details()
        messagebox.showinfo("Loaded", f"Loaded trained model:\n{model_zip.name}")

    def _on_load_config(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not filepath:
            return
        self._load_config_from_filepath(filepath, show_popup=True)

    def _load_config_from_filepath(self, filepath: str, show_popup: bool = True):
        """Load arm config from disk and refresh all UI dependencies."""
        try:
            if self.simulation_active:
                self._stop_simulation()
            loaded = ArmConfiguration3D.from_json(filepath)
            self.config = loaded
            self._force_shoulder_origin()
            self.controller = ArmController3D(self.config)
            self.prev_angles = self.controller.angles.copy()
            self.prev_velocities = np.zeros(self.config.dof, dtype=float)
            self._reset_histories()
            self._sync_ui_to_config()
            self._compute_positions()
            self._update_visualization()
            self._update_metrics_display()
            if self.sim_toggle_button is not None:
                self.sim_toggle_button.config(state="normal")
            if show_popup:
                messagebox.showinfo("Loaded", f"Loaded 3D configuration:\n{Path(filepath).name}")
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load configuration:\n{exc}")

    def _reset_histories(self):
        self.torque_time_hist.clear()
        self.shoulder_torque_hist.clear()
        self.elbow_torque_hist.clear()
        self.torque_plot_time = 0.0
        self.latest_shoulder_torque = 0.0
        self.latest_elbow_torque = 0.0

        self.ee_kin_time_hist.clear()
        self.ee_vx_hist.clear()
        self.ee_vy_hist.clear()
        self.ee_vz_hist.clear()
        self.ee_ax_hist.clear()
        self.ee_ay_hist.clear()
        self.ee_az_hist.clear()
        self._ee_prev_pos = None
        self._ee_prev_vel = np.zeros(3, dtype=float)
        self._ee_prev_time = None
        self._ee_plot_time = 0.0
        self.prev_update_time = time()

    def _estimate_joint_torques(
        self, angles: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray
    ) -> tuple[float, float]:
        inertias = np.asarray(self.config.inertias, dtype=float)
        damping = float(self.config.damping)
        q = np.asarray(angles, dtype=float)
        qd = np.asarray(velocities, dtype=float)
        qdd = np.asarray(accelerations, dtype=float)

        torques = inertias * qdd + damping * qd

        g = 9.81
        l1, l2 = [float(v) for v in self.config.link_lengths]
        m1, m2 = [float(v) for v in self.config.masses]
        torques[0] += g * (m1 * (0.5 * l1) + m2 * (l1 + 0.5 * l2)) * np.sin(q[0])
        torques[3] += g * (m2 * (0.5 * l2)) * np.sin(q[3])

        shoulder_tau = float(np.linalg.norm(torques[:3]))
        elbow_tau = float(torques[3])
        return shoulder_tau, elbow_tau

    def _append_torque_sample(self, shoulder_torque: float, elbow_torque: float, dt: float):
        self.torque_plot_time += max(float(dt), 0.0)
        self.torque_time_hist.append(self.torque_plot_time)
        self.shoulder_torque_hist.append(float(shoulder_torque))
        self.elbow_torque_hist.append(float(elbow_torque))
        self.latest_shoulder_torque = float(shoulder_torque)
        self.latest_elbow_torque = float(elbow_torque)

    def _update_torque_plot(self):
        if self.torque_ax is None or self.torque_canvas is None:
            return
        self.torque_ax.clear()
        if not self.simulation_active:
            self.torque_ax.set_title("Shoulder/Elbow Torque")
            self.torque_ax.set_xlabel("Time (s)")
            self.torque_ax.set_ylabel("Torque (Nm)")
            self.torque_ax.grid(True, alpha=0.3)
            self.torque_ax.text(
                0.5,
                0.5,
                "Start simulation to activate plot",
                ha="center",
                va="center",
                transform=self.torque_ax.transAxes,
                fontsize=8,
            )
            self.torque_canvas.draw_idle()
            return

        if self.torque_time_hist:
            t = np.asarray(self.torque_time_hist, dtype=float)
            shoulder = np.asarray(self.shoulder_torque_hist, dtype=float)
            elbow = np.asarray(self.elbow_torque_hist, dtype=float)
            self.torque_ax.plot(t, shoulder, linewidth=1.4, label="Shoulder Torque")
            self.torque_ax.plot(t, elbow, linewidth=1.4, label="Elbow Torque")
            self.torque_ax.legend(loc="upper right", fontsize=8)
            if t[-1] > 20.0:
                self.torque_ax.set_xlim(t[-1] - 20.0, t[-1])
        self.torque_ax.set_title("Shoulder/Elbow Torque")
        self.torque_ax.set_xlabel("Time (s)")
        self.torque_ax.set_ylabel("Torque (Nm)")
        self.torque_ax.grid(True, alpha=0.3)
        self.torque_canvas.draw_idle()

    def _append_ee_kin_sample(self, now_s: float):
        pos = np.asarray(self.controller.get_end_effector_position(), dtype=float)
        if self._ee_prev_time is None or self._ee_prev_pos is None:
            dt = 0.0
            vel = np.zeros(3, dtype=float)
            acc = np.zeros(3, dtype=float)
        else:
            dt = max(now_s - float(self._ee_prev_time), 1e-8)
            vel = (pos - self._ee_prev_pos) / dt
            acc = (vel - self._ee_prev_vel) / dt
            self._ee_prev_vel = vel.copy()

        self._ee_plot_time += dt
        self.ee_kin_time_hist.append(float(self._ee_plot_time))
        self.ee_vx_hist.append(float(vel[0]))
        self.ee_vy_hist.append(float(vel[1]))
        self.ee_vz_hist.append(float(vel[2]))
        self.ee_ax_hist.append(float(acc[0]))
        self.ee_ay_hist.append(float(acc[1]))
        self.ee_az_hist.append(float(acc[2]))

        self._ee_prev_pos = pos.copy()
        self._ee_prev_time = float(now_s)

    def _update_vel_acc_plot(self):
        if self.velacc_ax_vel is None or self.velacc_ax_acc is None or self.velacc_canvas is None:
            return

        self.velacc_ax_vel.clear()
        self.velacc_ax_acc.clear()
        if not self.simulation_active:
            self.velacc_ax_vel.set_title("End-Effector Velocity")
            self.velacc_ax_vel.set_ylabel("m/s")
            self.velacc_ax_vel.grid(True, alpha=0.3)
            self.velacc_ax_vel.text(
                0.5,
                0.5,
                "Start simulation to activate plot",
                ha="center",
                va="center",
                transform=self.velacc_ax_vel.transAxes,
                fontsize=8,
            )
            self.velacc_ax_acc.set_title("End-Effector Acceleration")
            self.velacc_ax_acc.set_xlabel("Time (s)")
            self.velacc_ax_acc.set_ylabel("m/s^2")
            self.velacc_ax_acc.grid(True, alpha=0.3)
            self.velacc_canvas.draw_idle()
            return

        if self.ee_kin_time_hist:
            t = np.asarray(self.ee_kin_time_hist, dtype=float)
            vx = np.asarray(self.ee_vx_hist, dtype=float)
            vy = np.asarray(self.ee_vy_hist, dtype=float)
            vz = np.asarray(self.ee_vz_hist, dtype=float)
            ax = np.asarray(self.ee_ax_hist, dtype=float)
            ay = np.asarray(self.ee_ay_hist, dtype=float)
            az = np.asarray(self.ee_az_hist, dtype=float)

            self.velacc_ax_vel.plot(t, vx, linewidth=1.1, label="Vx")
            self.velacc_ax_vel.plot(t, vy, linewidth=1.1, label="Vy")
            self.velacc_ax_vel.plot(t, vz, linewidth=1.1, label="Vz")
            self.velacc_ax_vel.legend(loc="upper right", fontsize=7)

            self.velacc_ax_acc.plot(t, ax, linewidth=1.1, label="Ax")
            self.velacc_ax_acc.plot(t, ay, linewidth=1.1, label="Ay")
            self.velacc_ax_acc.plot(t, az, linewidth=1.1, label="Az")
            self.velacc_ax_acc.legend(loc="upper right", fontsize=7)

            if t[-1] > 20.0:
                self.velacc_ax_vel.set_xlim(t[-1] - 20.0, t[-1])
                self.velacc_ax_acc.set_xlim(t[-1] - 20.0, t[-1])
        else:
            self.velacc_ax_vel.text(
                0.5,
                0.5,
                "Awaiting motion...",
                ha="center",
                va="center",
                transform=self.velacc_ax_vel.transAxes,
                fontsize=8,
            )

        self.velacc_ax_vel.set_title("End-Effector Velocity")
        self.velacc_ax_vel.set_ylabel("m/s")
        self.velacc_ax_vel.grid(True, alpha=0.3)

        self.velacc_ax_acc.set_title("End-Effector Acceleration")
        self.velacc_ax_acc.set_xlabel("Time (s)")
        self.velacc_ax_acc.set_ylabel("m/s^2")
        self.velacc_ax_acc.grid(True, alpha=0.3)
        self.velacc_canvas.draw_idle()

    def _calculate_axis_limits(self):
        shoulder = np.asarray(self.config.shoulder_position, dtype=float)
        reach = float(sum(self.config.link_lengths))
        margin = max(0.2, 0.2 * reach)
        min_xyz = shoulder - (reach + margin)
        max_xyz = shoulder + (reach + margin)
        min_xyz = np.minimum(min_xyz, np.array([-0.1, -0.1, -0.1]))
        max_xyz = np.maximum(max_xyz, np.array([0.1, 0.1, 0.1]))
        return min_xyz, max_xyz

    def _update_visualization(self):
        self.ax.clear()
        pos = self.controller.positions
        origin = pos[0]
        shoulder = pos[1]
        arm = pos[1:]

        self.ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], "b-o", linewidth=2.2, markersize=5, label="Arm links")
        self.ax.plot(
            [origin[0], shoulder[0]],
            [origin[1], shoulder[1]],
            [origin[2], shoulder[2]],
            "k--",
            linewidth=1,
            alpha=0.5,
            label="Origin to shoulder",
        )

        self.ax.scatter(origin[0], origin[1], origin[2], color="black", s=36, label="Origin")
        self.ax.scatter(shoulder[0], shoulder[1], shoulder[2], color="green", s=70, label="Fixed shoulder")
        self.ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color="red", s=90, marker="*", label="End-effector")

        # Fixed world-frame axis triad at origin for clear reference.
        axis_len = 0.18
        self.ax.quiver(0.0, 0.0, 0.0, axis_len, 0.0, 0.0, color="tab:blue", arrow_length_ratio=0.25, linewidth=1.4)
        self.ax.quiver(0.0, 0.0, 0.0, 0.0, axis_len, 0.0, color="tab:green", arrow_length_ratio=0.25, linewidth=1.4)
        self.ax.quiver(0.0, 0.0, 0.0, 0.0, 0.0, axis_len, color="tab:purple", arrow_length_ratio=0.25, linewidth=1.4)
        self.ax.text(axis_len + 0.01, 0.0, 0.0, "X", color="tab:blue", fontsize=8)
        self.ax.text(0.0, axis_len + 0.01, 0.0, "Y", color="tab:green", fontsize=8)
        self.ax.text(0.0, 0.0, axis_len + 0.01, "Z", color="tab:purple", fontsize=8)

        # Gravity direction is fixed in world frame: negative Y.
        g_anchor = shoulder + np.array([0.0, 0.25, 0.0], dtype=float)
        g_vec = np.array([0.0, -0.20, 0.0], dtype=float)
        self.ax.quiver(
            g_anchor[0],
            g_anchor[1],
            g_anchor[2],
            g_vec[0],
            g_vec[1],
            g_vec[2],
            color="crimson",
            linewidth=1.8,
            arrow_length_ratio=0.35,
        )
        self.ax.text(g_anchor[0], g_anchor[1] + 0.03, g_anchor[2], "g (-Y)", color="crimson", fontsize=8)

        if self.show_trajectory and self.trajectory_points:
            traj = np.asarray(self.trajectory_points, dtype=float)
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "m--", linewidth=1.2, alpha=0.7, label="Trajectory")

        self.ax.set_xlabel("X (Anterior/Posterior)")
        self.ax.set_ylabel("Y (Vertical)")
        self.ax.set_zlabel("Z (Medial/Lateral)")
        self.ax.set_title(f"3D Arm View - FPS: {self.fps_counter:.0f}")
        self.ax.legend(loc="upper left", fontsize=8)

        mins, maxs = self._calculate_axis_limits()
        self.ax.set_xlim(mins[0], maxs[0])
        self.ax.set_ylim(mins[1], maxs[1])
        self.ax.set_zlim(mins[2], maxs[2])
        self.ax.set_box_aspect((1, 1, 1))
        self._apply_fixed_camera_view()
        self.canvas_agg.draw_idle()

    def _update_metrics_display(self):
        ee = self.controller.get_end_effector_position()
        q_deg = np.rad2deg(self.controller.angles)
        qd_deg = np.rad2deg(self.controller.velocities)
        shoulder = np.asarray(self.config.shoulder_position, dtype=float)

        metrics_text = f"""
CURRENT STATE
================================
End-effector: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m
Shoulder (fixed): ({shoulder[0]:.3f}, {shoulder[1]:.3f}, {shoulder[2]:.3f}) m

Joint Angles (deg):
{", ".join(f"{v:6.2f}" for v in q_deg)}

Joint Velocities (deg/s):
{", ".join(f"{v:6.2f}" for v in qd_deg)}

JOINT LIMITS (deg)
================================
X shoulder: {self.config.joint_limits_deg_min[0]:.0f} .. {self.config.joint_limits_deg_max[0]:.0f}
Y shoulder: {self.config.joint_limits_deg_min[1]:.0f} .. {self.config.joint_limits_deg_max[1]:.0f}
Z shoulder: {self.config.joint_limits_deg_min[2]:.0f} .. {self.config.joint_limits_deg_max[2]:.0f}

RECORDING
================================
Status: {"REC" if self.recording else "IDLE"}
Frames: {self.recorder.get_num_frames()}
Playback: {"ON" if self.playing_back else "OFF"}
Simulation: {"RUNNING" if self.simulation_active else "IDLE"}

TORQUE (Nm)
================================
Shoulder torque: {self.latest_shoulder_torque:7.3f}
Elbow torque:    {self.latest_elbow_torque:7.3f}

PERFORMANCE
================================
FPS: {self.fps_counter:5.1f}
Trajectory points: {len(self.trajectory_points)}
        """
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert("1.0", metrics_text)
        self.metrics_text.config(state="disabled")

    def _update_dynamic_state(self):
        if not self.simulation_active:
            return
        now = time()
        dt = max(now - self.prev_update_time, 1e-8)
        angles = self.controller.angles.copy()
        velocities = (angles - self.prev_angles) / dt
        accelerations = (velocities - self.prev_velocities) / dt

        self.controller.velocities = velocities
        shoulder_tau, elbow_tau = self._estimate_joint_torques(angles, velocities, accelerations)
        self._append_torque_sample(shoulder_tau, elbow_tau, dt)
        self._append_ee_kin_sample(now)

        self.prev_angles = angles
        self.prev_velocities = velocities
        self.prev_update_time = now

    def _handle_keyboard(self, event):
        if event.keysym == "Left":
            self.selected_joint = max(0, self.selected_joint - 1)
        elif event.keysym == "Right":
            self.selected_joint = min(self.config.dof - 1, self.selected_joint + 1)
        elif event.keysym == "Up":
            self._increment_joint(self.selected_joint, 2.0)
        elif event.keysym == "Down":
            self._increment_joint(self.selected_joint, -2.0)
        elif event.char == "t":
            self.show_trajectory = not self.show_trajectory

    def create_window(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        title = ttk.Label(main_frame, text="RL Arm Motion - 3D Interactive Controller", font=("Arial", 16, "bold"))
        title.pack(fill="x", pady=(0, 10))
        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=5)

        content_pane = ttk.Panedwindow(main_frame, orient="horizontal")
        content_pane.pack(fill="both", expand=True)

        left_frame = ttk.Frame(content_pane)
        center_frame = ttk.Frame(content_pane)
        right_frame = ttk.Frame(content_pane)
        left_frame.configure(width=400)
        center_frame.configure(width=400)
        right_frame.configure(width=200)
        content_pane.add(left_frame, weight=3)
        content_pane.add(center_frame, weight=5)
        content_pane.add(right_frame, weight=2)

        left_pane = ttk.Panedwindow(left_frame, orient="vertical")
        left_pane.pack(fill="both", expand=True)
        props_host = ttk.Frame(left_pane)
        torque_host = ttk.Frame(left_pane)
        left_pane.add(props_host, weight=1)
        left_pane.add(torque_host, weight=1)

        center_pane = ttk.Panedwindow(center_frame, orient="vertical")
        center_pane.pack(fill="both", expand=True)
        ctrl_host = ttk.Frame(center_pane)
        model_host = ttk.Frame(center_pane)
        viz_host = ttk.Frame(center_pane)
        center_pane.add(ctrl_host, weight=30)
        center_pane.add(model_host, weight=20)
        center_pane.add(viz_host, weight=50)

        right_pane = ttk.Panedwindow(right_frame, orient="vertical")
        right_pane.pack(fill="both", expand=True)
        metrics_host = ttk.Frame(right_pane)
        velacc_host = ttk.Frame(right_pane)
        right_pane.add(metrics_host, weight=1)
        right_pane.add(velacc_host, weight=1)

        self._create_properties_frame(props_host).pack(fill="both", expand=True)
        self._create_torque_frame(torque_host).pack(fill="both", expand=True)
        self._create_control_frame(ctrl_host).pack(fill="both", expand=True)
        self._create_model_selection_frame(model_host).pack(fill="both", expand=True)
        self._create_visualization_frame(viz_host).pack(fill="both", expand=True)
        self._create_metrics_frame(metrics_host).pack(fill="both", expand=True)
        self._create_velocity_acceleration_frame(velacc_host).pack(fill="both", expand=True)

        bottom = ttk.Frame(main_frame)
        bottom.pack(fill="x", pady=(10, 0))
        ttk.Separator(bottom, orient="horizontal").pack(fill="x", pady=(0, 10))
        buttons = ttk.Frame(bottom)
        buttons.pack()
        ttk.Button(buttons, text="Reset Defaults", command=self._on_reset_defaults, width=16).pack(
            side="left", padx=6
        )
        ttk.Button(buttons, text="Exit", command=self.on_closing, width=12).pack(side="left", padx=6)

        def _set_initial_sashes(retries=10):
            try:
                total_w = int(content_pane.winfo_width())
                if total_w < 900 and retries > 0:
                    self.root.after(80, lambda: _set_initial_sashes(retries - 1))
                    return

                # Pane widths: left/center/right = 0.3 / 0.5 / 0.2.
                content_pane.sashpos(0, int(total_w * 0.30))
                content_pane.sashpos(1, int(total_w * 0.80))

                lh = int(left_pane.winfo_height())
                ch = int(center_pane.winfo_height())
                rh = int(right_pane.winfo_height())
                if lh > 200:
                    left_pane.sashpos(0, int(lh * 0.50))
                if ch > 300:
                    center_pane.sashpos(0, int(ch * 0.30))
                    center_pane.sashpos(1, int(ch * 0.50))
                if rh > 200:
                    right_pane.sashpos(0, int(rh * 0.50))
            except Exception:
                if retries > 0:
                    self.root.after(80, lambda: _set_initial_sashes(retries - 1))

        self.root.after(120, _set_initial_sashes)

    def run(self):
        self.create_window()
        self.root.bind("<Up>", self._handle_keyboard)
        self.root.bind("<Down>", self._handle_keyboard)
        self.root.bind("<Left>", self._handle_keyboard)
        self.root.bind("<Right>", self._handle_keyboard)
        self.root.bind("<t>", self._handle_keyboard)

        def update_loop():
            if not self.running:
                return

            if self.playing_back and self.playback_frames:
                if self.playback_index < len(self.playback_frames):
                    state = self.playback_frames[self.playback_index]
                    self.controller.apply_state(state)
                    self.playback_index += 1
                else:
                    self.playing_back = False
                    self.playback_button.config(text="Playback")

            if self.recording:
                self.recorder.record_frame(self.controller.get_state(timestamp=time()))

            self.trajectory_points.append(np.asarray(self.controller.get_end_effector_position(), dtype=float))
            if len(self.trajectory_points) > 2000:
                self.trajectory_points = self.trajectory_points[-2000:]

            if self.simulation_active:
                self._update_dynamic_state()
            self._update_visualization()
            self._update_metrics_display()
            self._update_torque_plot()
            self._update_vel_acc_plot()

            now = time()
            if now - self.last_frame_time >= 1.0:
                self.fps_counter = float(self.frame_count)
                self.frame_count = 0
                self.last_frame_time = now
            else:
                self.frame_count += 1

            self.root.after(50, update_loop)  # 20 FPS cadence

        update_loop()
        self.root.mainloop()

    def on_closing(self):
        self.running = False
        self.root.destroy()


def main():
    gui = ArmControllerGUI3D(ArmConfiguration3D.get_default())
    gui.run()


if __name__ == "__main__":
    main()
