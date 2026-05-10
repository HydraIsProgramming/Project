"""Combined launcher for the 2-D RL Arm Motion GUIs.

Running ``python -m rl_armMotion.two_d.gui`` opens a small launcher window
from which the user can:

  - Open the interactive arm-control GUI (rl_armMotion.two_d.gui.app),
    which is the visualisation/teleoperation front-end.
  - Configure and start the training GUI (rl_armMotion.two_d.gui.training_gui)
    with algorithm choice, timestep budget, and save directory exposed as
    form fields.

Each GUI is launched as an independent subprocess of the same Python
interpreter, so each gets its own clean Tk root (Tkinter does not support
two simultaneous Tk() roots in the same process). The launcher window
stays open after spawning a child so the user can launch additional
sessions concurrently.

The existing GUI entry points
    python -m rl_armMotion.two_d.gui.app
    python -m rl_armMotion.two_d.gui.training_gui [args]
remain unchanged and can still be invoked directly. This launcher simply
provides a single, friendlier dispatch point.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk
from typing import List, Optional


# Default training-GUI parameters. Match the values shown in the project
# README and the CP493 progress report so the launcher's defaults match
# the documented protocol.
DEFAULT_ALGORITHMS: List[str] = ["SAC", "PPO", "A2C"]
DEFAULT_TIMESTEPS: int = 100_000
DEFAULT_SAVE_DIR: str = "./project_assets/outputs/fischer_session"


class LauncherApp:
    """A small Tk launcher window that dispatches to the arm-control and
    training GUIs as separate subprocesses."""

    PAD = 8

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("RL Arm Motion - Launcher")
        self.root.geometry("520x460")
        self.root.minsize(460, 420)

        # Tracks live subprocesses so the launcher can report their status.
        self._launched: List[subprocess.Popen] = []

        # Form variables for the training-GUI panel.
        self.algorithm_var = tk.StringVar(value="SAC")
        self.timesteps_var = tk.StringVar(value=str(DEFAULT_TIMESTEPS))
        self.save_dir_var = tk.StringVar(value=DEFAULT_SAVE_DIR)
        self.status_var = tk.StringVar(value="Ready.")

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=self.PAD)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(
            outer,
            text="RL Arm Motion",
            font=("Helvetica", 16, "bold"),
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            outer,
            text="2-DOF Robotic Arm - Reinforcement Learning Workbench",
            font=("Helvetica", 10),
            foreground="gray30",
        )
        subtitle.pack(anchor="w", pady=(0, self.PAD))

        ttk.Separator(outer).pack(fill="x", pady=(0, self.PAD))

        # ---- Section 1: Arm Control GUI ------------------------------
        arm_frame = ttk.LabelFrame(
            outer, text="Interactive Arm Control", padding=self.PAD,
        )
        arm_frame.pack(fill="x", pady=(0, self.PAD))

        ttk.Label(
            arm_frame,
            text=(
                "Launch the live arm visualisation. Use the sliders to adjust\n"
                "link lengths, masses, and damping. Use the +/- buttons or the\n"
                "arrow keys to drive each joint."
            ),
            justify="left",
            foreground="gray20",
        ).pack(anchor="w", pady=(0, self.PAD))

        ttk.Button(
            arm_frame,
            text="Open Arm Control GUI",
            command=self._launch_arm_gui,
        ).pack(anchor="w")

        # ---- Section 2: Training GUI --------------------------------
        train_frame = ttk.LabelFrame(
            outer, text="Training", padding=self.PAD,
        )
        train_frame.pack(fill="x", pady=(0, self.PAD))

        # Form rows
        form = ttk.Frame(train_frame)
        form.pack(fill="x", pady=(0, self.PAD))
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Algorithm:").grid(
            row=0, column=0, sticky="w", padx=(0, self.PAD), pady=2,
        )
        algo_box = ttk.Combobox(
            form,
            textvariable=self.algorithm_var,
            values=DEFAULT_ALGORITHMS,
            state="readonly",
            width=10,
        )
        algo_box.grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(form, text="Timesteps:").grid(
            row=1, column=0, sticky="w", padx=(0, self.PAD), pady=2,
        )
        ttk.Entry(form, textvariable=self.timesteps_var, width=14).grid(
            row=1, column=1, sticky="w", pady=2,
        )

        ttk.Label(form, text="Save dir:").grid(
            row=2, column=0, sticky="w", padx=(0, self.PAD), pady=2,
        )
        save_row = ttk.Frame(form)
        save_row.grid(row=2, column=1, sticky="ew", pady=2)
        save_row.columnconfigure(0, weight=1)
        ttk.Entry(save_row, textvariable=self.save_dir_var).grid(
            row=0, column=0, sticky="ew",
        )
        ttk.Button(save_row, text="Browse...", command=self._browse_save_dir).grid(
            row=0, column=1, padx=(4, 0),
        )

        # Defaults note
        ttk.Label(
            train_frame,
            text=(
                "Defaults follow Fischer et al. (2021): SAC + adaptive\n"
                "curriculum (0.60 m -> 0.02 m) + motor babbling (5000 steps)."
            ),
            justify="left",
            foreground="gray30",
        ).pack(anchor="w", pady=(0, self.PAD))

        ttk.Button(
            train_frame,
            text="Start Training GUI",
            command=self._launch_training_gui,
        ).pack(anchor="w")

        # ---- Status + Quit -------------------------------------------
        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", side="bottom")

        status = ttk.Label(
            bottom, textvariable=self.status_var, foreground="gray20",
            anchor="w",
        )
        status.pack(side="left", fill="x", expand=True)

        ttk.Button(bottom, text="Quit", command=self._on_quit).pack(side="right")

    # ------------------------------------------------------------------ #
    # Subprocess launchers
    # ------------------------------------------------------------------ #
    def _launch_arm_gui(self) -> None:
        cmd = [sys.executable, "-m", "rl_armMotion.two_d.gui.app"]
        self._spawn(cmd, label="Arm Control GUI")

    def _launch_training_gui(self) -> None:
        algorithm = self.algorithm_var.get().strip().upper() or "SAC"
        if algorithm not in DEFAULT_ALGORITHMS:
            self.status_var.set(
                f"Unknown algorithm '{algorithm}'. Choose one of {DEFAULT_ALGORITHMS}."
            )
            return

        try:
            timesteps = int(self.timesteps_var.get().strip())
            if timesteps < 1:
                raise ValueError("timesteps must be positive")
        except ValueError as exc:
            self.status_var.set(f"Invalid timesteps value: {exc}")
            return

        save_dir = self.save_dir_var.get().strip() or DEFAULT_SAVE_DIR
        # Best-effort directory creation so the training GUI can write
        # outputs without first interacting with the user.
        try:
            Path(save_dir).expanduser().mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.status_var.set(f"Could not create save dir: {exc}")
            return

        cmd = [
            sys.executable,
            "-m",
            "rl_armMotion.two_d.gui.training_gui",
            "--algorithm", algorithm,
            "--timesteps", str(timesteps),
            "--save-dir", str(save_dir),
        ]
        self._spawn(cmd, label=f"Training GUI ({algorithm}, {timesteps} steps)")

    def _spawn(self, cmd: List[str], label: str) -> None:
        """Launch a child process detached from the launcher's event loop."""
        try:
            # Inherit the launcher's environment so PYTHONPATH and the
            # active venv carry through unchanged.
            popen = subprocess.Popen(
                cmd,
                env=os.environ.copy(),
                stdout=None,
                stderr=None,
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
        except OSError as exc:
            self.status_var.set(f"Failed to launch {label}: {exc}")
            return

        self._launched.append(popen)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_var.set(
            f"[{timestamp}] {label} launched (PID {popen.pid}). "
            f"Total live: {self._count_live()}"
        )

    def _count_live(self) -> int:
        """Return the number of subprocesses still running."""
        live = 0
        for p in self._launched:
            if p.poll() is None:
                live += 1
        return live

    # ------------------------------------------------------------------ #
    # Quit
    # ------------------------------------------------------------------ #
    def _on_quit(self) -> None:
        # The user explicitly closed the launcher. Do NOT terminate child
        # processes; they may be doing useful work (training, motion
        # recording) that the user wants to keep running.
        self.root.destroy()

    def _browse_save_dir(self) -> None:
        choice = filedialog.askdirectory(
            initialdir=self.save_dir_var.get() or os.getcwd(),
            title="Choose save directory for trained model and metrics",
        )
        if choice:
            self.save_dir_var.set(choice)

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    LauncherApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
