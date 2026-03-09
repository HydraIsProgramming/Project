"""Configuration objects for the 3D arm model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ArmConfiguration3D:
    """Configuration for a 3D arm with a spherical shoulder joint."""

    name: str = "3D_2Link_SphericalShoulder"
    dof: int = 4

    # Joints: spherical shoulder (J1x/J1y/J1z) + revolute elbow (J2)
    joint_names: List[str] = field(
        default_factory=lambda: ["J1x", "J1y", "J1z", "J2"]
    )

    # Degrees are easier for GUI display/editing.
    initial_angles_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # Link lengths: upper-arm, forearm.
    link_lengths: List[float] = field(default_factory=lambda: [0.42, 0.36])
    masses: List[float] = field(default_factory=lambda: [2.6, 1.9])
    inertias: List[float] = field(default_factory=lambda: [0.09, 0.09, 0.09, 0.05])
    damping: float = 0.08

    # Shoulder remains fixed in space.
    shoulder_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Requested constraints in degrees:
    # X (anterior/posterior): 0 .. 120
    # Y (vertical): -90 .. +90
    # Z (medial/lateral): -90 .. +120
    joint_limits_deg_min: List[float] = field(default_factory=lambda: [0.0, -90.0, -90.0, 0.0])
    joint_limits_deg_max: List[float] = field(default_factory=lambda: [120.0, 90.0, 120.0, 150.0])

    dt: float = 0.05
    velocity_limits_deg_per_s: float = 120.0

    def __post_init__(self) -> None:
        if self.dof != 4:
            self.dof = 4
        self.initial_angles_deg = self._fit_len(self.initial_angles_deg, self.dof, 0.0)
        self.joint_names = self._fit_names(self.joint_names, self.dof)
        self.inertias = self._fit_len(self.inertias, self.dof, 0.05)
        self.joint_limits_deg_min = self._fit_len(self.joint_limits_deg_min, self.dof, -180.0)
        self.joint_limits_deg_max = self._fit_len(self.joint_limits_deg_max, self.dof, 180.0)
        self.link_lengths = self._fit_len(self.link_lengths, 2, 0.2)
        self.masses = self._fit_len(self.masses, 2, 1.0)
        self.shoulder_position = self._fit_len(self.shoulder_position, 3, 0.0)

    @staticmethod
    def _fit_len(values: List[float], size: int, fill: float) -> List[float]:
        out = list(values[:size])
        while len(out) < size:
            out.append(fill)
        return out

    @staticmethod
    def _fit_names(values: List[str], size: int) -> List[str]:
        out = list(values[:size])
        while len(out) < size:
            out.append(f"joint_{len(out)}")
        return out

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ArmConfiguration3D":
        """
        Build config from 3D-native or legacy/2D-style dictionaries.

        Supports legacy keys like:
        - initial_angles (rad)
        - joint_limits_min/max (rad)
        - velocity_limits (rad/s)
        """
        if not isinstance(data, dict):
            return cls()

        cfg = cls()

        if "name" in data:
            cfg.name = str(data["name"])

        if "joint_names" in data and isinstance(data["joint_names"], list):
            cfg.joint_names = [str(v) for v in data["joint_names"]]

        if "initial_angles_deg" in data:
            cfg.initial_angles_deg = [float(v) for v in data["initial_angles_deg"]]
        elif "initial_angles" in data:
            cfg.initial_angles_deg = [float(np.rad2deg(float(v))) for v in data["initial_angles"]]

        if "link_lengths" in data:
            cfg.link_lengths = [float(v) for v in data["link_lengths"]]

        if "masses" in data:
            cfg.masses = [float(v) for v in data["masses"]]

        if "inertias" in data:
            cfg.inertias = [float(v) for v in data["inertias"]]

        if "damping" in data:
            cfg.damping = float(data["damping"])

        if "shoulder_position" in data:
            cfg.shoulder_position = [float(v) for v in data["shoulder_position"]]

        if "joint_limits_deg_min" in data:
            cfg.joint_limits_deg_min = [float(v) for v in data["joint_limits_deg_min"]]
        elif "joint_limits_min" in data:
            cfg.joint_limits_deg_min = [float(np.rad2deg(float(v))) for v in data["joint_limits_min"]]

        if "joint_limits_deg_max" in data:
            cfg.joint_limits_deg_max = [float(v) for v in data["joint_limits_deg_max"]]
        elif "joint_limits_max" in data:
            cfg.joint_limits_deg_max = [float(np.rad2deg(float(v))) for v in data["joint_limits_max"]]

        if "dt" in data:
            cfg.dt = float(data["dt"])

        if "velocity_limits_deg_per_s" in data:
            cfg.velocity_limits_deg_per_s = float(data["velocity_limits_deg_per_s"])
        elif "velocity_limits" in data:
            cfg.velocity_limits_deg_per_s = float(np.rad2deg(float(data["velocity_limits"])))

        cfg.__post_init__()
        return cfg

    def to_json(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "ArmConfiguration3D":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def get_joint_limits_rad(self) -> Tuple[np.ndarray, np.ndarray]:
        mins = np.deg2rad(np.asarray(self.joint_limits_deg_min, dtype=float))
        maxs = np.deg2rad(np.asarray(self.joint_limits_deg_max, dtype=float))
        return mins, maxs

    def clamp_angles_rad(self, angles_rad: np.ndarray) -> np.ndarray:
        mins, maxs = self.get_joint_limits_rad()
        return np.clip(np.asarray(angles_rad, dtype=float), mins, maxs)

    @classmethod
    def get_default(cls) -> "ArmConfiguration3D":
        return cls()
