"""3D kinematics and control helpers for the spherical-shoulder arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from rl_armMotion.three_d.config import ArmConfiguration3D


@dataclass
class ArmState3D:
    """Container for arm state snapshots."""

    angles: np.ndarray
    velocities: np.ndarray
    positions: np.ndarray
    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "angles": self.angles.tolist(),
            "velocities": self.velocities.tolist(),
            "positions": self.positions.tolist(),
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ArmState3D":
        return cls(
            angles=np.asarray(data["angles"], dtype=float),
            velocities=np.asarray(data["velocities"], dtype=float),
            positions=np.asarray(data["positions"], dtype=float),
            timestamp=float(data.get("timestamp", 0.0)),
        )


class ArmKinematics3D:
    """Forward kinematics for a 3D arm with spherical shoulder."""

    @staticmethod
    def _rot_x(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)

    @staticmethod
    def _rot_y(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)

    @staticmethod
    def _rot_z(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    @classmethod
    def shoulder_rotation(cls, shoulder_angles_xyz_rad: np.ndarray) -> np.ndarray:
        """
        Build shoulder orientation from extrinsic world-axis rotations.

        Convention used in 3D GUI:
        - J1x: rotation about global X axis
        - J1y: rotation about global Y axis (vertical axis)
        - J1z: rotation about global Z axis

        Ordering keeps J1y as a pure vertical-axis world rotation.
        """
        ax, ay, az = np.asarray(shoulder_angles_xyz_rad, dtype=float)
        return cls._rot_y(ay) @ cls._rot_z(az) @ cls._rot_x(ax)

    @classmethod
    def forward_kinematics(
        cls, angles_rad: np.ndarray, config: ArmConfiguration3D
    ) -> np.ndarray:
        """
        Compute 3D arm points: origin -> shoulder -> elbow -> end-effector.

        Shoulder is fixed in space at config.shoulder_position.
        """
        q = np.asarray(angles_rad, dtype=float)
        q = config.clamp_angles_rad(q)

        shoulder = np.asarray(config.shoulder_position, dtype=float)
        l_upper, l_forearm = [float(v) for v in config.link_lengths[:2]]

        shoulder_rot = cls.shoulder_rotation(q[:3])
        elbow_flex = float(q[3])

        # Local arm plane before applying shoulder orientation.
        # Rest pose points vertically downward along local -Y.
        v_upper_local = np.array([0.0, -l_upper, 0.0], dtype=float)
        v_forearm_local = np.array(
            [l_forearm * np.sin(elbow_flex), -l_forearm * np.cos(elbow_flex), 0.0],
            dtype=float,
        )

        elbow = shoulder + shoulder_rot @ v_upper_local
        ee = elbow + shoulder_rot @ v_forearm_local

        return np.vstack(
            [
                np.zeros(3, dtype=float),  # global origin
                shoulder,
                elbow,
                ee,
            ]
        )


class ArmController3D:
    """State controller for the 3D arm."""

    def __init__(self, config: ArmConfiguration3D):
        self.config = config
        self.dof = config.dof
        self.angles = np.deg2rad(np.asarray(config.initial_angles_deg, dtype=float))
        self.angles = self.config.clamp_angles_rad(self.angles)
        self.velocities = np.zeros(self.dof, dtype=float)
        self.positions = self._compute_positions()

    def _compute_positions(self) -> np.ndarray:
        return ArmKinematics3D.forward_kinematics(self.angles, self.config)

    def update_joint_angle(self, joint_id: int, target_angle_rad: float) -> None:
        if not 0 <= joint_id < self.dof:
            raise IndexError(f"Joint {joint_id} out of range [0, {self.dof - 1}]")
        mins, maxs = self.config.get_joint_limits_rad()
        self.angles[joint_id] = np.clip(float(target_angle_rad), mins[joint_id], maxs[joint_id])
        self.positions = self._compute_positions()

    def increment_joint(self, joint_id: int, delta_rad: float) -> None:
        if not 0 <= joint_id < self.dof:
            raise IndexError(f"Joint {joint_id} out of range [0, {self.dof - 1}]")
        max_delta = np.deg2rad(float(self.config.velocity_limits_deg_per_s)) * float(self.config.dt)
        delta_clamped = float(np.clip(delta_rad, -max_delta, max_delta))
        self.update_joint_angle(joint_id, self.angles[joint_id] + delta_clamped)
        self.velocities[joint_id] = delta_clamped / max(float(self.config.dt), 1e-8)

    def set_home_position(self) -> None:
        self.angles = np.deg2rad(np.asarray(self.config.initial_angles_deg, dtype=float))
        self.angles = self.config.clamp_angles_rad(self.angles)
        self.velocities = np.zeros(self.dof, dtype=float)
        self.positions = self._compute_positions()

    def get_state(self, timestamp: float = 0.0) -> ArmState3D:
        return ArmState3D(
            angles=self.angles.copy(),
            velocities=self.velocities.copy(),
            positions=self.positions.copy(),
            timestamp=float(timestamp),
        )

    def apply_state(self, state: ArmState3D) -> None:
        self.angles = self.config.clamp_angles_rad(state.angles)
        self.velocities = np.asarray(state.velocities, dtype=float).copy()
        self.positions = np.asarray(state.positions, dtype=float).copy()

    def get_end_effector_position(self) -> Tuple[float, float, float]:
        return tuple(self.positions[-1])


class MotionRecorder3D:
    """Simple recorder for manual joint motion playback."""

    def __init__(self):
        self.frames: List[ArmState3D] = []
        self.is_recording = False

    def start_recording(self) -> None:
        self.frames = []
        self.is_recording = True

    def stop_recording(self) -> None:
        self.is_recording = False

    def clear_frames(self) -> None:
        self.frames = []

    def record_frame(self, state: ArmState3D) -> None:
        if self.is_recording:
            self.frames.append(state)

    def get_num_frames(self) -> int:
        return len(self.frames)

    def get_frames(self) -> List[ArmState3D]:
        return list(self.frames)
