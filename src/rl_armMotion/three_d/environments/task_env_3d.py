"""Task environment for 3D arm with spherical shoulder and revolute elbow."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_armMotion.three_d.config import ArmConfiguration3D
from rl_armMotion.three_d.utils import ArmController3D, ArmKinematics3D


class ArmTaskEnv3D(gym.Env):
    """
    3D task environment.

    Arm model:
    - Shoulder spherical joint: J1x, J1y, J1z
    - Elbow revolute joint: J2

    Action (4-dim): normalized joint velocity commands in [-1, 1].

    Observation (18-dim):
    [sin(q0), cos(q0), ..., sin(q3), cos(q3),
     qd0_norm, qd1_norm, qd2_norm, qd3_norm,
     signed_axis_error,
     lateral_error,
     orientation_error,
     gradient_norm,
     in_goal_region_flag,
     hold_progress]
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    GOAL_DIRECTIONS = {"EAST", "WEST", "NORTH"}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[ArmConfiguration3D] = None,
        goal_direction: str = "EAST",
    ):
        self.render_mode = render_mode
        self.config = ArmConfiguration3D.from_dict((config or ArmConfiguration3D.get_default()).to_dict())
        self.num_dof = int(self.config.dof)

        self.workspace_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.shoulder_base_position = np.asarray(self.config.shoulder_position, dtype=np.float32)

        self.controller = ArmController3D(self.config)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_dof,),
            dtype=np.float32,
        )

        self.max_reach = float(np.sum(self.config.link_lengths[:2]))
        self.max_joint_speed = float(np.deg2rad(self.config.velocity_limits_deg_per_s))

        obs_low = np.array(
            [
                *([-1.0] * (2 * self.num_dof)),
                *([-1.0] * self.num_dof),
                -2.0 * self.max_reach,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                *([1.0] * (2 * self.num_dof)),
                *([1.0] * self.num_dof),
                2.0 * self.max_reach,
                2.0 * self.max_reach,
                np.pi,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.dt = float(self.config.dt)
        self.max_episode_steps = 900
        self.step_count = 0

        self.goal_direction = "EAST"
        self.goal_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.goal_position = self.shoulder_base_position.copy()
        self.goal_height = float(self.goal_position[1])
        self._configure_goal(goal_direction)

        self.goal_tolerance = 0.08
        self.orientation_tolerance = float(np.deg2rad(10.0))
        self.hold_velocity_tolerance = float(np.deg2rad(25.0))
        self.hold_steps_required = 70
        self.gradient_scale = 8.0

        self.state: Optional[np.ndarray] = None
        self.best_distance = float("inf")
        self.best_total_error = float("inf")
        self.previous_total_error = float("inf")
        self.hold_counter = 0
        self.last_gradient = 0.0

    @staticmethod
    def _unit_vector(v: np.ndarray) -> np.ndarray:
        vec = np.asarray(v, dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return np.zeros_like(vec)
        return vec / norm

    def _configure_goal(self, direction: str) -> None:
        direction = str(direction).strip().upper()
        if direction not in self.GOAL_DIRECTIONS:
            direction = "EAST"

        if direction == "WEST":
            axis = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        elif direction == "NORTH":
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            direction = "EAST"

        self.goal_direction = direction
        self.goal_axis = axis
        self.goal_position = self.shoulder_base_position + self.goal_axis * self.max_reach
        self.goal_height = float(self.goal_position[1])

    def _get_points(self, angles: np.ndarray) -> np.ndarray:
        return ArmKinematics3D.forward_kinematics(angles, self.config)

    def _get_end_effector_position(self, angles: np.ndarray) -> np.ndarray:
        return np.asarray(self._get_points(angles)[-1], dtype=np.float32)

    def _get_forearm_direction(self, angles: np.ndarray) -> np.ndarray:
        pts = self._get_points(angles)
        forearm_vec = np.asarray(pts[-1] - pts[-2], dtype=np.float32)
        unit = self._unit_vector(forearm_vec)
        if float(np.linalg.norm(unit)) < 1e-8:
            return np.array([0.0, -1.0, 0.0], dtype=np.float32)
        return unit.astype(np.float32)

    def _compute_goal_errors(self, end_effector_pos: np.ndarray) -> Tuple[float, float, float]:
        delta = np.asarray(end_effector_pos, dtype=float) - np.asarray(self.goal_position, dtype=float)
        signed_axis_error = float(np.dot(delta, self.goal_axis))
        lateral_vec = delta - signed_axis_error * np.asarray(self.goal_axis, dtype=float)
        lateral_error = float(np.linalg.norm(lateral_vec))
        goal_distance = float(np.linalg.norm(delta))
        return signed_axis_error, lateral_error, goal_distance

    def _compute_orientation_error(self, angles: np.ndarray) -> Tuple[float, float]:
        forearm_dir = np.asarray(self._get_forearm_direction(angles), dtype=float)
        alignment = float(np.clip(np.dot(forearm_dir, np.asarray(self.goal_axis, dtype=float)), -1.0, 1.0))
        orientation_error = float(np.arccos(alignment))
        return orientation_error, alignment

    def _is_goal_reached(
        self,
        goal_distance: float,
        orientation_error: float,
        velocity_norm: float,
    ) -> bool:
        return (
            goal_distance < self.goal_tolerance
            and orientation_error < self.orientation_tolerance
            and velocity_norm < self.hold_velocity_tolerance
        )

    def _get_observation(
        self,
        angles: np.ndarray,
        velocities: np.ndarray,
        signed_axis_error: float,
        lateral_error: float,
        orientation_error: float,
        gradient_norm: float,
        in_goal_region: bool,
    ) -> np.ndarray:
        vel_norm = np.asarray(velocities, dtype=np.float32) / max(self.max_joint_speed, 1e-8)
        vel_norm = np.clip(vel_norm, -1.0, 1.0)

        hold_progress = min(1.0, self.hold_counter / float(self.hold_steps_required))

        trig = []
        for q in np.asarray(angles, dtype=float):
            trig.extend([np.sin(q), np.cos(q)])

        obs = np.array(
            [
                *trig,
                *vel_norm.tolist(),
                float(signed_axis_error),
                float(lateral_error),
                float(orientation_error),
                float(gradient_norm),
                1.0 if in_goal_region else 0.0,
                float(hold_progress),
            ],
            dtype=np.float32,
        )
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if isinstance(options, dict) and "goal_direction" in options:
            self._configure_goal(str(options.get("goal_direction", self.goal_direction)))

        angles = np.deg2rad(np.asarray(self.config.initial_angles_deg, dtype=np.float32))
        angles = self.config.clamp_angles_rad(angles).astype(np.float32)
        velocities = np.zeros(self.num_dof, dtype=np.float32)

        self.state = np.concatenate([angles, velocities]).astype(np.float32)
        self.step_count = 0
        self.best_distance = float("inf")
        self.best_total_error = float("inf")
        self.previous_total_error = float("inf")
        self.hold_counter = 0
        self.last_gradient = 0.0

        self.controller.angles = angles.copy()
        self.controller.velocities = velocities.copy()
        self.controller.positions = self._get_points(angles)

        ee = self._get_end_effector_position(angles)
        signed_axis_error, lateral_error, goal_distance = self._compute_goal_errors(ee)
        orientation_error, _ = self._compute_orientation_error(angles)

        self.previous_total_error = 2.0 * goal_distance + orientation_error + 0.25 * lateral_error

        obs = self._get_observation(
            angles=angles,
            velocities=velocities,
            signed_axis_error=signed_axis_error,
            lateral_error=lateral_error,
            orientation_error=orientation_error,
            gradient_norm=0.0,
            in_goal_region=False,
        )
        return obs, self.get_state_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1

        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping")

        angles = self.state[: self.num_dof].copy()

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        joint_velocity_cmd = action * self.max_joint_speed

        new_angles = angles + joint_velocity_cmd * self.dt
        new_angles = self.config.clamp_angles_rad(new_angles).astype(np.float32)

        new_velocities = joint_velocity_cmd.astype(np.float32)
        self.state = np.concatenate([new_angles, new_velocities]).astype(np.float32)

        self.controller.angles = new_angles.copy()
        self.controller.velocities = new_velocities.copy()
        self.controller.positions = self._get_points(new_angles)

        ee = self._get_end_effector_position(new_angles)
        signed_axis_error, lateral_error, goal_distance = self._compute_goal_errors(ee)
        orientation_error, orientation_alignment = self._compute_orientation_error(new_angles)
        velocity_norm = float(np.linalg.norm(new_velocities))

        in_goal_region = self._is_goal_reached(goal_distance, orientation_error, velocity_norm)
        if in_goal_region:
            self.hold_counter += 1
        else:
            self.hold_counter = 0

        total_error = 2.0 * goal_distance + orientation_error + 0.25 * lateral_error
        if goal_distance < self.best_distance:
            self.best_distance = goal_distance
        if total_error < self.best_total_error:
            self.best_total_error = total_error

        gradient = abs(total_error - self.previous_total_error) / max(self.dt, 1e-8)
        gradient_norm = float(np.clip(gradient / self.gradient_scale, 0.0, 1.0))
        self.last_gradient = gradient_norm

        progress = self.previous_total_error - total_error
        self.previous_total_error = total_error

        reward = (
            -2.5 * goal_distance
            -0.90 * orientation_error
            -0.20 * lateral_error
            -0.15 * velocity_norm
            -0.20 * gradient_norm
            -0.01 * float(np.linalg.norm(action))
        )
        if progress > 0:
            reward += 1.30 * progress
        if in_goal_region:
            reward += 2.0 * float(self.hold_counter)

        terminated = self.hold_counter >= self.hold_steps_required
        if terminated:
            reward += 160.0

        truncated = self.step_count >= self.max_episode_steps
        hold_progress = min(1.0, self.hold_counter / float(self.hold_steps_required))

        info = {
            "goal_distance": float(goal_distance),
            "signed_axis_error": float(signed_axis_error),
            "lateral_error": float(lateral_error),
            "height_error": float(ee[1] - self.goal_height),
            "orientation_error": float(orientation_error),
            "orientation_alignment": float(orientation_alignment),
            "gradient": float(gradient),
            "gradient_norm": float(gradient_norm),
            "total_error": float(total_error),
            "hold_counter": int(self.hold_counter),
            "hold_steps_required": int(self.hold_steps_required),
            "hold_progress": float(hold_progress),
            "in_goal_region": bool(in_goal_region),
            "end_effector_position": ee.copy(),
            "joint_angles": new_angles.copy(),
            "joint_velocities": new_velocities.copy(),
            "shoulder_position": self.shoulder_base_position.copy(),
            "goal_position": self.goal_position.copy(),
            "goal_axis": self.goal_axis.copy(),
            "goal_direction": self.goal_direction,
            "goal_height": float(self.goal_height),
            "goal_reached": bool(terminated),
            "step": int(self.step_count),
            "best_distance": float(self.best_distance),
        }

        obs = self._get_observation(
            angles=new_angles,
            velocities=new_velocities,
            signed_axis_error=signed_axis_error,
            lateral_error=lateral_error,
            orientation_error=orientation_error,
            gradient_norm=gradient_norm,
            in_goal_region=in_goal_region,
        )

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "human" and self.state is not None:
            angles = self.state[: self.num_dof]
            ee = self._get_end_effector_position(angles)
            _, _, goal_distance = self._compute_goal_errors(ee)
            orientation_error, _ = self._compute_orientation_error(angles)
            print(
                f"Step {self.step_count:4d} | "
                f"GoalDist: {goal_distance:7.4f} | "
                f"OrientErr: {orientation_error:6.3f} | "
                f"Hold: {self.hold_counter}/{self.hold_steps_required}"
            )

    def close(self):
        pass

    def get_state_info(self) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Environment state not initialized. Call reset() first.")

        angles = self.state[: self.num_dof]
        velocities = self.state[self.num_dof :]
        ee = self._get_end_effector_position(angles)
        signed_axis_error, lateral_error, goal_distance = self._compute_goal_errors(ee)
        orientation_error, orientation_alignment = self._compute_orientation_error(angles)
        velocity_norm = float(np.linalg.norm(velocities))
        in_goal_region = self._is_goal_reached(goal_distance, orientation_error, velocity_norm)

        return {
            "joint_angles": angles.copy(),
            "joint_velocities": velocities.copy(),
            "end_effector_position": ee.copy(),
            "shoulder_position": self.shoulder_base_position.copy(),
            "workspace_origin": self.workspace_origin.copy(),
            "goal_position": self.goal_position.copy(),
            "goal_axis": self.goal_axis.copy(),
            "goal_direction": self.goal_direction,
            "goal_height": float(self.goal_height),
            "distance_to_goal": float(goal_distance),
            "goal_distance": float(goal_distance),
            "signed_axis_error": float(signed_axis_error),
            "lateral_error": float(lateral_error),
            "height_error": float(ee[1] - self.goal_height),
            "orientation_error": float(orientation_error),
            "orientation_alignment": float(orientation_alignment),
            "gradient_norm": float(self.last_gradient),
            "hold_counter": int(self.hold_counter),
            "hold_steps_required": int(self.hold_steps_required),
            "hold_progress": float(min(1.0, self.hold_counter / float(self.hold_steps_required))),
            "in_goal_region": bool(in_goal_region),
            "goal_reached": bool(self.hold_counter >= self.hold_steps_required),
            "step": int(self.step_count),
            "max_steps": int(self.max_episode_steps),
        }


__all__ = ["ArmTaskEnv3D"]
