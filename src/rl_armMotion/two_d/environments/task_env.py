"""Task-specific environment for 2-DOF robotic arm with workspace setup.

This environment defines a task where:
- Shoulder joint is fixed at position [1.0, 0] meters from workspace origin
- Initial configuration: Arm pointing vertically downward (shoulder -90°, elbow 0°)
- Goal configuration can be:
  - legacy height goal (reach shoulder height + orientation hold)
  - directional far-point goal (EAST, WEST, NORTH)
- Workspace: 2D plane with origin at [0, 0] and shoulder base at [1.0, 0]
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_armMotion.two_d.config import ArmConfiguration
from rl_armMotion.two_d.utils.arm_kinematics import ArmController, ArmKinematics


class ArmTaskEnv(gym.Env):
    """
    2-DOF Robotic Arm Task Environment.

    Task objective:
    1. Reach the selected target (height line or directional far point)
    2. Align end-effector orientation to direction-specific target orientation
    3. Hold the pose for a required number of consecutive steps

    Observation (11-dim):
    [sin(theta1), cos(theta1), sin(theta2), cos(theta2),
     vel1_norm, vel2_norm,
     signed_height_error,
     signed_orientation_error,
     gradient_norm,
     in_goal_region_flag,
     hold_progress]

    Action (2-dim):
    Normalized joint velocity commands in [-1, 1], scaled by velocity limits.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        shoulder_base_position: Optional[np.ndarray] = None,
        use_2dof: bool = True,
        goal_direction: str = "HEIGHT",
    ):
        self.render_mode = render_mode
        self.use_2dof = use_2dof
        self.goal_direction = str(goal_direction).strip().upper()
        if self.goal_direction not in {"HEIGHT", "EAST", "WEST", "NORTH"}:
            self.goal_direction = "HEIGHT"

        # Load 2-DOF arm configuration with constraints
        self.config = ArmConfiguration.get_preset("2dof_simple")
        assert self.config.dof == 2, "This environment requires 2-DOF arm"

        self.num_dof = self.config.dof

        # Workspace setup
        self.workspace_origin = np.array([0.0, 0.0], dtype=np.float32)
        self.shoulder_base_position = (
            np.asarray(shoulder_base_position, dtype=np.float32)
            if shoulder_base_position is not None
            else np.array([1.0, 0.0], dtype=np.float32)
        )

        # Arm controller for dynamics
        self.controller = ArmController(self.config)

        # Action space: normalized velocity command for each joint
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_dof,),
            dtype=np.float32,
        )

        # Observation space
        max_reach = float(np.sum(self.config.link_lengths))
        obs_low = np.array(
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -max_reach,
                -np.pi,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                max_reach,
                np.pi,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Simulation parameters
        self.dt = float(self.config.dt)
        self.max_episode_steps = 800
        self.step_count = 0

        # State: [angles, velocities]
        self.state: Optional[np.ndarray] = None

        # Goal specification (legacy height mode or directional far-point mode).
        self.goal_height = float(self.shoulder_base_position[1])
        self.goal_position = np.array(
            [self.shoulder_base_position[0], self.goal_height],
            dtype=np.float32,
        )
        self.target_orientation = 0.0
        self.goal_axis = np.array([0.0, 1.0], dtype=np.float32)
        self._configure_goal(self.goal_direction)

        # Tolerances and hold constraints
        self.height_tolerance = 0.03
        self.orientation_tolerance = float(np.deg2rad(5.0))
        self.hold_velocity_tolerance = 0.15
        self.hold_steps_required = 60
        self.gradient_scale = 5.0

        # Compatibility alias with previous code/tests
        self.goal_tolerance = self.height_tolerance

        # Track performance and hold state
        self.best_distance = float("inf")
        self.best_total_error = float("inf")
        self.previous_total_error = float("inf")
        self.hold_counter = 0
        self.last_gradient = 0.0

    @staticmethod
    def _angle_normalize(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _compute_orientation(self, angles: np.ndarray) -> float:
        """Compute end-effector orientation for planar serial chain."""
        return self._angle_normalize(float(np.sum(angles)))

    def _compute_orientation_error(self, angles: np.ndarray) -> Tuple[float, float]:
        """Return signed and absolute orientation error from target orientation."""
        orientation = self._compute_orientation(angles)
        signed_error = self._angle_normalize(orientation - self.target_orientation)
        return signed_error, abs(signed_error)

    def _configure_goal(self, direction: str) -> None:
        """Configure directional goal target and orientation."""
        max_reach = float(np.sum(self.config.link_lengths))
        direction = str(direction).strip().upper()
        if direction not in {"HEIGHT", "EAST", "WEST", "NORTH"}:
            direction = "HEIGHT"
        self.goal_direction = direction

        if direction == "EAST":
            self.goal_axis = np.array([1.0, 0.0], dtype=np.float32)
            self.goal_position = self.shoulder_base_position + self.goal_axis * max_reach
            self.target_orientation = 0.0
        elif direction == "WEST":
            self.goal_axis = np.array([-1.0, 0.0], dtype=np.float32)
            self.goal_position = self.shoulder_base_position + self.goal_axis * max_reach
            self.target_orientation = np.pi
        elif direction == "NORTH":
            self.goal_axis = np.array([0.0, 1.0], dtype=np.float32)
            self.goal_position = self.shoulder_base_position + self.goal_axis * max_reach
            self.target_orientation = np.pi / 2.0
        else:
            # Legacy mode: target line at shoulder height.
            self.goal_axis = np.array([0.0, 1.0], dtype=np.float32)
            self.goal_position = np.array(
                [self.shoulder_base_position[0], self.shoulder_base_position[1]],
                dtype=np.float32,
            )
            self.target_orientation = 0.0

        self.goal_height = float(self.goal_position[1])

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state (arm vertical downward)."""
        super().reset(seed=seed)

        angles = np.array(self.config.initial_angles, dtype=np.float32)
        velocities = np.zeros(self.num_dof, dtype=np.float32)

        self.state = np.concatenate([angles, velocities]).astype(np.float32)
        self.step_count = 0
        self.best_distance = float("inf")
        self.best_total_error = float("inf")
        self.hold_counter = 0
        self.last_gradient = 0.0

        self.controller.angles = angles.copy()

        end_effector = self._get_end_effector_position(angles)
        signed_height_error = self._compute_signed_goal_error(end_effector)
        _, abs_orientation_error = self._compute_orientation_error(angles)
        self.previous_total_error = 2.0 * abs(signed_height_error) + abs_orientation_error

        obs = self._get_observation(
            angles=angles,
            velocities=velocities,
            signed_height_error=signed_height_error,
            signed_orientation_error=self._compute_orientation_error(angles)[0],
            gradient_norm=0.0,
            in_goal_region=False,
        )
        return obs, self.get_state_info()

    def _get_end_effector_position(self, angles: np.ndarray) -> np.ndarray:
        """Compute end-effector position in workspace frame."""
        positions = ArmKinematics.forward_kinematics(angles, self.config)
        end_effector_shoulder_frame = positions[-1, :2]
        return end_effector_shoulder_frame + self.shoulder_base_position

    def _compute_goal_distance(self, end_effector_pos: np.ndarray) -> float:
        """Compute distance to current goal definition."""
        if self.goal_direction == "HEIGHT":
            return abs(float(end_effector_pos[1] - self.goal_height))
        return float(np.linalg.norm(np.asarray(end_effector_pos, dtype=float) - self.goal_position))

    def _compute_signed_goal_error(self, end_effector_pos: np.ndarray) -> float:
        """Compute signed error projected onto goal axis."""
        if self.goal_direction == "HEIGHT":
            return float(end_effector_pos[1] - self.goal_height)
        delta = np.asarray(end_effector_pos, dtype=float) - self.goal_position
        return float(np.dot(delta, self.goal_axis))

    def _is_goal_reached(
        self,
        height_error_abs: float,
        orientation_error_abs: float,
        velocity_norm: float,
    ) -> bool:
        """Check if arm is in stable goal region for this step."""
        return (
            height_error_abs < self.height_tolerance
            and orientation_error_abs < self.orientation_tolerance
            and velocity_norm < self.hold_velocity_tolerance
        )

    def _get_observation(
        self,
        angles: np.ndarray,
        velocities: np.ndarray,
        signed_height_error: float,
        signed_orientation_error: float,
        gradient_norm: float,
        in_goal_region: bool,
    ) -> np.ndarray:
        """Build observation vector for policy."""
        velocities_norm = np.asarray(velocities, dtype=np.float32) / np.asarray(
            self.config.velocity_limits, dtype=np.float32
        )
        velocities_norm = np.clip(velocities_norm, -1.0, 1.0)

        hold_progress = min(1.0, self.hold_counter / float(self.hold_steps_required))

        obs = np.array(
            [
                np.sin(angles[0]),
                np.cos(angles[0]),
                np.sin(angles[1]),
                np.cos(angles[1]),
                velocities_norm[0],
                velocities_norm[1],
                signed_height_error,
                signed_orientation_error,
                gradient_norm,
                1.0 if in_goal_region else 0.0,
                hold_progress,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.step_count += 1

        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping")

        angles = self.state[: self.num_dof].copy()

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Convert normalized action to physical joint velocity command.
        joint_velocity_cmd = action * np.asarray(self.config.velocity_limits, dtype=np.float32)

        new_angles = angles + joint_velocity_cmd * self.dt
        new_angles = np.clip(
            new_angles,
            self.config.joint_limits_min,
            self.config.joint_limits_max,
        ).astype(np.float32)

        new_velocities = joint_velocity_cmd.astype(np.float32)
        self.state = np.concatenate([new_angles, new_velocities]).astype(np.float32)
        self.controller.angles = new_angles

        end_effector_pos = self._get_end_effector_position(new_angles)
        signed_height_error = self._compute_signed_goal_error(end_effector_pos)
        goal_distance = self._compute_goal_distance(end_effector_pos)

        signed_orientation_error, orientation_error = self._compute_orientation_error(new_angles)
        velocity_norm = float(np.linalg.norm(new_velocities))

        in_goal_region = self._is_goal_reached(goal_distance, orientation_error, velocity_norm)

        if in_goal_region:
            self.hold_counter += 1
        else:
            self.hold_counter = 0

        total_error = 2.0 * goal_distance + orientation_error

        if goal_distance < self.best_distance:
            self.best_distance = goal_distance
        if total_error < self.best_total_error:
            self.best_total_error = total_error

        gradient = abs(total_error - self.previous_total_error) / max(self.dt, 1e-8)
        gradient_norm = float(np.clip(gradient / self.gradient_scale, 0.0, 1.0))
        self.last_gradient = gradient_norm

        progress = self.previous_total_error - total_error
        self.previous_total_error = total_error

        # Reward shaping for fast learning and stable hold behavior.
        reward = (
            -2.0 * goal_distance
            -1.0 * orientation_error
            -0.15 * velocity_norm
            -0.20 * gradient_norm
            -0.01 * float(np.linalg.norm(action))
        )

        if progress > 0:
            reward += 1.5 * progress

        if in_goal_region:
            reward += 2.0 * float(self.hold_counter)

        terminated = self.hold_counter >= self.hold_steps_required
        if terminated:
            reward += 150.0

        truncated = self.step_count >= self.max_episode_steps

        hold_progress = min(1.0, self.hold_counter / float(self.hold_steps_required))

        info = {
            "goal_distance": float(goal_distance),
            "height_error": float(signed_height_error),
            "orientation_error": float(signed_orientation_error),
            "orientation_error_abs": float(orientation_error),
            "gradient": float(gradient),
            "gradient_norm": float(gradient_norm),
            "total_error": float(total_error),
            "hold_counter": int(self.hold_counter),
            "hold_steps_required": int(self.hold_steps_required),
            "hold_progress": float(hold_progress),
            "in_goal_region": bool(in_goal_region),
            "end_effector_position": end_effector_pos.copy(),
            "joint_angles": new_angles.copy(),
            "joint_velocities": new_velocities.copy(),
            "shoulder_position": self.shoulder_base_position.copy(),
            "goal_height": self.goal_height,
            "goal_position": self.goal_position.copy(),
            "goal_direction": self.goal_direction,
            "target_orientation": self.target_orientation,
            "goal_reached": bool(terminated),
            "step": self.step_count,
            "best_distance": float(self.best_distance),
        }

        obs = self._get_observation(
            angles=new_angles,
            velocities=new_velocities,
            signed_height_error=signed_height_error,
            signed_orientation_error=signed_orientation_error,
            gradient_norm=gradient_norm,
            in_goal_region=in_goal_region,
        )

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        """Render environment information to console."""
        if self.render_mode == "human" and self.state is not None:
            angles = self.state[: self.num_dof]
            end_effector = self._get_end_effector_position(angles)
            goal_distance = self._compute_goal_distance(end_effector)
            _, orientation_error = self._compute_orientation_error(angles)
            print(
                f"Step {self.step_count:3d} | "
                f"Angles: [{angles[0]:6.3f}, {angles[1]:6.3f}] | "
                f"EE: [{end_effector[0]:6.3f}, {end_effector[1]:6.3f}] | "
                f"GoalDist: {goal_distance:6.3f} | "
                f"Orient Err: {orientation_error:6.3f} | "
                f"GoalDir: {self.goal_direction:>6s} | "
                f"Hold: {self.hold_counter}/{self.hold_steps_required}"
            )

    def close(self):
        """Close the environment."""
        pass

    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed information about current environment state."""
        if self.state is None:
            raise RuntimeError("Environment state not initialized. Call reset() first.")

        angles = self.state[: self.num_dof]
        velocities = self.state[self.num_dof :]
        end_effector = self._get_end_effector_position(angles)
        signed_height_error = self._compute_signed_goal_error(end_effector)
        signed_orientation_error, orientation_error_abs = self._compute_orientation_error(angles)
        velocity_norm = float(np.linalg.norm(velocities))
        in_goal_region = self._is_goal_reached(
            self._compute_goal_distance(end_effector), orientation_error_abs, velocity_norm
        )

        return {
            "joint_angles": angles.copy(),
            "joint_velocities": velocities.copy(),
            "end_effector_position": end_effector.copy(),
            "shoulder_position": self.shoulder_base_position.copy(),
            "workspace_origin": self.workspace_origin.copy(),
            "goal_height": self.goal_height,
            "goal_position": self.goal_position.copy(),
            "goal_direction": self.goal_direction,
            "target_orientation": self.target_orientation,
            "distance_to_goal": float(self._compute_goal_distance(end_effector)),
            "height_error": float(signed_height_error),
            "orientation_error": float(signed_orientation_error),
            "orientation_error_abs": float(orientation_error_abs),
            "gradient_norm": float(self.last_gradient),
            "hold_counter": int(self.hold_counter),
            "hold_steps_required": int(self.hold_steps_required),
            "hold_progress": float(min(1.0, self.hold_counter / float(self.hold_steps_required))),
            "in_goal_region": bool(in_goal_region),
            "goal_reached": bool(self.hold_counter >= self.hold_steps_required),
            "step": self.step_count,
            "max_steps": self.max_episode_steps,
        }


__all__ = ["ArmTaskEnv"]
