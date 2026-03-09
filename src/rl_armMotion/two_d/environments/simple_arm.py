"""Sample 7-DOF robotic arm environment using Gymnasium

This is a template environment for a 7-DOF arm that can be customized
for your specific arm model.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any


class SimpleArmEnv(gym.Env):
    """
    Simple 7-DOF robotic arm environment.

    State: Joint angles [7] + Joint velocities [7] = 14-dim observation
    Action: Target joint velocities [7] = 7-dim action
    Reward: Based on reaching target position + energy efficiency
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, num_dof: int = 7):
        """
        Initialize the arm environment.

        Args:
            render_mode: Rendering mode
            num_dof: Number of degrees of freedom (default: 7 for 7-DOF arm)
        """
        self.num_dof = num_dof
        self.render_mode = render_mode

        # Joint limits (radians) - typical for industrial arms
        self.joint_limits = np.array(
            [
                [-2.96, 2.96],  # Joint 1
                [-2.09, 2.09],  # Joint 2
                [-2.96, 2.96],  # Joint 3
                [-2.09, 2.09],  # Joint 4
                [-2.96, 2.96],  # Joint 5
                [-2.09, 2.09],  # Joint 6
                [-3.05, 3.05],  # Joint 7
            ]
        )

        # Velocity limits (rad/s)
        self.velocity_limits = 2.0

        # Action space: target velocities for each joint
        self.action_space = spaces.Box(
            low=-self.velocity_limits,
            high=self.velocity_limits,
            shape=(self.num_dof,),
            dtype=np.float32,
        )

        # Observation space: joint angles + velocities
        low = np.concatenate([self.joint_limits[:, 0], -self.velocity_limits * np.ones(self.num_dof)])
        high = np.concatenate([self.joint_limits[:, 1], self.velocity_limits * np.ones(self.num_dof)])

        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        # Target position for reaching task
        self.target_position = np.zeros(self.num_dof)

        # Simulation parameters
        self.dt = 0.01  # Time step
        self.max_episode_steps = 500
        self.step_count = 0

        self.state = None

    def reset(
        self, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Initialize with random joint angles
        self.state = np.random.uniform(
            low=self.joint_limits[:, 0],
            high=self.joint_limits[:, 1],
        ).astype(np.float32)

        # Add zero velocities
        self.state = np.concatenate([self.state, np.zeros(self.num_dof, dtype=np.float32)])

        # Random target position
        self.target_position = np.random.uniform(
            low=self.joint_limits[:, 0],
            high=self.joint_limits[:, 1],
        ).astype(np.float32)

        self.step_count = 0

        return self.state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.step_count += 1

        # Extract current state
        angles = self.state[:self.num_dof]
        velocities = self.state[self.num_dof :]

        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Simple dynamics: velocity = action, position += velocity * dt
        new_velocities = action.copy()
        new_angles = angles + new_velocities * self.dt

        # Enforce joint limits
        new_angles = np.clip(
            new_angles, self.joint_limits[:, 0], self.joint_limits[:, 1]
        )

        # Update state
        self.state = np.concatenate([new_angles, new_velocities]).astype(np.float32)

        # Compute reward
        position_error = np.linalg.norm(new_angles - self.target_position)
        energy_cost = np.linalg.norm(action) * 0.01

        reward = -position_error - energy_cost

        # Termination conditions
        terminated = False
        if position_error < 0.1:  # Reached target
            terminated = True
            reward += 100.0

        truncated = self.step_count >= self.max_episode_steps

        info = {
            "position_error": float(position_error),
            "energy": float(np.linalg.norm(action)),
            "reached_target": position_error < 0.1,
        }

        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
        """Render the environment (placeholder)"""
        if self.render_mode == "human":
            print(f"Step {self.step_count}: Angles = {self.state[:self.num_dof]}")

    def close(self):
        """Close the environment"""
        pass


__all__ = ["SimpleArmEnv"]
