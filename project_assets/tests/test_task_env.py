"""Tests for the ArmTaskEnv environment."""

import numpy as np

from rl_armMotion.environments.task_env import ArmTaskEnv
from rl_armMotion.utils.arm_kinematics import ArmKinematics


class TestArmTaskEnv:
    """Test suite for ArmTaskEnv."""

    def setup_method(self):
        self.env = ArmTaskEnv()

    def test_initialization(self):
        assert self.env.num_dof == 2
        assert self.env.action_space.shape == (2,)
        assert self.env.observation_space.shape == (11,)
        assert np.allclose(self.env.action_space.low, -1.0)
        assert np.allclose(self.env.action_space.high, 1.0)
        assert np.allclose(self.env.shoulder_base_position, [1.0, 0.0])
        assert self.env.goal_height == 0.0
        assert self.env.hold_steps_required > 0

    def test_reset_to_initial_state(self):
        obs, info = self.env.reset()

        assert obs.shape == (11,)

        # sin/cos encoding should match initial angles
        initial_angles = np.array(self.env.config.initial_angles, dtype=np.float32)
        assert np.isclose(obs[0], np.sin(initial_angles[0]))
        assert np.isclose(obs[1], np.cos(initial_angles[0]))
        assert np.isclose(obs[2], np.sin(initial_angles[1]))
        assert np.isclose(obs[3], np.cos(initial_angles[1]))

        # normalized velocities should start at zero
        assert np.allclose(obs[4:6], 0.0)
        assert obs[8] == 0.0  # gradient_norm
        assert obs[9] == 0.0  # in_goal_region flag
        assert obs[10] == 0.0  # hold progress

        assert "hold_counter" in info
        assert "hold_steps_required" in info

    def test_end_effector_position_vertical(self):
        self.env.reset()
        angles = self.env.state[:2]
        ee_pos = self.env._get_end_effector_position(angles)
        assert ee_pos[1] < self.env.shoulder_base_position[1]

    def test_goal_distance_from_initial(self):
        self.env.reset()
        angles = self.env.state[:2]
        ee_pos = self.env._get_end_effector_position(angles)
        distance = self.env._compute_goal_distance(ee_pos)

        expected_distance = abs(ee_pos[1] - self.env.goal_height)
        assert np.isclose(distance, expected_distance)
        assert distance > 0

    def test_step_updates_state(self):
        self.env.reset()
        initial_angles = self.env.state[:2].copy()

        action = np.array([1.0, 0.5], dtype=np.float32)
        obs, reward, done, truncated, info = self.env.step(action)

        new_angles = self.env.state[:2]
        new_velocities = self.env.state[2:4]
        expected_velocities = action * self.env.config.velocity_limits

        assert obs.shape == (11,)
        assert not np.allclose(new_angles, initial_angles)
        assert np.allclose(new_velocities, expected_velocities)

        assert "goal_distance" in info
        assert "height_error" in info
        assert "orientation_error" in info
        assert "gradient" in info
        assert "gradient_norm" in info
        assert "hold_counter" in info
        assert "hold_progress" in info
        assert "in_goal_region" in info
        assert np.isfinite(reward)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)

    def test_joint_limits_enforced(self):
        self.env.reset()

        for _ in range(200):
            action = np.array([1.0, 1.0], dtype=np.float32)
            self.env.step(action)

        angles = self.env.state[:2]
        assert np.all(angles >= self.env.config.joint_limits_min)
        assert np.all(angles <= self.env.config.joint_limits_max)

    def test_episode_truncation(self):
        self.env.reset()

        # Zero action keeps arm away from goal region initially, so truncation should happen.
        for step in range(self.env.max_episode_steps + 5):
            _, _, terminated, truncated, _ = self.env.step(np.zeros(2, dtype=np.float32))
            if step < self.env.max_episode_steps - 1:
                assert truncated is False
            if step == self.env.max_episode_steps - 1:
                assert truncated is True
                assert terminated is False
                break

    def test_custom_shoulder_position(self):
        custom_shoulder = np.array([2.0, 1.5])
        env = ArmTaskEnv(shoulder_base_position=custom_shoulder)

        assert np.allclose(env.shoulder_base_position, custom_shoulder)
        assert env.goal_height == 1.5

        obs, _ = env.reset()
        assert obs.shape == (11,)

    def test_render_mode(self):
        env = ArmTaskEnv(render_mode="human")
        env.reset()

        for _ in range(3):
            env.step(np.array([0.2, 0.2], dtype=np.float32))
            env.render()

    def test_get_state_info(self):
        self.env.reset()

        state_info = self.env.get_state_info()

        assert "joint_angles" in state_info
        assert "joint_velocities" in state_info
        assert "end_effector_position" in state_info
        assert "shoulder_position" in state_info
        assert "workspace_origin" in state_info
        assert "goal_height" in state_info
        assert "target_orientation" in state_info
        assert "distance_to_goal" in state_info
        assert "height_error" in state_info
        assert "orientation_error" in state_info
        assert "gradient_norm" in state_info
        assert "hold_counter" in state_info
        assert "hold_steps_required" in state_info
        assert "hold_progress" in state_info
        assert "in_goal_region" in state_info
        assert "goal_reached" in state_info
        assert "step" in state_info
        assert "max_steps" in state_info

        assert state_info["joint_angles"].shape == (2,)
        assert state_info["joint_velocities"].shape == (2,)
        assert state_info["end_effector_position"].shape == (2,)
        assert state_info["distance_to_goal"] > 0
        assert state_info["goal_reached"] is False

    def test_reward_computation(self):
        self.env.reset()

        _, reward, _, _, info = self.env.step(np.array([0.0, 0.4], dtype=np.float32))

        assert np.isfinite(reward)
        if not info["in_goal_region"]:
            assert reward < 150.0

    def test_forward_kinematics_integration(self):
        self.env.reset()
        angles = self.env.state[:2]

        env_ee = self.env._get_end_effector_position(angles)

        positions = ArmKinematics.forward_kinematics(angles, self.env.config)
        manual_ee = positions[-1, :2] + self.env.shoulder_base_position

        assert np.allclose(env_ee, manual_ee)

    def test_workspace_frame_transformation(self):
        env = ArmTaskEnv(shoulder_base_position=np.array([5.0, 3.0]))
        env.reset()

        angles = env.state[:2]
        ee_pos = env._get_end_effector_position(angles)

        positions = ArmKinematics.forward_kinematics(angles, env.config)
        ee_shoulder_frame = positions[-1, :2]
        expected_ee = ee_shoulder_frame + env.shoulder_base_position

        assert np.allclose(ee_pos, expected_ee)
