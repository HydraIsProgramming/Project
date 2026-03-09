"""Tests for visualization utilities"""

import sys
from pathlib import Path
import pytest
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_armMotion.utils import (
    ArmVisualizer,
    SimulationVisualizer,
    plot_arm_trajectory_comparison,
)


class TestArmVisualizer:
    """Test cases for ArmVisualizer"""

    def test_arm_visualizer_init(self):
        """Test ArmVisualizer initialization"""
        viz = ArmVisualizer(dof=7)
        assert viz.dof == 7
        assert len(viz.link_lengths) == 7

    def test_forward_kinematics(self):
        """Test forward kinematics computation"""
        viz = ArmVisualizer(dof=7)
        joint_angles = np.zeros(7)

        positions = viz.forward_kinematics(joint_angles)

        # Should return positions array
        assert positions.shape == (8, 3)  # 7 joints + 1 base
        # First position should be base at origin
        np.testing.assert_array_almost_equal(positions[0], [0, 0, 0])

    def test_plot_pose_2d(self):
        """Test 2D pose plotting"""
        viz = ArmVisualizer(dof=7)
        joint_angles = np.random.randn(7) * 0.5

        ax = viz.plot_pose_2d(joint_angles)

        assert ax is not None
        plt.close("all")

    def test_plot_pose_3d(self):
        """Test 3D pose plotting"""
        viz = ArmVisualizer(dof=7)
        joint_angles = np.random.randn(7) * 0.5

        ax = viz.plot_pose_3d(joint_angles)

        assert ax is not None
        plt.close("all")

    def test_plot_trajectory(self):
        """Test trajectory plotting"""
        viz = ArmVisualizer(dof=7)
        trajectory = np.random.randn(50, 7) * 0.5

        fig = viz.plot_trajectory(trajectory)

        assert fig is not None
        plt.close("all")

    def test_plot_joint_angles(self):
        """Test joint angles plotting"""
        viz = ArmVisualizer(dof=7)
        trajectory = np.random.randn(50, 7) * 0.5

        fig = viz.plot_joint_angles(trajectory)

        assert fig is not None
        plt.close("all")

    def test_animate_trajectory(self):
        """Test trajectory animation creation"""
        viz = ArmVisualizer(dof=7)
        trajectory = np.random.randn(10, 7) * 0.5

        # Test animation without saving
        anim = viz.animate_trajectory(trajectory)

        assert anim is not None
        plt.close("all")


class TestSimulationVisualizer:
    """Test cases for SimulationVisualizer"""

    def test_plot_rewards(self):
        """Test reward plotting"""
        rewards_list = [np.random.randn(100) for _ in range(4)]

        fig = SimulationVisualizer.plot_rewards(rewards_list)

        assert fig is not None
        plt.close("all")

    def test_plot_parallel_stats(self):
        """Test parallel statistics plotting"""
        results = [
            {"episode_reward": np.random.randn(), "episode_length": 100, "env_id": i}
            for i in range(4)
        ]

        fig = SimulationVisualizer.plot_parallel_stats(results)

        assert fig is not None
        plt.close("all")

    def test_create_interactive_dashboard(self):
        """Test interactive dashboard creation"""
        results = [
            {"episode_reward": np.random.randn(), "episode_length": 100, "env_id": i}
            for i in range(4)
        ]

        fig = SimulationVisualizer.create_interactive_dashboard(results)

        assert fig is not None


class TestTrajectoryComparison:
    """Test cases for trajectory comparison"""

    def test_plot_trajectory_comparison(self):
        """Test trajectory comparison plotting"""
        trajectories = [
            np.random.randn(50, 7) * 0.5 for _ in range(3)
        ]

        fig = plot_arm_trajectory_comparison(trajectories)

        assert fig is not None
        plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
