"""Test and demonstration of arm model visualization

This script demonstrates all visualization capabilities for the SimpleArmEnv
and parallel simulations.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_armMotion.utils import (
    ArmVisualizer,
    SimulationVisualizer,
    ParallelEnvironmentRunner,
    VectorEnvironment,
)
from rl_armMotion.environments import SimpleArmEnv


def test_arm_visualization_basic():
    """Test 1: Basic arm visualization"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Arm Visualization (2D and 3D)")
    print("=" * 70)

    viz = ArmVisualizer(dof=7)

    # Random pose
    joint_angles = np.random.uniform(-np.pi, np.pi, 7)

    # Plot 2D pose
    print("✓ Generating 2D pose visualization...")
    fig_2d = plt.figure(figsize=(12, 5))
    ax1 = fig_2d.add_subplot(121)
    viz.plot_pose_2d(joint_angles, ax=ax1, title="7-DOF Arm - 2D View")

    # Plot 3D pose
    ax2 = fig_2d.add_subplot(122, projection="3d")
    viz.plot_pose_3d(joint_angles, ax=ax2, title="7-DOF Arm - 3D View")

    plt.tight_layout()
    plt.savefig("test_arm_pose_visualization.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: test_arm_pose_visualization.png")
    print(f"  Joint angles: {joint_angles}")
    print(f"  Forward kinematics computed for chain with {len(joint_angles)} DOF")
    plt.close()


def test_trajectory_visualization():
    """Test 2: Trajectory visualization"""
    print("\n" + "=" * 70)
    print("TEST 2: Trajectory Visualization")
    print("=" * 70)

    viz = ArmVisualizer(dof=7)

    # Generate trajectory (simple oscillation)
    num_steps = 100
    trajectory = np.zeros((num_steps, 7))
    for i in range(7):
        trajectory[:, i] = np.sin(np.linspace(0, 4*np.pi, num_steps)) * (np.pi / 4)

    print(f"✓ Generated trajectory with {num_steps} timesteps and {7} joints")

    # Plot trajectory
    print("✓ Generating trajectory visualization...")
    fig = viz.plot_trajectory(trajectory, figsize=(14, 6))
    plt.savefig("test_arm_trajectory.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: test_arm_trajectory.png")
    print(f"  Trajectory shape: {trajectory.shape}")
    plt.close()

    # Plot joint angles
    print("✓ Generating joint angles plot...")
    fig = viz.plot_joint_angles(trajectory, figsize=(12, 10))
    plt.savefig("test_arm_joint_angles.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: test_arm_joint_angles.png")
    plt.close()


def test_simple_arm_env_visualization():
    """Test 3: SimpleArmEnv visualization"""
    print("\n" + "=" * 70)
    print("TEST 3: SimpleArmEnv Episodes Visualization")
    print("=" * 70)

    env = SimpleArmEnv()
    viz = ArmVisualizer(dof=7)

    # Run multiple episodes and collect trajectories
    episodes_data = []
    for episode in range(3):
        obs, _ = env.reset(seed=42 + episode)
        trajectory = []
        rewards = []
        episode_reward = 0

        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Extract joint angles from observation
            joint_angles = obs[:7]
            trajectory.append(joint_angles)
            rewards.append(reward)
            episode_reward += reward

            if terminated or truncated:
                break

        trajectory = np.array(trajectory)
        episodes_data.append({
            "trajectory": trajectory,
            "rewards": rewards,
            "total_reward": episode_reward,
            "episode_num": episode,
        })
        print(f"✓ Episode {episode}: steps={len(trajectory)}, reward={episode_reward:.2f}")

    env.close()

    # Visualize first episode
    print("✓ Generating episode visualization...")
    fig = viz.plot_trajectory(episodes_data[0]["trajectory"])
    plt.savefig("test_simple_arm_episode.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: test_simple_arm_episode.png")
    plt.close()

    # Plot rewards for all episodes
    print("✓ Generating episode rewards visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, episode_data in enumerate(episodes_data):
        axes[i].plot(np.cumsum(episode_data["rewards"]), linewidth=2, color="blue")
        axes[i].fill_between(range(len(episode_data["rewards"])),
                            np.cumsum(episode_data["rewards"]), alpha=0.3)
        axes[i].set_title(f"Episode {i+1} - Total Reward: {episode_data['total_reward']:.2f}")
        axes[i].set_xlabel("Timestep")
        axes[i].set_ylabel("Cumulative Reward")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("test_simple_arm_episode_rewards.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: test_simple_arm_episode_rewards.png")
    plt.close()

    return episodes_data


def test_parallel_simulation_visualization():
    """Test 4: Parallel simulation visualization"""
    print("\n" + "=" * 70)
    print("TEST 4: Parallel Simulation Visualization")
    print("=" * 70)

    # Run parallel simulations using VectorEnvironment
    print("✓ Running parallel simulations (4 environments)...")
    vec_env = VectorEnvironment(["CartPole-v1"] * 4, seed=42)

    all_rewards = [[] for _ in range(4)]
    for step in range(200):
        actions = np.array([vec_env.envs[i].action_space.sample() for i in range(4)])
        obs, rewards, terms, truncs, infos = vec_env.step(actions)

        for i, reward in enumerate(rewards):
            all_rewards[i].append(reward)

    vec_env.close()

    # Visualization using SimulationVisualizer
    print("✓ Generating parallel simulation statistics...")
    fig = SimulationVisualizer.plot_rewards(
        all_rewards,
        labels=[f"Env {i}" for i in range(4)],
        figsize=(12, 6)
    )
    plt.savefig("test_parallel_rewards.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: test_parallel_rewards.png")
    plt.close()

    # Create mock simulation results
    simulation_results = [
        {
            "env_id": i,
            "episode_reward": np.sum(all_rewards[i]),
            "episode_length": len(all_rewards[i]),
        }
        for i in range(4)
    ]

    print("✓ Generating parallel statistics dashboard...")
    fig = SimulationVisualizer.plot_parallel_stats(simulation_results, figsize=(14, 10))
    plt.savefig("test_parallel_statistics.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: test_parallel_statistics.png")
    plt.close()

    # Plotly interactive dashboard
    print("✓ Generating interactive Plotly dashboard...")
    fig_plotly = SimulationVisualizer.create_interactive_dashboard(simulation_results)
    fig_plotly.write_html("test_parallel_interactive_dashboard.html")
    print("✓ Saved: test_parallel_interactive_dashboard.html")

    # Print statistics
    print("\n  Parallel Simulation Statistics:")
    total_rewards = [np.sum(rewards) for rewards in all_rewards]
    print(f"    Mean Reward: {np.mean(total_rewards):.2f}")
    print(f"    Std Reward:  {np.std(total_rewards):.2f}")
    print(f"    Min Reward:  {np.min(total_rewards):.2f}")
    print(f"    Max Reward:  {np.max(total_rewards):.2f}")


def test_trajectory_comparison():
    """Test 5: Multiple trajectory comparison"""
    print("\n" + "=" * 70)
    print("TEST 5: Trajectory Comparison")
    print("=" * 70)

    from rl_armMotion.utils import plot_arm_trajectory_comparison

    # Generate multiple trajectories
    np.random.seed(42)
    trajectories = []
    for traj_num in range(3):
        num_steps = 100
        trajectory = np.random.randn(num_steps, 7) * 0.5
        trajectories.append(trajectory)
        print(f"✓ Generated trajectory {traj_num + 1}")

    # Compare trajectories
    print("✓ Generating trajectory comparison visualization...")
    fig = plot_arm_trajectory_comparison(
        trajectories,
        labels=["Random Path 1", "Random Path 2", "Random Path 3"],
        figsize=(14, 6)
    )
    plt.savefig("test_trajectory_comparison.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: test_trajectory_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║   RL Arm Motion: Visualization Test Suite                         ║")
    print("╚" + "=" * 68 + "╝")

    try:
        test_arm_visualization_basic()
        test_trajectory_visualization()
        episodes_data = test_simple_arm_env_visualization()
        test_parallel_simulation_visualization()
        test_trajectory_comparison()

        print("\n" + "=" * 70)
        print("✅ ALL VISUALIZATION TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nGenerated Files:")
        print("  - test_arm_pose_visualization.png")
        print("  - test_arm_trajectory.png")
        print("  - test_arm_joint_angles.png")
        print("  - test_simple_arm_episode.png")
        print("  - test_simple_arm_episode_rewards.png")
        print("  - test_parallel_rewards.png")
        print("  - test_parallel_statistics.png")
        print("  - test_parallel_interactive_dashboard.html")
        print("  - test_trajectory_comparison.png")
        print("\n📊 Open the .html file in a browser for interactive dashboard!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
