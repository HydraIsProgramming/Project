"""
Demonstration of the ArmTaskEnv - Virtual Environment for 2-DOF Robotic Arm

This script shows how to:
1. Initialize the virtual environment
2. Run the arm from initial state (vertical downward) toward goal (horizontal)
3. Monitor the environment's state and rewards
4. Visualize the task progression
"""

import numpy as np
import sys
from rl_armMotion.environments.task_env import ArmTaskEnv


def run_demo_episode(env, num_steps=500, action_mode="random"):
    """
    Run a demonstration episode in the environment.

    Args:
        env: ArmTaskEnv instance
        num_steps: Maximum number of steps
        action_mode: "random" for random actions or "heuristic" for goal-seeking
    """
    print("=" * 80)
    print(f"Starting Demo Episode - Action Mode: {action_mode}")
    print("=" * 80)
    print(f"Shoulder base position: {env.shoulder_base_position}")
    print(f"Goal height: {env.goal_height}")
    print(f"Initial arm state: {env.config.initial_angles} (vertical downward)")
    print()

    obs, info = env.reset()
    episode_reward = 0
    goal_reached = False
    best_distance = float("inf")

    # Monitor key metrics
    distances = []
    rewards = []
    angles_history = []

    for step in range(num_steps):
        # Choose action based on mode
        if action_mode == "random":
            # Random exploration
            action = env.action_space.sample()
        elif action_mode == "heuristic":
            # Simple heuristic: try to move elbow upward to reach horizontal
            angles = obs[:2]
            current_distance = env._compute_goal_distance(
                env._get_end_effector_position(angles)
            )

            if current_distance > env.goal_tolerance:
                # Move elbow upward (positive action)
                action = np.array([0.0, 1.0])
            else:
                # Reached goal, small actions
                action = np.array([0.0, 0.1])
        else:
            # Zero action
            action = np.zeros(2)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Track metrics
        distances.append(info["goal_distance"])
        rewards.append(reward)
        angles_history.append(obs[:2].copy())

        best_distance = min(best_distance, info["goal_distance"])

        # Print progress
        if step % 50 == 0 or info["goal_reached"]:
            angles = obs[:2]
            ee_pos = info["end_effector_position"]
            print(
                f"Step {step:3d} | "
                f"Angles: [{angles[0]:6.3f}, {angles[1]:6.3f}] | "
                f"E.E.: [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}] | "
                f"Distance: {info['goal_distance']:.3f} | "
                f"Reward: {reward:7.2f}"
            )

        if info["goal_reached"]:
            goal_reached = True
            print(f"\n✓ GOAL REACHED at step {step}!")
            break

        if truncated:
            print(f"\nEpisode truncated at max steps ({step})")
            break

    # Summary statistics
    print("\n" + "=" * 80)
    print("Episode Summary:")
    print("=" * 80)
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Goal reached: {goal_reached}")
    print(f"Best distance achieved: {best_distance:.4f}")
    print(f"Total steps: {step + 1}")
    print(f"Average reward per step: {episode_reward / (step + 1):.3f}")

    if distances:
        print(f"Final distance to goal: {distances[-1]:.4f}")
        print(f"Initial distance to goal: {distances[0]:.4f}")
        print(f"Distance improvement: {distances[0] - distances[-1]:.4f}")

    return {
        "episode_reward": episode_reward,
        "goal_reached": goal_reached,
        "best_distance": best_distance,
        "steps": step + 1,
        "distances": distances,
        "rewards": rewards,
        "angles_history": angles_history,
    }


def demo_workspace_setup():
    """Demonstrate workspace setup and configuration."""
    print("\n" + "=" * 80)
    print("WORKSPACE SETUP DEMONSTRATION")
    print("=" * 80)

    # Default setup
    print("\n1. Default Setup (shoulder at [1.0, 0]):")
    env1 = ArmTaskEnv()
    obs, _ = env1.reset()
    state_info = env1.get_state_info()
    print(f"   Shoulder position: {state_info['shoulder_position']}")
    print(f"   Workspace origin: {state_info['workspace_origin']}")
    print(f"   Goal height: {state_info['goal_height']}")
    print(f"   End-effector position: {state_info['end_effector_position']}")
    print(f"   Distance to goal: {state_info['distance_to_goal']:.4f}")

    # Custom setup
    print("\n2. Custom Setup (shoulder at [2.0, 1.5]):")
    custom_shoulder = np.array([2.0, 1.5])
    env2 = ArmTaskEnv(shoulder_base_position=custom_shoulder)
    obs, _ = env2.reset()
    state_info = env2.get_state_info()
    print(f"   Shoulder position: {state_info['shoulder_position']}")
    print(f"   Workspace origin: {state_info['workspace_origin']}")
    print(f"   Goal height: {state_info['goal_height']}")
    print(f"   End-effector position: {state_info['end_effector_position']}")
    print(f"   Distance to goal: {state_info['distance_to_goal']:.4f}")

    # Multiple configurations
    print("\n3. Multiple workspace configurations:")
    positions = [
        [0.5, 0.0],
        [1.0, 0.0],
        [1.5, 0.0],
        [1.0, 1.0],
        [1.0, -1.0],
    ]
    for pos in positions:
        env = ArmTaskEnv(shoulder_base_position=np.array(pos))
        obs, _ = env.reset()
        state_info = env.get_state_info()
        distance = state_info["distance_to_goal"]
        print(f"   Shoulder at {pos} → Initial distance to goal: {distance:.4f}")


def demo_action_modes():
    """Demonstrate different action modes."""
    print("\n" + "=" * 80)
    print("ACTION MODES DEMONSTRATION")
    print("=" * 80)

    # Random actions
    print("\n1. Random Action Policy:")
    env = ArmTaskEnv()
    results_random = run_demo_episode(env, num_steps=300, action_mode="random")

    # Heuristic actions
    print("\n\n2. Heuristic Goal-Seeking Policy:")
    env = ArmTaskEnv()
    results_heuristic = run_demo_episode(env, num_steps=300, action_mode="heuristic")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Random:    Goal reached: {results_random['goal_reached']}, "
          f"Best distance: {results_random['best_distance']:.4f}")
    print(f"Heuristic: Goal reached: {results_heuristic['goal_reached']}, "
          f"Best distance: {results_heuristic['best_distance']:.4f}")


def demo_arm_dynamics():
    """Demonstrate arm dynamics and joint constraints."""
    print("\n" + "=" * 80)
    print("ARM DYNAMICS & CONSTRAINTS DEMONSTRATION")
    print("=" * 80)

    env = ArmTaskEnv()
    obs, _ = env.reset()

    print("\n1. Initial Configuration:")
    state_info = env.get_state_info()
    print(f"   Shoulder angle: {state_info['joint_angles'][0]:.4f} rad ({np.degrees(state_info['joint_angles'][0]):.1f}°)")
    print(f"   Elbow angle: {state_info['joint_angles'][1]:.4f} rad ({np.degrees(state_info['joint_angles'][1]):.1f}°)")
    print(f"   Joint limits - Shoulder: [{env.config.joint_limits_min[0]:.4f}, {env.config.joint_limits_max[0]:.4f}]")
    print(f"   Joint limits - Elbow: [{env.config.joint_limits_min[1]:.4f}, {env.config.joint_limits_max[1]:.4f}]")
    print(f"                        [{np.degrees(env.config.joint_limits_min[1]):.1f}°, {np.degrees(env.config.joint_limits_max[1]):.1f}°]")

    print("\n2. Testing Joint Constraints:")
    print("   (Trying to move elbow backward - should stay at 0°)")
    for _ in range(20):
        action = np.array([0.0, -2.0])  # Extreme negative action
        obs, _, _, _, info = env.step(action)

    state_info = env.get_state_info()
    print(f"   Elbow angle after constraint test: {state_info['joint_angles'][1]:.4f} rad")
    print(f"   Constraint enforced: {state_info['joint_angles'][1] >= env.config.joint_limits_min[1]}")

    print("\n3. Reaching Horizontal Position:")
    obs, _ = env.reset()
    for step in range(200):
        action = np.array([0.0, 1.5])  # Move elbow upward
        obs, reward, terminated, _, info = env.step(action)
        if info["goal_reached"]:
            state_info = env.get_state_info()
            print(f"   Horizontal position reached at step {step}")
            print(f"   Final shoulder angle: {state_info['joint_angles'][0]:.4f} rad")
            print(f"   Final elbow angle: {state_info['joint_angles'][1]:.4f} rad ({np.degrees(state_info['joint_angles'][1]):.1f}°)")
            print(f"   End-effector height: {info['end_effector_position'][1]:.4f} m")
            print(f"   Goal height: {info['goal_height']:.4f} m")
            print(f"   Distance from goal: {info['goal_distance']:.4f} m")
            break


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + "  RL ARM MOTION - VIRTUAL ENVIRONMENT DEMONSTRATION".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    print("This demonstration shows the virtual environment for training the 2-DOF robotic arm")
    print("to move from vertical downward position to horizontal position.")
    print()

    # Run demonstrations
    demo_workspace_setup()
    demo_arm_dynamics()
    demo_action_modes()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Points:")
    print("- The environment uses correct forward kinematics (fixed serial chain)")
    print("- Joint constraints are properly enforced (elbow 0° to 120°)")
    print("- Goal detection works based on end-effector height")
    print("- Reward structure encourages smooth, goal-directed motion")
    print("- Environment is compatible with standard RL algorithms (gymnasium API)")
    print("\nReady for RL agent training!")
    print()


if __name__ == "__main__":
    main()
