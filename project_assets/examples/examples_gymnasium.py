"""Example: Using Gymnasium for Parallel Arm Simulations

This example demonstrates how to run parallel simulations using the
RL Arm Motion project's gymnasium utilities.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_armMotion.utils import (
    ParallelEnvironmentRunner,
    VectorEnvironment,
    run_single_simulation,
)
from rl_armMotion.environments import SimpleArmEnv


def example_1_single_simulation():
    """Example 1: Run a single environment simulation"""
    print("=" * 60)
    print("Example 1: Single Simulation")
    print("=" * 60)

    result = run_single_simulation(
        env_id=0,
        env_name="CartPole-v1",
        num_steps=200,
        seed=42,
    )

    print(f"✓ Completed simulation")
    print(f"  Episode reward: {result.episode_reward:.2f}")
    print(f"  Episode length: {result.episode_length}")
    print(f"  Final observation shape: {result.observation.shape}")
    print()


def example_2_vector_environment():
    """Example 2: Use vectorized environment for parallel execution"""
    print("=" * 60)
    print("Example 2: Vectorized Environment (Synchronous Parallel)")
    print("=" * 60)

    # Create 4 parallel CartPole environments
    vec_env = VectorEnvironment(
        env_names=["CartPole-v1"] * 4,
        seed=42,
    )

    print(f"✓ Created {len(vec_env)} parallel environments")

    total_rewards = np.zeros(len(vec_env))

    # Run for 100 steps
    for step in range(100):
        # Random actions for each environment
        actions = np.array([vec_env.envs[i].action_space.sample() for i in range(len(vec_env))])

        obs, rewards, terms, truncs, infos = vec_env.step(actions)
        total_rewards += rewards

    print(f"✓ Ran 100 steps across all {len(vec_env)} environments")
    print(f"  Total rewards: {total_rewards}")
    print(f"  Average reward: {total_rewards.mean():.2f}")

    vec_env.close()
    print()


def example_3_parallel_runner():
    """Example 3: Use parallel runner with multiprocessing"""
    print("=" * 60)
    print("Example 3: Parallel Runner (Multiprocessing)")
    print("=" * 60)

    with ParallelEnvironmentRunner(num_envs=4, num_processes=4) as runner:
        results = runner.run_simulations(
            env_name="CartPole-v1",
            num_steps=200,
            seed=42,
        )

    print(f"✓ Ran 4 parallel simulations with multiprocessing")
    for i, result in enumerate(results):
        print(f"  Env {i}: reward={result.episode_reward:.2f}, steps={result.episode_length}")

    avg_reward = np.mean([r.episode_reward for r in results])
    print(f"  Average reward: {avg_reward:.2f}")
    print()


def example_4_custom_arm_environment():
    """Example 4: Use custom arm environment"""
    print("=" * 60)
    print("Example 4: Custom SimpleArmEnv (7-DOF Arm)")
    print("=" * 60)

    env = SimpleArmEnv()

    observation, info = env.reset(seed=42)
    print(f"✓ Created SimpleArmEnv")
    print(f"  Observation shape: {observation.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Action shape: {env.action_space.shape}")

    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"  ✓ Reached target after {step} steps!")
            break

    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final position error: {info['position_error']:.4f}")
    print(f"  Reached target: {info['reached_target']}")

    env.close()
    print()


def example_5_parallel_arm_simulations():
    """Example 5: Parallel simulations with custom arm environment"""
    print("=" * 60)
    print("Example 5: Parallel SimpleArmEnv Simulations")
    print("=" * 60)

    # Create 4 independent arm environments
    envs = [SimpleArmEnv() for _ in range(4)]
    observations = [env.reset(seed=42 + i)[0] for i, env in enumerate(envs)]

    print(f"✓ Created {len(envs)} parallel arm environments")

    total_steps = 0
    total_rewards = np.zeros(len(envs))

    for step in range(100):
        actions = [env.action_space.sample() for env in envs]

        for i, (env, action) in enumerate(zip(envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            total_rewards[i] += reward
            observations[i] = obs

            if terminated or truncated:
                observations[i], _ = env.reset()

        total_steps += 1

    print(f"✓ Ran {total_steps} steps across {len(envs)} arm environments")
    print(f"  Total rewards: {total_rewards}")
    print(f"  Average reward: {total_rewards.mean():.2f}")

    for env in envs:
        env.close()
    print()


def example_6_batch_simulations():
    """Example 6: Run different environment types in parallel"""
    print("=" * 60)
    print("Example 6: Batch Parallel Simulations (Multiple Env Types)")
    print("=" * 60)

    with ParallelEnvironmentRunner(num_envs=2) as runner:
        results = runner.run_batch_simulations(
            env_names=["CartPole-v1", "Acrobot-v1"],
            num_steps=200,
            seed=42,
        )

    for env_name, env_results in results.items():
        rewards = [r.episode_reward for r in env_results]
        print(f"✓ {env_name}:")
        print(f"    Rewards: {[f'{r:.2f}' for r in rewards]}")
        print(f"    Average: {np.mean(rewards):.2f}")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  RL Arm Motion: Gymnasium Parallel Simulation Examples  ║")
    print("╚" + "=" * 58 + "╝")
    print()

    example_1_single_simulation()
    example_2_vector_environment()
    example_3_parallel_runner()
    example_4_custom_arm_environment()
    example_5_parallel_arm_simulations()
    example_6_batch_simulations()

    print("=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)
