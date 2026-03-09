"""Tests for parallel environment utilities"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_armMotion.utils.parallel_env import (
    ParallelEnvironmentRunner,
    VectorEnvironment,
    run_single_simulation,
)


class TestParallelEnvironments:
    """Test cases for parallel environment utilities"""

    def test_single_simulation(self):
        """Test running a single environment simulation"""
        result = run_single_simulation(
            env_id=0,
            env_name="CartPole-v1",
            num_steps=100,
            seed=42,
        )

        assert result.env_id == 0
        assert result.episode_length == 100
        assert isinstance(result.episode_reward, float)
        assert result.observation is not None

    def test_vector_environment(self):
        """Test vectorized environment wrapper"""
        env_names = ["CartPole-v1", "CartPole-v1"]
        vec_env = VectorEnvironment(env_names, seed=42)

        assert len(vec_env) == 2

        # Test step
        actions = np.array([0, 1])
        observations, rewards, terminateds, truncateds, infos = vec_env.step(actions)

        assert observations.shape[0] == 2
        assert rewards.shape[0] == 2
        assert len(infos) == 2

        vec_env.close()

    def test_parallel_environment_runner(self):
        """Test parallel environment runner"""
        with ParallelEnvironmentRunner(num_envs=2) as runner:
            results = runner.run_simulations(
                env_name="CartPole-v1",
                num_steps=50,
                seed=42,
            )

        assert len(results) == 2
        assert all(hasattr(r, "episode_reward") for r in results)
        assert all(hasattr(r, "episode_length") for r in results)

    def test_parallel_runner_multiple_envs(self):
        """Test parallel runner with different environments"""
        with ParallelEnvironmentRunner(num_envs=2) as runner:
            results = runner.run_batch_simulations(
                env_names=["CartPole-v1", "Acrobot-v1"],
                num_steps=50,
                seed=42,
            )

        assert "CartPole-v1" in results
        assert "Acrobot-v1" in results
        assert len(results["CartPole-v1"]) == 2
        assert len(results["Acrobot-v1"]) == 2

    def test_parallel_consistency(self):
        """Test that seeded simulations produce consistent results"""
        def policy(obs):
            return 1  # Always take action 1

        result1 = run_single_simulation(
            env_id=0,
            env_name="CartPole-v1",
            num_steps=100,
            policy_func=policy,
            seed=42,
        )

        result2 = run_single_simulation(
            env_id=0,
            env_name="CartPole-v1",
            num_steps=100,
            policy_func=policy,
            seed=42,
        )

        # Rewards should be identical with same seed
        np.testing.assert_almost_equal(result1.episode_reward, result2.episode_reward)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
