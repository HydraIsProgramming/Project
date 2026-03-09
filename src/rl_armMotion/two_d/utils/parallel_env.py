"""Parallel environment utilities for running multiple arm simulations

This module provides utilities for running gymnasium environments in parallel
using multiprocessing, suitable for distributed RL training.
"""

import gymnasium as gym
import numpy as np
from multiprocessing import Pool, Manager
from typing import Callable, List, Tuple, Dict, Any
import threading
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Data class for storing simulation results"""
    env_id: int
    episode_reward: float
    episode_length: int
    observation: Any
    info: Dict


def run_single_simulation(
    env_id: int,
    env_name: str,
    num_steps: int,
    policy_func: Callable = None,
    seed: int = None,
) -> SimulationResult:
    """
    Run a single environment simulation.

    Args:
        env_id: Unique identifier for this simulation
        env_name: Gymnasium environment name
        num_steps: Number of steps to run
        policy_func: Function to determine actions (if None, uses random actions)
        seed: Random seed for reproducibility

    Returns:
        SimulationResult containing episode statistics
    """
    env = gym.make(env_name)

    if seed is not None:
        env.reset(seed=seed + env_id)

    observation, info = env.reset()
    total_reward = 0.0

    for step in range(num_steps):
        if policy_func is not None:
            action = policy_func(observation)
        else:
            action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

    return SimulationResult(
        env_id=env_id,
        episode_reward=total_reward,
        episode_length=num_steps,
        observation=observation,
        info=info,
    )


class ParallelEnvironmentRunner:
    """Manages parallel execution of gymnasium environments"""

    def __init__(self, num_envs: int = 4, num_processes: int = None):
        """
        Initialize parallel environment runner.

        Args:
            num_envs: Number of parallel environments
            num_processes: Number of worker processes (default: num_envs)
        """
        self.num_envs = num_envs
        self.num_processes = num_processes or num_envs
        self.pool = None

    def __enter__(self):
        """Context manager entry"""
        self.pool = Pool(processes=self.num_processes)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def run_simulations(
        self,
        env_name: str,
        num_steps: int,
        policy_func: Callable = None,
        seed: int = None,
    ) -> List[SimulationResult]:
        """
        Run multiple environment simulations in parallel.

        Args:
            env_name: Gymnasium environment name
            num_steps: Steps per environment
            policy_func: Policy function for actions
            seed: Random seed

        Returns:
            List of SimulationResult objects
        """
        if self.pool is None:
            raise RuntimeError("ParallelEnvironmentRunner not used as context manager")

        tasks = [
            (env_id, env_name, num_steps, policy_func, seed)
            for env_id in range(self.num_envs)
        ]

        results = self.pool.starmap(run_single_simulation, tasks)
        return results

    def run_batch_simulations(
        self,
        env_names: List[str],
        num_steps: int,
        policy_func: Callable = None,
        seed: int = None,
    ) -> Dict[str, List[SimulationResult]]:
        """
        Run multiple environment types in parallel.

        Args:
            env_names: List of environment names
            num_steps: Steps per environment
            policy_func: Policy function
            seed: Random seed

        Returns:
            Dictionary mapping env_name to results
        """
        results = {}
        for env_name in env_names:
            results[env_name] = self.run_simulations(
                env_name, num_steps, policy_func, seed
            )
        return results


class VectorEnvironment:
    """Vectorized environment wrapper for synchronous parallel execution"""

    def __init__(self, env_names: List[str], seed: int = None):
        """
        Initialize vectorized environments.

        Args:
            env_names: List of environment names (can be duplicates for multiple instances)
            seed: Random seed
        """
        self.envs = [gym.make(name) for name in env_names]
        self.num_envs = len(self.envs)
        self.seed = seed
        self._reset_all()

    def _reset_all(self):
        """Reset all environments"""
        self.observations = []
        self.infos = []
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=self.seed + i if self.seed else None)
            self.observations.append(obs)
            self.infos.append(info)

    def step(self, actions: np.ndarray) -> Tuple:
        """
        Execute one step in all environments.

        Args:
            actions: Array of actions for each environment

        Returns:
            observations, rewards, terminateds, truncateds, infos
        """
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()

            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        self.observations = observations
        self.infos = infos

        return (
            np.array(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos,
        )

    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()

    def __len__(self):
        return self.num_envs


__all__ = [
    "SimulationResult",
    "run_single_simulation",
    "ParallelEnvironmentRunner",
    "VectorEnvironment",
]
