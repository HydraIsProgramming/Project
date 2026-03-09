"""Compare PPO and SAC on the Weng gait environment.

This script trains reinforcement learning agents using Proximal Policy
Optimisation (PPO) and Soft Actor–Critic (SAC) on the gait environment
defined in :mod:`rl_armMotion.environments.weng_gait_env`.  After
training, both agents are evaluated on identical episodes to assess
their performance in terms of success rate, time to target, smoothness
and effort.  Training logs for each algorithm can optionally be
saved to disk and visualised using the :func:`plot_training_progress`
helper defined in :mod:`rl_armMotion.training.weng_gait_trainer`.

Usage
-----
You can run this module as a script to perform a default comparison:

.. code-block:: bash

    python -m rl_armMotion.training.compare_algorithms

Custom settings can be provided via command-line arguments.  See
``python -m rl_armMotion.training.compare_algorithms --help`` for
options.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from rl_armMotion.training.weng_gait_trainer import WengGaitTrainer, plot_training_progress


def run_comparison(
    *,
    total_timesteps: int = 200_000,
    num_eval_episodes: int = 20,
    seed: Optional[int] = None,
    log_dir: Optional[str] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> None:
    """Train and evaluate PPO and SAC agents on the gait environment.

    This function instantiates two separate trainers—one for PPO and
    another for SAC—using the same environment configuration.  Each
    agent is trained for ``total_timesteps`` and subsequently
    evaluated on ``num_eval_episodes`` episodes.  The results are
    printed to stdout and, when ``log_dir`` is provided, training
    progress is logged to CSV files under that directory.

    Parameters
    ----------
    total_timesteps: int, optional
        Number of environment timesteps to train each agent.  Defaults
        to 200 000.
    num_eval_episodes: int, optional
        Number of evaluation episodes for computing the metrics.  Defaults
        to 20.
    seed: int, optional
        Random seed for reproducibility.  When ``None`` the default
        randomness of the underlying libraries is used.
    log_dir: str, optional
        Directory in which to save per‑episode training logs.  Two
        files will be created: ``ppo_log.csv`` and ``sac_log.csv``.
        When omitted, no log files are generated.
    hyperparams: dict, optional
        Hyperparameters to override defaults when constructing the
        trainers.  These hyperparameters are applied to both PPO
        and SAC agents.  See :func:`rl_armMotion.training.weng_gait_trainer._default_hyperparams`
        for details.
    """
    algorithms = ["PPO", "SAC"]
    results = {}
    # Ensure log directory exists when specified
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    for alg in algorithms:
        log_path = None
        if log_dir is not None:
            log_path = str(Path(log_dir) / f"{alg.lower()}_log.csv")
        trainer = WengGaitTrainer(
            env=None,
            total_timesteps=total_timesteps,
            algorithm=alg,
            hyperparams=hyperparams,
            seed=seed,
            log_path=log_path,
        )
        print(f"Training {alg} agent for {total_timesteps} timesteps...")
        trainer.train()
        res = trainer.evaluate(num_eval_episodes)
        results[alg] = res
        # Optionally plot training progress for each algorithm
        if log_dir is not None and log_path is not None:
            print(f"Plotting training curves for {alg} (saved to {log_dir})...")
            # Use default metrics list
            plot_training_progress(log_path, save_dir=str(Path(log_dir) / alg.lower()))
    # Print summary results
    print("\nComparison results:")
    for alg in algorithms:
        res = results[alg]
        print(
            f"{alg}: success_rate={res.success_rate:.2f}, mean_time_to_target={res.mean_time_to_target:.1f}, "
            f"mean_pose_error={res.mean_pose_error:.3f}, mean_smoothness={res.mean_smoothness:.2f}, "
            f"mean_effort={res.mean_effort:.2f}, safety_violations={res.safety_violations}"
        )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison script."""
    parser = argparse.ArgumentParser(description="Compare PPO and SAC on the gait environment")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps per agent")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory to save training logs and plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_comparison(
        total_timesteps=args.timesteps,
        num_eval_episodes=args.eval_episodes,
        seed=args.seed,
        log_dir=args.log_dir,
    )