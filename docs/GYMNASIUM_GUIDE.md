# Gymnasium Integration Guide

## Overview

Your RL Arm Motion project now includes **full Gymnasium support** for creating and managing parallel simulations of your robotic arm environments.

## Status

✅ **Gymnasium is installed** (v1.0.0)
✅ **Added to project dependencies** (requirements.txt & pyproject.toml)
✅ **Parallel execution utilities created**
✅ **Custom 7-DOF arm environment implemented**
✅ **Comprehensive examples provided**

---

## What's New

### 1. **Gymnasium Dependencies**

Added to your project:
- **gymnasium** (v1.0.0+) - RL environment standard
- **gymnasium-robotics** (v0.7.0+) - Robotics-specific environments

Install with:
```bash
pip install -r requirements.txt
```

Or for optional parallel features:
```bash
pip install -e ".[parallel]"
```

### 2. **Parallel Environment Utilities**
**File**: `src/rl_armMotion/utils/parallel_env.py`

#### **ParallelEnvironmentRunner** - Multiprocessing Support
Run multiple environments across CPU cores:

```python
from rl_armMotion.utils import ParallelEnvironmentRunner

with ParallelEnvironmentRunner(num_envs=4) as runner:
    results = runner.run_simulations(
        env_name="CartPole-v1",
        num_steps=200,
        seed=42,
    )

for result in results:
    print(f"Reward: {result.episode_reward}, Steps: {result.episode_length}")
```

**Use case**: Distributed training across multiple CPU cores

#### **VectorEnvironment** - Synchronous Parallel
Run multiple environments in lock-step:

```python
from rl_armMotion.utils import VectorEnvironment
import numpy as np

vec_env = VectorEnvironment(
    env_names=["CartPole-v1", "CartPole-v1", "CartPole-v1"],
    seed=42,
)

for _ in range(100):
    actions = np.array([vec_env.envs[i].action_space.sample()
                       for i in range(len(vec_env))])
    obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    print(f"Rewards: {rewards}")

vec_env.close()
```

**Use case**: Batch training, collecting experience efficiently

### 3. **Custom SimpleArmEnv**
**File**: `src/rl_armMotion/environments/simple_arm.py`

A 7-DOF robotic arm environment built with Gymnasium:

```python
from rl_armMotion.environments import SimpleArmEnv

env = SimpleArmEnv()
observation, info = env.reset(seed=42)

# observation: 14-dim [7 joint angles + 7 joint velocities]
# action: 7-dim target joint velocities
# reward: based on reaching target + energy efficiency

for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        print(f"Reached target after {step} steps!")
        break

print(f"Position error: {info['position_error']:.4f}")
print(f"Energy used: {info['energy']:.4f}")

env.close()
```

**Features**:
- 7-DOF joint configuration
- Joint limits and velocity constraints
- Reaching task with target position
- Energy efficiency rewards
- Extensible for custom dynamics

---

## Usage Examples

### Example 1: Running Parallel Simulations of Your Arm

```python
from rl_armMotion.utils import VectorEnvironment
from rl_armMotion.environments import SimpleArmEnv
import numpy as np

# Create 4 parallel arm environments
vec_env = VectorEnvironment(
    env_names=["simplearm-v0", "simplearm-v0", "simplearm-v0", "simplearm-v0"],
    seed=42,
)

# Collect experience
observations_batch = []
actions_batch = []
rewards_batch = []

for episode in range(100):
    obs, info = vec_env.reset()
    for step in range(500):
        # Your policy here
        actions = np.array([vec_env.envs[i].action_space.sample()
                           for i in range(len(vec_env))])

        obs, rewards, terms, truncs, infos = vec_env.step(actions)
        rewards_batch.append(rewards)

        if np.any(terms | truncs):
            obs, info = vec_env.reset()

vec_env.close()

print(f"Total experience collected: {len(rewards_batch)} batches")
print(f"Average batch reward: {np.mean(rewards_batch):.2f}")
```

### Example 2: Multiprocessing for CPU-Heavy Training

```python
from rl_armMotion.utils import ParallelEnvironmentRunner

def my_policy(observation):
    """Your RL policy function"""
    return observation  # Placeholder

with ParallelEnvironmentRunner(num_envs=8, num_processes=8) as runner:
    # Run 8 independent simulations in parallel
    results = runner.run_simulations(
        env_name="CartPole-v1",
        num_steps=1000,
        policy_func=my_policy,
        seed=42,
    )

    for i, result in enumerate(results):
        print(f"Process {i}: reward={result.episode_reward:.2f}")
```

### Example 3: Batch Testing Different Environments

```python
from rl_armMotion.utils import ParallelEnvironmentRunner

with ParallelEnvironmentRunner(num_envs=2) as runner:
    results = runner.run_batch_simulations(
        env_names=["CartPole-v1", "Acrobot-v1", "MountainCar-v0"],
        num_steps=500,
        seed=42,
    )

    for env_name, env_results in results.items():
        avg_reward = sum(r.episode_reward for r in env_results) / len(env_results)
        print(f"{env_name}: avg_reward={avg_reward:.2f}")
```

---

## Available Methods

### ParallelEnvironmentRunner

```python
# Initialize
runner = ParallelEnvironmentRunner(num_envs=4, num_processes=4)

# Context manager
with runner:
    # Run single environment type
    results = runner.run_simulations(
        env_name="CartPole-v1",
        num_steps=100,
        policy_func=None,  # Optional policy
        seed=42,
    )

    # Run multiple environment types
    results = runner.run_batch_simulations(
        env_names=["CartPole-v1", "Acrobot-v1"],
        num_steps=100,
        seed=42,
    )
```

### VectorEnvironment

```python
# Initialize with environment names
vec_env = VectorEnvironment(
    env_names=["CartPole-v1", "CartPole-v1"],
    seed=42,
)

# Step all environments
obs, rewards, terms, truncs, infos = vec_env.step(actions)

# Properties
len(vec_env)  # Number of environments

vec_env.close()
```

### SimpleArmEnv

```python
env = SimpleArmEnv()

# Reset
obs, info = env.reset(seed=42)

# Step
obs, reward, terminated, truncated, info = env.step(action)

# Properties
env.action_space      # Action bounds
env.observation_space # Observation bounds
env.num_dof          # Degrees of freedom (7)

# Environment info
info['position_error']  # Distance to target
info['energy']         # Energy consumed
info['reached_target']  # Success flag
```

---

## Running the Examples

All examples are in `project_assets/examples/examples_gymnasium.py`:

```bash
# Run all examples
python3 project_assets/examples/examples_gymnasium.py

# Run specific example by editing the main block
python3 -c "import sys; sys.path.append('project_assets/examples'); from examples_gymnasium import example_1_single_simulation; example_1_single_simulation()"
```

**Examples include**:
1. Single simulation
2. Vectorized environment (synchronous)
3. Parallel runner (multiprocessing)
4. Custom arm environment
5. Parallel arm simulations
6. Batch simulations with multiple env types

---

## Customizing SimpleArmEnv

Modify `src/rl_armMotion/environments/simple_arm.py` to:

### Change DOF count:
```python
env = SimpleArmEnv(num_dof=6)  # 6-DOF instead of 7-DOF
```

### Modify dynamics:
```python
# In step() method:
self.dt = 0.05  # Increase timestep
self.max_episode_steps = 1000  # Longer episodes
```

### Update reward function:
```python
# In step() method, modify reward calculation
reward = -position_error - 0.1 * energy_cost + bonus_terms
```

### Add constraints:
```python
# In __init__:
self.joint_limits = np.array([...])  # Custom limits
self.velocity_limits = 3.0  # Custom velocity bounds
```

---

## Testing

Run the test suite:

```bash
# Run all tests
pytest project_assets/tests/

# Run parallel environment tests
pytest project_assets/tests/test_parallel_env.py -v

# Run with coverage
pytest project_assets/tests/ --cov=src/rl_armMotion
```

Tests verify:
- Single simulations
- Vectorized environments
- Parallel runners
- Environment consistency
- Seeded reproducibility

---

## Performance Tips

### For VectorEnvironment:
- Use when you need synchronized stepping
- Fastest for collecting experience batches
- Good for policy gradient methods

### For ParallelEnvironmentRunner:
- Use for independent simulations
- Better for embarrassingly parallel tasks
- Good for evolution strategies, population-based training

### General Tips:
1. Use meaningful seeds for reproducibility
2. Batch size = num_envs * steps_per_env
3. Profile your policy function - it's often the bottleneck
4. Monitor CPU/memory usage with multiple environments

---

## Next Steps

1. **Implement your arm kinematics** in SimpleArmEnv
2. **Add reward shaping** based on your task
3. **Train RL agents** using the parallel utilities
4. **Benchmark performance** with multiple environments
5. **Integrate with PyTorch/TensorFlow** for learning

---

## Common Issues

**ImportError: No module named 'gymnasium'**
```bash
pip install gymnasium gymnasium-robotics
```

**VectorEnvironment shows different rewards than expected**
- Check random seeds are set correctly
- Vectorized execution may show stochasticity
- Use single environment for debugging

**Multiprocessing hangs**
- Ensure policy_func is picklable (avoid lambdas)
- Don't use GPU in worker processes
- Test with small num_envs first

---

## Files Modified/Created

✅ Created:
- `src/rl_armMotion/utils/parallel_env.py` - Parallel utilities
- `src/rl_armMotion/environments/simple_arm.py` - Custom arm env
- `project_assets/tests/test_parallel_env.py` - Test suite
- `project_assets/examples/examples_gymnasium.py` - Comprehensive examples

✅ Updated:
- `requirements.txt` - Added gymnasium
- `pyproject.toml` - Added gymnasium + parallel extras
- `src/rl_armMotion/utils/__init__.py` - Export parallel utils
- `src/rl_armMotion/environments/__init__.py` - Export SimpleArmEnv

---

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Gymnasium Robotics](https://robotics.farama.org/)
- [Python Multiprocessing](https://docs.python.org/3/library/multiprocessing.html)

---

**Last Updated**: 2026-02-23
**Gymnasium Version**: 1.0.0
**Status**: ✅ Ready for parallel arm motion simulations
