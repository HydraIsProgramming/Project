# Visualization Guide

## Overview

Your RL Arm Motion project now includes **comprehensive visualization utilities** for:
- 3D arm model visualization
- Trajectory plotting and analysis
- Parallel simulation monitoring
- Interactive dashboards
- Animation generation

---

## Visualization Components

### 1. **ArmVisualizer** - Arm Model Visualization

Visualize your 7-DOF arm in 2D and 3D with forward kinematics.

#### Basic Usage

```python
from rl_armMotion.utils import ArmVisualizer
import numpy as np

# Create visualizer
viz = ArmVisualizer(dof=7)

# Generate random joint angles
joint_angles = np.random.randn(7) * 0.5

# Plot 2D configuration
ax = viz.plot_pose_2d(joint_angles, title="Arm Configuration - Top View")

# Plot 3D configuration
ax = viz.plot_pose_3d(joint_angles, title="Arm Configuration - 3D View")
```

#### Trajectory Visualization

```python
# Generate trajectory (100 timesteps, 7 joints)
trajectory = np.random.randn(100, 7) * 0.5

# Plot end-effector trajectory
fig = viz.plot_trajectory(trajectory)

# Plot all joint angles over time
fig = viz.plot_joint_angles(trajectory)
```

#### Animation

```python
# Create animation (saves to file)
anim = viz.animate_trajectory(
    trajectory,
    output_file="arm_motion.gif",
    fps=10
)
```

#### Forward Kinematics

```python
# Get end-effector position from joint angles
positions = viz.forward_kinematics(joint_angles)
# positions shape: (8, 3) - 7 joint positions + 1 base
end_effector_pos = positions[-1]  # Last position
```

### 2. **SimulationVisualizer** - Parallel Simulation Analysis

Monitor and analyze results from parallel simulations.

#### Plot Rewards

```python
from rl_armMotion.utils import SimulationVisualizer

# Rewards from 4 parallel simulations
rewards_list = [
    np.random.randn(200),
    np.random.randn(200),
    np.random.randn(200),
    np.random.randn(200),
]

# Plot cumulative rewards
fig = SimulationVisualizer.plot_rewards(
    rewards_list,
    labels=["Env 1", "Env 2", "Env 3", "Env 4"],
    title="Parallel Simulation Rewards"
)
plt.show()
```

#### Parallel Statistics

```python
# Simulation results
results = [
    {
        "env_id": 0,
        "episode_reward": 245.5,
        "episode_length": 200,
    },
    {
        "env_id": 1,
        "episode_reward": 198.3,
        "episode_length": 180,
    },
    # ... more results
]

# Generate statistics dashboard
fig = SimulationVisualizer.plot_parallel_stats(results)
plt.show()
```

#### Interactive Dashboard (Plotly)

```python
# Create interactive dashboard
fig = SimulationVisualizer.create_interactive_dashboard(
    results,
    title="Simulation Dashboard"
)

# Save or display
fig.write_html("dashboard.html")
fig.show()
```

### 3. **Trajectory Comparison**

Compare multiple trajectories side-by-side.

```python
from rl_armMotion.utils import plot_arm_trajectory_comparison

# Multiple trajectories
trajectories = [
    trajectory1,  # (100, 7)
    trajectory2,  # (100, 7)
    trajectory3,  # (100, 7)
]

# Compare
fig = plot_arm_trajectory_comparison(
    trajectories,
    labels=["Policy A", "Policy B", "Policy C"]
)
plt.show()
```

---

## Complete Examples

### Example 1: Visualize SimpleArmEnv Episode

```python
import numpy as np
import matplotlib.pyplot as plt
from rl_armMotion.utils import ArmVisualizer
from rl_armMotion.environments import SimpleArmEnv

# Run episode
env = SimpleArmEnv()
obs, _ = env.reset(seed=42)
trajectory = []

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    trajectory.append(obs[:7])  # Extract joint angles

trajectory = np.array(trajectory)
env.close()

# Visualize
viz = ArmVisualizer()
viz.plot_trajectory(np.array(trajectory))
viz.plot_joint_angles(np.array(trajectory))
plt.show()
```

### Example 2: Monitor Parallel Simulations

```python
import numpy as np
from rl_armMotion.utils import (
    VectorEnvironment,
    SimulationVisualizer,
)

# Create parallel environments
vec_env = VectorEnvironment(["CartPole-v1"] * 4, seed=42)

# Collect experience
rewards_per_env = [[] for _ in range(4)]

for step in range(200):
    actions = np.array([vec_env.envs[i].action_space.sample()
                       for i in range(4)])
    obs, rewards, terms, truncs, infos = vec_env.step(actions)

    for i, r in enumerate(rewards):
        rewards_per_env[i].append(r)

vec_env.close()

# Analyze
fig = SimulationVisualizer.plot_rewards(
    rewards_per_env,
    labels=[f"Env {i}" for i in range(4)]
)
plt.show()
```

### Example 3: Create Interactive Dashboard

```python
import numpy as np
from rl_armMotion.utils import (
    ParallelEnvironmentRunner,
    SimulationVisualizer,
)

# Run parallel simulations
with ParallelEnvironmentRunner(num_envs=8) as runner:
    results = runner.run_simulations(
        env_name="CartPole-v1",
        num_steps=500,
        seed=42,
    )

# Create dashboard data
results_data = [
    {
        "env_id": r.env_id,
        "episode_reward": r.episode_reward,
        "episode_length": r.episode_length,
    }
    for r in results
]

# Generate interactive dashboard
fig = SimulationVisualizer.create_interactive_dashboard(results_data)
fig.write_html("parallel_results.html")
print("✓ Dashboard saved to parallel_results.html")
```

---

## Visualization Classes Reference

### ArmVisualizer

```python
class ArmVisualizer:
    def __init__(self, link_lengths=None, dof=7)
    def forward_kinematics(joint_angles) -> np.ndarray
    def plot_pose_2d(joint_angles, ax=None, title="", show_joints=True) -> plt.Axes
    def plot_pose_3d(joint_angles, ax=None, title="", show_joints=True) -> Axes3D
    def plot_trajectory(joint_trajectories, title="", figsize=(12,5)) -> plt.Figure
    def animate_trajectory(joint_trajectories, output_file=None, fps=10) -> FuncAnimation
    def plot_joint_angles(joint_trajectories, figsize=(14,8)) -> plt.Figure
```

### SimulationVisualizer

```python
class SimulationVisualizer:
    @staticmethod
    def plot_rewards(rewards_list, labels=None, figsize=(12,6), title="") -> plt.Figure

    @staticmethod
    def plot_parallel_stats(simulation_results, figsize=(15,10)) -> plt.Figure

    @staticmethod
    def create_interactive_dashboard(simulation_results, title="") -> go.Figure
```

---

## Output Formats

### Static Visualization
- PNG images for papers and reports
- PDF export for presentations
- High-resolution output (150 DPI default)

### Interactive Visualization
- HTML dashboards with Plotly
- Zoomable, pannable, rotatable 3D plots
- Hover information and statistics

### Animation
- GIF format for quick playback
- MP4 format for sharing
- Frame-by-frame control

---

## Customization

### Change Link Lengths

```python
link_lengths = np.array([1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.4])
viz = ArmVisualizer(link_lengths=link_lengths, dof=7)
```

### Custom Colors and Styling

```python
fig, ax = plt.subplots()
viz.plot_pose_2d(joint_angles, ax=ax)
ax.set_facecolor("lightgray")
ax.grid(True, alpha=0.5, linestyle="--")
```

### Save to Different Formats

```python
# PNG (default)
plt.savefig("arm.png", dpi=150, bbox_inches="tight")

# PDF
plt.savefig("arm.pdf", bbox_inches="tight")

# SVG (vector)
plt.savefig("arm.svg", bbox_inches="tight")
```

---

## Test and Demo Scripts

### Quick Test
```bash
python3 project_assets/test_runs/test_visualization_standalone.py
```

This generates:
- `project_assets/test_images/test_arm_pose_visualization.png` - 2D/3D arm poses
- `project_assets/test_images/test_arm_trajectory.png` - End-effector trajectory
- `project_assets/test_images/test_arm_joint_angles.png` - All joint angles over time
- `project_assets/test_images/test_simple_arm_episode.png` - Episode trajectory
- `project_assets/test_images/test_simple_arm_episode_rewards.png` - Episode rewards
- `project_assets/test_images/test_parallel_rewards.png` - Parallel simulation rewards
- `project_assets/test_images/test_parallel_statistics.png` - Statistics dashboard
- `project_assets/test_runs/test_parallel_interactive_dashboard.html` - Interactive dashboard
- `project_assets/test_images/test_trajectory_comparison.png` - Trajectory comparison

### Run Unit Tests
```bash
pytest project_assets/tests/test_visualization.py -v
```

---

## Tips & Best Practices

### For Publication-Quality Figures
```python
# High DPI, no borders
plt.savefig("figure.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

# Use matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")
```

### For Large Datasets
```python
# Sample trajectory for cleaner plots
trajectory_sampled = trajectory[::10]  # Every 10th frame
fig = viz.plot_trajectory(trajectory_sampled)
```

### For Real-Time Monitoring
```python
# Use interactive backend
%matplotlib widget  # In Jupyter

# Or use Plotly for better responsiveness
fig = SimulationVisualizer.create_interactive_dashboard(results)
fig.show()
```

### Memory Optimization
```python
# Close figures after saving
plt.close("all")

# For animations, use reasonable frame count
anim = viz.animate_trajectory(trajectory, fps=5)  # 5 FPS instead of 10
```

---

## Dependencies

**Included in requirements.txt:**
- `matplotlib>=3.8.0` - Static plots and 3D visualization
- `seaborn>=0.13.0` - Statistical visualizations
- `plotly>=5.17.0` - Interactive dashboards
- `pillow>=10.0.0` - Image processing
- `kaleido>=0.2.1` - Export to static formats
- `opencv-python>=4.8.0` - Video processing (optional)

Install visualization extras:
```bash
pip install -e ".[visualization]"
```

---

## Troubleshooting

**Issue**: "No module named plotly"
```bash
pip install plotly kaleido
```

**Issue**: Plots not showing in Jupyter
```python
%matplotlib inline
# or
%matplotlib notebook
```

**Issue**: Animation playback is slow
```python
# Reduce FPS or sample trajectory
anim = viz.animate_trajectory(trajectory[::5], fps=5)
```

**Issue**: 3D plots looking flat
```python
# Adjust viewing angle
ax.view_init(elev=20, azim=45)
```

---

## Architecture

```
visualization.py
├── ArmVisualizer
│   ├── forward_kinematics()      # Compute end-effector position
│   ├── plot_pose_2d()            # 2D arm configuration
│   ├── plot_pose_3d()            # 3D arm configuration
│   ├── plot_trajectory()         # End-effector path
│   ├── animate_trajectory()      # Motion animation
│   └── plot_joint_angles()       # Joint angle timeseries
│
├── SimulationVisualizer
│   ├── plot_rewards()            # Reward curves
│   ├── plot_parallel_stats()     # Statistics dashboard
│   └── create_interactive_dashboard()  # Plotly dashboard
│
└── Utility Functions
    └── plot_arm_trajectory_comparison()  # Multi-trajectory plots
```

---

## Next Steps

1. **Integrate with Your RL Training**
   - log trajectories during training
   - Create dashboards for hyperparameter studies

2. **Create Custom Visualizations**
   - Extend ArmVisualizer for your arm kinematics
   - Add reward shaping visualizations

3. **Generate Reports**
   - Save high-resolution figures for papers
   - Export interactive HTML dashboards

4. **Monitor Performance**
   - Real-time dashboard during training
   - Comparative analysis across methods

---

**Last Updated**: 2026-02-23
**Status**: ✅ Ready for use
**Tests**: All 11 visualization tests passing
