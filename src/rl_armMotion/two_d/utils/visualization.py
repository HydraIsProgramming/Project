"""Visualization utilities for robotic arm models and simulations

Provides 3D visualization, trajectory plotting, and parallel simulation monitoring.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Callable, Dict, Any
import io
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ArmVisualizer:
    """Visualizer for robotic arm configurations and trajectories"""

    def __init__(
        self,
        link_lengths: Optional[np.ndarray] = None,
        dof: int = 7,
    ):
        """
        Initialize arm visualizer.

        Args:
            link_lengths: Length of each link (if None, uses unit lengths)
            dof: Degrees of freedom
        """
        self.dof = dof
        self.link_lengths = link_lengths if link_lengths is not None else np.ones(dof)
        self.trajectories = []
        self.current_pose = np.zeros(dof)

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute end-effector position from joint angles.

        Simplified 2D serial chain kinematics (can be extended for 3D).

        Args:
            joint_angles: Joint angles in radians

        Returns:
            (x, y, z) position of end-effector
        """
        # Initialize positions with base at origin.
        positions = np.zeros((len(joint_angles) + 1, 3))

        # Compute serial-chain FK incrementally so each link keeps rigid length.
        x = 0.0
        y = 0.0
        cumulative_angle = 0.0
        for i, angle in enumerate(joint_angles):
            cumulative_angle += float(angle)
            x += float(self.link_lengths[i]) * np.cos(cumulative_angle)
            y += float(self.link_lengths[i]) * np.sin(cumulative_angle)
            positions[i + 1] = [x, y, 0.0]

        return positions

    def plot_pose_2d(
        self,
        joint_angles: np.ndarray,
        ax: Optional[plt.Axes] = None,
        title: str = "2D Arm Configuration",
        show_joints: bool = True,
    ) -> plt.Axes:
        """
        Plot 2D arm configuration.

        Args:
            joint_angles: Joint angles
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            show_joints: Whether to show joint markers

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        positions = self.forward_kinematics(joint_angles)

        # Plot links
        ax.plot(positions[:, 0], positions[:, 1], "b-o" if show_joints else "b-", linewidth=2)

        if show_joints:
            # Mark base
            ax.plot(positions[0, 0], positions[0, 1], "go", markersize=10, label="Base")
            # Mark end-effector
            ax.plot(positions[-1, 0], positions[-1, 1], "r*", markersize=20, label="End-effector")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend()

        return ax

    def plot_pose_3d(
        self,
        joint_angles: np.ndarray,
        ax: Optional[Axes3D] = None,
        title: str = "3D Arm Configuration",
        show_joints: bool = True,
    ) -> Axes3D:
        """
        Plot 3D arm configuration.

        Args:
            joint_angles: Joint angles
            ax: 3D axes (creates new if None)
            title: Plot title
            show_joints: Whether to show joint markers

        Returns:
            3D axes
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

        positions = self.forward_kinematics(joint_angles)

        # Plot links
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-o" if show_joints else "b-", linewidth=2)

        if show_joints:
            ax.scatter(*positions[0], color="green", s=100, label="Base")
            ax.scatter(*positions[-1], color="red", s=200, label="End-effector", marker="*")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.legend()

        return ax

    def plot_trajectory(
        self,
        joint_trajectories: np.ndarray,
        title: str = "End-Effector Trajectory",
        figsize: Tuple = (12, 5),
    ) -> plt.Figure:
        """
        Plot end-effector trajectory over time.

        Args:
            joint_trajectories: Array of shape (timesteps, dof) with joint angles
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Compute end-effector positions
        ee_positions = np.array([self.forward_kinematics(angles)[-1] for angles in joint_trajectories])

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 2D trajectory
        axes[0].plot(ee_positions[:, 0], ee_positions[:, 1], "b-", linewidth=2)
        axes[0].scatter(ee_positions[0, 0], ee_positions[0, 1], color="green", s=100, label="Start")
        axes[0].scatter(ee_positions[-1, 0], ee_positions[-1, 1], color="red", s=100, label="End")
        axes[0].set_xlabel("X (m)")
        axes[0].set_ylabel("Y (m)")
        axes[0].set_title("2D Trajectory")
        axes[0].grid(True, alpha=0.3)
        axes[0].axis("equal")
        axes[0].legend()

        # Plot 3D trajectory
        ax = fig.add_subplot(122, projection="3d")
        ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], "b-", linewidth=2)
        ax.scatter(*ee_positions[0], color="green", s=100, label="Start")
        ax.scatter(*ee_positions[-1], color="red", s=100, label="End")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Trajectory")
        ax.legend()

        fig.suptitle(title)
        return fig

    def animate_trajectory(
        self,
        joint_trajectories: np.ndarray,
        output_file: Optional[str] = None,
        fps: int = 10,
    ) -> Optional[FuncAnimation]:
        """
        Create animation of arm motion.

        Args:
            joint_trajectories: Array of shape (timesteps, dof)
            output_file: Save animation to file (MP4, GIF)
            fps: Frames per second

        Returns:
            FuncAnimation object (or None if saved to file)
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        def update(frame):
            ax.clear()
            angles = joint_trajectories[frame]
            self.plot_pose_2d(angles, ax=ax, title=f"Arm Motion - Frame {frame}")

        anim = FuncAnimation(
            fig,
            update,
            frames=len(joint_trajectories),
            interval=1000 / fps,
            repeat=True,
        )

        if output_file:
            anim.save(output_file, fps=fps)
            plt.close()
            return None

        return anim

    def plot_joint_angles(
        self,
        joint_trajectories: np.ndarray,
        figsize: Tuple = (14, 8),
    ) -> plt.Figure:
        """
        Plot all joint angles over time.

        Args:
            joint_trajectories: Array of shape (timesteps, dof)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(self.dof, 1, figsize=figsize)
        if self.dof == 1:
            axes = [axes]

        for i in range(self.dof):
            axes[i].plot(joint_trajectories[:, i], linewidth=1.5)
            axes[i].set_ylabel(f"Joint {i+1} (rad)")
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Timestep")
        fig.suptitle("Joint Angles Over Time")

        return fig


class SimulationVisualizer:
    """Visualizer for parallel simulation results"""

    @staticmethod
    def plot_rewards(
        rewards_list: List[np.ndarray],
        labels: Optional[List[str]] = None,
        figsize: Tuple = (12, 6),
        title: str = "Cumulative Rewards Over Episodes",
    ) -> plt.Figure:
        """
        Plot rewards from multiple simulations.

        Args:
            rewards_list: List of reward arrays from parallel simulations
            labels: Labels for each simulation
            figsize: Figure size
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        for i, rewards in enumerate(rewards_list):
            label = labels[i] if labels else f"Sim {i}"
            ax.plot(np.cumsum(rewards), label=label, linewidth=2)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig

    @staticmethod
    def plot_parallel_stats(
        simulation_results: List[Dict[str, Any]],
        figsize: Tuple = (15, 10),
    ) -> plt.Figure:
        """
        Plot statistics from parallel simulations.

        Args:
            simulation_results: List of result dicts with metrics
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        num_sims = len(simulation_results)
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Extract metrics
        rewards = [r.get("episode_reward", 0) for r in simulation_results]
        lengths = [r.get("episode_length", 0) for r in simulation_results]
        env_ids = [r.get("env_id", i) for i, r in enumerate(simulation_results)]

        # Plot 1: Rewards bar chart
        axes[0, 0].bar(range(num_sims), rewards, alpha=0.7, color="blue")
        axes[0, 0].set_xlabel("Simulation ID")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].set_title("Rewards by Simulation")
        axes[0, 0].grid(True, alpha=0.3, axis="y")

        # Plot 2: Episode lengths
        axes[0, 1].bar(range(num_sims), lengths, alpha=0.7, color="green")
        axes[0, 1].set_xlabel("Simulation ID")
        axes[0, 1].set_ylabel("Episode Length")
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        # Plot 3: Reward distribution
        axes[1, 0].hist(rewards, bins=10, alpha=0.7, color="blue", edgecolor="black")
        axes[1, 0].set_xlabel("Reward")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Reward Distribution")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        # Plot 4: Statistics table
        axes[1, 1].axis("off")
        stats_text = f"""
        Simulation Statistics (n={num_sims})

        Rewards:
            Mean: {np.mean(rewards):.2f}
            Std:  {np.std(rewards):.2f}
            Min:  {np.min(rewards):.2f}
            Max:  {np.max(rewards):.2f}

        Episode Lengths:
            Mean: {np.mean(lengths):.2f}
            Std:  {np.std(lengths):.2f}
            Min:  {int(np.min(lengths))}
            Max:  {int(np.max(lengths))}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family="monospace",
                       verticalalignment="center")

        fig.suptitle("Parallel Simulation Statistics")
        return fig

    @staticmethod
    def create_interactive_dashboard(
        simulation_results: List[Dict[str, Any]],
        title: str = "Parallel Simulations Dashboard",
    ) -> go.Figure:
        """
        Create interactive Plotly dashboard for simulation results.

        Args:
            simulation_results: List of result dicts
            title: Dashboard title

        Returns:
            Plotly figure
        """
        rewards = [r.get("episode_reward", 0) for r in simulation_results]
        lengths = [r.get("episode_length", 0) for r in simulation_results]
        env_ids = [str(r.get("env_id", i)) for i, r in enumerate(simulation_results)]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Total Rewards",
                "Episode Lengths",
                "Reward Distribution",
                "Statistics"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "table"}],
            ],
        )

        # Rewards bar
        fig.add_trace(
            go.Bar(x=env_ids, y=rewards, name="Rewards", marker_color="blue"),
            row=1, col=1
        )

        # Lengths bar
        fig.add_trace(
            go.Bar(x=env_ids, y=lengths, name="Lengths", marker_color="green"),
            row=1, col=2
        )

        # Reward histogram
        fig.add_trace(
            go.Histogram(x=rewards, name="Reward Distribution", nbinsx=10),
            row=2, col=1
        )

        # Statistics table
        stats = [
            ["Metric", "Value"],
            ["Mean Reward", f"{np.mean(rewards):.2f}"],
            ["Std Reward", f"{np.std(rewards):.2f}"],
            ["Mean Length", f"{np.mean(lengths):.2f}"],
            ["Total Simulations", f"{len(simulation_results)}"],
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=stats[0], fill_color="lightblue"),
                cells=dict(values=list(zip(*stats[1:])), fill_color="lightyellow")
            ),
            row=2, col=2
        )

        fig.update_layout(title_text=title, height=800, showlegend=False)
        return fig


def plot_arm_trajectory_comparison(
    trajectories: List[np.ndarray],
    labels: Optional[List[str]] = None,
    figsize: Tuple = (14, 6),
) -> plt.Figure:
    """
    Compare multiple arm trajectories.

    Args:
        trajectories: List of trajectory arrays
        labels: Labels for each trajectory
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    visualizer = ArmVisualizer()

    for i, trajectory in enumerate(trajectories):
        label = labels[i] if labels else f"Trajectory {i}"
        ee_positions = np.array([visualizer.forward_kinematics(angles)[-1]
                                for angles in trajectory])

        # 2D plot
        axes[0].plot(ee_positions[:, 0], ee_positions[:, 1], label=label, linewidth=2)

        # 3D plot
        if i == 0:
            ax3d = fig.add_subplot(122, projection="3d")

        ax3d.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                 label=label, linewidth=2)

    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title("2D Trajectories")
    axes[0].grid(True, alpha=0.3)
    axes[0].axis("equal")
    axes[0].legend()

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Trajectories")
    ax3d.legend()

    fig.suptitle("Trajectory Comparison")
    return fig


__all__ = [
    "ArmVisualizer",
    "SimulationVisualizer",
    "plot_arm_trajectory_comparison",
]
