"""
An environment implementing the key elements of the natural walking controller
proposed by Weng et al. (2021) for musculoskeletal gait.  The goal of this
environment is to provide a flexible sandbox where reinforcement learning
algorithms can learn to stabilise, step and walk by controlling a simplified
arm–torso model.  Although the original paper focuses on full‑body walking
with complex musculoskeletal dynamics, this implementation captures the core
ideas in a lightweight format that integrates with the existing RL
infrastructure of the project.

The state includes joint angles and velocities, pelvis (root) position and
velocity, relative segment positions, optional ground contact flags and the
centre‑of‑mass velocity.  Actions can be interpreted either as joint torques
or as muscle excitation signals; in both cases the environment applies
saturation and simple dynamics to update the underlying controller state.
Curriculum learning stages and penalty ramps are exposed through constructor
arguments so that training scripts can schedule progressive changes in the
reward and penalty structure.  A small collection of metrics are exposed via
the ``info`` dictionary returned at each step to aid downstream logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl_armMotion.three_d.config import ArmConfiguration3D
from rl_armMotion.three_d.utils.kinematics_3d import ArmKinematics3D
from rl_armMotion.three_d.utils.kinematics_3d import ArmController3D


@dataclass
class PenaltySchedule:
    """Container controlling penalty weighting for different reward terms.

    The schedule defines baseline weights and a multiplier that can be
    increased during the course of training to gradually emphasise effort,
    smoothness and stability penalties.  See `update_weights` for usage.
    """

    forward: float = 1.0
    effort: float = 1e-3
    smoothness: float = 1e-3
    joint_limit: float = 5e-3
    stability: float = 1e-1
    success: float = 100.0
    fall: float = -100.0
    multiplier: float = 1.0

    def update(self, progress: float) -> None:
        """Linearly scale penalties towards a higher multiplier.

        Args:
            progress: A float in [0, 1] indicating the fraction of training
                completed.  At progress=0 the multiplier is unchanged; at
                progress=1 the multiplier is doubled.  The forward weight is
                not scaled as it drives the task objective.
        """
        # Clamp progress for safety
        p = float(np.clip(progress, 0.0, 1.0))
        # We linearly increase the multiplier from 1 to 2 over training.
        self.multiplier = 1.0 + p

    def scaled_effort(self) -> float:
        return self.effort * self.multiplier

    def scaled_smoothness(self) -> float:
        return self.smoothness * self.multiplier

    def scaled_joint_limit(self) -> float:
        return self.joint_limit * self.multiplier

    def scaled_stability(self) -> float:
        return self.stability * self.multiplier


class WengGaitEnv(gym.Env):
    """Simplified gait environment inspired by Weng et al. (2021).

    This environment wraps a 3D arm–torso model from the existing project to
    expose a state, action and reward structure suitable for natural walking
    experiments.  It emphasises forward progress, energy efficiency,
    movement smoothness, joint safety and upright stability.  Three
    curriculum stages are supported:

    1. **Stabilise:** The agent learns to maintain an upright posture.
    2. **Step:**  The agent learns to take steps without falling.
    3. **Walk:**  The agent learns to progress forward toward a target.

    Parameters
    ----------
    config: ArmConfiguration3D, optional
        Configuration describing joint limits, link lengths and other
        physical parameters of the underlying arm model.  A default 4‑DOF
        configuration is used when none is provided.
    use_muscles: bool, default False
        When ``True``, actions are interpreted as muscle excitations in
        [0, 1].  Otherwise actions are interpreted as normalised joint
        torques in [−1, 1].
    include_grf: bool, default False
        When ``True``, three ground reaction force contact flags are
        appended to the observation.  Contact forces are not simulated in
        this simplified environment so they are currently always zero.
    curriculum_stage: int, default 1
        Initial curriculum stage.  Valid values are 1 (stabilise), 2
        (step) and 3 (walk).  Changing the stage adjusts the reward
        composition as described above.
    penalty_schedule: PenaltySchedule, optional
        Object controlling the weighting of different penalty terms.  If
        ``None`` a default schedule is constructed.
    domain_randomization: bool, default False
        When ``True`` the environment will randomise physical parameters such
        as link masses and joint damping at every reset for robustness.
    noise_std: float, default 0.0
        Standard deviation of Gaussian noise added independently to each
        observation element.  Useful for domain randomisation.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        config: Optional[ArmConfiguration3D] = None,
        *,
        use_muscles: bool = False,
        include_grf: bool = False,
        curriculum_stage: int = 1,
        penalty_schedule: Optional[PenaltySchedule] = None,
        domain_randomization: bool = False,
        noise_std: float = 0.0,
        max_episode_steps: int = 1000,
    ) -> None:
        super().__init__()
        # Underlying physical configuration and controller
        self.config: ArmConfiguration3D = config or ArmConfiguration3D.get_default()
        self.controller: ArmController3D = ArmController3D(self.config)
        self.num_joints: int = int(self.config.dof)

        # Domain randomisation options
        self.domain_randomization = bool(domain_randomization)
        self.noise_std: float = float(noise_std)

        # Curriculum and penalty scheduling
        if curriculum_stage not in (1, 2, 3):
            raise ValueError(
                f"curriculum_stage must be 1, 2 or 3; got {curriculum_stage}"
            )
        self.curriculum_stage: int = curriculum_stage
        self.penalty_schedule: PenaltySchedule = penalty_schedule or PenaltySchedule()

        # Action interpretation
        self.use_muscles: bool = bool(use_muscles)
        # Each joint has a torque limit; here we pick a nominal limit based on
        # velocity limits and inertia.  This is a simplification: in a full
        # musculoskeletal model torque limits depend on muscle strength.
        vel_limit = float(np.deg2rad(self.config.velocity_limits_deg_per_s))
        # Use inertia and dt to compute a rough torque limit.  The inertia
        # values are per joint; we broadcast a single nominal value.
        nominal_inertia = 0.1
        # torque_limit = inertia * (angular_acceleration) ~ inertia * vel_limit / dt
        self.torque_limits: np.ndarray = nominal_inertia * vel_limit / float(self.config.dt) * np.ones(self.num_joints, dtype=float)
        # For muscle actions we treat excitations as fractional torque; the same
        # torque limits are reused internally.

        # Store previous action for smoothness penalty and muscle dynamics
        self.prev_action: np.ndarray = np.zeros(self.num_joints, dtype=float)
        self.muscle_activation: np.ndarray = np.zeros(self.num_joints, dtype=float)

        # Episode control
        self.max_episode_steps: int = int(max_episode_steps)
        self.step_count: int = 0
        self._done: bool = False

        # Root (pelvis) state: we model the pelvis as moving along the
        # horizontal X–Z plane at a fixed height.  The y‑coordinate is the
        # vertical axis in our coordinate system.  Root orientation is not
        # explicitly represented; stability penalties derive from the arm
        # orientation.
        self.root_pos: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=float)
        self.root_vel: np.ndarray = np.zeros(3, dtype=float)

        # Contact flags (currently always false in this simplified model)
        self.include_grf: bool = bool(include_grf)

        # Metrics tracking
        self._success_counter: int = 0
        self._hold_required: int = 50  # steps required to consider success
        self.goal_tolerance: float = 0.05  # tolerance for end‑effector to target
        # Target position for walking: 1 m ahead along X from shoulder base
        shoulder_base = np.asarray(self.config.shoulder_position, dtype=float)
        self.target_position: np.ndarray = shoulder_base + np.array([1.0, 0.0, 0.0], dtype=float)

        # Build observation and action spaces
        # Observation: angles (rad) normalised by pi, velocities normalised by vel_limit,
        # root pos (3), root vel (3), relative positions of elbow, wrist and hand
        # (3 segments ×3 coords), optional ground contact flags (3) and COM velocity (3).
        # We treat the shoulder as the first segment relative to the root; the next
        # two segments correspond to elbow and end effector.
        n_segments = 3  # elbow, wrist (forearm end), hand (end effector)
        obs_dim = 2 * self.num_joints + 3 + 3 + n_segments * 3 + (3 if self.include_grf else 0) + 3
        # Lower and upper bounds are loose; normalisation is applied by the agent.
        obs_low = -np.ones(obs_dim, dtype=float) * 10.0
        obs_high = np.ones(obs_dim, dtype=float) * 10.0
        self.observation_space: spaces.Box = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        if self.use_muscles:
            # Muscle excitations in [0, 1]
            self.action_space: spaces.Box = spaces.Box(low=0.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        else:
            # Torques normalised to [−1, 1]; they are scaled internally by torque_limits
            self.action_space: spaces.Box = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Environment API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment state.

        Randomises domain parameters when requested and reinitialises the
        controller and root state.  The returned observation is the first
        state of a new episode with optional noise.
        """
        super().reset(seed=seed)
        self.step_count = 0
        self._done = False
        self._success_counter = 0
        # Reset controller angles and velocities to initial state
        self.controller.set_home_position()
        # Reset root state; small random shift along X for domain randomisation
        self.root_pos = np.array([0.0, 1.0, 0.0], dtype=float)
        self.root_vel = np.zeros(3, dtype=float)
        self.prev_action = np.zeros(self.num_joints, dtype=float)
        self.muscle_activation = np.zeros(self.num_joints, dtype=float)
        # Domain randomisation of physical parameters
        if self.domain_randomization:
            self._randomise_physics()
        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply an action and advance the simulation by one step.

        The action is clipped to the action space bounds and interpreted as
        either torque commands or muscle excitations.  A simple Euler–Euler
        integration scheme is used to update joint angles and root motion.
        Reward composition depends on the current curriculum stage and the
        penalty schedule.  Additional diagnostics are returned via the info
        dictionary.
        """
        if self._done:
            # Gymnasium API requires that calling step() on a terminated episode
            # raises an exception; however we choose to simply reset instead for
            # user convenience.  This mirrors the behaviour of many SB3
            # environments.
            obs, _ = self.reset()
            return obs, 0.0, True, True, {}

        # Clip and copy action
        act = np.asarray(action, dtype=float).copy()
        act = np.clip(act, self.action_space.low, self.action_space.high)

        # Interpret action
        if self.use_muscles:
            # Update muscle activation with a simple first‑order filter
            # (activation dynamics): a_t = 0.9 a_{t-1} + 0.1 act
            self.muscle_activation = 0.9 * self.muscle_activation + 0.1 * act
            # Convert activation to torque in [0, torque_limit]
            torque = self.muscle_activation * self.torque_limits
        else:
            # Convert normalised torques to physical torque range
            torque = act * self.torque_limits

        # Update joints: we approximate angular acceleration proportional to
        # applied torque and integrate once to update angles.  Velocities are
        # updated implicitly via the controller API; this is a simplification
        # relative to a full dynamics model.
        delta_angles = torque * float(self.config.dt) / (0.5 + 0.0)  # divide by pseudo‑inertia
        for j in range(self.num_joints):
            # Use ArmController3D to apply incremental angle changes.  It will
            # clamp to joint limits and update internal velocities.
            self.controller.increment_joint(j, float(delta_angles[j]))

        # Update root position: approximate forward progress based on elbow
        # extension.  As the agent swings the elbow forward (positive torque on
        # the last joint), we move the root slightly forward.  This is a very
        # coarse approximation but encourages agents to learn cyclical motion.
        forward_motion = float(np.sum(torque)) * 0.01 * float(self.config.dt)
        self.root_vel = np.array([forward_motion / float(self.config.dt), 0.0, 0.0], dtype=float)
        self.root_pos += self.root_vel * float(self.config.dt)

        # Compute reward and info before checking termination so that fall
        # penalties are applied appropriately
        reward, info = self._compute_reward(act)

        # Increment step counter and check termination
        self.step_count += 1
        terminated = False
        truncated = False

        # Terminate if pelvis height below threshold or torso tilt too large
        if self.root_pos[1] < 0.5 or self._is_tilt_unstable():
            terminated = True
            reward += self.penalty_schedule.fall

        # Terminate if any joint limit is violated beyond clamp (the controller
        # clamps joint angles, so here we penalise near‑limit angles instead)
        angles = self.controller.angles
        mins, maxs = self.config.get_joint_limits_rad()
        # if any angle equals limit (with small epsilon), treat as unsafe
        if np.any(np.isclose(angles, mins, atol=1e-3)) or np.any(np.isclose(angles, maxs, atol=1e-3)):
            terminated = True

        # Truncate episode if maximum number of steps reached
        if self.step_count >= self.max_episode_steps:
            truncated = True

        # Mark done flag
        self._done = terminated or truncated

        # Build observation with optional noise
        obs = self._get_observation()

        return obs, float(reward), bool(self._done), bool(truncated), info

    # ------------------------------------------------------------------
    # Internal helpers
    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector with optional Gaussian noise."""
        angles = self.controller.angles.copy()
        velocities = self.controller.velocities.copy()
        # Normalise angles to [−1, 1] by dividing by π
        angle_norm = angles / np.pi
        # Normalise velocities by the maximum velocity limit
        vel_limit = float(np.deg2rad(self.config.velocity_limits_deg_per_s))
        vel_norm = velocities / vel_limit

        # Root position and velocity
        root_pos = self.root_pos.copy()
        root_vel = self.root_vel.copy()

        # Segment positions relative to pelvis: elbow, wrist and hand.  The
        # forward_kinematics function returns a stack of four points:
        # [0] world origin (unused), [1] shoulder, [2] elbow, [3] end effector.
        pts = ArmKinematics3D.forward_kinematics(angles, self.config)
        # Compute positions relative to pelvis (root).  We treat the shoulder
        # position as being offset from the pelvis by ``self.config.shoulder_position``.
        pelvis_to_shoulder = np.asarray(self.config.shoulder_position, dtype=float)
        elbow_rel = pts[2] + pelvis_to_shoulder - root_pos
        wrist_rel = (pts[3] + pelvis_to_shoulder - root_pos)  # end effector is treated as wrist
        # In a 4‑DOF arm the end effector corresponds to the hand; there is no
        # explicit wrist joint.  We duplicate the end effector position for
        # completeness.
        hand_rel = pts[3] + pelvis_to_shoulder - root_pos
        segment_rel = np.concatenate([elbow_rel, wrist_rel, hand_rel])

        # Ground reaction force/contact flags (always zero in this model)
        grf = np.zeros(3, dtype=float) if self.include_grf else np.array([], dtype=float)

        # Centre of mass velocity approximated by pelvis velocity.  In a full
        # model this would be computed from segment masses and velocities.
        com_vel = self.root_vel.copy()

        obs = np.concatenate([
            angle_norm,
            vel_norm,
            root_pos,
            root_vel,
            segment_rel,
            grf,
            com_vel,
        ])

        # Add Gaussian noise if requested
        if self.noise_std > 0.0:
            obs = obs + np.random.normal(scale=self.noise_std, size=obs.shape)

        return obs.astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compose the reward from forward progress and penalty terms."""
        # Forward reward: depends on curriculum stage
        # We measure forward progress along the X axis of the root
        progress = float(self.root_vel[0]) * float(self.config.dt)
        # Distance error to target (end effector vs. target position)
        ee_pos = ArmKinematics3D.forward_kinematics(self.controller.angles, self.config)[-1] + np.asarray(self.config.shoulder_position, dtype=float)
        dist_to_target = float(np.linalg.norm(ee_pos - self.target_position))
        # The forward reward encourages reduction of distance to target or forward motion
        if self.curriculum_stage <= 1:
            # In stabilisation stage there is no forward progress term
            forward_reward = -dist_to_target
        elif self.curriculum_stage == 2:
            # Stepping stage encourages small forward motions
            forward_reward = progress - dist_to_target
        else:
            # Walking stage strongly encourages forward progress
            forward_reward = 2.0 * progress - dist_to_target

        # Effort penalty: penalise squared torque/activation
        eff_penalty = float(np.sum(np.square(action)))
        # Smoothness penalty: penalise changes in action relative to previous step
        smoothness_penalty = float(np.sum(np.square(action - self.prev_action)))
        # Update previous action for next call
        self.prev_action = action.copy()
        # Joint limit penalty: penalise proximity to joint limits
        angles = self.controller.angles.copy()
        mins, maxs = self.config.get_joint_limits_rad()
        # Compute fractional distance to limits: 0 at centre, 1 at limit
        frac = (angles - mins) / (maxs - mins + 1e-8)
        # Penalise near 0 or 1 (close to limits)
        joint_penalty = float(np.sum(np.exp(-4.0 * np.minimum(frac, 1 - frac))))
        # Stability penalty: measure tilt as deviation of the arm's vertical
        # orientation from the upright orientation.  We approximate tilt by
        # computing the end effector height relative to elbow and penalising
        # deviation from a downward pointing forearm.
        pts = ArmKinematics3D.forward_kinematics(angles, self.config)
        shoulder = pts[1]
        elbow = pts[2]
        hand = pts[3]
        forearm_vec = hand - elbow
        # Unit vector along global −Y (downward)
        desired_dir = np.array([0.0, -1.0, 0.0], dtype=float)
        unit = forearm_vec / (np.linalg.norm(forearm_vec) + 1e-8)
        tilt_err = float(np.arccos(np.clip(np.dot(unit, desired_dir), -1.0, 1.0)))
        stability_penalty = tilt_err

        # Accumulate reward components with penalty schedule
        ps = self.penalty_schedule
        reward = ps.forward * forward_reward
        reward -= ps.scaled_effort() * eff_penalty
        reward -= ps.scaled_smoothness() * smoothness_penalty
        reward -= ps.scaled_joint_limit() * joint_penalty
        reward -= ps.scaled_stability() * stability_penalty

        # Success reward: if the end effector remains within tolerance of
        # the target for a required number of consecutive steps
        if dist_to_target < self.goal_tolerance:
            self._success_counter += 1
            if self._success_counter >= self._hold_required:
                reward += ps.success
                self._done = True
        else:
            self._success_counter = 0

        info: Dict[str, float] = {
            "forward_reward": forward_reward,
            "effort_penalty": eff_penalty,
            "smoothness_penalty": smoothness_penalty,
            "joint_limit_penalty": joint_penalty,
            "stability_penalty": stability_penalty,
            "distance_to_target": dist_to_target,
            "progress": progress,
            "success_counter": float(self._success_counter),
        }
        return reward, info

    def _is_tilt_unstable(self) -> bool:
        """Return True if the arm orientation implies an unstable tilt."""
        angles = self.controller.angles.copy()
        # Compute orientation error as in stability penalty
        pts = ArmKinematics3D.forward_kinematics(angles, self.config)
        elbow = pts[2]
        hand = pts[3]
        forearm_vec = hand - elbow
        unit = forearm_vec / (np.linalg.norm(forearm_vec) + 1e-8)
        desired_dir = np.array([0.0, -1.0, 0.0], dtype=float)
        tilt_err = float(np.arccos(np.clip(np.dot(unit, desired_dir), -1.0, 1.0)))
        # Consider tilt unstable if deviation exceeds ~60 degrees
        return bool(tilt_err > np.deg2rad(60.0))

    # ------------------------------------------------------------------
    # Domain randomisation utilities
    def _randomise_physics(self) -> None:
        """Randomise link masses, joint damping and torque limits for robustness."""
        # Randomly scale link masses by ±10 %
        scale = 1.0 + 0.2 * (np.random.rand(len(self.config.masses)) - 0.5)
        self.config.masses = (np.asarray(self.config.masses) * scale).tolist()
        # Randomly scale joint damping by ±10 %
        self.config.damping *= float(1.0 + 0.2 * (np.random.rand() - 0.5))
        # Randomly scale torque limits by ±10 %
        self.torque_limits *= (1.0 + 0.2 * (np.random.rand(self.num_joints) - 0.5))

    # ------------------------------------------------------------------
    # Curriculum utilities
    def set_curriculum_stage(self, stage: int) -> None:
        """Set the curriculum stage and adjust internal parameters accordingly.

        Changing the curriculum stage adjusts how the reward is composed and
        modifies the success hold duration and positional tolerance.  Use
        this method to manually advance through the stabilise → step →
        walk curriculum.  Stage must be in {1, 2, 3}.

        Args
        ----
        stage: int
            The new curriculum stage (1: stabilise, 2: step, 3: walk).

        Raises
        ------
        ValueError
            If ``stage`` is not one of 1, 2 or 3.
        """
        if stage not in (1, 2, 3):
            raise ValueError(f"Invalid curriculum stage {stage}; must be 1, 2 or 3")
        self.curriculum_stage = stage
        # Adjust hold duration and goal tolerance for different stages.
        # In earlier stages we require longer holds and looser tolerances;
        # as the agent progresses the task becomes stricter.
        if stage == 1:
            self._hold_required = 50
            self.goal_tolerance = 0.05
        elif stage == 2:
            self._hold_required = 40
            self.goal_tolerance = 0.05
        else:  # stage == 3
            self._hold_required = 30
            # Reduce tolerance slightly to encourage precise control in walk stage
            self.goal_tolerance = 0.03

