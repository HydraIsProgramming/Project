# Reference Papers — RL Arm Motion Physics Constraints

All PDFs in this folder are the academic sources cited directly in the physics
constraint comments inside `src/rl_armMotion/two_d/environments/task_env.py`
(branch `claude/trusting-chaum`).

---

## A — Gravitational Potential Energy (ΔPE gravity penalty)

| File | Citation |
|------|----------|
| *(textbook — not downloaded; widely available)* | Goldstein, Poole & Safko, **Classical Mechanics** (3rd ed.), §1.4 — gravitational PE of a system of particles. Addison-Wesley, 2002. |

---

## C — Joint Acceleration Limit (8.0 rad/s²)

| File | Citation |
|------|----------|
| `UR5_Technical_Spec_Sheet.pdf` | Universal Robots, **UR5 Technical Specification** (Item 110105). Establishes ~5.24 rad/s² (300 deg/s²) as the physical limit for a 5 kg-payload industrial arm. |
| `ETA-IK_2024_arXiv_2411.14381_KUKA_iiwa_acceleration.pdf` | Fraunhofer IWU (2024). *ETA-IK: Efficient Trajectory Approximation using Inverse Kinematics for the KUKA LBR iiwa.* arXiv:2411.14381. Reports 2.0–5.0 rad/s² joint acceleration limits for the KUKA LBR iiwa 14 R820. |

---

## J — Mechanical Energy Budget (|τ·ω|·dt)

| File | Citation |
|------|----------|
| `Petrichenko_2024_arXiv_2411.03194_energy_modeling_robotics.pdf` | Petrichenko et al. (2024). *Energy Consumption in Robotics: A Simplified Modeling Approach.* arXiv:2411.03194. Validates P = τᵀ·q̇ to within 3.5–4% of measured electrical power on a Franka Emika Panda robot. |
| `Peri_2025_arXiv_2509.01765_non_regenerative_energy_RL.pdf` | Peri et al. (2025). *Non-conflicting Energy Minimization in RL-based Robot Control.* arXiv:2509.01765. Explicitly justifies using \|τ·ω\| (absolute value) for non-regenerative DC servo actuators that dissipate braking energy as heat. |
| `Zhang_2023_Sensors_23_5974_energy_reward_RL.pdf` | Zhang, Xia, Chen & Cheng (2023). *Multi-Objective Optimal Trajectory Planning for Robotic Arms Using Deep Reinforcement Learning.* **Sensors** 23(13):5974. DOI: 10.3390/s23135974. Uses a discrete ∫τ·ω·dt approximation as an energy reward term in RL training. |

---

## K — Jerk Penalty (minimum-jerk criterion)

| File | Citation |
|------|----------|
| `Flash_Hogan_1985_JNeurosci_minimum_jerk_arm_motion.pdf` | Flash & Hogan (1985). *The coordination of arm movements: an experimentally confirmed mathematical model.* **Journal of Neuroscience** 5(7):1688-1703. Proves that human arm trajectories minimise ∫(d³x/dt³)² dt (integral of squared jerk), producing straight-line paths with bell-shaped velocity profiles. |
| `Kim_2024_arXiv_2308.12517_jerk_RL_deployment.pdf` | Kim et al. (2024). *Jerk-Aware Reward Shaping for Deployment of RL Policies on Real Robots.* arXiv:2308.12517. Demonstrates that penalising jerk during RL training significantly improves sim-to-real transfer and reduces actuator wear. |

---

## Note on ISO/TS 15066:2016

ISO/TS 15066:2016 (*Robots and robotic devices — Collaborative robots*) governs
contact force and pressure limits for collaborative robot operation. It is
referenced in the code as the safety-standard context for our 8.0 rad/s²
acceleration limit. The standard is not freely redistributable; purchase it
from [ISO.org](https://www.iso.org/standard/62996.html).

---

*Downloaded: 2026-03-21 | Branch: `claude/trusting-chaum`*
