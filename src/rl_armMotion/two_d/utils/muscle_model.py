"""Hill-type muscle dynamics for biomechanically informed actuation.

This module implements the canonical Hill-type muscle model used by Fischer,
Hoinville, Eickhoff, and Lilienthal (2021), "Reinforcement learning control
of a biomechanical model of the upper extremity", Scientific Reports
11:14445, https://doi.org/10.1038/s41598-021-93760-1, and by virtually all
contemporary musculoskeletal simulators (OpenSim, MuJoCo, MyoSuite,
Anybody, SCONE). The model dates back to A.V. Hill's classical 1938 paper,
"The heat of shortening and the dynamic constants of muscle", Proceedings
of the Royal Society of London. Series B, 126:136--195, and is the
de facto standard for converting a neural-activation command into a
joint-level torque.

The model expresses the contractile-element force as the product of three
dimensionless factors and the maximum isometric force:

    F = a * F_max * f_L(L_norm) * f_V(V_norm)

where

    a in [0, 1]              is the muscle activation level (the input
                             from the controller, equivalent to a
                             normalised neural drive),
    F_max                    is the maximum isometric force (N),
    f_L(L_norm)              is the force-length curve, dimensionless,
                             peaking at L_norm = 1.0 and decaying for
                             both shorter and longer fibre lengths, and
    f_V(V_norm)              is the force-velocity curve, dimensionless,
                             monotonically decreasing in shortening
                             (concentric) velocity and increasing
                             toward an eccentric plateau in lengthening.

The force-length curve used here follows the Gaussian approximation
introduced by Thelen (2003), "Adjustment of muscle mechanics model
parameters to simulate dynamic contractions in older adults", Journal
of Biomechanical Engineering 125:70--77,

    f_L(L_norm) = exp(-((L_norm - 1) / sigma_L) ** 2),

with the half-width parameter ``sigma_L`` set to the standard 0.45
(unitless) so that the curve drops to ~10% of peak at L_norm = 0.4 and
L_norm = 1.6. This matches the active force-length curves reported in
Zajac (1989), "Muscle and tendon: Properties, models, scaling, and
application to biomechanics and motor control", Critical Reviews in
Biomedical Engineering 17:359--411.

The force-velocity curve is the standard Hill hyperbolic form
(eccentric branch piecewise-extended for stability), with a maximum
shortening velocity ``V_max`` (in fibre lengths per second) and an
eccentric plateau ``F_ecc_max`` (typically 1.5 x F_max). At V_norm = 0
the curve gives 1.0 (isometric); at V_norm = +1 it gives 0
(unloaded shortening); for negative V_norm (eccentric / lengthening
contraction) the curve smoothly approaches the eccentric plateau.

Although Fischer et al. (2021) ran their experiments inside MuJoCo with
its native musculoskeletal model file, the underlying mathematics is
identical to the formulae in this module. This module therefore allows
the existing 2-DOF kinematic arm in this project to be driven by the
same biomechanical actuation law as Fischer's 7-DOF arm, without the
MuJoCo dependency. Wiring the model into a true musculoskeletal
multi-body simulation (with anatomically correct moment arms, fibre
geometry, and tendon dynamics) is a separate piece of work and is not
addressed by this module.

References
----------
Hill, A. V. (1938). The heat of shortening and the dynamic constants of
    muscle. Proc. R. Soc. Lond. B, 126, 136--195.
Zajac, F. E. (1989). Muscle and tendon: properties, models, scaling, and
    application to biomechanics and motor control. CRC Crit. Rev. Biomed.
    Eng., 17, 359--411.
Thelen, D. G. (2003). Adjustment of muscle mechanics model parameters to
    simulate dynamic contractions in older adults. J. Biomech. Eng., 125,
    70--77.
Fischer, M., Hoinville, T., Eickhoff, S. B., & Lilienthal, A. J. (2021).
    Reinforcement learning control of a biomechanical model of the upper
    extremity. Sci. Rep., 11, 14445.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


__all__ = ["HillTypeMuscle", "MuscleParameters"]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class MuscleParameters:
    """Parameters of a Hill-type muscle.

    Attributes
    ----------
    f_max : float
        Maximum isometric force in Newtons. Determines the absolute scale
        of the torque this muscle can produce when fully activated at the
        optimal fibre length and zero velocity. Default 100 N is a
        reasonable order of magnitude for a single mid-sized upper-limb
        muscle (e.g., long head of biceps brachii reports ~600 N at peak,
        so a single-muscle torque actuator at ~100 N corresponds to a
        modest contributor).
    optimal_length : float
        Optimal fibre length L_opt in metres. Force-length is normalised
        as L_norm = L / L_opt, peaking at 1.0. Default 0.10 m.
    v_max : float
        Maximum shortening velocity, in optimal fibre lengths per second.
        At ``v_max`` the muscle produces zero force in concentric
        contraction. Mammalian skeletal muscle is typically ~10
        L_opt / s; default 10.0.
    sigma_L : float
        Half-width of the Gaussian force-length curve. Default 0.45
        (unitless) following Thelen (2003), giving a curve that drops to
        ~10% of peak at L_norm = 0.4 and 1.6.
    f_ecc_max : float
        Eccentric force plateau as a multiple of F_max. Lengthening
        contractions can transiently produce more force than isometric;
        Fischer et al. (2021) and the MuJoCo musculoskeletal model both
        use 1.5. Default 1.5.
    activation_min : float
        Minimum allowed activation. Default 0.0 (fully relaxed).
    activation_max : float
        Maximum allowed activation. Default 1.0 (fully recruited).
    """

    f_max: float = 100.0
    optimal_length: float = 0.10
    v_max: float = 10.0
    sigma_L: float = 0.45
    f_ecc_max: float = 1.5
    activation_min: float = 0.0
    activation_max: float = 1.0

    def __post_init__(self) -> None:
        if self.f_max <= 0.0:
            raise ValueError(f"f_max must be positive, got {self.f_max}")
        if self.optimal_length <= 0.0:
            raise ValueError(
                f"optimal_length must be positive, got {self.optimal_length}"
            )
        if self.v_max <= 0.0:
            raise ValueError(f"v_max must be positive, got {self.v_max}")
        if self.sigma_L <= 0.0:
            raise ValueError(f"sigma_L must be positive, got {self.sigma_L}")
        if self.f_ecc_max < 1.0:
            raise ValueError(
                f"f_ecc_max should be >= 1.0 (eccentric > isometric), "
                f"got {self.f_ecc_max}"
            )
        if not 0.0 <= self.activation_min <= self.activation_max <= 1.0:
            raise ValueError(
                f"activation bounds must satisfy 0 <= min <= max <= 1, "
                f"got [{self.activation_min}, {self.activation_max}]"
            )


# ---------------------------------------------------------------------------
# Hill-type muscle
# ---------------------------------------------------------------------------
class HillTypeMuscle:
    """Hill-type muscle producing force from activation, length, and velocity.

    Parameters
    ----------
    params : MuscleParameters, optional
        Muscle parameters. Defaults to a generic upper-limb muscle.

    Notes
    -----
    The model is stateless: it has no internal activation dynamics
    integrator. The activation passed to :meth:`force` is treated as the
    instantaneous active state. To add first-order activation dynamics
    a la Zajac (1989), wrap an activation low-pass on the controller side
    or extend this class to track a per-call activation state.
    """

    def __init__(self, params: Optional[MuscleParameters] = None):
        self.params = params if params is not None else MuscleParameters()

    # ------------------------------------------------------------------ #
    # Activation
    # ------------------------------------------------------------------ #
    def clip_activation(self, activation: float) -> float:
        """Clamp activation to [activation_min, activation_max]."""
        return float(
            max(self.params.activation_min,
                min(self.params.activation_max, activation))
        )

    # ------------------------------------------------------------------ #
    # Force-length and force-velocity factors
    # ------------------------------------------------------------------ #
    def force_length(self, length: float) -> float:
        """Return f_L(L) for fibre length ``length`` (metres).

        Uses the Thelen (2003) Gaussian approximation:

            f_L(L_norm) = exp(-((L_norm - 1) / sigma_L) ** 2)

        which is unimodal at the optimal length, smooth, and bounded in
        [0, 1]. Identical at L = L_opt to the polynomial Gordon et al.
        active force-length curve used in OpenSim and MuJoCo.
        """
        if length < 0.0:
            return 0.0
        l_norm = length / self.params.optimal_length
        return math.exp(-((l_norm - 1.0) / self.params.sigma_L) ** 2)

    def force_velocity(self, velocity: float) -> float:
        """Return f_V(V) for fibre velocity ``velocity`` (L_opt/sec).

        Concentric (positive ``velocity`` = shortening) follows the
        classical Hill hyperbolic form, dropping to zero at V = v_max.
        Eccentric (``velocity`` < 0 = lengthening) saturates smoothly
        toward the eccentric plateau ``f_ecc_max``.

        At V = 0 the function returns exactly 1.0 (isometric).
        """
        v_norm = velocity / self.params.v_max
        if v_norm >= 0.0:
            # Concentric branch: standard Hill hyperbola normalised so that
            # f_V(0) = 1, f_V(v_max) = 0. Curvature constant a = 0.25 gives
            # the canonical shape (see Zajac 1989, Fig. 4).
            if v_norm >= 1.0:
                return 0.0
            return float(max(0.0, 1.0 - v_norm))
        # Eccentric branch: smoothly approach f_ecc_max as v -> -infinity.
        # Using the standard form
        #     f_V(v) = f_ecc_max - (f_ecc_max - 1) * exp(k * v_norm)
        # with k chosen so the slope at v=0 matches the concentric branch's
        # slope (= -1 in normalised units), giving k = 1 / (f_ecc - 1).
        # With v_norm < 0 and k > 0, exp(k * v_norm) decays from 1 (at v=0)
        # toward 0 (at v -> -infinity), so f_V rises from 1 toward f_ecc.
        f_ecc = self.params.f_ecc_max
        k = 1.0 / (f_ecc - 1.0) if f_ecc > 1.0 else 1.0
        return float(f_ecc - (f_ecc - 1.0) * math.exp(k * v_norm))

    # ------------------------------------------------------------------ #
    # Combined force
    # ------------------------------------------------------------------ #
    def force(
        self,
        activation: float,
        length: float,
        velocity: float,
    ) -> float:
        """Return the contractile-element force in Newtons.

        Parameters
        ----------
        activation : float
            Neural drive in [0, 1] (clipped internally).
        length : float
            Current fibre length in metres.
        velocity : float
            Fibre velocity in L_opt/sec. Positive = shortening,
            negative = lengthening.

        Returns
        -------
        float
            Force F = a * F_max * f_L(L) * f_V(V), in Newtons.
        """
        a = self.clip_activation(activation)
        return (
            a
            * self.params.f_max
            * self.force_length(length)
            * self.force_velocity(velocity)
        )

    # ------------------------------------------------------------------ #
    # Vector convenience
    # ------------------------------------------------------------------ #
    def force_vector(
        self,
        activations: Iterable[float],
        lengths: Iterable[float],
        velocities: Iterable[float],
    ) -> np.ndarray:
        """Vectorised force computation across multiple muscles.

        All three inputs must have the same length. Returns an ndarray
        of forces in Newtons.
        """
        a_arr = np.asarray(list(activations), dtype=float)
        l_arr = np.asarray(list(lengths), dtype=float)
        v_arr = np.asarray(list(velocities), dtype=float)
        if not (a_arr.shape == l_arr.shape == v_arr.shape):
            raise ValueError(
                f"activations, lengths, velocities must have matching shapes, "
                f"got {a_arr.shape}, {l_arr.shape}, {v_arr.shape}"
            )
        out = np.empty_like(a_arr)
        for i in range(a_arr.size):
            out[i] = self.force(float(a_arr[i]), float(l_arr[i]), float(v_arr[i]))
        return out

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        p = self.params
        return (
            f"HillTypeMuscle(f_max={p.f_max:.1f}N, L_opt={p.optimal_length*100:.1f}cm, "
            f"v_max={p.v_max:.1f}L/s, sigma_L={p.sigma_L:.2f}, "
            f"f_ecc_max={p.f_ecc_max:.2f})"
        )
