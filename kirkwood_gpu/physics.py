"""Physics of the planar circular restricted three-body problem (CR3BP).

Unit system
-----------
Length:   AU
Time:     year
Mass:     solar mass (M_sun = 1)

In these units, Kepler's third law n^2 a^3 = G M gives G M_sun = 4 pi^2.

Frame
-----
We integrate in the inertial barycentric frame. The Sun and Jupiter move on
analytic circular orbits about the common barycenter; the test particle feels
only their gravity. Osculating elements and Jacobi constant are computed from
the inertial state.

The Jacobi constant takes its frame-independent form

    C_J = 2 (n L - E)

where L = x * vy - y * vx is the inertial specific angular momentum and
E = 0.5 * v^2 - GM_sun / r_sun - GM_J / r_J is the inertial specific energy.
This avoids mixing rotating-frame transformations into the measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---- fundamental constants in (AU, yr, M_sun) ----------------------------

GM_SUN: float = 4.0 * math.pi * math.pi  # AU^3 / yr^2, by construction

# Jupiter mass ratio m_J / m_sun. Murray & Dermott Table A.2: 1/1047.3486.
MASS_RATIO_JUPITER: float = 1.0 / 1047.3486

# Jupiter orbital semi-major axis (AU).
SEMIMAJOR_JUPITER: float = 5.2044


@dataclass(frozen=True)
class CR3BPParams:
    """Immutable parameter bundle for a CR3BP configuration.

    All quantities are in (AU, yr, M_sun) units. The default corresponds to
    the physical Sun-Jupiter system. Pass ``mass_ratio_scale > 1`` to inflate
    Jupiter's mass (didactic runs, not for headline science).
    """

    a_J: float = SEMIMAJOR_JUPITER
    mass_ratio: float = MASS_RATIO_JUPITER
    mass_ratio_scale: float = 1.0  # keep 1.0 for realistic science

    @property
    def q(self) -> float:
        """Effective m_J / m_sun after any scaling."""
        return self.mass_ratio * self.mass_ratio_scale

    @property
    def mu(self) -> float:
        """CR3BP reduced mass mu = m_J / (m_sun + m_J)."""
        q = self.q
        return q / (1.0 + q)

    @property
    def GM_sun(self) -> float:
        return GM_SUN

    @property
    def GM_jup(self) -> float:
        return GM_SUN * self.q

    @property
    def mean_motion(self) -> float:
        """Sun-Jupiter synodic angular frequency n (rad / yr)."""
        return math.sqrt(GM_SUN * (1.0 + self.q) / self.a_J**3)

    @property
    def jupiter_period(self) -> float:
        return 2.0 * math.pi / self.mean_motion


# ---- primary positions in inertial barycentric frame ---------------------


def primary_positions(t, params: CR3BPParams, xp=np):
    """Return Sun and Jupiter positions at time t (scalar or array).

    Both primaries sit on circles of radius mu*a_J and (1-mu)*a_J respectively,
    180 degrees apart, rotating at the mean motion. At t=0 Jupiter is on the
    +x axis.
    """
    n = params.mean_motion
    mu = params.mu
    a = params.a_J
    theta = n * t
    cos_t = xp.cos(theta)
    sin_t = xp.sin(theta)
    # Sun at -mu*a in the rotated direction; Jupiter at (1-mu)*a.
    xs = -mu * a * cos_t
    ys = -mu * a * sin_t
    xj = (1.0 - mu) * a * cos_t
    yj = (1.0 - mu) * a * sin_t
    return (xs, ys), (xj, yj)


_ACCEL_KERNEL = None


def _get_accel_kernel():
    """Fused CuPy ElementwiseKernel for the CR3BP Sun+Jupiter acceleration.

    Collapses ~10 per-particle elementwise ops (two vector subtractions,
    two inv_r3s, two force terms, and the sum) to a single GPU kernel
    launch. Primary positions are passed as scalar kernel parameters so
    they are broadcast once to all threads.
    """
    global _ACCEL_KERNEL
    if _ACCEL_KERNEL is not None:
        return _ACCEL_KERNEL
    import cupy as cp  # type: ignore

    _ACCEL_KERNEL = cp.ElementwiseKernel(
        in_params=(
            "float64 x, float64 y, "
            "float64 xs, float64 ys, float64 xj, float64 yj, "
            "float64 gm_s, float64 gm_j"
        ),
        out_params="float64 ax, float64 ay",
        operation=r"""
        double dxs = x - xs, dys = y - ys;
        double rs2 = dxs*dxs + dys*dys;
        double inv_rs3 = pow(rs2, -1.5);
        double dxj = x - xj, dyj = y - yj;
        double rj2 = dxj*dxj + dyj*dyj;
        double inv_rj3 = pow(rj2, -1.5);
        ax = -gm_s * dxs * inv_rs3 - gm_j * dxj * inv_rj3;
        ay = -gm_s * dys * inv_rs3 - gm_j * dyj * inv_rj3;
        """,
        name="cr3bp_accel_fused",
    )
    return _ACCEL_KERNEL


def _is_cupy_array(a):
    return type(a).__module__.startswith("cupy")


def acceleration(x, y, t, params: CR3BPParams, xp=np):
    """Gravitational acceleration on test particles from Sun + Jupiter.

    ``x, y`` are arrays of shape (N,). Returns (ax, ay) of the same shape.

    On CuPy arrays, dispatches to a fused ElementwiseKernel that does the
    entire force evaluation in one GPU launch. On NumPy, uses the
    vectorized reference formulation below.
    """
    (xs, ys), (xj, yj) = primary_positions(t, params, xp=xp)

    if _is_cupy_array(x):
        kernel = _get_accel_kernel()
        ax = xp.empty_like(x)
        ay = xp.empty_like(y)
        kernel(
            x, y,
            float(xs), float(ys), float(xj), float(yj),
            float(params.GM_sun), float(params.GM_jup),
            ax, ay,
        )
        return ax, ay

    dx_s = x - xs
    dy_s = y - ys
    r2_s = dx_s * dx_s + dy_s * dy_s
    inv_r3_s = r2_s ** (-1.5)

    dx_j = x - xj
    dy_j = y - yj
    r2_j = dx_j * dx_j + dy_j * dy_j
    inv_r3_j = r2_j ** (-1.5)

    gm_s = params.GM_sun
    gm_j = params.GM_jup
    ax = -gm_s * dx_s * inv_r3_s - gm_j * dx_j * inv_r3_j
    ay = -gm_s * dy_s * inv_r3_s - gm_j * dy_j * inv_r3_j
    return ax, ay


# ---- conserved quantities ------------------------------------------------


def inertial_energy(x, y, vx, vy, t, params: CR3BPParams, xp=np):
    """Specific inertial energy E = 0.5 v^2 - GM_sun/r_sun - GM_jup/r_j."""
    (xs, ys), (xj, yj) = primary_positions(t, params, xp=xp)
    r_s = xp.sqrt((x - xs) ** 2 + (y - ys) ** 2)
    r_j = xp.sqrt((x - xj) ** 2 + (y - yj) ** 2)
    v2 = vx * vx + vy * vy
    return 0.5 * v2 - params.GM_sun / r_s - params.GM_jup / r_j


def jacobi_constant(x, y, vx, vy, t, params: CR3BPParams, xp=np):
    """Jacobi constant C_J = 2 (n L - E), frame-independent.

    L = x*vy - y*vx is inertial specific angular momentum.
    """
    L = x * vy - y * vx
    E = inertial_energy(x, y, vx, vy, t, params, xp=xp)
    return 2.0 * (params.mean_motion * L - E)


# ---- osculating elements -------------------------------------------------


def osculating_semimajor_axis(x, y, vx, vy, t, params: CR3BPParams, xp=np):
    """Heliocentric osculating semi-major axis (AU).

    Computed relative to the Sun's instantaneous position and velocity,
    using the two-body vis-viva relation with mu = GM_sun. This is the
    natural coordinate in which Kirkwood gaps appear.
    """
    (xs, ys), _ = primary_positions(t, params, xp=xp)
    n = params.mean_motion
    # Sun's inertial velocity: derivative of -mu*a*(cos nt, sin nt).
    mu = params.mu
    a = params.a_J
    vxs = mu * a * n * xp.sin(n * t)
    vys = -mu * a * n * xp.cos(n * t)

    dx = x - xs
    dy = y - ys
    dvx = vx - vxs
    dvy = vy - vys
    r = xp.sqrt(dx * dx + dy * dy)
    v2 = dvx * dvx + dvy * dvy
    # vis-viva: v^2 = GM (2/r - 1/a)  =>  1/a = 2/r - v^2/GM
    inv_a = 2.0 / r - v2 / params.GM_sun
    return 1.0 / inv_a


def osculating_eccentricity(x, y, vx, vy, t, params: CR3BPParams, xp=np):
    """Heliocentric osculating eccentricity (planar)."""
    (xs, ys), _ = primary_positions(t, params, xp=xp)
    n = params.mean_motion
    mu = params.mu
    a = params.a_J
    vxs = mu * a * n * xp.sin(n * t)
    vys = -mu * a * n * xp.cos(n * t)

    dx = x - xs
    dy = y - ys
    dvx = vx - vxs
    dvy = vy - vys
    r = xp.sqrt(dx * dx + dy * dy)
    v2 = dvx * dvx + dvy * dvy
    gm = params.GM_sun
    # specific angular momentum (planar): scalar h_z = dx*dvy - dy*dvx
    hz = dx * dvy - dy * dvx
    # Laplace-Runge-Lenz vector components
    ex = (dvy * hz) / gm - dx / r
    ey = (-dvx * hz) / gm - dy / r
    return xp.sqrt(ex * ex + ey * ey)


# ---- resonance locations -------------------------------------------------


def resonance_semimajor_axis(p: int, q: int, a_J: float = SEMIMAJOR_JUPITER) -> float:
    """Semi-major axis of an interior p:q mean-motion resonance with Jupiter.

    A p:q interior MMR has T_asteroid / T_jupiter = q / p (with p > q),
    hence a_asteroid = a_J * (q/p)^(2/3). For example:

        >>> round(resonance_semimajor_axis(3, 1), 4)
        2.5015
        >>> round(resonance_semimajor_axis(2, 1), 4)
        3.2785
        >>> round(resonance_semimajor_axis(3, 2), 4)
        3.9714
    """
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive integers")
    ratio = q / p
    return a_J * ratio ** (2.0 / 3.0)
