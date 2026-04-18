"""Symplectic integrators for CR3BP test-particle integration.

We integrate in the inertial barycentric frame, where the Sun and Jupiter
move on analytic circles. The test-particle acceleration comes from the sum
of the two primary potentials at the current time.

Integrator: Stormer-Verlet / kick-drift-kick leapfrog. This is second-order
and symplectic for autonomous Hamiltonians, and remains time-reversible for
the time-dependent external potential that the moving primaries produce.
Over 10^5-year integrations the energy error stays bounded rather than
drifting linearly, which is what makes the Kirkwood-gap signal trustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .backend import xp
from .kepler import kepler_drift
from .physics import CR3BPParams, acceleration


@dataclass
class State:
    """Bundle of per-particle arrays carried through integration.

    ``x, y, vx, vy`` are 1-D arrays of length N (N test particles).
    ``alive`` is a boolean mask; particles that escape or collide with a
    primary are marked dead and their accelerations are forced to zero so
    their state freezes for the remainder of the run.
    """

    x: "xp.ndarray"
    y: "xp.ndarray"
    vx: "xp.ndarray"
    vy: "xp.ndarray"
    t: float
    alive: "xp.ndarray"


def _zero_dead(accel, alive):
    return xp.where(alive, accel, xp.zeros_like(accel))


# Yoshida (1990) 4th-order symplectic composition weights.
# Applying leapfrog(w1*dt), leapfrog(w0*dt), leapfrog(w1*dt) yields an
# overall 4th-order symplectic map per composite step. Reference:
# H. Yoshida, Phys. Lett. A 150 (1990), 262-268.
_YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_W0 = 1.0 - 2.0 * _YOSHIDA_W1


def leapfrog_step(
    state: State,
    dt: float,
    params: CR3BPParams,
) -> State:
    """One KDK leapfrog step in place on ``state`` (returns a new State)."""
    ax, ay = acceleration(state.x, state.y, state.t, params, xp=xp)
    ax = _zero_dead(ax, state.alive)
    ay = _zero_dead(ay, state.alive)

    half_dt = 0.5 * dt
    vx_half = state.vx + half_dt * ax
    vy_half = state.vy + half_dt * ay

    x_new = state.x + dt * vx_half
    y_new = state.y + dt * vy_half
    t_new = state.t + dt

    ax2, ay2 = acceleration(x_new, y_new, t_new, params, xp=xp)
    ax2 = _zero_dead(ax2, state.alive)
    ay2 = _zero_dead(ay2, state.alive)

    vx_new = vx_half + half_dt * ax2
    vy_new = vy_half + half_dt * ay2

    return State(x_new, y_new, vx_new, vy_new, t_new, state.alive)


def yoshida4_step(
    state: State,
    dt: float,
    params: CR3BPParams,
) -> State:
    """4th-order symplectic step: 3 leapfrog substeps per composite step.

    Costs 3x the force evaluations of plain leapfrog but gains two orders
    of accuracy in dt. At the same cost (same number of force evals) as
    leapfrog-with-dt/3, Yoshida4 is much more accurate, so for stringent
    conservation targets it is the better trade.
    """
    state = leapfrog_step(state, _YOSHIDA_W1 * dt, params)
    state = leapfrog_step(state, _YOSHIDA_W0 * dt, params)
    state = leapfrog_step(state, _YOSHIDA_W1 * dt, params)
    return state


# ---- Wisdom-Holman mapping ----------------------------------------------


def _jupiter_helio(t: float, params: CR3BPParams):
    """Heliocentric position of Jupiter at time t (scalar)."""
    n = params.mean_motion
    a = params.a_J
    c = math.cos(n * t)
    s = math.sin(n * t)
    return a * c, a * s


def _sun_bary(t: float, params: CR3BPParams):
    """Barycentric position and velocity of the Sun at time t."""
    n = params.mean_motion
    mu = params.mu
    a = params.a_J
    c = math.cos(n * t)
    s = math.sin(n * t)
    x_s = -mu * a * c
    y_s = -mu * a * s
    vx_s = mu * a * n * s
    vy_s = -mu * a * n * c
    return x_s, y_s, vx_s, vy_s


def wisdom_holman_step(state: State, dt: float, params: CR3BPParams) -> State:
    """DKD Wisdom-Holman map: Kepler drift dt/2, Jupiter kick dt, Kepler drift dt/2.

    The drift is analytic (exact Kepler advance in heliocentric coords),
    so dt is limited only by the Jupiter-perturbation timescale — in
    practice dt ~ T_J / 20 works where Yoshida4 needs T_J / 200. This
    is where the Wisdom-Holman map earns its name.
    """
    # 1. convert barycentric state -> heliocentric (positions and velocities)
    x_s, y_s, vx_s, vy_s = _sun_bary(state.t, params)
    rho_x = state.x - x_s
    rho_y = state.y - y_s
    rho_vx = state.vx - vx_s
    rho_vy = state.vy - vy_s

    # 2. Kepler drift dt/2
    rho_x, rho_y, rho_vx, rho_vy, valid1 = kepler_drift(
        rho_x, rho_y, rho_vx, rho_vy, 0.5 * dt, params.GM_sun, xp=xp
    )

    # 3. Kick dt under the Jupiter-interaction perturbation
    #    dH_pert/drho = GM_j * (rho - R_hel)/|rho - R_hel|^3 + GM_j * R_hel / a_J^3
    t_mid = state.t + 0.5 * dt
    Rx, Ry = _jupiter_helio(t_mid, params)
    dx = rho_x - Rx
    dy = rho_y - Ry
    r2 = dx * dx + dy * dy
    inv_r3 = r2 ** (-1.5)
    gm_j = params.GM_jup
    a_j3 = params.a_J ** 3
    # direct + indirect terms
    ax = -gm_j * dx * inv_r3 - gm_j * Rx / a_j3
    ay = -gm_j * dy * inv_r3 - gm_j * Ry / a_j3
    # respect dead-particle mask
    ax = xp.where(state.alive & valid1, ax, xp.zeros_like(ax))
    ay = xp.where(state.alive & valid1, ay, xp.zeros_like(ay))
    rho_vx = rho_vx + dt * ax
    rho_vy = rho_vy + dt * ay

    # 4. Kepler drift dt/2
    rho_x, rho_y, rho_vx, rho_vy, valid2 = kepler_drift(
        rho_x, rho_y, rho_vx, rho_vy, 0.5 * dt, params.GM_sun, xp=xp
    )

    # 5. back to barycentric inertial at t + dt
    t_new = state.t + dt
    x_s2, y_s2, vx_s2, vy_s2 = _sun_bary(t_new, params)
    x_new = rho_x + x_s2
    y_new = rho_y + y_s2
    vx_new = rho_vx + vx_s2
    vy_new = rho_vy + vy_s2

    alive_new = state.alive & valid1 & valid2
    return State(x_new, y_new, vx_new, vy_new, t_new, alive_new)


# ---- inertial-frame Kepler helper (for Kepler 2-body sanity test) -------


def kepler_acceleration(x, y, gm: float = None):
    """Inertial-frame acceleration of a test particle toward the origin
    under a point-mass ``gm``. Vectorized, used by the energy conservation
    test so that we can validate the integrator kernel against an analytic
    two-body problem before introducing Jupiter.
    """
    if gm is None:
        from .physics import GM_SUN

        gm = GM_SUN
    r2 = x * x + y * y
    inv_r3 = r2 ** (-1.5)
    return -gm * x * inv_r3, -gm * y * inv_r3


def kepler_rk4_step(x, y, vx, vy, dt: float, gm: float):
    """Classical RK4 step for the Kepler 2-body problem.

    Not used by production runs — provided purely as a reference against
    which to compare symplectic methods on a log-drift plot. RK4 is
    globally 4th-order accurate per step but has *secular* energy drift
    on Hamiltonian systems: the error grows linearly with integration
    time. This is the quintessential cautionary tale of using a
    general-purpose ODE solver on a long-horizon orbital integration.
    """
    def f(x, y, vx, vy):
        ax, ay = kepler_acceleration(x, y, gm=gm)
        return vx, vy, ax, ay

    k1x, k1y, k1vx, k1vy = f(x, y, vx, vy)
    k2x, k2y, k2vx, k2vy = f(
        x + 0.5 * dt * k1x, y + 0.5 * dt * k1y,
        vx + 0.5 * dt * k1vx, vy + 0.5 * dt * k1vy,
    )
    k3x, k3y, k3vx, k3vy = f(
        x + 0.5 * dt * k2x, y + 0.5 * dt * k2y,
        vx + 0.5 * dt * k2vx, vy + 0.5 * dt * k2vy,
    )
    k4x, k4y, k4vx, k4vy = f(
        x + dt * k3x, y + dt * k3y,
        vx + dt * k3vx, vy + dt * k3vy,
    )
    x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_new = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    vx_new = vx + (dt / 6.0) * (k1vx + 2.0 * k2vx + 2.0 * k3vx + k4vx)
    vy_new = vy + (dt / 6.0) * (k1vy + 2.0 * k2vy + 2.0 * k3vy + k4vy)
    return x_new, y_new, vx_new, vy_new


def kepler_yoshida4_step(x, y, vx, vy, dt: float, gm: float):
    """Yoshida4 wrapper for the Kepler 2-body kernel (drift comparison)."""
    s = kepler_leapfrog_step
    x, y, vx, vy = s(x, y, vx, vy, _YOSHIDA_W1 * dt, gm)
    x, y, vx, vy = s(x, y, vx, vy, _YOSHIDA_W0 * dt, gm)
    x, y, vx, vy = s(x, y, vx, vy, _YOSHIDA_W1 * dt, gm)
    return x, y, vx, vy


def kepler_leapfrog_step(x, y, vx, vy, dt: float, gm: float):
    """One KDK leapfrog step for a test particle around a fixed point mass.

    Not used by the main CR3BP run — isolated here so the Kepler
    conservation test can exercise the minimum symplectic kernel without
    dragging in the rotating-frame plumbing.
    """
    ax, ay = kepler_acceleration(x, y, gm=gm)
    half_dt = 0.5 * dt
    vx_half = vx + half_dt * ax
    vy_half = vy + half_dt * ay
    x_new = x + dt * vx_half
    y_new = y + dt * vy_half
    ax2, ay2 = kepler_acceleration(x_new, y_new, gm=gm)
    vx_new = vx_half + half_dt * ax2
    vy_new = vy_half + half_dt * ay2
    return x_new, y_new, vx_new, vy_new


# ---- bulk driver ---------------------------------------------------------


def integrate(
    state: State,
    params: CR3BPParams,
    dt: float,
    n_steps: int,
    snapshot_every: int = 0,
    on_snapshot: Callable[[State, int], None] | None = None,
    escape_radius: float = 100.0,
    collision_radius_sun: float = 0.005,  # AU (~1 solar radius)
    collision_radius_jup: float = 0.0005,  # AU (~1 Jupiter radius)
    integrator: str = "yoshida4",
):
    """Advance ``state`` by ``n_steps`` composite steps of size ``dt``.

    ``integrator`` selects between ``"leapfrog"`` (2nd-order KDK) and
    ``"yoshida4"`` (4th-order composition; 3x cost per step, much smaller
    error).  Every ``snapshot_every`` steps (0 disables), invoke
    ``on_snapshot(state, step_index)``. Particles that stray past
    ``escape_radius`` from the barycenter, or come within the collision
    radius of either primary, are flagged dead and stop accumulating state.
    """
    from .physics import primary_positions

    if integrator == "leapfrog":
        step = leapfrog_step
    elif integrator == "yoshida4":
        step = yoshida4_step
    elif integrator == "wisdom_holman":
        step = wisdom_holman_step
    else:
        raise ValueError(f"unknown integrator {integrator!r}")

    for i in range(n_steps):
        state = step(state, dt, params)

        # dead-particle housekeeping (cheap compared with force eval)
        r2 = state.x * state.x + state.y * state.y
        (xs, ys), (xj, yj) = primary_positions(state.t, params, xp=xp)
        r2_s = (state.x - xs) ** 2 + (state.y - ys) ** 2
        r2_j = (state.x - xj) ** 2 + (state.y - yj) ** 2
        alive = state.alive & (r2 < escape_radius**2)
        alive = alive & (r2_s > collision_radius_sun**2)
        alive = alive & (r2_j > collision_radius_jup**2)
        state = State(state.x, state.y, state.vx, state.vy, state.t, alive)

        if snapshot_every and on_snapshot and ((i + 1) % snapshot_every == 0):
            on_snapshot(state, i + 1)

    return state
