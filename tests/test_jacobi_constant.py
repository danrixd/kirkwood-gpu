"""Jacobi constant conservation in CR3BP.

For any test particle that does not undergo a close encounter with Jupiter
(and we pick a handful of sub-resonance semi-major axes that stay regular
over ~5 Jupiter periods), the Jacobi constant must be conserved. The
leapfrog symplectic property manifests as bounded, non-secular error in
C_J. Target: max |dC/C0| < 5e-6 over 5 Jupiter orbits at dt = T_J / 500.
"""

from __future__ import annotations

import math

import numpy as np

from kirkwood_gpu.backend import to_numpy, xp
from kirkwood_gpu.integrators import State, leapfrog_step
from kirkwood_gpu.physics import CR3BPParams, GM_SUN, jacobi_constant


def _initial_belt(n: int, a_min: float = 2.0, a_max: float = 3.5, seed: int = 0):
    """Circular heliocentric orbits at evenly spaced semi-major axes."""
    rng = np.random.default_rng(seed)
    a = np.linspace(a_min, a_max, n)
    phi0 = rng.uniform(0.0, 2.0 * math.pi, size=n)
    x = a * np.cos(phi0)
    y = a * np.sin(phi0)
    # circular speed relative to the Sun (barycentric Sun is near origin;
    # for the Jacobi test this small offset is fine — C_J still conserved).
    v = np.sqrt(GM_SUN / a)
    vx = -v * np.sin(phi0)
    vy = v * np.cos(phi0)
    return x, y, vx, vy


def test_jacobi_conserved_over_five_jupiter_periods():
    params = CR3BPParams()  # physical Jupiter mass
    # Pick semi-major axes that avoid the 3:1 (2.50) and 2:1 (3.28) resonances.
    a_choices = np.array([2.1, 2.3, 2.7, 2.9, 3.1, 3.45])
    n = len(a_choices)
    rng = np.random.default_rng(42)
    phi0 = rng.uniform(0.0, 2.0 * math.pi, size=n)
    x = xp.asarray(a_choices * np.cos(phi0))
    y = xp.asarray(a_choices * np.sin(phi0))
    v = np.sqrt(GM_SUN / a_choices)
    vx = xp.asarray(-v * np.sin(phi0))
    vy = xp.asarray(v * np.cos(phi0))
    alive = xp.ones(n, dtype=bool)
    state = State(x, y, vx, vy, 0.0, alive)

    T_J = params.jupiter_period
    steps_per_orbit = 500
    n_orbits = 5
    n_steps = steps_per_orbit * n_orbits
    dt = T_J / steps_per_orbit

    c0 = to_numpy(
        jacobi_constant(state.x, state.y, state.vx, state.vy, state.t, params, xp=xp)
    ).copy()
    c_hist = [c0.copy()]
    for _ in range(n_steps):
        state = leapfrog_step(state, dt, params)
        c_hist.append(
            to_numpy(
                jacobi_constant(
                    state.x, state.y, state.vx, state.vy, state.t, params, xp=xp
                )
            ).copy()
        )
    c_hist = np.stack(c_hist, axis=0)  # (n_steps+1, n)
    rel_err = np.abs(c_hist - c0[None, :]) / np.abs(c0[None, :])
    max_err = rel_err.max()
    assert max_err < 5e-6, (
        f"Jacobi constant drift too large: max |dC/C0| = {max_err:.3e}"
    )

    # bounded (not secular) drift: worst error in the 2nd half should be
    # comparable to the worst in the 1st half, not dramatically larger.
    first = rel_err[: n_steps // 2].max()
    second = rel_err[n_steps // 2 :].max()
    assert second < 3.0 * max(first, 1e-14), (
        f"Jacobi error appears to be drifting: first-half max {first:.2e}, "
        f"second-half max {second:.2e}"
    )
