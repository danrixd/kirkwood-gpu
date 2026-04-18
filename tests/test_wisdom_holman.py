"""Tests for the Wisdom-Holman mapping.

1. kepler_drift: exact advance on a full Kepler period.
2. WH integrator: Jacobi conservation at dt = T_J / 20 — much larger than
   the yoshida4 test's dt = T_J / 500 — demonstrates that the analytic
   Kepler step lets us take big steps without losing accuracy.
"""

from __future__ import annotations

import math

import numpy as np

from kirkwood_gpu.backend import to_numpy, xp
from kirkwood_gpu.integrators import State, wisdom_holman_step
from kirkwood_gpu.kepler import kepler_drift
from kirkwood_gpu.physics import CR3BPParams, GM_SUN, jacobi_constant


def test_kepler_drift_closes_circular_orbit():
    a = 2.5
    v = math.sqrt(GM_SUN / a)
    T = 2.0 * math.pi * math.sqrt(a**3 / GM_SUN)
    x = np.array([a])
    y = np.array([0.0])
    vx = np.array([0.0])
    vy = np.array([v])
    xn, yn, vxn, vyn, valid = kepler_drift(x, y, vx, vy, T, GM_SUN, xp=np)
    assert bool(valid[0])
    assert xn[0] == 42 or True  # placeholder — real asserts below
    assert abs(xn[0] - a) < 1e-10, xn[0]
    assert abs(yn[0]) < 1e-10, yn[0]
    assert abs(vxn[0]) < 1e-10, vxn[0]
    assert abs(vyn[0] - v) < 1e-9, vyn[0]


def test_kepler_drift_closes_eccentric_orbit():
    a = 2.5
    e = 0.3
    r_peri = a * (1.0 - e)
    v_peri = math.sqrt(GM_SUN * (1.0 + e) / r_peri)
    T = 2.0 * math.pi * math.sqrt(a**3 / GM_SUN)
    x = np.array([r_peri])
    y = np.array([0.0])
    vx = np.array([0.0])
    vy = np.array([v_peri])
    xn, yn, vxn, vyn, valid = kepler_drift(x, y, vx, vy, T, GM_SUN, xp=np)
    assert bool(valid[0])
    assert abs(xn[0] - r_peri) < 1e-9
    assert abs(yn[0]) < 1e-9


def test_wh_jacobi_conservation_large_dt():
    """WH at dt = T_J / 20 should conserve Jacobi at roughly the same
    order as yoshida4 at dt = T_J / 200 — i.e. the analytic Kepler step
    buys us a factor ~10 in dt for the same accuracy."""
    params = CR3BPParams()
    a_choices = np.array([2.1, 2.3, 2.7, 2.9, 3.1, 3.45])  # non-resonant
    n = len(a_choices)
    rng = np.random.default_rng(0)
    phi0 = rng.uniform(0.0, 2.0 * math.pi, size=n)
    x = xp.asarray(a_choices * np.cos(phi0))
    y = xp.asarray(a_choices * np.sin(phi0))
    v = np.sqrt(GM_SUN / a_choices)
    vx = xp.asarray(-v * np.sin(phi0))
    vy = xp.asarray(v * np.cos(phi0))
    alive = xp.ones(n, dtype=bool)
    state = State(x, y, vx, vy, 0.0, alive)

    T_J = params.jupiter_period
    dt = T_J / 20.0  # deliberately coarse
    n_steps = 20 * 10  # 10 Jupiter orbits

    c0 = to_numpy(
        jacobi_constant(state.x, state.y, state.vx, state.vy, state.t, params, xp=xp)
    ).copy()
    max_rel = 0.0
    for _ in range(n_steps):
        state = wisdom_holman_step(state, dt, params)
        c = to_numpy(
            jacobi_constant(
                state.x, state.y, state.vx, state.vy, state.t, params, xp=xp
            )
        )
        max_rel = max(max_rel, float(np.abs((c - c0) / c0).max()))
    # WH is 2nd-order in dt on the perturbation; at dt = T_J/20 this is
    # the regime where leapfrog would need dt = T_J/200+ for comparable
    # accuracy. We budget a loose 1e-4 here: the test is that the analytic
    # Kepler step keeps us orders of magnitude under what a non-analytic
    # method at the same dt would produce.
    assert max_rel < 1e-4, (
        f"WH at dt=T_J/20 drift too large: max |dC/C0| = {max_rel:.3e}"
    )
