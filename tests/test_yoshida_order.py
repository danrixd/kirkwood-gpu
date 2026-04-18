"""Yoshida4 must show 4th-order accuracy on a Kepler orbit.

This is a direct numerical verification that the composition of three
leapfrog substeps at the Yoshida weights gains two orders of accuracy
over plain leapfrog. We compare the position error at the end of one
orbit as a function of step size and fit the slope.
"""

from __future__ import annotations

import math

import numpy as np

from kirkwood_gpu.backend import to_numpy, xp
from kirkwood_gpu.integrators import (
    State,
    leapfrog_step,
    yoshida4_step,
)
from kirkwood_gpu.physics import CR3BPParams, GM_SUN


def _kepler_circular(a, steps, dt, step_fn):
    """Run a test particle around a fixed point mass (Sun) via the full
    State machinery with mass_ratio_scale=0 so Jupiter contributes nothing.
    Returns final (x, y, vx, vy)."""
    params = CR3BPParams(mass_ratio_scale=0.0)  # Jupiter zero -> pure Kepler
    v = math.sqrt(GM_SUN / a)
    x = xp.asarray([a], dtype="float64")
    y = xp.asarray([0.0], dtype="float64")
    vx = xp.asarray([0.0], dtype="float64")
    vy = xp.asarray([v], dtype="float64")
    alive = xp.ones(1, dtype=bool)
    state = State(x, y, vx, vy, 0.0, alive)
    for _ in range(steps):
        state = step_fn(state, dt, params)
    return state


def _final_error(a, orbits, steps_per_orbit, step_fn):
    period = 2.0 * math.pi * math.sqrt(a**3 / GM_SUN)
    dt = period / steps_per_orbit
    steps = orbits * steps_per_orbit
    state = _kepler_circular(a, steps, dt, step_fn)
    # circular orbit: should return to (a, 0) after an integer number of orbits
    x_final = float(to_numpy(state.x)[0])
    y_final = float(to_numpy(state.y)[0])
    dx = x_final - a
    dy = y_final - 0.0
    return math.sqrt(dx * dx + dy * dy)


def _fit_order(errors, hs):
    # slope of log(err) vs log(h)
    le = np.log(np.asarray(errors))
    lh = np.log(np.asarray(hs))
    slope, _ = np.polyfit(lh, le, 1)
    return slope


def test_leapfrog_is_second_order():
    a = 2.5
    orbits = 4
    spo_list = [100, 200, 400]
    errs = [_final_error(a, orbits, spo, leapfrog_step) for spo in spo_list]
    hs = [1.0 / spo for spo in spo_list]
    order = _fit_order(errs, hs)
    assert 1.7 < order < 2.3, f"leapfrog order {order:.2f} != 2"


def test_yoshida4_is_fourth_order():
    a = 2.5
    orbits = 4
    spo_list = [40, 80, 160]
    errs = [_final_error(a, orbits, spo, yoshida4_step) for spo in spo_list]
    hs = [1.0 / spo for spo in spo_list]
    order = _fit_order(errs, hs)
    assert 3.5 < order < 4.5, f"yoshida4 order {order:.2f} != 4"
