"""Leapfrog must conserve energy on a 2-body Kepler orbit.

This is the minimum credibility check for the integrator kernel: before
putting Jupiter back in, the KDK leapfrog must hold energy on a clean,
analytic two-body problem to roughly floating-point precision over 1e4
steps. A non-symplectic integrator (RK4 at a comparable step) would show
linear energy drift; leapfrog's error should be bounded and oscillatory.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kirkwood_gpu.integrators import kepler_leapfrog_step
from kirkwood_gpu.physics import GM_SUN


def _circular_orbit_ic(a: float):
    """Circular Kepler orbit at semi-major axis a."""
    v = math.sqrt(GM_SUN / a)
    return np.array([a]), np.array([0.0]), np.array([0.0]), np.array([v])


def _energy(x, y, vx, vy, gm):
    r = np.sqrt(x * x + y * y)
    return 0.5 * (vx * vx + vy * vy) - gm / r


def test_leapfrog_circular_orbit_energy_bounded():
    a = 2.5  # AU — well inside the Kirkwood belt
    x, y, vx, vy = _circular_orbit_ic(a)
    period = 2.0 * math.pi * math.sqrt(a**3 / GM_SUN)
    # 50 orbits, 200 steps/orbit -> 1e4 steps
    n_steps = 10_000
    dt = 50.0 * period / n_steps

    e0 = _energy(x, y, vx, vy, GM_SUN).item()
    e_hist = [e0]
    for _ in range(n_steps):
        x, y, vx, vy = kepler_leapfrog_step(x, y, vx, vy, dt, GM_SUN)
        e_hist.append(_energy(x, y, vx, vy, GM_SUN).item())

    e_hist = np.asarray(e_hist)
    rel_err = np.abs(e_hist - e0) / abs(e0)
    assert rel_err.max() < 1e-6, (
        f"leapfrog energy drift too large: max |dE/E| = {rel_err.max():.3e}"
    )


def test_leapfrog_eccentric_orbit_energy_bounded():
    """Modestly eccentric orbit (e=0.2) should still hold energy well."""
    a = 2.5
    e = 0.2
    # perihelion state: r = a(1-e), v = sqrt(GM (1+e)/(a(1-e)))
    r_peri = a * (1.0 - e)
    v_peri = math.sqrt(GM_SUN * (1.0 + e) / r_peri)
    x = np.array([r_peri])
    y = np.array([0.0])
    vx = np.array([0.0])
    vy = np.array([v_peri])

    period = 2.0 * math.pi * math.sqrt(a**3 / GM_SUN)
    n_steps = 10_000
    # shorter dt for eccentric case: 400 steps per orbit, 25 orbits
    dt = 25.0 * period / n_steps

    e0 = _energy(x, y, vx, vy, GM_SUN).item()
    e_hist = [e0]
    for _ in range(n_steps):
        x, y, vx, vy = kepler_leapfrog_step(x, y, vx, vy, dt, GM_SUN)
        e_hist.append(_energy(x, y, vx, vy, GM_SUN).item())

    e_hist = np.asarray(e_hist)
    rel_err = np.abs(e_hist - e0) / abs(e0)
    assert rel_err.max() < 1e-4, (
        f"eccentric leapfrog drift too large: max |dE/E| = {rel_err.max():.3e}"
    )
    # symplectic signature: drift should be bounded, not linear. Compare
    # the second half of the run to the first — they should be similar.
    first_half = rel_err[: n_steps // 2].max()
    second_half = rel_err[n_steps // 2 :].max()
    assert second_half < 2.0 * max(first_half, 1e-12), (
        "energy error should be bounded, not secularly growing"
    )
