"""Generate the "why symplectic" figure.

Integrates a single Kepler orbit with three methods at the same step size
and plots |E(t) - E_0| / |E_0| on a log-y axis. The purpose is to make
the backward-error argument in ``docs/derivation.md`` visual: RK4 drifts
linearly, leapfrog and Yoshida4 oscillate with bounded amplitude.

Run:

    python docs/make_drift_figure.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kirkwood_gpu.integrators import (
    kepler_leapfrog_step,
    kepler_rk4_step,
    kepler_yoshida4_step,
)
from kirkwood_gpu.physics import GM_SUN


def _energy(x, y, vx, vy, gm):
    xv, yv, vxv, vyv = float(x[0]), float(y[0]), float(vx[0]), float(vy[0])
    r = math.sqrt(xv * xv + yv * yv)
    return 0.5 * (vxv * vxv + vyv * vyv) - gm / r


def _run(step_fn, a, orbits, steps_per_orbit, gm):
    period = 2.0 * math.pi * math.sqrt(a**3 / gm)
    dt = period / steps_per_orbit
    x = np.array([a])
    y = np.array([0.0])
    vx = np.array([0.0])
    vy = np.array([math.sqrt(gm / a)])
    n_steps = orbits * steps_per_orbit
    times = np.empty(n_steps + 1)
    err = np.empty(n_steps + 1)
    e0 = _energy(x, y, vx, vy, gm)
    times[0] = 0.0
    err[0] = 0.0
    for i in range(1, n_steps + 1):
        x, y, vx, vy = step_fn(x, y, vx, vy, dt, gm)
        times[i] = i * dt
        err[i] = abs(_energy(x, y, vx, vy, gm) - e0) / abs(e0)
    return times / period, err


def main():
    a = 2.5
    orbits = 200
    spo = 100  # 100 steps per orbit — an aggressive but common choice

    t_rk, e_rk = _run(kepler_rk4_step, a, orbits, spo, GM_SUN)
    t_lf, e_lf = _run(kepler_leapfrog_step, a, orbits, spo, GM_SUN)
    t_y4, e_y4 = _run(kepler_yoshida4_step, a, orbits, spo, GM_SUN)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    ax.semilogy(t_rk, np.maximum(e_rk, 1e-18), color="#d62728", lw=1.2,
                label="RK4 (non-symplectic) — linear drift")
    ax.semilogy(t_lf, np.maximum(e_lf, 1e-18), color="#1f77b4", lw=1.0,
                label="Stormer-Verlet leapfrog — bounded")
    ax.semilogy(t_y4, np.maximum(e_y4, 1e-18), color="#2ca02c", lw=1.0,
                label="Yoshida 4 (default) — bounded, smaller")
    ax.set_xlabel("time (orbital periods)")
    ax.set_ylabel(r"$|E(t) - E_0| / |E_0|$")
    ax.set_title(
        f"Energy drift on a Kepler orbit (a = {a} AU, dt = T/{spo})"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(1e-16, 1.0)
    fig.tight_layout()

    out = Path(__file__).parent / "drift_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")
    print(
        f"end-of-run rel errors: RK4={e_rk[-1]:.2e}  leapfrog={e_lf[-1]:.2e}  "
        f"yoshida4={e_y4[-1]:.2e}"
    )


if __name__ == "__main__":
    main()
