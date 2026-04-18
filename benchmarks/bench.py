"""CPU (NumPy) vs GPU (CuPy) throughput benchmark.

Integrates a fixed number of leapfrog steps at several particle counts,
reports wall time and particle-steps/second. Run once per backend:

    KIRKWOOD_BACKEND=numpy python benchmarks/bench.py --out benchmarks/cpu.json
    KIRKWOOD_BACKEND=cupy  python benchmarks/bench.py --out benchmarks/gpu.json

then combine with render_table.py to generate gpu_vs_cpu.md.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from kirkwood_gpu.backend import backend_name, on_gpu, to_numpy, xp
from kirkwood_gpu.initial_conditions import BeltConfig, build_initial_belt
from kirkwood_gpu.integrators import (
    State,
    leapfrog_step,
    wisdom_holman_step,
    yoshida4_step,
)
from kirkwood_gpu.physics import CR3BPParams


def _sync():
    if backend_name() == "cupy":
        xp.cuda.Stream.null.synchronize()


def time_run(n_particles: int, n_steps: int, integrator: str, seed: int):
    params = CR3BPParams()
    cfg = BeltConfig(n_particles=n_particles, seed=seed)
    x0, y0, vx0, vy0 = build_initial_belt(cfg, params)
    x = xp.asarray(x0)
    y = xp.asarray(y0)
    vx = xp.asarray(vx0)
    vy = xp.asarray(vy0)
    alive = xp.ones(n_particles, dtype=bool)
    state = State(x, y, vx, vy, 0.0, alive)
    step = {
        "leapfrog": leapfrog_step,
        "yoshida4": yoshida4_step,
        "wisdom_holman": wisdom_holman_step,
    }[integrator]
    dt = params.jupiter_period / 200.0

    # warm up (first kernel launches, driver init, etc.)
    state = step(state, dt, params)
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = step(state, dt, params)
    _sync()
    wall = time.perf_counter() - t0

    # integrity check: keep a small state reduction so the compiler
    # can't elide the loop
    s = float(to_numpy(state.x).sum())
    return {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "integrator": integrator,
        "wall_s": wall,
        "particle_steps_per_sec": n_particles * n_steps / wall,
        "checksum": s,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--particle-counts",
        type=int,
        nargs="+",
        default=[1_000, 10_000, 100_000, 1_000_000],
    )
    p.add_argument("--steps", type=int, default=500)
    p.add_argument(
        "--integrator",
        default="yoshida4",
        choices=["leapfrog", "yoshida4", "wisdom_holman"],
    )
    p.add_argument("--seed", type=int, default=42)
    ns = p.parse_args()

    print(f"backend: {backend_name()} (gpu={on_gpu()})")
    print(f"integrator: {ns.integrator}   steps: {ns.steps}")
    results = []
    for n in ns.particle_counts:
        try:
            r = time_run(n, ns.steps, ns.integrator, ns.seed)
        except Exception as exc:
            print(f"  N={n:>10,d}   FAILED: {exc!r}")
            continue
        print(
            f"  N={n:>10,d}   wall={r['wall_s']:7.3f} s   "
            f"throughput={r['particle_steps_per_sec']:.3e} part-steps/s"
        )
        results.append(r)

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(
        json.dumps(
            {
                "backend": backend_name(),
                "on_gpu": on_gpu(),
                "integrator": ns.integrator,
                "steps": ns.steps,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"wrote {ns.out}")


if __name__ == "__main__":
    main()
