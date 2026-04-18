"""CLI entry point.

    python -m kirkwood_gpu.run --particles 100000 --years 100000 --seed 42

Runs a full Kirkwood-gap integration from an initial belt, tracks Jacobi
constant drift for a diagnostic subset of particles, periodically snapshots
to disk, and emits the headline GIF + hero PNG.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import Snapshot, summary_stats
from .backend import backend_name, on_gpu, to_numpy, xp
from .initial_conditions import BeltConfig, build_initial_belt
from .integrators import State, leapfrog_step, wisdom_holman_step, yoshida4_step
from .physics import CR3BPParams, jacobi_constant
from .render import write_hero_png, write_histogram_gif


def _log(msg: str):
    print(f"[kirkwood-gpu] {msg}", flush=True)


@dataclass
class RunConfig:
    particles: int
    years: float
    seed: int
    out_dir: Path
    steps_per_orbit: int = 200
    n_snapshots: int = 60
    mass_ratio_scale: float = 1.0
    bins: int = 260
    make_gif: bool = True
    integrator: str = "yoshida4"


def _parse_args(argv=None) -> RunConfig:
    p = argparse.ArgumentParser(
        prog="kirkwood_gpu.run",
        description="Simulate Kirkwood gap formation via CR3BP on CPU or GPU.",
    )
    p.add_argument("--particles", type=int, default=100_000)
    p.add_argument("--years", type=float, default=100_000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("runs/latest"))
    p.add_argument(
        "--steps-per-orbit",
        type=int,
        default=200,
        help="Leapfrog steps per Jupiter orbital period",
    )
    p.add_argument(
        "--snapshots",
        type=int,
        default=60,
        help="Number of snapshots captured uniformly across the run",
    )
    p.add_argument(
        "--mass-scale",
        type=float,
        default=1.0,
        help="Scale factor on Jupiter mass; keep 1.0 for realistic science",
    )
    p.add_argument("--bins", type=int, default=260)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument(
        "--integrator",
        choices=["leapfrog", "yoshida4", "wisdom_holman"],
        default="yoshida4",
        help=(
            "2nd-order KDK leapfrog, 4th-order Yoshida composition, or "
            "Wisdom-Holman map (analytic Kepler drift + Jupiter kick; "
            "permits ~10x larger dt than yoshida4)"
        ),
    )
    p.add_argument("--version", action="version", version=f"kirkwood-gpu {__version__}")
    ns = p.parse_args(argv)
    return RunConfig(
        particles=ns.particles,
        years=ns.years,
        seed=ns.seed,
        out_dir=ns.out,
        steps_per_orbit=ns.steps_per_orbit,
        n_snapshots=ns.snapshots,
        mass_ratio_scale=ns.mass_scale,
        bins=ns.bins,
        make_gif=not ns.no_gif,
        integrator=ns.integrator,
    )


def _snapshot_from_state(state: State) -> Snapshot:
    return Snapshot(
        t=float(state.t),
        x=to_numpy(state.x).copy(),
        y=to_numpy(state.y).copy(),
        vx=to_numpy(state.vx).copy(),
        vy=to_numpy(state.vy).copy(),
        alive=to_numpy(state.alive).copy(),
    )


def run(cfg: RunConfig):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    params = CR3BPParams(mass_ratio_scale=cfg.mass_ratio_scale)
    T_J = params.jupiter_period
    dt = T_J / cfg.steps_per_orbit
    n_total_steps = int(math.ceil(cfg.years / dt))
    # round up so we hit at least cfg.years
    step_fn = {
        "leapfrog": leapfrog_step,
        "yoshida4": yoshida4_step,
        "wisdom_holman": wisdom_holman_step,
    }[cfg.integrator]
    _log(
        f"backend={backend_name()}  particles={cfg.particles:,}  years={cfg.years:,.0f}"
        f"  dt={dt:.4f} yr  steps={n_total_steps:,}  integrator={cfg.integrator}"
        f"  mass_scale={cfg.mass_ratio_scale}"
    )

    belt = BeltConfig(n_particles=cfg.particles, seed=cfg.seed)
    x0, y0, vx0, vy0 = build_initial_belt(belt, params)

    # push to backend
    x = xp.asarray(x0)
    y = xp.asarray(y0)
    vx = xp.asarray(vx0)
    vy = xp.asarray(vy0)
    alive = xp.ones(cfg.particles, dtype=bool)
    state = State(x, y, vx, vy, 0.0, alive)

    # snapshot cadence: aim for n_snapshots evenly across the run
    snap_every = max(1, n_total_steps // cfg.n_snapshots)
    snapshots: list[Snapshot] = [_snapshot_from_state(state)]
    _log(f"snapshot every {snap_every:,} steps ({snap_every * dt:,.1f} yr)")

    # Jacobi diagnostic on a deterministic sample of particles
    n_diag = min(256, cfg.particles)
    diag_idx = np.linspace(0, cfg.particles - 1, n_diag, dtype=np.int64)
    diag_idx_xp = xp.asarray(diag_idx)
    c0 = to_numpy(
        jacobi_constant(
            state.x[diag_idx_xp],
            state.y[diag_idx_xp],
            state.vx[diag_idx_xp],
            state.vy[diag_idx_xp],
            state.t,
            params,
            xp=xp,
        )
    ).copy()
    max_rel_dC = 0.0
    median_rel_dC = 0.0
    p95_rel_dC = 0.0

    t0 = time.perf_counter()
    for i in range(1, n_total_steps + 1):
        state = step_fn(state, dt, params)
        if i % snap_every == 0 or i == n_total_steps:
            # housekeeping: flag escapers/colliders
            from .integrators import integrate  # noqa: F401 (silence unused)
            r2 = state.x * state.x + state.y * state.y
            from .physics import primary_positions

            (xs, ys), (xj, yj) = primary_positions(state.t, params, xp=xp)
            r2_s = (state.x - xs) ** 2 + (state.y - ys) ** 2
            r2_j = (state.x - xj) ** 2 + (state.y - yj) ** 2
            alive = state.alive & (r2 < 100.0**2)
            alive = alive & (r2_s > 0.005**2)
            alive = alive & (r2_j > 0.0005**2)
            state = State(state.x, state.y, state.vx, state.vy, state.t, alive)

            snap = _snapshot_from_state(state)
            snapshots.append(snap)

            # update Jacobi diagnostic
            cN = to_numpy(
                jacobi_constant(
                    state.x[diag_idx_xp],
                    state.y[diag_idx_xp],
                    state.vx[diag_idx_xp],
                    state.vy[diag_idx_xp],
                    state.t,
                    params,
                    xp=xp,
                )
            )
            alive_diag = to_numpy(state.alive[diag_idx_xp])
            rel_dC = np.abs(cN - c0) / np.abs(c0)
            rel_dC = rel_dC[alive_diag]
            if rel_dC.size:
                max_rel_dC = max(max_rel_dC, float(rel_dC.max()))
                median_rel_dC = max(median_rel_dC, float(np.median(rel_dC)))
                p95_rel_dC = max(p95_rel_dC, float(np.percentile(rel_dC, 95)))

            n_alive = int(to_numpy(state.alive).sum())
            elapsed = time.perf_counter() - t0
            _log(
                f"  step {i:>10,}/{n_total_steps:,}  t={state.t:>9,.1f} yr  "
                f"alive={n_alive:>9,}  |dC/C| med={median_rel_dC:.1e} "
                f"p95={p95_rel_dC:.1e} max={max_rel_dC:.1e}  "
                f"wall={elapsed:6.1f}s"
            )

    total_wall = time.perf_counter() - t0
    _log(f"integration complete in {total_wall:.1f} s")
    _log(
        f"|dC/C| diagnostic (256 particles): median={median_rel_dC:.2e}  "
        f"p95={p95_rel_dC:.2e}  max={max_rel_dC:.2e}"
    )

    # persist
    np.savez_compressed(
        out / "snapshots.npz",
        t=np.array([s.t for s in snapshots]),
        x=np.stack([s.x for s in snapshots]),
        y=np.stack([s.y for s in snapshots]),
        vx=np.stack([s.vx for s in snapshots]),
        vy=np.stack([s.vy for s in snapshots]),
        alive=np.stack([s.alive for s in snapshots]),
    )

    suffix = (
        f" (N={cfg.particles:,}, {backend_name()}"
        + (", GPU" if on_gpu() else "")
        + f", m_J x{cfg.mass_ratio_scale:g})"
        if cfg.mass_ratio_scale != 1.0
        else f" (N={cfg.particles:,}, {backend_name()})"
    )

    write_hero_png(snapshots[-1], params, out / "hero.png", bins=cfg.bins, title_suffix=suffix)
    if cfg.make_gif:
        write_histogram_gif(
            snapshots, params, out / "kirkwood.gif", bins=cfg.bins, title_suffix=suffix
        )

    # final stats
    stats = summary_stats(snapshots[-1], params)
    _log(f"final: {stats}")
    # also drop a small run-summary markdown
    (out / "run_summary.md").write_text(
        "# run summary\n"
        f"- backend: {backend_name()} (gpu={on_gpu()})\n"
        f"- particles: {cfg.particles:,}\n"
        f"- years: {cfg.years:,.0f}\n"
        f"- dt: {dt:.6f} yr ({cfg.steps_per_orbit} steps/Jupiter orbit)\n"
        f"- integrator: {cfg.integrator}\n"
        f"- total steps: {n_total_steps:,}\n"
        f"- wall time: {total_wall:.1f} s\n"
        f"- |dC/C| diagnostic (256 particles): "
        f"median {median_rel_dC:.2e}, p95 {p95_rel_dC:.2e}, max {max_rel_dC:.2e}\n"
        f"- mass-ratio scale: {cfg.mass_ratio_scale}\n"
        f"- alive at end: {stats['n_alive']:,}\n"
    )
    return snapshots, {
        "wall_time_s": total_wall,
        "max_rel_dC": max_rel_dC,
        "n_steps": n_total_steps,
    }


def main(argv=None):
    cfg = _parse_args(argv)
    return run(cfg)


if __name__ == "__main__":
    main()
