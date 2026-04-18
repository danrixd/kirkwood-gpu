"""Combine cpu.json + gpu.json into a markdown benchmark table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pretty_int(n: int) -> str:
    return f"{n:,}"


def _load(path: Path):
    data = json.loads(path.read_text())
    by_n = {r["n_particles"]: r for r in data["results"]}
    return data, by_n


def _merge_cpu(primary: Path, extra: Path | None):
    """Union-merge two CPU bench JSON files (e.g. main + 1M supplement)."""
    data, by_n = _load(primary)
    if extra and extra.exists():
        _, extra_by_n = _load(extra)
        for n, row in extra_by_n.items():
            by_n.setdefault(n, row)
    return data, by_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", type=Path, default=Path("benchmarks/cpu.json"))
    p.add_argument("--cpu-extra", type=Path, default=Path("benchmarks/cpu_1M.json"))
    p.add_argument("--gpu", type=Path, default=Path("benchmarks/gpu.json"))
    p.add_argument("--out", type=Path, default=Path("benchmarks/gpu_vs_cpu.md"))
    ns = p.parse_args()

    cpu_meta, cpu_by_n = _merge_cpu(ns.cpu, ns.cpu_extra)
    gpu_meta, gpu_by_n = _load(ns.gpu)

    all_n = sorted(set(cpu_by_n) | set(gpu_by_n))

    lines = [
        "# GPU vs CPU scaling benchmark",
        "",
        f"Integrator: **{cpu_meta['integrator']}**. "
        f"{cpu_meta['steps']} composite steps per run; each Yoshida4 step does 3 force evaluations.",
        "",
        "| N particles | CPU wall (s) | GPU wall (s) | CPU throughput | GPU throughput | GPU speedup |",
        "|------------:|-------------:|-------------:|---------------:|---------------:|------------:|",
    ]
    for n in all_n:
        c = cpu_by_n.get(n)
        g = gpu_by_n.get(n)
        c_wall = f"{c['wall_s']:.2f}" if c else "—"
        g_wall = f"{g['wall_s']:.2f}" if g else "—"
        c_tp = f"{c['particle_steps_per_sec']:.2e}" if c else "—"
        g_tp = f"{g['particle_steps_per_sec']:.2e}" if g else "—"
        if c and g:
            speedup = g["particle_steps_per_sec"] / c["particle_steps_per_sec"]
            speed = f"{speedup:5.1f}x"
        else:
            speed = "—"
        lines.append(
            f"| {_pretty_int(n):>11} | {c_wall:>12} | {g_wall:>12} | "
            f"{c_tp:>14} | {g_tp:>14} | {speed:>11} |"
        )

    lines += [
        "",
        "Throughput is reported in (particles × composite-steps) per second, so "
        "wall-clock time for a run of length `T` years at step `h` years is",
        "",
        "    wall ≈ N * (T / h) / throughput.",
        "",
        "## Reading the table",
        "",
        "- Below ~10⁴ particles the GPU is **kernel-launch bound**: a handful of "
        "tiny elementwise kernels per step dominate the timeline regardless of N.",
        "- At N ≳ 10⁵ the GPU is memory-bandwidth bound and scales as O(N).",
        "- At N = 10⁶ the RTX 3080 Ti reaches ≈10⁸ particle-steps/s — ~100× the "
        "NumPy CPU path, which is the break-even point for a 10⁵-year run.",
        "",
        "## Hardware / software",
        "",
        f"- CPU run: `{cpu_meta['backend']}` (gpu={cpu_meta['on_gpu']})",
        f"- GPU run: `{gpu_meta['backend']}` (gpu={gpu_meta['on_gpu']})",
    ]

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {ns.out}")


if __name__ == "__main__":
    main()
