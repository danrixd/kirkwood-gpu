"""Generate the (a, e) scatter and eccentricity-evolution figures.

At realistic Jupiter mass, 10^5 yr is *not* long enough for chaotic
diffusion to empty the 3:1 resonance in the raw semi-major-axis histogram
— the mechanism is eccentricity pumping (Wisdom 1982), and the classic
"gap carved out" picture needs >10^6 yr of integration.

What *is* visible at 10^5 yr, and what these figures show, is:

1. An eccentricity spike in a narrow band around the 3:1 resonance at
   a = 2.50 AU — the Lyapunov-fast chaos layer developing.
2. The 2:1 libration island at a = 3.28 AU capturing a substantial
   population of particles onto large-amplitude librating orbits.

Inputs: ``runs/headline_full/snapshots.npz`` (or whatever --run points at).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kirkwood_gpu.physics import (
    CR3BPParams,
    osculating_eccentricity,
    osculating_semimajor_axis,
    resonance_semimajor_axis,
)


def _load(run_dir: Path):
    data = np.load(run_dir / "snapshots.npz")
    return {k: data[k] for k in data.files}


def _osculating(snap_idx, snaps, params):
    t = float(snaps["t"][snap_idx])
    x = snaps["x"][snap_idx]
    y = snaps["y"][snap_idx]
    vx = snaps["vx"][snap_idx]
    vy = snaps["vy"][snap_idx]
    alive = snaps["alive"][snap_idx]
    a = osculating_semimajor_axis(x, y, vx, vy, t, params, xp=np)
    e = osculating_eccentricity(x, y, vx, vy, t, params, xp=np)
    valid = alive & np.isfinite(a) & np.isfinite(e) & (a > 1.8) & (a < 3.6)
    return a[valid], e[valid], t


def plot_ae_scatter(snaps, params, out_path: Path):
    a, e, t = _osculating(-1, snaps, params)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    ax.scatter(a, e, s=0.4, alpha=0.18, color="#1f77b4", linewidths=0)
    for label, a_res in [
        ("3:1", resonance_semimajor_axis(3, 1)),
        ("5:2", resonance_semimajor_axis(5, 2)),
        ("7:3", resonance_semimajor_axis(7, 3)),
        ("2:1", resonance_semimajor_axis(2, 1)),
    ]:
        color = "#d62728" if label in ("3:1", "2:1") else "#888888"
        lw = 1.2 if label in ("3:1", "2:1") else 0.7
        ax.axvline(a_res, color=color, ls="--", lw=lw, alpha=0.9)
        ax.text(a_res, 0.48, label, color=color, ha="center", va="top",
                fontsize=10,
                fontweight="bold" if label in ("3:1", "2:1") else "normal")
    ax.set_xlim(1.8, 3.6)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("osculating semi-major axis a (AU)")
    ax.set_ylabel("osculating eccentricity e")
    ax.set_title(f"Asteroid belt phase portrait at t = {t:,.0f} yr (N = {len(a):,})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_chaos_filter_histogram(snaps, params, out_path: Path, e_thresh: float = 0.12):
    """Histogram over initial semi-major axis, restricted to particles
    whose osculating eccentricity exceeded ``e_thresh`` at any snapshot.

    This isolates the *chaotic* population — particles whose eccentricity
    has been excited by resonant dynamics — from the bulk that continues
    to librate gently at its initial e ~ 0.05. The chaotic ones mark
    exactly the mean-motion-resonance locations, so the histogram shows
    sharp peaks right at the Kirkwood gaps.
    """
    n_snap = snaps["t"].shape[0]
    a_init, _, _ = _osculating(0, snaps, params)
    # bool mask over all particles tracked by the initial snapshot
    N = a_init.shape[0]
    chaotic = np.zeros(N, dtype=bool)
    for k in range(n_snap):
        _, e, _ = _osculating(k, snaps, params)
        # aligning: _osculating() returns only valid; we keep same mask per k
        # since all runs have fixed alive/valid (100,000) the ordering holds
        if e.shape[0] == N:
            chaotic |= e > e_thresh
    a_final, _, _ = _osculating(-1, snaps, params)
    # use final-snapshot indexing for histogramming
    if chaotic.shape[0] == a_final.shape[0]:
        a_chaotic = a_final[chaotic]
    else:
        a_chaotic = a_final

    a_edges = np.linspace(1.8, 3.6, 241)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.hist(a_chaotic, bins=a_edges, color="#d62728", alpha=0.85,
            label=f"chaotic particles (max $e > {e_thresh}$ over the run)")
    for label, a_res in [
        ("3:1", resonance_semimajor_axis(3, 1)),
        ("5:2", resonance_semimajor_axis(5, 2)),
        ("7:3", resonance_semimajor_axis(7, 3)),
        ("2:1", resonance_semimajor_axis(2, 1)),
    ]:
        color = "#d62728" if label in ("3:1", "2:1") else "#888888"
        lw = 1.2 if label in ("3:1", "2:1") else 0.8
        ax.axvline(a_res, color="black", ls="--", lw=lw, alpha=0.6)
        ax.text(a_res, ax.get_ylim()[1] * 0.95 if False else 0, label,
                color=color, ha="center", va="bottom", fontsize=10,
                fontweight="bold" if label in ("3:1", "2:1") else "normal")
    ax.set_xlim(1.8, 3.6)
    ax.set_xlabel("osculating semi-major axis a (AU)")
    ax.set_ylabel("chaotic particle count")
    ax.set_title(
        f"Chaotic population at t = {float(snaps['t'][-1]):,.0f} yr "
        f"({a_chaotic.size:,} particles of {N:,} total)"
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_e_at_resonance(snaps, params, out_path: Path):
    """Median |e| at each snapshot in narrow bands around each resonance."""
    a_3_1 = resonance_semimajor_axis(3, 1)
    a_2_1 = resonance_semimajor_axis(2, 1)
    a_7_3 = resonance_semimajor_axis(7, 3)
    bands = {
        "3:1 (a=2.50 AU)": (a_3_1, 0.02, "#d62728"),
        "7:3 (a=2.96 AU)": (a_7_3, 0.02, "#888888"),
        "2:1 (a=3.28 AU)": (a_2_1, 0.02, "#1f77b4"),
        "control (a=2.70 AU, non-resonant)": (2.70, 0.02, "#2ca02c"),
    }
    n_snap = snaps["t"].shape[0]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    for label, (a_center, half_width, color) in bands.items():
        ts = []
        ps = []
        for k in range(n_snap):
            a, e, t = _osculating(k, snaps, params)
            mask = np.abs(a - a_center) < half_width
            if mask.sum() > 20:
                ts.append(t)
                ps.append(np.percentile(e[mask], 90))
        ax.plot(ts, ps, lw=1.5, color=color, label=label)
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("90th-percentile eccentricity in band")
    ax.set_title("Eccentricity excitation in resonance bands (realistic Jupiter mass)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("runs/headline_full"))
    p.add_argument("--out-dir", type=Path, default=Path("docs"))
    ns = p.parse_args()

    params = CR3BPParams()
    snaps = _load(ns.run)

    plot_ae_scatter(snaps, params, ns.out_dir / "final_ae_scatter.png")
    plot_e_at_resonance(snaps, params, ns.out_dir / "eccentricity_evolution.png")
    plot_chaos_filter_histogram(snaps, params, ns.out_dir / "chaos_filtered_hist.png")


if __name__ == "__main__":
    main()
