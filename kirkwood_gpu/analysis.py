"""Post-processing: osculating elements and semi-major-axis histograms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import (
    CR3BPParams,
    osculating_eccentricity,
    osculating_semimajor_axis,
    resonance_semimajor_axis,
)


@dataclass
class Snapshot:
    """A single time snapshot of particle state (all numpy, host memory)."""

    t: float
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    alive: np.ndarray  # bool


def histogram_semimajor_axes(
    snap: Snapshot,
    params: CR3BPParams,
    bins: int = 240,
    a_range: tuple[float, float] = (1.8, 3.6),
):
    """Return (bin_centers, counts) for osculating a of alive particles."""
    a = osculating_semimajor_axis(
        snap.x, snap.y, snap.vx, snap.vy, snap.t, params, xp=np
    )
    valid = snap.alive & np.isfinite(a) & (a > a_range[0]) & (a < a_range[1])
    counts, edges = np.histogram(a[valid], bins=bins, range=a_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def resonance_overlay(a_J: float) -> list[tuple[str, float]]:
    """Labeled gap / resonance positions used in plots."""
    return [
        ("3:1", resonance_semimajor_axis(3, 1, a_J)),
        ("5:2", resonance_semimajor_axis(5, 2, a_J)),
        ("7:3", resonance_semimajor_axis(7, 3, a_J)),
        ("2:1", resonance_semimajor_axis(2, 1, a_J)),
    ]


def summary_stats(snap: Snapshot, params: CR3BPParams) -> dict:
    """Quick diagnostics for a snapshot."""
    a = osculating_semimajor_axis(snap.x, snap.y, snap.vx, snap.vy, snap.t, params, xp=np)
    e = osculating_eccentricity(snap.x, snap.y, snap.vx, snap.vy, snap.t, params, xp=np)
    valid = snap.alive & np.isfinite(a) & np.isfinite(e)
    return {
        "t_years": snap.t,
        "n_alive": int(snap.alive.sum()),
        "n_valid": int(valid.sum()),
        "a_median": float(np.median(a[valid])) if valid.any() else float("nan"),
        "e_median": float(np.median(e[valid])) if valid.any() else float("nan"),
    }
