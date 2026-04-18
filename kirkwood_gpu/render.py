"""Plotting + animation for the Kirkwood-gap histogram.

Kept matplotlib-only (no external animation dependency beyond Pillow) so the
outputs reproduce across any NumPy install. The GIF is built frame-by-frame
via matplotlib.animation.PillowWriter, which is reliable on Windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.animation import PillowWriter

from .analysis import Snapshot, histogram_semimajor_axes, resonance_overlay
from .physics import CR3BPParams

_BELT_XLIM = (1.8, 3.6)


def _draw_histogram_frame(
    ax,
    snap: Snapshot,
    params: CR3BPParams,
    bins: int,
    y_max: float,
    title_suffix: str = "",
):
    centers, counts = histogram_semimajor_axes(snap, params, bins=bins, a_range=_BELT_XLIM)
    ax.clear()
    ax.fill_between(centers, 0, counts, step="mid", alpha=0.85, color="#1f77b4")
    for label, a_res in resonance_overlay(params.a_J):
        color = "#d62728" if label in ("3:1", "2:1") else "#888888"
        lw = 1.4 if label in ("3:1", "2:1") else 0.9
        ax.axvline(a_res, color=color, ls="--", lw=lw, alpha=0.85)
        ax.text(
            a_res,
            y_max * 0.96,
            label,
            color=color,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold" if label in ("3:1", "2:1") else "normal",
        )
    ax.set_xlim(*_BELT_XLIM)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("osculating semi-major axis a (AU)")
    ax.set_ylabel("particle count")
    ax.set_title(f"Kirkwood belt @ t = {snap.t:,.0f} yr{title_suffix}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))


def write_histogram_gif(
    snapshots: Iterable[Snapshot],
    params: CR3BPParams,
    out_path: Path,
    bins: int = 240,
    fps: int = 12,
    title_suffix: str = "",
):
    """Build a GIF that steps through all snapshots at constant y-axis."""
    snaps = list(snapshots)
    if not snaps:
        raise ValueError("no snapshots given")

    # pick a stable y-axis from the first snapshot (pre-evolution baseline)
    _, c0 = histogram_semimajor_axes(snaps[0], params, bins=bins, a_range=_BELT_XLIM)
    y_max = max(1.0, 1.15 * float(c0.max()))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    writer = PillowWriter(fps=fps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out_path), dpi=120):
        for snap in snaps:
            _draw_histogram_frame(ax, snap, params, bins, y_max, title_suffix)
            writer.grab_frame()
    plt.close(fig)


def write_hero_png(
    snap: Snapshot,
    params: CR3BPParams,
    out_path: Path,
    bins: int = 300,
    title_suffix: str = "",
    dpi: int = 220,
):
    """Final high-DPI histogram with labeled gap locations."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    centers, counts = histogram_semimajor_axes(snap, params, bins=bins, a_range=_BELT_XLIM)
    y_max = max(1.0, 1.15 * float(counts.max()))
    _draw_histogram_frame(ax, snap, params, bins, y_max, title_suffix)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
