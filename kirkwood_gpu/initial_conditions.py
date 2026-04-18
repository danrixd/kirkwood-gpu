"""Build an initial belt of test particles for CR3BP integration.

The default is a uniform-in-semi-major-axis belt between a_min and a_max, on
nearly-circular orbits with small Rayleigh-distributed eccentricities and
uniform longitudes. This matches the Wisdom (1982) setup closely enough that
the Kirkwood gaps should carve themselves out of an initially smooth
distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .physics import GM_SUN, CR3BPParams


@dataclass(frozen=True)
class BeltConfig:
    n_particles: int
    a_min: float = 2.0
    a_max: float = 3.5
    e_rms: float = 0.05  # Rayleigh-distributed eccentricities
    seed: int = 42


def build_initial_belt(cfg: BeltConfig, params: CR3BPParams):
    """Return four numpy arrays (x, y, vx, vy) of length cfg.n_particles.

    The particles are drawn heliocentrically then shifted into the
    barycentric inertial frame so they are consistent with the Sun's
    barycentric position at t=0. This eliminates a spurious O(mu*a_J)
    offset that would otherwise show up as a bulk wobble.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_particles

    # uniform in a, uniform in mean anomaly proxy, Rayleigh e, uniform omega
    a = rng.uniform(cfg.a_min, cfg.a_max, size=n)
    e = rng.rayleigh(scale=cfg.e_rms, size=n)
    # clip to avoid hyperbolic / grazing orbits at the tails
    e = np.clip(e, 0.0, 0.3)
    f = rng.uniform(0.0, 2.0 * math.pi, size=n)  # true anomaly
    omega = rng.uniform(0.0, 2.0 * math.pi, size=n)  # argument of perihelion

    # heliocentric polar -> cartesian, planar
    r = a * (1.0 - e * e) / (1.0 + e * np.cos(f))
    theta = f + omega
    x_h = r * np.cos(theta)
    y_h = r * np.sin(theta)

    # circular-orbit velocity magnitude at r for given a, e
    # vis-viva: v = sqrt(GM (2/r - 1/a))
    v = np.sqrt(GM_SUN * (2.0 / r - 1.0 / a))
    # velocity direction: perpendicular to r (in osculating plane), with
    # sense set by angular momentum. For a Keplerian orbit at true anomaly f:
    #   phi = angle of velocity = theta + pi/2 - gamma
    # where gamma is the flight-path angle, tan(gamma) = e sin(f)/(1 + e cos(f))
    flight_path = np.arctan2(e * np.sin(f), 1.0 + e * np.cos(f))
    vphi = theta + 0.5 * math.pi - flight_path
    vx_h = v * np.cos(vphi)
    vy_h = v * np.sin(vphi)

    # shift from heliocentric to barycentric inertial frame
    # (Sun is at -mu*a_J * (cos 0, sin 0) = (-mu a_J, 0) at t=0,
    #  moving at velocity mu*a_J*n * (sin 0, -cos 0) = (0, -mu a_J n))
    mu = params.mu
    a_J = params.a_J
    n_mot = params.mean_motion
    x_sun = -mu * a_J
    y_sun = 0.0
    vx_sun = 0.0
    vy_sun = -mu * a_J * n_mot
    # wait: derivative of -mu a_J cos(n t) at t=0 is +mu a_J n sin(0) = 0;
    # derivative of -mu a_J sin(n t) at t=0 is -mu a_J n cos(0) = -mu a_J n
    # so vy_sun = -mu a_J n. Correct.

    x = x_h + x_sun
    y = y_h + y_sun
    vx = vx_h + vx_sun
    vy = vy_h + vy_sun
    return x, y, vx, vy
