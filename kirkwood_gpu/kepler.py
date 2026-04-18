"""Analytic Kepler propagator for the Wisdom-Holman drift step.

Given a test particle at heliocentric state (x, y, vx, vy) relative to a
point mass ``gm`` at the origin, advance it exactly along its Keplerian
orbit by a time interval ``dt``. This is the "drift" half of the
Wisdom-Holman symplectic map: because the Keplerian motion is integrable,
we pay zero discretization error on it, and the step size is limited only
by the perturbation (Jupiter) timescale.

The implementation uses Danby's f/g formulation with Newton iteration on
the change in eccentric anomaly DeltaE. It assumes elliptic orbits (a > 0,
e < 1). Test particles that temporarily go hyperbolic during a close
encounter with Jupiter are the only failure mode; they're flagged dead by
the integration driver so no NaN propagates.

Reference: Danby, J. M. A., *Fundamentals of Celestial Mechanics*, 2nd ed.,
Willmann-Bell (1988), Ch. 6.
"""

from __future__ import annotations

import math

_FUSED_KERNEL = None  # lazy-compiled CuPy ElementwiseKernel


def _get_fused_kernel():
    """Build (once) the fused single-kernel Kepler drift for CuPy.

    Bundles the Newton iteration, f/g construction, and final update into
    one GPU kernel. For N large, this collapses ~80 per-particle kernel
    launches to a single elementwise dispatch — typically 5-10x faster
    than the NumPy-style Python implementation at N = 10^5.
    """
    global _FUSED_KERNEL
    if _FUSED_KERNEL is not None:
        return _FUSED_KERNEL
    import cupy as cp  # type: ignore

    _FUSED_KERNEL = cp.ElementwiseKernel(
        in_params="float64 x0, float64 y0, float64 vx0, float64 vy0, float64 dt, float64 gm",
        out_params="float64 x_new, float64 y_new, float64 vx_new, float64 vy_new, bool valid",
        operation=r"""
        double r0 = sqrt(x0*x0 + y0*y0);
        double v2 = vx0*vx0 + vy0*vy0;
        double inv_a = 2.0/r0 - v2/gm;
        if (inv_a <= 1e-12) {
            valid = false;
            x_new = x0; y_new = y0; vx_new = vx0; vy_new = vy0;
        } else {
            double a = 1.0 / inv_a;
            double sqrt_gm_a = sqrt(gm * a);
            double rdot = (x0*vx0 + y0*vy0) / r0;
            double s0 = r0 * rdot / sqrt_gm_a;
            double c0 = 1.0 - r0 * inv_a;
            double n  = sqrt(gm * inv_a * inv_a * inv_a);
            double dE = n * dt;
            for (int i = 0; i < 8; ++i) {
                double sdE = sin(dE), cdE = cos(dE);
                double F  = dE + s0 * (1.0 - cdE) - c0 * sdE - n * dt;
                double Fp = 1.0 + s0 * sdE - c0 * cdE;
                dE -= F / Fp;
            }
            double sdE = sin(dE), cdE = cos(dE);
            double r  = a * (1.0 - c0 * cdE + s0 * sdE);
            double f  = 1.0 - (a / r0) * (1.0 - cdE);
            double g  = dt - (1.0 / n) * (dE - sdE);
            double fd = -sqrt_gm_a * sdE / (r0 * r);
            double gd = 1.0 - (a / r) * (1.0 - cdE);
            x_new  = f  * x0 + g  * vx0;
            y_new  = f  * y0 + g  * vy0;
            vx_new = fd * x0 + gd * vx0;
            vy_new = fd * y0 + gd * vy0;
            valid = true;
        }
        """,
        name="kepler_drift_fused",
    )
    return _FUSED_KERNEL


def _is_cupy_array(a):
    return type(a).__module__.startswith("cupy")


def kepler_drift(x, y, vx, vy, dt, gm, xp, newton_iters: int = 8):
    """Advance (x, y, vx, vy) by ``dt`` along Kepler orbits of parameter ``gm``.

    Vectorized over the leading (particle) dimension; ``xp`` is NumPy or CuPy.

    When ``xp`` is CuPy, dispatches to a single fused ElementwiseKernel
    that handles Newton + f/g in one GPU launch. On NumPy the code uses
    the readable Python-loop formulation below — clear enough to audit
    against the fused kernel.

    Returns (x_new, y_new, vx_new, vy_new, valid) where ``valid`` is a bool
    mask of particles whose orbit was safely elliptic throughout the step.
    Particles with ``valid == False`` are returned unchanged; the caller
    should flag them as dead so they stop accumulating state.
    """
    if _is_cupy_array(x):
        kernel = _get_fused_kernel()
        # broadcast scalars to arrays of the right dtype for the kernel
        x_new = xp.empty_like(x)
        y_new = xp.empty_like(y)
        vx_new = xp.empty_like(vx)
        vy_new = xp.empty_like(vy)
        valid = xp.empty(x.shape, dtype=xp.bool_)
        kernel(x, y, vx, vy, float(dt), float(gm),
               x_new, y_new, vx_new, vy_new, valid)
        return x_new, y_new, vx_new, vy_new, valid

    r0 = xp.sqrt(x * x + y * y)
    v2 = vx * vx + vy * vy
    inv_a = 2.0 / r0 - v2 / gm  # = 1/a; elliptic iff positive
    # anywhere inv_a <= 0 the orbit is parabolic/hyperbolic (escape)
    valid = inv_a > 1.0e-12
    # clamp so downstream division doesn't NaN for invalid entries; we'll
    # mask the result at the end.
    inv_a_safe = xp.where(valid, inv_a, xp.ones_like(inv_a))
    a = 1.0 / inv_a_safe

    # s0 = e sin E0 = r0 * rdot0 / sqrt(gm*a)
    # c0 = e cos E0 = 1 - r0/a
    sqrt_gm_a = xp.sqrt(gm * a)
    rdot = (x * vx + y * vy) / r0
    s0 = r0 * rdot / sqrt_gm_a
    c0 = 1.0 - r0 * inv_a_safe
    n = xp.sqrt(gm * inv_a_safe * inv_a_safe * inv_a_safe)  # = sqrt(gm / a^3)

    # Newton on F(dE) = dE + s0 (1 - cos dE) - c0 sin dE - n*dt
    dE = n * dt  # initial guess (exact for e = 0)
    for _ in range(newton_iters):
        sin_dE = xp.sin(dE)
        cos_dE = xp.cos(dE)
        F = dE + s0 * (1.0 - cos_dE) - c0 * sin_dE - n * dt
        Fp = 1.0 + s0 * sin_dE - c0 * cos_dE
        dE = dE - F / Fp

    sin_dE = xp.sin(dE)
    cos_dE = xp.cos(dE)

    # new r:  r = a (1 - c0 cos dE + s0 sin dE)
    r = a * (1.0 - c0 * cos_dE + s0 * sin_dE)

    # f / g functions (Danby)
    f = 1.0 - (a / r0) * (1.0 - cos_dE)
    g = dt - (1.0 / n) * (dE - sin_dE)
    f_dot = -sqrt_gm_a * sin_dE / (r0 * r)
    g_dot = 1.0 - (a / r) * (1.0 - cos_dE)

    x_new = f * x + g * vx
    y_new = f * y + g * vy
    vx_new = f_dot * x + g_dot * vx
    vy_new = f_dot * y + g_dot * vy

    # for invalid (hyperbolic) particles, hold state so the caller can flag them
    x_new = xp.where(valid, x_new, x)
    y_new = xp.where(valid, y_new, y)
    vx_new = xp.where(valid, vx_new, vx)
    vy_new = xp.where(valid, vy_new, vy)
    return x_new, y_new, vx_new, vy_new, valid
