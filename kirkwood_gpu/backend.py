"""Array-backend resolution.

Exposes a single ``xp`` name that points at CuPy when it is installed AND a
CUDA runtime is reachable, otherwise at NumPy. A single import site keeps the
rest of the package backend-agnostic.

Respects the environment variable ``KIRKWOOD_BACKEND``:

    ``numpy``  -> force the NumPy fallback, even if CuPy is importable.
    ``cupy``   -> force CuPy; raise if it cannot be used.
    unset / anything else -> auto (CuPy when usable, else NumPy).
"""

from __future__ import annotations

import os
import warnings

import numpy as _np


def _try_cupy():
    # On Windows, pip-installed nvidia-* wheels put DLLs in per-package
    # bin dirs that aren't on PATH. Register them before importing cupy.
    from . import _cuda_dll_loader  # noqa: F401

    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None
    # Verify a GPU is actually reachable AND that the runtime libraries
    # needed for elementwise ops are loadable. CuPy imports fine even
    # when the shared libs are missing, so we probe with a trivial kernel.
    try:
        cp.cuda.runtime.getDeviceCount()
        probe = cp.arange(4, dtype=cp.float64)
        _ = float((probe * probe).sum())
    except Exception as exc:  # pragma: no cover - hardware dependent
        warnings.warn(
            f"CuPy imported but GPU kernels fail to execute ({exc!r}); "
            "falling back to NumPy.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return cp


_backend_env = os.environ.get("KIRKWOOD_BACKEND", "").lower()

if _backend_env == "numpy":
    xp = _np
    _BACKEND_NAME = "numpy"
    _on_gpu = False
elif _backend_env == "cupy":
    cp = _try_cupy()
    if cp is None:
        raise RuntimeError("KIRKWOOD_BACKEND=cupy but CuPy / CUDA unavailable")
    xp = cp
    _BACKEND_NAME = "cupy"
    _on_gpu = True
else:
    cp = _try_cupy()
    if cp is not None:
        xp = cp
        _BACKEND_NAME = "cupy"
        _on_gpu = True
    else:
        xp = _np
        _BACKEND_NAME = "numpy"
        _on_gpu = False


def backend_name() -> str:
    """Return 'cupy' or 'numpy' depending on the resolved backend."""
    return _BACKEND_NAME


def on_gpu() -> bool:
    return _on_gpu


def to_numpy(arr):
    """Move ``arr`` to NumPy regardless of backend (for plotting, I/O)."""
    if _BACKEND_NAME == "cupy":
        return xp.asnumpy(arr)
    return _np.asarray(arr)
