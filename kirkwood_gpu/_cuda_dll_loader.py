"""Windows DLL-directory glue for pip-installed NVIDIA CUDA wheels.

When the user installs CuPy via ``pip`` together with the
``nvidia-cuda-*`` wheels (nvidia-cuda-nvrtc-cu11, nvidia-curand-cu11, ...),
the DLLs land in per-package ``bin/`` directories that are NOT on PATH.
CuPy then fails at import-time of cuBLAS / cuRAND / NVRTC / ... Windows
11 requires ``os.add_dll_directory()`` for every such directory before
the first load.

Import this module **before** ``import cupy``. It is a no-op on non-Windows
platforms and on environments where the nvidia-* wheels aren't present
(e.g. a system CUDA Toolkit install that already sits on PATH).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_SUBDIRS = (
    "cuda_nvrtc",
    "cuda_runtime",
    "cuda_cupti",
    "curand",
    "cublas",
    "cusparse",
    "cusolver",
    "cufft",
    "nvjitlink",
)


def ensure_nvidia_dlls_on_path() -> list[str]:
    """Add nvidia-*/bin directories to the Windows DLL search path.

    Returns the list of directories added (empty on non-Windows or when
    the nvidia-* wheels aren't installed).
    """
    if os.name != "nt":
        return []
    added: list[str] = []
    for sp in [Path(p) for p in sys.path if "site-packages" in p]:
        base = sp / "nvidia"
        if not base.is_dir():
            continue
        for sub in _SUBDIRS:
            bin_dir = base / sub / "bin"
            if bin_dir.is_dir():
                try:
                    os.add_dll_directory(str(bin_dir))  # type: ignore[attr-defined]
                except (FileNotFoundError, OSError):
                    continue
                added.append(str(bin_dir))
                # also prepend to PATH for subprocesses / child loaders
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    return added


_ADDED = ensure_nvidia_dlls_on_path()
