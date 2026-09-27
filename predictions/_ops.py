"""Loads the compiled CUDA extension and exposes the custom ops.

Resolution order for each op:
  1. Prebuilt extension `predictions._C` (from `pip install -e .`).
  2. JIT build via `torch.utils.cpp_extension.load` on first import.
  3. Pure-PyTorch reference (when CUDA / a compiler is unavailable).

This keeps the package importable everywhere — the reference fallback lets you
develop and test the sampling logic on CPU-only machines, while the real kernels
kick in automatically on a CUDA box.
"""

from __future__ import annotations

import os
import warnings

import torch

from . import reference

_EXT = None
_EXT_SOURCE = "reference"


def _try_prebuilt():
    try:
        from . import _C  # type: ignore

        return _C, "prebuilt"
    except Exception:
        return None, None


def _try_jit():
    if not torch.cuda.is_available():
        return None, None
    try:
        from torch.utils.cpp_extension import load

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csrc = os.path.join(here, "csrc")
        ext = load(
            name="predictions_C_jit",
            sources=[
                os.path.join(csrc, "bindings.cpp"),
                os.path.join(csrc, "kernels", "denoise_step.cu"),
            ],
            extra_include_paths=[os.path.join(csrc, "include")],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        return ext, "jit"
    except Exception as exc:  # pragma: no cover - depends on toolchain
        warnings.warn(f"JIT build of predictions kernels failed: {exc}")
        return None, None


def _load():
    global _EXT, _EXT_SOURCE
    if _EXT is not None:
        return
    for loader in (_try_prebuilt, _try_jit):
        ext, source = loader()
        if ext is not None:
            _EXT, _EXT_SOURCE = ext, source
            return


_load()


def extension_available() -> bool:
    """True if the compiled CUDA kernels are in use (not the reference)."""
    return _EXT is not None


def backend() -> str:
    """One of 'prebuilt', 'jit', or 'reference'."""
    return _EXT_SOURCE


def denoise_step(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    coef_x: float,
    coef_eps: float,
) -> torch.Tensor:
    """Fused one-step denoiser update.

    x_prev = coef_x * x_t + coef_eps * eps
    """
    if _EXT is not None and x_t.is_cuda:
        return _EXT.denoise_step(x_t, eps, float(coef_x), float(coef_eps))
    return reference.denoise_step(x_t, eps, coef_x, coef_eps)


def clipped_ddim_step(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    alpha_t: float,
    alpha_prev: float,
    clip_range: float = 1.0,
) -> torch.Tensor:
    if _EXT is not None and x_t.is_cuda:
        return _EXT.clipped_ddim_step(x_t, eps, alpha_t, alpha_prev, clip_range)
    return reference.clipped_ddim_step(x_t, eps, alpha_t, alpha_prev, clip_range)
