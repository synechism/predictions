"""Pure-PyTorch reference implementations of every custom op.

These are the source of truth for correctness. Each CUDA kernel must match its
reference here within tolerance (see tests/test_parity.py). They are also the
CPU fallback used when the compiled extension is unavailable.

Keep these simple and obviously-correct — do not optimize them. Their job is to
be right, not fast.
"""
from __future__ import annotations

import torch


def denoise_step(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    coef_x: float,
    coef_eps: float,
) -> torch.Tensor:
    """x_prev = coef_x * x_t + coef_eps * eps."""
    return coef_x * x_t + coef_eps * eps
