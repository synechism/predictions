"""Pure-PyTorch reference implementations of every custom op.

These are the source of truth for correctness. Each CUDA kernel must match its
reference here within tolerance (see tests/test_parity.py). They are also the
CPU fallback used when the compiled extension is unavailable.

Keep these simple and obviously-correct — do not optimize them. Their job is to
be right, not fast.
"""

from __future__ import annotations

import torch


def _math_inputs(x: torch.Tensor, eps: torch.Tensor):
    # Match CUDA's opmath precision: fp32 for half/bfloat16, otherwise unchanged.
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float(), eps.float()
    return x, eps


def denoise_step(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    coef_x: float,
    coef_eps: float,
) -> torch.Tensor:
    """x_prev = coef_x * x_t + coef_eps * eps."""
    x, noise = _math_inputs(x_t, eps)
    return (coef_x * x + coef_eps * noise).to(x_t.dtype)


def clipped_ddim_step(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    alpha_t: float,
    alpha_prev: float,
    clip_range: float = 1.0,
) -> torch.Tensor:
    """DDIM eta=0, epsilon prediction, with clipped reconstructed clean image."""
    x, noise = _math_inputs(x_t, eps)
    x0 = (x - (1.0 - alpha_t) ** 0.5 * noise) / alpha_t**0.5
    x0 = x0.clamp(-clip_range, clip_range)
    return (alpha_prev**0.5 * x0 + (1.0 - alpha_prev) ** 0.5 * noise).to(x_t.dtype)
