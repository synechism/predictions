"""Sampling loop and scheduler math.

The scheduler produces the scalar coefficients for each step on the host; the
per-step tensor update is delegated to the custom kernel via `denoise_step`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

from ._ops import denoise_step


def linear_beta_schedule(
    num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


def ddim_coefficients(
    alphas_cumprod: torch.Tensor, t: int, t_prev: int
) -> Tuple[float, float]:
    """DDIM (eta=0) coefficients mapping (x_t, eps) -> x_{t_prev}.

    Returns (coef_x, coef_eps) such that
        x_prev = coef_x * x_t + coef_eps * eps
    """
    a_t = float(alphas_cumprod[t])
    a_prev = float(alphas_cumprod[t_prev]) if t_prev >= 0 else 1.0

    sqrt_a_t = a_t ** 0.5
    sqrt_a_prev = a_prev ** 0.5

    coef_x = sqrt_a_prev / sqrt_a_t
    coef_eps = ((1.0 - a_prev) ** 0.5) - coef_x * ((1.0 - a_t) ** 0.5)
    return coef_x, coef_eps


@dataclass
class Sampler:
    """Minimal DDIM-style sampler.

    `model_fn(x, t)` must return the predicted noise eps for sample `x` at
    integer timestep `t`.
    """

    num_train_steps: int = 1000
    num_sample_steps: int = 50

    def __post_init__(self) -> None:
        betas = linear_beta_schedule(self.num_train_steps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def timesteps(self) -> List[int]:
        step = self.num_train_steps // self.num_sample_steps
        return list(range(self.num_train_steps - 1, -1, -step))

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable[[torch.Tensor, int], torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        ts = self.timesteps()
        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            eps = model_fn(x, t)
            coef_x, coef_eps = ddim_coefficients(self.alphas_cumprod, t, t_prev)
            x = denoise_step(x, eps, coef_x, coef_eps)
        return x
