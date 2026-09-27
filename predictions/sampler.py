"""Sampling loop and scheduler math.

The scheduler produces the scalar coefficients for each step on the host; the
per-step tensor update is delegated to the custom kernel via `denoise_step`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from ._ops import clipped_ddim_step, denoise_step


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

    sqrt_a_t = a_t**0.5
    sqrt_a_prev = a_prev**0.5

    coef_x = sqrt_a_prev / sqrt_a_t
    coef_eps = ((1.0 - a_prev) ** 0.5) - coef_x * ((1.0 - a_t) ** 0.5)
    return coef_x, coef_eps


@dataclass
class Sampler:
    """Minimal DDIM-style sampler.

    `model_fn(x, t)` must return the predicted noise eps for sample `x` at
    integer timestep `t`. Supports deterministic eta=0, leading timestep
    spacing, and optional x0 clipping. Half/bfloat16 sampler math uses fp32.
    """

    num_train_steps: int = 1000
    num_sample_steps: int = 50
    clip_sample: bool = False
    clip_sample_range: float = 1.0
    alphas_cumprod: Optional[torch.Tensor] = None
    final_alpha_cumprod: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.num_train_steps, int) or self.num_train_steps < 1:
            raise ValueError("num_train_steps must be a positive integer")
        if (
            not isinstance(self.num_sample_steps, int)
            or not 1 <= self.num_sample_steps <= self.num_train_steps
        ):
            raise ValueError("num_sample_steps must be between 1 and num_train_steps")
        if self.clip_sample_range <= 0 or not torch.isfinite(
            torch.tensor(self.clip_sample_range)
        ):
            raise ValueError("clip_sample_range must be finite and positive")
        if self.alphas_cumprod is None:
            betas = linear_beta_schedule(self.num_train_steps)
            self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        else:
            self.alphas_cumprod = self.alphas_cumprod.detach().cpu().float().clone()
        if self.alphas_cumprod.shape != (self.num_train_steps,):
            raise ValueError(
                "alphas_cumprod must contain one value per training timestep"
            )
        if not ((self.alphas_cumprod > 0) & (self.alphas_cumprod <= 1)).all():
            raise ValueError("alphas_cumprod must be in (0, 1]")
        if not 0 < self.final_alpha_cumprod <= 1:
            raise ValueError("final_alpha_cumprod must be in (0, 1]")

        # Prepare CPU scalar coefficients once, outside the repeated frame loop.
        ts = self.timesteps()
        self._updates = []
        for i, t in enumerate(ts):
            a_t = float(self.alphas_cumprod[t])
            a_prev = (
                float(self.alphas_cumprod[ts[i + 1]])
                if i + 1 < len(ts)
                else self.final_alpha_cumprod
            )
            if self.clip_sample:
                args = (a_t, a_prev, self.clip_sample_range)
            else:
                coef_x = (a_prev / a_t) ** 0.5
                args = (coef_x, (1.0 - a_prev) ** 0.5 - coef_x * (1.0 - a_t) ** 0.5)
            self._updates.append((t, args))

    @classmethod
    def from_diffusers(cls, scheduler, num_sample_steps: int) -> "Sampler":
        """Import a DDIM schedule; this implementation supports epsilon, eta=0.

        Deliberately reject features the custom kernels do not implement.
        Diffusers supplies the trained alpha schedule, not a guessed replacement.
        """
        if scheduler.__class__.__name__ != "DDIMScheduler":
            raise ValueError("expected a DDIMScheduler")
        config = scheduler.config
        if config.prediction_type != "epsilon" or config.thresholding:
            raise ValueError(
                "only epsilon prediction without dynamic thresholding is supported"
            )
        if config.timestep_spacing != "leading" or config.steps_offset != 0:
            raise ValueError(
                "only leading timestep spacing with steps_offset=0 is supported"
            )
        return cls(
            num_train_steps=config.num_train_timesteps,
            num_sample_steps=num_sample_steps,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            alphas_cumprod=scheduler.alphas_cumprod,
            final_alpha_cumprod=float(scheduler.final_alpha_cumprod),
        )

    def timesteps(self) -> List[int]:
        step = self.num_train_steps // self.num_sample_steps
        # Match DDIMScheduler's "leading" spacing, including non-divisors.
        return [i * step for i in reversed(range(self.num_sample_steps))]

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable[[torch.Tensor, int], torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        update = clipped_ddim_step if self.clip_sample else denoise_step
        for t, args in self._updates:
            with torch.profiler.record_function("predictions::denoiser"):
                eps = model_fn(x, t)
            with torch.profiler.record_function("predictions::sampler_update"):
                x = update(x, eps, *args)
        return x
