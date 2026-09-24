"""End-to-end realtime pipeline.

Wraps a model + sampler and exposes a `step`/`generate` API suited to an
interactive loop where each produced frame should land as fast as possible.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .sampler import Sampler


class RealtimePipeline:
    def __init__(
        self,
        model_fn: Callable[[torch.Tensor, int], torch.Tensor],
        shape: tuple,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        num_sample_steps: int = 8,
    ) -> None:
        self.model_fn = model_fn
        self.shape = shape
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.sampler = Sampler(num_sample_steps=num_sample_steps)

    def _noise(self, generator: Optional[torch.Generator]) -> torch.Tensor:
        return torch.randn(
            self.shape, device=self.device, dtype=self.dtype, generator=generator
        )

    @torch.no_grad()
    def generate(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Produce a single sample from fresh noise."""
        x = self._noise(generator)
        return self.sampler.sample(self.model_fn, x)

    @torch.no_grad()
    def stream(self, num_frames: int, generator: Optional[torch.Generator] = None):
        """Yield successive samples for an interactive/realtime loop."""
        for _ in range(num_frames):
            yield self.generate(generator)
