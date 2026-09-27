"""End-to-end realtime pipeline.

Wraps a model + sampler and exposes a `generate`/`stream` API suited to an
interactive loop where each produced frame should land as fast as possible.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .sampler import Sampler


DEFAULT_MODEL = "google/ddpm-cifar10-32"
DEFAULT_REVISION = "267b167dc01f0e4e61923ea244e8b988f84deb80"


class RealtimePipeline:
    def __init__(
        self,
        model_fn: Callable[[torch.Tensor, int], torch.Tensor],
        shape: tuple,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        num_sample_steps: int = 8,
        sampler: Optional[Sampler] = None,
    ) -> None:
        self.model_fn = model_fn
        self.shape = shape
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.sampler = (
            sampler
            if sampler is not None
            else Sampler(num_sample_steps=num_sample_steps)
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL,
        *,
        revision: Optional[str] = None,
        batch_size: int = 1,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        num_sample_steps: int = 50,
        local_files_only: bool = False,
    ) -> "RealtimePipeline":
        """Load a pretrained unconditional RGB UNet and its training schedule.

        The default checkpoint generates 32x32 CIFAR-10 images directly in pixel
        space. There is no text encoder or VAE. All sampler updates use our ops.
        Only safetensors weights are loaded; no repository code is executed.
        """
        from diffusers import DDIMScheduler, UNet2DModel

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        revision = revision or (DEFAULT_REVISION if model_id == DEFAULT_MODEL else None)
        load_args = dict(revision=revision, local_files_only=local_files_only)
        scheduler = DDIMScheduler.from_pretrained(model_id, **load_args)
        sampler = Sampler.from_diffusers(scheduler, num_sample_steps)
        model = (
            UNet2DModel.from_pretrained(
                model_id, use_safetensors=True, torch_dtype=dtype, **load_args
            )
            .to(device)
            .eval()
        )
        model.requires_grad_(False)
        if model.config.in_channels != 3 or model.config.out_channels != 3:
            raise ValueError(
                "this example expects an unconditional three-channel pixel-space model"
            )
        if (
            model.config.class_embed_type is not None
            or model.config.num_class_embeds is not None
        ):
            raise ValueError(
                "class-conditioned models are not supported by this example"
            )
        size = model.config.sample_size
        height, width = (size, size) if isinstance(size, int) else size

        # Keep timestep tensors on the GPU instead of copying a scalar each step.
        timesteps = {
            t: torch.tensor(t, device=device, dtype=torch.long)
            for t in sampler.timesteps()
        }

        def model_fn(x, t):
            return model(x, timesteps[t]).sample

        pipeline = cls(
            model_fn, (batch_size, 3, height, width), device, dtype, sampler=sampler
        )
        pipeline.model = model
        pipeline.model_id = model_id
        pipeline.revision = revision
        return pipeline

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
