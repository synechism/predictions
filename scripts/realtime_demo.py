"""Minimal interactive/realtime sampling loop.

Uses a dummy model so the skeleton runs end-to-end without weights. Swap
`dummy_model` for a real denoiser (UNet / DiT) to get actual samples.

    python scripts/realtime_demo.py --frames 30
"""
import argparse
import time

import torch

from predictions import RealtimePipeline, extension_available


def dummy_model(x, t):
    # Stand-in denoiser: predicts a small, timestep-dependent perturbation.
    return 0.01 * t * torch.ones_like(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = RealtimePipeline(
        model_fn=dummy_model,
        shape=(1, 4, args.size, args.size),
        device=device,
        dtype=dtype,
        num_sample_steps=args.steps,
    )

    print(f"device={device} kernels={'compiled' if extension_available() else 'reference'}")

    t0 = time.perf_counter()
    n = 0
    for frame in pipe.stream(args.frames):
        n += 1
    dt = time.perf_counter() - t0

    print(f"generated {n} frames in {dt:.3f}s  ->  {n / dt:.1f} fps")


if __name__ == "__main__":
    main()
