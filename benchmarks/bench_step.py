"""Per-step latency / throughput for the denoise_step op.

    python benchmarks/bench_step.py

Reports the compiled-kernel path on CUDA and the reference path otherwise, so
you can watch the gap close as kernels get optimized.
"""
import time

import torch

from predictions import _ops


def bench(shape=(1, 4, 128, 128), iters=200, warmup=20, dtype=torch.float16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(shape, device=device, dtype=dtype)
    eps = torch.randn_like(x)

    def run():
        return _ops.denoise_step(x, eps, 0.9, -0.1)

    for _ in range(warmup):
        run()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    print(f"backend      : {_ops.backend()}")
    print(f"device       : {device}  dtype={dtype}")
    print(f"shape        : {tuple(shape)}")
    print(f"per-step     : {dt * 1e6:8.2f} us")
    print(f"throughput   : {x.numel() / dt / 1e9:8.2f} Gelem/s")


if __name__ == "__main__":
    bench()
