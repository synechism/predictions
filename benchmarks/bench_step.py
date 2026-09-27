"""Compare identical math in eager PyTorch, torch.compile, and custom CUDA.

Run: python -m benchmarks.bench_step
Wall time includes Python dispatch and allocation. CUDA graph timing amortizes
host submission across 50 calls to expose the device-side cost of the ops.
Compilation, graph capture, and warmup are excluded from both measurements.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from predictions import _ops, reference


@torch.inference_mode()
def measure(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    wall_samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        wall_samples.append((time.perf_counter() - start) * 1e6 / iters)

    # Retain the final output so its graph-owned allocation stays alive.
    graph = torch.cuda.CUDAGraph()
    calls_per_graph = 50
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            output = fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    graph_samples = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        graph_samples.append(start.elapsed_time(end) * 1000 / (iters * calls_per_graph))
    return statistics.median(wall_samples), statistics.median(graph_samples)


@torch.inference_mode()
def bench(shapes, dtype, iters=200, warmup=30):
    if not torch.cuda.is_available() or not _ops.extension_available():
        raise RuntimeError(
            "This benchmark requires CUDA and the compiled extension; no fallback is timed."
        )
    rows = []
    for name, eager, custom, args in [
        ("affine", reference.denoise_step, _ops.denoise_step, (0.87, -0.13)),
        (
            "clipped_ddim",
            reference.clipped_ddim_step,
            _ops.clipped_ddim_step,
            (0.1, 0.2, 1.0),
        ),
    ]:
        compiled = torch.compile(
            eager, fullgraph=True, options={"triton.cudagraphs": False}
        )
        for shape in shapes:
            generator = torch.Generator(device="cuda").manual_seed(0)
            x = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
            eps = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
            expected = eager(x, eps, *args)
            for label, op in [
                ("eager", eager),
                ("torch.compile", compiled),
                ("custom CUDA", custom),
            ]:
                fn = lambda: op(x, eps, *args)
                tolerance = {
                    torch.float32: 2e-5,
                    torch.float16: 2e-3,
                    torch.bfloat16: 2e-2,
                }[dtype]
                torch.testing.assert_close(
                    fn(), expected, rtol=tolerance, atol=tolerance
                )
                wall_us, graph_us = measure(fn, iters, warmup)
                rows.append(
                    dict(
                        op=name,
                        shape=list(shape),
                        backend=label,
                        wall_us=wall_us,
                        graph_us=graph_us,
                    )
                )
                print(
                    f"{name:13} {str(shape):22} {label:14} wall={wall_us:8.2f} us  graph={graph_us:7.2f} us",
                    flush=True,
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--shape", type=int, nargs="+", help="One custom tensor shape")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 1 or (args.shape and min(args.shape) < 1):
        parser.error("iterations, warmup, and shape dimensions must be positive")
    shapes = (
        [tuple(args.shape)]
        if args.shape
        else [(1, 3, 32, 32), (1, 4, 128, 128), (1, 4, 512, 512)]
    )
    rows = bench(shapes, getattr(torch, args.dtype), args.iters, args.warmup)
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        extension=_ops.backend(),
        dtype=args.dtype,
        iters=args.iters,
        warmup=args.warmup,
        results=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
