"""Generate real 32x32 RGB images with a pretrained UNet and custom CUDA DDIM.

    python -m scripts.realtime_demo --validate --profile

The default is unconditional CIFAR-10 generation, without text prompts or a VAE.
"""

import argparse
import json
import math
from pathlib import Path
import time

from PIL import Image
import torch

from predictions import RealtimePipeline, _ops
from predictions.pipeline import DEFAULT_MODEL


def save_grid(samples, path):
    pixels = ((samples.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    pixels = pixels.permute(0, 2, 3, 1).cpu().numpy()
    columns = math.ceil(math.sqrt(len(pixels)))
    height, width = pixels.shape[1:3]
    grid = Image.new(
        "RGB", (columns * width, math.ceil(len(pixels) / columns) * height)
    )
    for i, sample in enumerate(pixels):
        grid.paste(
            Image.fromarray(sample), ((i % columns) * width, (i // columns) * height)
        )
    grid.save(path)
    grid.resize((grid.width * 4, grid.height * 4), Image.Resampling.NEAREST).save(
        path.with_stem(path.stem + "_preview")
    )


@torch.inference_mode()
def validate_against_diffusers(pipe, seed):
    """Same weights, initial noise, and schedule; independently compare the loop."""
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler.from_pretrained(
        pipe.model_id, revision=pipe.revision, local_files_only=True
    )
    scheduler.set_timesteps(pipe.sampler.num_sample_steps)
    expected = pipe._noise(torch.Generator(device=pipe.device).manual_seed(seed))
    for timestep in scheduler.timesteps:
        eps = pipe.model_fn(expected, int(timestep))
        expected = scheduler.step(eps, timestep, expected, eta=0).prev_sample
    actual = pipe.generate(torch.Generator(device=pipe.device).manual_seed(seed))
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    error = (actual - expected).abs()
    return dict(
        max_abs_error=error.max().item(),
        mean_abs_error=error.mean().item(),
        rtol=2e-3,
        atol=2e-3,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of independently generated batches",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warmup", type=int, default=1, help="Untimed full sampling runs"
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/cifar10"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Compare full float32 generation with Diffusers",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Save a separate CPU/CUDA profile and Chrome trace",
    )
    args = parser.parse_args()
    if args.frames < 1 or args.batch_size < 1 or args.warmup < 0:
        parser.error(
            "frames and batch size must be positive; warmup must be nonnegative"
        )
    if args.validate and args.dtype != "float32":
        parser.error(
            "--validate uses float32 to compare equivalent scheduler precision"
        )
    if args.device == "cuda" and not _ops.extension_available():
        raise RuntimeError(
            "CUDA demo requires the compiled extension; fix the build before benchmarking."
        )
    if args.device == "cpu":
        print("CPU mode: using the PyTorch reference sampler.", flush=True)

    pipe = RealtimePipeline.from_pretrained(
        args.model,
        batch_size=args.batch_size,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        num_sample_steps=args.steps,
        local_files_only=args.local_files_only,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"model={pipe.model_id} params={sum(p.numel() for p in pipe.model.parameters()):,}"
    )
    print(
        f"device={args.device} dtype={args.dtype} backend={_ops.backend()} steps={args.steps} batch={args.batch_size}",
        flush=True,
    )

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    warmup_generator = torch.Generator(device=args.device).manual_seed(args.seed)
    for _ in range(args.warmup):
        pipe.generate(warmup_generator)
    synchronize()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    latencies = []
    images = []
    for i in range(args.frames):
        synchronize()
        start = time.perf_counter()
        samples = pipe.generate(generator)
        synchronize()
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)
        if not torch.isfinite(samples).all():
            raise RuntimeError("Model produced nonfinite samples")
        path = args.output / f"samples_{i:03d}.png"
        # Image conversion, transfer, and PNG encoding are outside sampling time.
        save_grid(samples, path)
        images.append(str(path))
        print(
            f"batch {i}: {elapsed * 1000:.1f} ms, {args.batch_size / elapsed:.2f} images/s; saved {path}",
            flush=True,
        )
    mean_ms = sum(latencies) / len(latencies)
    report = dict(
        model=pipe.model_id,
        revision=pipe.revision,
        seed=args.seed,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        dtype=args.dtype,
        backend=_ops.backend() if args.device == "cuda" else "reference",
        steps=args.steps,
        batch_size=args.batch_size,
        warmup_runs=args.warmup,
        sampling_ms=latencies,
        mean_batch_ms=mean_ms,
        batches_per_second=1000 / mean_ms,
        images_per_second=args.batch_size * 1000 / mean_ms,
        timing="Synchronized sampling, including noise generation; excludes loading, warmup, image transfer and PNG encoding.",
        images=images,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20
        if args.device == "cuda"
        else None,
    )
    if args.validate:
        print("Comparing the complete sampling loop with Diffusers...", flush=True)
        report["validation"] = validate_against_diffusers(pipe, args.seed)
        print(f"Validation passed: {report['validation']}", flush=True)
    if args.profile:
        print("Profiling a separate sampling run...", flush=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            pipe.generate(torch.Generator(device=args.device).manual_seed(args.seed))
            synchronize()
        prof.export_chrome_trace(str(args.output / "trace.json"))
        table = prof.key_averages().table(
            sort_by="self_device_time_total", row_limit=30
        )
        (args.output / "profile.txt").write_text(table)
        print(table)
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Run details: {args.output / 'run.json'}")


if __name__ == "__main__":
    main()
