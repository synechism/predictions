# Verified example — 2026-09-26

Hardware: RTX 4070, 12 GB VRAM. Environment: Python 3.14.7, PyTorch
2.14.0+cu130, Diffusers 0.40.0, Triton 3.8.0, nvcc 13.3, GCC 15.

## Correctness

- 73 tests passed, including float32/float16/bfloat16/float64 affine kernels,
  clipped DDIM, uneven and empty tensors, noncontiguous inputs, invalid input
  checks, nondefault CUDA streams, and trajectories against Diffusers.
- The pretrained UNet generated 16 recognizable 32×32 images with 50 steps,
  float32, and seed 0. Weights: `google/ddpm-cifar10-32`, revision
  `267b167dc01f0e4e61923ea244e8b988f84deb80`.
- Complete generation matched Diffusers with maximum absolute error 0.001684
  and mean absolute error 0.0000306, in the output range [-1, 1]. Validation uses
  `rtol=0.002, atol=0.002`. Fused floating-point evaluation is not bitwise identical.

The output and machine-readable metadata are under `outputs/cifar10/`.
Generated files and downloaded weights are intentionally gitignored.

## Sampling and profiling

The warmed 50-step float32 run took **697.5 ms for a batch of 16**, or 22.94
images/s. This is batch throughput, not 22.94 FPS for an interactive single
image. The measured interval includes noise generation and synchronized
sampling, and excludes model loading, warmup, image transfer, and PNG encoding.
Peak PyTorch allocated GPU memory was approximately 271 MiB for that run;
this excludes CUDA context/library and other processes' memory.

A separate batch-size-one run measured 308.4, 310.1, and 277.6 ms per image
(298.7 ms mean) at the same 50-step float32 setting. The documented 20-step
float16 experiment also ran successfully: 164.8 ms for 16 images. It changes
both precision and step count, so it is a different quality/performance setting;
its full trajectory was not compared against the float32 Diffusers reference.
Reports are saved in `outputs/single/` and `outputs/fp16/`.

A separate profiler run recorded **50 custom clipped-DDIM kernel launches**.
Together they used **0.088 ms**, out of **683.0 ms** summed CUDA kernel time.
The denoising network dominates; accelerating only the scheduler cannot yield a
large end-to-end improvement here. Convolutions were the largest measured model
operation. Profiling adds overhead and its numbers are separate from the normal
sampling timer.

## Kernel comparison

Float32, 200 iterations, 30 warmup calls; median of three measurements.
These are one local run on a display GPU, not guaranteed performance figures.

| Op / shape | Backend | Wall µs/call | CUDA graph µs/call |
| --- | --- | ---: | ---: |
| Affine / 1×3×32×32 | Eager | 10.45 | 2.63 |
| | torch.compile | 16.48 | 0.81 |
| | Custom CUDA | 3.09 | 0.95 |
| Affine / 1×4×128×128 | Eager | 9.99 | 3.12 |
| | torch.compile | 19.82 | 1.17 |
| | Custom CUDA | 2.99 | 1.25 |
| Clipped DDIM / 1×3×32×32 | Eager | 23.10 | 5.60 |
| | torch.compile | 15.95 | 0.73 |
| | Custom CUDA | 2.98 | 0.89 |
| Clipped DDIM / 1×4×128×128 | Eager | 23.39 | 6.91 |
| | torch.compile | 19.78 | 1.17 |
| | Custom CUDA | 2.97 | 1.31 |

Full results, including 1×4×512×512, are in `outputs/benchmark.json`.
The custom wrapper has lower Python-call overhead in this experiment, while
`torch.compile` has slightly faster device execution. Both fuse the pointwise
math. CUDA graph measurements use repeated tensors and benefit from cache reuse.

## Next experiments

1. Read or rewrite the float32 affine kernel, checking parity after each change.
2. Change the block size and measure both small and large tensors; the default
   is 256 threads. Preserve the baseline results for comparison.
3. Use the real-model profile to select a model operation for the next kernel,
   or measure model compilation / CUDA graph capture. Keep an independent
   numerical reference and continue measuring complete generation latency.
