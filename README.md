# predictions

A working CUDA-kernel learning project with a small pretrained diffusion model.
It generates actual 32×32 CIFAR-10 images using
[`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32), a
35.7M-parameter unconditional UNet. Python runs the neural network; our CUDA
extension performs every DDIM sampler update. There is no text prompt or VAE.

The current example is a correctness and profiling baseline. See
[measured results](docs/results.md) for the RTX 4070; it is not yet video-rate.

## Run on this box

The `.venv`, compiled extension, and model cache are already prepared:

```bash
.venv/bin/python -m scripts.realtime_demo --local-files-only
```

This saves 16 samples in `outputs/cifar10/samples_000.png`, an enlarged
`samples_000_preview.png`, and timings/configuration in `run.json`. Each image is
32×32; enlarging the preview does not add detail. The default uses 50 DDIM steps
and seed 0. `--frames` means independently generated batches, not coherent video.

```bash
# Full output comparison against Diffusers, plus a separate profiling run.
.venv/bin/python -m scripts.realtime_demo --local-files-only --validate --profile

# Single-image latency; warmup is excluded and CUDA is synchronized.
.venv/bin/python -m scripts.realtime_demo --local-files-only --batch-size 1 --frames 5 --output outputs/single

# Experiment with the speed/quality tradeoff and half precision.
.venv/bin/python -m scripts.realtime_demo --local-files-only --steps 20 --dtype float16 --output outputs/fp16

# Numerical tests and a comparison with both PyTorch baselines.
.venv/bin/python -m pytest -q
.venv/bin/python -m benchmarks.bench_step
```

`--validate` uses float32, identical weights/noise/timesteps, and compares the
complete custom sampling trajectory against `DDIMScheduler.step(eta=0)`.
`--profile` writes `profile.txt` and a `trace.json` that can be opened in Perfetto
or a Chrome trace viewer. The trace can be large. Loading, warmup, PNG encoding,
and the extra validation/profiling runs are excluded from sampling latency.
Batch throughput (images/s) is distinct from latency (ms/batch).

## Build or rebuild

Requires an NVIDIA driver, CUDA toolkit (`nvcc`), and a CUDA-enabled PyTorch.
The tested environment is Linux, Python 3.14, PyTorch 2.14 / CUDA 13.0,
CUDA toolkit 13.3, GCC 15, and an RTX 4070 (compute capability 8.9).

To recreate the environment on this machine:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt

# This machine's default GCC 16 is newer than nvcc's supported GCC range.
export CC=gcc-15
export CXX=g++-15
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.9
uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e .

# Omit --local-files-only on the first run to download the checkpoint (~143 MB).
.venv/bin/python -m scripts.realtime_demo --validate
```

For another system, install a compatible PyTorch/toolkit first, then the
`dev` and `demo` extras with `--no-build-isolation`; the lock file records this
Linux machine's exact versions. Use a host compiler supported by your toolkit,
and set the architecture for your own GPU (or let PyTorch detect it).

After editing CUDA, rerun the editable-install command above with those compiler
variables set. Each fresh Python process loads the rebuilt extension. If no
prebuilt extension is installed, `_ops.py` tries a JIT build. Both build paths
are tested. CPU use falls back to the PyTorch reference; the CUDA demo and
benchmark require an actual compiled extension and fail if it is unavailable.

## Follow the code

| File | What to learn |
| --- | --- |
| `predictions/reference.py` | Readable equations and the precision contract |
| `csrc/kernels/denoise_step.cu` | Grid-stride indexing, fused math, dtype dispatch, streams, launch checks |
| `csrc/bindings.cpp` / `predictions/_ops.py` | Calling CUDA from Python and selecting the backend |
| `tests/test_parity.py` | Numerical agreement, empty/strided inputs, invalid inputs, nondefault streams |
| `benchmarks/bench_step.py` | Eager PyTorch vs `torch.compile` vs handwritten CUDA |
| `predictions/sampler.py` | Exact timestep count and precomputed scheduler coefficients |
| `predictions/pipeline.py` | Loading a real UNet and preserving its trained noise schedule |
| `scripts/realtime_demo.py` | Generating images, saving results, validating, and profiling |

The first kernel evaluates `out[i] = a * x[i] + b * eps[i]`. A thread starts at
`blockIdx.x * blockDim.x + threadIdx.x` and advances by
`blockDim.x * gridDim.x`. Neighboring threads read neighboring elements. Start
with the float32 instantiation; the template adds float16, bfloat16, and float64.
Half/bfloat16 inputs are computed in float32 and rounded when stored. See
[NVIDIA's grid-stride explanation](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/).

The model's scheduler also clips its clean-image estimate. A second kernel
fuses these equations into one launch:

```text
x0     = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
x0     = clamp(x0, -clip_range, clip_range)
x_prev = sqrt(alpha_prev) * x0 + sqrt(1 - alpha_prev) * eps
```

Here `alpha_t` is the cumulative product of the training alphas. The default
checkpoint and revision are pinned in `pipeline.py`, safetensors weights are
loaded, and the trained alpha schedule comes from Diffusers. The adapter
supports epsilon prediction, deterministic DDIM (`eta=0`), and leading timestep
spacing with zero offset. It rejects unsupported scheduler features instead of
silently substituting different mathematics. These are inference-only ops.

## Read the benchmark correctly

```bash
.venv/bin/python -m benchmarks.bench_step --dtype float32
.venv/bin/python -m benchmarks.bench_step --dtype float16 --shape 1 4 128 128 --output outputs/benchmark_fp16.json
```

Both ops run on identical inputs in all three implementations. Compilation and
warmup are excluded; outputs are checked before measurement. Each result is the
median of three measurements:

- `wall_us`: synchronized Python-to-result time per call, including dispatch
  and allocation.
- `graph_us`: CUDA-event time per call within repeated 50-call CUDA graphs,
  reducing host submission overhead equally for all implementations.

Graph timing repeatedly accesses the same tensors and is a warm-cache workload,
not a claim about peak DRAM bandwidth. `torch.compile` already fuses pointwise
math; a handwritten kernel is not guaranteed to beat its device execution time.
See [PyTorch's fusion guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations).
