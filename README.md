# predictions

Realtime diffusion with custom CUDA kernels.

The goal: drive a diffusion model fast enough to sample interactively (ideally
video-rate) by pushing the sampling hot path into hand-written CUDA kernels
instead of relying on generic framework ops. The Python layer stays thin — it
orchestrates the schedule and owns the model weights; the per-step math that
runs every frame lives in `csrc/`.

## Layout

```
predictions/
├── csrc/                     # C++/CUDA extension sources
│   ├── bindings.cpp          # pybind11 entry points
│   ├── kernels/
│   │   ├── denoise_step.cu   # fused sampler update (the hot path)
│   │   └── utils.cuh         # shared device helpers
│   └── include/
│       └── predictions.h     # C++ declarations shared across TUs
├── predictions/              # Python package
│   ├── __init__.py
│   ├── _ops.py               # loads the compiled extension, exposes ops
│   ├── sampler.py            # sampling loop / schedulers
│   ├── pipeline.py           # end-to-end realtime pipeline
│   └── reference.py          # pure-PyTorch reference impls (for parity tests)
├── tests/
│   ├── test_parity.py        # CUDA kernels vs. reference, numerically
│   └── test_sampler.py
├── benchmarks/
│   └── bench_step.py         # per-step latency / throughput
├── scripts/
│   └── realtime_demo.py      # interactive demo entry point
├── setup.py                  # builds the CUDA extension
├── pyproject.toml
└── requirements.txt
```

## Build

Requires an NVIDIA GPU, the CUDA toolkit, and a matching PyTorch build.

```bash
pip install -r requirements.txt
pip install -e .          # compiles csrc/ against your local CUDA/torch
```

The extension is also compiled just-in-time on first import via
`torch.utils.cpp_extension.load` (see `predictions/_ops.py`), so you can iterate
on a kernel without a full reinstall.

## Develop

```bash
pytest tests/                        # parity + sampler tests
python benchmarks/bench_step.py      # measure per-step latency
python scripts/realtime_demo.py      # interactive loop
```

## Approach

Every custom kernel ships with a pure-PyTorch twin in `reference.py`. The parity
tests assert the two agree within tolerance, so a kernel is only "correct" when
it matches the reference — that keeps the fast path honest as it gets optimized.

## Status

Skeleton. Kernels are stubs that fall back to the reference implementation until
filled in.
