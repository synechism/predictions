"""Build the predictions CUDA extension.

Falls back to a CPU-only / no-extension install when CUDA is unavailable, so the
Python package still imports (kernels then route through the PyTorch reference
implementations). See predictions/_ops.py.
"""

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).parent.resolve()

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    HAS_CUDA = torch.version.cuda is not None and CUDA_HOME is not None
except Exception:  # torch not importable at build time
    HAS_CUDA = False

ext_modules = []
cmdclass = {}

if HAS_CUDA:
    ext_modules = [
        CUDAExtension(
            name="predictions._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/kernels/denoise_step.cu",
            ],
            include_dirs=[str(ROOT / "csrc" / "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
