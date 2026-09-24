"""Build the predictions CUDA extension.

Falls back to a CPU-only / no-extension install when CUDA is unavailable, so the
Python package still imports (kernels then route through the PyTorch reference
implementations). See predictions/_ops.py.
"""
from setuptools import setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    HAS_CUDA = torch.cuda.is_available() or torch.version.cuda is not None
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
            include_dirs=["csrc/include"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
