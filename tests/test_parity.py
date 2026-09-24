"""CUDA kernels must match the PyTorch reference within tolerance."""
import pytest
import torch

from predictions import _ops, reference


def test_denoise_step_reference_math():
    x = torch.randn(4, 3, 16, 16)
    eps = torch.randn_like(x)
    out = reference.denoise_step(x, eps, 0.9, -0.1)
    expected = 0.9 * x - 0.1 * eps
    assert torch.allclose(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_denoise_step_kernel_matches_reference():
    x = torch.randn(4, 3, 64, 64, device="cuda")
    eps = torch.randn_like(x)
    coef_x, coef_eps = 0.87, -0.13

    ref = reference.denoise_step(x, eps, coef_x, coef_eps)
    got = _ops.denoise_step(x, eps, coef_x, coef_eps)

    assert _ops.extension_available(), "compiled kernel not loaded on a CUDA box"
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_denoise_step_dtypes(dtype):
    x = torch.randn(1024, device="cuda", dtype=dtype)
    eps = torch.randn_like(x)
    ref = reference.denoise_step(x, eps, 0.5, 0.5)
    got = _ops.denoise_step(x, eps, 0.5, 0.5)
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
