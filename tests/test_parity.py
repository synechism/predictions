"""Check real CUDA execution, awkward sizes, precision, and stream semantics."""

import pytest
import torch

from predictions import _ops, reference

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_denoise_step_reference_math():
    x = torch.tensor([1.0, -2.0, 3.0])
    eps = torch.tensor([2.0, 3.0, -4.0])
    torch.testing.assert_close(
        reference.denoise_step(x, eps, 2.0, -0.5), torch.tensor([1.0, -5.5, 8.0])
    )


@pytest.fixture
def require_extension():
    assert _ops.extension_available(), (
        "CUDA tests must not silently exercise the PyTorch fallback"
    )


@CUDA
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
@pytest.mark.parametrize("shape", [(0,), (1,), (257,), (2, 3, 32, 32)])
@pytest.mark.parametrize("coefficients", [(0.87, -0.13), (157.41, -157.405)])
def test_affine_parity(require_extension, dtype, shape, coefficients):
    generator = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    eps = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    actual = _ops.denoise_step(x, eps, *coefficients)
    expected = reference.denoise_step(x, eps, *coefficients)
    tol = {
        torch.float64: 1e-12,
        torch.float32: 1e-4,
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
    }[dtype]
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alphas", [(0.00004, 0.0001), (0.1, 0.2), (0.9999, 1.0)])
def test_clipped_ddim_parity(require_extension, dtype, alphas):
    generator = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(2, 3, 32, 32, device="cuda", dtype=dtype, generator=generator)
    eps = torch.randn(x.shape, device="cuda", dtype=dtype, generator=generator)
    actual = _ops.clipped_ddim_step(x, eps, *alphas)
    expected = reference.clipped_ddim_step(x, eps, *alphas)
    tol = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    if alphas[1] == 1.0:
        assert actual.abs().max() <= 1.0


@CUDA
@pytest.mark.parametrize(
    "op,args", [("denoise_step", (0.87, -0.13)), ("clipped_ddim_step", (0.1, 0.2))]
)
def test_noncontiguous_and_empty(require_extension, op, args):
    for shape in [(7, 11), (0, 3)]:
        x = torch.randn(shape, device="cuda").T
        eps = torch.randn(shape, device="cuda").T
        torch.testing.assert_close(
            getattr(_ops, op)(x, eps, *args),
            getattr(reference, op)(x, eps, *args),
            atol=1e-5,
            rtol=1e-5,
        )


@CUDA
@pytest.mark.parametrize(
    "op,args", [("denoise_step", (0.87, -0.13)), ("clipped_ddim_step", (0.1, 0.2))]
)
def test_rejects_incompatible_inputs(require_extension, op, args):
    x = torch.zeros(32, device="cuda")
    fn = getattr(_ops, op)
    with pytest.raises(RuntimeError, match="equal dtype"):
        fn(x, x.half(), *args)
    with pytest.raises(RuntimeError, match="equal shape"):
        fn(x, x[:16], *args)
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        fn(x, x.cpu(), *args)


@CUDA
def test_current_stream(require_extension):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.ones(65537, device="cuda") * 3
        eps = torch.ones_like(x) * 2
        out = _ops.denoise_step(x, eps, 2.0, -0.5)
    stream.synchronize()
    torch.testing.assert_close(out, torch.full_like(out, 5.0))
