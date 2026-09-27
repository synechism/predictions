#include "predictions.h"
#include "utils.cuh"

#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cmath>

namespace predictions {

namespace {

template <typename scalar_t>
__global__ void denoise_step_kernel(const scalar_t* __restrict__ x_t,
                                    const scalar_t* __restrict__ eps,
                                    scalar_t* __restrict__ out,
                                    at::opmath_type<scalar_t> coef_x,
                                    at::opmath_type<scalar_t> coef_eps,
                                    int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    // Adjacent threads read adjacent elements. Half/bfloat16 arithmetic is
    // performed in float32, rounding only when the result is stored.
    using math_t = at::opmath_type<scalar_t>;
    out[i] = coef_x * static_cast<math_t>(x_t[i]) +
             coef_eps * static_cast<math_t>(eps[i]);
  }
}

template <typename scalar_t>
__global__ void clipped_ddim_step_kernel(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ eps,
    scalar_t* __restrict__ out, at::opmath_type<scalar_t> sqrt_alpha_t,
    at::opmath_type<scalar_t> sqrt_beta_t,
    at::opmath_type<scalar_t> sqrt_alpha_prev,
    at::opmath_type<scalar_t> sqrt_beta_prev,
    at::opmath_type<scalar_t> clip_range, int64_t n) {
  using math_t = at::opmath_type<scalar_t>;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    const math_t noise = static_cast<math_t>(eps[i]);
    math_t x0 = (static_cast<math_t>(x[i]) - sqrt_beta_t * noise) / sqrt_alpha_t;
    x0 = x0 < -clip_range ? -clip_range : (x0 > clip_range ? clip_range : x0);
    out[i] = sqrt_alpha_prev * x0 + sqrt_beta_prev * noise;
  }
}

void check_inputs(const torch::Tensor& x, const torch::Tensor& eps) {
  TORCH_CHECK(x.is_cuda() && eps.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.device() == eps.device(), "inputs must be on the same device");
  TORCH_CHECK(x.sizes() == eps.sizes(), "inputs must have equal shape");
  TORCH_CHECK(x.scalar_type() == eps.scalar_type(), "inputs must have equal dtype");
  TORCH_CHECK(x.is_floating_point(), "inputs must have floating point dtype");
}

}  // namespace

// Fused affine sampler update. Elementwise for now; the intent is to grow this
// into a fused kernel that folds in the model's output normalization and the
// next-step noise injection so a full sampler step is one launch.
torch::Tensor denoise_step(torch::Tensor x_t,
                           torch::Tensor eps,
                           double coef_x,
                           double coef_eps) {
  check_inputs(x_t, eps);
  const c10::cuda::CUDAGuard guard(x_t.device());

  x_t = x_t.contiguous();
  eps = eps.contiguous();
  auto out = torch::empty_like(x_t);
  const int64_t n = x_t.numel();
  if (n == 0) {
    return out;
  }

  const int blocks = static_cast<int>(
      std::min<int64_t>(ceil_div(n, kThreadsPerBlock), 65535));
  auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x_t.scalar_type(),
      "denoise_step", [&] {
        denoise_step_kernel<scalar_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
            x_t.data_ptr<scalar_t>(), eps.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), static_cast<at::opmath_type<scalar_t>>(coef_x),
            static_cast<at::opmath_type<scalar_t>>(coef_eps), n);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}

torch::Tensor clipped_ddim_step(torch::Tensor x, torch::Tensor eps,
                                double alpha_t, double alpha_prev,
                                double clip_range) {
  check_inputs(x, eps);
  TORCH_CHECK(alpha_t > 0 && alpha_t <= 1, "alpha_t must be in (0, 1]");
  TORCH_CHECK(alpha_prev > 0 && alpha_prev <= 1, "alpha_prev must be in (0, 1]");
  TORCH_CHECK(std::isfinite(clip_range) && clip_range > 0,
              "clip_range must be finite and positive");
  const c10::cuda::CUDAGuard guard(x.device());
  x = x.contiguous();
  eps = eps.contiguous();
  auto out = torch::empty_like(x);
  const int64_t n = x.numel();
  if (n == 0) return out;
  const int blocks = static_cast<int>(
      std::min<int64_t>(ceil_div(n, kThreadsPerBlock), 65535));
  auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "clipped_ddim_step", [&] {
        using math_t = at::opmath_type<scalar_t>;
        clipped_ddim_step_kernel<scalar_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
            x.data_ptr<scalar_t>(), eps.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            static_cast<math_t>(std::sqrt(alpha_t)),
            static_cast<math_t>(std::sqrt(1.0 - alpha_t)),
            static_cast<math_t>(std::sqrt(alpha_prev)),
            static_cast<math_t>(std::sqrt(1.0 - alpha_prev)),
            static_cast<math_t>(clip_range), n);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace predictions
