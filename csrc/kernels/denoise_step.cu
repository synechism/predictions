#include "predictions.h"
#include "kernels/utils.cuh"

#include <c10/cuda/CUDAStream.h>

namespace predictions {

namespace {

template <typename scalar_t>
__global__ void denoise_step_kernel(const scalar_t* __restrict__ x_t,
                                    const scalar_t* __restrict__ eps,
                                    scalar_t* __restrict__ out,
                                    scalar_t coef_x,
                                    scalar_t coef_eps,
                                    int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = coef_x * x_t[i] + coef_eps * eps[i];
  }
}

}  // namespace

// Fused affine sampler update. Elementwise for now; the intent is to grow this
// into a fused kernel that folds in the model's output normalization and the
// next-step noise injection so a full sampler step is one launch.
torch::Tensor denoise_step(torch::Tensor x_t,
                           torch::Tensor eps,
                           double coef_x,
                           double coef_eps) {
  TORCH_CHECK(x_t.is_cuda(), "x_t must be a CUDA tensor");
  TORCH_CHECK(eps.is_cuda(), "eps must be a CUDA tensor");
  TORCH_CHECK(x_t.sizes() == eps.sizes(), "x_t and eps must have equal shape");

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
            out.data_ptr<scalar_t>(), static_cast<scalar_t>(coef_x),
            static_cast<scalar_t>(coef_eps), n);
      });

  return out;
}

}  // namespace predictions
