#pragma once

#include <cuda_runtime.h>

namespace predictions {

// Standard grid-stride loop bound helper.
__host__ __device__ inline int64_t ceil_div(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// Default threads-per-block for elementwise kernels. Tune per-arch later.
constexpr int kThreadsPerBlock = 256;

}  // namespace predictions
