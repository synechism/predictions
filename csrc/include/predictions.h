#pragma once

#include <torch/extension.h>

namespace predictions {

// Fused one-step denoiser update.
//
// Given the current sample x_t, the model's predicted noise eps, and the
// scalar schedule coefficients for this step, produce x_{t-1} in a single
// kernel launch. This accelerates scheduler math; the neural network is separate.
//
//   x_prev = coef_x * x_t + coef_eps * eps
//
// (The exact coefficient meaning depends on the sampler; the kernel just
// applies the fused affine combination the scheduler hands it.)
torch::Tensor denoise_step(torch::Tensor x_t,
                           torch::Tensor eps,
                           double coef_x,
                           double coef_eps);

// DDIM eta=0 with epsilon prediction and clipping of the reconstructed x0.
torch::Tensor clipped_ddim_step(torch::Tensor x, torch::Tensor eps,
                                double alpha_t, double alpha_prev,
                                double clip_range);

}  // namespace predictions
