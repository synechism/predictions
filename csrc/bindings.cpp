#include "predictions.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "predictions: custom CUDA kernels for realtime diffusion";
  m.def("denoise_step", &predictions::denoise_step,
        "Fused one-step denoiser update (x_prev = coef_x * x_t + coef_eps * eps)",
        py::arg("x_t"), py::arg("eps"), py::arg("coef_x"), py::arg("coef_eps"));
}
