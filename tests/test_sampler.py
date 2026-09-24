import torch

from predictions.sampler import Sampler, ddim_coefficients, linear_beta_schedule


def test_timesteps_are_descending_and_bounded():
    s = Sampler(num_train_steps=1000, num_sample_steps=50)
    ts = s.timesteps()
    assert ts[0] == 999
    assert all(earlier > later for earlier, later in zip(ts, ts[1:]))
    assert min(ts) >= 0


def test_ddim_coefficients_final_step_reconstructs_x0():
    # At the final step (t_prev = -1 => a_prev = 1) DDIM maps to the x0
    # estimate: x0 = (x_t - sqrt(1-a_t) eps) / sqrt(a_t).
    alphas_cumprod = torch.cumprod(1.0 - linear_beta_schedule(1000), dim=0)
    t = 100
    coef_x, coef_eps = ddim_coefficients(alphas_cumprod, t, -1)
    a_t = float(alphas_cumprod[t])
    assert abs(coef_x - 1.0 / a_t**0.5) < 1e-5
    assert abs(coef_eps - (-((1 - a_t) ** 0.5) / a_t**0.5)) < 1e-5


def test_sample_runs_end_to_end_on_cpu():
    s = Sampler(num_train_steps=1000, num_sample_steps=5)
    x0 = torch.randn(2, 3, 8, 8)

    # A trivial "model" that predicts zero noise; sampling should stay finite.
    def model_fn(x, t):
        return torch.zeros_like(x)

    out = s.sample(model_fn, x0)
    assert out.shape == x0.shape
    assert torch.isfinite(out).all()
