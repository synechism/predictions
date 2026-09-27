import torch
import pytest

from predictions.sampler import Sampler, ddim_coefficients, linear_beta_schedule


def test_timesteps_are_descending_and_bounded():
    s = Sampler(num_train_steps=1000, num_sample_steps=50)
    ts = s.timesteps()
    assert ts[0] == 980
    assert len(ts) == 50
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


@pytest.mark.parametrize("steps", [1, 3, 8, 50, 333, 1000])
def test_exact_step_count(steps):
    sampler = Sampler(num_sample_steps=steps)
    assert len(sampler.timesteps()) == steps
    assert len(set(sampler.timesteps())) == steps


@pytest.mark.parametrize("steps", [0, -1, 1001, 3.5])
def test_invalid_step_count(steps):
    with pytest.raises(ValueError, match="num_sample_steps"):
        Sampler(num_sample_steps=steps)


@pytest.mark.parametrize("steps", [3, 8, 50])
@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matches_diffusers_schedule_and_trajectory(steps, clip, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    diffusers = pytest.importorskip("diffusers")
    scheduler = diffusers.DDIMScheduler(clip_sample=clip, beta_schedule="scaled_linear")
    scheduler.set_timesteps(steps)
    sampler = Sampler.from_diffusers(scheduler, steps)
    assert sampler.timesteps() == scheduler.timesteps.tolist()
    generator = torch.Generator(device=device).manual_seed(19)
    start = torch.randn(1, 3, 8, 8, device=device, generator=generator)

    def model(x, t):
        return 0.1 * x.tanh()

    expected = start.clone()
    for t in scheduler.timesteps:
        expected = scheduler.step(
            model(expected, int(t)), t, expected, eta=0
        ).prev_sample
    actual = sampler.sample(model, start)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_rejects_unsupported_scheduler():
    diffusers = pytest.importorskip("diffusers")
    for config in [
        {"prediction_type": "v_prediction"},
        {"thresholding": True},
        {"timestep_spacing": "trailing"},
    ]:
        with pytest.raises(ValueError):
            Sampler.from_diffusers(diffusers.DDIMScheduler(**config), 10)
