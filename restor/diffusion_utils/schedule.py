"""Diffusion and flow-matching schedules used across the experiments.

The strongest SAMECFM setting used the straight linear flow-matching path for
training and a uniform (linear-time) Euler ODE grid for inference.  The
schedule comparison that selected this combination was not included in the
paper.  DDPM and log-SNR alternatives remain here for reproducibility of the
ablations.
"""

import torch


def make_linear_beta_schedule(num_train_timesteps, beta_start, beta_end, device=None):
    """Create the standard DDPM linear beta schedule."""
    return torch.linspace(
        beta_start,
        beta_end,
        num_train_timesteps,
        device=device,
        dtype=torch.float32,
    )


def precompute_ddpm_buffers(
    num_train_timesteps=1000,
    beta_start=1.0e-4,
    beta_end=2.0e-2,
    device=None,
):
    """Precompute scalar DDPM schedule tensors used in training and sampling."""
    betas = make_linear_beta_schedule(
        num_train_timesteps, beta_start, beta_end, device=device
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([
        torch.ones(1, device=device, dtype=torch.float32),
        alphas_cumprod[:-1],
    ])

    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
        "posterior_variance": torch.clamp(posterior_variance, min=1e-20),
    }


def move_ddpm_buffers(buffers, device):
    """Move a DDPM buffer dict onto a device."""
    return {name: value.to(device) for name, value in buffers.items()}


def extract(buffer, t, target_shape):
    """Gather timestep values and reshape them for broadcasting to a target."""
    if t.ndim != 1:
        raise ValueError(f"Expected t to be rank-1, got shape {tuple(t.shape)}")
    values = buffer.gather(0, t)
    return values.reshape(t.shape[0], *([1] * (len(target_shape) - 1)))


def q_sample(z_clean, t, epsilon, buffers):
    """Apply the DDPM forward noising process to clean latents."""
    sqrt_alpha_bar_t = extract(
        buffers["sqrt_alphas_cumprod"], t, z_clean.shape
    ).to(z_clean.dtype)
    sqrt_one_minus_alpha_bar_t = extract(
        buffers["sqrt_one_minus_alphas_cumprod"], t, z_clean.shape
    ).to(z_clean.dtype)
    return sqrt_alpha_bar_t * z_clean + sqrt_one_minus_alpha_bar_t * epsilon


def make_cfm_euler_time_grid(
    num_steps,
    cfm_cfg,
    device=None,
    dtype=torch.float32,
):
    """Return ``num_steps + 1`` increasing CFM Euler time points.

    ``uniform`` is the best-performing final setting and preserves the
    linear-in-time Euler grid. ``log_snr`` is an ablation that uses
    a fixed, audio-length-independent logistic coordinate:

        lambda(t) = log((1 - t) / t),  t = sigmoid(-lambda)

    The finite lambda bounds define the interior Euler points. Exact t=0/t=1
    endpoints are added separately so all schedules integrate the full CFM
    ODE interval. No audio or latent sequence length enters this calculation.
    """
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")

    schedule = cfm_cfg.get("inference_timestep_schedule", "uniform")
    if schedule == "uniform":
        # arange/steps is bit-identical to the former Python step_idx/steps
        # values after conversion to float32.
        time_grid = (
            torch.arange(
                num_steps + 1,
                device=device,
                dtype=torch.float32,
            )
            / num_steps
        )
    elif schedule == "log_snr":
        logsnr_min = float(cfm_cfg.get("logsnr_min", -6.2))
        logsnr_max = float(cfm_cfg.get("logsnr_max", 6.2))
        if not logsnr_min < logsnr_max:
            raise ValueError(
                "logsnr_min must be less than logsnr_max, got "
                f"{logsnr_min} and {logsnr_max}"
            )

        if num_steps == 1:
            time_grid = torch.tensor(
                [0.0, 1.0],
                device=device,
                dtype=torch.float32,
            )
        else:
            interior_count = num_steps - 1
            if interior_count == 1:
                # Preserve symmetry for the two-step edge case.
                lambdas = torch.tensor(
                    [(logsnr_max + logsnr_min) / 2.0],
                    device=device,
                    dtype=torch.float32,
                )
            else:
                # CFM integrates t: 0 -> 1, so lambda runs max -> min.
                lambdas = torch.linspace(
                    logsnr_max,
                    logsnr_min,
                    interior_count,
                    device=device,
                    dtype=torch.float32,
                )
            interior_times = torch.sigmoid(-lambdas)
            time_grid = torch.cat(
                (
                    torch.zeros(1, device=device, dtype=torch.float32),
                    interior_times,
                    torch.ones(1, device=device, dtype=torch.float32),
                )
            )
    else:
        raise ValueError(
            "CFM inference_timestep_schedule must be 'uniform' or 'log_snr', "
            f"got {schedule!r}"
        )

    time_grid = time_grid.to(dtype=dtype)
    if not torch.all(time_grid[1:] > time_grid[:-1]):
        raise ValueError(
            f"CFM {schedule!r} Euler time grid is not strictly increasing"
        )
    return time_grid
