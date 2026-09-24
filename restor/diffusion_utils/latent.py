"""SAME-L latent layout and normalization helpers.

Per-channel normalization was essential for CFM training because native
SAME-L channels have unequal offsets and scales.  Standardizing them prevents
high-variance channels from dominating the velocity MSE, improves optimizer
conditioning, and places the clean target on a scale compatible with the
unit-Gaussian CFM source.  The saved ``latent_mean`` and ``latent_std`` are
training-set statistics used both to normalize before the denoiser and to
restore SAME-L's native latent scale before decoding.

Forgetting the final unnormalization sends standardized latents to a decoder
trained on native SAME-L latents and causes a very large quality drop.
"""

import torch


def ensure_channel_first(latent, channels=256):
    """Return latent as [B, channels, T]."""
    if latent.ndim != 3:
        raise ValueError(f"Expected rank-3 latent, got shape {tuple(latent.shape)}")
    if latent.shape[1] == channels:
        return latent
    if latent.shape[2] == channels:
        return latent.transpose(1, 2)
    raise ValueError(
        f"Latent must have channel dimension {channels}; got {tuple(latent.shape)}"
    )


def align_latent_pair(z_cond, z_clean):
    """Crop paired latents to the same valid time length."""
    if z_cond.ndim != 3 or z_clean.ndim != 3:
        raise ValueError("Expected paired latents to both be rank-3 tensors")
    if z_cond.shape[:2] != z_clean.shape[:2]:
        raise ValueError(
            "Latent batch/channel dimensions differ: "
            f"{tuple(z_cond.shape)} vs {tuple(z_clean.shape)}"
        )
    valid_t = min(z_cond.shape[-1], z_clean.shape[-1])
    return z_cond[..., :valid_t], z_clean[..., :valid_t], valid_t


def normalize_latent(latent, mean, std, eps=1e-6):
    """Standardize each SAME-L channel using saved training-set statistics."""
    return (latent - mean.to(latent.device, latent.dtype)) / (
        std.to(latent.device, latent.dtype) + eps
    )


def unnormalize_latent(latent, mean, std, eps=1e-6):
    """Restore native SAME-L scale; this is mandatory before codec decoding."""
    return latent * (std.to(latent.device, latent.dtype) + eps) + mean.to(
        latent.device, latent.dtype
    )


@torch.no_grad()
def compute_dataset_latent_stats(codec, loader, device, channels=256):
    """Compute per-channel clean-latent mean/std over a training loader."""
    total = None
    total_sq = None
    count = 0

    was_training = codec.training
    codec.eval()
    for clean in loader:
        clean = codec.preprocess(clean.to(device))
        z_clean = ensure_channel_first(codec.encode(clean), channels=channels)
        z_clean = z_clean.float()

        batch_sum = z_clean.sum(dim=(0, 2))
        batch_sum_sq = (z_clean * z_clean).sum(dim=(0, 2))
        batch_count = z_clean.shape[0] * z_clean.shape[2]

        total = batch_sum if total is None else total + batch_sum
        total_sq = batch_sum_sq if total_sq is None else total_sq + batch_sum_sq
        count += batch_count

    if was_training:
        codec.train()

    if count == 0:
        raise ValueError("Cannot compute latent stats from an empty loader")

    mean = total / count
    variance = (total_sq / count) - mean.square()
    std = torch.sqrt(torch.clamp(variance, min=1e-12))
    return mean.reshape(1, channels, 1), std.reshape(1, channels, 1)
