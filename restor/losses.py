"""Training objectives retained for the final model and its ablations.

For the submitted SAMECFM-40M model, the only function used here is
``latent_mse_loss``.  Specifically, it compares the predicted CFM velocity
with the straight-path target velocity in normalized SAME-L space; it is not
a direct restored-latent reconstruction loss.  The waveform spectral losses
below support earlier deterministic models and objective ablations.
"""

import torch
import torch.nn.functional as F


def latent_mse_loss(pred, target, reduction="mean"):
    """MSE helper used by final CFM velocity training and latent ablations."""
    return F.mse_loss(pred, target.to(pred.device), reduction=reduction)


def multiscale_spectral_loss(pred, target, fft_sizes=(512, 1024, 2048),
                             hop_ratio=0.25, eps=1e-7):
    """Multi-resolution spectral convergence + log-magnitude L1 loss.

    Args:
        pred, target: (B, 1, T) audio tensors
        fft_sizes: tuple of FFT sizes
    Returns:
        scalar loss
    """
    device = pred.device
    target = target.to(device)   # ensure both tensors are on the same device
    loss = 0.0
    for n_fft in fft_sizes:
        hop = int(n_fft * hop_ratio)
        window = torch.hann_window(n_fft, device=device)

        pred_spec = torch.stft(
            pred.squeeze(1), n_fft, hop_length=hop, win_length=n_fft,
            window=window, return_complex=True,
        ).abs()
        target_spec = torch.stft(
            target.squeeze(1), n_fft, hop_length=hop, win_length=n_fft,
            window=window, return_complex=True,
        ).abs()

        # spectral convergence
        loss += (pred_spec - target_spec).norm(p="fro") / (target_spec.norm(p="fro") + eps)
        # log-magnitude L1
        loss += F.l1_loss(torch.log(pred_spec + eps), torch.log(target_spec + eps))

    return loss / len(fft_sizes)


def multiscale_spectral_loss_DDSP(
    pred,
    target,
    fft_sizes=(4096, 2048, 1024, 512),
    hop_sizes=(1024, 512, 256, 128),
    logmag_weight=0.1,
    eps=1e-7,
):
    """Mono adaptation of the attached DDSP multi-scale spectral objective.

    At each scale this sums a linear-magnitude L1 loss and a log10-magnitude
    MSE. Their weights match the supplied implementation: 0.9 magnitude and
    0.1 log magnitude. Only magnitudes are compared, so phase is disabled.

    Based on DDSP: https://arxiv.org/abs/2001.04643
    """
    if len(fft_sizes) != len(hop_sizes):
        raise ValueError("DDSP FFT and hop lists must have equal lengths")
    if not 0.0 <= logmag_weight <= 1.0:
        raise ValueError("logmag_weight must be between 0 and 1")

    target = target.to(pred.device, dtype=pred.dtype)
    total_mag_loss = pred.new_zeros(())
    total_logmag_loss = pred.new_zeros(())
    for n_fft, hop_size in zip(fft_sizes, hop_sizes):
        window = torch.hann_window(
            int(n_fft), device=pred.device, dtype=pred.dtype
        )
        pred_mag = torch.stft(
            pred.squeeze(1),
            n_fft=int(n_fft),
            hop_length=int(hop_size),
            win_length=int(n_fft),
            window=window,
            return_complex=True,
        ).abs()
        target_mag = torch.stft(
            target.squeeze(1),
            n_fft=int(n_fft),
            hop_length=int(hop_size),
            win_length=int(n_fft),
            window=window,
            return_complex=True,
        ).abs()

        total_mag_loss = total_mag_loss + F.l1_loss(pred_mag, target_mag)
        total_logmag_loss = total_logmag_loss + F.mse_loss(
            torch.log10(pred_mag + eps),
            torch.log10(target_mag + eps),
        )

    return (
        (1.0 - logmag_weight) * total_mag_loss
        + logmag_weight * total_logmag_loss
    )
