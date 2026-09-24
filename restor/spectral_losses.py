"""Mel-MSE analysis and spectral ablation utilities for restor.

Important: Mel-spectrogram MSE is an analysis metric in the final paper (for
example, the codec reconstruction comparison); it is not the SAMECFM-40M
training objective.  The final model minimizes CFM velocity MSE in normalized
SAME-L latent space.  ``mel_stft_mse_loss`` is therefore placed before the
other spectral objectives below to make its analysis role easy to find.

The defaults intentionally mirror auraloss' frequency-domain defaults:
single-resolution STFT uses ``n_fft=1024, hop=256, win=1024`` and
multi-resolution STFT uses ``[(1024, 120, 600), (2048, 240, 1200),
(512, 50, 240)]``.  The training path for spectrogram-domain Conformer/U-Net
uses complex STFT tensors represented as real-valued channels:

    [B, 2, F, T] where channel 0 is real and channel 1 is imaginary.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable

import soundfile as sf
import torch
import torchaudio

from auraloss import freq as _auraloss_freq


DEFAULT_FFT_SIZE = 1024
DEFAULT_HOP_SIZE = 256
DEFAULT_WIN_LENGTH = 1024
DEFAULT_CENTER = True
DEFAULT_N_MELS = 128

MULTIRES_FFT_SIZES = (1024, 2048, 512)
MULTIRES_HOP_SIZES = (120, 240, 50)
MULTIRES_WIN_LENGTHS = (600, 1200, 240)

SPECTRAL_LOSS_KEYS = (
    "complex_stft_mse",
    "mel_stft_mse",
    "multi_resolution_stft_mse",
)


def default_spectral_config() -> dict:
    """Return serializable STFT defaults for configs/metadata."""
    return {
        "fft_size": DEFAULT_FFT_SIZE,
        "hop_size": DEFAULT_HOP_SIZE,
        "win_length": DEFAULT_WIN_LENGTH,
        "center": DEFAULT_CENTER,
        "n_mels": DEFAULT_N_MELS,
        "multires_fft_sizes": list(MULTIRES_FFT_SIZES),
        "multires_hop_sizes": list(MULTIRES_HOP_SIZES),
        "multires_win_lengths": list(MULTIRES_WIN_LENGTHS),
    }


def _cfg_value(cfg: dict | None, key: str, default):
    return default if cfg is None else cfg.get(key, default)


def _as_batched_mono(audio: torch.Tensor) -> torch.Tensor:
    """Return audio as float tensor with shape [B, 1, samples]."""
    if audio.ndim == 1:
        audio = audio.reshape(1, 1, -1)
    elif audio.ndim == 2:
        audio = audio.unsqueeze(0)
    elif audio.ndim != 3:
        raise ValueError(f"Expected audio rank 1, 2, or 3; got {tuple(audio.shape)}")

    audio = audio.float()
    if audio.shape[1] > 1:
        audio = audio.mean(dim=1, keepdim=True)
    return audio


def prepare_audio_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downmix to mono, move target to pred device, and crop to common length."""
    pred = _as_batched_mono(pred)
    target = _as_batched_mono(target).to(pred.device, dtype=pred.dtype)
    if pred.shape[0] != target.shape[0]:
        raise ValueError(
            "Predicted and target audio must have the same batch size: "
            f"{pred.shape[0]} != {target.shape[0]}"
        )
    length = min(pred.shape[-1], target.shape[-1])
    if length <= 0:
        raise ValueError("Cannot compute spectral losses on empty audio")
    return pred[..., :length], target[..., :length]


def audio_to_complex_stft(
    audio: torch.Tensor,
    spectral_cfg: dict | None = None,
) -> torch.Tensor:
    """Convert audio to complex STFT with shape [B, F, T]."""
    audio = _as_batched_mono(audio)
    n_fft = int(_cfg_value(spectral_cfg, "fft_size", DEFAULT_FFT_SIZE))
    hop = int(_cfg_value(spectral_cfg, "hop_size", DEFAULT_HOP_SIZE))
    win = int(_cfg_value(spectral_cfg, "win_length", DEFAULT_WIN_LENGTH))
    center = bool(_cfg_value(spectral_cfg, "center", DEFAULT_CENTER))
    window = torch.hann_window(win, device=audio.device, dtype=audio.dtype)
    return torch.stft(
        audio.squeeze(1),
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=center,
        return_complex=True,
    )


def complex_to_channels(stft: torch.Tensor) -> torch.Tensor:
    """Represent complex STFT [B, F, T] as real channels [B, 2, F, T]."""
    if not torch.is_complex(stft):
        if stft.ndim == 4 and stft.shape[1] == 2:
            return stft.float()
        raise ValueError(f"Expected complex [B,F,T] or [B,2,F,T], got {tuple(stft.shape)}")
    return torch.stack([stft.real, stft.imag], dim=1).float()


def channels_to_complex(stft_channels: torch.Tensor) -> torch.Tensor:
    """Represent real channels [B, 2, F, T] as complex STFT [B, F, T]."""
    if torch.is_complex(stft_channels):
        return stft_channels
    if stft_channels.ndim != 4 or stft_channels.shape[1] != 2:
        raise ValueError(
            "Expected STFT channels with shape [B, 2, F, T], "
            f"got {tuple(stft_channels.shape)}"
        )
    return torch.complex(stft_channels[:, 0], stft_channels[:, 1])


def audio_to_stft_channels(
    audio: torch.Tensor,
    spectral_cfg: dict | None = None,
) -> torch.Tensor:
    """Convert audio to real/imag STFT channels [B, 2, F, T]."""
    return complex_to_channels(audio_to_complex_stft(audio, spectral_cfg))


def stft_channels_to_audio(
    stft_channels: torch.Tensor,
    length: int | None = None,
    spectral_cfg: dict | None = None,
) -> torch.Tensor:
    """Invert real/imag STFT channels to mono audio [B, 1, samples]."""
    stft = channels_to_complex(stft_channels)
    n_fft = int(_cfg_value(spectral_cfg, "fft_size", DEFAULT_FFT_SIZE))
    hop = int(_cfg_value(spectral_cfg, "hop_size", DEFAULT_HOP_SIZE))
    win = int(_cfg_value(spectral_cfg, "win_length", DEFAULT_WIN_LENGTH))
    center = bool(_cfg_value(spectral_cfg, "center", DEFAULT_CENTER))
    window = torch.hann_window(win, device=stft.real.device, dtype=stft.real.dtype)
    audio = torch.istft(
        stft,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=center,
        length=length,
    )
    return audio.unsqueeze(1)


def normalize_stft(stft_channels: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize STFT channels with stats shaped [1, 2, F, 1]."""
    return (stft_channels - mean.to(stft_channels.device, stft_channels.dtype)) / (
        std.to(stft_channels.device, stft_channels.dtype) + 1e-6
    )


def unnormalize_stft(stft_channels: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Undo ``normalize_stft``."""
    return stft_channels * (std.to(stft_channels.device, stft_channels.dtype) + 1e-6) + mean.to(
        stft_channels.device, stft_channels.dtype
    )


def per_file_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average MSE per file/window, then average equally across the batch."""
    target = target.to(pred.device, dtype=pred.dtype)
    dims = tuple(range(1, pred.ndim))
    return (pred - target).square().mean(dim=dims).mean()


def mel_stft_mse_loss(
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
    spectral_cfg: dict | None = None,
) -> torch.Tensor:
    """Analysis-only MSE over mel-scaled STFT magnitudes.

    SAMECFM-40M does not backpropagate this quantity; its training loss is the
    latent CFM velocity MSE described in ``restor.losses``.
    """
    pred_audio, target_audio = prepare_audio_pair(pred_audio, target_audio)
    n_fft = int(_cfg_value(spectral_cfg, "fft_size", DEFAULT_FFT_SIZE))
    hop = int(_cfg_value(spectral_cfg, "hop_size", DEFAULT_HOP_SIZE))
    win = int(_cfg_value(spectral_cfg, "win_length", DEFAULT_WIN_LENGTH))
    center = bool(_cfg_value(spectral_cfg, "center", DEFAULT_CENTER))
    n_mels = int(_cfg_value(spectral_cfg, "n_mels", DEFAULT_N_MELS))
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        n_mels=n_mels,
        center=center,
        power=1.0,
    ).to(pred_audio.device, dtype=pred_audio.dtype)
    pred_mel = mel(pred_audio.squeeze(1))
    target_mel = mel(target_audio.squeeze(1))
    return per_file_mse(pred_mel, target_mel)


def complex_stft_mse_loss(pred_stft, target_stft) -> torch.Tensor:
    """Per-file MSE over complex STFT real/imag channels."""
    return per_file_mse(complex_to_channels(pred_stft), complex_to_channels(target_stft))


def waveform_mse_loss(pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
    """Per-file waveform MSE for waveform-domain deterministic models."""
    pred_audio, target_audio = prepare_audio_pair(pred_audio, target_audio)
    return per_file_mse(pred_audio, target_audio)


def _mse_over_stfts(
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    center: bool = DEFAULT_CENTER,
) -> torch.Tensor:
    cfg = {
        "fft_size": fft_size,
        "hop_size": hop_size,
        "win_length": win_length,
        "center": center,
    }
    return complex_stft_mse_loss(
        audio_to_stft_channels(pred_audio, cfg),
        audio_to_stft_channels(target_audio, cfg),
    )


def multi_resolution_stft_mse_loss(
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor,
    spectral_cfg: dict | None = None,
) -> torch.Tensor:
    """Average complex-STFT squared error at three analysis resolutions.

    This is the legacy project-specific objective, not Auraloss'
    ``MultiResolutionSTFTLoss``. Squared error is applied directly to real and
    imaginary STFT values, so phase differences are penalized implicitly.
    """
    pred_audio, target_audio = prepare_audio_pair(pred_audio, target_audio)
    fft_sizes = _cfg_value(spectral_cfg, "multires_fft_sizes", MULTIRES_FFT_SIZES)
    hop_sizes = _cfg_value(spectral_cfg, "multires_hop_sizes", MULTIRES_HOP_SIZES)
    win_lengths = _cfg_value(spectral_cfg, "multires_win_lengths", MULTIRES_WIN_LENGTHS)
    if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
        raise ValueError(
            "Multi-resolution FFT, hop, and window lists must have equal lengths"
        )
    center = bool(_cfg_value(spectral_cfg, "center", DEFAULT_CENTER))
    losses = [
        _mse_over_stfts(pred_audio, target_audio, int(n_fft), int(hop), int(win), center)
        for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
    ]
    return torch.stack(losses).mean()


_AURALOSS_MRSTFT_CACHE = {}


def auralossMRSTFT(
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor,
    spectral_cfg: dict | None = None,
) -> torch.Tensor:
    """Apply Auraloss' Yamamoto-style ``MultiResolutionSTFTLoss``.

    Each resolution uses spectral convergence plus an L1 norm on log
    magnitudes. Linear-magnitude and phase losses are explicitly disabled
    (``w_lin_mag=0`` and ``w_phs=0``), so this objective does not penalize
    phase difference directly.

    Source: https://github.com/csteinmetz1/auraloss/blob/main/auraloss/freq.py
    """
    pred_audio, target_audio = prepare_audio_pair(pred_audio, target_audio)
    fft_sizes = tuple(
        int(value)
        for value in _cfg_value(
            spectral_cfg, "multires_fft_sizes", MULTIRES_FFT_SIZES
        )
    )
    hop_sizes = tuple(
        int(value)
        for value in _cfg_value(
            spectral_cfg, "multires_hop_sizes", MULTIRES_HOP_SIZES
        )
    )
    win_lengths = tuple(
        int(value)
        for value in _cfg_value(
            spectral_cfg, "multires_win_lengths", MULTIRES_WIN_LENGTHS
        )
    )
    if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
        raise ValueError(
            "Auraloss FFT, hop, and window lists must have equal lengths"
        )
    if not bool(_cfg_value(spectral_cfg, "center", DEFAULT_CENTER)):
        raise ValueError(
            "auralossMRSTFT requires spectral.center=true to match "
            "Auraloss MultiResolutionSTFTLoss"
        )

    # Cache the stateless loss module instead of rebuilding its STFT windows
    # at every optimizer step. Device is included because Auraloss moves its
    # window tensors to the input device during forward.
    cache_key = (
        fft_sizes,
        hop_sizes,
        win_lengths,
        str(pred_audio.device),
        str(pred_audio.dtype),
    )
    loss_module = _AURALOSS_MRSTFT_CACHE.get(cache_key)
    if loss_module is None:
        loss_module = _auraloss_freq.MultiResolutionSTFTLoss(
            fft_sizes=list(fft_sizes),
            hop_sizes=list(hop_sizes),
            win_lengths=list(win_lengths),
            window="hann_window",
            w_sc=1.0,
            w_log_mag=1.0,
            w_lin_mag=0.0,
            w_phs=0.0,
            scale=None,
            perceptual_weighting=False,
            scale_invariance=False,
            mag_distance="L1",
        )
        _AURALOSS_MRSTFT_CACHE[cache_key] = loss_module
    return loss_module(pred_audio, target_audio)


def compute_spectral_mse_losses(
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
    spectral_cfg: dict | None = None,
    pred_stft=None,
    target_stft=None,
) -> Dict[str, torch.Tensor]:
    """Compute the three spectral MSE metrics.

    ``pred_stft``/``target_stft`` may be supplied by STFT-domain models to avoid
    recomputing the single-resolution complex STFT metric.
    """
    pred_audio, target_audio = prepare_audio_pair(pred_audio, target_audio)
    if pred_stft is None or target_stft is None:
        pred_stft = audio_to_stft_channels(pred_audio, spectral_cfg)
        target_stft = audio_to_stft_channels(target_audio, spectral_cfg)
    return {
        "complex_stft_mse": complex_stft_mse_loss(pred_stft, target_stft),
        "mel_stft_mse": mel_stft_mse_loss(pred_audio, target_audio, sample_rate, spectral_cfg),
        "multi_resolution_stft_mse": multi_resolution_stft_mse_loss(
            pred_audio, target_audio, spectral_cfg
        ),
    }


@torch.no_grad()
def compute_spectral_mse_values(
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
    spectral_cfg: dict | None = None,
    pred_stft=None,
    target_stft=None,
) -> Dict[str, float]:
    """Return all spectral MSE metrics as Python floats."""
    return {
        key: float(value.detach().cpu())
        for key, value in compute_spectral_mse_losses(
            pred_audio,
            target_audio,
            sample_rate,
            spectral_cfg=spectral_cfg,
            pred_stft=pred_stft,
            target_stft=target_stft,
        ).items()
    }


@torch.no_grad()
def compute_stft_channel_stats(
    clean_audio_batches: Iterable[torch.Tensor],
    spectral_cfg: dict | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute clean STFT mean/std with output shape [1, 2, F, 1]."""
    total = None
    total_sq = None
    count = 0
    for audio in clean_audio_batches:
        stft = audio_to_stft_channels(audio.to(device, non_blocking=True), spectral_cfg).float()
        batch_sum = stft.sum(dim=(0, 3))
        batch_sum_sq = stft.square().sum(dim=(0, 3))
        batch_count = stft.shape[0] * stft.shape[-1]
        total = batch_sum if total is None else total + batch_sum
        total_sq = batch_sum_sq if total_sq is None else total_sq + batch_sum_sq
        count += batch_count
    if count == 0:
        raise ValueError("Cannot compute STFT stats from an empty iterable")
    mean = total / count
    variance = (total_sq / count) - mean.square()
    std = torch.sqrt(torch.clamp(variance, min=1e-12))
    return mean.reshape(1, *mean.shape, 1), std.reshape(1, *std.shape, 1)


def _load_audio_pair(
    denoised_path: str,
    target_path: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    denoised_np, denoised_sr = sf.read(denoised_path, dtype="float32", always_2d=True)
    target_np, target_sr = sf.read(target_path, dtype="float32", always_2d=True)
    if denoised_sr != target_sr:
        raise ValueError(
            f"Sample rates differ: denoised={denoised_sr}, target={target_sr}"
        )
    denoised = torch.from_numpy(denoised_np.T.copy())
    target = torch.from_numpy(target_np.T.copy())
    return denoised.to(device), target.to(device), denoised_sr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute spectral MSE metrics between denoised and target WAVs."
    )
    parser.add_argument("--denoised", required=True, help="Path to denoised audio")
    parser.add_argument("--target", required=True, help="Path to ground-truth audio")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc. Default: auto",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    denoised, target, sample_rate = _load_audio_pair(args.denoised, args.target, device)
    values = compute_spectral_mse_values(denoised, target, sample_rate)
    print(json.dumps({
        "denoised": args.denoised,
        "target": args.target,
        "sample_rate": sample_rate,
        "losses": values,
    }, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
