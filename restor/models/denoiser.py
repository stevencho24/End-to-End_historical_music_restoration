"""Denoiser architectures used in the restoration experiments.

The paper's proposed model is :class:`CFMDiTVelocityNet40M`, selected by
``denoiser.type: cfm_40m``.  It is intentionally listed first in
``build_denoiser`` below.  The remaining MLP, TCN, U-Net, Conformer, direct
DiT-MSE, DDPM, and alternate-capacity CFM implementations are retained to
document architecture, objective, and model-size ablations.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DIT_MSE_DENOISER_TYPES = frozenset({
    "dit_mse",
    "dit_mse_8m",
    "dit_mse_16m",
    "dit_mse_40m",
    "dit_mse_80m",
})


def build_denoiser(cfg, latent_dim=None, spectral_cfg=None):
    dtype = cfg["type"]

    # Final paper methodology: a 40M-parameter 1-D DiT velocity field trained
    # with conditional flow matching in normalized SAME-L latent space.
    if dtype == "cfm_40m":
        cfm_cfg = dict(cfg["cfm_40m"])
        cfm_cfg.pop("latent_channels", None)
        return CFMDiTVelocityNet40M(latent_channels=latent_dim, **cfm_cfg)

    # Everything below this point supports preliminary experiments or the
    # architecture/objective/capacity ablations, not the proposed CFM40 model.
    if dtype == "mlp":
        return MLPDenoiser(latent_dim, **cfg.get("mlp", {}))
    elif dtype == "tcn":
        return TCNDenoiser(1, **cfg.get("tcn", {}))
    elif dtype == "tcn_8m":
        return TCNDenoiser8M(1, **cfg.get("tcn_8m", {}))
    elif dtype in {"unet", "unet1d"}:
        fft_size = int((spectral_cfg or {}).get("fft_size", 1024))
        return UNet2DSpectrogramDenoiser(
            freq_bins=fft_size // 2 + 1,
            **cfg.get("unet", cfg.get("unet1d", {})),
        )
    elif dtype == "conformer":
        fft_size = int((spectral_cfg or {}).get("fft_size", 1024))
        return SpectrogramConformerDenoiser(
            freq_bins=fft_size // 2 + 1,
            **cfg["conformer"],
        )
    elif dtype == "conformer_8m":
        fft_size = int((spectral_cfg or {}).get("fft_size", 1024))
        return SpectrogramConformerDenoiser8M(
            freq_bins=fft_size // 2 + 1,
            **cfg.get("conformer_8m", {}),
        )
    elif dtype in DIT_MSE_DENOISER_TYPES:
        dit_classes = {
            "dit_mse": DiTMSEDenoiser,
            "dit_mse_8m": DiTMSEDenoiser8M,
            "dit_mse_16m": DiTMSEDenoiser16M,
            "dit_mse_40m": DiTMSEDenoiser40M,
            "dit_mse_80m": DiTMSEDenoiser80M,
        }
        # Index with the active type. This prevents one capacity variant from
        # inheriting architecture or optimizer fields from another variant's
        # config block.
        dit_cfg = dict(cfg[dtype])
        dit_cfg.pop("latent_channels", None)
        return dit_classes[dtype](latent_channels=latent_dim, **dit_cfg)
    elif dtype == "ddpm":
        ddpm_cfg = dict(cfg["ddpm"])
        ddpm_cfg.pop("latent_channels", None)
        return DDPMDiTDenoiser(latent_channels=latent_dim, **ddpm_cfg)
    elif dtype == "cfm":
        cfm_cfg = dict(cfg["cfm"])
        cfm_cfg.pop("latent_channels", None)
        return CFMDiTVelocityNet(latent_channels=latent_dim, **cfm_cfg)
    elif dtype == "ddpm_8m":
        ddpm_cfg = dict(cfg["ddpm_8m"])
        ddpm_cfg.pop("latent_channels", None)
        return DDPMDiTDenoiser8M(latent_channels=latent_dim, **ddpm_cfg)
    elif dtype == "cfm_8m":
        cfm_cfg = dict(cfg["cfm_8m"])
        cfm_cfg.pop("latent_channels", None)
        return CFMDiTVelocityNet8M(latent_channels=latent_dim, **cfm_cfg)
    elif dtype == "cfm_4m":
        cfm_cfg = dict(cfg["cfm_4m"])
        cfm_cfg.pop("latent_channels", None)
        return CFMDiTVelocityNet4M(latent_channels=latent_dim, **cfm_cfg)
    elif dtype == "cfm_16m":
        cfm_cfg = dict(cfg["cfm_16m"])
        cfm_cfg.pop("latent_channels", None)
        return CFMDiTVelocityNet16M(latent_channels=latent_dim, **cfm_cfg)
    elif dtype == "cfm_80m":
        cfm_cfg = dict(cfg["cfm_80m"])
        cfm_cfg.pop("latent_channels", None)
        return CFMDiTVelocityNet80M(latent_channels=latent_dim, **cfm_cfg)
    else:
        raise ValueError(f"Unknown denoiser type: {dtype}")


def _activation(name):
    return {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }[name]()


def _check_latent_shape(z, channels, module_name):
    if z.ndim != 3:
        raise ValueError(
            f"{module_name} expects z with shape [B, D, T], got {tuple(z.shape)}"
        )
    if z.shape[1] != channels:
        raise ValueError(
            f"{module_name} expected {channels} latent channels, got {z.shape[1]}"
        )


def _zero_init(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _group_norm(channels, max_groups):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SameLengthConv1d(nn.Module):
    """Conv1d wrapper that preserves the input time length for odd/even kernels."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=1, groups=1):
        super().__init__()
        self.pad_total = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x):
        left = self.pad_total // 2
        right = self.pad_total - left
        return self.conv(F.pad(x, (left, right)))


# ------------------------------------------------------------------ MLP ----


class MLPDenoiser(nn.Module):
    """No-temporal-context control model for latent channel remapping.

    This baseline applies the same MLP independently to each latent time frame:
    [B, D, T] -> [B, T, D] -> MLP(D) -> [B, D, T]. It deliberately cannot see
    neighboring frames, so it tests whether per-frame SAME-L channel correction
    alone is enough before comparing against TCN, U-Net, and Conformer models.
    """

    def __init__(self, latent_dim, hidden_dims=(1024, 1024, 1024),
                 activation="gelu", dropout=0.0, layer_norm=False,
                 zero_init_final=True):
        super().__init__()
        layers = []
        if layer_norm:
            layers.append(nn.LayerNorm(latent_dim))
        in_dim = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), _activation(activation)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        final = nn.Linear(in_dim, latent_dim)
        if zero_init_final:
            _zero_init(final)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, z):
        """z: (B, D, T) -> (B, D, T)"""
        _check_latent_shape(z, self.latent_dim, "MLPDenoiser")
        x = z.permute(0, 2, 1)        # (B, T, D)
        x = self.net(x) + x           # residual
        return x.permute(0, 2, 1)     # (B, D, T)


# ------------------------------------------------------------------ TCN ----


class TCNBlock(nn.Module):
    """Residual non-causal dilated Conv1d block at one temporal scale.

    GroupNorm keeps training stable for small batches; the dilated Conv1d is
    the temporal-context mechanism; the 1x1 Conv1d remixes latent channels.
    The residual add makes the block learn a restoration correction, not a
    full latent rewrite.
    """

    def __init__(self, channels, kernel_size, dilation, dropout,
                 activation="gelu", norm_groups=8):
        super().__init__()
        self.net = nn.Sequential(
            _group_norm(channels, norm_groups),
            _activation(activation),
            SameLengthConv1d(channels, channels, kernel_size, dilation=dilation),
            _group_norm(channels, norm_groups),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TCNDenoiser(nn.Module):
    """Single-resolution temporal-context ablation for SAME-L restoration.

    The TCN isolates long temporal receptive field from U-Net-style hierarchy:
    it can use neighboring latent frames through non-causal dilated convolutions,
    but it does not downsample, upsample, or use attention.

    As an ablation, it tests whether offline bidirectional temporal convolution
    alone is enough: stronger than a per-frame MLP, but simpler than multiscale
    U-Net or attention-based Conformer models.
    """

    def __init__(self, latent_dim, hidden_channels=512, kernel_size=3,
                 dilations=(1, 2, 4, 8, 16, 32, 64, 128),
                 dropout=0.1, activation="gelu", norm_groups=8,
                 zero_init_final=True):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("TCNDenoiser kernel_size must be positive")
        self.latent_dim = latent_dim
        self.input_proj = nn.Conv1d(latent_dim, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(
                hidden_channels,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
                activation=activation,
                norm_groups=norm_groups,
            )
            for d in dilations
        ])
        self.output_norm = _group_norm(hidden_channels, norm_groups)
        self.output_act = _activation(activation)
        self.output_proj = nn.Conv1d(hidden_channels, latent_dim, 1)
        if zero_init_final:
            _zero_init(self.output_proj)

    def forward(self, z):
        """z: (B, D, T) -> (B, D, T)"""
        _check_latent_shape(z, self.latent_dim, "TCNDenoiser")
        x = self.input_proj(z)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(self.output_act(self.output_norm(x)))
        return z + x


class TCNBlock8M(nn.Module):
    """Width-reduced residual TCN block for the 8M waveform model.

    The block retains the original non-causal dilated Conv1d, pointwise Conv1d,
    two GroupNorms, dropout, and residual path. Only the channel width supplied
    by TCNDenoiser8M is reduced. Keeping the exponentially increasing dilation
    stack follows the generic TCN design and preserves its effective memory:
    https://arxiv.org/abs/1803.01271.

    Conv-TasNet found that larger receptive fields improve audio separation and
    that, at a fixed receptive field, deeper TCNs perform better. This motivates
    narrowing every block instead of removing the highest-dilation block:
    https://arxiv.org/abs/1809.07454 (Sections II-D and IV-B).

    For channel width C and kernel size K, one block contains
    (K + 1)*C^2 + 6*C parameters. With C=496 and K=3, this is
    4*496^2 + 6*496 = 987,040 parameters.
    """

    def __init__(self, channels, kernel_size, dilation, dropout,
                 activation="gelu", norm_groups=8):
        super().__init__()
        self.net = nn.Sequential(
            _group_norm(channels, norm_groups),
            _activation(activation),
            SameLengthConv1d(
                channels, channels, kernel_size, dilation=dilation
            ),
            _group_norm(channels, norm_groups),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TCNDenoiser8M(nn.Module):
    """TCN scaled just below 8M parameters without reducing temporal context.

    Reduction method and rationale:
      * Reduce hidden_channels from 512 to 496, a 3.125% width reduction. The
        dominant convolution weights and MACs scale quadratically with width,
        so this removes about 6.15% of their cost with a small capacity change.
      * Keep 496 divisible by norm_groups=8, preserving eight GroupNorm groups.
        The closer 499-channel solution would make _group_norm fall back to one
        group and would therefore change normalization behavior.
      * Preserve all eight residual blocks, kernel size 3, and dilations
        1..128. Their receptive field remains 511 waveform samples; deleting
        dilation 128 would halve it to 255 samples. Conv-TasNet reports benefits
        from both greater TCN depth and larger receptive fields in audio:
        https://arxiv.org/abs/1809.07454 (Sections II-D and IV-B).

    Exact default count for the one-channel waveform path constructed by
    build_denoiser:
      input projection = 1*496 + 496 = 992
      eight blocks = 8 * 987,040 = 7,896,320
      output GroupNorm = 2*496 = 992
      output projection = 496*1 + 1 = 497
      total = 992 + 7,896,320 + 992 + 497 = 7,898,801 parameters.

    Relative to the 512-channel model's 8,415,745 parameters, this saves
    516,944 parameters (6.14%). Per-timestep convolution MACs in each block
    fall from 4*512^2 = 1,048,576 to 4*496^2 = 984,064 (6.15%), while hidden
    activation storage falls by 3.125%. Actual speed and energy savings remain
    hardware- and workload-dependent.

    The cited results are for sequence modeling and speech separation; they
    motivate the scaling choice but do not guarantee music-restoration quality.
    """

    def __init__(self, latent_dim, hidden_channels=496, kernel_size=3,
                 dilations=(1, 2, 4, 8, 16, 32, 64, 128),
                 dropout=0.1, activation="gelu", norm_groups=8,
                 zero_init_final=True):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("TCNDenoiser8M kernel_size must be positive")
        self.latent_dim = latent_dim
        self.input_proj = nn.Conv1d(latent_dim, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            TCNBlock8M(
                hidden_channels,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
                activation=activation,
                norm_groups=norm_groups,
            )
            for d in dilations
        ])
        self.output_norm = _group_norm(hidden_channels, norm_groups)
        self.output_act = _activation(activation)
        self.output_proj = nn.Conv1d(hidden_channels, latent_dim, 1)
        if zero_init_final:
            _zero_init(self.output_proj)

    def forward(self, z):
        """z: [B, 1, T] -> [B, 1, T] for the configured waveform path."""
        _check_latent_shape(z, self.latent_dim, "TCNDenoiser8M")
        x = self.input_proj(z)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(self.output_act(self.output_norm(x)))
        return z + x


# --------------------------------------------------------------- U-Net ----


class UNetResBlock1D(nn.Module):
    """Residual Conv1d block used by the latent-time U-Net.

    Same-length Conv1d preserves the SAME latent time axis, GroupNorm supports
    small batches, and the optional 1x1 skip projection keeps residual learning
    valid when U-Net levels change channel count.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 dropout=0.0, activation="gelu", norm_groups=8):
        super().__init__()
        self.norm1 = _group_norm(in_channels, norm_groups)
        self.act1 = _activation(activation)
        self.conv1 = SameLengthConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.norm2 = _group_norm(out_channels, norm_groups)
        self.act2 = _activation(activation)
        self.drop = nn.Dropout(dropout)
        self.conv2 = SameLengthConv1d(out_channels, out_channels, kernel_size)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        y = self.conv1(self.act1(self.norm1(x)))
        y = self.drop(y)
        y = self.conv2(self.drop(self.act2(self.norm2(y))))
        return self.skip(x) + y


class UNet1DDenoiser(nn.Module):
    """Multiscale deterministic latent-time U-Net.

    This is the strongest convolutional deterministic baseline: encoder skips
    preserve local SAME-L detail while deeper levels see coarser musical context.
    Upsampling uses interpolation plus Conv1d blocks to keep output length stable.

    As an ablation, it asks whether multiscale temporal context and skip-carried
    local detail improve over the single-resolution TCN before adding attention.
    """

    def __init__(self, latent_dim, base_channels=256,
                 channel_mults=(1, 2, 4), kernel_size=3,
                 bottleneck_dilations=(1, 2, 4), dropout=0.1,
                 activation="gelu", norm_groups=8, zero_init_final=True):
        super().__init__()
        if not channel_mults:
            raise ValueError("UNet1DDenoiser requires at least one channel multiplier")
        self.latent_dim = latent_dim
        self.input_proj = nn.Conv1d(latent_dim, base_channels, 1)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        skip_channels = []
        in_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.enc_blocks.append(
                UNetResBlock1D(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                    norm_groups=norm_groups,
                )
            )
            skip_channels.append(out_ch)
            if i < len(channel_mults) - 1:
                self.downs.append(
                    nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
                )
            in_ch = out_ch

        self.bottleneck = nn.ModuleList([
            UNetResBlock1D(
                in_ch,
                in_ch,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
                activation=activation,
                norm_groups=norm_groups,
            )
            for d in bottleneck_dilations
        ])

        self.dec_blocks = nn.ModuleList()
        current_ch = in_ch
        for skip_ch in reversed(skip_channels):
            self.dec_blocks.append(
                UNetResBlock1D(
                    current_ch + skip_ch,
                    skip_ch,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                    norm_groups=norm_groups,
                )
            )
            current_ch = skip_ch

        self.output_norm = _group_norm(current_ch, norm_groups)
        self.output_act = _activation(activation)
        self.output_proj = nn.Conv1d(current_ch, latent_dim, 1)
        if zero_init_final:
            _zero_init(self.output_proj)

    def forward(self, z):
        """z: (B, D, T) -> (B, D, T)"""
        _check_latent_shape(z, self.latent_dim, "UNet1DDenoiser")
        x = self.input_proj(z)
        skips = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)

        for block in self.bottleneck:
            x = block(x)

        for block in self.dec_blocks:
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
            x = block(torch.cat([x, skip], dim=1))

        x = self.output_proj(self.output_act(self.output_norm(x)))
        if x.shape[-1] != z.shape[-1]:
            x = F.interpolate(x, size=z.shape[-1], mode="linear", align_corners=False)
        return z + x


class UNetResBlock2D(nn.Module):
    """Residual Conv2d block for complex-STFT spectrograms."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 dropout=0.0, activation="gelu", norm_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = _group_norm(in_channels, norm_groups)
        self.act1 = _activation(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = _group_norm(out_channels, norm_groups)
        self.act2 = _activation(activation)
        self.drop = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        y = self.conv1(self.act1(self.norm1(x)))
        y = self.drop(y)
        y = self.conv2(self.drop(self.act2(self.norm2(y))))
        return self.skip(x) + y


class UNet2DSpectrogramDenoiser(nn.Module):
    """2D U-Net that predicts normalized clean complex STFT channels.

    Input/output shape is [B, 2, F, T] with real/imag channels. The final output
    is residual: corrupted normalized STFT + predicted correction.
    """

    def __init__(self, freq_bins, base_channels=32, channel_mults=(1, 2, 4, 8),
                 kernel_size=3, dropout=0.1, activation="gelu",
                 norm_groups=8, zero_init_final=True, **unused):
        super().__init__()
        if not channel_mults:
            raise ValueError("UNet2DSpectrogramDenoiser requires channel_mults")
        self.freq_bins = freq_bins
        self.input_proj = nn.Conv2d(2, base_channels, 1)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        skip_channels = []
        in_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.enc_blocks.append(
                UNetResBlock2D(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                    norm_groups=norm_groups,
                )
            )
            skip_channels.append(out_ch)
            if i < len(channel_mults) - 1:
                self.downs.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            in_ch = out_ch

        self.bottleneck = UNetResBlock2D(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation,
            norm_groups=norm_groups,
        )

        self.dec_blocks = nn.ModuleList()
        current_ch = in_ch
        for skip_ch in reversed(skip_channels):
            self.dec_blocks.append(
                UNetResBlock2D(
                    current_ch + skip_ch,
                    skip_ch,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                    norm_groups=norm_groups,
                )
            )
            current_ch = skip_ch

        self.output_norm = _group_norm(current_ch, norm_groups)
        self.output_act = _activation(activation)
        self.output_proj = nn.Conv2d(current_ch, 2, 1)
        if zero_init_final:
            _zero_init(self.output_proj)

    def forward(self, stft):
        """stft: [B, 2, F, T] -> [B, 2, F, T]."""
        if stft.ndim != 4 or stft.shape[1] != 2:
            raise ValueError(
                "UNet2DSpectrogramDenoiser expects [B, 2, F, T], "
                f"got {tuple(stft.shape)}"
            )
        x = self.input_proj(stft)
        skips = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)

        x = self.bottleneck(x)

        for block in self.dec_blocks:
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = block(torch.cat([x, skip], dim=1))

        x = self.output_proj(self.output_act(self.output_norm(x)))
        if x.shape[-2:] != stft.shape[-2:]:
            x = F.interpolate(
                x,
                size=stft.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return stft + x


# ------------------------------------------------------------- Conformer ----


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(self.norm(x))


class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, 1),
            nn.GLU(dim=1),
            nn.Conv1d(
                d_model, d_model, kernel_size,
                padding=kernel_size // 2, groups=d_model,
            ),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, T, D)
        x = self.norm(x)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, conv_kernel_size, dropout):
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_dim, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForward(d_model, ff_dim, dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        res = x
        x = self.norm_attn(x)
        x = res + self.attn_drop(self.attn(x, x, x)[0])
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)


class SpectrogramConformerDenoiser(nn.Module):
    """Conformer over STFT time frames with learned feature projections.

    Input/output shape is [B, 2, F, T]. Each time frame is represented by
    concatenated real/imag frequency bins, projected to the configured
    conformer width, then projected back to complex-STFT channels.
    """

    def __init__(self, freq_bins, d_model=512, num_layers=4, num_heads=8,
                 ff_dim=2048, conv_kernel_size=31, dropout=0.1,
                 zero_init_final=True):
        super().__init__()
        self.freq_bins = freq_bins
        self.feature_dim = 2 * freq_bins
        self.d_model = d_model
        self.input_proj = nn.Linear(self.feature_dim, d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, ff_dim, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, self.feature_dim)
        if zero_init_final:
            _zero_init(self.output_proj)

    def forward(self, stft):
        """stft: [B, 2, F, T] -> [B, 2, F, T]."""
        if stft.ndim != 4 or stft.shape[1] != 2:
            raise ValueError(
                "SpectrogramConformerDenoiser expects [B, 2, F, T], "
                f"got {tuple(stft.shape)}"
            )
        if stft.shape[2] != self.freq_bins:
            raise ValueError(f"Expected {self.freq_bins} frequency bins, got {stft.shape[2]}")
        bsz, _, freq, frames = stft.shape
        x = stft.permute(0, 3, 1, 2).reshape(bsz, frames, 2 * freq)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        x = x.reshape(bsz, frames, 2, freq).permute(0, 2, 3, 1)
        return stft + x


class ConformerBlock8M(nn.Module):
    """Width-reduced Conformer block used by the 8M spectrogram model.

    The block structure is deliberately unchanged: Macaron-style half-step
    FFN -> self-attention -> depthwise convolution -> half-step FFN -> norm.
    Only the widths supplied by SpectrogramConformerDenoiser8M are reduced.
    This follows the original Conformer design, whose small/medium/large models
    were selected by jointly scaling depth, model dimension, and attention
    heads while retaining the complete block structure:
    https://arxiv.org/abs/2005.08100 (Sections 2.4 and 3.2, Table 1).

    For model width D, FFN width M, and depthwise kernel K, this implementation
    has 7*D^2 + 4*D*M + 2*M + D*K + 22*D trainable parameters per block.
    At D=288, M=1088, and K=31, that is 1,851,424 parameters.
    """

    def __init__(self, d_model, num_heads, ff_dim, conv_kernel_size, dropout):
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_dim, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForward(d_model, ff_dim, dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        residual = x
        x = self.norm_attn(x)
        x = residual + self.attn_drop(
            self.attn(x, x, x, need_weights=False)[0]
        )
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)


class SpectrogramConformerDenoiser8M(nn.Module):
    """Eight-million-parameter Conformer for complex-STFT restoration.

    Reduction method and rationale:
      * Preserve all four complete Conformer blocks instead of deleting half
        the network depth. Directly reducing a Conformer encoder to two blocks
        has been reported to hurt performance, while retaining effective depth
        through repeated blocks performed better at the same parameter count:
        https://arxiv.org/abs/2209.08326 (Sections 3.1 and 5.2).
      * Reduce d_model from 512 to 288. Attention, pointwise convolution, and
        FFN weights scale mainly with d_model squared, so width reduction gives
        the largest parameter saving without changing the Conformer topology.
      * Reduce ff_dim from 2048 to 1088. The resulting expansion ratio is
        1088/288 = 3.78, kept close to the original Conformer's 4x FFN expansion:
        https://arxiv.org/abs/2005.08100 (Section 2.3, Figure 4).
      * Keep eight attention heads, the 31-frame depthwise convolution, and all
        STFT frequency bins. Changing the head count does not change parameter
        count at fixed d_model, while shrinking the depthwise kernel saves few
        parameters and reducing frequency bins would discard spectral detail.

    Exact count for a 1024-point FFT (F=513, feature_dim=2*F=1026):
      four blocks = 4 * 1,851,424 = 7,405,696
      input/output projections = 2*1026*288 + 288 + 1026 = 592,290
      total = 7,405,696 + 592,290 = 7,997,986 trainable parameters.

    The literature results are from speech recognition and motivate the scaling
    choice; restoration quality for this complex-STFT task must still be
    established experimentally.
    """

    def __init__(self, freq_bins, d_model=288, num_layers=4, num_heads=8,
                 ff_dim=1088, conv_kernel_size=31, dropout=0.1,
                 zero_init_final=True):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )
        self.freq_bins = freq_bins
        self.feature_dim = 2 * freq_bins
        self.d_model = d_model
        self.input_proj = nn.Linear(self.feature_dim, d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock8M(
                d_model, num_heads, ff_dim, conv_kernel_size, dropout
            )
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, self.feature_dim)
        if zero_init_final:
            _zero_init(self.output_proj)

    def forward(self, stft):
        """stft: [B, 2, F, T] -> [B, 2, F, T]."""
        if stft.ndim != 4 or stft.shape[1] != 2:
            raise ValueError(
                "SpectrogramConformerDenoiser8M expects [B, 2, F, T], "
                f"got {tuple(stft.shape)}"
            )
        if stft.shape[2] != self.freq_bins:
            raise ValueError(
                f"Expected {self.freq_bins} frequency bins, got {stft.shape[2]}"
            )
        bsz, _, freq, frames = stft.shape
        x = stft.permute(0, 3, 1, 2).reshape(bsz, frames, 2 * freq)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        x = x.reshape(bsz, frames, 2, freq).permute(0, 2, 3, 1)
        return stft + x


class ConformerDenoiser(nn.Module):
    """Strong deterministic sequence baseline with local conv + global attention.

    Unlike the TCN's fixed dilated filters or the U-Net's multiscale hierarchy,
    the Conformer treats latent frames as a sequence: self-attention can relate
    distant musical events, the depthwise convolution keeps local audio texture,
    and feed-forward layers mix SAME channels. This tests whether global
    content-aware context plus local convolution improves deterministic latent
    restoration before moving to stochastic DDPM sampling.

    It keeps the simple deterministic API forward(z) -> z_hat.
    """

    def __init__(self, d_model, num_layers, num_heads, ff_dim,
                 conv_kernel_size, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, ff_dim, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, z):
        """z: (B, D, T) -> (B, D, T)"""
        _check_latent_shape(z, self.d_model, "ConformerDenoiser")
        x = z.permute(0, 2, 1)       # (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = self.proj(x)
        return x.permute(0, 2, 1) + z  # residual


# -------------------------------------------------------------- DDPM DiT ----


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class SinusoidalPositionalEncoding1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, length, device):
        half = self.dim // 2
        pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = pos * freqs[None]
        enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            enc = torch.nn.functional.pad(enc, (0, 1))
        return enc.unsqueeze(0)


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock1D(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0,
                 dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(d_model, elementwise_affine=False)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, time_emb):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN(time_emb).chunk(6, dim=1)
        )

        attn_in = _modulate(self.norm_attn(x), shift_attn, scale_attn)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + gate_attn.unsqueeze(1) * self.attn_drop(attn_out)

        mlp_in = _modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class DiTBlock1D8M(nn.Module):
    """Lightweight adaLN-zero DiT block for the ~8M DDPM/CFM variants.

    Identical to DiTBlock1D but uses mlp_ratio=2.0 by default to target
    the ~8M parameter budget at d_model=288, num_layers=6. Keeping the
    adaLN-zero conditioning unchanged preserves the DDPM/CFM training
    objective while halving the MLP parameter cost.
    """

    def __init__(self, d_model, num_heads, mlp_ratio=2.0,
                 dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(d_model, elementwise_affine=False)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, time_emb):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN(time_emb).chunk(6, dim=1)
        )

        attn_in = _modulate(self.norm_attn(x), shift_attn, scale_attn)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + gate_attn.unsqueeze(1) * self.attn_drop(attn_out)

        mlp_in = _modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class DeterministicDiTBlock1D(nn.Module):
    """Pre-norm transformer block for latent MSE DiT restoration.

    DDPMDiTDenoiser uses AdaLN-zero timestep conditioning because it predicts
    noise at a sampled diffusion time.  This deterministic ablation removes
    timestep/noise conditioning, so the block is a plain self-attention + MLP
    transformer over SAME latent frames.
    """

    def __init__(self, d_model, num_heads, mlp_ratio=4.0,
                 dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_in = self.norm_attn(x)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + self.attn_drop(attn_out)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class DeterministicDiTBlock1D8M(nn.Module):
    """Lightweight pre-norm transformer block for DiTMSEDenoiser8M (~7.4M total).

    Identical to DeterministicDiTBlock1D but targets d_model=384 and
    mlp_ratio=2.0 by default to hit the ~8M parameter budget.
    """

    def __init__(self, d_model, num_heads, mlp_ratio=2.0,
                 dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_in = self.norm_attn(x)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + self.attn_drop(attn_out)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class DiTMSEDenoiser(nn.Module):
    """Deterministic SAME latent DiT ablation trained with z_clean MSE.

    Input and output are normalized SAME-L latents with shape [B, 256, T].  The
    model removes DDPM's timestep/noise conditioning and predicts a residual
    cleanup over z_cond, making the ablation about deterministic DiT restoration
    versus the DDPM objective/sampler.
    """

    def __init__(self, latent_channels=256, d_model=768, num_layers=12,
                 num_attention_heads=12, mlp_ratio=4.0, dropout=0.0,
                 attention_dropout=0.0, zero_init_final=True, **unused):
        super().__init__()
        if latent_channels != 256:
            raise ValueError("DiTMSEDenoiser is SAME-L-only and expects 256 latent channels")
        self.latent_channels = latent_channels
        self.input_proj = nn.Linear(latent_channels, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding1D(d_model)
        self.blocks = nn.ModuleList([
            DeterministicDiTBlock1D(
                d_model=d_model,
                num_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_channels)
        if zero_init_final:
            _zero_init(self.output_proj)

    def forward(self, z_cond):
        """z_cond: [B, 256, T] -> z_pred: [B, 256, T]."""
        if z_cond.ndim != 3:
            raise ValueError("DiTMSEDenoiser expects rank-3 latent tensors")
        if z_cond.shape[1] != self.latent_channels:
            raise ValueError(f"Expected 256 latent channels, got {z_cond.shape[1]}")
        x = z_cond.permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + self.pos_encoding(x.shape[1], x.device).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
        delta = self.output_proj(self.final_norm(x)).permute(0, 2, 1)
        return z_cond + delta


class DiTMSEDenoiser8M(nn.Module):
    """Lightweight deterministic SAME latent DiT ablation (~7.4M parameters).

    Identical to DiTMSEDenoiser in structure and training objective but uses
    d_model=384, num_layers=6, num_attention_heads=6, mlp_ratio=2.0 to target
    ~7.4M parameters vs the full model's ~85.5M. This isolates whether model
    capacity or architecture type drives restoration quality.

    Per the DiT paper (Peebles & Xie, ICCV 2023), GFlops predict quality better
    than parameter count. At d_model=384 x 6 layers the GFlop budget is
    proportionally higher than d_model=256 x 12 layers at similar param count,
    making this the better small-scale config.
    """

    def __init__(self, latent_channels=256, d_model=384, num_layers=6,
                 num_attention_heads=6, mlp_ratio=2.0, dropout=0.0,
                 attention_dropout=0.0, zero_init_final=True, **unused):
        super().__init__()
        if latent_channels != 256:
            raise ValueError("DiTMSEDenoiser8M is SAME-L-only and expects 256 latent channels")
        if d_model % num_attention_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_attention_heads={num_attention_heads}"
            )
        self.latent_channels = latent_channels
        self.input_proj = nn.Linear(latent_channels, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding1D(d_model)
        self.blocks = nn.ModuleList([
            DeterministicDiTBlock1D8M(
                d_model=d_model,
                num_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_channels)
        if zero_init_final:
            _zero_init(self.output_proj)

    def forward(self, z_cond):
        """z_cond: [B, 256, T] -> z_pred: [B, 256, T]."""
        model_name = type(self).__name__
        if z_cond.ndim != 3:
            raise ValueError(f"{model_name} expects rank-3 latent tensors")
        if z_cond.shape[1] != self.latent_channels:
            raise ValueError(
                f"{model_name} expected 256 latent channels, got {z_cond.shape[1]}"
            )
        x = z_cond.permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + self.pos_encoding(x.shape[1], x.device).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
        delta = self.output_proj(self.final_norm(x)).permute(0, 2, 1)
        return z_cond + delta


class DiTMSEDenoiser16M(DiTMSEDenoiser8M):
    """Fixed-capacity 16M deterministic SAME latent DiT.

    The 8M model's six-layer, ratio-2, pre-norm residual architecture and
    64-channel attention heads are preserved. Capacity is added only through
    width: d_model=576 and nine heads produce 16,260,160 trainable parameters.

    Architecture fields are validated rather than treated as freely tunable
    defaults. This keeps ``dit_mse_16m`` stable if another model's defaults or
    shared configuration conventions change in the future.
    """

    D_MODEL = 576
    NUM_LAYERS = 6
    NUM_ATTENTION_HEADS = 9
    MLP_RATIO = 2.0
    HEAD_DIM = 64
    EXPECTED_PARAMETER_COUNT = 16_260_160

    def __init__(self, latent_channels=256, d_model=D_MODEL,
                 num_layers=NUM_LAYERS,
                 num_attention_heads=NUM_ATTENTION_HEADS,
                 mlp_ratio=MLP_RATIO, dropout=0.0,
                 attention_dropout=0.0, zero_init_final=True, **unused):
        architecture = {
            "d_model": d_model,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "mlp_ratio": float(mlp_ratio),
        }
        expected = {
            "d_model": self.D_MODEL,
            "num_layers": self.NUM_LAYERS,
            "num_attention_heads": self.NUM_ATTENTION_HEADS,
            "mlp_ratio": self.MLP_RATIO,
        }
        if architecture != expected:
            raise ValueError(
                "DiTMSEDenoiser16M has a fixed architecture; expected "
                f"{expected}, got {architecture}. Define a separate model type "
                "for architecture experiments."
            )
        if latent_channels != 256:
            raise ValueError(
                "DiTMSEDenoiser16M is SAME-L-only and expects 256 latent channels"
            )
        if d_model // num_attention_heads != self.HEAD_DIM:
            raise ValueError(
                "DiTMSEDenoiser16M must preserve 64 channels per attention head"
            )

        super().__init__(
            latent_channels=latent_channels,
            d_model=d_model,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            zero_init_final=zero_init_final,
            **unused,
        )
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != self.EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                "DiTMSEDenoiser16M parameter-count invariant failed: expected "
                f"{self.EXPECTED_PARAMETER_COUNT:,}, got {parameter_count:,}"
            )


class DiTMSEDenoiser40M(DiTMSEDenoiser8M):
    """Fixed-capacity 40M deterministic SAME latent DiT.

    This follows the same scaling rule as the 16M and 80M variants: preserve
    six ratio-2 pre-norm residual blocks and 64-channel attention heads, and
    add capacity only through width. A 896-wide backbone with fourteen heads
    contains exactly 39,056,000 trainable parameters, which is the closest
    64-head-dimension configuration to the 40M target.

    Architecture fields are validated rather than treated as freely tunable
    defaults. This keeps ``dit_mse_40m`` stable if another model's defaults or
    shared configuration conventions change in the future.
    """

    D_MODEL = 896
    NUM_LAYERS = 6
    NUM_ATTENTION_HEADS = 14
    MLP_RATIO = 2.0
    HEAD_DIM = 64
    EXPECTED_PARAMETER_COUNT = 39_056_000

    def __init__(self, latent_channels=256, d_model=D_MODEL,
                 num_layers=NUM_LAYERS,
                 num_attention_heads=NUM_ATTENTION_HEADS,
                 mlp_ratio=MLP_RATIO, dropout=0.0,
                 attention_dropout=0.0, zero_init_final=True, **unused):
        architecture = {
            "d_model": d_model,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "mlp_ratio": float(mlp_ratio),
        }
        expected = {
            "d_model": self.D_MODEL,
            "num_layers": self.NUM_LAYERS,
            "num_attention_heads": self.NUM_ATTENTION_HEADS,
            "mlp_ratio": self.MLP_RATIO,
        }
        if architecture != expected:
            raise ValueError(
                "DiTMSEDenoiser40M has a fixed architecture; expected "
                f"{expected}, got {architecture}. Define a separate model type "
                "for architecture experiments."
            )
        if latent_channels != 256:
            raise ValueError(
                "DiTMSEDenoiser40M is SAME-L-only and expects 256 latent channels"
            )
        if d_model // num_attention_heads != self.HEAD_DIM:
            raise ValueError(
                "DiTMSEDenoiser40M must preserve 64 channels per attention head"
            )

        super().__init__(
            latent_channels=latent_channels,
            d_model=d_model,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            zero_init_final=zero_init_final,
            **unused,
        )
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != self.EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                "DiTMSEDenoiser40M parameter-count invariant failed: expected "
                f"{self.EXPECTED_PARAMETER_COUNT:,}, got {parameter_count:,}"
            )


class DiTMSEDenoiser80M(DiTMSEDenoiser8M):
    """Fixed-capacity 80M deterministic SAME latent DiT.

    This follows the 8M-to-16M scaling rule exactly: preserve six ratio-2
    pre-norm residual blocks and 64-channel attention heads, and add capacity
    only through width. A 1,280-wide backbone with twenty heads contains
    exactly 79,387,136 trainable parameters.
    """

    D_MODEL = 1280
    NUM_LAYERS = 6
    NUM_ATTENTION_HEADS = 20
    MLP_RATIO = 2.0
    HEAD_DIM = 64
    EXPECTED_PARAMETER_COUNT = 79_387_136

    def __init__(self, latent_channels=256, d_model=D_MODEL,
                 num_layers=NUM_LAYERS,
                 num_attention_heads=NUM_ATTENTION_HEADS,
                 mlp_ratio=MLP_RATIO, dropout=0.0,
                 attention_dropout=0.0, zero_init_final=True, **unused):
        architecture = {
            "d_model": d_model,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "mlp_ratio": float(mlp_ratio),
        }
        expected = {
            "d_model": self.D_MODEL,
            "num_layers": self.NUM_LAYERS,
            "num_attention_heads": self.NUM_ATTENTION_HEADS,
            "mlp_ratio": self.MLP_RATIO,
        }
        if architecture != expected:
            raise ValueError(
                "DiTMSEDenoiser80M has a fixed architecture; expected "
                f"{expected}, got {architecture}. Define a separate model type "
                "for architecture experiments."
            )
        if latent_channels != 256:
            raise ValueError(
                "DiTMSEDenoiser80M is SAME-L-only and expects 256 latent channels"
            )
        if d_model // num_attention_heads != self.HEAD_DIM:
            raise ValueError(
                "DiTMSEDenoiser80M must preserve 64 channels per attention head"
            )

        super().__init__(
            latent_channels=latent_channels,
            d_model=d_model,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            zero_init_final=zero_init_final,
            **unused,
        )
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != self.EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                "DiTMSEDenoiser80M parameter-count invariant failed: expected "
                f"{self.EXPECTED_PARAMETER_COUNT:,}, got {parameter_count:,}"
            )


class DDPMDiTDenoiser(nn.Module):
    """1D DiT epsilon predictor for SAME latent DDPM restoration."""

    def __init__(self, latent_channels=256, condition_channels=256,
                 d_model=768, num_layers=12, num_attention_heads=12,
                 mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0,
                 time_embed_dim=None, **unused):
        super().__init__()
        if latent_channels != 256 or condition_channels != 256:
            raise ValueError(
                "DDPMDiTDenoiser is SAME-L-only and expects 256 latent/condition channels"
            )
        time_embed_dim = time_embed_dim or d_model
        if time_embed_dim != d_model:
            raise ValueError("time_embed_dim must match d_model in this implementation")

        self.latent_channels = latent_channels
        self.condition_channels = condition_channels
        self.input_proj = nn.Linear(latent_channels + condition_channels, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding1D(d_model)
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.blocks = nn.ModuleList([
            DiTBlock1D(
                d_model=d_model,
                num_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_channels)

    def forward(self, z_t, t, z_cond):
        """
        z_t: [B, 256, T], t: [B], z_cond: [B, 256, T]
        returns eps_pred: [B, 256, T]
        """
        if z_t.ndim != 3 or z_cond.ndim != 3:
            raise ValueError("DDPMDiTDenoiser expects rank-3 latent tensors")
        if z_t.shape != z_cond.shape:
            raise ValueError(
                f"z_t and z_cond shapes must match, got {z_t.shape} and {z_cond.shape}"
            )
        if z_t.shape[1] != self.latent_channels:
            raise ValueError(f"Expected 256 latent channels, got {z_t.shape[1]}")
        if t.shape != (z_t.shape[0],):
            raise ValueError(f"Expected t shape {(z_t.shape[0],)}, got {tuple(t.shape)}")

        x = torch.cat([z_t, z_cond], dim=1).permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + self.pos_encoding(x.shape[1], x.device).to(dtype=x.dtype)
        time_emb = self.time_embed(t)
        for block in self.blocks:
            x = block(x, time_emb)
        x = self.output_proj(self.final_norm(x))
        return x.permute(0, 2, 1)


class DDPMDiTDenoiser8M(nn.Module):
    """Lightweight 1D DiT epsilon predictor for SAME latent DDPM restoration (~7.9M parameters).

    Reduces DDPMDiTDenoiser from ~132.9M to ~7.9M by using
    d_model=288, num_layers=6, num_attention_heads=6, mlp_ratio=2.0.
    Keeps 6 blocks instead of reducing to fewer to preserve depth,
    since the DiT paper (Peebles & Xie, ICCV 2023) shows GFlops ∝
    num_layers × d_model² and depth reduction hurts quality
    disproportionately at fixed compute.
    """

    def __init__(self, latent_channels=256, condition_channels=256,
                 d_model=288, num_layers=6, num_attention_heads=6,
                 mlp_ratio=2.0, dropout=0.0, attention_dropout=0.0,
                 time_embed_dim=None, **unused):
        super().__init__()
        if latent_channels != 256 or condition_channels != 256:
            raise ValueError(
                "DDPMDiTDenoiser8M is SAME-L-only and expects 256 latent/condition channels"
            )
        if d_model % num_attention_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_attention_heads={num_attention_heads}"
            )
        time_embed_dim = time_embed_dim or d_model
        if time_embed_dim != d_model:
            raise ValueError("time_embed_dim must match d_model in this implementation")

        self.latent_channels = latent_channels
        self.condition_channels = condition_channels
        self.input_proj = nn.Linear(latent_channels + condition_channels, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding1D(d_model)
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.blocks = nn.ModuleList([
            DiTBlock1D8M(
                d_model=d_model,
                num_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_channels)

    def forward(self, z_t, t, z_cond):
        """
        z_t: [B, 256, T], t: [B], z_cond: [B, 256, T]
        returns eps_pred: [B, 256, T]
        """
        if z_t.ndim != 3 or z_cond.ndim != 3:
            raise ValueError("DDPMDiTDenoiser8M expects rank-3 latent tensors")
        if z_t.shape != z_cond.shape:
            raise ValueError(
                f"z_t and z_cond shapes must match, got {z_t.shape} and {z_cond.shape}"
            )
        if z_t.shape[1] != self.latent_channels:
            raise ValueError(f"Expected 256 latent channels, got {z_t.shape[1]}")
        if t.shape != (z_t.shape[0],):
            raise ValueError(f"Expected t shape {(z_t.shape[0],)}, got {tuple(t.shape)}")

        x = torch.cat([z_t, z_cond], dim=1).permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + self.pos_encoding(x.shape[1], x.device).to(dtype=x.dtype)
        time_emb = self.time_embed(t)
        for block in self.blocks:
            x = block(x, time_emb)
        x = self.output_proj(self.final_norm(x))
        return x.permute(0, 2, 1)


class CFMDiTVelocityNet(nn.Module):
    """1D DiT velocity predictor for conditional flow matching in SAME latents.

    This intentionally mirrors the DDPM DiT backbone: the architecture sees the
    current latent state and corrupted latent condition at every time frame,
    adds temporal position encoding, injects continuous flow time through
    AdaLN-Zero transformer blocks, and predicts a SAME-shaped velocity field.
    Keeping the backbone equal to DDPM makes the ablation about the training
    objective and sampler, not about model capacity.
    """

    def __init__(self, latent_channels=256, condition_channels=256,
                 d_model=768, num_layers=12, num_attention_heads=12,
                 mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0,
                 time_embed_dim=None, **unused):
        super().__init__()
        if latent_channels != 256 or condition_channels != 256:
            raise ValueError(
                "CFMDiTVelocityNet is SAME-L-only and expects 256 latent/condition channels"
            )
        time_embed_dim = time_embed_dim or d_model
        if time_embed_dim != d_model:
            raise ValueError("time_embed_dim must match d_model in this implementation")

        self.latent_channels = latent_channels
        self.condition_channels = condition_channels
        self.input_proj = nn.Linear(latent_channels + condition_channels, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding1D(d_model)
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.blocks = nn.ModuleList([
            DiTBlock1D(
                d_model=d_model,
                num_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_channels)

    def forward(self, z_t, t, z_cond):
        """
        z_t: [B, 256, T], t: [B] in [0, 1], z_cond: [B, 256, T]
        returns velocity: [B, 256, T]
        """
        if z_t.ndim != 3 or z_cond.ndim != 3:
            raise ValueError("CFMDiTVelocityNet expects rank-3 latent tensors")
        if z_t.shape != z_cond.shape:
            raise ValueError(
                f"z_t and z_cond shapes must match, got {z_t.shape} and {z_cond.shape}"
            )
        if z_t.shape[1] != self.latent_channels:
            raise ValueError(f"Expected 256 latent channels, got {z_t.shape[1]}")
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.shape != (z_t.shape[0],):
            raise ValueError(f"Expected t shape {(z_t.shape[0],)}, got {tuple(t.shape)}")

        x = torch.cat([z_t, z_cond], dim=1).permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + self.pos_encoding(x.shape[1], x.device).to(dtype=x.dtype)
        time_emb = self.time_embed(t)
        for block in self.blocks:
            x = block(x, time_emb)
        x = self.output_proj(self.final_norm(x))
        return x.permute(0, 2, 1)


class CFMDiTVelocityNet8M(nn.Module):
    """Lightweight 1D DiT velocity predictor for CFM in SAME latents (~7.9M parameters).

    Reduces CFMDiTVelocityNet from ~132.9M to ~7.9M by using
    d_model=288, num_layers=6, num_attention_heads=6, mlp_ratio=2.0.
    Architecture is otherwise identical to CFMDiTVelocityNet, keeping
    the ablation between CFM and DDPM isolated to the objective/sampler
    rather than model capacity.
    """

    def __init__(self, latent_channels=256, condition_channels=256,
                 d_model=288, num_layers=6, num_attention_heads=6,
                 mlp_ratio=2.0, dropout=0.0, attention_dropout=0.0,
                 time_embed_dim=None, **unused):
        super().__init__()
        if latent_channels != 256 or condition_channels != 256:
            raise ValueError(
                "CFMDiTVelocityNet8M is SAME-L-only and expects 256 latent/condition channels"
            )
        if d_model % num_attention_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_attention_heads={num_attention_heads}"
            )
        time_embed_dim = time_embed_dim or d_model
        if time_embed_dim != d_model:
            raise ValueError("time_embed_dim must match d_model in this implementation")

        self.latent_channels = latent_channels
        self.condition_channels = condition_channels
        self.input_proj = nn.Linear(latent_channels + condition_channels, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding1D(d_model)
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.blocks = nn.ModuleList([
            DiTBlock1D8M(
                d_model=d_model,
                num_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_channels)

    def forward(self, z_t, t, z_cond):
        """
        z_t: [B, 256, T], t: [B] in [0, 1], z_cond: [B, 256, T]
        returns velocity: [B, 256, T]
        """
        if z_t.ndim != 3 or z_cond.ndim != 3:
            raise ValueError("CFMDiTVelocityNet8M expects rank-3 latent tensors")
        if z_t.shape != z_cond.shape:
            raise ValueError(
                f"z_t and z_cond shapes must match, got {z_t.shape} and {z_cond.shape}"
            )
        if z_t.shape[1] != self.latent_channels:
            raise ValueError(f"Expected 256 latent channels, got {z_t.shape[1]}")
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.shape != (z_t.shape[0],):
            raise ValueError(f"Expected t shape {(z_t.shape[0],)}, got {tuple(t.shape)}")

        x = torch.cat([z_t, z_cond], dim=1).permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + self.pos_encoding(x.shape[1], x.device).to(dtype=x.dtype)
        time_emb = self.time_embed(t)
        for block in self.blocks:
            x = block(x, time_emb)
        x = self.output_proj(self.final_norm(x))
        return x.permute(0, 2, 1)


class CFMDiTVelocityNet4M(CFMDiTVelocityNet8M):
    """Approximately-4M CFM DiT with GPU-friendly 32-wide attention heads.

    This retains the CFM8M ratio-2 AdaLN-zero block and native velocity
    objective, but uses d_model=192 and seven blocks. The resulting model has
    exactly 4,074,304 trainable parameters with six attention heads.
    """

    def __init__(self, latent_channels=256, condition_channels=256,
                 d_model=192, num_layers=7, num_attention_heads=6,
                 mlp_ratio=2.0, dropout=0.0, attention_dropout=0.0,
                 time_embed_dim=None, **unused):
        super().__init__(
            latent_channels=latent_channels,
            condition_channels=condition_channels,
            d_model=d_model,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            time_embed_dim=time_embed_dim,
            **unused,
        )


class CFMDiTVelocityNet16M(CFMDiTVelocityNet8M):
    """Approximately-16M CFM DiT with GPU-friendly 64-wide attention heads.

    This retains the CFM8M ratio-2 AdaLN-zero block and native velocity
    objective, but uses d_model=384 and seven blocks. The resulting model has
    exactly 15,963,520 trainable parameters with six attention heads.
    """

    def __init__(self, latent_channels=256, condition_channels=256,
                 d_model=384, num_layers=7, num_attention_heads=6,
                 mlp_ratio=2.0, dropout=0.0, attention_dropout=0.0,
                 time_embed_dim=None, **unused):
        super().__init__(
            latent_channels=latent_channels,
            condition_channels=condition_channels,
            d_model=d_model,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            time_embed_dim=time_embed_dim,
            **unused,
        )


class CFMDiTVelocityNet40M(CFMDiTVelocityNet8M):
    """Approximately-40M CFM DiT with GPU-friendly 64-wide attention heads.

    This follows the same scaling path as the 16M and 80M variants: retain the
    ratio-2 AdaLN-Zero block and native velocity objective while increasing both
    backbone width and depth. Interpolating between the 16M configuration
    (d_model=384, seven blocks) and 80M configuration (d_model=768, nine blocks)
    gives d_model=576 and eight blocks. With nine attention heads, each head
    remains 64 channels wide and the model has exactly 40,320,256 trainable
    parameters.
    """

    D_MODEL = 576
    NUM_LAYERS = 8
    NUM_ATTENTION_HEADS = 9
    MLP_RATIO = 2.0
    HEAD_DIM = 64
    EXPECTED_PARAMETER_COUNT = 40_320_256

    def __init__(self, latent_channels=256, condition_channels=256,
                 d_model=D_MODEL, num_layers=NUM_LAYERS,
                 num_attention_heads=NUM_ATTENTION_HEADS,
                 mlp_ratio=MLP_RATIO, dropout=0.0, attention_dropout=0.0,
                 time_embed_dim=None, **unused):
        architecture = {
            "d_model": d_model,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "mlp_ratio": float(mlp_ratio),
        }
        expected = {
            "d_model": self.D_MODEL,
            "num_layers": self.NUM_LAYERS,
            "num_attention_heads": self.NUM_ATTENTION_HEADS,
            "mlp_ratio": self.MLP_RATIO,
        }
        if architecture != expected:
            raise ValueError(
                "CFMDiTVelocityNet40M has a fixed architecture; expected "
                f"{expected}, got {architecture}. Define a separate model type "
                "for architecture experiments."
            )
        if latent_channels != 256 or condition_channels != 256:
            raise ValueError(
                "CFMDiTVelocityNet40M is SAME-L-only and expects 256 channels"
            )
        if d_model // num_attention_heads != self.HEAD_DIM:
            raise ValueError(
                "CFMDiTVelocityNet40M must preserve 64 channels per attention head"
            )
        super().__init__(
            latent_channels=latent_channels,
            condition_channels=condition_channels,
            d_model=d_model,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            time_embed_dim=time_embed_dim,
            **unused,
        )
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != self.EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                "CFMDiTVelocityNet40M parameter-count invariant failed: expected "
                f"{self.EXPECTED_PARAMETER_COUNT:,}, got {parameter_count:,}"
            )


class CFMDiTVelocityNet80M(CFMDiTVelocityNet8M):
    """Approximately-80M CFM DiT with GPU-friendly 64-wide attention heads.

    This follows the same scaling rule as the 8M-to-16M variant: retain the
    ratio-2 AdaLN-Zero block and native velocity objective while increasing
    backbone width and depth. With d_model=768, nine blocks, and twelve heads,
    each attention head remains 64 channels wide and the model has exactly
    79,722,496 trainable parameters.
    """

    def __init__(self, latent_channels=256, condition_channels=256,
                 d_model=768, num_layers=9, num_attention_heads=12,
                 mlp_ratio=2.0, dropout=0.0, attention_dropout=0.0,
                 time_embed_dim=None, **unused):
        super().__init__(
            latent_channels=latent_channels,
            condition_channels=condition_channels,
            d_model=d_model,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            time_embed_dim=time_embed_dim,
            **unused,
        )
