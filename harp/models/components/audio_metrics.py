"""
Audio Quality Metrics for HARP

Evaluation metrics for audio reconstruction quality.
"""

import torch
import torch.nn as nn


class LogSpectralDistance(nn.Module):
    """
    Log Spectral Distance (LSD) metric.

    Measures the distance between log power spectra.
    Lower is better. Typical good values: < 1.0 dB
    """

    def __init__(self, n_fft: int = 2048, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.reduction = reduction
        self.eps = eps
        self.register_buffer('window', torch.hann_window(n_fft))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute LSD between reconstructed (x) and reference (y) audio.

        Args:
            x: Reconstructed audio [B, 1, T] or [B, T]
            y: Reference audio [B, 1, T] or [B, T]

        Returns:
            LSD in dB
        """
        if x.dim() == 3:
            x = x.squeeze(1)
            y = y.squeeze(1)

        stft_x = torch.stft(
            x, self.n_fft, self.hop_length, window=self.window,
            return_complex=True, pad_mode='reflect'
        )
        stft_y = torch.stft(
            y, self.n_fft, self.hop_length, window=self.window,
            return_complex=True, pad_mode='reflect'
        )

        power_x = stft_x.abs() ** 2
        power_y = stft_y.abs() ** 2

        log_diff = (power_x.clamp(min=self.eps).log10() - power_y.clamp(min=self.eps).log10()) ** 2
        lsd = torch.sqrt(log_diff.mean(dim=1) + self.eps).mean(dim=1)

        if self.reduction == 'mean':
            return lsd.mean()
        return lsd


class SignalToNoiseRatio(nn.Module):
    """
    Signal-to-Noise Ratio (SNR) metric.

    Measures the ratio of signal power to noise power.
    Higher is better. Typical good values: > 20 dB
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8, max_snr: float = 100.0):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.max_snr = max_snr  # Cap SNR to prevent infinity

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute SNR between reconstructed (x) and reference (y) audio.

        Args:
            x: Reconstructed audio [B, 1, T] or [B, T]
            y: Reference audio [B, 1, T] or [B, T]

        Returns:
            SNR in dB (clamped to max_snr)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
            y = y.squeeze(1)

        signal_power = (y ** 2).sum(dim=-1)
        noise_power = ((x - y) ** 2).sum(dim=-1)

        # Clamp ratio to prevent infinity
        ratio = signal_power.clamp(min=self.eps) / noise_power.clamp(min=self.eps)
        snr = 10 * torch.log10(ratio)

        # Clamp to maximum SNR value
        snr = snr.clamp(max=self.max_snr)

        if self.reduction == 'mean':
            return snr.mean()
        return snr


class ScaleInvariantSDR(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric.

    Measures audio quality independent of scale.
    Higher is better. Typical good values: > 10 dB
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8, max_sdr: float = 100.0):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.max_sdr = max_sdr  # Cap SI-SDR to prevent infinity

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute SI-SDR between reconstructed (x) and reference (y) audio.

        Args:
            x: Reconstructed audio [B, 1, T] or [B, T]
            y: Reference audio [B, 1, T] or [B, T]

        Returns:
            SI-SDR in dB (capped to max_sdr)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
            y = y.squeeze(1)

        # Zero mean
        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)

        # Optimal scaling
        dot = (x * y).sum(dim=-1, keepdim=True)
        s_target_energy = (y ** 2).sum(dim=-1, keepdim=True) + self.eps
        s_target = (dot / s_target_energy) * y

        e_noise = x - s_target

        # Clamp both numerator and denominator
        s_target_power = (s_target ** 2).sum(dim=-1).clamp(min=self.eps)
        e_noise_power = (e_noise ** 2).sum(dim=-1).clamp(min=self.eps)

        si_sdr = 10 * torch.log10(s_target_power / e_noise_power)

        # Clamp to maximum SI-SDR value
        si_sdr = si_sdr.clamp(min=-self.max_sdr, max=self.max_sdr)

        if self.reduction == 'mean':
            return si_sdr.mean()
        return si_sdr
