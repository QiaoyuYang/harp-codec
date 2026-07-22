"""
HARP Loss Functions
"""

from typing import List, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools import STFTParams


# =============================================================================
# Basic Losses
# =============================================================================

class L1Loss(nn.Module):
    """L1 Loss between AudioSignals with detached reference."""
    
    def __init__(self, attribute: str = "audio_data", weight: float = 1.0, **kwargs):
        super().__init__()
        self.attribute = attribute
        self.weight = weight

    def forward(self, x: AudioSignal, y: AudioSignal) -> torch.Tensor:
        if isinstance(x, AudioSignal):
            x = getattr(x, self.attribute)
            y = getattr(y, self.attribute).detach()  # Detach reference
        return F.l1_loss(x, y)

class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-scale STFT loss (magnitude).
    """
    
    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: Callable = None,
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: Optional[str] = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn if loss_fn is not None else nn.L1Loss()
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            
            loss += self.log_weight * self.loss_fn(
                x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
        return loss


class MelSpectrogramLoss(nn.Module):
    """
    Multi-scale mel spectrogram loss.
    """
    
    def __init__(
        self,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: Callable = None,
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None],
        window_type: Optional[str] = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn if loss_fn is not None else nn.L1Loss()
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            
            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


# =============================================================================
# GAN Losses (Optimized)
# =============================================================================

class GANLoss(nn.Module):
    """GAN loss for adversarial training."""
    
    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator

    def discriminator_loss(
        self, 
        fake: AudioSignal, 
        real: AudioSignal, 
        label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """Compute discriminator loss."""
        # Detach fake to prevent generator gradients
        fake_audio = fake.audio_data.detach()
        real_audio = real.audio_data.detach()
        
        d_fake = self.discriminator(fake_audio)
        d_real = self.discriminator(real_audio)
        
        real_target = 1.0 - label_smoothing
        loss = 0.0
        
        for x_fake, x_real in zip(d_fake, d_real):
            loss = loss + (x_fake[-1] ** 2).mean()
            loss = loss + ((real_target - x_real[-1]) ** 2).mean()
        
        return loss

    def generator_loss(
        self, 
        fake: AudioSignal, 
        real: AudioSignal
    ) -> tuple:
        """Compute generator adversarial and feature matching losses."""
        fake_audio = fake.audio_data
        real_audio = real.audio_data.detach()
        
        d_fake = self.discriminator(fake_audio)
        
        # For feature matching, we need real discriminator outputs
        with torch.no_grad():
            d_real = self.discriminator(real_audio)
        
        # Generator adversarial loss
        loss_g = 0.0
        for x_fake in d_fake:
            loss_g = loss_g + ((1 - x_fake[-1]) ** 2).mean()
        
        # Feature matching loss
        loss_feat = 0.0
        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feat = loss_feat + F.l1_loss(d_fake[i][j], d_real[i][j])
        
        return loss_g, loss_feat