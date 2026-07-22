"""
Neural network layers for HARP.
Based on DAC's layers.py structure.
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


def WNConv1d(*args, **kwargs):
    """Weight-normalized 1D convolution."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """Weight-normalized 1D transposed convolution."""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolution."""

    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class AMPBlock(nn.Module):
    """
    Anti-aliased Multi-Periodicity block for BigVGAN-style decoder.
    Combines multiple dilated convolutions with different periodicities.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple = (1, 3, 5),
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            padding = (kernel_size - 1) * d // 2
            self.convs.append(
                nn.Sequential(
                    Snake1d(channels),
                    WNConv1d(channels, channels, kernel_size, dilation=d, padding=padding),
                    Snake1d(channels),
                    WNConv1d(channels, channels, kernel_size=1),
                )
            )
        self.num_layers = len(dilations)

    def forward(self, x):
        out = 0
        for conv in self.convs:
            out = out + conv(x)
        return x + out / self.num_layers