"""
model/encoders.py — Fixed sinusoidal feature encoders (no learnable parameters)

These encoders convert raw conditioning signals (F0, velocity, loudness) into
smooth, high-dimensional representations that are easier for the GRU to process.
"""

import math
import torch

from synth.constants import F0_MIN, F0_MAX, F0_BINS, VEL_DIM, LO_DIM


def encode_f0(f0_hz: torch.Tensor, bins: int = F0_BINS) -> torch.Tensor:
    """Sinusoidal log-frequency encoding.
    f0_hz : (B, T) in Hz, 0 = unvoiced
    returns (B, T, bins)
    """
    B, T   = f0_hz.shape
    device = f0_hz.device
    f0_safe = f0_hz.clamp(min=F0_MIN)
    f0_norm = (torch.log(f0_safe) - math.log(F0_MIN)) / (math.log(F0_MAX) - math.log(F0_MIN))
    f0_norm = f0_norm.clamp(0.0, 1.0)
    i   = torch.arange(1, bins // 2 + 1, dtype=torch.float32, device=device)
    enc = torch.zeros(B, T, bins, device=device)
    enc[:, :, 0::2] = torch.sin(math.pi * i * f0_norm.unsqueeze(-1))
    enc[:, :, 1::2] = torch.cos(math.pi * i * f0_norm.unsqueeze(-1))
    voiced = (f0_hz > 0).float().unsqueeze(-1)
    return enc * voiced


def encode_velocity(velocity: torch.Tensor, dim: int = VEL_DIM) -> torch.Tensor:
    """Sinusoidal velocity encoding.
    velocity : (B,) bucket 0–7
    returns (B, dim)
    """
    device   = velocity.device
    vel_norm = (velocity.float() / 7.0).clamp(0.0, 1.0)
    i   = torch.arange(1, dim // 2 + 1, dtype=torch.float32, device=device)
    enc = torch.zeros(velocity.shape[0], dim, device=device)
    enc[:, 0::2] = torch.sin(math.pi * i * vel_norm.unsqueeze(-1))
    enc[:, 1::2] = torch.cos(math.pi * i * vel_norm.unsqueeze(-1))
    return enc


def encode_loudness(loudness_db: torch.Tensor, dim: int = LO_DIM) -> torch.Tensor:
    """Sinusoidal loudness encoding.
    loudness_db : (B, T) in dB
    returns (B, T, dim)
    """
    B, T   = loudness_db.shape
    device = loudness_db.device
    lo_norm = ((loudness_db + 80.0) / 80.0).clamp(0.0, 1.0)
    i   = torch.arange(1, dim // 2 + 1, dtype=torch.float32, device=device)
    enc = torch.zeros(B, T, dim, device=device)
    enc[:, :, 0::2] = torch.sin(math.pi * i * lo_norm.unsqueeze(-1))
    enc[:, :, 1::2] = torch.cos(math.pi * i * lo_norm.unsqueeze(-1))
    return enc
