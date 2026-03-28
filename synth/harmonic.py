"""
synth/harmonic.py — Additive harmonic synthesizer

Signal model:
    f_k(t) = k · F0(t) · √(1 + inh · k²)          [inharmonic partial frequencies]
    φ_k(t) = cumsum(2π · f_k(t) / SR)               [phase accumulation]

    Physics decay (Simionato 2024, Bensa 2003):
    σ_k = b1 + b3 · (2π · k · f0_mean)²             [freq-dependent decay rate]
    d_k(t) = exp(-σ_k · t)                           [decay envelope per partial]

    y(t) = Σ_k  a_k(t) · d_k(t) · sin(φ_k(t))
           / n_active

Active harmonics: n_active = min(N_HARM_MAX, floor(nyquist / f0)).
Decay envelope applied at frame rate then upsampled — avoids (B, N, n_samples) tensor.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import SR, FRAME_HOP, N_HARM_MAX


class HarmonicSynth(nn.Module):
    """
    Additive synthesis with inharmonicity and per-partial physics decay.

    Output: (B, 1, n_samples)
    """

    def forward(self, harm_amps: torch.Tensor, f0_hz: torch.Tensor,
                n_samples: int, inh: torch.Tensor,
                b1: torch.Tensor, b3: torch.Tensor) -> torch.Tensor:
        """
        harm_amps : (B, T, N)   per-partial amplitude envelope (frame rate)
        f0_hz     : (B, T)      fundamental frequency in Hz
        n_samples : int         output length in samples
        inh       : (B,)        inharmonicity coefficient B
        b1        : (B,)        baseline decay rate  [s⁻¹]
        b3        : (B,)        freq-dependent decay coefficient  [s⁻¹ / (rad/s)²]

        Physics decay: σ_k = b1 + b3 · (2π · k · f0_mean)²
        Realistic Steinway D values: b1 ≈ 0.3 s⁻¹,  b3 ≈ 1e-7
        """
        B, T_frames, N = harm_amps.shape
        device = harm_amps.device

        k = torch.arange(1, N + 1, dtype=torch.float32, device=device)  # (N,)

        # ------------------------------------------------------------------ #
        # Per-partial physics decay — computed at frame rate                  #
        # Memory: B × N × T_frames  (vs B × N × n_samples at sample rate)    #
        # For bass: 4 × 128 × 2000 ≈ 4 MB  (vs ~1 GB at sample rate)         #
        # ------------------------------------------------------------------ #
        f0_mean = f0_hz.clamp(min=20.0).mean(dim=1)            # (B,)
        omega_k = 2.0 * math.pi * f0_mean.unsqueeze(1) * k    # (B, N) rad/s
        sigma_k = b1.unsqueeze(1) + b3.unsqueeze(1) * omega_k ** 2   # (B, N) s⁻¹

        t_frames = torch.arange(T_frames, dtype=torch.float32, device=device) \
                   * (FRAME_HOP / SR)                          # (T,) seconds
        # decay_env shape: (B, N, T)
        decay_env = torch.exp(-sigma_k.unsqueeze(2) * t_frames.unsqueeze(0).unsqueeze(0))

        # Apply decay to per-partial amplitudes at frame rate
        # harm_amps: (B, T, N) → permute to (B, N, T), multiply, permute back
        harm_amps_decayed = harm_amps.permute(0, 2, 1) * decay_env   # (B, N, T)
        harm_amps_decayed = harm_amps_decayed.permute(0, 2, 1)       # (B, T, N)

        # Upsample to sample rate
        a     = F.interpolate(harm_amps_decayed.permute(0, 2, 1),
                              size=n_samples, mode='linear', align_corners=False)
        f0_up = F.interpolate(f0_hz.unsqueeze(1).float(),
                              size=n_samples, mode='linear', align_corners=False).squeeze(1)

        # Piano inharmonicity: partial k at k · f0 · √(1 + inh · k²)
        stretch   = torch.sqrt(1.0 + inh.unsqueeze(1) * k ** 2)                     # (B, N)
        inst_freq = f0_up.unsqueeze(1) * (k.unsqueeze(0) * stretch).unsqueeze(2)    # (B, N, n_samples)

        # Zero out harmonics above Nyquist
        nyq_mask = (inst_freq < SR * 0.45).float()
        a = a * nyq_mask

        # Adaptive normalization: divide by n_active
        n_active = ((k.unsqueeze(0) * f0_mean.unsqueeze(1)) < SR * 0.45) \
                   .sum(dim=1).clamp(min=1).float()            # (B,)

        # Phase accumulation and synthesis
        phase  = torch.cumsum(2.0 * math.pi * inst_freq / SR, dim=-1)
        signal = (a * torch.sin(phase)).sum(dim=1, keepdim=True)
        return signal / n_active.view(B, 1, 1)
