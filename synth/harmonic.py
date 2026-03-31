"""
synth/harmonic.py — Additive harmonic synthesizer

Signal model:
    f_k(t) = k · F0(t) · √(1 + inh · k²)          [inharmonic partial frequencies]
    φ_k(t) = cumsum(2π · f_k(t) / SR)               [phase accumulation]

    Two-component physics decay (Phase 2 — Weinreich 1977, Bensa 2003):
    Each piano string vibrates in two polarizations with separate decay rates:

        σ_k_fast = b1_f + b3_f · (2π · k · f0_mean)²   [vertical polarization, faster]
        σ_k_slow = b1_s + b3_s · (2π · k · f0_mean)²   [horizontal polarization, slower]

    d_k(t) = α · exp(-σ_k_fast · t) + (1-α) · exp(-σ_k_slow · t)

    Perceptual result: initial fast decay + sustained slower component →
    characteristic piano "two-stage" doznívání + AM-like beating in the envelope.

    y(t) = Σ_k  a_k(t) · d_k(t) · sin(φ_k(t))
           / n_active

Decay computed at frame rate then upsampled — avoids (B, N, n_samples) tensor.
Memory: B×N×T_frames  (e.g. 4×128×2000 ≈ 4 MB for bass 10s window).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import SR, FRAME_HOP, N_HARM_MAX


class HarmonicSynth(nn.Module):
    """
    Additive synthesis with inharmonicity and two-component physics decay.

    Output: (B, 1, n_samples)
    """

    def forward(self, harm_amps: torch.Tensor, f0_hz: torch.Tensor,
                n_samples: int, inh: torch.Tensor,
                b1_f: torch.Tensor, b3_f: torch.Tensor,
                b1_s: torch.Tensor, b3_s: torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
        """
        harm_amps : (B, T, N)   per-partial amplitude envelope (frame rate)
        f0_hz     : (B, T)      fundamental frequency in Hz
        n_samples : int         output length in samples
        inh       : (B,)        inharmonicity coefficient B
        b1_f      : (B,)        fast-component baseline decay rate  [s⁻¹]
        b3_f      : (B,)        fast-component freq-dependent decay [s⁻¹/(rad/s)²]
        b1_s      : (B,)        slow-component baseline decay rate  [s⁻¹]
        b3_s      : (B,)        slow-component freq-dependent decay [s⁻¹/(rad/s)²]
        alpha     : (B,)        fast-component fraction [0,1]

        Physical init targets (Steinway D):
            b1_f ≈ 1.0 s⁻¹  (τ_fast ≈ 1s — vertical polarization)
            b3_f ≈ 2e-7
            b1_s ≈ 0.15 s⁻¹ (τ_slow ≈ 6.7s — horizontal polarization)
            b3_s ≈ 5e-9
            alpha ≈ 0.5      (equal polarization excitation)
        """
        B, T_frames, N = harm_amps.shape
        device = harm_amps.device

        k = torch.arange(1, N + 1, dtype=torch.float32, device=device)  # (N,)

        # ------------------------------------------------------------------ #
        # Two-component decay — computed at frame rate (memory efficient)     #
        # ------------------------------------------------------------------ #
        f0_mean = f0_hz.clamp(min=20.0).mean(dim=1)            # (B,)
        omega_k = 2.0 * math.pi * f0_mean.unsqueeze(1) * k    # (B, N) rad/s

        sigma_fast = b1_f.unsqueeze(1) + b3_f.unsqueeze(1) * omega_k ** 2   # (B, N)
        sigma_slow = b1_s.unsqueeze(1) + b3_s.unsqueeze(1) * omega_k ** 2   # (B, N)

        t_frames = torch.arange(T_frames, dtype=torch.float32, device=device) \
                   * (FRAME_HOP / SR)                          # (T,) seconds

        decay_fast = torch.exp(-sigma_fast.unsqueeze(2) * t_frames)   # (B, N, T)
        decay_slow = torch.exp(-sigma_slow.unsqueeze(2) * t_frames)   # (B, N, T)
        a_f = alpha.view(B, 1, 1)
        decay_env  = a_f * decay_fast + (1.0 - a_f) * decay_slow      # (B, N, T)

        # Apply decay to per-partial amplitudes at frame rate
        harm_amps_decayed = harm_amps.permute(0, 2, 1) * decay_env    # (B, N, T)
        harm_amps_decayed = harm_amps_decayed.permute(0, 2, 1)        # (B, T, N)

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
