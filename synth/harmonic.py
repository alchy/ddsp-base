"""
synth/harmonic.py — Additive harmonic synthesizer

Signal model:
    f_k(t) = k · F0(t) · √(1 + inh · k²)     [inharmonic partial frequencies]
    φ_k(t) = cumsum(2π · f_k(t) / SR)          [phase accumulation]
    y(t)   = Σ_k  a_k(t) · sin(φ_k(t))        [weighted sum of sinusoids]
             / n_active                          [normalized by active count]

Active harmonics adapt per batch item: n_active = min(N_HARM_MAX, floor(nyquist / f0))
so bass notes automatically use more harmonics than treble notes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import SR, N_HARM_MAX


class HarmonicSynth(nn.Module):
    """
    Additive synthesis: weighted sum of up to N_HARM_MAX sinusoids at k * F0(t).

    Output: (B, 1, n_samples)
    """

    def forward(self, harm_amps: torch.Tensor, f0_hz: torch.Tensor,
                n_samples: int, inh: torch.Tensor) -> torch.Tensor:
        """
        harm_amps : (B, T, N)   per-partial amplitude envelope (frame rate)
        f0_hz     : (B, T)      fundamental frequency in Hz
        n_samples : int         output length in samples
        inh       : (B,)        inharmonicity coefficient B — f_k = k·f0·√(1 + B·k²)
        """
        B, T_frames, N = harm_amps.shape
        device = harm_amps.device

        # Upsample frame-rate signals to sample rate
        a     = F.interpolate(harm_amps.permute(0, 2, 1),
                              size=n_samples, mode='linear', align_corners=False)
        f0_up = F.interpolate(f0_hz.unsqueeze(1).float(),
                              size=n_samples, mode='linear', align_corners=False).squeeze(1)

        k = torch.arange(1, N + 1, dtype=torch.float32, device=device)   # (N,)

        # Piano inharmonicity: partial k at k·f0·√(1 + inh·k²)
        stretch   = torch.sqrt(1.0 + inh.unsqueeze(1) * k ** 2)                       # (B, N)
        inst_freq = f0_up.unsqueeze(1) * (k.unsqueeze(0) * stretch).unsqueeze(2)      # (B, N, n_samples)

        # Zero out harmonics above Nyquist
        nyq_mask = (inst_freq < SR * 0.45).float()
        a = a * nyq_mask

        # Adaptive normalization: divide by n_active, not by fixed N
        f0_mean  = f0_up.mean(dim=-1)                                                  # (B,)
        n_active = ((k.unsqueeze(0) * f0_mean.unsqueeze(1)) < SR * 0.45) \
                   .sum(dim=1).clamp(min=1).float()                                    # (B,)

        # Phase accumulation
        phase = torch.cumsum(2.0 * math.pi * inst_freq / SR, dim=-1)

        signal = (a * torch.sin(phase)).sum(dim=1, keepdim=True)
        return signal / n_active.view(B, 1, 1)
