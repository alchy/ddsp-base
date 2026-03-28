"""
model/envelope.py — EnvelopeNet: loudness envelope predictor

Tiny MLP that learns instrument-specific amplitude envelopes from training data.
Used during generation when no reference audio is available.

Architecture: (midi_norm, vel_norm) → (duration_s, loudness_shape[N_ENV])
The output uses a warped time axis (power-law) to concentrate resolution
near the attack transient.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synth.constants import SR, FRAME_HOP

N_ENV         = 512     # number of envelope control points
ENVELOPE_WARP = 4.0     # power-law time warp exponent
                        # t_store[i] = (i/(N-1))^WARP
                        # → first N/2 points cover first 6% of note (attack region)


class EnvelopeNet(nn.Module):
    """
    Tiny MLP: (midi_norm, vel_norm) → (dur_s, shape[N_ENV])
    ~30 K parameters (hidden=64, N_ENV=512)
    """

    def __init__(self, hidden: int = 64, n_env: int = N_ENV):
        super().__init__()
        self.n_env = n_env
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1 + n_env),
        )

    def forward(self, midi_norm: torch.Tensor,
                vel_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        midi_norm, vel_norm : (B,) in [0, 1]
        Returns:
            dur_s  : (B,)       predicted duration in seconds (≥ 0.5 s)
            shape  : (B, n_env) loudness curve in dB at warped time steps
        """
        x   = torch.stack([midi_norm, vel_norm], dim=-1)
        out = self.net(x)
        dur_s = F.softplus(out[:, 0]) + 0.5
        shape = out[:, 1:]
        return dur_s, shape

    def predict_envelope(self, midi: int, vel: int,
                         warp: float = ENVELOPE_WARP) -> np.ndarray:
        """Return loudness np.ndarray (n_frames,) for a single (midi, vel)."""
        device    = next(self.parameters()).device
        midi_t    = torch.tensor([midi / 127.0], dtype=torch.float32, device=device)
        vel_t     = torch.tensor([vel  /   7.0], dtype=torch.float32, device=device)
        with torch.no_grad():
            dur_s, shape = self(midi_t, vel_t)
        dur_s    = float(dur_s[0])
        shape_np = shape[0].cpu().numpy()
        n_frames = max(1, round(dur_s * SR / FRAME_HOP))
        t_store  = np.power(np.linspace(0.0, 1.0, self.n_env), warp)
        t_query  = np.linspace(0.0, 1.0, n_frames)
        return np.interp(t_query, t_store, shape_np).astype(np.float32)
