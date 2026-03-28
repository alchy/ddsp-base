"""
model/vocoder.py — DDSPVocoder: neural network + synthesis pipeline

Architecture (Decoupled Timbre):
    Input:  F0 + velocity  (no loudness — applied post-synthesis)
    GRU:    learns timbre over time
    Heads:  predict synthesis parameters for HarmonicSynth + NoiseSynth
    Output: (B, 2, n_samples) stereo audio at sample rate SR

Loudness is applied as a linear envelope after synthesis so the network
learns pure timbre independently of dynamics.

Per-partial physics decay (Phase 1 — bass refactor):
    σ_k = b1 + b3 · (2π · k · f0_mean)²     [Simionato 2024, Bensa 2003]
    b1, b3 are learned scalars pooled over T (note-level constants).
    Init: b1 ≈ 0.31 s⁻¹,  b3 ≈ 1.1e-7  (close to Steinway D values).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from synth.constants import (
    SR, FRAME_HOP, N_HARM, N_NOISE, F0_BINS, VEL_DIM, F0_MIN, MODEL_SIZES,
)
from synth.harmonic import HarmonicSynth
from synth.noise import NoiseSynth
from model.encoders import encode_f0, encode_velocity


class DDSPVocoder(nn.Module):
    """
    Fully stereo DDSP vocoder conditioned on (F0, loudness, velocity).

    Independent harmonic and noise parameters for L and R channels
    allow the network to capture spatial timbre differences.

    Returns: (B, 2, n_samples)
    """

    def __init__(self, gru_hidden: int = 64, gru_layers: int = 1, mlp_dim: int = 128):
        super().__init__()

        feat_dim = F0_BINS + VEL_DIM   # 72

        self.pre_mlp = nn.Sequential(
            nn.Linear(feat_dim, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim,  mlp_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(mlp_dim, gru_hidden, num_layers=gru_layers, batch_first=True)
        self.post_mlp = nn.Sequential(
            nn.Linear(gru_hidden, mlp_dim), nn.ReLU(),
        )

        # Harmonic heads: softmax distribution × softplus global amplitude
        self.head_harm_L = nn.Linear(mlp_dim, N_HARM)
        self.head_harm_R = nn.Linear(mlp_dim, N_HARM)
        self.head_amp_L  = nn.Linear(mlp_dim, 1)
        self.head_amp_R  = nn.Linear(mlp_dim, 1)

        # Noise heads: spectral shape + global amplitude scalar
        self.head_noise_L     = nn.Linear(mlp_dim, N_NOISE)
        self.head_noise_R     = nn.Linear(mlp_dim, N_NOISE)
        self.head_noise_amp_L = nn.Linear(mlp_dim, 1)
        self.head_noise_amp_R = nn.Linear(mlp_dim, 1)

        # Inharmonicity: single scalar B per note (pooled over T)
        # f_k = k·f0·√(1 + B·k²),  B_MAX is MIDI-dependent
        # init bias=-5 → sigmoid≈0.007 → B≈0 at start
        self.head_B = nn.Linear(mlp_dim, 1)
        nn.init.constant_(self.head_B.bias, -5.0)

        # Per-partial physics decay: σ_k = b1 + b3 · (2π · k · f0)²
        # Pooled over T — note-level constants (physics: decay rate doesn't vary per frame)
        # b1 init: softplus(-1.0) ≈ 0.31 s⁻¹  (Steinway D: ~0.3)
        # b3 init: softplus(-16.0) ≈ 1.1e-7    (Steinway D: ~1e-7)
        self.head_b1 = nn.Linear(mlp_dim, 1)
        self.head_b3 = nn.Linear(mlp_dim, 1)
        nn.init.constant_(self.head_b1.bias, -1.0)
        nn.init.constant_(self.head_b3.bias, -16.0)

        self.harm_synth  = HarmonicSynth()
        self.noise_synth = NoiseSynth()

        nn.init.constant_(self.head_noise_L.bias,     -3.0)
        nn.init.constant_(self.head_noise_R.bias,     -3.0)
        nn.init.constant_(self.head_noise_amp_L.bias, -3.0)
        nn.init.constant_(self.head_noise_amp_R.bias, -3.0)

    def forward(self, f0_hz: torch.Tensor, loudness_db: torch.Tensor,
                velocity=None, n_frames: int = None,
                inh_scale: float = 1.0) -> torch.Tensor:
        """
        f0_hz       : (B, T)  Hz, 0 = unvoiced
        loudness_db : (B, T)  dB — applied post-synthesis as amplitude envelope
        velocity    : (B,)    MIDI velocity bucket 0–7
        inh_scale   : float   0 = no inharmonicity, 1 = learned B, 2 = double
        """
        B, T = f0_hz.shape
        if n_frames is None:
            n_frames = T
        if velocity is None:
            velocity = torch.full((B,), 5.0, dtype=torch.float32, device=f0_hz.device)

        # Encode conditioning — loudness excluded from timbre network
        f0_enc  = encode_f0(f0_hz, F0_BINS)
        vel_enc = encode_velocity(velocity, VEL_DIM).unsqueeze(1).expand(-1, T, -1)
        feat    = torch.cat([f0_enc, vel_enc], dim=-1)

        feat    = self.pre_mlp(feat)
        feat, _ = self.gru(feat)
        feat    = self.post_mlp(feat)

        # Harmonic parameters
        harm_dist_L = torch.softmax(self.head_harm_L(feat), dim=-1)
        harm_dist_R = torch.softmax(self.head_harm_R(feat), dim=-1)
        harm_amp_L  = F.softplus(self.head_amp_L(feat))
        harm_amp_R  = F.softplus(self.head_amp_R(feat))
        harm_amps_L = harm_dist_L * harm_amp_L
        harm_amps_R = harm_dist_R * harm_amp_R

        # Noise parameters
        noise_L = torch.sigmoid(self.head_noise_L(feat)) \
                  * F.softplus(self.head_noise_amp_L(feat))
        noise_R = torch.sigmoid(self.head_noise_R(feat)) \
                  * F.softplus(self.head_noise_amp_R(feat))

        # Inharmonicity: MIDI-dependent B_MAX, scalar per note (pooled over T)
        f0_mean  = f0_hz.clamp(min=F0_MIN).mean(dim=1)
        midi_est = 69.0 + 12.0 * torch.log2(f0_mean / 440.0)
        b_max    = 0.0008 * torch.exp(-(midi_est - 21.0) / 88.0 * math.log(10.0))
        inh      = torch.sigmoid(self.head_B(feat)).squeeze(-1).mean(dim=1) \
                   * b_max * inh_scale

        # Per-partial physics decay (note-level scalars pooled over T)
        b1 = F.softplus(self.head_b1(feat)).squeeze(-1).mean(dim=1)  # (B,)
        b3 = F.softplus(self.head_b3(feat)).squeeze(-1).mean(dim=1)  # (B,)

        # Synthesis
        n_samples   = n_frames * FRAME_HOP
        harmonic_L  = self.harm_synth(harm_amps_L, f0_hz, n_samples, inh=inh, b1=b1, b3=b3)
        harmonic_R  = self.harm_synth(harm_amps_R, f0_hz, n_samples, inh=inh, b1=b1, b3=b3)
        noise_sig_L = self.noise_synth(noise_L, f0_hz, n_samples)
        noise_sig_R = self.noise_synth(noise_R, f0_hz, n_samples)

        # Loudness envelope — post-synthesis linear multiplier
        lo_lin = 10.0 ** (loudness_db / 20.0)
        lo_up  = F.interpolate(lo_lin.unsqueeze(1).float(),
                               size=n_samples, mode='linear', align_corners=False)

        L = (harmonic_L + noise_sig_L) * lo_up
        R = (harmonic_R + noise_sig_R) * lo_up
        return torch.cat([L, R], dim=1)   # (B, 2, n_samples)


def build_ddsp(cfg: dict = None, size: str = None) -> DDSPVocoder:
    """Build DDSPVocoder from config dict or size name."""
    cfg = cfg or {}
    if size and size in MODEL_SIZES:
        params = MODEL_SIZES[size]
    else:
        params = dict(
            gru_hidden = cfg.get('gru_hidden', 64),
            gru_layers = cfg.get('gru_layers', 1),
            mlp_dim    = cfg.get('mlp_dim',    128),
        )
    return DDSPVocoder(**params)


def count_params(model: DDSPVocoder) -> int:
    return sum(p.numel() for p in model.parameters())
