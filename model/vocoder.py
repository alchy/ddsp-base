"""
model/vocoder.py — DDSPVocoder: neural network + synthesis pipeline

Architecture (Decoupled Timbre):
    Input:  F0 + velocity  (no loudness — applied post-synthesis)
    GRU:    learns timbre over time
    Heads:  predict synthesis parameters for HarmonicSynth + NoiseSynth
    Output: (B, 2, n_samples) stereo audio at sample rate SR

Loudness is applied as a linear envelope after synthesis.

Physics decay (Phase 2 — two-component model, Weinreich 1977, Bensa 2003):
    Two string polarizations (vertical fast, horizontal slow):
    σ_k_fast = b1_f + b3_f · (2π·k·f0)²
    σ_k_slow = b1_s + b3_s · (2π·k·f0)²
    d_k(t) = α · exp(-σ_k_fast · t) + (1-α) · exp(-σ_k_slow · t)

    decay_scale (inference only): 0=no decay, 1=learned, >1=faster doznívání
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

        # Inharmonicity: scalar B per note (pooled over T), MIDI-dependent B_MAX
        # init: sigmoid≈0.007 → B≈0 at start
        self.head_B = nn.Linear(mlp_dim, 1)
        nn.init.constant_(self.head_B.bias, -5.0)

        # Two-component physics decay (note-level scalars, pooled over T)
        #
        # Fast component (vertical polarization, stronger coupling to soundboard):
        #   b1_f init: softplus(0.5) ≈ 0.97 s⁻¹  → τ_fast ≈ 1.0 s
        #   b3_f init: softplus(-15) ≈ 3e-7        (freq-dependent term)
        #
        # Slow component (horizontal polarization, weaker coupling):
        #   b1_s init: softplus(-1.7) ≈ 0.15 s⁻¹ → τ_slow ≈ 6.7 s
        #   b3_s init: softplus(-19) ≈ 5e-9        (very weak freq-dep)
        #
        # alpha: fast fraction — sigmoid(0) = 0.5 (equal excitation at start)
        self.head_b1_f  = nn.Linear(mlp_dim, 1)
        self.head_b3_f  = nn.Linear(mlp_dim, 1)
        self.head_b1_s  = nn.Linear(mlp_dim, 1)
        self.head_b3_s  = nn.Linear(mlp_dim, 1)
        self.head_alpha = nn.Linear(mlp_dim, 1)
        nn.init.constant_(self.head_b1_f.bias,   0.5)
        nn.init.constant_(self.head_b3_f.bias,  -15.0)
        nn.init.constant_(self.head_b1_s.bias,  -1.7)
        nn.init.constant_(self.head_b3_s.bias,  -19.0)
        nn.init.constant_(self.head_alpha.bias,   0.0)

        self.harm_synth  = HarmonicSynth()
        self.noise_synth = NoiseSynth()

        nn.init.constant_(self.head_noise_L.bias,     -3.0)
        nn.init.constant_(self.head_noise_R.bias,     -3.0)
        nn.init.constant_(self.head_noise_amp_L.bias, -3.0)
        nn.init.constant_(self.head_noise_amp_R.bias, -3.0)

    def forward(self, f0_hz: torch.Tensor, loudness_db: torch.Tensor,
                velocity=None, n_frames: int = None,
                inh_scale: float = 1.0,
                decay_scale: float = 1.0) -> torch.Tensor:
        """
        f0_hz       : (B, T)  Hz, 0 = unvoiced
        loudness_db : (B, T)  dB — applied post-synthesis as amplitude envelope
        velocity    : (B,)    MIDI velocity bucket 0–7
        inh_scale   : float   0 = no inharmonicity, 1 = learned B, 2 = exaggerated
        decay_scale : float   0 = no physics decay, 1 = learned, >1 = faster decay
        """
        B, T = f0_hz.shape
        if n_frames is None:
            n_frames = T
        if velocity is None:
            velocity = torch.full((B,), 5.0, dtype=torch.float32, device=f0_hz.device)

        # Encode conditioning
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

        # Inharmonicity: MIDI-dependent B_MAX, pooled over T
        f0_mean  = f0_hz.clamp(min=F0_MIN).mean(dim=1)
        midi_est = 69.0 + 12.0 * torch.log2(f0_mean / 440.0)
        b_max    = 0.0008 * torch.exp(-(midi_est - 21.0) / 88.0 * math.log(10.0))
        inh      = torch.sigmoid(self.head_B(feat)).squeeze(-1).mean(dim=1) \
                   * b_max * inh_scale

        # Two-component physics decay (note-level scalars, pooled over T)
        b1_f  = F.softplus(self.head_b1_f(feat)).squeeze(-1).mean(dim=1)  * decay_scale
        b3_f  = F.softplus(self.head_b3_f(feat)).squeeze(-1).mean(dim=1)  * decay_scale
        b1_s  = F.softplus(self.head_b1_s(feat)).squeeze(-1).mean(dim=1)  * decay_scale
        b3_s  = F.softplus(self.head_b3_s(feat)).squeeze(-1).mean(dim=1)  * decay_scale
        alpha = torch.sigmoid(self.head_alpha(feat)).squeeze(-1).mean(dim=1)

        # Synthesis
        n_samples   = n_frames * FRAME_HOP
        harmonic_L  = self.harm_synth(harm_amps_L, f0_hz, n_samples, inh,
                                      b1_f, b3_f, b1_s, b3_s, alpha)
        harmonic_R  = self.harm_synth(harm_amps_R, f0_hz, n_samples, inh,
                                      b1_f, b3_f, b1_s, b3_s, alpha)
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
