"""
model_ddsp.py - DDSP Neural Vocoder

Signal model:
    φ_k(t) = cumsum(2π · f_k(t) / SR)               [phase accumulation]
    f_k(t) = k · F0(t) · √(1 + inh · k²)            [inharmonic partial frequencies]
    L(t) = A_L(t) · Σ_k  d_k_L(t) · sin(φ_k(t))   [harmonic oscillators, left]
         + noise_STFT_shaped_L(t)                     [filtered noise, left]
    R(t) = A_R(t) · Σ_k  d_k_R(t) · sin(φ_k(t))   [harmonic oscillators, right]
         + noise_STFT_shaped_R(t)                     [filtered noise, right]

where d_k  (harmonic distribution) = softmax(head_harm),  sum_k d_k = 1
      A    (global amplitude)       = softplus(head_amp)
      inh  (inharmonicity coeff)    = sigmoid(head_B).mean(T) × B_MAX(f0)
           B_MAX(f0) = 0.0008 · exp(-(midi(f0) − 21) / 88 · ln10)
           pools over time → single scalar per note, physically correct
      n_active (adaptive harmonic count) = min(N_HARM_MAX, floor(nyquist / f0))
           bass notes use more harmonics than treble notes automatically

Conditioned on (F0, loudness, velocity) per 5 ms frame.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SR        = 48000
FRAME_HOP = 240           # 5 ms per frame @ 48 kHz
N_HARM_MAX = 128          # max harmonic oscillators; active count = min(N_HARM_MAX, floor(nyquist/f0))
N_HARM     = N_HARM_MAX   # alias used throughout (head sizes, normalization)
NOISE_FFT = 1024
N_NOISE   = NOISE_FFT // 2 + 1   # = 513 spectral bins

F0_MIN  = 20.0
F0_MAX  = 5000.0
F0_BINS = 64
LO_DIM  = 16
VEL_DIM = 8

# Model size presets
# small:  ~84K  params — fast CPU training, baseline quality
# medium: ~400K params — good balance, recommended default
# large:  ~1.9M params — best quality, slow on CPU
MODEL_SIZES = {
    'small':  dict(gru_hidden=64,  gru_layers=1, mlp_dim=128),
    'medium': dict(gru_hidden=128, gru_layers=2, mlp_dim=256),
    'large':  dict(gru_hidden=256, gru_layers=3, mlp_dim=512),
}


# ---------------------------------------------------------------------------
# Feature encoders (fixed, no parameters)
# ---------------------------------------------------------------------------

def encode_f0(f0_hz: torch.Tensor, bins: int = F0_BINS) -> torch.Tensor:
    """Sinusoidal log-frequency encoding.  f0_hz: (B,T) -> (B,T,bins)"""
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
    """Sinusoidal velocity encoding.  velocity: (B,) -> (B,dim)"""
    device   = velocity.device
    vel_norm = (velocity.float() / 7.0).clamp(0.0, 1.0)
    i   = torch.arange(1, dim // 2 + 1, dtype=torch.float32, device=device)
    enc = torch.zeros(velocity.shape[0], dim, device=device)
    enc[:, 0::2] = torch.sin(math.pi * i * vel_norm.unsqueeze(-1))
    enc[:, 1::2] = torch.cos(math.pi * i * vel_norm.unsqueeze(-1))
    return enc


def encode_loudness(loudness_db: torch.Tensor, dim: int = LO_DIM) -> torch.Tensor:
    """Sinusoidal loudness encoding.  loudness_db: (B,T) -> (B,T,dim)"""
    B, T   = loudness_db.shape
    device = loudness_db.device
    lo_norm = ((loudness_db + 80.0) / 80.0).clamp(0.0, 1.0)
    i   = torch.arange(1, dim // 2 + 1, dtype=torch.float32, device=device)
    enc = torch.zeros(B, T, dim, device=device)
    enc[:, :, 0::2] = torch.sin(math.pi * i * lo_norm.unsqueeze(-1))
    enc[:, :, 1::2] = torch.cos(math.pi * i * lo_norm.unsqueeze(-1))
    return enc


# ---------------------------------------------------------------------------
# Harmonic Synthesizer
# ---------------------------------------------------------------------------

class HarmonicSynth(nn.Module):
    """
    Additive synthesis: weighted sum of up to N_HARM_MAX sinusoids at k * F0(t).

    The number of active harmonics is adaptive per batch item:
        n_active = min(N_HARM_MAX, floor(nyquist / f0))
    so bass notes automatically use more harmonics than treble notes.
    Signal is normalized by n_active so output level is consistent.

    Output: (B, 1, n_samples)
    """

    def forward(self, harm_amps, f0_hz, n_samples, inh):
        """
        harm_amps : (B, T, N)
        f0_hz     : (B, T)
        n_samples : int
        inh       : (B,)  inharmonicity coefficient — f_k = k · f0 · √(1 + inh · k²)
        """
        B, T_frames, N = harm_amps.shape
        device = harm_amps.device

        a     = F.interpolate(harm_amps.permute(0, 2, 1), size=n_samples, mode='linear', align_corners=False)
        f0_up = F.interpolate(f0_hz.unsqueeze(1).float(), size=n_samples, mode='linear', align_corners=False).squeeze(1)

        k = torch.arange(1, N + 1, dtype=torch.float32, device=device)   # (N,)

        # Piano inharmonicity: partial k at k·f0·√(1 + inh·k²)
        stretch   = torch.sqrt(1.0 + inh.unsqueeze(1) * k ** 2)          # (B, N)
        inst_freq = f0_up.unsqueeze(1) * (k.unsqueeze(0) * stretch).unsqueeze(2)  # (B, N, n_samples)

        nyq_mask = (inst_freq < SR * 0.45).float()
        a = a * nyq_mask

        # Adaptive normalization: divide by n_active harmonics per batch item,
        # not by fixed N — bass notes use more harmonics than treble notes.
        # n_active = number of harmonics with k·f0 < nyquist (using mean f0)
        f0_mean  = f0_up.mean(dim=-1)                                      # (B,)
        n_active = ((k.unsqueeze(0) * f0_mean.unsqueeze(1)) < SR * 0.45) \
                   .sum(dim=1).clamp(min=1).float()                        # (B,)

        # Phase accumulation: cumsum over instantaneous frequency
        phase = torch.cumsum(2.0 * math.pi * inst_freq / SR, dim=-1)

        signal = (a * torch.sin(phase)).sum(dim=1, keepdim=True)
        return signal / n_active.view(B, 1, 1)


# ---------------------------------------------------------------------------
# Noise Synthesizer
# ---------------------------------------------------------------------------

class NoiseSynth(nn.Module):
    """
    Filtered noise via STFT shaping.
    Output: (B, 1, n_samples)
    """

    def __init__(self, n_fft=NOISE_FFT):
        super().__init__()
        self.n_fft = n_fft
        self.hop   = n_fft // 4

    def forward(self, noise_mags, f0_hz, n_samples):
        """
        noise_mags : (B, T_frames, N)  spectral shape in harmonic-relative space
                     bin k → absolute freq  k · f0 · N_HARM / N
                     keeps the same spectral shape relative to f0 across all pitches
        f0_hz      : (B, T)            fundamental frequency (Hz); mean used for warping
        n_samples  : int
        """
        B, T_frames, N = noise_mags.shape
        device = noise_mags.device
        n_fft, hop = self.n_fft, self.hop
        n_bins = n_fft // 2 + 1

        window = torch.hann_window(n_fft, device=device)
        noise  = torch.randn(B, n_samples, device=device)
        stft   = torch.stft(noise, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                             window=window, return_complex=True)
        T_stft = stft.shape[-1]

        # Upsample noise_mags from model frame rate to STFT frame rate
        mag_up = F.interpolate(noise_mags.permute(0, 2, 1), size=T_stft,
                                mode='linear', align_corners=False)   # (B, N, T_stft)

        # Warp harmonic-relative bins → absolute STFT bins
        # Noise bin k represents freq  k · f0 · N_HARM / N
        # STFT bin i represents freq   i · SR / n_fft
        # → for STFT bin i: source noise bin = i · (SR/n_fft) / f0 · N / N_HARM
        f0_mean    = f0_hz.clamp(min=F0_MIN).mean(dim=1)                        # (B,)
        stft_freqs = torch.arange(n_bins, dtype=torch.float32, device=device) \
                     * (SR / n_fft)                                              # (n_bins,)
        src_pos    = stft_freqs.unsqueeze(0) / f0_mean.unsqueeze(1) \
                     * (N / N_HARM)                                              # (B, n_bins)
        src_pos    = src_pos.clamp(0.0, N - 1.0)

        # Linear interpolation: gather at floor/ceil, blend with fractional part
        idx_lo  = src_pos.long()                                                 # (B, n_bins)
        idx_hi  = (idx_lo + 1).clamp(max=N - 1)
        frac    = (src_pos - idx_lo.float()) \
                  .unsqueeze(-1).expand(-1, -1, T_stft)                          # (B, n_bins, T_stft)
        idx_lo  = idx_lo.unsqueeze(-1).expand(-1, -1, T_stft)                   # (B, n_bins, T_stft)
        idx_hi  = idx_hi.unsqueeze(-1).expand(-1, -1, T_stft)
        val_lo  = torch.gather(mag_up, 1, idx_lo)
        val_hi  = torch.gather(mag_up, 1, idx_hi)
        mag_abs = val_lo + frac * (val_hi - val_lo)                              # (B, n_bins, T_stft)

        out = torch.istft(stft * mag_abs, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=window, length=n_samples)
        return out.unsqueeze(1)


# ---------------------------------------------------------------------------
# DDSPVocoder
# ---------------------------------------------------------------------------

class DDSPVocoder(nn.Module):
    """
    DDSP Neural Vocoder conditioned on (F0, loudness, velocity).

    Fully stereo: independent harmonic amplitude profiles and noise filters
    for L and R channels.  The model learns per-channel overtone balance
    (e.g. piano string position, mic placement) directly from training data.

    Returns: (B, 2, n_samples) stereo audio.
    """

    def __init__(self, gru_hidden=64, gru_layers=1, mlp_dim=128):
        super().__init__()

        # Loudness removed from GRU input — applied post-synthesis as a
        # linear multiplier so the network learns pure timbre independently.
        feat_dim = F0_BINS + VEL_DIM   # 72

        self.pre_mlp = nn.Sequential(
            nn.Linear(feat_dim, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim,  mlp_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(mlp_dim, gru_hidden, num_layers=gru_layers, batch_first=True)

        # Post-GRU projection: richer output conditioning
        self.post_mlp = nn.Sequential(
            nn.Linear(gru_hidden, mlp_dim), nn.ReLU(),
        )

        # Independent harmonic amplitude profiles per channel
        # harm_dist: softmax → normalized harmonic distribution (sums to 1)
        # harm_amp:  softplus → global amplitude scalar per frame
        self.head_harm_L  = nn.Linear(mlp_dim, N_HARM)
        self.head_harm_R  = nn.Linear(mlp_dim, N_HARM)
        self.head_amp_L   = nn.Linear(mlp_dim, 1)
        self.head_amp_R   = nn.Linear(mlp_dim, 1)
        self.head_noise_L     = nn.Linear(mlp_dim, N_NOISE)
        self.head_noise_R     = nn.Linear(mlp_dim, N_NOISE)
        # Global noise amplitude scalar — decouples spectral shape from level
        self.head_noise_amp_L = nn.Linear(mlp_dim, 1)
        self.head_noise_amp_R = nn.Linear(mlp_dim, 1)

        # Piano inharmonicity: single scalar B per note (pooled over T)
        # f_k = k·f0·√(1 + B·k²),  B_MAX is MIDI-dependent (see forward)
        # init bias=-5 → sigmoid≈0.007 → B≈0 at start (safe, no pitch disruption)
        self.head_B = nn.Linear(mlp_dim, 1)
        nn.init.constant_(self.head_B.bias, -5.0)

        self.harm_synth  = HarmonicSynth()
        self.noise_synth = NoiseSynth()

        nn.init.constant_(self.head_noise_L.bias,     -3.0)
        nn.init.constant_(self.head_noise_R.bias,     -3.0)
        nn.init.constant_(self.head_noise_amp_L.bias, -3.0)
        nn.init.constant_(self.head_noise_amp_R.bias, -3.0)

    def forward(self, f0_hz, loudness_db, velocity=None, n_frames=None, inh_scale=1.0):
        """
        f0_hz       : (B, T)  in Hz, 0 = unvoiced
        loudness_db : (B, T)  in dB (used as post-synthesis amplitude envelope)
        velocity    : (B,)    MIDI velocity bucket 0-7
        """
        B, T = f0_hz.shape
        if n_frames is None:
            n_frames = T
        if velocity is None:
            velocity = torch.full((B,), 5.0, dtype=torch.float32, device=f0_hz.device)

        # Timbre network: F0 + velocity only (no loudness)
        f0_enc  = encode_f0(f0_hz, F0_BINS)
        vel_enc = encode_velocity(velocity, VEL_DIM).unsqueeze(1).expand(-1, T, -1)
        feat    = torch.cat([f0_enc, vel_enc], dim=-1)

        feat    = self.pre_mlp(feat)
        feat, _ = self.gru(feat)
        feat    = self.post_mlp(feat)

        # Canonical DDSP: softmax distribution × softplus global amplitude
        harm_dist_L = torch.softmax(self.head_harm_L(feat), dim=-1)
        harm_dist_R = torch.softmax(self.head_harm_R(feat), dim=-1)
        harm_amp_L  = F.softplus(self.head_amp_L(feat))   # (B, T, 1)
        harm_amp_R  = F.softplus(self.head_amp_R(feat))
        harm_amps_L = harm_dist_L * harm_amp_L
        harm_amps_R = harm_dist_R * harm_amp_R
        noise_L     = torch.sigmoid(self.head_noise_L(feat)) \
                      * F.softplus(self.head_noise_amp_L(feat))
        noise_R     = torch.sigmoid(self.head_noise_R(feat)) \
                      * F.softplus(self.head_noise_amp_R(feat))

        # Inharmonicity: MIDI-dependent B_MAX from mean F0, pooled scalar per note
        # B_MAX = 0.0008 · exp(-(midi-21)/88·ln10)  — basy ~0.0008, výšky ~0.00008
        f0_mean  = f0_hz.clamp(min=F0_MIN).mean(dim=1)                          # (B,)
        midi_est = 69.0 + 12.0 * torch.log2(f0_mean / 440.0)                   # (B,)
        b_max    = 0.0008 * torch.exp(-(midi_est - 21.0) / 88.0 * math.log(10.0))
        inh      = torch.sigmoid(self.head_B(feat)).squeeze(-1).mean(dim=1) \
                   * b_max * inh_scale                                           # (B,)

        n_samples   = n_frames * FRAME_HOP
        harmonic_L  = self.harm_synth(harm_amps_L, f0_hz, n_samples, inh=inh)
        harmonic_R  = self.harm_synth(harm_amps_R, f0_hz, n_samples, inh=inh)
        noise_sig_L = self.noise_synth(noise_L, f0_hz, n_samples)
        noise_sig_R = self.noise_synth(noise_R, f0_hz, n_samples)

        # Loudness applied as a linear amplitude envelope (post-synthesis)
        # Convert dB → linear, upsample from frame rate to sample rate
        lo_lin = 10.0 ** (loudness_db / 20.0)                                    # (B, T)
        lo_up  = F.interpolate(lo_lin.unsqueeze(1).float(),
                               size=n_samples, mode='linear',
                               align_corners=False)                               # (B, 1, n_samples)

        L = (harmonic_L + noise_sig_L) * lo_up
        R = (harmonic_R + noise_sig_R) * lo_up
        return torch.cat([L, R], dim=1)   # (B, 2, n_samples)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ddsp(cfg: dict = None, size: str = None) -> DDSPVocoder:
    """
    Build DDSPVocoder from config dict or size name.
    size: 'small' | 'medium' | 'large'  (overrides cfg keys)
    """
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


# ---------------------------------------------------------------------------
# EnvelopeNet — tiny MLP for loudness envelope prediction
# ---------------------------------------------------------------------------

N_ENV          = 512    # number of envelope control points output by EnvelopeNet
ENVELOPE_WARP  = 4.0   # power-law time warp — concentrates resolution near t=0 (attack)
                        # t_store[i] = (i/(N-1))^WARP  → first N/2 points cover first 6% of note


class EnvelopeNet(nn.Module):
    """
    Tiny MLP: (midi_norm, vel_norm) -> (dur_s, shape[N_ENV])

    Learns instrument-specific loudness envelopes from NPZ training data.
    The shape vector uses a warped time axis (power-law, exponent ENVELOPE_WARP)
    so that early control points cover the attack region at ~2 ms resolution,
    while later points spread across the sustain/release at coarser resolution.

    Normalization: midi_norm = midi / 127, vel_norm = vel / 7
    Parameters: ~30 K (hidden=64, N_ENV=512)
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
        midi_norm, vel_norm: (B,) in [0, 1]
        Returns:
          dur_s  — (B,)       predicted duration in seconds (≥ 0.5 s)
          shape  — (B, n_env) loudness curve in dB at warped time steps
        """
        x   = torch.stack([midi_norm, vel_norm], dim=-1)
        out = self.net(x)
        dur_s = F.softplus(out[:, 0]) + 0.5   # floor at 0.5 s
        shape = out[:, 1:]                      # (B, n_env) unconstrained dB
        return dur_s, shape

    def predict_envelope(self, midi: int, vel: int,
                         warp: float = ENVELOPE_WARP) -> 'np.ndarray':
        """Return loudness np.ndarray (n_frames,) for a single (midi, vel).

        The warped control points are interpolated back to a uniform frame grid
        (FRAME_HOP = 5 ms) for the DDSP vocoder.
        """
        import numpy as np
        device    = next(self.parameters()).device
        midi_t    = torch.tensor([midi / 127.0], dtype=torch.float32, device=device)
        vel_t     = torch.tensor([vel  /   7.0], dtype=torch.float32, device=device)
        with torch.no_grad():
            dur_s, shape = self(midi_t, vel_t)
        dur_s    = float(dur_s[0])
        shape_np = shape[0].cpu().numpy()
        n_frames = max(1, round(dur_s * SR / FRAME_HOP))
        # Warped positions where the N_ENV values are stored
        t_store = np.power(np.linspace(0.0, 1.0, self.n_env), warp)
        # Uniform frame positions to reconstruct
        t_query = np.linspace(0.0, 1.0, n_frames)
        return np.interp(t_query, t_store, shape_np).astype(np.float32)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for size, cfg in MODEL_SIZES.items():
        m = DDSPVocoder(**cfg)
        n = count_params(m)
        print(f'{size:8s}  {n:>9,} params  gru_hidden={cfg["gru_hidden"]}  '
              f'layers={cfg["gru_layers"]}  mlp_dim={cfg["mlp_dim"]}')
