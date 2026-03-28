"""
synth/constants.py — shared physical and model constants

All modules import from here to avoid circular dependencies and
to have a single authoritative source for every tunable constant.
"""

SR         = 48000
FRAME_HOP  = 240          # 5 ms per frame @ 48 kHz

N_HARM_MAX = 128          # max harmonic oscillators; active count = min(N_HARM_MAX, floor(nyquist/f0))
N_HARM     = N_HARM_MAX   # alias used in head sizes and normalization

NOISE_FFT  = 1024
N_NOISE    = NOISE_FFT // 2 + 1   # = 513 spectral bins

F0_MIN  = 20.0
F0_MAX  = 5000.0
F0_BINS = 64
LO_DIM  = 16
VEL_DIM = 8

MODEL_SIZES = {
    'small':  dict(gru_hidden=64,  gru_layers=1, mlp_dim=128),   # ~84 K params
    'medium': dict(gru_hidden=128, gru_layers=2, mlp_dim=256),   # ~400 K params
    'large':  dict(gru_hidden=256, gru_layers=3, mlp_dim=512),   # ~1.9 M params
}
