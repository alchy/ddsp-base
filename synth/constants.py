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

# Model size presets — optimized for Grand Piano (88 notes × 8 velocity layers)
#
# small  (~598K):  quick diagnostics / first-run test (50 epochs)
#                  1-layer GRU, mlp_dim=256
#
# medium (~2.08M): recommended default for piano production
#                  2-layer GRU (handles long bass sequences from Phase 0),
#                  GRU dominates (47% of total) — correct balance for piano
#
# large  (~4.4M):  best quality; 2-layer GRU with 512 hidden
#                  GPU recommended (CPU very slow)
#                  noise heads plateau at mlp_dim=512 — GRU is the differentiator
MODEL_SIZES = {
    'small':  dict(gru_hidden=128, gru_layers=1, mlp_dim=256),
    'medium': dict(gru_hidden=256, gru_layers=2, mlp_dim=512),
    'large':  dict(gru_hidden=512, gru_layers=2, mlp_dim=512),
}
