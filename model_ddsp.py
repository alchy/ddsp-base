"""
model_ddsp.py — backward-compatibility shim

All production code has moved to the framework packages:
    synth/      — HarmonicSynth, NoiseSynth, constants
    model/      — DDSPVocoder, EnvelopeNet, encoders
    training/   — SourceDataset, mrstft_loss, _attack_weight

This file re-exports every name that ddsp.py (and any external scripts)
previously imported from model_ddsp directly, so existing code keeps working
without modification.
"""

# Constants (re-exported from synth.constants)
from synth.constants import (
    SR, FRAME_HOP,
    N_HARM_MAX, N_HARM, NOISE_FFT, N_NOISE,
    F0_MIN, F0_MAX, F0_BINS, LO_DIM, VEL_DIM,
    MODEL_SIZES,
)

# Encoders
from model.encoders import encode_f0, encode_velocity, encode_loudness

# Vocoder
from model.vocoder import DDSPVocoder, build_ddsp, count_params

# Envelope predictor
from model.envelope import EnvelopeNet, N_ENV, ENVELOPE_WARP

# Synthesizers (for scripts that import them directly)
from synth.harmonic import HarmonicSynth
from synth.noise    import NoiseSynth

__all__ = [
    'SR', 'FRAME_HOP',
    'N_HARM_MAX', 'N_HARM', 'NOISE_FFT', 'N_NOISE',
    'F0_MIN', 'F0_MAX', 'F0_BINS', 'LO_DIM', 'VEL_DIM',
    'MODEL_SIZES',
    'encode_f0', 'encode_velocity', 'encode_loudness',
    'DDSPVocoder', 'build_ddsp', 'count_params',
    'EnvelopeNet', 'N_ENV', 'ENVELOPE_WARP',
    'HarmonicSynth', 'NoiseSynth',
]
