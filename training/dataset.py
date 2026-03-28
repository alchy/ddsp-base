"""
training/dataset.py — SourceDataset: loads NPZ feature files for DDSP training

Adaptive window size (Phase 0 of bass refactor):
    Bass notes have τ_slow ≈ 5 s (two-stage decay from zig-zag polarization).
    A fixed 250 ms window covers only 0.8 % of a 29 s bass sample — the model
    never observes slow decay, so it cannot learn it.

    crop_frames(midi) grows the training window for low MIDI notes:
        MIDI ≤ 24 (A0): 2000 frames = 10 s  — captures ~2× τ_slow
        MIDI = 48 (C3): ~1025 frames = 5 s
        MIDI ≥ 72 (C5):   50 frames = 0.25 s  — original behaviour
"""

from __future__ import annotations

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from synth.constants import SR, FRAME_HOP

# Default window sizes (frames)
CROP_FRAMES_MAX  = 2000   # 10 s — deepest bass
CROP_FRAMES_MIN  =   50   # 0.25 s — treble (original)
CROP_MIDI_LOW    =   24   # A0: full long window
CROP_MIDI_HIGH   =   72   # C5: short window


def crop_frames(midi: int) -> int:
    """Return training window length in frames for a given MIDI note.

    Linear interpolation between CROP_FRAMES_MAX at MIDI≤24
    and CROP_FRAMES_MIN at MIDI≥72.
    """
    t = (midi - CROP_MIDI_LOW) / (CROP_MIDI_HIGH - CROP_MIDI_LOW)
    t = max(0.0, min(1.0, t))
    return int(round(CROP_FRAMES_MAX + t * (CROP_FRAMES_MIN - CROP_FRAMES_MAX)))


class SourceDataset(Dataset):
    """Loads all NPZ extract files and yields fixed-length training windows.

    Each NPZ contains:
        audio        : (2, T_samples)   float32 stereo audio
        f0           : (T_frames,)      float32 Hz, 0 = unvoiced
        loudness_L   : (T_frames,)      float32 dB
        loudness_R   : (T_frames,)      float32 dB
        voiced_prob  : (T_frames,)      float32 [0, 1]
        vel_frames   : (T_frames,)      float32 velocity bucket 0–7

    Filename convention:  mXXX-velY-fXX.npz  (parsed for MIDI note).
    """

    def __init__(self, extracts_dir: str, min_voiced: float = 0.1):
        from audio_io import parse_filename
        self.hop   = FRAME_HOP
        self.items = []     # list of (file_index, start_frame, crop_len)
        self.data  = []
        self.midi_per_file = []

        npz_files = sorted(glob.glob(os.path.join(extracts_dir, '*.npz')))
        if not npz_files:
            raise FileNotFoundError(
                f'No NPZ files in {extracts_dir}.\n'
                f'Run: python ddsp.py extract --instrument <path>'
            )

        total_frames = 0
        for fi, path in enumerate(npz_files):
            print(f'  loading {os.path.basename(path)} ...', flush=True)
            raw = np.load(path)
            d   = {k: raw[k].copy() for k in raw.files}
            self.data.append(d)

            parsed = parse_filename(path)
            midi   = parsed[0] if parsed else 60
            self.midi_per_file.append(midi)

            crop = crop_frames(midi)
            T    = len(d['f0'])
            if T < crop + 1:
                continue
            step = max(1, crop // 2)
            for start in range(0, T - crop, step):
                if d['voiced_prob'][start:start + crop].mean() >= min_voiced:
                    self.items.append((fi, start, crop))
            total_frames += T

        if not self.items:
            print('[dataset] WARNING: no voiced windows found -- using all windows.')
            for fi, d in enumerate(self.data):
                midi = self.midi_per_file[fi]
                crop = crop_frames(midi)
                T    = len(d['f0'])
                for start in range(0, T - crop, max(1, crop // 4)):
                    self.items.append((fi, start, crop))

        dur = total_frames / (SR / FRAME_HOP)
        print(f'[dataset] {len(npz_files)} file(s)  {dur:.0f}s  {len(self.items)} windows')

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fi, start, crop = self.items[idx]
        d   = self.data[fi]
        sl  = slice(start, start + crop)

        f0  = d['f0'][sl].astype(np.float32)
        loL = d['loudness_L'][sl].astype(np.float32)
        loR = d['loudness_R'][sl].astype(np.float32)
        vp  = d['voiced_prob'][sl].astype(np.float32)

        s_s   = start * self.hop
        audio = d['audio'][:, s_s:s_s + crop * self.hop].astype(np.float32)

        # Random gain augmentation ±2 dB
        if random.random() < 0.5:
            db    = random.uniform(-2.0, 2.0)
            g     = 10.0 ** (db / 20.0)
            audio = audio * g
            loL  += db
            loR  += db

        vel_mean = float(d['vel_frames'][sl].mean())
        midi     = self.midi_per_file[fi]
        return (
            torch.from_numpy(f0),
            torch.from_numpy(loL),
            torch.from_numpy(loR),
            torch.from_numpy(vp),
            torch.from_numpy(audio),
            torch.tensor(vel_mean, dtype=torch.float32),
            torch.tensor(midi,     dtype=torch.long),
            torch.tensor(start,    dtype=torch.long),
        )
