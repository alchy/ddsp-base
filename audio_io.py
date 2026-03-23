"""
audio_io.py - DDSP Neural Vocoder — Audio I/O Utilities

Loading, saving and scanning WAV/MP3/FLAC files.
Mono files are automatically expanded to stereo.
"""

import os
import glob
from typing import List, Optional, Tuple

import numpy as np

SR = 48000


def load_wav_stereo(path: str, target_sr: int = SR) -> np.ndarray:
    """
    Load a WAV file as stereo float32 (2, T).
    Mono files are duplicated to stereo.
    Files at a different sample rate are resampled to target_sr.
    """
    import soundfile as sf

    audio, file_sr = sf.read(path, dtype='float32', always_2d=True)
    audio = audio.T   # (C, T)

    if audio.shape[0] == 1:
        audio = np.vstack([audio, audio])
    elif audio.shape[0] > 2:
        audio = audio[:2]

    if file_sr != target_sr:
        import librosa
        print(f'  [audio_io] Resampling {os.path.basename(path)}: '
              f'{file_sr} Hz -> {target_sr} Hz')
        audio = np.stack([
            librosa.resample(audio[ch], orig_sr=file_sr, target_sr=target_sr)
            for ch in range(2)
        ])

    return audio.astype(np.float32)


def save_wav(path: str, audio: np.ndarray, sr: int = SR):
    """
    Save (2, T) float32 to 16-bit PCM WAV.
    Peak-normalizes if necessary to prevent clipping.
    """
    import soundfile as sf

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    peak = float(np.abs(audio).max())
    if peak > 1.0:
        print(f'  [audio_io] Peak {peak:.3f} > 1.0 -- normalizing: '
              f'{os.path.basename(path)}')
        audio = audio / peak

    sf.write(path, audio.T, sr, subtype='PCM_16')


def scan_instrument_dir(directory: str) -> List[str]:
    """
    Find all instrument WAV files in directory.
    Prefers *-f48.wav (48 kHz), falls back to *-f44.wav (44.1 kHz).
    Returns sorted list of absolute paths.
    """
    files = sorted(glob.glob(os.path.join(directory, '*-f48.wav')))
    if not files:
        files = sorted(glob.glob(os.path.join(directory, '*-f44.wav')))
    if not files:
        print(f'  [audio_io] WARNING: no WAV files found in {directory}')
    else:
        print(f'  [audio_io] Found {len(files)} instrument files in {directory}')
    return files


def parse_filename(path: str) -> Optional[Tuple[int, int]]:
    """
    Parse instrument filename mXXX-velY-f44.wav -> (midi, vel).
    Returns None on parse failure.
      midi: MIDI note number (0–127)
      vel:  velocity level (0–7)
    """
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        parts = name.split('-')
        return int(parts[0][1:]), int(parts[1][3:])
    except Exception:
        return None
