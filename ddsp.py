"""
ddsp.py - DDSP Neural Vocoder — unified entry point

Usage:
    python ddsp.py extract   --instrument <path>
    python ddsp.py learn     --instrument <path> [--model small|medium|large] [--epochs 100]
    python ddsp.py generate  --instrument <path> [--wet 1.0]
    python ddsp.py status    --instrument <path>

Directory layout:
    Source samples (READ-ONLY):
        C:\\SoundBanks\\ddsp\\<instrument>\\   ← mXXX-velX-fXX.wav + instrument-definition.json

    Workspace (auto-created next to source, or --workspace override):
        <instrument>-ddsp\\
          extracts\\          ← cached NPZ features
          checkpoints\\       ← best.pt, last.pt
          instrument.json    ← config & training status
          train.log          ← training log

    Generated output (IthacaPlayer):
        C:\\SoundBanks\\IthacaPlayer\\<instrument>\\   ← synthesized WAV bank
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import random
import glob
import shutil
import argparse
import datetime
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model_ddsp import (
    build_ddsp, count_params, FRAME_HOP, SR, F0_MAX,
    DDSPVocoder, MODEL_SIZES,
    EnvelopeNet, N_ENV, ENVELOPE_WARP,
)
from audio_io import load_wav_stereo, save_wav, scan_instrument_dir, parse_filename

CROP_FRAMES = 50   # 0.25 s of frames per training window

# IthacaPlayer output root — generated sample banks land here
ITHACA_PLAYER_ROOT = os.environ.get('ITHACA_ROOT', r'C:\SoundBanks\IthacaPlayer')


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

@dataclass
class Workspace:
    source_dir: str   # READ-ONLY: original instrument WAV files
    work_dir:   str   # all outputs live here

    @property
    def extracts_dir(self):    return os.path.join(self.work_dir, 'extracts')
    @property
    def checkpoints_dir(self): return os.path.join(self.work_dir, 'checkpoints')
    @property
    def generated_dir(self):   return os.path.join(self.work_dir, 'generated')
    @property
    def config_path(self):     return os.path.join(self.work_dir, 'instrument.json')
    @property
    def log_path(self):        return os.path.join(self.work_dir, 'train.log')
    @property
    def name(self):
        return os.path.basename(os.path.normpath(self.source_dir))

    def makedirs(self):
        for d in [self.work_dir, self.extracts_dir,
                  self.checkpoints_dir, self.generated_dir]:
            os.makedirs(d, exist_ok=True)


def resolve_device(requested: str = 'auto') -> 'torch.device':
    """Return torch.device from 'auto' | 'cpu' | 'mps' | 'cuda'."""
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(requested)


def make_workspace(instrument: str, workspace: str = None) -> Workspace:
    source_dir = os.path.abspath(instrument)
    if workspace:
        work_dir = os.path.abspath(workspace)
    else:
        work_dir = source_dir.rstrip('/\\') + '-ddsp'
    return Workspace(source_dir=source_dir, work_dir=work_dir)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')


def load_config(ws: Workspace) -> dict:
    if os.path.exists(ws.config_path):
        with open(ws.config_path, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_config(ws: Workspace, cfg: dict):
    ws.makedirs()
    with open(ws.config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def load_audio_any(path: str) -> np.ndarray:
    """Load MP3/WAV/FLAC as (2, T) float32 at SR.  Mono -> duplicated to stereo."""
    try:
        import soundfile as sf
        audio, file_sr = sf.read(path, dtype='float32', always_2d=True)
        audio = audio.T
    except Exception:
        import librosa
        y, file_sr = librosa.load(path, sr=None, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        audio = y.astype(np.float32)

    if audio.shape[0] == 1:
        audio = np.vstack([audio, audio])
    elif audio.shape[0] > 2:
        audio = audio[:2]

    if file_sr != SR:
        import librosa
        audio = np.stack([
            librosa.resample(audio[ch], orig_sr=file_sr, target_sr=SR)
            for ch in range(2)
        ])
    return audio.astype(np.float32)


def scan_source_dir(source_dir: str):
    """Return all audio files in source_dir."""
    exts = ('*.wav','*.WAV','*.mp3','*.MP3','*.flac','*.FLAC','*.ogg','*.aiff','*.aif')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
    return sorted(set(files))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def loudness_db(audio: np.ndarray, hop: int = FRAME_HOP) -> np.ndarray:
    """Per-frame RMS loudness in dB.  audio: (2,T) or (T,) -> (T_frames,)"""
    mono     = audio.mean(axis=0) if audio.ndim > 1 else audio
    n_frames = max(1, len(mono) // hop)
    lo       = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s, e = i * hop, min((i + 1) * hop, len(mono))
        lo[i] = 20.0 * np.log10(float(np.sqrt(np.mean(mono[s:e] ** 2) + 1e-10)) + 1e-10)
    return lo


def extract_features(audio: np.ndarray):
    """Extract features using pyin F0 estimation (slow, ~20s/file).
    Used as fallback when MIDI note is not known from filename."""
    import librosa
    mono = audio.mean(axis=0)
    f0, _, voiced_prob = librosa.pyin(
        mono.astype(np.float64), fmin=27.5, fmax=float(F0_MAX),
        sr=SR, hop_length=FRAME_HOP, frame_length=4096, fill_na=0.0,
    )
    f0          = np.nan_to_num(f0,          nan=0.0).astype(np.float32)
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0).astype(np.float32)
    lo_L = loudness_db(audio[0])
    lo_R = loudness_db(audio[1])
    T    = min(len(f0), len(lo_L), len(lo_R))
    return dict(f0=f0[:T], loudness_L=lo_L[:T], loudness_R=lo_R[:T], voiced_prob=voiced_prob[:T])


def extract_features_known_f0(audio: np.ndarray, midi: int):
    """Extract features using known MIDI note as F0 (fast, <0.1s/file).
    Default path for mXXX-velX-fXX.wav banks where note is in filename.

    voiced_prob is derived from RMS loudness: frames above -60 dB are
    considered voiced. This is accurate for isolated note samples.
    """
    lo_L = loudness_db(audio[0])
    lo_R = loudness_db(audio[1])
    T    = min(len(lo_L), len(lo_R))
    freq = float(440.0 * (2.0 ** ((midi - 69) / 12.0)))
    f0   = np.full(T, freq, dtype=np.float32)
    # Voiced = loudness above noise floor (-60 dB), smoothed over 3 frames
    lo_m = (lo_L[:T] + lo_R[:T]) * 0.5
    raw  = (lo_m > -60.0).astype(np.float32)
    voiced_prob = np.convolve(raw, np.ones(3) / 3.0, mode='same').astype(np.float32)
    return dict(f0=f0, loudness_L=lo_L[:T], loudness_R=lo_R[:T], voiced_prob=voiced_prob)


def extract_and_cache(source_dir: str, extracts_dir: str, chunk_sec: int = 0,
                      force_pyin: bool = False):
    """Extract & cache features as NPZ.  Skips already-cached files.

    When the filename follows the mXXX-velY-fXX.wav convention the MIDI note
    is used directly as F0 (fast, <0.1 s/file).  Pass force_pyin=True to
    always run the slower pyin estimator instead.
    """
    os.makedirs(extracts_dir, exist_ok=True)
    files = scan_source_dir(source_dir)
    if not files:
        print(f'[extract] ERROR: no audio files in {source_dir}')
        sys.exit(1)
    print(f'[extract] {len(files)} file(s)  cache->{extracts_dir}'
          + (f'  chunk_sec={chunk_sec}' if chunk_sec else ''))
    n_new = 0
    for path in files:
        name   = os.path.splitext(os.path.basename(path))[0]
        legacy = os.path.join(extracts_dir, name + '.npz')
        chunk0 = os.path.join(extracts_dir, f'{name}_chunk000.npz')
        if os.path.exists(legacy) or os.path.exists(chunk0):
            continue
        try:
            t0    = time.time()
            audio = load_audio_any(path)
            dur_s = audio.shape[-1] / SR

            parsed = parse_filename(path)
            known_midi = parsed[0] if parsed else None
            if parsed and parsed[1] is not None:
                vel_constant, vel_dynamic = float(parsed[1]), None
            else:
                lo_full = (loudness_db(audio[0]) + loudness_db(audio[1])) * 0.5
                vel_dynamic  = (np.percentile(lo_full, 5), np.percentile(lo_full, 95))
                vel_constant = None

            n_chunks  = max(1, math.ceil(audio.shape[1] / (chunk_sec * SR))) if chunk_sec else 1
            use_known = (known_midi is not None) and (not force_pyin)
            for ci in range(n_chunks):
                if chunk_sec:
                    s, e   = ci * chunk_sec * SR, min((ci + 1) * chunk_sec * SR, audio.shape[1])
                    chunk  = audio[:, s:e]
                    label  = f'{os.path.basename(path)} chunk {ci+1}/{n_chunks}'
                    out_name = f'{name}_chunk{ci:03d}.npz'
                else:
                    chunk, label, out_name = audio, os.path.basename(path), name + '.npz'

                mode_tag = 'known-f0' if use_known else 'pyin'
                print(f'  {label} [{mode_tag}] ... ', end='', flush=True)
                t1    = time.time()
                if use_known:
                    feats = extract_features_known_f0(chunk, known_midi)
                else:
                    feats = extract_features(chunk)
                T_fr  = len(feats['f0'])
                if vel_constant is not None:
                    vel_frames = np.full(T_fr, vel_constant, dtype=np.float32)
                else:
                    lo_p5, lo_p95 = vel_dynamic
                    lo_m  = (feats['loudness_L'] + feats['loudness_R']) * 0.5
                    vel_frames = np.clip(7.0 * (lo_m - lo_p5) / max(lo_p95 - lo_p5, 1.0), 0.0, 7.0).astype(np.float32)
                np.savez_compressed(os.path.join(extracts_dir, out_name),
                                    audio=chunk, vel_frames=vel_frames, **feats)
                print(f'{time.time()-t1:.1f}s')
            print(f'  ok {os.path.basename(path)}  {n_chunks} chunk(s)  {dur_s:.1f}s  [{time.time()-t0:.1f}s total]')
            n_new += 1
        except Exception as exc:
            print(f'  ERROR {os.path.basename(path)}: {exc}')
            import traceback; traceback.print_exc()
    print(f'[extract] {n_new} new file(s) cached.')
    return n_new


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SourceDataset(Dataset):
    def __init__(self, extracts_dir: str, min_voiced: float = 0.1):
        self.crop  = CROP_FRAMES
        self.hop   = FRAME_HOP
        self.items = []
        self.data  = []
        self.midi_per_file = []   # midi note per NPZ file (for coupled training)
        npz_files  = sorted(glob.glob(os.path.join(extracts_dir, '*.npz')))
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
            self.midi_per_file.append(parsed[0] if parsed else 60)
            T   = len(d['f0'])
            if T < self.crop + 1:
                continue
            step = max(1, self.crop // 2)
            for start in range(0, T - self.crop, step):
                if d['voiced_prob'][start:start + self.crop].mean() >= min_voiced:
                    self.items.append((fi, start))
            total_frames += T
        if not self.items:
            print('[dataset] WARNING: no voiced windows found -- using all windows.')
            for fi, d in enumerate(self.data):
                T = len(d['f0'])
                for start in range(0, T - self.crop, self.crop // 4):
                    self.items.append((fi, start))
        dur = total_frames / (SR / FRAME_HOP)
        print(f'[dataset] {len(npz_files)} file(s)  {dur:.0f}s  {len(self.items)} windows')

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fi, start = self.items[idx]
        d  = self.data[fi]
        sl = slice(start, start + self.crop)
        f0  = d['f0'][sl].astype(np.float32)
        loL = d['loudness_L'][sl].astype(np.float32)
        loR = d['loudness_R'][sl].astype(np.float32)
        vp  = d['voiced_prob'][sl].astype(np.float32)
        s_s = start * self.hop
        audio = d['audio'][:, s_s:s_s + self.crop * self.hop].astype(np.float32)
        if random.random() < 0.5:
            db  = random.uniform(-2.0, 2.0)
            g   = 10.0 ** (db / 20.0)
            audio = audio * g; loL += db; loR += db
        vel_mean = float(d['vel_frames'][sl].mean())
        midi     = self.midi_per_file[fi]
        return (torch.from_numpy(f0), torch.from_numpy(loL), torch.from_numpy(loR),
                torch.from_numpy(vp), torch.from_numpy(audio),
                torch.tensor(vel_mean, dtype=torch.float32),
                torch.tensor(midi,     dtype=torch.long),
                torch.tensor(start,    dtype=torch.long))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def mrstft_loss(pred, target, fft_sizes=(256, 1024, 4096)):
    B, C, T = pred.shape
    p_flat  = pred.reshape(B * C, T)
    t_flat  = target.reshape(B * C, T)
    total, n_valid = pred.sum() * 0.0, 0
    for n_fft in fft_sizes:
        if n_fft >= T: continue
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=pred.device)
        Sp  = torch.stft(p_flat, n_fft, hop, n_fft, win, return_complex=True)
        St  = torch.stft(t_flat, n_fft, hop, n_fft, win, return_complex=True)
        mp  = (Sp.real.pow(2) + Sp.imag.pow(2) + 1e-8).sqrt()
        mt  = (St.real.pow(2) + St.imag.pow(2) + 1e-8).sqrt()
        sc  = ((mp - mt).pow(2).sum() + 1e-8).sqrt() / (mt.pow(2).sum() + 1e-8).sqrt()
        lm  = F.l1_loss(torch.log(mp + 1e-7), torch.log(mt + 1e-7))
        total += sc + lm; n_valid += 1
    return total / max(n_valid, 1)


# ---------------------------------------------------------------------------
# Generation utilities
# ---------------------------------------------------------------------------

MIDI_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
_NOTE_MAP  = {
    'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,
    'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11,
}

def midi_to_name(m): return f'{MIDI_NAMES[m % 12]}{m // 12 - 1}'

def parse_note_name(name: str):
    name = name.strip()
    for length in (2, 1):
        np_, op_ = name[:length], name[length:]
        if np_ in _NOTE_MAP and op_.lstrip('-').isdigit():
            midi = (int(op_) + 1) * 12 + _NOTE_MAP[np_]
            if 0 <= midi <= 127: return midi
    return None

def extract_loudness_from_audio(audio: np.ndarray, n_frames: int) -> np.ndarray:
    mono = audio.mean(axis=0)
    lo   = np.full(n_frames, -80.0, dtype=np.float32)
    for i in range(n_frames):
        s, e = i * FRAME_HOP, min((i + 1) * FRAME_HOP, len(mono))
        if s < len(mono):
            lo[i] = 20.0 * np.log10(float(np.sqrt(np.mean(mono[s:e] ** 2) + 1e-10)) + 1e-10)
    return lo

def load_envelope_templates(extracts_dir: str) -> dict:
    """Load loudness envelopes from NPZ cache.

    Returns dict: (midi, vel) -> np.ndarray of loudness_dB per frame.
    Uses the average of L and R channels.  For chunk files keeps the
    longest chunk (captures full decay better than shortest).
    """
    import glob as _glob
    templates: dict = {}
    for path in sorted(_glob.glob(os.path.join(extracts_dir, '*.npz'))):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        midi, vel = parsed
        d  = np.load(path)
        lo = ((d['loudness_L'] + d['loudness_R']) * 0.5).astype(np.float32)
        key = (midi, vel)
        if key not in templates or len(lo) > len(templates[key]):
            templates[key] = lo
    return templates


def find_envelope(templates: dict, midi: int, vel: int) -> np.ndarray:
    """Return the envelope template closest to (midi, vel).

    Search order:
      1. Exact (midi, vel) match
      2. Nearest midi at the same vel layer
      3. Any entry — nearest by midi + vel distance
    """
    if (midi, vel) in templates:
        return templates[(midi, vel)]
    # Same vel, nearest midi
    same_vel = [(abs(m - midi), m) for m, v in templates if v == vel]
    if same_vel:
        _, best_midi = min(same_vel)
        return templates[(best_midi, vel)]
    # Fallback: any entry
    best = min(templates, key=lambda k: abs(k[0] - midi) + abs(k[1] - vel) * 3)
    return templates[best]


@torch.no_grad()
def synthesize(model: DDSPVocoder, midi: int, velocity: float,
               loudness: np.ndarray, device: torch.device,
               inh_scale: float = 1.0) -> np.ndarray:
    """Synthesize one note -> (2, T) float32"""
    n_frames = len(loudness)
    freq     = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    f0_t  = torch.from_numpy(np.full(n_frames, freq, np.float32)).unsqueeze(0).to(device)
    lo_t  = torch.from_numpy(loudness).unsqueeze(0).to(device)
    vel_t = torch.tensor([float(velocity)], dtype=torch.float32, device=device)
    out   = model(f0_t, lo_t, vel_t, n_frames, inh_scale=inh_scale)
    return out[0].cpu().numpy()   # (2, T)


def apply_attack_ramp(audio: np.ndarray, ramp_ms: float, sr: int = SR) -> np.ndarray:
    """Apply a short amplitude ramp-up to the start of audio (2, T) or (T,).

    The ramp is a raised-cosine (half-Hann) shape over the first ramp_ms ms,
    ensuring a crisp, natural onset transient.  ramp_ms=0 is a no-op.
    """
    if ramp_ms <= 0:
        return audio
    n_ramp = min(int(ramp_ms * sr / 1000), audio.shape[-1])
    ramp   = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_ramp) / n_ramp)).astype(np.float32)
    if audio.ndim == 2:
        audio = audio.copy()
        audio[:, :n_ramp] *= ramp
    else:
        audio = audio.copy()
        audio[:n_ramp] *= ramp
    return audio


# ---------------------------------------------------------------------------
# Command: extract
# ---------------------------------------------------------------------------

def cmd_extract(args):
    ws = make_workspace(args.instrument, getattr(args, "workspace", None))
    ws.makedirs()
    print(f'[ddsp extract]  source  -> {ws.source_dir}')
    print(f'                cache   -> {ws.extracts_dir}')
    force_pyin = getattr(args, 'force_pyin', False)
    n = extract_and_cache(ws.source_dir, ws.extracts_dir, chunk_sec=args.chunk_sec,
                          force_pyin=force_pyin)
    cfg = load_config(ws)
    cfg.update({'instrument': ws.name, 'source_dir': ws.source_dir,
                'extract': {'completed': True, 'n_new': n, 'completed_at': _now(),
                            'chunk_sec': args.chunk_sec}})
    save_config(ws, cfg)


# ---------------------------------------------------------------------------
# Command: learn
# ---------------------------------------------------------------------------

def cmd_learn(args):
    ws = make_workspace(args.instrument, getattr(args, "workspace", None))
    ws.makedirs()

    cfg = load_config(ws)
    model_size = args.model or cfg.get('model_size', 'small')
    if model_size not in MODEL_SIZES:
        print(f'[ddsp learn] ERROR: unknown model size "{model_size}". '
              f'Choose from: {list(MODEL_SIZES)}')
        sys.exit(1)

    # Auto-extract if no NPZ found
    npz_files = glob.glob(os.path.join(ws.extracts_dir, '*.npz'))
    if not npz_files:
        print('[ddsp learn] No extracts found -- running extraction first.')
        extract_and_cache(ws.source_dir, ws.extracts_dir, chunk_sec=60)

    device = resolve_device(getattr(args, 'device', 'auto'))
    print(f'[ddsp learn]  instrument={ws.name}  model={model_size}  device={device}')
    print(f'              source  -> {ws.source_dir}')
    print(f'              work    -> {ws.work_dir}')

    # --- coupled mode: train EnvelopeNet first, then use its predictions during DDSP training ---
    coupled   = getattr(args, 'coupled', False)
    env_mix   = getattr(args, 'env_mix', 0.5)   # fraction of batches using EnvelopeNet loudness
    env_model = None
    if coupled:
        env_pt = os.path.join(ws.checkpoints_dir, 'envelope.pt')
        if not os.path.exists(env_pt):
            print('[ddsp learn]  Coupled mode: EnvelopeNet not found — training first...')
            os.makedirs(ws.checkpoints_dir, exist_ok=True)
            train_envelope_model(ws.extracts_dir, ws.checkpoints_dir, device=device,
                                 epochs=1000, lr=1e-3,
                                 warp=ENVELOPE_WARP, n_env=N_ENV, attack_weight=5.0)
        env_model = load_envelope_model(env_pt, device)
        print(f'[ddsp learn]  Coupled mode: env_mix={env_mix:.0%}  '
              f'(n_env={env_model.n_env}  warp={env_model._warp})')

    dataset = SourceDataset(ws.extracts_dir, min_voiced=args.min_voiced)
    n_val   = max(1, int(len(dataset) * 0.08))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model    = build_ddsp(size=model_size).to(device)
    n_params = count_params(model)
    print(f'[ddsp learn]  params={n_params:,}  n_train={n_train}  n_val={n_val}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    start_epoch, best_val = 0, float('inf')
    last_pt = os.path.join(ws.checkpoints_dir, 'last.pt')
    best_pt = os.path.join(ws.checkpoints_dir, 'best.pt')

    if args.resume and os.path.exists(last_pt):
        ckpt = torch.load(last_pt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f'[ddsp learn]  resumed from epoch {start_epoch}  best_val={best_val:.4f}')

    # Training log
    log_f = open(ws.log_path, 'a', buffering=1)
    log_f.write(f'\n--- {_now()}  model={model_size}  epochs={args.epochs}  lr={args.lr} ---\n')

    def _get_loudness(loL, loR, vel_b, midi_b, start_b):
        """Return loudness tensor for a batch.
        Coupled mode: with probability env_mix, replace real NPZ loudness with
        EnvelopeNet prediction (sliced to the same crop window).
        """
        lo_real = (loL + loR) * 0.5   # (B, CROP_FRAMES) from NPZ
        if env_model is None or random.random() >= env_mix:
            return lo_real.to(device)
        lo_list = []
        for b in range(lo_real.shape[0]):
            midi_i  = int(midi_b[b].item())
            vel_i   = round(float(vel_b[b].item()))
            st      = int(start_b[b].item())
            env_arr = env_model.predict_envelope(midi_i, vel_i, warp=env_model._warp)
            end     = st + CROP_FRAMES
            if len(env_arr) >= end:
                chunk = env_arr[st:end]
            else:
                chunk = np.pad(env_arr, (0, max(0, end - len(env_arr))),
                               constant_values=-80.0)[st:end]
            lo_list.append(torch.from_numpy(chunk))
        return torch.stack(lo_list).to(device)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        t0, train_losses = time.time(), []
        for f0, loL, loR, vp, audio, vel, midi_b, start_b in train_loader:
            f0    = f0.to(device)
            audio = audio.to(device)
            vel   = vel.to(device)
            lo    = _get_loudness(loL, loR, vel, midi_b, start_b)
            pred  = model(f0, lo, vel, CROP_FRAMES)
            loss  = mrstft_loss(pred, audio) + 0.2 * F.l1_loss(pred, audio)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for f0, loL, loR, vp, audio, vel, midi_b, start_b in val_loader:
                f0    = f0.to(device)
                audio = audio.to(device)
                vel   = vel.to(device)
                lo    = _get_loudness(loL, loR, vel, midi_b, start_b)
                pred  = model(f0, lo, vel, CROP_FRAMES)
                val_losses.append((mrstft_loss(pred, audio) + 0.2 * F.l1_loss(pred, audio)).item())

        tl, vl  = float(np.mean(train_losses)), float(np.mean(val_losses))
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        best_tag = ''
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_pt)
            best_tag = '  <- best'

        line = (f'ep {epoch:4d}  train={tl:.4f}  val={vl:.4f}  '
                f'lr={lr_now:.2e}  {elapsed:.1f}s{best_tag}')
        print(f'  {line}')
        log_f.write(line + '\n')

        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'best_val': best_val, 'model_size': model_size}, last_pt)

    log_f.write(f'[done]  best_val={best_val:.4f}\n')
    log_f.close()
    print(f'\n[ddsp learn]  done.  best_val={best_val:.4f}  checkpoint->{best_pt}')

    # --- train envelope model (fast, ~seconds) ---
    print('\n[ddsp learn]  training EnvelopeNet ...')
    try:
        train_envelope_model(ws.extracts_dir, ws.checkpoints_dir, device=device,
                             epochs=1000, lr=1e-3,
                             warp=ENVELOPE_WARP, n_env=N_ENV, attack_weight=5.0)
    except Exception as exc:
        print(f'[ddsp learn]  WARNING: EnvelopeNet training failed: {exc}')

    cfg.update({'instrument': ws.name, 'source_dir': ws.source_dir,
                'model_size': model_size, 'sr': SR,
                'training': {
                    'epochs_completed': start_epoch + args.epochs,
                    'best_val': best_val,
                    'last_trained': _now(),
                }})
    save_config(ws, cfg)


# ---------------------------------------------------------------------------
# Command: learn-envelope
# ---------------------------------------------------------------------------

def train_envelope_model(extracts_dir: str, checkpoint_dir: str,
                         epochs: int = 1000, lr: float = 1e-3,
                         warp: float = ENVELOPE_WARP,
                         n_env: int = N_ENV,
                         attack_weight: float = 5.0,
                         device: 'torch.device' = None) -> 'EnvelopeNet':
    """Train EnvelopeNet on NPZ loudness curves.  Returns trained model.

    Uses 2 ms loudness resolution from stored audio (hop_fine=96 @ 48 kHz) so
    the attack region is sampled at fine resolution even though the DDSP vocoder
    runs at 5 ms frames.  The target is resampled onto a warped time axis so
    that early control points (attack) get proportionally more coverage.

    attack_weight: MSE multiplier for the first N_ATK warped points (attack region).
    """
    import glob as _glob

    HOP_FINE  = 96                # 2 ms @ 48 kHz — fine resolution for attack
    N_ATK     = max(1, n_env // 25)  # ~4 % of points weighted extra (≈ first 300 ms)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Warped time axis: t_store[i] = (i/(n_env-1))^warp
    t_store = np.power(np.linspace(0.0, 1.0, n_env), warp)

    # --- gather training samples ---
    samples = []
    for path in sorted(_glob.glob(os.path.join(extracts_dir, '*.npz'))):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        midi, vel = parsed
        d = np.load(path)

        # Compute 2 ms loudness from stored audio for fine attack resolution
        audio_np = d['audio']                             # (2, N_samples)
        lo_fine  = loudness_db(audio_np, hop=HOP_FINE)   # ~2 ms / frame

        dur_s = len(lo_fine) * HOP_FINE / SR
        # Sample loudness at warped positions (finer near t=0)
        xs    = np.linspace(0.0, 1.0, len(lo_fine))
        shape = np.interp(t_store, xs, lo_fine).astype(np.float32)
        samples.append((midi / 127.0, vel / 7.0, dur_s, shape))

    if not samples:
        raise RuntimeError(f'No NPZ files with parseable filenames in {extracts_dir}')
    print(f'  EnvelopeNet: {len(samples)} samples  n_env={n_env}  warp={warp}'
          f'  attack_weight={attack_weight}  device={device}')

    # --- tensors ---
    midi_t  = torch.tensor([s[0] for s in samples], dtype=torch.float32, device=device)
    vel_t   = torch.tensor([s[1] for s in samples], dtype=torch.float32, device=device)
    dur_t   = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=device)
    shape_t = torch.tensor(np.stack([s[3] for s in samples]), dtype=torch.float32, device=device)

    # Loss weight vector: attack_weight for first N_ATK points, 1.0 for the rest
    loss_w         = torch.ones(n_env, dtype=torch.float32, device=device)
    loss_w[:N_ATK] = attack_weight

    model_env = EnvelopeNet(n_env=n_env).to(device)
    opt       = torch.optim.Adam(model_env.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    for epoch in range(1, epochs + 1):
        pred_dur, pred_shape = model_env(midi_t, vel_t)
        # Weighted MSE: attack region contributes attack_weight× more to shape loss
        err_shape  = (pred_shape - shape_t) ** 2          # (B, n_env)
        loss_shape = (err_shape * loss_w).mean()
        loss_dur   = F.mse_loss(torch.log(pred_dur), torch.log(dur_t.clamp(min=0.5)))
        loss       = loss_shape + loss_dur
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        if epoch % 100 == 0 or epoch == epochs:
            print(f'    ep {epoch:4d}  loss={loss.item():.5f}'
                  f'  (shape={loss_shape.item():.5f}  dur={loss_dur.item():.5f})')

    env_pt = os.path.join(checkpoint_dir, 'envelope.pt')
    # Save model together with its hyperparameters so predict_envelope uses correct warp
    torch.save({'state_dict': model_env.state_dict(),
                'n_env': n_env, 'warp': warp}, env_pt)
    print(f'  EnvelopeNet saved -> {env_pt}')
    return model_env


def load_envelope_model(env_pt: str, device: 'torch.device') -> 'EnvelopeNet':
    """Load EnvelopeNet from checkpoint (dict with state_dict + hyperparams)."""
    ckpt      = torch.load(env_pt, map_location=device, weights_only=True)
    n_env     = ckpt['n_env']
    warp      = ckpt['warp']
    model_env = EnvelopeNet(n_env=n_env).to(device)
    model_env.load_state_dict(ckpt['state_dict'])
    model_env._warp = warp
    model_env.eval()
    return model_env


def cmd_learn_envelope(args):
    ws = make_workspace(args.instrument, getattr(args, 'workspace', None))
    if not os.path.isdir(ws.extracts_dir):
        print(f'[learn-envelope] ERROR: no extracts dir at {ws.extracts_dir}')
        print('[learn-envelope] Run: python ddsp.py extract --instrument <path>')
        sys.exit(1)
    os.makedirs(ws.checkpoints_dir, exist_ok=True)
    device = resolve_device(getattr(args, 'device', 'auto'))
    print(f'[learn-envelope]  instrument={ws.name}  device={device}')
    train_envelope_model(ws.extracts_dir, ws.checkpoints_dir,
                         epochs=args.epochs, lr=args.lr,
                         warp=args.envelope_warp,
                         n_env=args.n_env,
                         attack_weight=args.attack_weight,
                         device=device)
    print(f'[learn-envelope]  done.')


# ---------------------------------------------------------------------------
# Command: generate
# ---------------------------------------------------------------------------

def cmd_generate(args):
    ws = make_workspace(args.instrument, getattr(args, "workspace", None))

    cfg       = load_config(ws)
    model_size = cfg.get('model_size', 'small')
    best_pt   = os.path.join(ws.checkpoints_dir, 'best.pt')
    if not os.path.exists(best_pt):
        print(f'[ddsp generate] ERROR: no checkpoint at {best_pt}')
        print('[ddsp generate] Run: python ddsp.py learn --instrument <path>')
        sys.exit(1)

    device = resolve_device(getattr(args, 'device', 'auto'))
    model  = build_ddsp(size=model_size).to(device)
    model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))
    model.eval()
    print(f'[ddsp generate]  instrument={ws.name}  model={model_size}  device={device}')
    print(f'                 checkpoint -> {best_pt}')

    # Load envelope model if available
    env_pt    = os.path.join(ws.checkpoints_dir, 'envelope.pt')
    env_model = None
    env_src   = getattr(args, 'envelope_source', 'auto')   # auto | envelopenet | npz
    if env_src != 'npz' and os.path.exists(env_pt):
        env_model = load_envelope_model(env_pt, device)
        print(f'                 envelope   -> {env_pt}  (n_env={env_model.n_env}  warp={env_model._warp})')
    elif env_src == 'envelopenet' and not os.path.exists(env_pt):
        print(f'[ddsp generate] ERROR: --envelope-source envelopenet but no envelope.pt found')
        sys.exit(1)

    attack_ramp_ms = getattr(args, 'attack_ramp_ms', 10)
    inh_scale      = float(getattr(args, 'inh_scale', 1.0))

    # Source list: NPZ cache — pure model generation, no WAV reference needed
    all_npz  = sorted(glob.glob(os.path.join(ws.extracts_dir, '*.npz')))
    chunk0   = [f for f in all_npz if '_chunk000' in os.path.basename(f)]
    src_list = chunk0 if chunk0 else all_npz
    if not src_list:
        print(f'[ddsp generate] ERROR: zadne NPZ soubory v {ws.extracts_dir} — spust extract')
        sys.exit(1)

    # Filters
    note_set, vel_set = None, None
    if args.notes:
        note_set = set()
        for n in args.notes:
            m = parse_note_name(n)
            if m is None:
                print(f'[ddsp generate] ERROR: unknown note "{n}"'); sys.exit(1)
            note_set.add(m)
        available = {(parse_filename(f) or [60])[0] for f in src_list}
        missing   = [n for n, m in zip(args.notes, [parse_note_name(x) for x in args.notes])
                     if m not in available]
        if missing:
            print(f'[ddsp generate] WARNING: tyto noty nejsou v datasetu: {" ".join(missing)}'
                  f'  (pouzij full-range mod pro syntezu libovolne noty)')
    if args.vel:
        vel_set = set(int(v) for v in args.vel)
    if note_set:
        src_list = [f for f in src_list if (parse_filename(f) or [60])[0] in note_set]
    if vel_set:
        src_list = [f for f in src_list if (parse_filename(f) or [0, None])[1] in vel_set]
    if (note_set or vel_set) and not src_list:
        print('[ddsp generate] ERROR: zadne soubory neodpovidaji filtru notes/vel')
        sys.exit(1)

    ithaca_out = os.path.join(ITHACA_PLAYER_ROOT, ws.name)
    output_dir = args.output or ithaca_out
    output_dir = output_dir if os.path.isabs(output_dir) else os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    skip = not args.no_skip

    # ------------------------------------------------------------------ #
    # Full-range mode: generate complete chromatic bank from model alone  #
    # ------------------------------------------------------------------ #
    if getattr(args, 'full_range', False):
        midi_lo   = args.midi_lo
        midi_hi   = args.midi_hi
        n_vel     = args.vel_layers
        sr_kHz    = SR // 1000
        midi_list = list(range(midi_lo, midi_hi + 1))
        total     = len(midi_list) * n_vel

        if env_model is not None:
            env_source = f'EnvelopeNet (n_env={env_model.n_env}  warp={env_model._warp})'
            templates  = None
        else:
            env_source = 'NPZ cache (nearest-neighbour)'
            templates  = load_envelope_templates(ws.extracts_dir)
            if not templates:
                print('[ddsp generate] ERROR: no NPZ extracts and no envelope.pt'
                      ' — run extract + learn first')
                sys.exit(1)

        print(f'                 attack_ramp={attack_ramp_ms} ms')

        print(f'[ddsp generate]  FULL RANGE  {midi_to_name(midi_lo)}-{midi_to_name(midi_hi)}'
              f'  ({len(midi_list)} notes x {n_vel} vel = {total} samples)')
        print(f'                 envelopes: {env_source}  output->{output_dir}\n')

        done = 0
        for midi in midi_list:
            freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            for vel in range(n_vel):
                out_name = f'm{midi:03d}-vel{vel}-f{sr_kHz}.wav'
                out_path = os.path.join(output_dir, out_name)
                if skip and os.path.exists(out_path):
                    done += 1; continue
                try:
                    if env_model is not None:
                        loudness = env_model.predict_envelope(midi, vel, warp=env_model._warp)
                    else:
                        loudness = find_envelope(templates, midi, vel)
                    audio    = synthesize(model, midi, float(vel), loudness, device, inh_scale=inh_scale)
                    audio    = apply_attack_ramp(audio, attack_ramp_ms)
                    dur_s    = len(loudness) * FRAME_HOP / SR
                    save_wav(out_path, audio, SR)
                    done += 1
                    print(f'  {midi_to_name(midi):4s} m{midi:03d} vel{vel}  {freq:.1f}Hz  {dur_s:.1f}s  -> {out_name}')
                except Exception as exc:
                    print(f'  ERROR m{midi:03d} vel{vel}: {exc}')

        # Copy instrument-definition.json if present in source
        idef_src = os.path.join(ws.source_dir, 'instrument-definition.json')
        if os.path.exists(idef_src):
            idef_dst = os.path.join(output_dir, 'instrument-definition.json')
            import json as _json
            with open(idef_src, encoding='utf-8') as f:
                idef = _json.load(f)
            idef['velocityMaps'] = str(n_vel)
            idef['sampleCount']  = len(midi_list)
            with open(idef_dst, 'w', encoding='utf-8') as f:
                _json.dump(idef, f, indent=4, ensure_ascii=False)

        print(f'\n[ddsp generate]  done: {done}/{total}  -> {output_dir}')
        cfg.update({'generated': {'n_files': done, 'mode': 'full_range',
                                   'midi_range': f'{midi_lo}-{midi_hi}',
                                   'vel_layers': n_vel, 'output_dir': output_dir,
                                   'generated_at': _now()}})
        save_config(ws, cfg)
        return

    # ------------------------------------------------------------------ #
    # Standard mode: pure model generation — one output per NPZ source  #
    # Envelope: EnvelopeNet (preferred) or NPZ loudness_L (fallback)    #
    # ------------------------------------------------------------------ #
    sr_kHz = SR // 1000
    total  = len(src_list)
    env_src_label = (f'EnvelopeNet (n_env={env_model.n_env}  warp={env_model._warp})'
                     if env_model is not None else 'NPZ loudness_L')
    print(f'[ddsp generate]  {total} sample(s)  envelopes={env_src_label}  output->{output_dir}\n')

    done = 0
    for idx, src_path in enumerate(src_list):
        parsed    = parse_filename(src_path)
        midi      = parsed[0] if parsed else 60
        vel_layer = parsed[1] if parsed else 0
        out_name  = f'm{midi:03d}-vel{vel_layer}-f{sr_kHz}.wav'
        out_path  = os.path.join(output_dir, out_name)
        if skip and os.path.exists(out_path):
            print(f'  [skip] {out_name}')
            continue
        try:
            if env_model is not None:
                loudness = env_model.predict_envelope(midi, vel_layer, warp=env_model._warp)
            else:
                loudness = np.load(src_path)['loudness_L']
            T     = len(loudness) * FRAME_HOP
            audio = synthesize(model, midi, vel_layer, loudness, device, inh_scale=inh_scale)
            audio = apply_attack_ramp(audio, attack_ramp_ms)
            save_wav(out_path, audio, SR)
            done += 1
            note_name = midi_to_name(midi)
            freq      = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            print(f'  [{idx+1:4d}/{total}] {note_name:4s} m{midi:03d} vel{vel_layer}  '
                  f'{freq:.1f}Hz  {T/SR:.2f}s  -> {out_name}')
        except Exception as exc:
            print(f'  ERROR {out_name}: {exc}')
    print(f'\n[ddsp generate]  done: {done}/{total}')

    cfg.update({'generated': {'n_files': done,
                               'output_dir': output_dir, 'generated_at': _now()}})
    save_config(ws, cfg)


# ---------------------------------------------------------------------------
# Command: status
# ---------------------------------------------------------------------------

def cmd_status(args):
    ws  = make_workspace(args.instrument, getattr(args, 'workspace', None))
    cfg = load_config(ws)

    def tick(cond): return 'ok' if cond else '--'

    wav_count = len(scan_instrument_dir(ws.source_dir))
    npz_count = len(glob.glob(os.path.join(ws.extracts_dir, '*.npz')))
    has_ckpt  = os.path.exists(os.path.join(ws.checkpoints_dir, 'best.pt'))
    gen_count = len(glob.glob(os.path.join(ws.generated_dir, '*.wav')))

    extract_info = cfg.get('extract', {})
    train_info   = cfg.get('training', {})
    gen_info     = cfg.get('generated', {})

    print(f'\nDDSP Instrument: {ws.name}')
    print(f'  Source      {ws.source_dir}  ({wav_count} WAV files)')
    print(f'  Workspace   {ws.work_dir}')
    print()
    print(f'  [{tick(npz_count)}] Extract    {npz_count} NPZ'
          + (f'  [{extract_info.get("completed_at","")}]' if extract_info else ''))
    model_size = cfg.get('model_size', '?')
    print(f'  [{tick(has_ckpt)}] Model      {model_size}'
          + (f'  epochs={train_info.get("epochs_completed","?")}  '
             f'best_val={train_info.get("best_val","?")}  '
             f'[{train_info.get("last_trained","")}]' if train_info else '  (not trained)'))
    print(f'  [{tick(gen_count)}] Generated  {gen_count} WAV'
          + (f'  wet={gen_info.get("wet","?")}  [{gen_info.get("generated_at","")}]' if gen_info else ''))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog='ddsp',
        description='DDSP Neural Vocoder -- learn instrument timbre and generate sample banks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subs = parser.add_subparsers(dest='command', required=True)

    _ws_help = 'Workspace directory for extracts/checkpoints (default: <instrument>-ddsp/)'

    # --- extract ---
    p_ext = subs.add_parser('extract', help='Extract & cache audio features')
    p_ext.add_argument('--instrument', required=True, metavar='DIR',
                       help='Source instrument directory (READ-ONLY)')
    p_ext.add_argument('--workspace', default=None, metavar='DIR', help=_ws_help)
    p_ext.add_argument('--chunk-sec', type=int, default=60, metavar='SEC',
                       dest='chunk_sec',
                       help='Split long files into chunks of SEC seconds (0=off, default: 60)')
    p_ext.add_argument('--force-pyin', action='store_true', dest='force_pyin',
                       help='Force slow pyin F0 estimation even when MIDI is known from filename')

    # --- learn ---
    p_lrn = subs.add_parser('learn', help='Train model (runs extract if needed)')
    p_lrn.add_argument('--instrument', required=True, metavar='DIR')
    p_lrn.add_argument('--workspace', default=None, metavar='DIR', help=_ws_help)
    p_lrn.add_argument('--model', choices=list(MODEL_SIZES), default=None,
                       help='Model size: small (~115K), medium (~452K), large (~1.99M)')
    p_lrn.add_argument('--epochs',     type=int,   default=100)
    p_lrn.add_argument('--lr',         type=float, default=3e-4)
    p_lrn.add_argument('--resume',     action='store_true',
                       help='Continue from last checkpoint')
    p_lrn.add_argument('--batch-size', type=int,   default=4, dest='batch_size')
    p_lrn.add_argument('--min-voiced', type=float, default=0.1, dest='min_voiced',
                       help='Minimum voiced frame fraction per window (default: 0.1)')
    p_lrn.add_argument('--coupled', action='store_true',
                       help='Coupled mode: train EnvelopeNet first, then use its loudness '
                            'predictions during DDSP training (aligns train/inference distributions)')
    p_lrn.add_argument('--env-mix', type=float, default=0.5, dest='env_mix',
                       help='Fraction of training batches using EnvelopeNet loudness in '
                            'coupled mode (default: 0.5)')
    p_lrn.add_argument('--device', default='auto',
                       choices=['auto', 'cpu', 'mps', 'cuda'],
                       help='Compute device: auto (default), cpu, mps (Apple Silicon), cuda')

    # --- generate ---
    p_gen = subs.add_parser('generate', help='Generate sample bank from trained model')
    p_gen.add_argument('--instrument', required=True, metavar='DIR')
    p_gen.add_argument('--workspace', default=None, metavar='DIR', help=_ws_help)
    p_gen.add_argument('--wet',     type=float, default=1.0, metavar='0-1',
                       help='Wet/dry blend: 1.0=full DDSP, 0.0=original (default: 1.0)')
    p_gen.add_argument('--inharmonicity-scale', type=float, default=1.0, metavar='0-2',
                       dest='inh_scale',
                       help='Inharmonicity multiplier: 0=pure harmonic, 1=learned B, 2=exaggerated (default: 1.0)')
    p_gen.add_argument('--output',  default=None, metavar='DIR',
                       help=f'Output directory (default: {ITHACA_PLAYER_ROOT}\\<instrument>)')
    p_gen.add_argument('--notes',   nargs='+', default=None, metavar='NOTE',
                       help='Generate only these notes, e.g. --notes C4 A3 G3')
    p_gen.add_argument('--vel',     nargs='+', type=int, default=None, metavar='VEL',
                       help='Velocity filter, e.g. --vel 5 7')
    p_gen.add_argument('--no-skip', action='store_true', dest='no_skip',
                       help='Overwrite existing output files')
    # Full-range mode
    p_gen.add_argument('--full-range', action='store_true', dest='full_range',
                       help='Generate complete chromatic bank without source WAVs')
    p_gen.add_argument('--midi-lo', type=int, default=21, dest='midi_lo', metavar='MIDI',
                       help='Lowest MIDI note for --full-range (default: 21 = A0)')
    p_gen.add_argument('--midi-hi', type=int, default=108, dest='midi_hi', metavar='MIDI',
                       help='Highest MIDI note for --full-range (default: 108 = C8)')
    p_gen.add_argument('--vel-layers', type=int, default=8, dest='vel_layers', metavar='N',
                       help='Number of velocity layers for --full-range (default: 8)')
    p_gen.add_argument('--envelope-source', choices=['auto', 'envelopenet', 'npz'],
                       default='auto', dest='envelope_source',
                       help='Envelope source: auto (EnvelopeNet if available), '
                            'envelopenet (force NN), npz (force nearest-neighbour template). '
                            'Default: auto')
    p_gen.add_argument('--attack-ramp-ms', type=float, default=10.0,
                       dest='attack_ramp_ms',
                       help='Hard onset ramp length in ms (0=off, default: 10)')
    p_gen.add_argument('--device', default='auto',
                       choices=['auto', 'cpu', 'mps', 'cuda'],
                       help='Compute device: auto (default), cpu, mps (Apple Silicon), cuda')

    # --- learn-envelope ---
    p_env = subs.add_parser('learn-envelope',
                             help='Train EnvelopeNet on NPZ loudness curves (fast, ~minutes)')
    p_env.add_argument('--instrument', required=True, metavar='DIR')
    p_env.add_argument('--workspace', default=None, metavar='DIR', help=_ws_help)
    p_env.add_argument('--epochs', type=int, default=1000,
                       help='Training epochs (default: 1000)')
    p_env.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    p_env.add_argument('--envelope-warp', type=float, default=ENVELOPE_WARP,
                       dest='envelope_warp',
                       help=f'Power-law time warp for attack resolution (default: {ENVELOPE_WARP})')
    p_env.add_argument('--n-env', type=int, default=N_ENV,
                       dest='n_env',
                       help=f'Number of envelope control points (default: {N_ENV})')
    p_env.add_argument('--attack-weight', type=float, default=5.0,
                       dest='attack_weight',
                       help='MSE weight multiplier for attack region (default: 5.0)')
    p_env.add_argument('--device', default='auto',
                       choices=['auto', 'cpu', 'mps', 'cuda'],
                       help='Compute device: auto (default), cpu, mps (Apple Silicon), cuda')

    # --- status ---
    p_sts = subs.add_parser('status', help='Show instrument training status')
    p_sts.add_argument('--instrument', required=True, metavar='DIR')
    p_sts.add_argument('--workspace', default=None, metavar='DIR', help=_ws_help)

    args = parser.parse_args()

    dispatch = {
        'extract':        cmd_extract,
        'learn':          cmd_learn,
        'learn-envelope': cmd_learn_envelope,
        'generate':       cmd_generate,
        'status':         cmd_status,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
