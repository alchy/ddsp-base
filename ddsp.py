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
        return (torch.from_numpy(f0), torch.from_numpy(loL), torch.from_numpy(loR),
                torch.from_numpy(vp), torch.from_numpy(audio),
                torch.tensor(vel_mean, dtype=torch.float32))


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
               loudness: np.ndarray, device: torch.device) -> np.ndarray:
    """Synthesize one note -> (2, T) float32"""
    n_frames = len(loudness)
    freq     = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    f0_t  = torch.from_numpy(np.full(n_frames, freq, np.float32)).unsqueeze(0).to(device)
    lo_t  = torch.from_numpy(loudness).unsqueeze(0).to(device)
    vel_t = torch.tensor([float(velocity)], dtype=torch.float32, device=device)
    out   = model(f0_t, lo_t, vel_t, n_frames)
    return out[0].cpu().numpy()   # (2, T)


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[ddsp learn]  instrument={ws.name}  model={model_size}  device={device}')
    print(f'              source  -> {ws.source_dir}')
    print(f'              work    -> {ws.work_dir}')

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

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        t0, train_losses = time.time(), []
        for f0, loL, loR, vp, audio, vel in train_loader:
            f0, loL, loR = f0.to(device), loL.to(device), loR.to(device)
            audio, vel   = audio.to(device), vel.to(device)
            pred  = model(f0, (loL + loR) * 0.5, vel, CROP_FRAMES)
            loss  = mrstft_loss(pred, audio) + 0.2 * F.l1_loss(pred, audio)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for f0, loL, loR, vp, audio, vel in val_loader:
                f0, loL, loR = f0.to(device), loL.to(device), loR.to(device)
                audio, vel   = audio.to(device), vel.to(device)
                pred = model(f0, (loL + loR) * 0.5, vel, CROP_FRAMES)
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

    cfg.update({'instrument': ws.name, 'source_dir': ws.source_dir,
                'model_size': model_size, 'sr': SR,
                'training': {
                    'epochs_completed': start_epoch + args.epochs,
                    'best_val': best_val,
                    'last_trained': _now(),
                }})
    save_config(ws, cfg)


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_ddsp(size=model_size).to(device)
    model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))
    model.eval()
    print(f'[ddsp generate]  instrument={ws.name}  model={model_size}  device={device}')
    print(f'                 checkpoint -> {best_pt}')

    wav_files = scan_instrument_dir(ws.source_dir)
    if not wav_files:
        print(f'[ddsp generate] ERROR: no WAV files in {ws.source_dir}')
        sys.exit(1)

    # Filters
    if args.notes:
        note_set = set()
        for n in args.notes:
            m = parse_note_name(n)
            if m is None:
                print(f'[ddsp generate] ERROR: unknown note "{n}"'); sys.exit(1)
            note_set.add(m)
        wav_files = [f for f in wav_files if (parse_filename(f) or [60])[0] in note_set]
    if args.vel:
        vel_set   = set(args.vel)
        wav_files = [f for f in wav_files if (parse_filename(f) or [0, None])[1] in vel_set]

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

        print(f'[ddsp generate]  FULL RANGE  {midi_to_name(midi_lo)}-{midi_to_name(midi_hi)}'
              f'  ({len(midi_list)} notes x {n_vel} vel = {total} samples)')
        print(f'                 envelopes from NPZ cache  output->{output_dir}\n')

        templates = load_envelope_templates(ws.extracts_dir)
        if not templates:
            print('[ddsp generate] ERROR: no NPZ extracts found — run extract first')
            sys.exit(1)
        print(f'  loaded {len(templates)} envelope template(s) from {ws.extracts_dir}\n')

        done = 0
        for midi in midi_list:
            freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            for vel in range(n_vel):
                out_name = f'm{midi:03d}-vel{vel}-f{sr_kHz}.wav'
                out_path = os.path.join(output_dir, out_name)
                if skip and os.path.exists(out_path):
                    done += 1; continue
                try:
                    loudness = find_envelope(templates, midi, vel)
                    audio    = synthesize(model, midi, float(vel), loudness, device)
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
    # Standard mode: one output per source WAV                            #
    # ------------------------------------------------------------------ #
    wet   = float(np.clip(args.wet, 0.0, 1.0))
    total = len(wav_files)
    print(f'[ddsp generate]  {total} sample(s)  wet={wet:.2f}  output->{output_dir}\n')

    done = 0
    for idx, wav_path in enumerate(wav_files):
        out_name = os.path.basename(wav_path)
        out_path = os.path.join(output_dir, out_name)
        if skip and os.path.exists(out_path):
            done += 1; continue
        parsed    = parse_filename(wav_path)
        midi      = parsed[0] if parsed else 60
        vel_layer = parsed[1] if parsed else 0
        try:
            ref      = load_wav_stereo(wav_path, target_sr=SR)
            T        = ref.shape[-1]
            n_frames = math.ceil(T / FRAME_HOP)
            loudness = extract_loudness_from_audio(ref, n_frames)
            audio    = synthesize(model, midi, vel_layer, loudness, device)
            if audio.shape[-1] > T:
                audio = audio[:, :T]
            elif audio.shape[-1] < T:
                audio = np.pad(audio, ((0,0),(0, T - audio.shape[-1])))
            if wet < 1.0:
                audio = wet * audio + (1.0 - wet) * ref
            save_wav(out_path, audio, SR)
            done += 1
            note_name = midi_to_name(midi)
            freq      = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            print(f'  [{idx+1:4d}/{total}] {note_name:4s} m{midi:03d} vel{vel_layer}  '
                  f'{freq:.1f}Hz  {T/SR:.2f}s  -> {out_name}')
        except Exception as exc:
            print(f'  ERROR {out_name}: {exc}')
    print(f'\n[ddsp generate]  done: {done}/{total}')

    cfg.update({'generated': {'n_files': done, 'wet': wet,
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

    # --- generate ---
    p_gen = subs.add_parser('generate', help='Generate sample bank from trained model')
    p_gen.add_argument('--instrument', required=True, metavar='DIR')
    p_gen.add_argument('--workspace', default=None, metavar='DIR', help=_ws_help)
    p_gen.add_argument('--wet',     type=float, default=1.0, metavar='0-1',
                       help='Wet/dry blend: 1.0=full DDSP, 0.0=original (default: 1.0)')
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

    # --- status ---
    p_sts = subs.add_parser('status', help='Show instrument training status')
    p_sts.add_argument('--instrument', required=True, metavar='DIR')
    p_sts.add_argument('--workspace', default=None, metavar='DIR', help=_ws_help)

    args = parser.parse_args()

    dispatch = {
        'extract':  cmd_extract,
        'learn':    cmd_learn,
        'generate': cmd_generate,
        'status':   cmd_status,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
