"""
sfz_convert.py - Convert SFZ instrument bank to mXXX-velX-fXX.wav format

Usage:
    python sfz_convert.py <file.sfz> <output_dir> [options]

Examples:
    python sfz_convert.py Salamander.sfz C:/samples/salamander
    python sfz_convert.py Salamander.sfz C:/samples/salamander --vel-layers 8 --sr 48000
    python sfz_convert.py Salamander.sfz C:/samples/salamander --dry-run

Output:
    <output_dir>/
        m060-vel0-f48.wav
        m060-vel1-f48.wav
        ...
        instrument-definition.json

Filename format:
    mXXX  - MIDI note 0-127 (3 digits, zero-padded)
    velX  - velocity layer 0-(N-1)
    fXX   - sample rate in kHz (e.g. f48 = 48000 Hz)
"""

import os
import sys
import re
import json
import shutil
import argparse
import math
from collections import defaultdict

try:
    import numpy as np
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


# ---------------------------------------------------------------------------
# SFZ parser
# ---------------------------------------------------------------------------

def _parse_sfz(sfz_path: str) -> list[dict]:
    """
    Minimální SFZ parser. Vrátí seznam regionů jako slovníků opcode→hodnota.
    Podporuje <group>, <region>, #include, řádkové komentáře (//).
    """
    sfz_dir = os.path.dirname(os.path.abspath(sfz_path))
    regions = []

    def _load(path, depth=0):
        if depth > 8:
            print(f'  WARN: #include zanoreni > 8, preskakuji {path}')
            return
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                text = f.read()
        except FileNotFoundError:
            print(f'  WARN: soubor nenalezen: {path}')
            return

        # Odstraň komentáře // …
        text = re.sub(r'//[^\n]*', '', text)
        # Odstraň /* … */ blokové komentáře
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

        group_ctx  = {}
        region_ctx = None

        tokens = re.split(r'(<\w+>|#include\s+"[^"]+")', text)

        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue

            # #include
            m = re.match(r'#include\s+"([^"]+)"', tok)
            if m:
                inc_path = os.path.join(sfz_dir, m.group(1).replace('\\', '/'))
                _load(inc_path, depth + 1)
                continue

            # Sekce
            if tok.lower() == '<group>':
                if region_ctx is not None:
                    regions.append({**group_ctx, **region_ctx})
                region_ctx = None
                group_ctx  = {}
                continue
            if tok.lower() == '<region>':
                if region_ctx is not None:
                    regions.append({**group_ctx, **region_ctx})
                region_ctx = {}
                continue
            if tok.startswith('<'):
                # <control>, <curve> apod. — ignoruj
                if region_ctx is not None:
                    regions.append({**group_ctx, **region_ctx})
                region_ctx = None
                continue

            # Opkódy key=value
            for m in re.finditer(r'(\w+)\s*=\s*(\S+)', tok):
                key, val = m.group(1).lower(), m.group(2)
                if region_ctx is not None:
                    region_ctx[key] = val
                else:
                    group_ctx[key] = val

        if region_ctx is not None:
            regions.append({**group_ctx, **region_ctx})

    _load(sfz_path)
    return regions


# ---------------------------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------------------------

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _note_name_to_midi(name: str) -> int | None:
    """'C4' → 60,  'A#3' → 46,  '60' → 60"""
    name = name.strip()
    if name.lstrip('-').isdigit():
        return int(name)
    m = re.match(r'([A-Ga-g][#b]?)(-?\d+)$', name)
    if not m:
        return None
    note_str = m.group(1).upper().replace('B', 'A#') if 'b' in m.group(1) else m.group(1).upper()
    # enharmonics
    enharmonics = {'DB': 'C#', 'EB': 'D#', 'FB': 'E', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#', 'CB': 'B'}
    note_str = enharmonics.get(note_str, note_str)
    if note_str not in NOTE_NAMES:
        return None
    octave = int(m.group(2))
    return NOTE_NAMES.index(note_str) + (octave + 1) * 12


def _midi_to_hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def _parse_int(val, default=None):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Jednoduchý resample přes scipy nebo librosa."""
    if src_sr == dst_sr:
        return audio
    try:
        import scipy.signal as sig
        gcd = math.gcd(src_sr, dst_sr)
        up, down = dst_sr // gcd, src_sr // gcd
        if audio.ndim == 1:
            return sig.resample_poly(audio, up, down).astype(np.float32)
        else:
            ch = [sig.resample_poly(audio[:, c], up, down).astype(np.float32)
                  for c in range(audio.shape[1])]
            return np.stack(ch, axis=1)
    except ImportError:
        pass
    try:
        import librosa
        if audio.ndim == 2:
            audio = audio.T  # librosa: (channels, samples)
        out = librosa.resample(audio.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr)
        if out.ndim == 2:
            out = out.T
        return out
    except ImportError:
        pass
    print('  WARN: scipy ani librosa nejsou dostupne — resample preskocen')
    return audio


# ---------------------------------------------------------------------------
# Hlavní konverze
# ---------------------------------------------------------------------------

def convert(sfz_path: str, out_dir: str,
            vel_layers: int = 8,
            target_sr: int = 48000,
            dry_run: bool = False,
            instrument_name: str = None):

    if not HAS_SF:
        print('Chyba: soundfile neni nainstalovano. Spust: pip install soundfile')
        sys.exit(1)

    print(f'[sfz_convert] Nacitam: {sfz_path}')
    regions = _parse_sfz(sfz_path)
    print(f'[sfz_convert] Nalezeno {len(regions)} regionu v SFZ')

    sfz_dir = os.path.dirname(os.path.abspath(sfz_path))

    # Sestavení struktury: {midi_note: [(lovel, hivel, sample_path), ...]}
    note_map = defaultdict(list)
    skipped  = 0

    for r in regions:
        sample = r.get('sample') or r.get('sample_path', '')
        if not sample:
            skipped += 1
            continue

        # Normalizace cesty
        sample_path = os.path.join(sfz_dir, sample.replace('\\', '/'))

        # MIDI nota
        midi = None
        for key in ('pitch_keycenter', 'key', 'lokey'):
            raw = r.get(key)
            if raw:
                midi = _note_name_to_midi(raw)
                if midi is not None:
                    break
        if midi is None:
            skipped += 1
            continue

        lovel = _parse_int(r.get('lovel', '0'),   0)
        hivel = _parse_int(r.get('hivel', '127'), 127)

        note_map[midi].append((lovel, hivel, sample_path))

    print(f'[sfz_convert] Platnych regionu: {sum(len(v) for v in note_map.values())}  '
          f'preskoceno: {skipped}')
    print(f'[sfz_convert] Rozsah not: MIDI {min(note_map)} – {max(note_map)}  '
          f'({len(note_map)} not)')

    sr_kHz = target_sr // 1000

    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)

    total_written = 0
    total_notes   = len(note_map)

    for midi, layers in sorted(note_map.items()):
        # Seřaď vrstvy podle střední velocity
        layers.sort(key=lambda x: (x[0] + x[1]) / 2)
        n_src = len(layers)

        # Výběr vel_layers rovnoměrně z dostupných
        if n_src <= vel_layers:
            chosen = list(range(n_src))
        else:
            # Rovnoměrný výběr vel_layers indexů z n_src
            chosen = [round(i * (n_src - 1) / (vel_layers - 1))
                      for i in range(vel_layers)]

        for vel_idx, src_idx in enumerate(chosen):
            lovel, hivel, sample_path = layers[src_idx]

            out_name = f'm{midi:03d}-vel{vel_idx}-f{sr_kHz}.wav'
            out_path = os.path.join(out_dir, out_name)

            if dry_run:
                src_exists = os.path.exists(sample_path)
                print(f'  [dry] {out_name}  <-  {os.path.basename(sample_path)}'
                      f'  vel={lovel}-{hivel}'
                      + ('' if src_exists else '  [SOUBOR NENALEZEN]'))
                total_written += 1
                continue

            if not os.path.exists(sample_path):
                print(f'  WARN: {os.path.basename(sample_path)} nenalezen, preskakuji')
                continue

            try:
                audio, src_sr = sf.read(sample_path, dtype='float32', always_2d=True)
                if src_sr != target_sr:
                    audio = _resample_audio(audio, src_sr, target_sr)

                # Stereo: (samples, 2) → zapis
                if audio.shape[1] == 1:
                    audio = np.concatenate([audio, audio], axis=1)

                sf.write(out_path, audio, target_sr, subtype='PCM_24')
                total_written += 1

            except Exception as e:
                print(f'  ERROR {os.path.basename(sample_path)}: {e}')

        if not dry_run and (midi % 10 == 0 or midi == min(note_map)):
            print(f'  nota {midi:3d}  {len(chosen)} vrst.  ok')

    # instrument-definition.json
    iname  = instrument_name or os.path.splitext(os.path.basename(sfz_path))[0]
    idef   = {
        'instrumentName': iname,
        'velocityMaps':   str(vel_layers),
        'instrumentVersion': '1',
        'author': 'sfz_convert',
        'description': f'Konvertovano z {os.path.basename(sfz_path)}',
        'category': 'Acoustic Piano',
        'sampleCount': total_notes,
    }
    idef_path = os.path.join(out_dir, 'instrument-definition.json')
    if not dry_run:
        with open(idef_path, 'w', encoding='utf-8') as f:
            json.dump(idef, f, indent=4, ensure_ascii=False)
        print(f'\n[sfz_convert] Hotovo. Zapsano {total_written} souboru do {out_dir}')
        print(f'[sfz_convert] instrument-definition.json ulozen')
    else:
        print(f'\n[sfz_convert] DRY RUN — bylo by zapsano {total_written} souboru do {out_dir}')
        print(f'[sfz_convert] instrument-definition.json:')
        print(json.dumps(idef, indent=4, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Convert SFZ instrument bank to mXXX-velX-fXX.wav format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('sfz',      metavar='SFZ',    help='Input .sfz file')
    ap.add_argument('out_dir',  metavar='OUT_DIR', help='Output directory')
    ap.add_argument('--vel-layers', type=int, default=8, metavar='N',
                    help='Number of velocity layers in output (default: 8)')
    ap.add_argument('--sr',  type=int, default=48000, metavar='HZ',
                    help='Target sample rate in Hz (default: 48000)')
    ap.add_argument('--name', metavar='NAME',
                    help='Instrument name for instrument-definition.json')
    ap.add_argument('--dry-run', action='store_true',
                    help='Show what would happen without writing files')
    args = ap.parse_args()

    convert(
        sfz_path        = args.sfz,
        out_dir         = args.out_dir,
        vel_layers      = args.vel_layers,
        target_sr       = args.sr,
        dry_run         = args.dry_run,
        instrument_name = args.name,
    )


if __name__ == '__main__':
    main()
