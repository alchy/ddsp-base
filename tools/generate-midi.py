"""
generate-midi.py — Vygeneruje MIDI sekvenci pro sampling VST piana

Vystup:
  piano_sample_session.mid   — nacti do DAW (Studio One apod.)
  timing_map.json            — referenci cas pro rezani nahravky

Pouziti:
  python generate-midi.py
  python generate-midi.py --hold 20 --gap 5 --step 1
  python generate-midi.py --hold 20 --gap 5 --step 1 --split-octaves
  python generate-midi.py --hold 8 --gap 10 --step 6 --out my_session.mid
"""

import json
import struct
import argparse

# ---------------------------------------------------------------------------
# Vychozi parametry
# ---------------------------------------------------------------------------

MIDI_MIN   = 21    # A0
MIDI_MAX   = 108   # C8
MIDI_STEP  = 3     # krok v pultonech (mala tercie)

VELOCITY_MAP = {
    0: 15,
    1: 30,
    2: 45,
    3: 60,
    4: 75,
    5: 90,
    6: 105,
    7: 120,
}

HOLD_SEC  = 12   # doba drzeni klavesy (sec)
GAP_SEC   = 13   # pauza po pusteni — decay tail (sec); slot = hold + gap
SR        = 48000

# ---------------------------------------------------------------------------
# Interni MIDI writer (bez externich zavislosti)
# ---------------------------------------------------------------------------

TICKS_PER_BEAT = 480
BPM            = 60   # 1 beat = 1 sekunda -> jednoduche pocitani

def _var_len(value):
    result = [value & 0x7F]
    value >>= 7
    while value:
        result.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(result)

def _tempo_event(bpm):
    uspb = int(60_000_000 / bpm)
    return b'\x00\xff\x51\x03' + struct.pack('>I', uspb)[1:]

def build_midi(notes):
    """notes: list of (start_beat, midi_note, velocity, duration_beats)"""
    events = [(0, _tempo_event(BPM))]
    for start_beat, note, vel, dur_beats in notes:
        t_on  = int(start_beat * TICKS_PER_BEAT)
        t_off = int((start_beat + dur_beats) * TICKS_PER_BEAT)
        events.append((t_on,  bytes([0x90, note, vel])))
        events.append((t_off, bytes([0x80, note, 0])))
    events.sort(key=lambda e: e[0])

    track_data = b''
    prev_tick  = 0
    for tick, data in events:
        delta = tick - prev_tick
        track_data += _var_len(delta) + data
        prev_tick = tick
    track_data += b'\x00\xff\x2f\x00'

    header = b'MThd' + struct.pack('>IHHH', 6, 0, 1, TICKS_PER_BEAT)
    track  = b'MTrk' + struct.pack('>I', len(track_data)) + track_data
    return header + track

# ---------------------------------------------------------------------------
# Hlavni logika
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Vygeneruje MIDI sekvenci pro sampling VST piana.'
    )
    parser.add_argument('--hold',    type=int, default=HOLD_SEC,
                        help=f'Doba drzeni klavesy v sekundach (default: {HOLD_SEC})')
    parser.add_argument('--gap',     type=int, default=GAP_SEC,
                        help=f'Pauza po pusteni / decay tail v sekundach (default: {GAP_SEC})')
    parser.add_argument('--step',    type=int, default=MIDI_STEP,
                        help=f'Krok mezi notami v pultonech (default: {MIDI_STEP})')
    parser.add_argument('--midi-min', type=int, default=MIDI_MIN, dest='midi_min',
                        help=f'Nejnizsi MIDI nota (default: {MIDI_MIN} = A0)')
    parser.add_argument('--midi-max', type=int, default=MIDI_MAX, dest='midi_max',
                        help=f'Nejvyssi MIDI nota (default: {MIDI_MAX} = C8)')
    parser.add_argument('--out',          default='piano_sample_session.mid',
                        help='Vystupni MIDI soubor (default: piano_sample_session.mid)')
    parser.add_argument('--map',          default='timing_map.json',
                        help='Vystupni timing map JSON (default: timing_map.json)')
    parser.add_argument('--split-octaves', action='store_true', dest='split_octaves',
                        help='Rozdel vystup na vice souboru po oktavach (C-based hranice)')
    args = parser.parse_args()

    sampled_notes = list(range(args.midi_min, args.midi_max + 1, args.step))
    slot_sec      = args.hold + args.gap
    total_sec     = len(sampled_notes) * len(VELOCITY_MAP) * slot_sec

    print(f'Notes:     {len(sampled_notes)}  (every {args.step} semitones, '
          f'MIDI {sampled_notes[0]}-{sampled_notes[-1]})')
    print(f'Velocity:  {len(VELOCITY_MAP)} layers  (vel0-vel{len(VELOCITY_MAP)-1})')
    print(f'Slot:      {args.hold}s hold + {args.gap}s gap = {slot_sec}s per note')
    print(f'Total:     {total_sec // 60} min {total_sec % 60} sec')
    if args.split_octaves:
        oct_notes = 12 // args.step if args.step <= 12 else 1
        print(f'Split:     ~{oct_notes * len(VELOCITY_MAP) * slot_sec // 60} min/oktava')
    print()

    if args.split_octaves:
        _write_split_octaves(sampled_notes, slot_sec, args)
    else:
        _write_single(sampled_notes, slot_sec, args)


def _build_events_map(sampled_notes, slot_sec, hold_sec):
    """Vraci (midi_events, timing_map) s casem od 0."""
    midi_events = []
    timing_map  = []
    beat = 0
    for midi_note in sampled_notes:
        for vel_idx, midi_vel in VELOCITY_MAP.items():
            timing_map.append({
                'name':      f'm{midi_note:03d}-vel{vel_idx}-f48',
                'midi_note': midi_note,
                'vel_idx':   vel_idx,
                'start_sec': beat,
                'end_sec':   beat + slot_sec,
            })
            midi_events.append((beat, midi_note, midi_vel, hold_sec))
            beat += slot_sec
    return midi_events, timing_map


def _save(midi_path, map_path, midi_events, timing_map, args):
    with open(midi_path, 'wb') as f:
        f.write(build_midi(midi_events))
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump({'sr': SR, 'hold_sec': args.hold, 'gap_sec': args.gap,
                   'samples': timing_map}, f, indent=2)
    n   = len(set(e['midi_note'] for e in timing_map))
    sec = timing_map[-1]['end_sec']
    print(f'  {midi_path}  ({n} not, {sec // 60} min {sec % 60} sec)')


def _write_single(sampled_notes, slot_sec, args):
    events, tmap = _build_events_map(sampled_notes, slot_sec, args.hold)
    _save(args.out, args.map, events, tmap, args)
    print()
    print('Next: open', args.out, 'in Studio One')
    print('  -> assign instrument plugin, export/record as WAV 48kHz stereo')


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def _note_name(midi):
    return f'{NOTE_NAMES[midi % 12]}{midi // 12 - 1}'


def _write_split_octaves(sampled_notes, slot_sec, args):
    # Rozdel noty po oktavach: C_n = MIDI 12*(n+1) .. 12*(n+2)-1
    # Hranice: first_C <= note < next_C
    from collections import defaultdict
    octave_groups = defaultdict(list)
    for note in sampled_notes:
        oct_n = note // 12   # C-based cislo oktavy (C-1=0, C0=1, C1=2, ...)
        octave_groups[oct_n].append(note)

    stem = args.out.replace('.mid', '')
    map_stem = args.map.replace('.json', '')

    print('Generuji soubory:')
    for oct_n in sorted(octave_groups):
        notes = octave_groups[oct_n]
        lo, hi = notes[0], notes[-1]
        tag = f'{_note_name(lo)}-{_note_name(hi)}'
        midi_path = f'{stem}_{tag}.mid'
        map_path  = f'{map_stem}_{tag}.json'
        events, tmap = _build_events_map(notes, slot_sec, args.hold)
        _save(midi_path, map_path, events, tmap, args)

    print()
    print('Next: nahraj kazdy MIDI soubor zvlast v Studio One')
    print('  -> assign instrument plugin, export/record as WAV 48kHz stereo')

if __name__ == '__main__':
    main()
