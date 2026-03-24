"""
generate-midi.py — Vygeneruje MIDI sekvenci pro sampling VST piana

Vystup:
  piano_sample_session.mid   — nacti do DAW (Studio One apod.)
  timing_map.json            — referenci cas pro rezani nahravky

Pouziti:
  python generate-midi.py
  python generate-midi.py --hold 12 --gap 13
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
    parser.add_argument('--out',     default='piano_sample_session.mid',
                        help='Vystupni MIDI soubor (default: piano_sample_session.mid)')
    parser.add_argument('--map',     default='timing_map.json',
                        help='Vystupni timing map JSON (default: timing_map.json)')
    args = parser.parse_args()

    sampled_notes = list(range(args.midi_min, args.midi_max + 1, args.step))
    slot_sec      = args.hold + args.gap
    total_sec     = len(sampled_notes) * len(VELOCITY_MAP) * slot_sec

    print(f'Notes:     {len(sampled_notes)}  (every {args.step} semitones, '
          f'MIDI {sampled_notes[0]}-{sampled_notes[-1]})')
    print(f'Velocity:  {len(VELOCITY_MAP)} layers  (vel0-vel{len(VELOCITY_MAP)-1})')
    print(f'Slot:      {args.hold}s hold + {args.gap}s gap = {slot_sec}s per note')
    print(f'Total:     {total_sec // 60} min {total_sec % 60} sec')
    print()

    midi_events = []
    timing_map  = []

    beat = 0
    for midi_note in sampled_notes:
        for vel_idx, midi_vel in VELOCITY_MAP.items():
            name = f'm{midi_note:03d}-vel{vel_idx}-f48'
            timing_map.append({
                'name':      name,
                'midi_note': midi_note,
                'vel_idx':   vel_idx,
                'start_sec': beat,
                'end_sec':   beat + slot_sec,
            })
            midi_events.append((beat, midi_note, midi_vel, args.hold))
            beat += slot_sec

    with open(args.out, 'wb') as f:
        f.write(build_midi(midi_events))
    print(f'Saved: {args.out}')

    with open(args.map, 'w', encoding='utf-8') as f:
        json.dump({
            'sr':       SR,
            'hold_sec': args.hold,
            'gap_sec':  args.gap,
            'samples':  timing_map,
        }, f, indent=2)
    print(f'Saved: {args.map}')
    print()
    print('Next: open', args.out, 'in Studio One')
    print('  -> assign instrument plugin, export/record as WAV 48kHz stereo')

if __name__ == '__main__':
    main()
