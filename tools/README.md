# tools/

Pomocné skripty pro přípravu a konverzi sample banků pro DDSP pipeline.

---

## generate-midi.py

Vygeneruje MIDI sekvenci pro sampling VST/SFZ piana v DAW (Studio One, Reaper, …)
a k ní `timing_map.json` s přesnými časy každé noty.

### Použití

```bash
python generate-midi.py                        # výchozí nastavení
python generate-midi.py --hold 12 --gap 13    # 12s drzeni + 13s decay = 25s/nota
python generate-midi.py --step 6              # kazda druha mala tercie (ridsí banka)
python generate-midi.py --midi-min 36 --midi-max 96  # omezeny rozsah
python generate-midi.py --out moje_session.mid --map moje_mapa.json
```

### Parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--hold` | 12 | Doba držení klávesy v sekundách |
| `--gap`  | 13 | Pauza po puštění (decay tail) v sekundách |
| `--step` | 3  | Krok mezi notami v půltónech (3 = malá tercie) |
| `--midi-min` | 21 | Nejnižší MIDI nota (21 = A0) |
| `--midi-max` | 108 | Nejvyšší MIDI nota (108 = C8) |
| `--out`  | `piano_sample_session.mid` | Výstupní MIDI soubor |
| `--map`  | `timing_map.json` | Výstupní timing mapa (JSON) |

Slot na notu = `hold + gap`. Výchozí 25 s/nota × 30 not × 8 velocity = **100 min**.

### Workflow

```
python generate-midi.py
  ↓
piano_sample_session.mid  →  otevři v Studio One
                               přiřaď VST nástroj (EZkeys, Kontakt, …)
                               exportuj jako WAV 48 kHz stereo
  ↓
recorded.wav  →  rozřež pomocí sample-slicer (https://github.com/alchy/sample-slicer)
                 namapuj pitche v sample-editor (https://github.com/alchy/sample-editor)
  ↓
mXXX-velX-f48.wav  →  python ddsp.py extract --instrument <slozka>
```

---

## sfz2ithacabank.py

Konvertuje SFZ instrument bank do formátu `mXXX-velX-fXX.wav` pro DDSP pipeline.

### Použití

```bash
python sfz2ithacabank.py Salamander.sfz C:/SoundBanks/ddsp/salamander
python sfz2ithacabank.py Salamander.sfz C:/SoundBanks/ddsp/salamander --vel-layers 8 --sr 48000
python sfz2ithacabank.py Salamander.sfz C:/SoundBanks/ddsp/salamander --dry-run
```

### Parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `SFZ` | — | Vstupní `.sfz` soubor |
| `OUT_DIR` | — | Výstupní adresář |
| `--vel-layers` | 8 | Počet velocity vrstev ve výstupu |
| `--sr` | 48000 | Cílová sample rate v Hz |
| `--name` | z SFZ | Název nástroje pro `instrument-definition.json` |
| `--dry-run` | — | Zobraz co by se stalo bez zápisu souborů |

### Výstup

```
<OUT_DIR>/
    m021-vel0-f48.wav
    m021-vel1-f48.wav
    ...
    instrument-definition.json
```

Pokud SFZ obsahuje více velocity vrstev než `--vel-layers`, skript vybere
rovnoměrně rozmístěné vrstvy. Pokud méně, použije dostupné.

Resample: automaticky přes `scipy` nebo `librosa` (je-li nainstalováno).

---

## ithacabank2sfz.py

Exportuje DDSP sample banku (`mXXX-velX-fXX.wav`) jako SFZ soubor
pro přehrávání v libovolném SFZ přehrávači (sfizz, ARIA, Plogue Sforzando, …).

### Použití

```bash
python ithacabank2sfz.py C:/SoundBanks/ddsp/salamander
python ithacabank2sfz.py C:/SoundBanks/ddsp/salamander --out salamander.sfz
python ithacabank2sfz.py C:/SoundBanks/ddsp/salamander --sr 48
python ithacabank2sfz.py C:/SoundBanks/ddsp/salamander --absolute-paths
```

### Parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `BANK_DIR` | — | Adresář s `mXXX-velX-fXX.wav` soubory |
| `--out` | `<BANK_DIR>.sfz` | Výstupní `.sfz` soubor |
| `--sr` | — | Filtruj jen soubory s touto SR (kHz), např. `48` |
| `--name` | z `instrument-definition.json` | Název nástroje v SFZ |
| `--absolute-paths` | — | Absolutní cesty místo relativních |

### Key mapping

Každá samplovaná nota pokrývá rozsah do půli vzdálenosti k sousedním notám
(standardní SFZ keymap). Nejnižší nota od MIDI 0, nejvyšší do MIDI 127.

### Velocity mapping

Pro 8 vrstev: vel0→0–15, vel1→16–31, … vel7→112–127.
