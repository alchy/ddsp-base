# DDSP Workflow — praktické příklady

Tento dokument ukazuje kompletní workflow pro dvě konkrétní banky:
- **vintage-vibe** — Rhodes elektrické piano (každý půltón, vel 0–7, 48 kHz)
- **salamander** — Steinway grand piano (každé 3 půltóny, vel 0–7, 48 kHz)

Oba nástroje jsou uloženy ve standardní adresářové struktuře `C:\SoundBanks\`.

---

## 1. Adresářová struktura

```
C:\SoundBanks\
  ddsp\
    vintage-vibe\          <- zdrojové WAV (READ-ONLY)  [MIDI 33–94, vel 0–7]
    salamander\            <- zdrojové WAV (READ-ONLY)  [MIDI 21–108, vel 0–7, každé 3 půltóny]
  SFZ\
    SalamanderGrandPianoV3\  <- originální SFZ banka (READ-ONLY)
  IthacaPlayer\
    vintage-vibe\          <- výstup generate (vytvořeno automaticky)
    salamander\            <- výstup generate (vytvořeno automaticky)

C:\SoundBanks\ddsp\vintage-vibe-ddsp\    <- workspace (extracts, checkpoints)
C:\SoundBanks\ddsp\salamander-ddsp\      <- workspace (extracts, checkpoints)
```

Workspace se vytvoří automaticky jako `<instrument>-ddsp\` vedle zdrojového adresáře.
Zdrojový adresář se nikdy nemodifikuje.

---

## 2. vintage-vibe (Rhodes elektrické piano)

### Zdroj
- 62 not × 8 velocity vrstev × 2 SR = 950 WAV souborů
- MIDI 33 (A1) – 94 (Bb6), každý půltón
- Filename: `mXXX-velY-f48.wav` → F0 se odvozuje přímo z názvu (known-F0 mod)

### Krok 1: Extrakce příznaků

```bash
python ddsp.py extract \
  --instrument C:\SoundBanks\ddsp\vintage-vibe \
  --chunk-sec 0
```

- `--chunk-sec 0` — soubory nepřerušovat na chunky (każý WAV je 1 nota, krátký)
- Known-F0 mod se aktivuje automaticky (název obsahuje `mXXX`)
- Výstup: `C:\SoundBanks\ddsp\vintage-vibe-ddsp\extracts\*.npz`
- Rychlost: ~1–2 s/soubor (IO dominuje), celkem ~15–20 min

Pokud chcete ověřit s pyin místo known-F0 (pomalé, kontrolní):
```bash
python ddsp.py extract --instrument C:\SoundBanks\ddsp\vintage-vibe --force-pyin
```

### Krok 2: Trénování

```bash
python ddsp.py learn \
  --instrument C:\SoundBanks\ddsp\vintage-vibe \
  --model medium \
  --epochs 200 \
  --lr 3e-4 \
  --batch-size 4
```

- `--model medium` — doporučeno pro Rhodes (~452K params, dobrý kompromis kvalita/rychlost)
- `--epochs 200` — 200 epoch pro plný timbre; pro rychlý test postačí 50
- Checkpointy: `vintage-vibe-ddsp\checkpoints\best.pt`, `last.pt`, `envelope.pt`
- Pokračování po přerušení: přidejte `--resume`
- **EnvelopeNet** se trénuje automaticky na konci — trvá sekundy, uloží se jako `envelope.pt`

Pro samostatný (re)trénink obálkového modelu:
```bash
python ddsp.py learn-envelope --instrument C:\SoundBanks\ddsp\vintage-vibe
```

Orientační doby trénování (model medium):

| Hardware | ~čas na 200 epoch |
|----------|-------------------|
| CPU 8 jader | 2–4 hodiny |
| GPU RTX 3080 | 15–30 min |

### Krok 3: Generování kompletní sample banky

```bash
python ddsp.py generate \
  --instrument C:\SoundBanks\ddsp\vintage-vibe \
  --full-range \
  --midi-lo 33 \
  --midi-hi 94 \
  --vel-layers 8 \
  --wet 1.0
```

- `--full-range` — generuje každý MIDI chromatic v rozsahu `--midi-lo` až `--midi-hi`
- `--midi-lo 33` / `--midi-hi 94` — rozsah odpovídá zdrojovým vzorkům (A1–Bb6)
- `--vel-layers 8` — 8 velocity vrstev (0–7), odpovídá zdrojové bance
- `--wet 1.0` — čistý DDSP výstup (0.0 = originál, 1.0 = plný synth)
- **Délka každého samplu** — odvozena automaticky z NPZ extraktů (reálná naučená obálka);
  pro každou (midi, vel) kombinaci se najde nejbližší uložená obálka z trénovacích dat.
  Basy mohou mít přirozeně 20+ sekund dozvuku, výšky kratší.
- Výstup: `C:\SoundBanks\IthacaPlayer\vintage-vibe\mXXX-velY-f48.wav`
- Celkem: 62 not × 8 vel = **496 WAV souborů**

Generování jen pro konkrétní noty nebo velocity:
```bash
# pouze noty C3–C5 a velocity 5,7
python ddsp.py generate \
  --instrument C:\SoundBanks\ddsp\vintage-vibe \
  --notes C3 D3 E3 F3 G3 A3 B3 C4 D4 E4 F4 G4 A4 B4 C5 \
  --vel 5 7
```

### Krok 4: Kontrola stavu

```bash
python ddsp.py status --instrument C:\SoundBanks\ddsp\vintage-vibe
```

---

## 3. salamander (Steinway grand piano)

### Zdroj
- 30 not × 8 velocity vrstev = 240 WAV souborů
- MIDI 21 (A0) – 108 (C8), vzorkováno každé 3 půltóny
- Filename: `mXXX-velY-f48.wav` → F0 se odvozuje přímo z názvu (known-F0 mod)
- Původ: Salamander Grand Piano V3 (SFZ banka) → převod přes `sfz_convert.py`

### Krok 1: Extrakce příznaků

```bash
python ddsp.py extract \
  --instrument C:\SoundBanks\ddsp\salamander \
  --chunk-sec 0
```

- `--chunk-sec 0` — bez dělení na chunky (salamander soubory jsou ~20 s, bez dlouhých stop)
- Known-F0 mod: F0 = 440 × 2^((midi−69)/12), žádné pyin
- Výstup: `C:\SoundBanks\ddsp\salamander-ddsp\extracts\*.npz`
- Rychlost: ~2 s/soubor (IO velké soubory), celkem ~8–10 min

### Krok 2: Trénování

```bash
python ddsp.py learn \
  --instrument C:\SoundBanks\ddsp\salamander \
  --model medium \
  --epochs 300 \
  --lr 3e-4 \
  --batch-size 4
```

- `--model medium` — doporučeno; pro vyšší kvalitu `--model large`
- `--epochs 300` — piano má složitější timbre, více epoch pomáhá
- Checkpointy: `salamander-ddsp\checkpoints\best.pt`, `last.pt`, `envelope.pt`
- **EnvelopeNet** (~25K params) se trénuje automaticky na konci `learn`; naučí se
  interpolovat délku doznívání mezi vzorkovanými notami (ob 3 půltóny → každý půltón)

Pro maximální kvalitu (pomalé):
```bash
python ddsp.py learn \
  --instrument C:\SoundBanks\ddsp\salamander \
  --model large \
  --epochs 500 \
  --lr 3e-4
```

### Krok 3: Generování kompletní chromatické banky

```bash
python ddsp.py generate \
  --instrument C:\SoundBanks\ddsp\salamander \
  --full-range \
  --midi-lo 21 \
  --midi-hi 108 \
  --vel-layers 8 \
  --wet 1.0
```

- `--midi-lo 21` / `--midi-hi 108` — plný rozsah klavíru A0–C8
- Model automaticky interpoluje mezilehlé noty (zdrojové vzorky jsou ob 3 půltóny)
- **Délka každého samplu** — odvozena z NPZ extraktů; basové noty mohou mít reálně
  20+ sekund dozvuku (A0 vel 7 ≈ 25 s v Salamander). Obálka se nikde neumělě neořezává.
- Výstup: `C:\SoundBanks\IthacaPlayer\salamander\mXXX-velY-f48.wav`
- Celkem: 88 not × 8 vel = **704 WAV souborů**

### Krok 4: Export do SFZ (volitelné)

```bash
python tools/bank_to_sfz.py C:\SoundBanks\IthacaPlayer\salamander \
  --out C:\SoundBanks\SFZ\salamander-ddsp.sfz \
  --sr 48 \
  --name "Salamander DDSP"
```

---

## 4. Rychlý referenční přehled

### Extrakce

```
extract --instrument <dir>          # known-F0 (výchozí), bez chunků = --chunk-sec 0
        --chunk-sec 0               # nedělí dlouhé soubory (doporučeno pro sample banky)
        --force-pyin                # přepne na pomalý pyin odhad F0
        --workspace <dir>           # override workspace (pokud se data přesunula)
```

### Trénování

```
learn   --instrument <dir>
        --model small|medium|large  # small=115K, medium=452K, large=1.99M params
        --epochs N                  # doporučeno: 200 (Rhodes), 300–500 (piano)
        --lr 3e-4                   # výchozí learning rate (Adam)
        --batch-size 4              # výchozí, zvyšte pokud máte dost RAM/VRAM
        --resume                    # pokračovat od last.pt
        --workspace <dir>
```

### Generování

```
generate --instrument <dir>
         --full-range               # kompletní chromatická banka (bez zdrojových WAV)
         --midi-lo 21               # spodní nota (default A0)
         --midi-hi 108              # horní nota (default C8)
         --vel-layers 8             # počet velocity vrstev (default 8)
                                    # délka: automaticky z NPZ obálek (žádný --duration)
         --wet 1.0                  # mix DDSP/originál (default 1.0 = čistý synth)
         --notes C4 A3 ...          # filtr: jen tyto noty (bez --full-range)
         --vel 5 7                  # filtr: jen tyto velocity vrstvy
         --output <dir>             # override výstupního adresáře
         --no-skip                  # přepsat existující soubory
         --workspace <dir>
```

### Doporučené nastavení per instrument

| Parametr | vintage-vibe (Rhodes) | salamander (Steinway) |
|----------|----------------------|-----------------------|
| `--model` | medium | medium / large |
| `--epochs` | 200 | 300–500 |
| délka samplu (generate) | z NPZ obálek (~1–8 s) | z NPZ obálek (~3–25 s) |
| `--midi-lo` | 33 (A1) | 21 (A0) |
| `--midi-hi` | 94 (Bb6) | 108 (C8) |
| `--vel-layers` | 8 | 8 |
| Výsledných WAV | 496 | 704 |

---

## 5. Shrnutí datového toku

```
C:\SoundBanks\SFZ\SalamanderGrandPianoV3\   (originální SFZ, READ-ONLY)
        |
        v  python tools/sfz_convert.py *.sfz C:\SoundBanks\ddsp\salamander --vel-layers 8
        |
C:\SoundBanks\ddsp\<nastroj>\               (zdrojové WAV mXXX-velY-f48.wav, READ-ONLY)
        |
        v  python ddsp.py extract --instrument <dir> --chunk-sec 0
        |  [known-F0: freq z názvu souboru, <0.1 s/soubor]
        |
C:\SoundBanks\ddsp\<nastroj>-ddsp\extracts\*.npz
        |
        v  python ddsp.py learn --instrument <dir> --model medium --epochs 300
        |  [MRSTFT loss + L1, Adam, CosineAnnealingLR]
        |
C:\SoundBanks\ddsp\<nastroj>-ddsp\checkpoints\best.pt
        |
        v  python ddsp.py generate --instrument <dir> --full-range ...
        |  [inference: každý MIDI chromatic × 8 vel, syntetická ADSR obálka]
        |
C:\SoundBanks\IthacaPlayer\<nastroj>\mXXX-velY-f48.wav   (výstup pro IthacaPlayer)
        |
        v  (volitelně) python tools/bank_to_sfz.py <dir> --out <nastroj>.sfz
        |
C:\SoundBanks\SFZ\<nastroj>-ddsp.sfz        (pro Kontakt / sforzando / jiné SFZ playery)
```
