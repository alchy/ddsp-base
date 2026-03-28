# DDSP Neural Vocoder

Naučí timbre Grand Piana z WAV vzorků a syntetizuje stereo WAV banku
kompatibilní se samplerovými pluginy (IthacaPlayer, Kontakt, SFZ).

Technická architektura a roadmap jsou v `docs/`.

---

## Požadavky

- Python 3.10+
- PyTorch ≥ 2.3 (plně podporuje Python 3.13)

```bash
pip install -r requirements.txt
```

GPU není povinné, ale výrazně urychlí trénink. Bez GPU program automaticky
použije preset `piano-cpu` (kratší okna, menší model).

**CUDA (volitelně):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Rychlý start

```bash
# 1. Extrakce příznaků ze zdrojových WAV
python ddsp.py extract --instrument C:\SoundBanks\ddsp\ks-grand

# 2. Trénování (nastavení z presetu, auto-detekce zařízení)
python ddsp.py learn --instrument C:\SoundBanks\ddsp\ks-grand

# 3. Generování vzorků
python ddsp.py generate --instrument C:\SoundBanks\ddsp\ks-grand

# Grafické rozhraní (alternativa ke CLI)
python gui.py
```

---

## Adresářová struktura

```
C:\SoundBanks\
  ddsp\
    <nástroj>\          ← zdrojové WAV (READ-ONLY vstup)
      mXXX-velY-fZZ.wav
      instrument-definition.json   (volitelné)
  IthacaPlayer\
    <nástroj>\          ← vygenerovaná WAV banka (výstup)

<nástroj>-ddsp\         ← workspace (vedle zdrojového adresáře)
  extracts\             ← NPZ cache extrahovaných příznaků
  checkpoints\
    best.pt             ← nejlepší checkpoint
    last.pt             ← poslední epocha (pro --resume)
    preview\            ← diagnostické WAV po každém best (přepisují se)
  train.json            ← per-instrument přepisy presetu (volitelné)
  instrument.json       ← metadata a stav workspace
  train.log             ← log trénování
```

Zdrojový adresář není nikdy modifikován. Workspace lze přesunout pomocí
`--workspace <cesta>`.

---

## Formát názvů souborů

```
mXXX-velY-fZZ.wav
^^^  ^^^  ^^^
|    |    +-- vzorkovací frekvence: f48 = 48 kHz
|    +------- velocity vrstva: 0–7
+------------ MIDI číslo noty: 000–127  (060 = C4, 021 = A0, 108 = C8)
```

Soubory bez tohoto formátu jsou podporovány — velocity se odhadne automaticky,
F0 pomocí pyin (pomalejší).

---

## Tréninkové presety

Všechna nastavení tréninku jsou uložena v **JSON presetech** v adresáři
`model-presets/`. Program automaticky vybere preset podle zařízení.

| Preset | Zařízení | Model | Batch | Epochy | max_crop | Popis |
|--------|----------|-------|-------|--------|----------|-------|
| `piano-cpu` | CPU | small (~598K) | 4 | 200 | 50 fr (0,25 s) | Rychlé epochy na CPU; test konvergence |
| `piano-cuda` | CUDA GPU | medium (~2,1M) | 16 | 200 | adaptive | Produkce na GPU; plná kvalita |
| `piano-mps` | Apple Silicon (M1–M3) | medium (~2,1M) | 8 | 200 | adaptive | Konzervativní pro starší MPS |
| `piano-m5` | **Apple M5** (24 GB) | **large (~4,4M)** | 16 | **1000** | adaptive | **Produkční preset M5 — maximální kvalita** |
| `piano-m5-medium` | Apple M5 (24 GB) | medium (~2,1M) | 16 | 500 | adaptive | M5 diagnostika / přechodný trénink |

### Úprava presetu

Edituj přímo soubor v `model-presets/`, nebo přidej vlastní:

```json
// model-presets/muj-preset.json
{
  "_description": "Můj vlastní preset",
  "model": "large",
  "epochs": 500,
  "lr": 0.0002,
  "batch_size": 32,
  "max_crop": null,
  "min_voiced": 0.1
}
```

```bash
python ddsp.py learn --instrument ... --preset muj-preset
```

### Per-instrument přepisy (train.json)

Pro přepis jednotlivých hodnot bez změny presetu vytvoř
`<workspace>/train.json`:

```json
{
  "preset": "piano-cuda",
  "epochs": 500,
  "lr": 0.0001
}
```

Priorita: **CLI `--preset` > `train.json` > preset soubor**.

---

## Parametry příkazů

### `extract` — extrakce příznaků

```bash
python ddsp.py extract --instrument <DIR> [--workspace <DIR>] [--chunk-sec 60] [--force-pyin]
```

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--instrument` | — | Adresář se zdrojovými WAV (povinné) |
| `--workspace` | `<nástroj>-ddsp/` | Výstupní adresář workspace |
| `--chunk-sec` | 60 | Rozdělí soubory delší než N sekund na úseky |
| `--force-pyin` | off | Odhad F0 přes pyin místo z názvu souboru (pomalejší) |

### `learn` — trénování

```bash
python ddsp.py learn --instrument <DIR> [--preset <NAME>] [--resume] [--device auto|cpu|cuda|mps]
```

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--instrument` | — | Adresář se zdrojovými WAV (povinné) |
| `--workspace` | `<nástroj>-ddsp/` | Workspace |
| `--preset` | auto dle zařízení | Název presetu z `model-presets/` |
| `--resume` | off | Pokračovat od `last.pt` |
| `--device` | auto | `auto` / `cpu` / `cuda` / `mps` (Apple Silicon) |

Všechna ostatní nastavení (model, epochy, lr, batch…) jsou v presetu nebo `train.json`.

### `generate` — generování vzorků

```bash
python ddsp.py generate --instrument <DIR> [volby]
```

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--instrument` | — | Povinné |
| `--full-range` | off | Syntetizuj celý rozsah not (vyžaduje EnvelopeNet nebo NPZ) |
| `--midi-lo` | `21` | Nejnižší nota (A0) |
| `--midi-hi` | `108` | Nejvyšší nota (C8) |
| `--vel-layers` | `8` | Počet velocity vrstev |
| `--envelope-source` | `auto` | `auto` / `envelopenet` / `npz` |
| `--wet` | `1.0` | Mix DDSP/originál (0,0–1,0) |
| `--inharmonicity-scale` | `1.0` | `0`=čistě harmonické · `1`=naučené · `2`=zesílené |
| `--decay-scale` | `1.0` | `0`=bez fyzikálního decay · `1`=naučené · `2`=rychlejší |
| `--output` | IthacaPlayer/... | Výstupní adresář |
| `--no-skip` | off | Přepsat existující soubory |

### `learn-envelope` — trénování EnvelopeNet

```bash
python ddsp.py learn-envelope --instrument <DIR> [--epochs 1000]
```

Volitelná malá síť (~30 K param) pro predikci hlasitostní obálky z (MIDI, velocity).
Potřebná pro `--full-range` bez referenčních NPZ.

### `status` — stav workspace

```bash
python ddsp.py status --instrument <DIR>
```

---

## Grafické rozhraní

```bash
python gui.py [--port 7860] [--share]
```

Otevře Gradio aplikaci (`http://127.0.0.1:7860`).

---

## Velikosti modelů

| Preset | Model | Parametry | Popis |
|--------|-------|-----------|-------|
| `piano-cpu` | small | ~598 K | Rychlá diagnostika na CPU |
| `piano-cuda` / `piano-mps` | medium | ~2,1 M | **Výchozí produkční model** — 2vrstvý GRU |
| vlastní | large | ~4,4 M | Nejvyšší kvalita; GPU doporučeno |

Klíčová charakteristika: GRU vrstvy jsou dominantní složka medium/large
(47 % parametrů). Noise heads platí při mlp_dim=512.

---

## Dokumentace

| Dokument | Obsah |
|----------|-------|
| `docs/ARCHITECTURE.md` | Signálový model, síťová architektura, konstanty |
| `docs/MODEL_ROADMAP.md` | Implementované funkce, roadmap |
| `docs/NPZ_FORMAT.md` | Formát NPZ cache souborů |
| `docs/WORKFLOW_EXAMPLES.md` | Příklady pro různé nástroje |
| `docs/BASS_REFACTOR_CONCEPT.md` | Fyzikální základ dvoukomponentního decay |
