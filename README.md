# DDSP Neural Vocoder

Naučí timbre monofónního nástroje z WAV vzorků a syntetizuje nové vzorky
podmíněné trojicí (F0, hlasitost, velocity). Výstupem je stereo WAV banka
kompatibilní se samplerovými pluginy (IthacaPlayer, Kontakt, sfz).

Technické detaily architektury a roadmap jsou v `docs/`.

---

## Požadavky

- Python 3.10+
- PyTorch ≥ 2.0, librosa ≥ 0.10, soundfile ≥ 0.12, gradio ≥ 4.0
- GPU není nutné — model `small` trénuje na CPU za přijatelnou dobu

```bash
pip install -r requirements.txt
```

Na Windows s CUDA (volitelně pro rychlejší trénování):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Adresářová struktura

```
C:\SoundBanks\
  ddsp\
    <nástroj>\          <- zdrojové WAV soubory (READ-ONLY vstup)
  IthacaPlayer\
    <nástroj>\          <- vygenerované vzorky (výstup)

<nástroj>-ddsp\         <- workspace (vedle zdrojového adresáře)
  extracts\             <- NPZ cache extrahovaných příznaků
  checkpoints\          <- best.pt, last.pt, envelope.pt
  instrument.json       <- konfigurace a stav
  train.log             <- log trénování
```

Zdrojový adresář je nikdy nemodifikován. Workspace lze přepsat parametrem
`--workspace <cesta>`.

---

## Rychlý start

```bash
# 1. Extrakce příznaků ze zdrojových WAV
python ddsp.py extract --instrument C:\SoundBanks\ddsp\ks-grand

# 2. Trénování
python ddsp.py learn --instrument C:\SoundBanks\ddsp\ks-grand --model small --epochs 100

# 3. Generování vzorků
python ddsp.py generate --instrument C:\SoundBanks\ddsp\ks-grand

# Grafické rozhraní (alternativa)
python gui.py
```

Výsledek: `C:\SoundBanks\IthacaPlayer\ks-grand\` obsahuje syntetizované WAV soubory.

---

## Formát názvů souborů

```
mXXX-velY-fZZ.wav
^^^  ^^^  ^^^
|    |    +-- vzorkovací frekvence: f48 = 48 kHz
|    +------- velocity vrstva: 0–7
+------------ MIDI číslo noty: 000–127 (060 = C4)
```

Soubory bez tohoto formátu jsou podporovány — velocity se odhadne automaticky,
F0 se odhadne pomocí pyin (pomalejší).

---

## Krok za krokem

### 1. Extrakce příznaků

```bash
python ddsp.py extract --instrument <adresář> [--chunk-sec 60] [--force-pyin]
```

- Pokud název souboru obsahuje MIDI číslo (`mXXX`), F0 se odvozuje přímo
  z názvu — rychlé (<0.1 s/soubor), žádný pyin.
- `--chunk-sec`: rozdělí dlouhé soubory na úseky (doporučeno pro soubory > 30 s).
- `--force-pyin`: vynucuje pomalý odhad F0 pyin (~20 s/soubor).

### 2. Trénování

```bash
python ddsp.py learn --instrument <adresář> [--model small|medium|large] [--epochs 100] [--resume]
```

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--model` | `small` | Velikost modelu (viz níže) |
| `--epochs` | `100` | Počet tréninkových epoch |
| `--lr` | `3e-4` | Learning rate (Adam) |
| `--batch-size` | `4` | Velikost batch |
| `--resume` | off | Pokračovat od `last.pt` |

Uložená data: `best.pt`, `last.pt`, `train.log`.

### 3. Trénování EnvelopeNet (volitelné)

```bash
python ddsp.py learn-envelope --instrument <adresář> [--epochs 1000]
```

Malá síť (~30 K param) pro predikci hlasitostní obálky z (MIDI, velocity).
Potřebná pro generování bez referenčního audia (`--full-range` nebo `--envelope-source envelopenet`).

### 4. Generování

```bash
python ddsp.py generate --instrument <adresář> [volby]
```

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--full-range` | off | Syntetizuj všechny noty MIDI lo–hi (vyžaduje EnvelopeNet nebo NPZ) |
| `--midi-lo` | `21` | Nejnižší nota pro full-range (A0) |
| `--midi-hi` | `108` | Nejvyšší nota pro full-range (C8) |
| `--vel-layers` | `8` | Počet velocity vrstev |
| `--envelope-source` | `auto` | `auto` / `envelopenet` / `npz` |
| `--wet` | `1.0` | Mix DDSP / originál (0.0–1.0) |
| `--notes` | vše | Filtr not, např. `C4 A3` |
| `--inharmonicity-scale` | `1.0` | `0`=čistě harmonické, `1`=naučené, `2`=zesílené |
| `--decay-scale` | `1.0` | `0`=bez fyzikálního decay, `1`=naučené, `2`=rychlejší |
| `--attack-ramp-ms` | `10` | Délka náběhu (ms), `0`=vypnuto |
| `--output` | IthacaPlayer/... | Výstupní adresář |
| `--no-skip` | off | Přepsat existující soubory |

### 5. Stav

```bash
python ddsp.py status --instrument <adresář>
```

---

## Grafické rozhraní

```bash
python gui.py [--port 7860] [--share]
```

Otevře Gradio aplikaci v prohlížeči (`http://127.0.0.1:7860`). Záložky:

- **Nástroj & Stav** — zadání cesty, zobrazení stavu workspace
- **Extrakce** — extrakce příznaků s nastavením chunk-sec
- **EnvelopeNet** — trénování prediktoru hlasitostní obálky
- **DDSP Model** — trénování hlavního modelu
- **Generování** — inference s ovládacími slidery (inharmonicity, decay)

Log se aktualizuje každé 2 sekundy. Tlačítko Stop přeruší běh příkazu.

---

## Velikosti modelu

Projekt je určen pro Grand Piano (88 not × 8 velocity vrstev). Piano pokrývá
extrémně široký rozsah — `small` slouží jen pro diagnostiku, ne produkci.

| Preset | Parametry | Doporučení |
|--------|-----------|------------|
| `small` | ~598 K | Rychlý test / diagnostika (50 epoch) |
| `medium` | ~2.1 M | **Výchozí pro piano** — 2-vrstvý GRU, zvládne celý rozsah 88 not |
| `large` | ~4.4 M | Produkční banka, nejlepší kvalita (GPU doporučeno) |

---

## Konvertory

### SFZ banka → mXXX-velX-fXX.wav

```bash
python tools/sfz2ithacabank.py <soubor.sfz> <výstupní_adresář> [--vel-layers 8] [--sr 48000]
```

### mXXX-velX-fXX.wav → SFZ

```bash
python tools/ithacabank2sfz.py <adresář_banky> [--out <soubor.sfz>]
```

---

## Dokumentace

| Dokument | Obsah |
|----------|-------|
| `docs/ARCHITECTURE.md` | Signálový model, architektura sítě, konstanty |
| `docs/MODEL_ROADMAP.md` | Implementované funkce, roadmap vylepšení |
| `docs/NPZ_FORMAT.md` | Formát NPZ cache souborů |
| `docs/WORKFLOW_EXAMPLES.md` | Příklady použití pro různé nástroje |
| `docs/BASS_REFACTOR_CONCEPT.md` | Analýza problémů basů, fyzikální základ oprav |
