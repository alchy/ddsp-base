# DDSP Neural Vocoder

Neuronová síť, která se naučí zvuk akustického klavíru z nahrávek a syntetizuje
stereo WAV banku kompatibilní se samplerovými pluginy (IthacaPlayer, Kontakt, SFZ).

Technologie: **DDSP** (Differentiable Digital Signal Processing) — fyzikálně
interpretovatelné bloky místo black-boxu. Model obsahuje sinusové oscilátory
pro harmonické složky, šumový syntetizátor pro přechodové jevy a explicitní
parametry pro inharmonicitu a dvoustupňový decay strun.

**Aktuální trénink:** `ks-grand` (Yamaha grand piano, 88 not × 8 velocity),
medium model (2.1M params), max_crop=200 (1.07 s), ep 38/500, val loss 1.8398.

> Podrobný popis projektu, co se počítalo a kde jsme: **[PROGRESS.md](PROGRESS.md)**

Technická architektura a roadmap jsou v `docs/`.

---

## Zprovoznění

### 1. Python prostředí

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Závislosti

```bash
pip install -r requirements.txt
```

GPU není povinné — bez něj se automaticky použije CPU preset.

**CUDA (volitelně, výrazně urychlí trénink):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Apple Silicon (MPS):**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 3. Zdrojová data

Zkopíruj nahrávky do:
```
C:\SoundBanks\ddsp\<nazev-nastroje>\
```

Formát názvů souborů:
```
mXXX-velY-fZZ.wav
^^^  ^^^  ^^^
|    |    +-- vzorkovací frekvence: f44 = 44.1 kHz, f48 = 48 kHz
|    +------- velocity vrstva: 0–7 (0 = nejslabší, 7 = nejsilnější)
+------------ MIDI číslo noty: 021 = A0, 060 = C4, 108 = C8
```

Soubory bez tohoto formátu jsou podporovány — velocity se odhadne automaticky,
F0 pomocí pyin (pomalejší).

### 4. Kompletní workflow (od nuly k bance)

```bash
# Krok 1 — extrakce příznaků (stačí jednou, výsledky se cachují)
python ddsp.py extract --instrument C:\SoundBanks\ddsp\ks-grand

# Krok 2 — trénování (spustí extract automaticky pokud chybí)
python ddsp.py learn --instrument C:\SoundBanks\ddsp\ks-grand --preset piano-cpu-quality

# Krok 3 — generování celé banky z nejlepšího checkpointu
python ddsp.py generate --instrument C:\SoundBanks\ddsp\ks-grand \
    --out C:\SoundBanks\IthacaPlayer\generated\ks-grand \
    --full-range --no-skip

# Volitelně — grafické rozhraní
python gui.py
```

---

## Adresářová struktura

```
C:\SoundBanks\
  ddsp\
    <nástroj>\              ← zdrojové WAV (READ-ONLY — nikdy se nemění)
      mXXX-velY-fZZ.wav
      instrument-definition.json   (volitelné)
  IthacaPlayer\
    generated\
      <nástroj>\            ← vygenerovaná WAV banka (výstup generate)

<nástroj>-ddsp\             ← workspace (vedle zdrojového adresáře)
  extracts\                 ← NPZ cache extrahovaných příznaků
  checkpoints\
    best.pt                 ← nejlepší checkpoint (použije generate)
    last.pt                 ← poslední epocha (pro --resume)
    preview\                ← diagnostické WAV po každém best
  train.json                ← per-instrument přepisy presetu (volitelné)
  instrument.json           ← metadata a stav workspace
  train.log                 ← log trénování
```

Workspace lze přemístit: `--workspace <cesta>`.

---

## Příkazy — kompletní reference

### `extract` — extrakce příznaků

Analyzuje zdrojové WAV → uloží F0, loudness, audio do NPZ cache.
Spouští se automaticky před `learn` pokud cache chybí.

```bash
python ddsp.py extract --instrument <DIR> [volby]
```

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--instrument <DIR>` | — | Zdrojový adresář (povinné) |
| `--workspace <DIR>` | `<nástroj>-ddsp/` | Výstupní workspace |
| `--chunk-sec <N>` | `60` | Rozdělí soubory delší než N sekund na úseky |
| `--force-pyin` | off | Odhad F0 přes pyin místo z názvu souboru (pomalejší, ale přesnější pro nestandardní názvy) |

---

### `learn` — trénování modelu

```bash
python ddsp.py learn --instrument <DIR> [volby]
```

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--instrument <DIR>` | — | Zdrojový adresář (povinné) |
| `--workspace <DIR>` | `<nástroj>-ddsp/` | Workspace |
| `--preset <NAME>` | auto dle zařízení | Název presetu z `model-presets/` (bez `.json`) |
| `--resume` | off | Pokračovat od `last.pt` místo nového tréninku |
| `--device <auto\|cpu\|cuda\|mps>` | `auto` | Zařízení; `auto` = CUDA → MPS → CPU |
| `--max-crop <N>` | z presetu | Přepis max délky trénovacího okna v snímcích |

Všechna ostatní nastavení (model, epochy, lr, batch…) jsou v presetu nebo `train.json`.

**Příklady:**
```bash
# CPU trénink s quality presetem
python ddsp.py learn --instrument C:\SoundBanks\ddsp\ks-grand --preset piano-cpu-quality

# Pokračovat od minulé session
python ddsp.py learn --instrument C:\SoundBanks\ddsp\ks-grand --preset piano-cpu-quality --resume

# Explicitní zařízení
python ddsp.py learn --instrument C:\SoundBanks\ddsp\ks-grand --device cuda
```

---

### `generate` — generování vzorků / banky

```bash
python ddsp.py generate --instrument <DIR> [volby]
```

#### Výběr not

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--full-range` | off | Syntetizuj celý rozsah (midi-lo až midi-hi) |
| `--midi-lo <N>` | `21` | Nejnižší nota v MIDI (21 = A0) |
| `--midi-hi <N>` | `108` | Nejvyšší nota v MIDI (108 = C8) |
| `--vel-layers <N>` | `8` | Počet velocity vrstev (0–N-1) |
| `--notes <NOTE …>` | — | Seznam konkrétních not (např. `C3 A4 C5`); alternativa k --full-range |
| `--vel <N …>` | — | Seznam konkrétních velocity vrstev (např. `0 4 7`) |

#### Výstup

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--out <DIR>` | `C:\SoundBanks\IthacaPlayer\<nástroj>\` | Výstupní adresář |
| `--no-skip` | off | Přepsat existující soubory (výchozí = přeskočit) |
| `--device <auto\|cpu\|cuda\|mps>` | `auto` | Zařízení pro inferenci |

#### Zvukové parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--wet <0-1>` | `1.0` | Mix DDSP vs originál; `0.0` = originál, `1.0` = čistý DDSP |
| `--inharmonicity-scale <0-2>` | `1.0` | `0` = čistě harmonické, `1` = naučené, `2` = zesílená inharmonicita |
| `--decay-scale <0-2>` | `1.0` | `0` = bez fyzikálního decay, `1` = naučené, `2` = rychlejší útlum; **pokud bas zní příliš mohutně, zkus `0.7`** |
| `--attack-ramp-ms <ms>` | `10.0` | Délka attack rampy v ms (fade-in na začátku noty) |
| `--envelope-source <auto\|envelopenet\|npz>` | `auto` | Zdroj hlasitostní obálky; `npz` = z nahrávky, `envelopenet` = predikce sítí |

**Příklady:**
```bash
# Celá banka (88 × 8 = 704 souborů)
python ddsp.py generate --instrument C:\SoundBanks\ddsp\ks-grand \
    --out C:\SoundBanks\IthacaPlayer\generated\ks-grand \
    --full-range --no-skip

# Jen bas (A0–C3) pro rychlý test
python ddsp.py generate --instrument C:\SoundBanks\ddsp\ks-grand \
    --full-range --midi-lo 21 --midi-hi 48 --no-skip

# Konkrétní noty
python ddsp.py generate --instrument C:\SoundBanks\ddsp\ks-grand \
    --notes A0 C1 C2 C3 C4 --vel 0 4 7

# Ztlumit přehnaný bas
python ddsp.py generate --instrument C:\SoundBanks\ddsp\ks-grand \
    --full-range --decay-scale 0.7 --no-skip \
    --out C:\SoundBanks\IthacaPlayer\generated\ks-grand-decay07
```

---

### `learn-envelope` — trénování EnvelopeNet

Volitelná malá síť (~30K params) pro predikci hlasitostní obálky z (MIDI, velocity).
Potřebná pro `generate --envelope-source envelopenet`.

```bash
python ddsp.py learn-envelope --instrument <DIR> [volby]
```

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--instrument <DIR>` | — | Povinné |
| `--epochs <N>` | `1000` | Počet epoch |
| `--lr <float>` | `0.001` | Learning rate |
| `--envelope-warp <float>` | z konstant | Warp faktor obálky |
| `--n-env <N>` | z konstant | Počet bodů obálky |
| `--attack-weight <float>` | `5.0` | Váha attack části v loss funkci |
| `--device <auto\|cpu\|cuda\|mps>` | `auto` | Zařízení |

---

### `status` — stav workspace

```bash
python ddsp.py status --instrument C:\SoundBanks\ddsp\ks-grand
```

Vypíše aktuální epochu, val loss, cestu k checkpointu a stav extrakce.

---

## Grafické rozhraní

```bash
python gui.py [--port 7860] [--share]
```

Otevře Gradio UI na `http://127.0.0.1:7860`.

| Parametr | Popis |
|----------|-------|
| `--port <N>` | Port (výchozí 7860) |
| `--share` | Veřejný odkaz přes Gradio tunel (sdílení přes internet) |

---

## Tréninkové presety

Uloženy v `model-presets/*.json`. Program vybere preset automaticky dle zařízení,
nebo explicitně přes `--preset`.

| Preset | Zařízení | Model | max_crop | Popis |
|--------|----------|-------|----------|-------|
| `piano-cpu` | CPU | small ~598K | 50 fr (0.25 s) | Rychlý test konvergence |
| `piano-cpu-quality` | CPU | medium ~2.1M | 200 fr (1.07 s) | **Aktuální trénink** — bas potřebuje dlouhá okna |
| `piano-cuda` | CUDA GPU | medium ~2.1M | adaptive | Produkce na GPU |
| `piano-mps` | Apple Silicon M1–M3 | medium ~2.1M | adaptive | Konzervativní MPS |
| `piano-m5` | Apple M5 24 GB | large ~4.4M | adaptive | Maximální kvalita |
| `piano-m5-medium` | Apple M5 24 GB | medium ~2.1M | adaptive | M5 diagnostika |

### Vlastní preset

```json
// model-presets/muj-preset.json
{
  "_description": "Popis",
  "model": "large",
  "epochs": 500,
  "lr": 0.0002,
  "batch_size": 32,
  "max_crop": 300,
  "min_voiced": 0.1
}
```

```bash
python ddsp.py learn --instrument ... --preset muj-preset
```

### Per-instrument přepisy (`train.json`)

Přepis konkrétních hodnot bez změny presetu — vytvoř `<workspace>/train.json`:

```json
{
  "preset": "piano-cuda",
  "epochs": 500,
  "lr": 0.0001
}
```

Priorita: **CLI args > `train.json` > preset soubor**.

---

## Velikosti modelů

| Model | Parametry | GRU hidden | Vrstvy | MLP dim | Použití |
|-------|-----------|------------|--------|---------|---------|
| small | ~598 K | 128 | 1 | 256 | Rychlá diagnostika |
| **medium** | **~2.1 M** | **256** | **2** | **512** | **Výchozí produkční** |
| large | ~4.4 M | 512 | 2 | 512 | Maximální kvalita, GPU doporučeno |

---

## Monitoring tréninku

```bash
# Poslední záznamy z logu
tail -5 C:\SoundBanks\ddsp\ks-grand-ddsp\train.log

# Zkontrolovat běžící proces (Windows)
wmic process where "name='python.exe'" get ProcessId,CommandLine | grep learn
```

Log formát:
```
ep   38  train=1.8339  val=1.8398  lr=2.96e-04  14853.4s  <- best
```

---

## Dokumentace

| Dokument | Obsah |
|----------|-------|
| `PROGRESS.md` | Co se počítalo, výsledky, co dál |
| `docs/ARCHITECTURE.md` | Signálový model, síťová architektura, konstanty |
| `docs/MODEL_ROADMAP.md` | Implementované funkce, roadmap |
| `docs/NPZ_FORMAT.md` | Formát NPZ cache souborů |
| `docs/WORKFLOW_EXAMPLES.md` | Příklady pro různé nástroje |
| `docs/BASS_REFACTOR_CONCEPT.md` | Fyzikální základ dvoukomponentního decay |
