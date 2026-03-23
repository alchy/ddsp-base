# DDSP Neural Vocoder

DDSP Neural Vocoder se naučí timbre monofónního nástroje z WAV vzorků
a syntetizuje nové vzorky podmíněné trojicí (F0, hlasitost, velocity).
Výstupem je stereo WAV banka kompatibilní se samplerovými pluginy (Kontakt, sfz).

Určeno pro: zvukové designéry, výzkumníky a vývojáře, kteří potřebují rychle
vytrénovat timbrový model z existujících nahrávek bez ručního anotování.

Závislosti: Python 3.10+, PyTorch ≥ 2.0, librosa ≥ 0.10, soundfile ≥ 0.12.
GPU není nutné — model `small` trénuje na CPU za přijatelnou dobu.

---

## Jak to funguje?

Vstupem jsou monofónní WAV soubory pojmenované ve formátu `mXXX-velY-f48.wav`
(MIDI nota, velocity vrstva, vzorkovací frekvence). Model se naučí mapovat
trojici (F0, loudness, velocity) na spektrální obálku nástroje.

```
Zdrojove WAV soubory
        |
        v  extract -- pyin F0, RMS loudness, velocity
        |
NPZ cache (extracts/*.npz)
        |
        v  learn -- DDSP trenovani (MRSTFT loss + L1)
        |
best.pt checkpoint
        |
        v  generate -- inference pro kazdy WAV ze zdroje
        |
Generovane WAV soubory (<nastroj>-ddsp/generated/)
```

Signalovy model:

```
audio(t) = SUM_k [ a_k(t) * sin(2*pi * k * F0(t) * t / SR) ]   // harmonicke oscilatory
         + ISTFT( STFT(white_noise) * mag_spectrum(t) )          // tvarovany sum
```

Neuronova sit predikuje amplitudy `a_k` a `mag_spectrum` pro kazdy 5ms ramec.
Podrobna architektura je v `docs/ARCHITECTURE.md`.

---

## Rychly start

Tri prikazy pro prvni pouziti:

```bash
# 1. Instalace zavislosti
pip install -r requirements.txt

# 2. Extrakce priznaku ze zdrojovych WAV
python ddsp.py extract --instrument C:\samples\rhodes

# 3. Trenovani (auto-spusti extrakci pokud chybi)
python ddsp.py learn --instrument C:\samples\rhodes --model small --epochs 100

# 4. Generovani vzorku
python ddsp.py generate --instrument C:\samples\rhodes
```

Vysledek: adresa `C:\samples\rhodes-ddsp\generated\` obsahuje syntezovane WAV soubory
se stejnymi nazvy jako originaly.

> Tip: graficke rozhrani spustite prikazem `python gui.py`. Otevre se v prohlizeci na `http://127.0.0.1:7860`.

---

## Krok za krokem

### 1. Instalace

```bash
pip install -r requirements.txt
```

Na Windows s CUDA (volitelne pro rychlejsi trenovani):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Priprava zdrojovych souboru

Zdrojovy adresar musi obsahovat WAV soubory. Doporuceny format nazvu:

```
m060-vel5-f48.wav     # MIDI 60 (C4), velocity vrstva 5, 48 kHz
m060-vel7-f48.wav     # MIDI 60 (C4), velocity vrstva 7 (forte)
```

Soubory bez velocity v nazvu jsou podporovany — velocity se odhadne dynamicky
z hlasitosti. Podpora formatu MP3, FLAC a OGG je k dispozici pres `extract`.

> **Pozor:** zdrojovy adresar je vzdy jen ke cteni. Vsechny vystupy jdou do `<nastroj>-ddsp/`.

### 3. Extrakce priznaku

```bash
python ddsp.py extract --instrument <adresar> [--chunk-sec 60]
```

Prikaz pro kazdy WAV:
- extrahuje F0 pomoci pYIN (librosa)
- vypocita RMS loudness per kanal per ramec
- ulozi do `<nastroj>-ddsp/extracts/*.npz`

Dlouhe soubory se automaticky rozdeli na chunky (`--chunk-sec`, vychozi: 60 s).
Uz zpracovane soubory se preskoci (lze bezpecne spustit opakovane).

### 4. Trenovani

```bash
python ddsp.py learn --instrument <adresar> [--model small|medium|large] [--epochs 100] [--resume]
```

Trenovani ulozi:
- `best.pt` — vahy s nejnizsi validacni ztrátou
- `last.pt` — posledni stav (pro pokracovani s `--resume`)
- `train.log` — ztrata po epochach

Priblizna doba trenovani modelu `small` na CPU (100 epoch, 60s audia):

| Hardware | Cas |
|----------|-----|
| CPU (8 jader) | 15–30 min |
| GPU RTX 3080 | 2–5 min |

### 5. Generovani

```bash
python ddsp.py generate --instrument <adresar> [--wet 1.0] [--notes C4 A3] [--vel 5 7]
```

Pro kazdy zdrojovy WAV:
1. nacte referencni loudness obálku
2. spusti inference DDSPVocoder
3. ulozi WAV do `<nastroj>-ddsp/generated/`

Parametr `--wet` (0.0–1.0) richa michani: `1.0` = plny DDSP, `0.0` = original.
Filtry `--notes` a `--vel` omezi generovani na podmnozinu vzorku.

### 6. Kontrola stavu

```bash
python ddsp.py status --instrument <adresar>
```

Vypise:
- pocet zdrojovych WAV souboru
- pocet NPZ v cache
- stav checkpointu (velikost modelu, epochy, best_val)
- pocet generovanych souboru

---

## Graficke rozhrani

```bash
python gui.py [--port 7860] [--share]
```

Otevre Gradio aplikaci v prohlizeci. Obsahuje zalozky:
- **Nastroj & Stav** — zadani cesty a zobrazeni stavu
- **Extrakce** — spusteni extrakce s nastavenim chunk-sec
- **Uceni** — vyber velikosti modelu, poctu epoch, learning rate a resume
- **Generovani** — filtrovani not a velocity, nastaveni wet/dry

Log se aktualizuje kazdé 2 sekundy. Tlacitko Stop prerusi beh prikazu.

---

## Reference

### Prikazy CLI

| Prikaz | Popis |
|--------|-------|
| `extract` | Extrahuje a cachuje priznaky ze zdrojovych WAV |
| `learn` | Trenuje model (auto-spusti extract pokud chybi NPZ) |
| `generate` | Generuje WAV banka z nauceneho modelu |
| `status` | Zobrazi stav nastroje |

### Parametry `learn`

| Parametr | Vychozi | Popis |
|----------|---------|-------|
| `--model` | `small` | Velikost modelu: `small`, `medium`, `large` |
| `--epochs` | `100` | Pocet trenicich epoch |
| `--lr` | `3e-4` | Learning rate (Adam) |
| `--batch-size` | `4` | Velikost batch |
| `--resume` | off | Pokracovat od `last.pt` |
| `--min-voiced` | `0.1` | Min. podil voiced ramcu v okne |

### Parametry `generate`

| Parametr | Vychozi | Popis |
|----------|---------|-------|
| `--wet` | `1.0` | Pomer DDSP / original (0.0–1.0) |
| `--notes` | vse | Filtr not, napr. `C4 A3 G3` |
| `--vel` | vse | Filtr velocity vrstev, napr. `5 7` |
| `--output` | `<nastroj>-ddsp/generated/` | Vystupni adresar |
| `--no-skip` | off | Prepsat existujici soubory |

### Velikosti modelu

| Preset | Parametry | gru_hidden | Vrstvy GRU | mlp_dim |
|--------|-----------|-----------|------------|---------|
| `small` | ~84 K | 64 | 1 | 128 |
| `medium` | ~400 K | 128 | 2 | 256 |
| `large` | ~1.9 M | 256 | 3 | 512 |

### Format nazvu souboru

```
mXXX-velY-fZZ.wav
^^^  ^^^  ^^^
|    |    +-- vzorkovaci frekvence: f48 = 48 kHz, f44 = 44.1 kHz
|    +------- velocity vrstva: 0–7 (0 = nejticejsi, 7 = nejhlasitejsi)
+------------ MIDI cislo noty: 000–127 (060 = C4, 069 = A4)
```

Soubory bez tohoto formatu jsou podporovany — velocity se odhadne automaticky.

### Struktura workspace

```
<nastroj>-ddsp/
  extracts/          NPZ cache (audio + f0 + loudness_L/R + voiced_prob + vel_frames)
  checkpoints/       best.pt, last.pt
  generated/         vystupni WAV soubory
  instrument.json    konfigurace a stav
  train.log          log trenovani
```
