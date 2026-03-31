# DDSP Neural Vocoder — Architektura

DDSP Neural Vocoder je neuronový model, který se naučí timbre monofónního nástroje
z WAV vzorků a syntetizuje nové vzorky podmíněné trojicí (F0, hlasitost, velocity).
Cílový výstup je stereo WAV ve 48 kHz, který se kvalitou přibližuje originálním nahrávkám.

Dokument je určen pro vývojáře, kteří chtějí pochopit vnitřní fungování modelu,
rozšířit architekturu nebo ladit trénování.

Závislosti: PyTorch ≥ 2.0, librosa ≥ 0.10, soundfile ≥ 0.12, gradio ≥ 4.0.

---

## Co je DDSP a proč ho používáme?

DDSP (Differentiable Digital Signal Processing) kombinuje klasické DSP bloky
(harmonické oscilátory, tvarování šumu) s neuronovými sítěmi tak, aby byl celý
systém diferencovatelný a trénovatelný gradientním sestupu.

Oproti čistě neuronovým vocoderům (WaveNet, HiFi-GAN) má DDSP tyto výhody:

- **Fyzikální interpretovatelnost** — harmonické složky odpovídají reálným parciálům
- **Datová efektivita** — stačí desítky sekund zdrojového audia místo hodin
- **Plynulá kontrola výšky** — F0 vstup přímo moduluje fázi oscilátorů bez artefaktů
- **Nízká latence inference** — generování probíhá blokově, ne sample-by-sample

Nevýhody: nezachytí aperiodické přechodové jevy (útok kladívka klavíru) tak dobře
jako čistě neuronové metody.

---

## Signálový model — Decoupled Timbre Architektura

Výstup je plně stereo — každý kanál má nezávislé harmonické amplitudy i šumový profil.
Hlasitostní obálka se aplikuje **post-synthesis** jako lineární multiplikátor:

```
# Piano inharmonicita — frekvence k-tého parciálu
# inh: (B,) — learned B koeficient, MIDI-dependent rozsah
stretch_k = sqrt( 1 + inh * k^2 )              # k = 1 .. N_HARM
f_k(t)    = k * F0(t) * stretch_k              # f_k > k*F0 pro k > 1

# Fázová akumulace (správná pro časově proměnné F0)
phi_k(t) = cumsum( 2*pi * f_k(t) / SR )

# Harmonická distribuce: softmax → součet přes k = 1
d_k_L(t) = softmax( head_harm_L(feat) )
d_k_R(t) = softmax( head_harm_R(feat) )

# Globální amplituda: softplus → vždy > 0
A_L(t) = softplus( head_amp_L(feat) )
A_R(t) = softplus( head_amp_R(feat) )

timbre_L(t) = (1/N) * A_L(t) * SUM_k [ d_k_L(t) * sin(phi_k(t)) ]
            + ISTFT( STFT(white_noise_L) * mag_spectrum_L(t) )

timbre_R(t) = (1/N) * A_R(t) * SUM_k [ d_k_R(t) * sin(phi_k(t)) ]
            + ISTFT( STFT(white_noise_R) * mag_spectrum_R(t) )

lo_linear(t) = 10 ^ (loudness_dB(t) / 20)     # dB → lineární amplituda

L(t) = timbre_L(t) * lo_linear(t)
R(t) = timbre_R(t) * lo_linear(t)

k = 1 .. N_HARM (32 harmonických oscilátorů)
mag_spectrum: N_NOISE (129) spektrálních koeficientů
inh = sigmoid(head_B(feat)).mean(T) * B_MAX(midi) * inh_scale
B_MAX(midi) = 0.0008 * exp(-(midi-21)/88 * ln10)   # A0: 8e-4, C8: 8e-5
```

**Klíčový princip**: síť se učí **pouze timbre** (overtone balance, barvu zvuku) z F0 a velocity.
Hlasitost se do sítě nepodává a neovlivňuje naučené harmonické profily.
Výsledkem je čistá separace:

- **Co se naučí síť**: jak nástroj *zní* (poměr harmonických, šum, stereo panning)
- **Co přichází zvenčí**: jak *hlasitý* je (obálka z EnvelopeNet nebo z NPZ)

**Harmonická distribuce (softmax)**: `d_k` jsou normalizované váhy — jejich součet přes všechny harmonické
je vždy 1. Globální amplituda `A(t)` (softplus) škáluje celkovou hlasitost harmonické složky nezávisle
na timbrovém rozložení. Tato dekompozice (canonical DDSP) vede k lepší podmíněnosti trénování
oproti dřívějšímu přístupu se `sigmoid` (každá harmonická nezávisle, bez normalizace).

**Fázová akumulace (cumsum)**: fáze oscilátorů se počítá kumulativním součtem okamžité frekvence,
nikoli jako `freq × t`. To je správné pro časově proměnné F0 a eliminuje fázové skoky při
interpolaci frekvencí.

Amplitudy `harm_amps_L/R` produkují dvě nezávislé hlavy (`head_harm_L`, `head_harm_R`).
Model se tak z dat naučí, jak se liší overtone balance mezi kanály — například
prostorové rozložení strun klavíru (bas vlevo, výšky vpravo) nebo rozdílný přenos
frekvencí u různých mikrofonních pozic.

---

## Extrakce příznaků

Příznaky jsou počítány z WAV souborů před trénováním a ukládány do NPZ cache.

### F0 (základní frekvence)

Výchozí cesta — **known-F0 mod** (platí pokud název souboru obsahuje MIDI notu `mXXX`):

```
freq = 440.0 * 2^((midi - 69) / 12)       # Hz přímo z MIDI čísla
f0   = np.full(T, freq)                    # konstantní po celou dobu
voiced_prob = RMS_envelope > -60 dB        # klouzavý průměr přes 3 rámce
```

Výhoda: <0.1 s/soubor vs. ~20 s pyin. Extrakce 240 souborů Salamander za ~10 s.

Záložní cesta — **pyin mod** (aktivuje se pokud MIDI nelze parseovat z názvu, nebo je zadán
příznak `--force-pyin`):

Extrakce metodou pYIN (librosa.pyin) s parametry:
- `fmin = 27.5 Hz` (A0)
- `fmax = 5000 Hz`
- `hop_length = 240` vzorků = 5 ms rámec
- `frame_length = 4096` vzorků

Výstup: pole `f0` [Hz] a `voiced_prob` [0–1] na rámec.
Nerámcované části (ticho, šum) mají f0 = 0 a voiced_prob ≈ 0.

### Hlasitost (loudness)

RMS energie v dB per rámec, vypočítaná zvlášť pro levý a pravý kanál:

```
loudness_dB(i) = 20 * log10( sqrt( mean( audio[i*hop:(i+1)*hop]^2 ) + eps ) + eps )
```

Rozsah přibližně -80 dB (ticho) až 0 dB (plná hlasitost).

### Velocity

Pokud název souboru obsahuje velocity ve formátu `mXXX-velY`, použije se přímá
hodnota Y (0–7). Jinak se velocity odhadne dynamicky z loudness percentilů:
vel = 7 * (loudness - p5) / (p95 - p5), ořezané na [0, 7].

---

## Architektura modelu

Model `DDSPVocoder` zpracovává sekvenci rámců (B, T) a produkuje stereo audio (B, 2, T*HOP).

### Enkodéry příznaků (bez parametrů)

Dva sinusoidální enkodéry převádí skalární příznaky na vektory pro timbre síť.
Loudness se do sítě **nepodává** — viz decoupled architektura výše.

| Enkodér | Vstup | Výstup | Dim |
|---------|-------|--------|-----|
| `encode_f0` | F0 v Hz (log-normalizace) | (B, T, F0_BINS) | 64 |
| `encode_velocity` | velocity 0–7 (statická) | (B, VEL_DIM) | 8 |

Velocity je statická pro celou notu a je broadcastována na (B, T, VEL_DIM).
Výsledný vstupní vektor na rámec: 64 + 8 = **72 dimenzí**.

`encode_loudness` je definován v `model/encoders.py` pro případné pozdější rozšíření,
ale není součástí trénovací cesty.

### pre_mlp

Dvouvrstvá MLP předpřipravuje příznaky před GRU:

```
feat (B, T, 72)
  -> Linear(72, mlp_dim) -> ReLU
  -> Linear(mlp_dim, mlp_dim) -> ReLU
  -> (B, T, mlp_dim)
```

### GRU

Rekurentní vrstva zachytí časové závislosti (přechody, vibrato, dozvuk):

```
(B, T, mlp_dim) -> GRU(mlp_dim, gru_hidden, num_layers) -> (B, T, gru_hidden)
```

### post_mlp

Jednovrstvá MLP po GRU rozšíří reprezentaci před výstupními hlavami:

```
(B, T, gru_hidden) -> Linear(gru_hidden, mlp_dim) -> ReLU -> (B, T, mlp_dim)
```

### Výstupní hlavy

Šest lineárních vrstev z `mlp_dim`:

| Hlava | Aktivace | Výstup | Popis |
|-------|----------|--------|-------|
| `head_harm_L` | softmax | (B, T, 32) | Normalizovaná harmonická distribuce — levý kanál |
| `head_harm_R` | softmax | (B, T, 32) | Normalizovaná harmonická distribuce — pravý kanál |
| `head_amp_L` | softplus | (B, T, 1) | Globální amplituda harmonické složky — levý kanál |
| `head_amp_R` | softplus | (B, T, 1) | Globální amplituda harmonické složky — pravý kanál |
| `head_noise_L` | sigmoid | (B, T, 129) | Spektrum šumu — levý kanál |
| `head_noise_R` | sigmoid | (B, T, 129) | Spektrum šumu — pravý kanál |
| `head_B` | sigmoid→pool | (B, 1) | Inharmonicity koeficient B (pooled přes T) |

`head_harm_L/R` produkují **softmax** distribuci (součet = 1), vynásobenou `softplus` amplitudou z `head_amp_L/R`.
Tato canonical DDSP dekompozice odděluje timbre od hlasitosti a vede k lepší podmíněnosti trénování.
`head_B` je poolován přes T (`.mean(dim=1)`) — B je konstantní pro celou notu, aby nevznikly
frekvenční skoky mezi framy.
Bias šumových hlav je inicializován na -3.0 (ticho na začátku trénování).
Bias `head_B` je inicializován na -5.0 (`sigmoid ≈ 0.007` → B ≈ 0 na startu, model se naučí B z dat).
Hlasitost se aplikuje post-synthesis (viz decoupled architektura).

### Syntetizátory

**HarmonicSynth**: upsamplingem převede rámcové amplitudy na vzorkovací frekvenci,
aplikuje Nyquistovu masku (harmonické nad 0.45 * SR jsou vynulovány) a sečte sinusoidy.

**NoiseSynth**: STFT bílého šumu je vynásobena spektrem z `head_noise_L/R`,
ISTFT produkuje tvarovaný šum.

---

## Trénovací smyčka

### Ztráta (loss)

Multi-Resolution STFT Loss (MRSTFT) kombinuje tři FFT velikosti (256, 1024, 4096):

```
loss = mrstft(pred, target) + 0.2 * L1(pred, target)

mrstft = mean over fft_sizes of:
    spectral_convergence = ||mag_pred - mag_target||_F / ||mag_target||_F
  + log_magnitude_L1     = L1( log(mag_pred), log(mag_target) )
```

MRSTFT zachytí jak hrubou spektrální strukturu (velké FFT) tak detaily útoku (malé FFT).
L1 regularizace ve vzorkovém prostoru zabraňuje fázovým artefaktům.

### Optimalizátor a scheduler

- Adam, výchozí lr = 3e-4
- CosineAnnealingLR: lr klesá kosinusově z lr_max na 0.05 * lr_max za T_max epoch
- Gradient clipping: norm ≤ 1.0

### Data augmentace

Při načítání okna (50 rámců = 0.25 s) se s pravděpodobností 0.5 aplikuje
náhodné zesílení ±2 dB, které se promítne i do loudness příznaků.

---

## Adresářová struktura

```
C:\SoundBanks\
  ddsp\
    <nastroj>\                  <- zdrojové WAV soubory (READ-ONLY)
      mXXX-velX-fXX.wav
      instrument-definition.json

  SFZ\
    <nastroj>\                  <- původní SFZ banka (READ-ONLY)
      *.sfz
      *.wav

  IthacaPlayer\
    <nastroj>\                  <- vygenerované vzorky pro IthacaPlayer
      mXXX-velX-fXX.wav
      instrument-definition.json

<nastroj>-ddsp\                 <- workspace (vedlejší data, vedle zdroje)
  extracts\                     <- NPZ cache extrahovaných příznaků
    m060-vel5-f48_chunk000.npz  <- audio + f0 + loudness_L/R + voiced_prob + vel_frames
    ...
  checkpoints\
    best.pt                     <- váhy modelu s nejnižší validační ztrátou
    last.pt                     <- váhy + stav optimalizátoru (pro resume)
  instrument.json               <- konfigurace a stav trénování
  train.log                     <- log epocha po epochě
```

Zdrojový adresář je nikdy nemodifikován.
Workspace lze přepsat parametrem `--workspace <cesta>` — užitečné pokud
zdrojová data změnila umístění, ale extracts/checkpoints zůstaly na místě.

---

## Datový tok

```
C:\SoundBanks\ddsp\<nastroj>\   (zdrojove WAV, READ-ONLY)
        |
        v  [extract_and_cache]
        |  F0: known-f0 (z nazvu mXXX) nebo pyin zaloha
        |  RMS loudness per kanal
        |  velocity z nazvu / dynamicka
        |
        v
NPZ cache (extracts/*.npz)
        |
        v  [SourceDataset]
        |  nahodne okno 50 ramcu = 0.25 s
        |  augmentace hlasitosti +-2 dB
        |
        v
Trenovaci batch (B, T=50)
  f0 [Hz], loudness_L/R [dB], velocity [0-7], audio (2, T*HOP)
        |
        v  [DDSPVocoder.forward]  — Decoupled Timbre Architektura
        |
        |  encode_f0 -> (B, T, 64)
        |  encode_velocity -> (B, T, 8)      ] -> cat -> (B, T, 72)   [loudness NENÍ vstupem]
        |
        |  pre_mlp -> (B, T, mlp_dim)
        |  GRU     -> (B, T, gru_hidden)
        |  post_mlp -> (B, T, mlp_dim)
        |
        |  head_harm_L  -> softmax -> (B, T, 32)  --.
        |  head_amp_L   -> softplus-> (B, T, 1)  --+--> HarmonicSynth(inh) -> timbre_L (B, 1, T*HOP)
        |  head_harm_R  -> softmax -> (B, T, 32)  --.
        |  head_amp_R   -> softplus-> (B, T, 1)  --+--> HarmonicSynth(inh) -> timbre_R (B, 1, T*HOP)
        |  head_B       -> sigmoid -> pool(T) -> (B,) * B_MAX(midi) * inh_scale = inh (B,)
        |  head_noise_L -> sigmoid -> (B, T, 129) -----> NoiseSynth_L  -> (B, 1, T*HOP)
        |  head_noise_R -> sigmoid -> (B, T, 129) -----> NoiseSynth_R  -> (B, 1, T*HOP)
        |
        |  loudness_db (B, T) -> 10^(dB/20) -> upsample -> lo_up (B, 1, T*HOP)
        |
        v
Stereo audio (B, 2, T*HOP)
  L = (timbre_L + noise_L) * lo_up
  R = (timbre_R + noise_R) * lo_up
        |
        v  [mrstft_loss + 0.2 * L1]
        |
Gradient -> Adam -> aktualizace vah
        |
        v  [generate]
        |
C:\SoundBanks\IthacaPlayer\<nastroj>\   (vygenerovane WAV pro IthacaPlayer)
```

---

## Velikosti modelu

| Preset | gru_hidden | gru_layers | mlp_dim | Parametry | Doporucene pouziti |
|--------|-----------|------------|---------|-----------|-------------------|
| `small` | 64 | 1 | 128 | ~113 K | Rychle CPU trenovani, baseline |
| `medium` | 128 | 2 | 256 | ~452 K | Dobry kompromis, doporuceny default |
| `large` | 256 | 3 | 512 | ~1.99 M | Nejlepsi kvalita, pomale na CPU |

Preset se vybira pres CLI: `python ddsp.py learn --model medium --instrument <path>`.
Trenovaci checkpointy (`best.pt`, `last.pt`) obsahuji informaci o velikosti modelu,
takze inference po nacteni checkpointu nepotrebuje znovu specifikovat preset.

---

## Rozsiritelnost

- **Pridani novych enkoderu**: implementuj funkci `encode_X(tensor) -> (B, T, D)` a pricti D do `feat_dim`
- **Zmena syntetizatoru**: `HarmonicSynth` a `NoiseSynth` jsou samostatne `nn.Module`, lze nahradit
- **Vicenasobna F0 (akordy)**: rozsir `head_harm` na (B, T, N_HARM * N_VOICES) a uprav `HarmonicSynth`
- **Vlastni loss**: nahrad `mrstft_loss` v `cmd_learn` libovolnou diferencovatelnou funkci

---

## Coupled vs. Decoupled trénink EnvelopeNet + DDSP

Existují dva přístupy k tréninku, které se liší ve vztahu mezi EnvelopeNet a DDSP modelem.

### Decoupled (výchozí, větev `main`)

```
NPZ extracts ──► DDSP trénink   (loudness z NPZ, reálná dynamika)
NPZ extracts ──► EnvelopeNet trénink  (samostatně, kdykoli)

Inference (full-range):
  EnvelopeNet.predict(midi, vel) → loudness → DDSPVocoder → audio
```

**Výhoda**: jednodušší, rychlejší, DDSP vidí přesnou reálnou dynamiku nahrávek.
**Nevýhoda**: mismatch distribuce — DDSP byl trénován s originální loudness,
ale při full-range generování dostane EnvelopeNet aproximaci (hladší, warped osa).
Tento rozdíl je v praxi malý pro standardní mód (generate ze zdrojových WAV),
ale může ovlivnit full-range mode.

### Coupled (příznak `--coupled`)

```
NPZ extracts ──► EnvelopeNet trénink  (PRVNÍ)
                     │
                     ▼
NPZ extracts ──► DDSP trénink  (--env-mix % batchů: loudness z EnvelopeNet,
                                 zbytek: reálná NPZ loudness)

Inference (full-range):
  EnvelopeNet.predict(midi, vel) → loudness → DDSPVocoder → audio
  ✓ stejná distribuce jako při tréninku
```

**Výhoda**: DDSP model vidí EnvelopeNet loudness i při tréninku → zarovnaná distribuce.
**Výhoda**: `--env-mix 0.5` zachovává 50 % batchů s reálnou loudness → model je robustní pro standardní mód.
**Nevýhoda**: pomalejší start (EnvelopeNet trénink ~minuty před DDSP), mírně složitější pipeline.

### Volba přístupu

| Situace | Doporučení |
|---------|-----------|
| Standardní generování (ze zdrojových WAV) | Decoupled — mismatch nevadí |
| Full-range banka (chromatická, bez zdrojových WAV) | Coupled — lepší konzistence |
| Rychlý experiment / prototyp | Decoupled |
| Produkční banka pro IthacaPlayer | Coupled s `--env-mix 0.5` |

### Použití

```bash
# Decoupled (default)
python ddsp.py learn --instrument C:\SoundBanks\ddsp\salamander --model small --epochs 300

# Coupled
python ddsp.py learn --instrument C:\SoundBanks\ddsp\salamander --model small --epochs 300 \
    --coupled --env-mix 0.5
```

---

## Struktura kódu — framework balíčky

Po refaktorizaci ([dev-bass-refactor / Phase 0]) je kód rozdělen do čitelných balíčků:

```
synth/
  constants.py   — SR, FRAME_HOP, N_HARM_MAX, N_NOISE, F0_BINS, VEL_DIM, MODEL_SIZES
  harmonic.py    — HarmonicSynth: additivní syntéza s inharmonicitou
  noise.py       — NoiseSynth: harmonic-relative STFT tvarování šumu
  __init__.py    — re-exportuje vše

model/
  encoders.py    — encode_f0, encode_velocity, encode_loudness (sinusoidální enkodéry)
  vocoder.py     — DDSPVocoder, build_ddsp, count_params
  envelope.py    — EnvelopeNet (tiny MLP pro predikci loudness obálky)
  __init__.py    — re-exportuje vše

training/
  dataset.py     — SourceDataset s adaptivním crop_frames(midi)
  loss.py        — mrstft_loss, _attack_weight
  __init__.py    — re-exportuje vše

ddsp.py          — CLI vstupní bod (extract / learn / generate / status)
model_ddsp.py    — backward-compat shim (re-exportuje z výše uvedených balíčků)
audio_io.py      — načítání/ukládání WAV, scan_instrument_dir, parse_filename
```

### Adaptivní tréninkové okno (`training/dataset.py`)

Délka trénovacího okna je závislá na MIDI notě:

| MIDI | Nota | Okno | Délka | Důvod |
|------|------|------|-------|-------|
| ≤24  | A0   | 2000 fr | 10 s | 2× τ_slow (zig-zag polarizace) |
| 48   | C3   | 1025 fr | 5 s  | překrývá celý sustain |
| ≥72  | C5   | 50 fr   | 0.25 s | původní chování |

Funkce `crop_frames(midi)` provede lineární interpolaci mezi těmito hodnotami.
