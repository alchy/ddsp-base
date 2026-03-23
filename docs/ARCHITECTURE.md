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

## Signálový model

Výstup je plně stereo — každý kanál má nezávislé harmonické amplitudy i šumový profil:

```
L(t) = overall_amp(t) * SUM_k [ a_k_L(t) * sin(2*pi * k * F0(t) * t / SR) ]
     + ISTFT( STFT(white_noise_L) * mag_spectrum_L(t) )

R(t) = overall_amp(t) * SUM_k [ a_k_R(t) * sin(2*pi * k * F0(t) * t / SR) ]
     + ISTFT( STFT(white_noise_R) * mag_spectrum_R(t) )

k = 1 .. N_HARM (32 harmonických oscilátorů)
mag_spectrum: N_NOISE (129) spektrálních koeficientů
```

Amplitudy `a_k_L(t)`, `a_k_R(t)` produkují dvě nezávislé hlavy (`head_harm_L`, `head_harm_R`).
Model se tak z dat naučí, jak se liší overtone balance mezi kanály — například
prostorové rozložení strun klavíru (bas vlevo, výšky vpravo) nebo rozdílný přenos
frekvencí u různých mikrofonních pozic. `overall_amp` je sdílená obálka pro oba kanály.

---

## Extrakce příznaků

Příznaky jsou počítány z WAV souborů před trénováním a ukládány do NPZ cache.

### F0 (základní frekvence)

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

Tři sinusoidální enkodéry převádí skalární příznaky na vektory:

| Enkodér | Vstup | Výstup | Dim |
|---------|-------|--------|-----|
| `encode_f0` | F0 v Hz (log-normalizace) | (B, T, F0_BINS) | 64 |
| `encode_loudness` | loudness dB (lineár. norm.) | (B, T, LO_DIM) | 16 |
| `encode_velocity` | velocity 0–7 (statická) | (B, VEL_DIM) | 8 |

Velocity je statická pro celou notu a je broadcastována na (B, T, VEL_DIM).
Výsledný vstupní vektor na rámec: 64 + 16 + 8 = **88 dimenzí**.

### pre_mlp

Dvouvrstvá MLP předpřipravuje příznaky před GRU:

```
feat (B, T, 88)
  -> Linear(88, mlp_dim) -> ReLU
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

Pět lineárních vrstev z `mlp_dim`:

| Hlava | Aktivace | Výstup | Popis |
|-------|----------|--------|-------|
| `head_harm_L` | sigmoid | (B, T, 32) | Per-harmonické amplitudy a_k — levý kanál |
| `head_harm_R` | sigmoid | (B, T, 32) | Per-harmonické amplitudy a_k — pravý kanál |
| `head_amp` | softplus | (B, T, 1) | Celková obálka overall_amp (sdílená) |
| `head_noise_L` | sigmoid | (B, T, 129) | Spektrum šumu — levý kanál |
| `head_noise_R` | sigmoid | (B, T, 129) | Spektrum šumu — pravý kanál |

`head_harm_L` a `head_harm_R` jsou zcela nezávislé — model se naučí jiný overtone
profil pro každý kanál podle toho, co data obsahují (stereo rozložení, rozdíly mikrofonů).
Bias šumových hlav je inicializován na -3.0 (ticho na začátku trénování).

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
        |  pyin F0 extrakce
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
        v  [DDSPVocoder.forward]
        |
        |  encode_f0 -> (B, T, 64)
        |  encode_loudness -> (B, T, 16)     ]
        |  encode_velocity -> (B, T, 8)      ] -> cat -> (B, T, 88)
        |
        |  pre_mlp -> (B, T, mlp_dim)
        |  GRU     -> (B, T, gru_hidden)
        |  post_mlp -> (B, T, mlp_dim)
        |
        |  head_harm_L  -> sigmoid -> (B, T, 32)  --.--> HarmonicSynth -> harmonic_L (B, 1, T*HOP)
        |  head_harm_R  -> sigmoid -> (B, T, 32)  --'--> HarmonicSynth -> harmonic_R (B, 1, T*HOP)
        |  head_amp     -> softplus -> (B, T, 1)  -----> (sdilena obalka pro oba kanaly)
        |  head_noise_L -> sigmoid -> (B, T, 129) -----> NoiseSynth_L  -> (B, 1, T*HOP)
        |  head_noise_R -> sigmoid -> (B, T, 129) -----> NoiseSynth_R  -> (B, 1, T*HOP)
        |
        v
Stereo audio (B, 2, T*HOP)
  L = harmonic_L + noise_L
  R = harmonic_R + noise_R
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
| `small` | 64 | 1 | 128 | ~115 K | Rychle CPU trenovani, baseline |
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
