# NPZ Cache — formát a obsah

Každý zdrojový WAV soubor se při extrakci (`ddsp.py extract`) převede na NPZ soubor
uložený do `<nastroj>-ddsp/extracts/`. Tyto soubory jsou vstupem pro trénování.

---

## Obsah jednoho NPZ souboru

Příklad: `m030-vel6-f48.npz` (nota B1, velocity vrstva 6, 48 kHz)

| Klíč | Shape | Dtype | Popis |
|------|-------|-------|-------|
| `audio` | `(2, N_samples)` | float32 | Stereo audio — L a R kanál; vzorky normalizované na −1..+1 |
| `f0` | `(T,)` | float32 | Základní frekvence v Hz per 5ms rámec |
| `loudness_L` | `(T,)` | float32 | RMS hlasitost levého kanálu v dB per rámec (rozsah −80..0 dB) |
| `loudness_R` | `(T,)` | float32 | RMS hlasitost pravého kanálu v dB per rámec |
| `voiced_prob` | `(T,)` | float32 | "Znělost" per rámec — hodnoty 0 nebo 1, průměr klouzavý přes 3 rámce |
| `vel_frames` | `(T,)` | float32 | Velocity layer per rámec — konstantní hodnota 0–7 z názvu souboru |

`T` = počet 5ms rámců = `N_samples / 240`

---

## Jak se f0 a voiced_prob počítají

### Known-F0 mod (výchozí, rychlý)

Pokud název souboru obsahuje MIDI číslo (`mXXX`):

```
f0[t]          = 440 × 2^((midi − 69) / 12)    # Hz, konstantní pro celý soubor
voiced_prob[t] = klouzavý průměr( RMS_mono[t] > −60 dB )   # 0 nebo 1
```

Výhoda: < 0.1 s/soubor. `f0` je přesné (vychází z fyziky ladění), `voiced_prob`
odpovídá skutečné přítomnosti zvuku odvozené z hlasitosti.

### pyin mod (záložní, pomalý)

Pokud MIDI nelze parsovat z názvu, nebo je zadán `--force-pyin`:

```
f0, voiced_prob = librosa.pyin(audio_mono,
    fmin=27.5, fmax=5000, hop_length=240, frame_length=4096)
```

Výhoda: zachytí vibrato a nezpůsobuje konstantní f0 pro melodické nástroje.
Nevýhoda: ~20 s/soubor.

---

## Délka záznamu a dozvuk

Délka NPZ odpovídá délce originálního WAV souboru — **nekrátí se**. Pro klavírní tóny
to znamená, že soubor obsahuje celý přirozený dozvuk:

| Nota | f0 [Hz] | Délka záznamu |
|------|---------|---------------|
| A0 (MIDI 21) | 27.5 Hz | ~21 s |
| B1 (MIDI 30) | 61.7 Hz | ~23 s |
| C4 (MIDI 60) | 261.6 Hz | ~10 s |
| C8 (MIDI 108) | 4186 Hz | ~4 s |

Basové tóny mají výrazně delší dozvuk než výšky — model se to naučí
a EnvelopeNet tuto závislost interpoluje pro chromatické noty.

---

## Velikost v paměti

Salamander (240 souborů):

| Statistika | Hodnota |
|------------|---------|
| Minimum | 1.0 MB / soubor (krátké výšky) |
| Maximum | 10.0 MB / soubor (dlouhé basy) |
| Průměr | 5.3 MB / soubor |
| **Celkem v RAM** | **~1.25 GB** |

SourceDataset načítá všechny NPZ při inicializaci do RAM (`self.data`).
Při trénování dvou nástrojů současně může být celková spotřeba RAM 2–5 GB.

---

## Jak se NPZ používá v pipeline

```
NPZ soubor
  │
  ├─► Trénování (SourceDataset.__getitem__)
  │     audio         → target pro MRSTFT loss (porovnání se synth výstupem)
  │     f0            → vstup do DDSPVocoder (řídí frekvence oscilátorů)
  │     loudness_L/R  → vstup do DDSPVocoder (řídí amplitudy)
  │     vel_frames    → vstup do DDSPVocoder (velocity conditioning)
  │     voiced_prob   → filtr oken (min. podíl znělých rámců = 0.1)
  │
  ├─► EnvelopeNet trénování
  │     loudness_L/R  → tvar obálky (resampling na 256 bodů)
  │     délka         → dur_s = T × FRAME_HOP / SR
  │
  └─► generate --full-range (fallback bez envelope.pt)
        loudness_L/R  → šablona obálky pro nejbližší (midi, vel)
```

---

## Nástroje pro inspekci

```python
import numpy as np

f = np.load('C:/SoundBanks/ddsp/salamander-ddsp/extracts/m060-vel4-f48.npz')
print(f.files)                    # seznam klíčů
print(f['audio'].shape)           # (2, N_samples)
print(f['f0'][:5])                # prvních 5 hodnot f0
print(f['loudness_L'].min())      # minimální hlasitost (tiché části)
print(len(f['f0']) * 240 / 48000) # délka záznamu v sekundách
```
