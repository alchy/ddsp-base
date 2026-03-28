# Model Roadmap

Živý dokument. Zachycuje architektonické směry, jejich motivaci a stav.
Pořadí odpovídá skutečné prioritě — ne abecedě ani historii vzniku.

---

## Implementováno

### [dev-softmax] Fázová akumulace + softmax harmonická distribuce

**Problém**: fáze počítána jako `mean_freq × t` (chybné pro časově proměnné F0);
`sigmoid` na harmonických amplitudách bez normalizace.

**Řešení**:
```python
phase     = cumsum(2π · inst_freq / SR)          # správná akumulace
harm_dist = softmax(head_harm)                    # normalizovaná distribuce
harm_amp  = softplus(head_amp)                    # globální amplituda
harm_amps = harm_dist * harm_amp
```

**Výsledek**: pitch správně, rychlejší konvergence, val loss ~1.13 vs ~1.49 (small model, ep 13).

---

### [dev-inharmonicity] Piano inharmonicita strun

**Problém**: model předpokládal `f_k = k·f0`. Reálné piano má vyšší parciály výše:
```
f_k = k · f0 · √(1 + B · k²)
```
Bez inharmonicity parciály „splývají" — basy znějí jako zvon místo jako klavír.

**Implementace**: learned scalar B per nota, poolovaný přes T (konstantní B pro celou notu):

```python
# DDSPVocoder.__init__
self.head_B = nn.Linear(mlp_dim, 1)
nn.init.constant_(self.head_B.bias, -5.0)   # sigmoid ≈ 0.007 → B ≈ 0 na startu

# DDSPVocoder.forward
f0_mean  = f0_hz.clamp(min=F0_MIN).mean(dim=1)
midi_est = 69.0 + 12.0 * torch.log2(f0_mean / 440.0)
b_max    = 0.0008 * torch.exp(-(midi_est - 21.0) / 88.0 * math.log(10.0))
inh      = torch.sigmoid(self.head_B(feat)).squeeze(-1).mean(dim=1) * b_max * inh_scale

# HarmonicSynth.forward — inh: (B,)
stretch   = torch.sqrt(1.0 + inh.unsqueeze(1) * k**2)   # (B, N)
inst_freq = f0_up.unsqueeze(1) * (k * stretch).unsqueeze(2)
```

**MIDI-dependent B_MAX**: `B_0 · exp(-(midi-21)/88·ln10)` — A0: B_MAX ≈ 0.0008, C8: B_MAX ≈ 0.00008.
Basy mají řádově vyšší inharmonicitu než výšky.

**inh_scale**: parametr inference (0–2), výchozí 1.0.
- 0.0 = čistě harmonické (diagnostický mód, bez inharmonicity)
- 1.0 = naučená hodnota B (fyzikálně věrné)
- 2.0 = zesílená inharmonicita

CLI: `--inharmonicity-scale`, GUI: slider „Inharmonicity scale" (0.0–2.0).

**+129 parametrů** (head_B: `mlp_dim → 1`).

---

## Implementováno (pokračování)

### [dev-noise-fft] NOISE_FFT 256 → 1024 + globální noise amplitude + MRSTFT 16384

**Problém**: `NOISE_FFT = 256` dává rozlišení 187.5 Hz/bin. Pro A0 (27.5 Hz) leží
celé basové spektrum v prvních 1–2 binech ze 129 → model nemůže naučit tvar šumové
textury v basech. Výsledek: "filtrovaný/tlumený" zvuk v nízkých rejstřících,
zatímco transpozice vyšších not do basu zní čistěji — potvrzeno poslechem D4→D2.

**Řešení** (tři změny v jednom branchi):

**B — NOISE_FFT: 256 → 1024** (model_ddsp.py):
- `N_NOISE = 513` místo 129, rozlišení 46.9 Hz/bin (4× lepší)
- A0: h1–h6 leží v binech 0–3 místo binu 0; model má prostor naučit texturní tvar
- Nekompatibilní s předchozími checkpointy (head_noise L/R mají jiný výstup)

**C — noise_amp globální scalar** (model_ddsp.py):
- `head_noise_amp_L/R: mlp_dim → 1` (softplus), analogie s `head_amp_L/R` u harmonických
- Oddělen tvar spektra (sigmoid, 513 binů) od celkové hlasitosti šumu (softplus, 1 hodnota)
- Init bias = -3.0 → tiché na začátku, model se naučí zvyšovat dle potřeby

**A — MRSTFT fft_sizes přidán 16384** (ddsp.py):
- `fft_sizes=(256, 1024, 4096, 16384)` → 16384 dá 2.9 Hz/bin
- A0 (27.5 Hz) leží v binu 9 místo binu 0 → loss konečně vidí basové harmonické

**D — Harmonic-relative noise warping** (model_ddsp.py):

**Problém**: `NoiseSynth` generoval spektrální tvar v **absolutních Hz binech**
(bin 0 = 0 Hz, bin 1 = 46.9 Hz, …). Model se musel naučit úplně jiné hodnoty
pro každou výšku noty — peak šumu na 523 Hz pro C4 se při transpozici na C5
nepřesune na 1046 Hz automaticky. Noise spektrum nesledovalo f0.

Důsledek: noise přidával energii na pevných absolutních frekvencích, které
nesedí na harmonické jiné noty → vnímaný pitch detunovaný, "posunutý" zvuk.

**Řešení**: `NoiseSynth.forward()` přijímá `f0_hz` a warpuje spektrální tvar
z harmonic-relative prostoru na absolutní STFT biny:

```
Noise bin k  →  absolutní freq  =  k · f0 · N_HARM / N
STFT bin i   →  absolutní freq  =  i · SR / n_fft

Pro STFT bin i:  zdrojový noise bin  =  i · (SR/n_fft) / f0 · N / N_HARM
```

Model se učí spektrální tvar **vždy relativně k f0** — "peak mezi 2. a 3.
harmonikem" se automaticky škáluje pro libovolnou výšku noty.

```python
# NoiseSynth.forward (zjednodušeno)
f0_mean    = f0_hz.clamp(min=F0_MIN).mean(dim=1)              # (B,)
stft_freqs = torch.arange(n_bins) * (SR / n_fft)             # (n_bins,)
src_pos    = stft_freqs / f0_mean * (N / N_HARM)             # (B, n_bins)
src_pos    = src_pos.clamp(0, N - 1)
# bilineární interpolace mag_up na pozicích src_pos → mag_abs
out = istft(stft * mag_abs, ...)
```

**Pokrytí noise binů** (N=513, N_HARM=128 po dev-adaptive-nharm):

| Nota | f0 | Noise bins pokrývají do |
|------|----|------------------------|
| A0   | 27.5 Hz | 3 520 Hz (= N_HARM × f0) → zbytek flat |
| A2   | 110 Hz  | 14 080 Hz |
| A4   | 440 Hz  | Nyquist |

Biny nad `N_HARM × f0` jsou clamped na poslední bin → šum je tam konstantní,
ale harmonická struktura je zachycena správně v pokrytém pásmu.

---

### [dev-adaptive-nharm] Adaptivní počet harmonických — konec "muddy/LPF" basů

**Problém**: `N_HARM = 32` nastavoval pevný strop pro všechny noty. Výsledkem bylo,
že harmonický synth pokrýval jen zlomek slyšitelného spektra basových not:

| Nota | f0 | N_HARM=32 pokryto do | Zbytek spektra |
|------|----|----------------------|----------------|
| A0   | 27.5 Hz | **880 Hz**    | 23 kHz = jen noise |
| A1   | 55 Hz   | **1 760 Hz**  | 22 kHz = jen noise |
| A2   | 110 Hz  | **3 520 Hz**  | 20 kHz = jen noise |
| A4   | 440 Hz  | 14 080 Hz     | OK |

Noise synth nad mezní frekvencí generuje filtrovaný šum bez harmonické struktury →
basy znějí jako klavír za LPF filtrem, "muddy", bez "znělosti" v horním pásmu.

**Řešení**: `N_HARM_MAX = 128`, počet aktivních harmonických adaptivní per batch item:

```python
# HarmonicSynth.forward
f0_mean  = f0_up.mean(dim=-1)                                   # (B,)
n_active = ((k.unsqueeze(0) * f0_mean.unsqueeze(1)) < SR * 0.45) \
           .sum(dim=1).clamp(min=1).float()                     # (B,)
...
return signal / n_active.view(B, 1, 1)   # normalizace n_active, ne pevným N
```

Normalizace opravena z pevného `/N` na `/n_active` — hlasitost konzistentní bez ohledu
na počet aktivních harmonických.

**Výsledek po zvýšení N_HARM_MAX=32 → 128**:

| Nota | f0 | n_active | Pokryto do | Zlepšení |
|------|----|----------|-----------|----------|
| A0   | 27.5 Hz | 128 | **3 520 Hz** | +4× |
| A1   | 55 Hz   | 128 | **7 040 Hz** | +4× |
| A2   | 110 Hz  | 109 | **11 925 Hz** | skoro plné |
| A3   | 220 Hz  |  54 | plné (nyq.)  | = |
| A4+  | ≥440 Hz |  ≤54| plné (nyq.)  | beze změny |

Treble noty nejsou dotčeny — nyquist maska přirozeně omezí n_active.

**Možné zvýšení pokud A0–A1 stále problematické po poslechu**:

| N_HARM_MAX | A0 pokryto | Paměť tensoru (B=16, crop=200 fr) |
|-----------|------------|-----------------------------------|
| 128        | 3 520 Hz  | ~196 MB — **aktuální** |
| 256        | 7 040 Hz  | ~393 MB — doporučený next step |
| 512        | 14 080 Hz | ~786 MB — hranice CPU RAM |
| 872 (full) | 24 000 Hz | ~1.3 GB — pouze GPU |

Zvýšení na **256 je přirozený next step** po poslechu výsledků s 128.
Provést změnou jediné konstanty: `N_HARM_MAX = 256` v `model_ddsp.py` + re-trénink.

**Kombinace s [dev-noise-fft]**: harmonic-relative noise warping zajistí, že noise synth
pokrývá zbytek spektra nad `N_HARM_MAX × f0` ve správném harmonickém kontextu.

---

## Implementováno (pokračování)

### [dev-bass-refactor / Phase 0] Framework restructure + adaptivní tréninkové okno

**Motivace (z analýzy 18 physics papers)**:
Diagnóza zjistila 6 příčin problematického zvuku basů. Nejzávažnější je
**tréninkové okno příliš krátké na zachycení basového doznívání**:
- CROP_FRAMES = 50 → 250 ms
- A0 (MIDI 21): délka samplu ~29 s, τ_slow ≈ 5 s
- Model vidí méně než 1 % délky noty — nikdy nepozoruje pomalé doznívání

**Provedené změny**:

1. **Framework restructure** — čistá reorganizace, žádná logická změna:
   ```
   synth/    constants.py, harmonic.py, noise.py, __init__.py
   model/    vocoder.py, encoders.py, envelope.py, __init__.py
   training/ dataset.py, loss.py, __init__.py
   model_ddsp.py → backward-compat shim (re-exportuje z frameworku)
   ```

2. **Adaptivní `crop_frames(midi)`** (`training/dataset.py`):
   ```python
   def crop_frames(midi: int) -> int:
       t = (midi - 24) / (72 - 24)          # 0.0 = A0-ish,  1.0 = C5
       t = max(0.0, min(1.0, t))
       return int(round(2000 + t * (50 - 2000)))
   # MIDI 21  → 2000 fr = 10 s  (2× τ_slow)
   # MIDI 48  → 1025 fr = 5 s
   # MIDI 72+ →   50 fr = 0.25 s (původní chování)
   ```

---

## Roadmap — [dev-bass-refactor]

Pořadí fází odpovídá přínosu a závislosti. Každá fáze je testovatelná samostatně.

---

### Phase 1. [dev-bass-refactor] Per-parciální σ_j decay

**Zdroj**: Simionato 2024, Bensa 2003 — freq-dependent decay rate ověřený ve studii.

**Fyzika**: každý parciál j doznívá s vlastní rychlostí:
```
σ_j = b1 + b3 · ωj²        # ωj = 2π · f_j,  b1 ≈ 0.3, b3 ≈ 1e-7 (Steinway D)
A_j(t) = A_j(0) · exp(-σ_j · t)
```
Vyšší parciály mizí rychleji → přirozené "otevírání a zavírání" spektra při doznívání.
Bez toho: všechny parciály doznívají stejně → zvuk znít umele staticky.

**Nové hlavy v DDSPVocoder**:
```python
self.head_b1 = nn.Linear(mlp_dim, 1)    # baseline decay (sdílený L+R)
self.head_b3 = nn.Linear(mlp_dim, 1)    # freq-dep. koefficient (sdílený L+R)
nn.init.constant_(self.head_b1.bias, -2.0)   # softplus → ~0.13 na startu
nn.init.constant_(self.head_b3.bias, -16.0)  # softplus → ~1e-7 na startu
```

**HarmonicSynth.forward** — per-parciální amplituda s decay envelope:
```python
b1 = F.softplus(head_b1(feat)).mean(dim=1)   # (B,)
b3 = F.softplus(head_b3(feat)).mean(dim=1)   # (B,)
f_j = f0_mean.unsqueeze(1) * k               # (B, N)  Hz
omega_j = 2 * pi * f_j
sigma_j = b1.unsqueeze(1) + b3.unsqueeze(1) * omega_j**2  # (B, N)
t_frames = torch.arange(T, device) * FRAME_HOP / SR       # (T,)
decay_env = torch.exp(-sigma_j.unsqueeze(2) * t_frames)   # (B, N, T)
harm_amps_decayed = harm_amps.unsqueeze(2) * decay_env     # (B, N, T)
```

**Kompatibilita**: nové checkpointy (head_b1/b3 přidány). Starý checkpoint nelze načíst.

---

### Phase 2. [dev-bass-refactor] AM beating — zig-zag polarizace strun

**Zdroj**: Weinreich 1977, Hall 1986, Conklin 1990 — zig-zag polarizace ověřena měřením.

**Fyzika**: každá struna vibruje ve dvou polarizacích (vertikální + horizontální).
- Vertikální (rychlejší doznívání): τ_fast ≈ 0.5–2 s
- Horizontální (pomalejší doznívání): τ_slow ≈ 3–8 s

Výsledek = AM modulace na každém parciálu j s beat rate 1–5 Hz:
```
A_j(t) = a_fast_j · exp(-t/τ_fast) + a_slow_j · exp(-t/τ_slow)
```
Nikoli frekvenční detuning (to je inharmonicita) — ale amplitudová modulace.

**Nové hlavy**:
```python
self.head_am_j     = nn.Linear(mlp_dim, 1)   # AM hloubka (α_j)
self.head_am_rate  = nn.Linear(mlp_dim, 1)   # beat rate Hz (1–5 Hz)
self.head_tau_fast = nn.Linear(mlp_dim, 1)   # τ_fast v sekundách
self.head_tau_slow = nn.Linear(mlp_dim, 1)   # τ_slow v sekundách
```

**Požadavek na okno**: τ_slow = 5 s → nutné ≥ 2000 framů (10 s okno z Phase 0).
Bez Phase 0 nelze Phase 2 natrénovat — závislost je tedy pevná.

---

### Phase 3. [dev-bass-refactor] Phantom partials (geometrická nelinearita)

**Zdroj**: Conklin 1996, Chaigne & Kergomard — podélné vlny + geometrická nelinearita.

**Fyzika**: velké výchylky struna (basy) způsobují podélné vlny.
- Podélná vlna f_L ≈ 28 × f_T (transverzální) → krátký burst na začátku
- Geometrická nelinearita generuje phantom partials: 2·f_j a f_i ± f_j
- Tyto frekvence nejsou v harmonické řadě → NoiseSynth je aktuálně pohltí jako šum

**Implementace**: přidat N_PHANTOM = 8–16 dedikovaných oscilátorů:
```python
self.head_phantom   = nn.Linear(mlp_dim, N_PHANTOM)   # magnitudy
self.head_phantom_f = nn.Linear(mlp_dim, N_PHANTOM)   # relative frekvence (2·k)
```

**Priorita**: nižší než Phase 1 + 2. Phantom partials jsou perceptuálně viditelné
hlavně v hlubokých basech A0–A2, po opravě decay a AM to bude otázka doladění.

---

### Phase 4. [dev-frame-rate] Adaptivní frame rate (čeká na M5)

**Proč odloženo**: FRAME_HOP = 240 samples = 5 ms dává C7 attacku 2 framy.
Na CPU je to neúnosné — implementace by stála NPZ re-extrakci + 4× více výpočtu.

**Správné okno**: příchod M5. Pak implementovat jako první velkou změnu.

```python
# dL/dt > 0 (attack)  → FRAME_HOP_ATTACK  = 60 samples (1.25 ms)
# dL/dt ≤ 0 (sustain) → FRAME_HOP_SUSTAIN = 240 samples (5 ms)
```

---

## Poznámky

- Phase 0 (framework + adaptivní okno): **implementováno** na dev-bass-refactor
- Phase 1 (σ_j decay): nové checkpointy; testovat small model 50 epoch
- Phase 2 (AM beating): závislé na Phase 0 (potřebuje dlouhá okna)
- Phase 3 (phantom partials): nezávislé, doladění po Phase 1+2
- Phase 4 (frame rate): čeká na M5, největší pipeline změna projektu
- Testovat vždy small model (50 epoch) před medium (300 epoch)
