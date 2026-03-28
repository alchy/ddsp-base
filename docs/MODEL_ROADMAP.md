# Model Roadmap

Živý dokument. Zachycuje architektonické směry, jejich motivaci a stav.
Pořadí v sekci Roadmap odpovídá skutečné prioritě.

---

## Implementováno

### [dev-softmax] Fázová akumulace + softmax harmonická distribuce

**Problém**: fáze počítána jako `mean_freq × t` (chybné pro časově proměnné F0);
`sigmoid` na harmonických amplitudách bez normalizace.

**Řešení**: cumsum fázová akumulace, softmax distribuce × softplus globální amplituda.

**Výsledek**: pitch správně, rychlejší konvergence, val loss ~1.13 vs ~1.49 (small, ep 13).

---

### [dev-inharmonicity] Piano inharmonicita strun

**Zdroj**: standardní fyzika pianových strun.

**Fyzika**: `f_k = k · f0 · √(1 + B · k²)` — vyšší parciály leží výše než harmonická řada.
Bez inharmonicity basy znějí jako zvon místo klavíru.

**Implementace**: learned scalar B per nota (pooled over T), MIDI-dependent B_MAX:
```
b_max = 0.0008 · exp(-(midi−21)/88·ln10)   # A0: 8×10⁻⁴, C8: 8×10⁻⁵
inh   = sigmoid(head_B) · b_max · inh_scale
```
Inference: `--inharmonicity-scale` (0–2), default 1.0.

---

### [dev-noise-fft] Noise FFT 256→1024 + harmonic-relative warping + MRSTFT 16384

**Problémy**:
- `NOISE_FFT=256` → 187.5 Hz/bin; A0 (27.5 Hz) leží celé v prvních 1–2 binech
- NoiseSynth tvaroval šum v absolutních Hz binech → nesedí při transpozici not

**Řešení**:
- `NOISE_FFT=1024` → 46.9 Hz/bin (4× lepší rozlišení pro basy)
- Noise tvar učí se v **harmonicko-relativním prostoru** (bin k → relativní frekvence k × f0)
- `fft_sizes` pro MRSTFT rozšířeny o 16384 → 2.9 Hz/bin pro basové harmonické
- Globální noise amplitude scalar (oddělený tvar spektra od celkové hlasitosti)

---

### [dev-adaptive-nharm] Adaptivní počet harmonických

**Problém**: pevné N_HARM=32 pokrývalo A0 jen do 880 Hz (37 % nyquist).
Zbytek spektra vyplňoval NoiseSynth bez harmonické struktury → "muddy" zvuk.

**Řešení**: `N_HARM_MAX=128`, n_active = min(N_HARM_MAX, floor(nyquist/f0)):
- A0: 128 harmonických → pokryto do 3 520 Hz
- A3+: n_active přirozeně omezeno nyquist maskou (beze změny chování treble)
- Normalizace `/n_active` místo `/N` — konzistentní hlasitost

---

### [dev-bass-refactor / Phase 0] Framework restructure + adaptivní tréninkové okno

**Zdroj**: analýza 18 physics papers (Weinreich, Bensa, Conklin, Simionato, Chaigne…).

**Diagnnóza**: CROP_FRAMES=50 → 250 ms; A0 sampel ~29 s, τ_slow ≈ 5 s.
Model viděl < 1 % délky noty — nikdy nepozoroval doznívání.

**Provedené změny**:

1. **Framework restructure** (čistá reorganizace, žádná logická změna):
   ```
   synth/    — HarmonicSynth, NoiseSynth, konstanty
   model/    — DDSPVocoder, EnvelopeNet, enkodéry
   training/ — SourceDataset, mrstft_loss
   ```
   `model_ddsp.py` smazán.

2. **Adaptivní `crop_frames(midi)`** (`training/dataset.py`):
   ```
   MIDI 21  → 2000 fr = 10 s  (2× τ_slow, zachytí celé doznívání)
   MIDI 48  → 1025 fr = 5 s
   MIDI 72+ →   50 fr = 0.25 s  (původní chování)
   ```

---

### [dev-bass-refactor / Phase 1] Per-parciální fyzikální decay σ_k

**Zdroj**: Simionato 2024 (měření Steinway D), Bensa 2003.

**Fyzika**: každý parciál k doznívá s vlastní rychlostí závislou na frekvenci:
```
σ_k = b1 + b3 · (2π · k · f0)²
```
Vyšší parciály doznívají rychleji → přirozené "uzavírání" spektra v průběhu sustainu.

**Implementace**: HarmonicSynth.forward() aplikuje decay na frame rate (memory-safe):
- Nové hlavy `head_b1_f`, `head_b3_f` v DDSPVocoder (note-level skalár, pooled over T)
- Init: b1 ≈ 0.97 s⁻¹, b3 ≈ 3×10⁻⁷ (blízko Steinway D)
- `decay_scale` (inference 0–2) škáluje obě konstanty

Tato fáze byla záhy zobecněna Phase 2 — viz níže.

---

### [dev-bass-refactor / Phase 2] Dvousložkový decay — zig-zag polarizace

**Zdroj**: Weinreich 1977 (měření), Hall 1986, Bensa 2003.

**Fyzika**: každá struna vibruje ve dvou polarizacích s odlišnými časy doznívání:
```
d_k(t) = α · exp(-σ_k_fast · t)   +   (1-α) · exp(-σ_k_slow · t)
         ^^^^^^^^^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
         vertikální (silnější vazba     horizontální (slabší vazba
         na soundboard, τ ≈ 1 s)        na soundboard, τ ≈ 5–8 s)
```
Superpozikou vzniká charakteristický dvoustupňový decay a AM-like beating obálky.

**Implementace** (zobecňuje Phase 1):
- 5 nových hlav: `head_b1_f`, `head_b3_f`, `head_b1_s`, `head_b3_s`, `head_alpha`
- Init: τ_fast ≈ 1 s, τ_slow ≈ 6.7 s, α = 0.5 (rovnoměrné buzení obou polarizací)
- Inference: `--decay-scale` (0 = bez decay, 1 = naučené, 2 = rychlejší doznívání)
- Výpočet na frame rate → (B × N × T_frames) místo (B × N × n_samples) → ~4 MB vs ~1 GB

---

## Roadmap

Pořadí reflektuje poměr přínos / náklady. Fáze jsou nezávislé pokud není uvedena závislost.

---

### 1. [dev-attack-loss] Attack-weighted MRSTFT loss

**Proč**: 20 řádků kódu, žádná změna architektury. Pomocí obálky loudness identifikuje
attack framy a zdůrazní je v loss funkci. Levný test, zda trénovací záležitost přispívá
k nedokonalému útoku (vs. architektonický limit).

**Fyzikální základ**: biphasní úder kladívka (Chaigne & Kergomard) způsobuje
dvoupulsový útok — model to musí vidět s vyšší vahou v loss.

```python
dl_pos   = relu(diff(gaussian_smooth(lo_db)))   # pozitivní derivace = attack
attack_w = 1.0 + alpha * (dl_pos / max(dl_pos))  # váhová mapa, floor=1.0
loss     = mrstft(pred, target) + β · mrstft(pred * w_up, target * w_up)
# β=0.3, w_up upsample na sample rate
```

**Strop**: treble noty mají attack 1–2 framy → limit je architektonický (frame rate).
Pro basy (80 ms attack → 16 framů) by měl být přínos viditelnější.

**Implementace**: `training/loss.py` (_attack_weight je připraven), přidat do trénovací smyčky.
Zpětně kompatibilní, žádné nové checkpointy.

---

### 2. [dev-bass-refactor / Phase 3] Phantom partials

**Zdroj**: Conklin 1996 (longitudinal waves), Chaigne & Kergomard (geometric nonlinearity).

**Fyzika**: velké výchylky basové struny způsobují podélné vlny a geometrickou nelinearitu:
- Podélná vlna: krátký burst před začátkem harmonického tónu (D♭1: 2 ms vs 12 ms)
- Phantom partials: energie na 2·f_j a f_i ± f_j — nejsou v harmonické řadě
- Aktuálně: NoiseSynth pohltí phantom partials jako nestrukturovaný šum → rozmazání

**Implementace**: N_PHANTOM = 16 dedikovaných oscilátorů s naučenými frekvencemi a amplitudami:
```python
self.head_phantom_mag = nn.Linear(mlp_dim, N_PHANTOM)   # magnitudy
self.head_phantom_rel = nn.Linear(mlp_dim, N_PHANTOM)   # relativní frekvence (≈ 2·k·f0)
```

**Závislost**: Phase 3 je nezávislá na Phase 1+2 — může jít paralelně.
Přínos hlavně pro A0–A2. Nejprve ověřit poslechem výsledky Phase 2.

---

### 3. [dev-unison-spread] Fyzikální rozladění strun unisonu

**Zdroj**: Hall 1986, obecná fyzika pianových strun.

**Fyzika**: piano má 2–3 struny na notu záměrně rozladěné o 0.3–2 cent
→ pomalý chorus (beating ~0.5–5 Hz), "živost" a "dýchání" sustainu.
Toto je FREKVENČNÍ efekt, odlišný od AM beatingu v Phase 2 (amplitudový efekt).

**Implementace**: druhý HarmonicSynth průchod s `f × (1 + delta)`:
```python
self.head_detune = nn.Linear(mlp_dim, 1)   # delta: 0–3 cent max
delta = sigmoid(head_detune).mean(dim=1) * DELTA_MAX
# 2. průchod: inst_freq_B = inst_freq_A * (1 + delta)
signal = a * (sin(phase_A) + sin(phase_B)) / 2
```

**Priorita**: nižší — Phase 2 již řeší majority AM beatingu, unison spread je doladění.
Dvojí HarmonicSynth průchod = 2× výpočet harmonické části. Počkat na test Phase 2.

---

### 4. [dev-frame-rate] Adaptivní frame rate řízený obálkou (čeká na M5)

**Diagnóza**: FRAME_HOP=240 samples=5 ms → C7 attack = 2 framy.
Lineární interpolace mezi 2 framy nemůže zachytit rychlou spektrální evoluci.
Toto je architektonický limit, ne trénovací záležitost.

**Proč odloženo**: implementace stojí NPZ re-extrakci + 4× více výpočtu v attack části.
Na CPU notebooku neúnosné. Správné okno = příchod M5 nebo GPU.

**Řešení**:
```
dL/dt > 0 (attack)  → FRAME_HOP_ATTACK  = 60 samples (1.25 ms)
dL/dt ≤ 0 (sustain) → FRAME_HOP_SUSTAIN = 240 samples (5 ms)
```
C7 attack: 10 ms → 8 framů místo 2. A0 attack: 80 ms → 64 framů místo 16.

---

### 5. [future] Biphasní úder kladívka — dvoupulsový attack per parciál

**Zdroj**: Chaigne & Kergomard (kapitola o hammer-string interaction).

**Fyzika**: tělo kladívka + stopka tvoří mechanický rezonátor (~40–50 Hz).
Při nárazu vznikají dva kontaktní pulsy (primární + sekundární od stopky).
Výsledek: dvoustupňový attack per parciál — patrné hlavně v basech a středním registru.

**Implementace**: learnable secondary attack pulse:
```python
self.head_t2   = nn.Linear(mlp_dim, 1)   # delay sekundárního pulsu (ms)
self.head_a2   = nn.Linear(mlp_dim, 1)   # amplituda sekundárního pulsu
# Composit attack: primary(t) + a2 · primary(t - t2)
```

**Závislost**: přínos omezený bez dev-frame-rate (attack = 2 framy u treble).
Smysluplné až po adaptivním frame rate nebo solo pro basy (80 ms attack = 16 framů).

---

## Kapacita modelu — analýza po Phase 0–2

### Parametrické složení

| Složka | small | medium | large |
|--------|-------|--------|-------|
| pre/post MLP | 34K | 118K | 432K |
| GRU | 37K | 247K | 1 381K |
| Harm heads (L+R) | 33K | 66K | 131K |
| Noise heads (L+R) | **133K** | **264K** | **527K** |
| Physics (b1/b3/alpha/B) | 0.8K | 1.5K | 3K |
| **Celkem** | **238K** | **697K** | **2 475K** |

### Upozornění: small + Phase 0 (dlouhé sekvence)

Phase 0 dává A0 notám okno 2000 framů (10 s). `small` model má:
- 1-vrstvý GRU, hidden=64
- Gradient přes 2000 kroků v mělkém GRU je náchylný k úniku/zániku
- Výsledek: model se nenaučí dlouhodobé decay vzory, přestože trénovací okno je dostatečné

**Doporučení**: pro nástroje s basovým registrem používat `medium` jako výchozí
(2-vrstvý GRU, hidden=128). `small` ponechat pro rychlé diagnostické trénování (50 epoch).

### Nevyváženost noise vs GRU (small model)

`small`: noise heads 133K vs GRU 37K — síť má 3.6× více kapacity pro tvar šumu
než pro dynamiku. Po Phase 2 (physics decay) je GRU méně zatížen, ale tato nevyváženost
zůstává nevhodná pokud je cílem naučit se jemné registrové rozdíly barvy.

Zvážit v budoucnosti: konfigurovatelné N_NOISE per model size (nebo sdílené L/R noise head).

### Strategie postupného zvyšování hloubky

Analogie z ML praxe: trénovat na daném hardware do dosažení plateau val loss,
pak přidat vrstvu GRU (gru_layers += 1) nebo zvýšit hidden.

Konkrétně pro tento projekt:
1. `small` (1 vrstva): rychlý test 50 epoch — ověří, zda física decay funguje
2. `medium` (2 vrstvy): produkční trénování pro bass instrument
3. Pokud `medium` plateau < 100 ep → zvážit `large` nebo custom preset

Přidání vrstvy GRU je zpětně nekompatibilní (nový checkpoint), ale levné —
stejný dataset, stejná pipeline, jen nová architektura.

---

## Přehled stavu

| Branch / Fáze | Stav | Klíčová změna |
|---------------|------|---------------|
| dev-softmax | ✓ hotovo | cumsum fáze, softmax distribuce |
| dev-inharmonicity | ✓ hotovo | head_B, MIDI-dep. B_MAX |
| dev-noise-fft | ✓ hotovo | NOISE_FFT 1024, harmonic-relative warping |
| dev-adaptive-nharm | ✓ hotovo | N_HARM_MAX=128, adaptivní n_active |
| bass-refactor Phase 0 | ✓ hotovo | framework, crop_frames(midi) 10s pro basy |
| bass-refactor Phase 1+2 | ✓ hotovo | dvousložkový physics decay, zig-zag polarizace |
| dev-attack-loss | ○ připraven | _attack_weight připraven, přidat do smyčky |
| bass-refactor Phase 3 | ○ plánováno | phantom partials (N_PHANTOM dedik. oscilátorů) |
| dev-unison-spread | ○ plánováno | fyzikální rozladění strun (nízká priorita) |
| dev-frame-rate | ○ čeká na HW | adaptivní frame rate (M5 / GPU) |
| biphasní hammer | ○ future | dvoupulsový attack modeling |
| progressive depth | ○ future | GRU layers++ po dosažení plateau |
