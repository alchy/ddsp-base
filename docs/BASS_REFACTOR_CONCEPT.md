# Koncept: dev-bass-refactor

Vychází z průzkumu 18 fyzikálních paperů (Bensa 2003, Chabassier 2013, Bank & Chabassier 2019,
Simionato 2024, aj.). Cíl: eliminovat muddy bass C1–C5 a zlepšit attack.

---

## Diagnóza kořenových příčin

### Proč jsou basy muddy
Papery i kompletní FEM model Steinway D (300 CPU, 24h/s) explicitně přiznávají selhání na
"bass depth". Příčin je několik — seřazeny od nejvyššího dopadu:

1. **Training window 250 ms — největší bottleneck**
   - A0 sample = 29 s, τ_slow ≈ 5 s → model vidí 0.8 % celé noty
   - Dvoustupňový decay (rychlá polarizace ~0.3 s + pomalá ~5 s) je v datech ale model ho nevidí
   - NoiseSynth kompenzuje "chybějící" sustain jako perzistentní šum

2. **Chybí per-partial frekvenčně závislý decay**
   - Aktuální model: jeden globální amp scalar na frame → všechny parciály stejný decay
   - Fyzika (Bensa 2003): `σ_j = b₁ + b₃·ωⱼ²`
   - C2: b₁=0.25 s⁻¹, b₃=7.5×10⁻⁵ → parciál 20 (f=1.3 kHz) zanikne 5× rychleji než parciál 1
   - Model drží vyšší parciály příliš dlouho → residual jde do NoiseSynthu

3. **Chybí dvoustupňový decay + AM beating**
   - Fyzika: zig-zag polarizace struny (SMAC13, Piano-model-revis)
   - Vertikální polarizace zanikne rychle (high admittance mostu), horizontální přetrvává
   - Výsledek: amplitudová modulace každého parciálu (1–5 Hz)
   - Beating NENÍ frekvenční posun (δF) — je to AM jednoho parciálu
   - NoiseSynth generuje statický šum místo AM modulace

4. **Phantom partials absorbovány NoiseSynthem**
   - Geometrická nelinearita struny → parciály na 2·fⱼ (longitudinální vlny)
   - D#1 forte: naměřeny na 1.2, 1.7, 2.0, 2.3, 2.8, 3.3, 3.9 kHz
   - Leží mimo harmonické pozice → NoiseSynth je musí pokrýt

5. **Precursor transient nemodelován**
   - Longitudinální vlny: c_∥ ≈ 5840 m/s, c_⊥ ≈ 209 m/s (ratio ~28×)
   - D♭1: precursor okno 2–12 ms (10 ms burst PŘED harmonickým attackem)
   - Model interpretuje jako šum v attacku

### Proč attack není ideální
- Biphasický hammer force profil (shank oscillation ~40-50 Hz resonance)
- Jeden hammer — primární peak + sekundární peak (rebound) 0–8 ms
- V basu (D#1): oblast 500–1200 Hz se mění až 25 dB podle způsobu hry
- Hammer p: 2.27 (C2 bas) → 3.0 (G6 výšky) — měkčí filc = více low-pass

---

## Navrhovaná architektura

### Klíčová změna: Parametrický syntézní model

**Aktuálně (frame-by-frame):**
```python
# GRU predikuje distribuci a amplitude na každém framu
harm_dist[B, T, N]   # softmax — per-frame overtone balance
harm_amp[B, T, 1]    # softplus — per-frame global scale
# → A_j[t] = harm_dist[j, t] * harm_amp[t]   (jeden frame = jedna hodnota)
```

**Nově (parametrický envelope per partial):**
```python
# GRU predikuje parametry envelopu pro celou notu
alpha_j[B, N]        # per-partial scale (softmax → softplus)
b1[B, 1]             # freq-independent decay (softplus)
b3[B, 1]             # freq-dependent decay (softplus)
AM_j[B, N]           # AM hloubka per partial (sigmoid)
AM_rate_j[B, N]      # AM frekvence 1-5 Hz (sigmoid)

# Syntéza:
omega_j = 2*pi * j * f0
sigma_j = b1 + b3 * omega_j**2          # fyzikální prior
A_j(t) = alpha_j * exp(-sigma_j * t)    # jednostupňový (fáze 1)

# Dvoustupňový (fáze 2):
A_j(t) = alpha1_j * exp(-gamma1_j * t) * (1 + AM_j * sin(2*pi*AM_rate_j * t))
        + alpha2_j * exp(-gamma2_j * t)
```

Loudness se stále aplikuje post-synthesis (decoupled timbre zachován).

### Nové výstupní hlavy DDSPVocoderV2

```
Stávající hlavy (zachovány):
  head_harm_L/R    → softmax(N_HARM)    [overtone distribution — init timbre]
  head_B           → sigmoid → B_MAX    [inharmonicita]

Nové hlavy (přidány):
  head_b1_L/R      → softplus(1)        [freq-independent decay rate]
  head_b3_L/R      → softplus(1)        [freq-dependent decay rate]
  head_alpha_L/R   → softplus(N_HARM)   [per-partial amplitude scale]
  head_AM_L/R      → sigmoid(N_HARM)    [AM hloubka per partial, 0-1]
  head_AM_rate_L/R → sigmoid(N_HARM)    [AM rate 1-5 Hz per partial]

Volitelné (fáze 3):
  head_dF          → tanh(1)            [beating frequency offset δF]
```

### HarmonicSynthV2 — nový synthesis kernel

```python
def synthesize(alpha_j, b1, b3, f0_hz, AM_j, AM_rate_j, inh, n_samples):
    t = torch.arange(n_samples) / SR                    # časová osa

    # frekvence parciálů (inharmonické)
    k = torch.arange(1, N_HARM + 1)
    stretch = torch.sqrt(1 + inh * k**2)
    f_j = f0_hz * k * stretch                           # (N_HARM,)

    # frekvenčně závislý decay
    omega_j = 2 * pi * f_j
    sigma_j = b1 + b3 * omega_j**2                      # (N_HARM,)

    # amplitudový envelope per partial
    A_j = alpha_j * torch.exp(-sigma_j * t[:, None])   # (T, N_HARM)

    # AM modulace (beating z zig-zag polarizace)
    AM = 1 + AM_j * torch.sin(2 * pi * AM_rate_j * t[:, None])
    A_j = A_j * AM                                       # (T, N_HARM)

    # Nyquist masking
    A_j[:, f_j > 0.45 * SR] = 0

    # fázová akumulace (cumsum)
    phase = torch.cumsum(2 * pi * f_j / SR, dim=0)      # (T, N_HARM)
    signal = torch.sum(A_j * torch.sin(phase), dim=-1)  # (T,)

    # phantom partials (longitudinální vlny) — hard-coded, 0 params
    f_phantom = 2 * f_j                                 # 2·fⱼ
    A_phantom = alpha_j * 0.05 * torch.exp(-2 * sigma_j * t[:, None])
    phase_ph = torch.cumsum(2 * pi * f_phantom / SR, dim=0)
    signal += torch.sum(A_phantom * torch.sin(phase_ph), dim=-1)

    return signal / (N_HARM + 1)
```

---

## Pořadí implementace (od nejvyššího ROI)

### Fáze 0 — Training Window Fix (diagnostická, ~30 řádků, ŽÁDNÁ arch změna)

**Problém:** `SourceDataset` vytváří náhodná 50-frame (250 ms) okna pro všechny noty.
**Fix:** Adaptivní délka okna dle MIDI noty.

```python
# V SourceDataset.__getitem__:
def get_window_size(midi):
    # bass: 4000 frames (20s), střed: 1000 frames (5s), výšky: 200 frames (1s)
    if midi < 48:   return 4000
    elif midi < 72: return 1000
    else:           return 200

window = get_window_size(self.midi)
```

**Proč nejdřív:** Tato změna nevyžaduje arch změnu, ale okamžitě umožní modelu vidět celý decay.
Výsledek poslouží jako diagnostika — pokud se bass zlepší, příčina je v window size.
Pokud ne, problém je čistě architektonický.

**Rizika:**
- Batch size musí být snížen pro bass sekvence (paměť)
- Možné řešení: per-register batche nebo gradient accumulation
- GRU bude muset zpracovat delší sekvence → pomalejší training

---

### Fáze 1 — Physics-informed per-partial decay (core fix, ~100 řádků)

Nahradit `head_amp_L/R` (jeden scalar) za `head_b1_L/R + head_b3_L/R + head_alpha_L/R`.

Nový synthesis kernel: `A_j(t) = alpha_j * exp(-(b1 + b3*omega_j^2) * t)`

**Fyzikální priory pro init:**
```python
# b1 init: malé, ale nenulové (b1 ≈ 0.25 s⁻¹ pro C2)
head_b1.bias.data.fill_(-2.0)   # softplus(-2) ≈ 0.13

# b3 init: velmi malé pro bass
head_b3.bias.data.fill_(-5.0)   # softplus(-5) ≈ 0.007
```

Tato fáze přidá frekvenčně závislý decay bez dvoustupňového modelu.

---

### Fáze 2 — Phantom partials (triviální, ~10 řádků)

Přidat do HarmonicSynth sekundární bank na `f_phantom = 2 * f_j` s pevnou amplitudou
`A_phantom = 0.05 * A_j` a `decay_phantom = 2 * sigma_j`. Hard-coded, žádné nové parametry.

---

### Fáze 3 — AM beating + dvoustupňový decay (~80 řádků)

Přidat `head_AM_L/R` + `head_AM_rate_L/R`. Rozsah AM_rate: sigmoid → [0.5, 5] Hz.
Dvoustupňový decay: `head_alpha1, head_gamma1, head_alpha2, head_gamma2` per partial.

**Pozor na parametrický explozi:** N_HARM=128 × 4 params = 512 čísel per partial per kanal.
Možné zjednodušení: sdílené b1/b3/AM_rate přes všechny parciály, pouze alpha_j per-partial.

---

### Fáze 4 — NoiseSynth temporal gating (volitelné, ~30 řádků)

Přidat `note_age` (čas od onsetu, normalizovaný) jako kondicionovací vstup NoiseSynthu.
Bass precursor okno: 0–12 ms → NoiseSynth aktivní; sustain → útlum noise amplitudy.

---

## Training protokol (Simionato 2024)

Two-phase training — zabraňuje B-distortion při optimalizaci decay:

1. **Fáze A:** Trénuj pouze `head_B` (inharmonicita) — F loss na f1
   - Zmraz všechny ostatní hlavy
   - 20–30 epoch

2. **Fáze B:** Zmraz `head_B`, trénuj vše ostatní
   - MRSTFT + 0.2×L1 + 0.3×attack_loss
   - Přidat: RMS envelope supervision per partial (nový loss term)

---

## Co se NEMĚNÍ

- B_MAX = 8×10⁻⁴ pro A0 — realistické (ovinuté struny mají nižší B než vzorec)
- N_HARM_MAX = 128 — dostačující pro C2+ (A0 má ~140 parciálů, ale nárůst na 256 není prioritou)
- MRSTFT fft_sizes — zachovat (256→512 je drobná optimalizace, ne blokátor)
- Decoupled timbre architektura — loudness stále post-synthesis
- EnvelopeNet — nezměněn
- NoiseSynth harmonic-relative warping — zachovat

---

## Shrnutí: Co je nové vs. stávající roadmap

| Položka | Stávající roadmap | Nový koncept |
|---|---|---|
| attack-loss | dev-attack-loss (20 řádků) | Zachovat, ale není priorita |
| unison-spread | dev-unison-spread (δF) | Nahrazeno AM per partial (fyzikálně správnější) |
| frame-rate | dev-frame-rate (čeká na GPU) | Nahrazeno adaptive window size (levnější) |
| **[NOVÉ]** | — | Training window adaptation (Fáze 0) |
| **[NOVÉ]** | — | Per-partial σ_j = b1+b3·ωj² decay (Fáze 1) |
| **[NOVÉ]** | — | Phantom partials 2·fj (Fáze 2) |
| **[NOVÉ]** | — | AM beating per partial (Fáze 3) |
| **[NOVÉ]** | — | Two-phase training protokol |

---

## Reference papers

- Bensa et al. 2003 (JASA) — b₁/b₂ hodnoty, tabulka C2/C4/C7
- Bank & Chabassier 2019 (IEEE SPM) — 5 jevů, review
- Simionato et al. 2024 (Frontiers) — DDSP architecture, two-phase training
- Chabassier, Chaigne, Joly 2013 (JASA) — Steinway D params, phantom partials
- Castera & Chabassier 2023 (RR-9530) — precursor timing D♭1=10ms
- Chabassier & Duruflé 2014 (JSV) — hammer shank, biphasic force
- Chabassier, Chaigne, Joly 2013 (ESAIM Part 1) — c_∥/c_⊥≈10, 5-unknown model
- Chabassier, Duruflé, Joly 2016 (ESAIM Part 2) — model ablation: stiffness+nonlin nutné zároveň
