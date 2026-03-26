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

---

## Roadmap

Pořadí reflektuje poměr přínos/náklady při **aktuálním hardwaru (CPU notebook)**.
dev-frame-rate je správný dlouhodobý směr — ale stojí nejvíc a vyžaduje M5.
Nejdřív levné změny s vysokým přínosem, velká přestavba pipeline až s výkonným hardwarem.

---

### 1. [dev-attack-loss] Attack-weighted loss pomocí obálky

**Proč první**: 20 řádků kódu, žádná změna architektury, žádné nové checkpointy.
Primárně diagnostická — odpoví na otázku *kolik z chybějícího attacku je tréninková
záležitost* před tím, než investujeme do dev-frame-rate. Pokud přínos bude viditelný,
kombinovat s dev-unison-spread (stejný trénink). Pokud marginální, potvrdit závěr
a přejít na dev-unison-spread bez dalšího ladění lossu.

```python
lo_smooth  = gaussian_filter1d(lo_db, sigma=2, dim=-1)
dl         = torch.diff(lo_smooth, prepend=lo_smooth[:, :1], dim=1)
dl_pos     = torch.clamp(dl, min=0.0)
dl_norm    = dl_pos / (dl_pos.amax(dim=1, keepdim=True) + 1e-6)
attack_w   = 1.0 + alpha * dl_norm          # alpha=4.0, floor=1.0
attack_w_up = upsample(attack_w, n_samples)

loss = mrstft(pred, target) + beta * mrstft(pred * attack_w_up, target * attack_w_up)
# beta=0.3; váhy jdou na vstup MRSTFT (ne na loss číslo) → musí být hladké
```

**Strop**: pro treble noty (C7 = 2 framy attacku) je limit architektonický —
attack-loss nemůže překročit informaci, která v sekvenci není.

---

### 2. [dev-unison-spread] Unison spread — rozladění strun

**Proč druhý**: fyzikální model klavíru je bez unison spreadu fundamentálně neúplný.
Inharmonicita (dev-inharmonicity) opravila frekvenční rozmístění parciálů — ale každý
parciál stále pochází z jednoho oscilátoru. Reálné piano má 2–3 struny na notu,
záměrně rozladěné o ~0.3–2 cent → **beating** = charakteristická "živost" a "dýchání"
sustainu. Bez toho bude sustain vždy znít staticky bez ohledu na loss funkci.
Implementace nepotřebuje změny pipeline, zvládne to CPU.

**Fyzika**: jev je steady-state (přítomen po celou dobu tónu), nejsilnější v basech.
Není to inharmonicita ani transient — jde o interakci mezi strunami v unisonové skupině.

```python
# DDSPVocoder.__init__
self.head_detune = nn.Linear(mlp_dim, 1)
nn.init.constant_(self.head_detune.bias, -6.0)   # sigmoid ≈ 0.002 → delta ≈ 0 na startu

# DDSPVocoder.forward
DELTA_MAX = 0.003    # ~5 cent horní mez (fyzikálně: basy až 2 cent, treble 0.3 cent)
delta = torch.sigmoid(self.head_detune(feat)).squeeze(-1).mean(dim=1) * DELTA_MAX

# HarmonicSynth.forward — druhý oscilátor s f × (1 + delta)
inst_freq_A = f0_up.unsqueeze(1) * (k * stretch).unsqueeze(2)
inst_freq_B = inst_freq_A * (1.0 + delta.view(B, 1, 1))
phase_A = torch.cumsum(2π * inst_freq_A / SR, dim=-1)
phase_B = torch.cumsum(2π * inst_freq_B / SR, dim=-1)
signal  = (a * (torch.sin(phase_A) + torch.sin(phase_B))).sum(dim=1) / (2 * N)
```

Beating frekvence = `delta × f_k` → A4 (440 Hz), delta=0.001:
beat na k=1: 0.44 Hz, na k=10: 4.4 Hz — slyšitelný chorusing.

Kombinovat s **dev-noise-amp** (viz níže) — oboje jde do stejného branche, +5 řádků navíc.
GUI slider `unison_scale` (0–2), analogicky `inh_scale`.

---

### ~~3. [dev-noise-amp] Globální amplituda šumové složky~~ → implementováno v dev-noise-fft

---

### 4. [dev-frame-rate] Adaptivní frame rate řízený obálkou

**Proč čtvrtý**: správná diagnóza, špatný timing na CPU. `FRAME_HOP = 240 samples = 5 ms`
dává C7 attacku **2 framy** — lineární interpolace mezi nimi nemůže zachytit rychlou
spektrální evoluci. Toto je fundamentální limit architektury, ne tréninková záležitost.
Implementace ale stojí: NPZ re-extrakce (~55 min), změny dataset loaderu, nové pozicové
kódování, 4× hustší sekvence v attack části = 4× více výpočtu. Na CPU je to neúnosné.

**Správné okno**: příchod M5. Pak implementovat jako první velkou změnu.

**Řešení**: envelope-guided variable frame rate

```python
# dL/dt > 0 (attack)  → FRAME_HOP_ATTACK  = 60 samples (1.25 ms)
# dL/dt ≤ 0 (sustain) → FRAME_HOP_SUSTAIN = 240 samples (5 ms)

# Pozicové kódování pro GRU — v sample jednotkách, normalizované FRAME_HOP
t_enc = frame_samples / FRAME_HOP   # attack step=0.25, sustain step=1.0
feat  = cat([f0_enc, vel_enc, time_enc], dim=-1)
```

C7 attack (10ms) → 8 framů místo 2. A0 attack (80ms) → 64 framů místo 16.

| Komponenta | Změna |
|-----------|-------|
| NPZ extrakce | ukládat frame_samples (absolutní pozice v samples), adaptivní hop |
| Dataset loader | variabilní frame sekvence, předávat frame_samples |
| DDSPVocoder | přidat `time_enc` do `feat_dim` |
| HarmonicSynth | beze změny (pracuje v samples) |

Po dev-frame-rate odblokovat **transient spectral evolution** (time-varying B(t) —
nelinearita kontaktu kladívka, rychlá spektrální evoluce prvních 5–20 ms attacku).

---

## Poznámky

- dev-attack-loss: zpětně kompatibilní, žádné nové checkpointy
- dev-unison-spread + dev-noise-amp: jeden branch, nové checkpointy
- dev-frame-rate: čeká na M5, největší pipeline změna projektu
- Testovat vždy small model (50 epoch) před medium (300 epoch)
