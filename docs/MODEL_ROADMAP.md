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

## Roadmap

### 1. [dev-frame-rate] Adaptivní frame rate řízený obálkou  ← nejvyšší priorita

**Kořen problému**:

Aktuální `FRAME_HOP = 240 samples = 5 ms`. Piano attack C7 trvá ~10 ms = **2 framy**.
Model má 2 datové body k popisu celého nástupu — lineární interpolace mezi nimi
nemůže zachytit rychlou spektrální evoluci bez ohledu na loss funkci nebo architekturu.
Toto je fundamentální limit, ne tréninková záležitost.

**Řešení**: envelope-guided variable frame rate

Obálka (derivace loudness) říká kde se "děje něco zajímavého":
- `dL/dt > 0` (attack) → `FRAME_HOP_ATTACK = 60 samples (1.25 ms)`
- `dL/dt ≤ 0` (sustain/decay) → `FRAME_HOP_SUSTAIN = 240 samples (5 ms)`

C7 attack (10ms) → 8 framů místo 2. A0 attack (80ms) → 64 framů místo 16.

**Pozicové kódování pro GRU**:

GRU potřebuje vědět kolik reálného času mezi framy uplynulo — jinak se temporální
vzory naučí špatně. Řešení: přidat `time_enc` jako vstup, kódovaný v **sample jednotkách**
normalizovaných hodnotou `FRAME_HOP` (ne timestamps v sekundách — model pracuje v samples):

```python
# frame_samples: (T,) absolutní pozice každého framu
t_enc = frame_samples / FRAME_HOP        # "standardní jednotky"
# attack frame (hop=60):   t_enc step = 60/240  = 0.25
# sustain frame (hop=240): t_enc step = 240/240 = 1.0

# sinusoidální kódování → (T, time_dim), analogicky jako F0 encoding
feat = cat([f0_enc, vel_enc, time_enc], dim=-1)
```

GRU přímo vidí hustotu framů — krok 0.25 = attack, krok 1.0 = sustain.
Relativní rozestupy 4:1 odpovídají fyzikální realitě.

**Dopady na pipeline**:

| Komponenta | Změna |
|-----------|-------|
| NPZ extrakce | ukládat framy s timestamps (sample index), adaptivní hop dle obálky |
| Dataset loader | číst variabilní frame sekvence, předávat frame_samples |
| DDSPVocoder | přidat `time_enc` do `feat_dim` |
| HarmonicSynth | beze změny (pracuje v samples) |

**Kdy implementovat**: dev-softmax test dokončen (50 epoch, val loss 1.1205). Ideálně na M5
(4× hustší sekvence = 4× více výpočtu v attack části).

---

### 2. [dev-attack-loss] Attack-weighted loss pomocí obálky  ← rychlý test, omezený strop

**Smysl tohoto kroku**:

Levná změna (20 řádků v `ddsp.py`, bez změny architektury), která rychle odpoví
na otázku: *kolik z chybějícího bright attacku je tréninková záležitost vs. fundamentální limit?*

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

**Známý strop**: pro treble noty (C7 = 2 framy attacku) je limit architektonický —
attack-loss nemůže překročit informaci, která v sekvenci není.
Pokud zlepšení bude marginální → přeskočit na dev-frame-rate bez dalšího ladění.

---

### 3. [dev-unison-spread] Unison spread — rozladění strun

**Fyzika**: piano má 2 struny v basovém rejstříku, 3 ve středním a vysokém.
Každá struna je záměrně naladěna o ~0.3–2 cent jinak → **beating** mezi strunami
= charakteristická "živost" a "dýchání" sustainu. Aktuální model má jeden oscilátor
na parciál → sustain je statický, nemá chorusing.

**Co to není**: nejde o inharmonicitu (B koeficient) ani o transient — je to steady-state
jev přítomný po celou dobu tónu, nejsilnější v basech.

**Implementace**: zdvojení oscilátorů s learned detuning `delta` per nota:

```python
# DDSPVocoder.__init__
self.head_detune = nn.Linear(mlp_dim, 1)
nn.init.constant_(self.head_detune.bias, -6.0)   # sigmoid ≈ 0.002 → delta ≈ 0 na startu

# DDSPVocoder.forward
DELTA_MAX = 0.003    # ~5 cent horní mez (fyzikálně: basy až 2 cent, treble 0.3 cent)
delta = torch.sigmoid(self.head_detune(feat)).squeeze(-1).mean(dim=1) * DELTA_MAX

# HarmonicSynth.forward — přidat druhý oscilátor s f * (1 + delta)
inst_freq_A = f0_up.unsqueeze(1) * (k * stretch).unsqueeze(2)
inst_freq_B = inst_freq_A * (1.0 + delta.view(B, 1, 1))
phase_A = torch.cumsum(2π * inst_freq_A / SR, dim=-1)
phase_B = torch.cumsum(2π * inst_freq_B / SR, dim=-1)
signal  = (a * (torch.sin(phase_A) + torch.sin(phase_B))).sum(dim=1) / (2 * N)
```

**Výsledek**: beating frekvence = `delta * f_k` → pro A4 (440 Hz) a delta=0.001
beat na k=1: 0.44 Hz, k=10: 4.4 Hz — přesně v rozsahu slyšitelného "live" efektu.

**Poznámky**:
- `delta` je MIDI-independent na začátek; pokud basy potřebují výrazně jiný rozsah
  než výšky, přidat analogický MIDI-dependent prior jako u B_MAX
- Alternativa s amplitudovým poměrem (`a1 : a2 = 1:1` vs. `1:0.7`): možné rozšíření,
  ale pravděpodobně není potřeba — model má `head_harm` k vyvážení
- Výpočetní cena: 2× více cumsum + sin v HarmonicSynth (~+30 % inference time)
- GUI slider `unison_scale` (0–2), analogicky jako `inh_scale`

**Zaparkováno — transient spectral evolution**: časově proměnné B při útoku kladívka
(nelinearita kontaktu → rychlá spektrální evoluce prvních 5–20 ms) vyžaduje time-varying
B(t) místo poolovaného skaláru + dev-frame-rate pro dostatečné časové rozlišení.
Implementovat až po dev-frame-rate.

---

### 4. [dev-noise-amp] Globální amplituda šumové složky

**Problém**: po zavedení `head_amp_L/R` pro harmonické vznikla asymetrie —
noise nemá globální scalar, model musí ladit všech 129 binů najednou.

```python
noise_amp_L = softplus(head_noise_amp_L(feat))   # nová hlava
noise_L     = sigmoid(head_noise_L(feat)) * noise_amp_L
```

Nízká priorita — jde o conditioning cleanup, ne hlavní řešení kvality attacku.
Vhodné kombinovat s jiným branchem.

---

## Poznámky

- Každá změna architektury = nové checkpointy (dev-frame-rate, dev-unison-spread, dev-noise-amp)
- dev-attack-loss je zpětně kompatibilní (pouze loss funkce)
- Testovat vždy small model (50 epoch) před medium (300 epoch)
- dev-frame-rate je preferovaný hlavní směr pro attack; dev-unison-spread pro sustain živost
- Transient spectral evolution (time-varying B) zaparkováno — závisí na dev-frame-rate
