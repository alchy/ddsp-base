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

**Kdy implementovat**: po dokončení dev-softmax testu (50 epoch). Ideálně na M5
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

### 3. [dev-inharmonicity] Piano inharmonicita strun

**Problém**: model předpokládá `f_k = k·f0`. Reálné piano má:
```
f_k = k · f0 · √(1 + B · k²)
```
Vyšší parciály jsou výše → chybí strunový charakter, beating, "živost" sustainu.
Bez inharmonicity parciály "splývají" v MRSTFT bez ohledu na váhování lossu —
inharmonicita může pomoci bright attacku víc než dev-attack-loss.

**Implementace**: learned scalar B per nota (pooled přes T)

```python
self.head_B = nn.Linear(mlp_dim, 1)
nn.init.constant_(self.head_B.bias, -5.0)   # start: B ≈ 0

B_raw   = torch.sigmoid(self.head_B(feat)).mean(dim=1) * B_MAX   # pool → (B,1)
stretch = torch.sqrt(1 + B_raw.unsqueeze(1) * k**2)
inst_freq = f0_up.unsqueeze(1) * k * stretch.unsqueeze(2)
```

**B_MAX**: nepoužívat jeden globální bound — basy mají výrazně vyšší B než výšky.
Lepší MIDI-dependent prior:
```python
B_MAX(midi) = B_0 · exp(-(midi - 21) / 88 · ln(10))
# B_0 ≈ 0.0008 (A0), klesá na ~0.00008 (C8)
```

**Doporučený postup**: nejdřív fixed `B(midi)` jako baseline (žádné nové parametry),
pak learned scalar. Pokud fixed dá 80 % benefitu — zvážit zda learned přidává hodnotu.

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

- Každá změna architektury (dev-frame-rate, dev-inharmonicity, dev-noise-amp) = nové checkpointy
- dev-attack-loss je zpětně kompatibilní (pouze loss funkce)
- Testovat vždy small model (50 epoch) před medium (300 epoch)
- dev-frame-rate je preferovaný hlavní směr; ostatní kroky jsou buď rychlé testy nebo doplňky
