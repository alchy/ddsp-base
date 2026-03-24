# Model Roadmap — navrhované vylepšení

Tento dokument zachycuje navrhované architektonické změny modelu, jejich motivaci
a stav implementace. Slouží jako backlog pro budoucí větve.

---

## Implementováno

### [dev-softmax] Fázová akumulace + softmax harmonická distribuce

**Větev**: `dev-softmax` (merged/in-progress)

**Problém**:
- `HarmonicSynth` počítal fázi jako `mean_freq × t`, což je nesprávné pro časově proměnné F0.
  Správné je kumulovat okamžitou frekvenci přes čas (`cumsum`).
- `head_harm_L/R` používal `sigmoid` — každá harmonická nezávisle v [0,1], bez normalizace.
  Výsledek závisel na absolutní hodnotě výstupu sítě, ne na relativním rozložení.

**Řešení**:
```python
# Fáze
phase = torch.cumsum(2π · inst_freq / SR, dim=-1)

# Harmonická distribuce
harm_dist = softmax(head_harm)          # normalizované váhy, suma = 1
harm_amp  = softplus(head_amp)          # globální amplituda (nová hlava)
harm_amps = harm_dist * harm_amp
```

**Efekt**: Správný pitch (cumsum), lepší podmíněnost trénování (softmax), rychlejší konvergence.

---

## Navrhované změny

### [dev-inharmonicity] Piano inharmonicita strun

**Priorita**: vysoká — výrazný vliv na „živost" zvuku

**Problém**:
Reálné piano má harmonické posunuté od přesných násobků f0:
```
f_k = k · f0 · √(1 + B · k²)
```
Vyšší harmonické jsou lehce výše než by odpovídalo čistému harmonickému řadě.
Aktuální model předpokládá `f_k = k · f0` — výsledek zní jako „ideální oscilátor".

**Navrhované řešení**:

Síť predikuje jeden skalár `B` per nota (pooled přes T):
```python
# Nová hlava (1 Linear, ~256 params)
self.head_B = nn.Linear(mlp_dim, 1)
nn.init.constant_(self.head_B.bias, -5.0)  # start: B ≈ 0 (bezpečné)

# V HarmonicSynth.forward:
B_raw   = torch.sigmoid(self.head_B(feat))       # (B, T, 1)
B       = B_raw.mean(dim=1) * B_MAX              # pool → (B, 1), B_MAX = 0.0002
stretch = torch.sqrt(1 + B.unsqueeze(1) * k**2) # (B, N)
inst_freq = f0_up.unsqueeze(1) * k * stretch.unsqueeze(2)
```

**Proč pooled scalar**:
- B je fyzikální vlastnost struny — nemění se per-frame
- Pooling přes T zabraňuje pitch jitteru
- Inicializace na −5 → B ≈ 0 na startu = žádné riziko rozbití pitche při přechodu

**Proč learned (ne fixní)**:
- Každý nástroj (Salamander, vintage-vibe, …) má jiné B
- Fixní fyzikální model by vyžadoval ruční kalibraci per-banka

**Alternativa**: fixní `B(midi)` odvozené z frekvence — rychlejší implementace,
ale neadaptuje se na data. Vhodné jako baseline pro srovnání.

---

### [dev-attack-loss] Attack-weighted loss pomocí obálky

**Priorita**: vysoká — přímé řešení slabého bright attacku

**Problém**:
MRSTFT průměruje loss přes celý signál. Sustain trvá sekundy, attack 10–80 ms
(C7 ≈ 10 ms, A0 ≈ 80 ms). Model minimalizuje loss tím, že se naučí sustain
a attack „odpíše" — výsledek má správný pitch a sustain, ale slabé bright tóny při náběhu.

Klíčové: model **kapacitně umí** generovat bright attack (softmax to nevylučuje),
ale loss ho k tomu **netlačí**. Jde o problém tréninku, ne architektury.

**Navrhované řešení**: váhování loss podle derivace obálky

Místo fixního okna (100 ms) použít `dL/dt` — loudness roste → extra váha,
loudness klesá → normální váha. Automaticky se přizpůsobí délce attacku per nota
(basové struny vs. výšky).

```python
# lo_db: (B, T) loudness z NPZ — již k dispozici v tréninku
dl = torch.diff(lo_db, prepend=lo_db[:, :1], dim=1)   # dL/dt per frame
dl = gaussian_smooth(dl, sigma=2)                       # vyhlazení — NPZ derivative je zašuměná
attack_w = torch.clamp(dl * alpha, min=0.0) + 1.0      # alpha ~ 5–10, floor = 1.0 (nikdy < norm)

# Upsample na sample rate pro STFT loss
attack_w_up = F.interpolate(attack_w.unsqueeze(1), size=n_samples, mode='linear')

# Loss: standard MRSTFT + attack-weighted term
loss = mrstft(pred, target) + beta * mrstft(pred * attack_w_up, target * attack_w_up)
```

**Proč obálka místo fixního okna**:
- Bass (A0): attack ~80 ms → fixní 100 ms ok, ale přesah do decay
- Treble (C7): attack ~10 ms → fixní 100 ms by tlačil na sustain zbytečně
- Obálka to ví sama — adaptivní, bez nových hyperparametrů mimo `alpha` a `beta`

**Výhrada**: `gaussian_smooth` nutný — raw derivace NPZ loudness je zašuměná,
zejména v prvních framech. Bez vyhlazení → noisy gradient signal.

**Parametry**: `alpha=5.0`, `beta=0.5` jako výchozí — attack frames dostanou
~6× vyšší váhu než sustain, celková loss je součet obou termů.

---

### [dev-noise-amp] Globální amplituda šumové složky

**Priorita**: střední

**Problém**:
Po zavedení `head_amp_L/R` pro harmonickou složku vznikla asymetrie:

| složka | aktivace | globální škálování |
|--------|----------|--------------------|
| harmonická | softmax × softplus | `head_amp_L/R` (naučená) |
| šumová | sigmoid per bin | žádné |

Šumová složka nemá vlastní globální amplitudu — model musí zvyšovat/snižovat
všech 129 binů najednou, což zpomaluje učení balance harm/noise.

**Navrhované řešení**:
```python
self.head_noise_amp_L = nn.Linear(mlp_dim, 1)
self.head_noise_amp_R = nn.Linear(mlp_dim, 1)
nn.init.constant_(self.head_noise_amp_L.bias, -3.0)
nn.init.constant_(self.head_noise_amp_R.bias, -3.0)

# V forward:
noise_amp_L = F.softplus(self.head_noise_amp_L(feat))   # (B, T, 1)
noise_L     = torch.sigmoid(self.head_noise_L(feat)) * noise_amp_L
```

**Efekt**: Symetrie s harmonickou složkou — síť může rychle ztišit/zesílit
celou šumovou složku, zatímco spektrální tvar (distribuce přes 129 binů) zůstává nezávislý.

---

## Pořadí implementace

1. **[dev-softmax]** — hotovo, testování (50 epoch small model)
2. **[dev-attack-loss]** — po dokončení dev-softmax; řeší slabý bright attack (bez změny architektury)
3. **[dev-inharmonicity]** — sustain realism, strunový charakter, beating
4. **[dev-noise-amp]** — balance harm/noise, velocity energie; kombinovat s dev-inharmonicity

## Poznámky

- Každá změna architektury = nové checkpointy (staré nekompatibilní)
- Testovat vždy na small modelu (50 epoch) před spuštěním medium (300 epoch)
- `B_MAX = 0.0002` odpovídá typickým pianům; lze vystavit jako `--inharmonicity-max` CLI arg
