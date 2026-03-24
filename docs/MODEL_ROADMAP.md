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
2. **[dev-inharmonicity]** — po dokončení dev-softmax testu
3. **[dev-noise-amp]** — kombinovat s dev-inharmonicity nebo samostatně

## Poznámky

- Každá změna architektury = nové checkpointy (staré nekompatibilní)
- Testovat vždy na small modelu (50 epoch) před spuštěním medium (300 epoch)
- `B_MAX = 0.0002` odpovídá typickým pianům; lze vystavit jako `--inharmonicity-max` CLI arg
