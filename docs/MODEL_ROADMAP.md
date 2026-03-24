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

**Doporučený A/B experiment**:

| Varianta | Popis | Kdy použít |
|----------|-------|-----------|
| A — fixed `B(midi)` | `B = B_0 · 2^((midi−69)/6)`, žádné nové params | baseline, ověření slyšitelnosti efektu |
| B — learned scalar | `head_B → sigmoid → pool → B_MAX`, ~256 params | finální produkční model |

Pokud fixed baseline (A) dá skoro stejný poslechový benefit jako learned (B),
je levnější, stabilnější a interpretovatelný. Learned varianta je správný finální cíl.

> **B_MAX = 0.0002** je konzervativní výchozí hodnota. Reálné hodnoty pro grand piano
> jsou přibližně 0.0001–0.0008 podle rejstříku (basy vyšší, výšky nižší).
> Lepší alternativa než jeden globální bound je **MIDI-dependent prior**:
> `B(midi) = B_0 · exp(-(midi − 21) / 88 · ln(10))` — fyzikálně přesnější a automaticky
> škáluje s rejstříkem bez nutnosti ruční kalibrace. Pokud efekt nebude slyšet,
> není chyba v architektuře — je to příliš těsný bound.

**Vazba na attack**: inharmonicita může pomoci bright attacku více než dev-attack-loss,
protože vyšší parciály bez inharmonicity "splývají" v čase a MRSTFT je netrestá
bez ohledu na váhování. Pořadí attack-loss → inharmonicity je rozumné pro rychlé
ověření, ale pokud attack-loss přinese jen marginální zlepšení, přeskočit rovnou sem.

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

# (1) Vyhlazení derivace — raw dL/dt z NPZ je zašuměná
lo_smooth = gaussian_filter1d(lo_db, sigma=2, dim=-1)
dl = torch.diff(lo_smooth, prepend=lo_smooth[:, :1], dim=1)

# (2) Clamp + normalizace per sample — alpha má stabilní sémantiku napříč batchi
dl_pos  = torch.clamp(dl, min=0.0)
dl_norm = dl_pos / (dl_pos.amax(dim=1, keepdim=True) + 1e-6)   # [0, 1] per sample

# (3) Hladké váhy, floor = 1.0 — váhy se aplikují na vstup MRSTFT, ne na loss číslo
#     → STFT nesmí vidět ostré přechody váhy (artefakty)
attack_w = 1.0 + alpha * dl_norm                                # alpha ≈ 3–5

# Upsample na sample rate pro STFT loss
attack_w_up = F.interpolate(attack_w.unsqueeze(1), size=n_samples,
                             mode='linear', align_corners=False).squeeze(1)

# Loss: standard MRSTFT + attack-weighted term
# Váhy na vstupu: mrstft(pred * w, target * w) — ne loss * w
loss = mrstft(pred, target) + beta * mrstft(pred * attack_w_up, target * attack_w_up)
```

**Proč obálka místo fixního okna**:
- Bass (A0): attack ~80 ms → fixní 100 ms ok, ale přesah do decay
- Treble (C7): attack ~10 ms → fixní 100 ms by tlačil na sustain zbytečně
- Obálka to ví sama — adaptivní, bez nových hyperparametrů mimo `alpha` a `beta`

**Tři klíčové implementační detaily** (potvrzeno odbornou konzultací):
1. **Vyhlazení** (`gaussian_filter1d`, sigma≈2) — raw derivace NPZ je zašuměná; bez toho noisy gradient
2. **Normalizace per sample** po clampu — `alpha` má pak stabilní sémantiku bez závislosti na absolutní loudnosti
3. **Váhy na vstupu MRSTFT**, ne na loss hodnotě — proto musí být hladké; ostré přechody váhy = STFT artefakty

**Parametry**: doporučený start `alpha=4.0`, `beta=0.3`.

> **Scope tohoto kroku**: dev-attack-loss řeší *underweighted onset energy / brightness* —
> tedy to, že model má kapacitu na bright attack, ale loss ho k tomu netlačí.
> Neřeší fyzikální inharmonicitu horních partials. Pokud se po tomto kroku attack zlepší,
> ale stále chybí „strunová kovovost", není to selhání branch — je to signál pro dev-inharmonicity.

**Otevřená pochybnost**: výše uvedená hypotéza ("model umí, loss netlačí") nemusí být pravdivá.
Alternativa: frame-rate interpolace harmonických amplitud (5ms/frame) + STFT loss na 48kHz
může mít fundamentálně nedostatečné časové rozlišení pro onset bez ohledu na váhování.
V tom případě attack-loss pomůže jen částečně. Pokud zlepšení bude marginální,
přeskočit rovnou na dev-inharmonicity bez dalšího ladění tohoto kroku.

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
