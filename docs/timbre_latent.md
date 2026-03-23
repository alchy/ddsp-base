# Multi-Instrument DDSP s latentním timbre vektorem

Rozšíření stávajícího single-instrument modelu na systém, který sdílí váhy sítě
napříč nástroji a reprezentuje barvu zvuku jako spojitý latentní vektor `z`.

Dokument popisuje motivaci, navrhovanou architekturu a postupné kroky implementace.
Zatím **není implementováno** — slouží jako plán pro budoucí fázi.

---

## Motivace

Stávající `DDSPVocoder` trénuje jeden model per nástroj. Timbre je implicitně
zakódován ve vahách GRU a MLP. To má dvě omezení:

1. **Chybějící noty**: pianino natrénované na C3–C6 si musí C1 domyslet
   extrapolací. Děje se to, ale nespolehlivě.
2. **Nový nástroj = nový trénink od nuly**: každých 65 hodin CPU per nástroj.

Latentní timbre vektor tyto problémy řeší:

- Síť se naučí **obecné fyzikální zákony** tvorby timbru (overtone decay, brightness vs. F0)
  sdílené napříč nástroji.
- Každý nástroj je pak jen **jeden bod v latentním prostoru** — nový nástroj
  stačí "zařadit" za cenu pár vzorků.

---

## Navrhovaná architektura

```
Vstup na rámec:
  encode_f0(f0)           →  (B, T, 64)
  encode_velocity(vel)    →  (B, T,  8)
  z (timbre embedding)    →  (B, T, 32)   ← broadcast ze (B, 32) na (B, T, 32)
                              ───────────
  concat                  →  (B, T, 104)

  pre_mlp → GRU → post_mlp → head_harm_L/R, head_noise_L/R

Výstup:
  timbre_signal(t) * loudness_envelope(t)   →  (B, 2, T*HOP)
```

Změny oproti stávajícímu modelu:
- `feat_dim`: 72 → 104 (přidáno 32D z)
- Nový parametr `z_dim` (default: 32)
- Lookup table `instrument_embeddings: nn.Embedding(n_instruments, z_dim)`
- Dataset vrací `instrument_id` (int) místo nic

Počet nových parametrů: pouze `n_instruments × z_dim` = 10 nástrojů × 32 = 320 čísel.
Zbytek sítě se nemění.

---

## Doporučená architektura (piano rodina, 20+ not)

### Zaměření: adaptace nového nástroje, ne interpolace

Pro piano rodinu (grand piano, upright, Rhodes, electric piano) s dostupnými
sample bankami (20+ not, více velocity layers) je optimální **dvoustupňový přístup**:

1. **Multi-instrument trénink** — síť se naučí obecnou fyziku pianin
2. **z-optimalizace** pro nový nástroj — zmrazíš síť, hledáš z gradientním sestupen

VAE regularizace ani audio encoder **nejsou potřeba**:
- VAE řeší problém nesouvisejících nástrojů (piano + organ + flute) — pro piano rodinu
  přirozená podobnost dat zajistí dostatečně hladký prostor sama o sobě
- Audio encoder je užitečný pro 3–5 vzorků — s 20+ notami v NPZ formátu
  je gradient descent na z přesnější a jednodušší

---

## Implementace

### Krok 1 — Multi-instrument trénink (Lookup embedding)

Každý nástroj dostane trénovaný vektor `z_i ∈ R^16`.
Trénujeme vše dohromady: váhy sítě + všechny z najednou.

```python
# Přidáno do DDSPVocoder.__init__:
self.instrument_embeddings = nn.Embedding(n_instruments, z_dim)

# forward dostane instrument_id (B,):
z = self.instrument_embeddings(instrument_id)          # (B, z_dim)
z = z.unsqueeze(1).expand(-1, T, -1)                  # (B, T, z_dim)
feat = torch.cat([f0_enc, vel_enc, z], dim=-1)
# feat_dim: 64 + 8 + 16 = 88
```

Změny oproti stávajícímu kódu: minimální — pouze `feat_dim` a předání `instrument_id`.

Doporučené nástroje pro základ:
1. **Salamander Grand Piano** — reference, největší dataset
2. **Rhodes / Fender Rhodes** — odlišný mechanismus (vidlička), ale podobná F0 charakteristika
3. **Upright piano** — varianta grand, jiný rezonátor
4. (volitelně) **Honky tonk** — detuning, výrazně jiný timbre

Pestrost je důležitá — příliš podobné nástroje síť nenaučí generalizovat.

### Krok 2 — Adaptace nového nástroje (z-optimalizace)

```
Nový nástroj (NPZ extracts, 20+ not):

  1. Načti multi-instrument model (zmrazené váhy)
  2. Inicializuj z_new = průměr existujících z (nebo náhodně)
  3. Gradient descent jen na z_new:
       for step in range(200):
           pred = model(f0, loudness, vel, z=z_new)
           loss = mrstft(pred, target)
           loss.backward()
           optimizer_z.step()   # Adam, lr=0.01, only z_new
  4. Ulož z_new → nový nástroj hotov

Čas: ~5 minut na CPU, ~30 sekund na M5
```

Výsledek: z_new je "souřadnice nového nástroje" v naučeném prostoru pianin.
Síť zná fyziku — hledáš jen správné místo v prostoru.

---

## Datová příprava

Stávající NPZ formát je kompatibilní — stačí přidat:
- `instrument_id` jako integer (mapování nástroj → číslo uloženo v `instrument.json`)
- Multi-instrument `SourceDataset` načítající NPZ ze **všech nástrojů najednou**

Žádná změna NPZ struktury není potřeba.

---

## Jak vypadá latentní prostor po tréninku

Pro piano rodinu se latentní prostor přirozeně uspořádá podle fyzikálního mechanismu
nástroje — bez explicitní supervize. Příklad (2D projekce 16D prostoru přes PCA):

```
        Upright ●
                  ● Salamander Grand
   Bösendorfer ●
                        ● Steinway
                                      ● Rhodes
                                            ● Wurlitzer
                                                  ● DX7 E-Piano
  ◄─────────────────────────────────────────────────────────►
  akustická piana                              elektrická piana
```

Klíčový závěr: nový akustický klavír se při z-optimalizaci přirozeně přitáhne
do levé části prostoru (oblast akustických pian) i bez toho, abys mu to řekl.
Síť "ví", co dělá akustické piano akustickým pianem.

---

## Rozhodnutí pro piano rodinu

Na základě diskuze (20+ not, různé velocity, sample banky podobné Salamanderu):

| Otázka | Rozhodnutí | Důvod |
|--------|-----------|-------|
| z_dim | **16** | Piano rodina je homogenní; větší prostor zbytečný |
| VAE regularizace | **ne** | Podobnost dat zajistí hladký prostor bez ní |
| Audio encoder | **ne** | S 20+ notami v NPZ je z-optimalizace přesnější |
| EnvelopeNet | **per-nástroj** (nezměněno) | Rhodes a grand piano mají různé obálky |
| Nový nástroj | **z-optimalizace** (~5 min) | Extrakce NPZ + gradient descent na z |

---

## Závislost na stávajícím systému

Implementace nevyžaduje změnu:
- NPZ formátu
- EnvelopeNet (funguje per-nástroj nezávisle)
- Inference pipeline (generování z checkpointu)

Přechod je opt-in: stávající single-instrument model zůstane funkční.
Multi-instrument model bude mít jiný checkpoint formát (`model_type: multi`).

---

## Časový odhad

| Krok | CPU notebook | Apple M5 (odhad) |
|------|-------------|-----------------|
| Multi-instrument trénink (3 nástroje, small) | ~200 hodin | ~20–40 hodin |
| z-optimalizace nového nástroje | ~5 minut | ~30 sekund |
| Implementace (kód) | 3–4 hodiny | — |

Trénink na CPU je náročný — multi-instrument je první kandidát na spuštění na M5.
