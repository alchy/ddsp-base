# Projekt: Co se dělo a kde jsme

## Projekt jednou větou

DDSP Neural Vocoder — neuronová síť, která se naučí zvuk klavíru z nahrávek
a pak ho dokáže syntetizovat pro libovolnou notu a dynamiku. Výstup je zvuková
banka pro IthacaPlayer.

---

## Proč to trvalo tak dlouho

**Problém:** Předchozí model zněl v basu (A0–C3) špatně — "noise bomb", tmavý
tunel, statický šum. Po analýze ~15 vědeckých paperů o fyzice klavíru jsme
identifikovali příčinu:

Model dostával trénovací okna o délce **267 ms = 7 cyklů A0**. Jenže basová
struna dokmitá za **5–8 sekund**. Model nikdy neviděl celý průběh dozvuku,
takže se ho nenaučil a přesunul energii do šumového syntetizátoru.

**Řešení:** Prodloužit okna na **1.07 s = 28 cyklů A0**, zvětšit model na
medium (2.1M parametrů) a obnovit nejdetailnější spektrální analýzu
(FFT 16384 = rozlišení 2.93 Hz pro bas).

---

## Fyzikální základ problému

Hluboký bas je perceptuálně nejnáročnější část klavíru. Pět klíčových jevů
(Bank & Chabassier 2019), které musí model zachytit:

1. **Dvoustupňový decay** — každá struna vibruje ve dvou polarizacích.
   Vertikální utichne za ~0.5 s, horizontální za ~5 s. Implementováno v modelu:
   `d_k(t) = α·exp(−σ_fast·t) + (1−α)·exp(−σ_slow·t)` per partial.

2. **Beating** — basové klávesy mají 2–3 struny mírně rozladěné → AM modulace
   1–5 Hz → živost tónu. Bez toho zní staticky. *Plánováno (dev-unison-spread).*

3. **Frekvenčně závislý útlum** — vyšší harmonické utichají rychleji.

4. **Phantom partialy** — geometrická nelinearita generuje frekvence mimo
   harmonickou řadu. Viditelné od C3 dolů.

5. **Precursor transients** — longitudinální vlny dorazí ~10 ms před příčnými.

---

## Co bylo spuštěno (31. března 2026)

```
python ddsp.py learn --instrument C:/SoundBanks/ddsp/ks-grand --preset piano-cpu-quality
```

Preset `piano-cpu-quality` (`model-presets/piano-cpu-quality.json`):
- Model: **medium** (2 080 524 parametrů)
- max_crop: **200 snímků = 1.07 s**
- MRSTFT FFT: 256 / 1024 / 4096 / **16384**
- Zařízení: CPU, batch_size=4, lr=0.0003, 500 epoch

---

## Průběh konvergence (ke dni 9. dubna 2026)

| Epocha | Val loss | Poznámka |
|--------|----------|----------|
| 0 | 2.0580 | Start |
| 5 | 1.8967 | — |
| 8 | 1.8668 | Ep 6–8 trvaly ~10 h (systém pod zátěží) |
| 17 | 1.8612 | Rychlost se ustálila ~4.1 h/ep |
| 25 | 1.8531 | LR začal klesat (CosineAnnealing) |
| 33 | 1.8461 | — |
| **38** | **1.8398** | **Aktuální best** |

Val loss klesá stabilně. LR: 3.00e-04 → 2.96e-04. Zbývá 462 epoch z 500.

---

## Vygenerované výstupy

| Výstup | Model | Hodnocení |
|--------|-------|-----------|
| `C:\SoundBanks\IthacaPlayer\generated\ks-grand\` | small, max_crop=50 | Bas nepřijatelný |
| `C:\SoundBanks\IthacaPlayer\generated\ks-grand-q38\` | **medium, max_crop=200, ep 38** | K poslechu |

---

## Co dál

### Krátkodobě
- Perceptuální hodnocení `ks-grand-q38` — fokus na bas m021–m039
- Pokračování tréninku

### Střednědobě (seřazeno dle priority)
1. **Per-partial frekvenčně závislý decay** — MLP predikuje d_k pro každý
   partial zvlášť; fyzikální prior: d_k ∝ b₁ + b₃·k²
2. **Beating / unison-spread** — druhá oscilátorová banka na F₀ + δF;
   tiny FC (F0, velocity) → δF; téměř nulová výpočetní cena, velký dopad
3. **NoiseSynth temporal gating** — šum jen v attack fázi (0–200 ms)

### Dlouhodobě
- Large model (4.4M params) na GPU
- Phantom partial modul pro C3 a níže
- Export SFZ/Kontakt banky
