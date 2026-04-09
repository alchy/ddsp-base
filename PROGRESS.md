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
| `C:\SoundBanks\IthacaPlayer\generated\ks-grand\` | small, max_crop=50 | Bas nepřijatelný — noise bomb, tmavý tunel |
| `C:\SoundBanks\IthacaPlayer\generated\ks-grand-q38\` | **medium, max_crop=200, ep 38** | Bas hluboký ✓, ale přehnaný + šum across all notes |

---

## Perceptuální hodnocení ep 38 (9. dubna 2026)

### Co se zlepšilo
- **Bas má hloubku** — max_crop fix funguje. Předchozí model měl "tmavý tunel",
  teď je reálná basová hloubka přítomna. Dvoustupňový decay se model naučil.

### Přetrvávající problémy

**1. Šum napříč celým rozsahem**

Příčina: NoiseSynth stále funguje jako záchranná síť — všude kde harmonická
syntéza není jistá, přidá šum. Ep 38 z 500 je příliš brzo. Druhá příčina:
chybí **NoiseSynth temporal gating** — šum se přidává po celou dobu tónu,
i v sustain fázi kde fyzikálně nepatří. Soundboard shock decay je rychlý,
trvá ~200 ms, pak by mělo být ticho (nebo jen harmonické).

**2. Bas příliš akcentovaný / mohutný**

Pravděpodobné příčiny:
- Dvoustupňový decay σ_slow je v basu příliš silný — model přehnal pomalou složku
- MRSTFT FFT=16384 nutí model agresivně posilovat každou basovou harmonickou
- NoiseSynth přidává nízko frekvenční šum na vrch harmonické složky

**Okamžitý workaround:** generovat s `--decay-scale 0.7` ztlumí fyzikální decay.

---

## Závěry a priority pro další vývoj

### Co víme jistě
1. `max_crop=200` (1.07 s) je správná cesta — bas získal hloubku
2. Šum bude klesat s dalšími epochami — trénink je na 38/500
3. NoiseSynth temporal gating je **akutní priorita** — bez něj bude šum
   přetrvávat bez ohledu na počet epoch

### Co dál (seřazeno dle priority)

1. **Nechat dotrénovat** — konvergence je aktivní, zbývá 462 epoch.
   Harmonická větev bude jistější → NoiseSynth ustoupí přirozeně.

2. **NoiseSynth temporal gating** *(střední obtížnost, vysoký dopad)*
   Šum jen v attack fázi (0–200 ms), po té pouze harmonické.
   Implementace: přidat note-age jako kondicionovací vstup NoiseSynthu.

3. **Beating / unison-spread** *(nízká obtížnost, vysoký dopad)*
   Druhá oscilátorová banka na F₀ + δF. Tiny FC (F0, velocity) → δF.
   Dá basu "živost" místo statického sustain. Téměř nulová výpočetní cena.

4. **Per-partial frekvenčně závislý decay** *(střední obtížnost)*
   MLP predikuje d_k pro každý partial zvlášť.
   Fyzikální prior: d_k ∝ b₁ + b₃·k² (vyšší parciály utichají rychleji).

5. **Large model na GPU** *(až bude přístup k M5 nebo CUDA)*
   4.4M params, adaptive max_crop, 1000 epoch = produkční kvalita.

6. **Phantom partial modul** *(vysoká obtížnost, jen pro C3 a níže)*
   Sekundární additive bank na f_m + f_n, |f_m − f_n|.
