# Sample banky pro trénink latentního timbre prostoru

Všechny nástroje jsou dostupné na **https://sfzinstruments.github.io/pianos/**
ve formátu SFZ + FLAC/WAV, kompatibilním s naším konverzním pipeline.

---

## Doporučený výběr pro piano rodinu

Cíl: 3–4 různé nástroje pokrývající spektrum akustické → elektrické piano.
Pestrost je důležitá — příliš podobné nástroje latentní prostor nenaučí generalizovat.

### Základ (již máme)

| Nástroj | Typ | Vel. vrstvy | Poznámka |
|---------|-----|-------------|---------|
| **Salamander Grand Piano V3** | Akustický grand | 16 | Yamaha C5, referenční kvalita |
| **Vintage Vibe** | Elektrické piano | 8 | Rhodes/Wurlitzer charakter |

### Doporučené přidání

| Nástroj | Typ | Vel. vrstvy | Velikost | Licence | URL |
|---------|-----|-------------|----------|---------|-----|
| **jRhodes3d** | Rhodes MK I (1977) | 5 | 92 MB | CC BY-NC 4.0 | https://sfzinstruments.github.io/pianos/jrhodes3d |
| **Splendid Grand Piano** | Akustický (Steinway) | 4+1 | 77 MB | Public Domain | https://sfzinstruments.github.io/pianos/splendid_grand_piano |
**jRhodes3d** je klíčový — unlooped samples až 25 sekund dávají přirozený decay,
ideální pro trénink. Liší se od Vintage Vibe (jiný rok, jiné ladění).

**Splendid Grand** je jiný akustický grand než Salamander (jiný výrobce, jiná mikrofonní
technika) — síť se naučí, co je společné všem grand pianům, ne jen Yamahou C5.

---

## Volitelné rozšíření

| Nástroj | Typ | Vel. vrstvy | Velikost | Licence | URL |
|---------|-----|-------------|----------|---------|-----|
| Headroom Piano (Yamaha C3) | Akustický grand | 5 | 156 MB | CC BY 4.0 | https://sfzinstruments.github.io/pianos/headroom_piano |
| Maestro Concert Grand (CF-3) | Akustický grand | 5 | 269 MB | Custom (free) | https://sfzinstruments.github.io/pianos/maestro_concert_grand_piano |

**Poznámka k upright pianům**: přestože se fyzicky liší od grandu (vertikální struny,
kratší rezonátor), kvalitně nahrané upright je v nahrávce od grandu k nerozeznání.
Pro latentní prostor by přidal bod příliš blízko existujících grandů — nepřidáváme.

---

## Konverzní pipeline (SFZ → ddsp formát)

Stávající `sfz_convert` tool (použitý pro Salamander) by měl fungovat pro všechny
SFZ nástroje ze sfzinstruments.github.io — používají stejný formát.

Postup pro nový nástroj:
```
1. Stáhnout ZIP z GitHubu (odkaz na stránce nástroje)
2. Rozbalit do C:\SoundBanks\SFZ\<nastroj>\
3. Spustit sfz_convert → C:\SoundBanks\ddsp\<nastroj>\
4. python ddsp.py extract --instrument C:\SoundBanks\ddsp\<nastroj>
```

Název souboru musí obsahovat MIDI číslo ve formátu `mXXX` pro known-F0 mod
(rychlá extrakce). sfz_convert toto zajišťuje automaticky.

---

## Priorita stažení

Pro první multi-instrument trénink (latentní prostor fáze 1):

1. **jRhodes3d** — 92 MB (zásadní — Rhodes vidlička vs. piano struna)
2. **Splendid Grand** — 77 MB (druhý akustický grand, jiný výrobce)

Celkem ~170 MB navíc. S Salamanderem a Vintage Vibe: **4 nástroje, ~2 GB dat**.

**Wurlitzer vynechán**: rákosový mechanismus je fyzikálně příliš odlišný od
struna/vidlička rodiny — v latentním prostoru by tvořil izolovaný shluk a
interpolace s ostatními nástroji by nedávala smysl. Vhodný jako samostatný
single-instrument model.
