# HOW2WRITE – Průvodce tvorbou technické dokumentace

Tento dokument popisuje styl a strukturu technické dokumentace používané v projektu UniForms.
Vzorový příklad ke studiu: [`docs/UNIFORMS_JS.md`](UNIFORMS_JS.md) – dokumentace klientské knihovny pro renderování formulářů.

Dokumentace je **návodná** – vede čtenáře od „co to je" přes „jak to funguje" až po „jak to konkrétně použiji".
Není to jen reference. Čtenář by po přečtení dokumentu měl být schopen samostatně vytvořit funkční výstup.

---

## Filosofie: proč → jak → reference

Každý dokument odpovídá na tři otázky **v tomto pořadí**:

1. **Proč to existuje?** – kontext, účel, pro koho je dokument určen
2. **Jak to celkově funguje?** – přehled, datový tok, klíčové koncepty
3. **Jak to konkrétně udělám?** – kroky, příklady, reference

Reference (tabulky klíčů, výčty hodnot, přehled funkcí) patří **na konec**, ne na začátek.

---

## Struktura dokumentu

Každý dokument by měl obsahovat tyto bloky (v tomto pořadí):

```
# Název dokumentu

  Krátký úvod: co to je, k čemu slouží, pro koho je určeno.
  Závislosti a předpoklady.

---

## Co to je a k čemu slouží?   (kontext)

## Jak to funguje?              (přehled, datový tok, ASCII diagram)

## Rychlý start                 (kompletní minimální příklad)

## Krok za krokem               (průvodce pro první použití)

## Přehled konceptů             (každý pojem/typ jako vlastní ### sekce)

## Reference – povinné a volitelné klíče

## Speciální témata             (bezpečnost, výkon, rozšiřitelnost...)

## Přehled API / funkcí         (tabulka)
```

Části, které nejsou relevantní pro daný dokument, vynechej.

---

## Pravidla pro každou `###` sekci konceptu

Každý popisovaný typ, endpoint, sekce nebo příkaz má **tři povinné části**:

### 1. Krátký popis (1–3 věty)

Co to je, kdy se to používá, čím se liší od podobných věcí.
Nepíšeš encyklopedii – píšeš orientaci.

```markdown
### Formulář (`form`)

Nejjednodušší sekce. Pole jsou zobrazena jako dvousloupcový grid – label vlevo, input vpravo.
Volitelný klíč `hint` zobrazí modrý informační box nad formulářem.
```

### 2. Pracovní příklad (kompletní, funkční)

Příklad musí být **kompletní** – čtenář ho může zkopírovat a okamžitě použít.
Nepiš fragmenty (žádné `...` nebo `// zbytek`). Pokud je příklad dlouhý, zkrať ho výběrem representativní podmnožiny, ne zkrácením syntaxe.

```markdown
```jsonc
{
  "id": "reported_by",
  "type": "form",
  "title": "Hlášeno osobou",
  "fields": [
    { "key": "name", "label": "Jméno", "type": "text", "editable": true, "value": null }
  ]
}
```‌
```

### 3. Tabulka klíčů

Tabulka se třemi sloupci: `Klíč | ✓ | Popis`.
Sloupec `✓` označuje povinné klíče; volitelné jsou prázdné.
Pro podmíněně povinné (např. „povinný pokud `allow_append_row: true`") použij `◐`.

```markdown
| Klíč | ✓ | Popis |
|------|:-:|-------|
| `id` | ✓ | Unikátní identifikátor |
| `type` | ✓ | `"form"` |
| `fields[]` | ✓ | Formulářová pole |
| `hint` | | Modrý informační box nad formulářem |
| `append_row_template{}` | ◐ | Povinný pokud `allow_append_row: true` |
```

---

## Formátování kódu

| Situace | Formát |
|---------|--------|
| JSON bez komentářů | ` ```json ` |
| JSON s komentáři (`//`) | ` ```jsonc ` |
| Shell / příkazy | ` ```bash ` |
| Inline klíč nebo hodnota | `` `klíč` `` |
| Název souboru, cesta | `` `data/workbooks/soubor.json` `` |
| HTTP metoda + path | `` `GET /api/v1/cases/` `` |

---

## Datový tok – ASCII diagram

Pro procesy s více kroky použij ASCII diagram – je to rychle srozumitelné:

```markdown
```
JSON šablona
    │
    ▼
Backend: klon šablony → přiřaď case_id → ulož
    │
    ▼
Frontend: Forms4SOC.render(sections, container)
    │
    ▼
Analytik edituje → field.value se aktualizuje živě
    │
    ▼
Uložit → JSON.stringify(doc) → PATCH /api/v1/cases/{id}
```‌
```

---

## Tón a slovník

**Přímý.** Vynechej úvodní fráze typu „V této sekci se dozvíte..." – začni rovnou věcí.

**Praktický.** Po příkladu napiš „Výsledek:", co se stane, co uživatel uvidí.

**Konzistentní.** Jeden pojem = jedno slovo. Přehled termínů pro UniForms:

| Pojem | Používej | Nepoužívej |
|-------|----------|------------|
| JSON soubor záznamu | záznam, dokument | případ, incident (není-li název kolekce) |
| YAML soubor šablony | šablona | template (jako standalone) |
| Uživatel vyplňující formulář | analytik | uživatel, operátor |
| Skupina šablon stejné povahy | kolekce | kategorie, množina |
| Předdefinované hodnoty výběru | options | hodnoty, možnosti výběru |
| Sloupec tabulky | sloupec | column (v UI textu) |

---

## Blockquotes – kdy a jak

Použij `>` pro tři typy poznámek:

```markdown
> Tip nebo doporučení – pozitivní doplněk.

> **Poznámka:** Výjimka nebo upřesnění, které je důležité ale ne kritické.

> **Pozor:** Chyba, která snadno nastane a způsobí problém.
```

Blockquoty nenahrazují tabulky ani příklady. Jeden blockquote ke každé sekci je maximum.

---

## Nadpisy

```markdown
# Název dokumentu            – vždy jeden, na začátku
## Hlavní sekce              – max 6–8 na dokument
### Pojem / typ / endpoint   – každý popis má vlastní ###
```

Nadpisy pišou **bez závorek** s jedinou výjimkou: typ nebo název identifikátoru v backtick závorce je součástí nadpisu:

```markdown
### Tabulka akcí (`table`)
### POST `/api/v1/cases/`
```

---

## Rychlý start – co musí obsahovat

Sekce Rychlý start musí čtenáři umožnit dosáhnout viditelného výsledku za 5 minut:

1. Kompletní minimální příklad (copy-paste funkční)
2. Výsledek – co uvidí / co se stane
3. Odkaz na další kroky

Příklad ze vzorového dokumentu:
```markdown
## Rychlý start

Minimální stránka, která vykreslí formulář z JSON:

```html
<!DOCTYPE html>
...
<script src="forms4soc.js"></script>
<script>
    const doc = { sections: [{ id: "info", type: "form", ... }] };
    Forms4SOC.render(doc.sections, document.getElementById('form-container'));
</script>
```‌

Po načtení stránky se zobrazí karta s nadpisem „Základní informace" a třemi poli.
```

---

## Reference na konci

Referenční tabulky (přehled všech klíčů, všech funkcí, všech endpoint) patří **na konec** dokumentu – jako příloha. Čtenář k nim přichází po tom, co pochopil kontext.

Struktura závěrečné reference:

```markdown
## Povinné a volitelné klíče   (souhrn všech tabulek klíčů z těla dokumentu)
## Přehled funkcí / API        (tabulka)
```

---

## Checklist před vydáním dokumentu

Před commitováním dokumentu ověř:

- [ ] Dokument začíná kontextem – co to je a pro koho
- [ ] Existuje sekce „Jak to funguje?" nebo ekvivalent
- [ ] Existuje „Rychlý start" nebo „Krok za krokem"
- [ ] Každý hlavní koncept má: krátký popis + příklad + tabulku klíčů
- [ ] Příklady jsou kompletní a funkční (ne fragmenty)
- [ ] Tabulky klíčů mají sloupec `✓` pro povinné klíče
- [ ] Reference je na konci, ne na začátku
- [ ] Všechny type identifikátory jsou v backtick: `` `form` ``, ne `form`
- [ ] Terminologie je konzistentní (viz tabulka slovníku výše)

---

## Příklad: špatně vs. správně

### Špatně – referenční styl bez kontextu

```markdown
## Typ `form`

| Klíč | Typ | Popis |
|------|-----|-------|
| id | string | Identifikátor |
| type | string | Musí být "form" |
| fields | array | Pole |
```

### Správně – návodný styl s příkladem

```markdown
### Formulář (`form`)

Nejjednodušší sekce. Pole jsou zobrazena jako dvousloupcový grid – label vlevo, input vpravo.
Volitelný klíč `hint` zobrazí modrý informační box nad formulářem.

```jsonc
{
  "id": "reported_by",
  "type": "form",
  "title": "Hlášeno osobou",
  "hint": "Vyplní SOC Analytik po přijetí hlášení.",
  "fields": [
    { "key": "name", "label": "Jméno", "type": "text", "editable": true, "value": null }
  ]
}
```‌

| Klíč | ✓ | Popis |
|------|:-:|-------|
| `id` | ✓ | Unikátní identifikátor sekce |
| `type` | ✓ | `"form"` |
| `title` | ✓ | Nadpis karty |
| `fields[]` | ✓ | Formulářová pole |
| `hint` | | HTML text zobrazený jako modrý informační box nad formulářem |
```

Rozdíl je zřejmý: návodný styl dává čtenáři **kontext** (proč), **příklad** (jak to vypadá) a **referenci** (co přesně zadat).
