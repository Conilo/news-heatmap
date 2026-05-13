# Extraction design

`src/extract.py` — runs each article through a local Ollama SLM to extract
structured fields (`state`, `municipality`, `group`, `event_type`,
`confidence`) and writes them back to the article record.

---

## Inputs and outputs

| | |
|---|---|
| Input | Article dict with at least `title` and `body` |
| Output | Same dict with SLM-extracted fields merged in, plus `processed_at` timestamp |
| Skipped | Articles with empty `body` receive fallback values (`"Desconocido"` / `"otro"` / `0.0`) without an Ollama call |

Entry point: `extract_articles(articles, skip_urls)` for batches;
`extract_article(article)` for single articles.  `skip_urls` avoids
re-running the SLM for URLs already present in `data/articles.csv`.

---

## SLM call

```python
ollama.chat(
    model=config.MODEL_NAME,
    messages=[system_prompt, user_message],
    format=_OUTPUT_SCHEMA,   # JSON schema with enum constraints
    options={"temperature": 0.0},
)
```

`format=_OUTPUT_SCHEMA` passes a JSON Schema to Ollama's constrained-decoding
layer so the model physically cannot emit values outside the `state` and
`event_type` enums.  Temperature 0 makes sampling deterministic.

The user message is `"Title: {title}\nArticle: {body[:ARTICLE_BODY_MAX_CHARS_SLM]}"`.
The description field is intentionally excluded — empirical analysis showed it
adds noise in ambiguous cases (it is usually a duplicate or subset of the
title).

---

## Prompt structure

The system prompt contains three sections:

1. **Field definitions** — natural-language instructions for each output field,
   including geographic edge cases (Internacional vs. Desconocido, how to
   handle municipal-level hints for state inference).

2. **Priority rules** — ordered rules that enforce deterministic
   `event_type` classification when headline signals are unambiguous.
   Rules are checked in order; the first match wins.

   | # | Type | Headline trigger |
   |---|---|---|
   | 1 | `disturbio` | incendian / queman / bloqueos / narcobloqueo / motín |
   | 2 | `detención` | detienen / detuvo / detenidos / detenidas / detención / detener / detuvieron / detenido / detenida / capturan / capturó / captura / capturado / capturada / capturas / arrestan / arrestados / arrestadas / arresta / arrestó / arrestado / arrestada / aprehenden / aprehendió / aprehendido / vinculan a proceso |
   | 3 | `incautación` | decomisan / aseguran / incautan + quantity |
   | 4 | `extorsión` | extortion demands, forced payments (cobro de piso), or criminal harassment (hostigamiento) against businesses or communities |
   | 5 | `narcotráfico` | trafficking operations, networks, supply chains, or formal trafficking charges |
   | 6 | `homicidio` | asesinado / ejecutado / balaceado / matan / mató / ultimaron |
   | 7 | `muerte` | fallece / muere / muerto (without a deliberate-killing verb) |
   | 8 | `otro` | biographical profile, InSight Crime entry, narcocorrido, political statement, retrospective, public-opinion survey, editorial |

3. **Few-shot examples** — labelled input/output pairs covering the most
   common classification patterns and known ambiguous cases.

---

## Post-SLM validation (`_validate_fields`)

After parsing the JSON response, `_validate_fields` ensures:

- All required keys are present (missing ones fall back to `_FALLBACK`).
- `event_type` and `state` are members of their respective enums; out-of-vocab
  values (e.g. accent or case mismatches that slip past constrained decoding)
  are replaced with `"otro"` / `"Desconocido"` respectively.
- `confidence` is clamped to `[0.0, 1.0]`.

---

## State inference (`_infer_state_from_municipality`)

When the SLM returns `state = "Desconocido"` but provides a recognisable
municipality name, `_maybe_fill_state_from_municipality` maps it to the
corresponding state via `config.LOCATION_TO_STATE` (accent-folded,
longest-match-first).  This recovers geographic signal in cases where the SLM
correctly identifies the city but fails to name the state.

---

## Known limitations and planned improvements

### Priority rules are advisory, not enforced in code

The SLM can ignore prompt priority rules when the article body strongly
contradicts the headline signal.  A common failure pattern observed in
production data:

> A `detención` article (capture of a cartel figure) also describes the
> fires and blockades the capture triggered.  The SLM reads the body and
> outputs `disturbio` instead of `detención`, despite Rule 2 being
> unambiguous from the headline alone.

**Analysis:** The headline alone contains sufficient signal to apply Rules 1–3
deterministically (disturbio, detención, incautación trigger words are
unambiguous).  The body only adds value for the lower-priority types
(narcotráfico, homicidio, extorsión, otro) where the headline wording is less
predictable.

**Planned improvement — title-only classification + SLM disambiguation:**

Split extraction into two stages:

1. **Title classifier** — apply Rules 1–3 from the headline using a fast
   rule-based or lightweight model pass (no body required).  For articles
   where the headline matches a high-confidence rule (e.g. "incendian",
   "capturan", "decomisan"), emit the classification directly without any
   Ollama call.

2. **SLM stage** — run the full body-aware SLM prompt only for articles whose
   headline does not match a high-confidence rule (Rules 4–8 and ambiguous
   cases).  The SLM result can be validated against the title-classifier
   output: if they disagree, surface the conflict as a low-confidence signal
   rather than silently accepting the SLM's answer.

This approach would eliminate the category of body-overrides-headline
misclassifications entirely for Rules 1–3, and reduce Ollama calls for the
unambiguous cases.

---

## Configuration reference

| Key | Description |
|---|---|
| `MODEL_NAME` | Ollama model identifier (e.g. `"llama3.2:3b"`) |
| `ARTICLE_BODY_MAX_CHARS_SLM` | Body truncation limit passed to the SLM (`8000`) |
