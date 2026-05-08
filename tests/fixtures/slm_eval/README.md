# Live SLM evaluation (`slm_eval`)

Curated article snapshots for **opt-in** regression testing against the real Ollama model (`user_config.MODEL_NAME`).

## Golden file: `cases.csv`

- **Input columns** (feed `extract_article`): all of `advanced_config.CSV_COLUMNS` — at minimum `url`, `title`, `description`, `body`, `published_date`, `source` populated from your cache.
- **Review columns**
  - `case_id` — short stable id (e.g. `c00`, `c01` from the labeling UI).
  - `expected_state` — **human-verified** ground truth (do not trust a prior SLM column without reading the article).
  - `notes` — optional.
- **Optional** (for extra printed stats in the eval test; **only `expected_state` gates pass/fail** today): `expected_municipality`, `expected_group`, `expected_event_type`.

Tests **skip** if no row has a non-empty `expected_state`.

## Labeling in Streamlit (preferred)

1. Run the dashboard: `streamlit run src/dashboard.py`.
2. Open **Label eval cases** in the sidebar (multipage app under `src/pages/`).
3. Filter and pick articles from `data/articles.csv`, correct **especially `event_type`** when the SLM is wrong, and **Add to golden buffer**.
4. **Download cases CSV** and merge rows into this file (`tests/fixtures/slm_eval/cases.csv`), keeping the header row.

The UI pre-fills expectations from current SLM fields so you only change what is wrong.

**State and event type** fields are dropdowns aligned with `src/extract.py` (`VALID_STATES`, `VALID_EVENT_TYPES`), plus an optional `(custom)` row for edge cases.

The golden buffer is saved automatically to **`data/slm_eval_label_draft.csv`** (same gitignore bucket as other `data/*.csv`), so it survives **Streamlit or server restarts**. **Clear golden buffer** removes that file.

## Running the evaluation

Requires: Ollama running locally, model available for `MODEL_NAME`.

```bash
pytest tests/test_extract_slm_eval.py --slm-live -v
```

Pass rule: **`correct_state / eligible_rows >= 0.95`** (with a tiny float epsilon). With **20** labelled rows, **at most one** wrong state passes the threshold.

Increase row count in `cases.csv` for a stronger regression signal.

## Legacy script

The old command-line candidate picker under `scripts/` was removed; use the Streamlit page above.

---

## Governor-tier tests: `governor_cases.csv`

A second fixture file, inspired by MOGO's validation strategy (Coscia & Rios, CIKM 2012). In MOGO, governors were used as a proxy ground truth: since they are known to operate only in their own states, correctly assigning them validated the geo-extraction machinery before applying it to the unknown problem (cartel territories). Here, each case is a headline where the correct `state` **and** `event_type` are unambiguous to any human reader — the trigger keyword appears explicitly in the title and the state is named directly or via an unmistakable city.

### Test file: `tests/test_extract_governor.py`

Two tests are included:

- `test_governor_fixture_is_valid` — always runs (no Ollama). Validates that every row has a non-empty `title`, `body`, `expected_state`, and `expected_event_type`, and that each value is a member of `VALID_STATES` / `VALID_EVENT_TYPES`. Catches typos before any live run is attempted.

- `test_governor_slam_dunks` — `--slm-live` only. Calls `extract_article` for each row and requires **100% accuracy** on `state` and `event_type`. All failures are collected before reporting (not fail-fast), so a single run shows the full picture. Municipality mismatches are printed as warnings but do not fail the test.

### Threshold semantics

| Test | Fixture | Threshold | A failure means… |
|---|---|---|---|
| `test_extract_slm_eval.py` | `cases.csv` | ≥ 95% state accuracy | Model drift on hard/ambiguous cases |
| `test_extract_governor.py` | `governor_cases.csv` | **100%** state + event_type | Extractor is broken |

Run the governor tier first after any prompt or model change. If it fails, fix the extractor before running the full eval.

### Running

```bash
pytest tests/test_extract_governor.py                  # fixture check only (no Ollama)
pytest tests/test_extract_governor.py --slm-live -v    # full evaluation
```

### Case breakdown

The 16 cases cover every priority rule in `src/extract.py`, with body text deliberately written to introduce the most common confusion signal for each rule:

| Case(s) | Rule | Body noise introduced |
|---|---|---|
| g01 — `narcobloqueo` in Culiacán | Rule 1 disturbio | Body describes the arrest that triggered the disturbio |
| g02 — `incendian` in Celaya | Rule 1 disturbio | No confound — clean baseline |
| g03 — `bloqueos`+`quema` in Chilpancingo | Rule 1 over Rule 2 | Body reads entirely like a detención article |
| g04 — `capturan` in Mazatlán | Rule 2 detención | Body describes fentanilo trafficking routes |
| g05 — `detienen` in Monterrey | Rule 2 detención | Body describes the homicide that prompted the arrest |
| g06 — `decomisan` in Hermosillo | Rule 3 incautación | Body includes arrests of driver and passengers |
| g07 — `aseguran` in Manzanillo | Rule 3 incautación | Body includes detentions |
| g08 — `ejecutan` in Zamora | Rule 5 homicidio | Straightforward — no confound |
| g09 — `matan` in Ciudad Juárez | Rule 5 homicidio | Straightforward — no confound |
| g10 — `fallece` in Tijuana | Rule 6 muerte | Body mentions balazo and police pursuit |
| g11 — `acusa` in Tamaulipas | Rule 4 narcotráfico | Body has corruption framing (sobornos) |
| g12 — InSight Crime profile | Rule 7 otro, Desconocido state | Body names Sinaloa as general operating territory |
| g13 — opinion survey | Rule 7 otro, Desconocido state | No event, no location |
| g14 — DEA `arresta` in Chicago | Internacional + detención | Body is about money laundering / narcotráfico |
| g15 — `muere` in Culiacán | Rule 6 muerte | Body confirms balazo but intent/cause ambiguous |
| g16 — `fallece` in Reynosa | Rule 6 muerte | Cause undetermined; no killing verb anywhere |

### Adding new cases

Add a row to `governor_cases.csv` following the same column schema as `cases.csv`. Required fields: `case_id`, `title`, `body`, `expected_state`, `expected_event_type`. Design criteria:

1. The correct answer must be **obvious from the headline alone** — if a reasonable person would hesitate, it belongs in `cases.csv` instead.
2. The body should introduce **at least one plausible confusion signal** — a pure positive with no noise doesn't stress-test anything.
3. Include a `notes` field explaining which priority rule is being tested and what body noise was added.

The fixture-validation test (`test_governor_fixture_is_valid`) will catch invalid enum values immediately without needing Ollama.
