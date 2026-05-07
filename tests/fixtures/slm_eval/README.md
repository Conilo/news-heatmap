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
