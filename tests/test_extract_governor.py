"""Governor-tier deterministic SLM tests.

Inspired by MOGO's validation strategy (Coscia & Rios, 2012): rather than
testing the unknown directly, validate the extractor against cases where the
correct answer is unambiguous from the headline alone — the "governor" analogy.

In MOGO, governors were used as a proxy: they are known to operate only in their
own states, so a correct extraction confirmed the geo machinery worked before
applying it to the unknown problem (cartel territories).

Here, each case is a headline whose correct state AND event_type is unambiguous
to any human reader. The trigger keyword for the event_type priority rule appears
explicitly in the title; the state is named directly or via an unmistakable city.

Threshold: 100% — a failure here means the extractor is broken, not merely
imprecise. This is structurally different from test_extract_slm_eval.py, which
uses a probabilistic floor (95%) on harder, ambiguous cases.

Run:
    pytest tests/test_extract_governor.py                  # fixture check only
    pytest tests/test_extract_governor.py --slm-live       # full evaluation
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

import config
from advanced_config import VALID_EVENT_TYPES, VALID_STATES
from geo_normalize import normalize_state
from src import extract

GOVERNOR_CSV = (
    Path(__file__).resolve().parent / "fixtures" / "slm_eval" / "governor_cases.csv"
)

_VALID_STATES_SET = frozenset(VALID_STATES)
_VALID_EVENT_TYPES_SET = frozenset(VALID_EVENT_TYPES)


# ---------------------------------------------------------------------------
# Helpers (mirrors test_extract_slm_eval.py conventions)
# ---------------------------------------------------------------------------

def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_to_article(row: dict[str, str]) -> dict[str, str]:
    return {col: str(row.get(col, "") or "") for col in config.CSV_COLUMNS}


# ---------------------------------------------------------------------------
# Test 1 — fixture integrity (no Ollama, always runs)
# ---------------------------------------------------------------------------

def test_governor_fixture_is_valid() -> None:
    """Validates the CSV structure and expected values without touching Ollama.

    Catches typos in state/event_type names and missing required fields before
    any live run is attempted.
    """
    rows = _load_rows(GOVERNOR_CSV)
    assert rows, f"governor_cases.csv is empty or missing at {GOVERNOR_CSV}"

    for row in rows:
        cid = row.get("case_id", "?")

        assert row.get("title", "").strip(), f"[{cid}] missing title"
        assert row.get("body", "").strip(), f"[{cid}] missing body"

        exp_state = row.get("expected_state", "").strip()
        assert exp_state, f"[{cid}] missing expected_state"
        assert exp_state in _VALID_STATES_SET, (
            f"[{cid}] expected_state {exp_state!r} not in VALID_STATES"
        )

        exp_event = row.get("expected_event_type", "").strip()
        assert exp_event, f"[{cid}] missing expected_event_type"
        assert exp_event in _VALID_EVENT_TYPES_SET, (
            f"[{cid}] expected_event_type {exp_event!r} not in VALID_EVENT_TYPES"
        )


# ---------------------------------------------------------------------------
# Test 2 — slam-dunk live evaluation (requires --slm-live)
# ---------------------------------------------------------------------------

@pytest.mark.slm_live
def test_governor_slam_dunks() -> None:
    """Calls extract_article for each governor case and asserts 100% accuracy.

    State and event_type are hard assertions — any miss fails the test
    immediately with a full report of all failures (collect-all, not fail-fast).

    Municipality is a soft check: mismatches are printed as warnings but do not
    fail the test, since municipality extraction is inherently harder and the
    governor analogy only requires the geo machinery (state) and classification
    (event_type) to be correct.
    """
    rows = _load_rows(GOVERNOR_CSV)
    if not rows:
        pytest.skip(f"No governor cases found at {GOVERNOR_CSV}")

    failures: list[str] = []
    mun_warnings: list[str] = []

    use_progress = sys.stderr.isatty()
    if use_progress:
        try:
            from tqdm import tqdm
            rows = list(tqdm(rows, desc="Governor eval", unit="case", file=sys.stderr))
        except ImportError:
            pass

    for row in rows:
        cid = row.get("case_id", "?")
        article = _row_to_article(row)
        out = extract.extract_article(dict(article))

        # ── Hard checks: state + event_type ──────────────────────────────────
        exp_state = normalize_state(row["expected_state"].strip())
        got_state = normalize_state(str(out.get("state", "")))

        exp_event = row["expected_event_type"].strip()
        got_event = str(out.get("event_type", "")).strip()

        state_ok = got_state == exp_state
        event_ok = got_event == exp_event

        if not state_ok or not event_ok:
            parts = []
            if not state_ok:
                parts.append(f"state expected={exp_state!r} got={got_state!r}")
            if not event_ok:
                parts.append(f"event_type expected={exp_event!r} got={got_event!r}")
            note = row.get("notes", "")
            failures.append(f"  [{cid}] {'; '.join(parts)}")
            if note:
                failures.append(f"         rule: {note}")

        # ── Soft check: municipality ──────────────────────────────────────────
        exp_mun = row.get("expected_municipality", "").strip()
        if exp_mun and config.strip_accents(exp_mun).casefold() != "desconocido":
            got_mun = str(out.get("municipality", "")).strip()
            if config.strip_accents(exp_mun).casefold() != config.strip_accents(got_mun).casefold():
                mun_warnings.append(
                    f"  [{cid}] municipality expected={exp_mun!r} got={got_mun!r}"
                )

    n = len(rows)
    n_failed = len(failures)
    n_passed = n - n_failed

    summary_lines = [f"Governor eval: {n_passed}/{n} passed (threshold: 100%)"]
    if mun_warnings:
        summary_lines.append(
            f"Municipality warnings ({len(mun_warnings)} soft):\n" + "\n".join(mun_warnings)
        )
    summary = "\n".join(summary_lines)

    if failures:
        pytest.fail(
            f"\n{summary}\n\n"
            f"BROKEN EXTRACTOR — {n_failed} case(s) failed (must be 0):\n"
            + "\n".join(failures)
        )

    print(f"\n{summary}")
