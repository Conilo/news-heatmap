"""Opt-in live Ollama evaluation against curated snapshots in fixtures/slm_eval/."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from tqdm import tqdm

import config
from geo_normalize import normalize_state
from src import extract

CASES_CSV = Path(__file__).resolve().parent / "fixtures" / "slm_eval" / "cases.csv"
STATE_ACCURACY_FLOOR = 0.95 - 1e-9
OPTIONAL_EXPECTED_FIELDS = (
    ("expected_municipality", "municipality"),
    ("expected_group", "group"),
    ("expected_event_type", "event_type"),
)


def _optional_field_matches(exp_col: str, exp_raw: str, got_raw: str) -> bool:
    """Alignment with pipeline: group via normalize_group; otherwise accent-fold + casefold."""
    if exp_col == "expected_group":
        return config.normalize_group(exp_raw) == config.normalize_group(got_raw)
    ke = config.strip_accents(exp_raw).casefold()
    kg = config.strip_accents(got_raw).casefold()
    return ke == kg


def _load_case_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_to_article(row: dict[str, str]) -> dict[str, str]:
    return {col: str(row.get(col, "") or "") for col in config.CSV_COLUMNS}


def test_optional_field_matches_accent_and_group_canonicalization() -> None:
    assert _optional_field_matches("expected_municipality", "Culiacán", "Culiacan")
    assert _optional_field_matches("expected_event_type", "detención", "detencion")
    assert _optional_field_matches(
        "expected_group", "Cartel de Sinaloa", "Cártel de Sinaloa"
    )
    assert _optional_field_matches(
        "expected_group", "CJNG", "Cártel Jalisco Nueva Generación"
    )


@pytest.mark.slm_live
def test_slm_eval_batch() -> None:
    """One batched evaluation: each eligible row invokes extract_article (N Ollama calls)."""

    rows = _load_case_rows(CASES_CSV)
    eligible = [
        r
        for r in rows
        if (r.get("expected_state") or "").strip()
    ]
    if not eligible:
        pytest.skip(
            f"No labelled cases in {CASES_CSV} — fill expected_state rows or run the picker in "
            "tests/fixtures/slm_eval/README.md",
        )

    state_hits = 0
    optional_hits = {exp: 0 for exp, _ in OPTIONAL_EXPECTED_FIELDS}
    optional_totals = {exp: 0 for exp, _ in OPTIONAL_EXPECTED_FIELDS}
    failures: list[str] = []
    optional_misses: list[str] = []

    n_eligible = len(eligible)
    _use_progress = sys.stderr.isatty()
    bar = tqdm(
        eligible,
        desc="SLM eval",
        unit="case",
        total=n_eligible,
        file=sys.stderr,
        disable=not _use_progress,
    )
    for row in bar:
        case_id = (row.get("case_id") or "").strip() or "?"
        if _use_progress:
            bar.set_postfix_str(case_id, refresh=False)
        article = _row_to_article(row)
        out = extract.extract_article(dict(article))

        exp_state = normalize_state((row.get("expected_state") or "").strip())
        got_state = normalize_state(str(out.get("state", "")))
        if got_state == exp_state:
            state_hits += 1
        else:
            failures.append(
                f"  [{case_id}] state expected={exp_state!r} got={got_state!r} "
            )

        for exp_col, actual_key in OPTIONAL_EXPECTED_FIELDS:
            exp_raw = (row.get(exp_col) or "").strip()
            if not exp_raw:
                continue
            optional_totals[exp_col] += 1
            got_raw = str(out.get(actual_key, "") or "").strip()
            if _optional_field_matches(exp_col, exp_raw, got_raw):
                optional_hits[exp_col] += 1
            else:
                optional_misses.append(
                    f"  [{case_id}] {exp_col} expected={exp_raw!r} got={got_raw!r}"
                )

    n = n_eligible
    acc = state_hits / n
    lines = [
        f"SLM eval state accuracy: {state_hits}/{n} ({acc:.3f})",
    ]
    for exp_col, _akey in OPTIONAL_EXPECTED_FIELDS:
        t = optional_totals[exp_col]
        if t:
            h = optional_hits[exp_col]
            lines.append(f"  {exp_col}: {h}/{t} ({h/t:.3f})")
    summary = "\n".join(lines)

    detail = ""
    if failures:
        detail += "State mismatches:\n" + "\n".join(failures) + "\n"
    if optional_misses:
        detail += "Optional field mismatches:\n" + "\n".join(optional_misses) + "\n"

    if acc < STATE_ACCURACY_FLOOR:
        pytest.fail("\n" + summary + "\n" + detail)

    if optional_misses:
        print("\n" + summary + "\n(Warning) \n" + detail)
    else:
        print("\n" + summary)
