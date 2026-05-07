"""Tests for src/extract.py — SLM-based structured-data extraction."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytest

from src import extract


FIXTURE_STATE_INFERENCE_CSV = (
    Path(__file__).resolve().parent / "fixtures" / "articles_state_inference_sample.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_response(content: str) -> dict:
    """Shape a fake `ollama.chat` return value the way the real client returns it."""
    return {"message": {"content": content}}


def _make_fake_chat(content: str):
    """Build a fake `ollama.chat` that always returns the same content string."""
    def fake_chat(*args, **kwargs):
        return _ollama_response(content)
    return fake_chat


# ---------------------------------------------------------------------------
# Group A — _parse_json_response()
# ---------------------------------------------------------------------------

def test_parse_json_response_parses_plain_json():
    assert extract._parse_json_response('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_json_response_strips_markdown_fences():
    fenced = '```json\n{"state": "Sinaloa", "confidence": 0.9}\n```'
    assert extract._parse_json_response(fenced) == {
        "state": "Sinaloa",
        "confidence": 0.9,
    }


def test_parse_json_response_extracts_embedded_object():
    prose = 'Sure! Here is the result: {"a": 1} hope this helps.'
    assert extract._parse_json_response(prose) == {"a": 1}


def test_parse_json_response_returns_empty_on_garbage():
    assert extract._parse_json_response("not json at all, no braces here") == {}


# ---------------------------------------------------------------------------
# Group B — _validate_fields()
# ---------------------------------------------------------------------------

def test_validate_fields_fills_missing_keys_from_fallback():
    result = extract._validate_fields({})
    assert result == {
        "state": "Desconocido",
        "municipality": "Desconocido",
        "group": "Desconocido",
        "event_type": "otro",
        "confidence": 0.0,
    }


def test_validate_fields_treats_none_empty_and_null_string_as_missing():
    result = extract._validate_fields({
        "state": None,
        "municipality": "",
        "group": "null",
        "event_type": "homicidio",
        "confidence": 0.5,
    })
    assert result["state"] == "Desconocido"
    assert result["municipality"] == "Desconocido"
    assert result["group"] == "Desconocido"
    assert result["event_type"] == "homicidio"
    assert result["confidence"] == 0.5


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.7", 0.7),
        (1.5, 1.0),
        (-0.5, 0.0),
        ("abc", 0.0),
        (None, 0.0),
    ],
)
def test_validate_fields_clamps_and_coerces_confidence(raw, expected):
    result = extract._validate_fields({"confidence": raw})
    assert result["confidence"] == expected


@pytest.mark.parametrize("bad_type", [
    "multihomicidio",
    "feminicidio",
    "narcocorrupción",
    "colaboración con el narcotráfico",
    "unknown_category",
])
def test_validate_fields_rejects_invalid_event_type(bad_type, capsys):
    result = extract._validate_fields({"event_type": bad_type, "confidence": 0.9})
    assert result["event_type"] == "otro"
    assert bad_type in capsys.readouterr().out


@pytest.mark.parametrize("bad_state", [
    "EEUU",
    "EE.UU.",
    "Estados Unidos",
    "México",
    "USA",
])
def test_validate_fields_rejects_invalid_state(bad_state, capsys):
    result = extract._validate_fields({"state": bad_state, "confidence": 0.5})
    assert result["state"] == "Desconocido"
    assert bad_state in capsys.readouterr().out


@pytest.mark.parametrize("valid_type", extract.VALID_EVENT_TYPES)
def test_validate_fields_accepts_all_valid_event_types(valid_type):
    result = extract._validate_fields({"event_type": valid_type, "confidence": 0.8})
    assert result["event_type"] == valid_type


@pytest.mark.parametrize("valid_state", extract.VALID_STATES)
def test_validate_fields_accepts_all_valid_states(valid_state):
    result = extract._validate_fields({"state": valid_state, "confidence": 0.8})
    assert result["state"] == valid_state


# ---------------------------------------------------------------------------
# Group C — extract_article() (with stubbed ollama.chat)
# ---------------------------------------------------------------------------

def _sample_article() -> dict:
    return {
        "url": "https://example.com/1",
        "title": "Cártel de Sinaloa ejecuta a 3 personas en Culiacán",
        "description": "Sicarios abrieron fuego en Culiacán, Sinaloa.",
        "body": "Reportes indican enfrentamiento en el centro de Culiacán con tres víctimas.",
        "published_date": "2024-01-01",
        "source": "El Universal",
    }


def _sample_article_no_geo_hints() -> dict:
    return {
        "url": "https://example.com/2",
        "title": "Autoridades reportan una detención en operativo rutinario",
        "description": "Elementos federales dieron cuenta de cargos formulados ante el Ministerio Público.",
        "body": "El comunicado oficial no ubicó colonias específicas en el norte del país.",
        "published_date": "2024-01-02",
        "source": "Generic Wire",
    }


def test_extract_article_merges_slm_fields_and_adds_processed_at(monkeypatch):
    monkeypatch.setattr(
        extract.ollama,
        "chat",
        _make_fake_chat(
            '{"state": "Sinaloa", "municipality": "Culiacán", "group": "CDS", '
            '"event_type": "homicidio", "confidence": 0.95}'
        ),
    )

    article = _sample_article()
    result = extract.extract_article(article)

    # Original article fields preserved
    assert result["url"] == article["url"]
    assert result["title"] == article["title"]
    assert result["source"] == article["source"]

    # Extracted fields merged in
    assert result["state"] == "Sinaloa"
    assert result["municipality"] == "Culiacán"
    assert result["group"] == "CDS"
    assert result["event_type"] == "homicidio"
    assert result["confidence"] == 0.95

    # processed_at is a valid ISO-format timestamp
    assert "processed_at" in result
    datetime.fromisoformat(result["processed_at"])


def test_extract_article_handles_markdown_fenced_response(monkeypatch):
    fenced = (
        '```json\n'
        '{"state": "Jalisco", "municipality": "Guadalajara", "group": "CJNG", '
        '"event_type": "narcotráfico", "confidence": 0.8}\n'
        '```'
    )
    monkeypatch.setattr(extract.ollama, "chat", _make_fake_chat(fenced))

    result = extract.extract_article(_sample_article())

    assert result["state"] == "Jalisco"
    assert result["municipality"] == "Guadalajara"
    assert result["group"] == "CJNG"
    assert result["event_type"] == "narcotráfico"
    assert result["confidence"] == 0.8


def test_extract_article_falls_back_on_ollama_exception(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("ollama is down")

    monkeypatch.setattr(extract.ollama, "chat", boom)

    article = _sample_article_no_geo_hints()
    result = extract.extract_article(article)
    out = capsys.readouterr().out

    # Original fields preserved
    assert result["url"] == article["url"]
    assert result["title"] == article["title"]

    # Fallback values applied
    assert result["state"] == "Desconocido"
    assert result["municipality"] == "Desconocido"
    assert result["group"] == "Desconocido"
    assert result["event_type"] == "otro"
    assert result["confidence"] == 0.0

    # processed_at still set
    datetime.fromisoformat(result["processed_at"])

    # Warning printed
    assert "[extract] Warning" in out


def test_extract_article_skips_slm_when_body_empty(monkeypatch, capsys):
    calls = []

    def fake_chat(*args, **kwargs):
        calls.append(1)
        return _ollama_response("{}")

    monkeypatch.setattr(extract.ollama, "chat", fake_chat)

    article = {**_sample_article_no_geo_hints(), "body": ""}
    result = extract.extract_article(article)

    assert calls == []
    assert result["state"] == "Desconocido"
    assert result["confidence"] == 0.0
    assert "Skipping SLM" in capsys.readouterr().out


def test_extract_article_skips_slm_when_body_whitespace_only(monkeypatch):
    calls = []

    def fake_chat(*args, **kwargs):
        calls.append(1)
        return _ollama_response("{}")

    monkeypatch.setattr(extract.ollama, "chat", fake_chat)

    article = {**_sample_article_no_geo_hints(), "body": "  \n\t  "}
    result = extract.extract_article(article)

    assert calls == []
    assert result["group"] == "Desconocido"


def test_extract_article_fills_state_when_slm_returns_desconocido(monkeypatch):
    monkeypatch.setattr(
        extract.ollama,
        "chat",
        _make_fake_chat(
            '{"state": "Desconocido", "municipality": "Culiacán", "group": "Desconocido", '
            '"event_type": "otro", "confidence": 0.2}'
        ),
    )
    result = extract.extract_article(_sample_article())
    assert result["state"] == "Sinaloa"


def test_infer_state_from_fixture_csv_matches_expected_state():
    with FIXTURE_STATE_INFERENCE_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mun_col = row.get("municipality") or ""
            got = extract._infer_state_from_municipality(mun_col)
            exp = (row.get("expected_state") or "").strip()
            if exp:
                assert got == exp, (mun_col, got, exp)
            else:
                assert got is None, (mun_col, got)


def test_infer_state_from_municipality_handles_substrings():
    assert extract._infer_state_from_municipality("centro de Culiacán") == "Sinaloa"
# ---------------------------------------------------------------------------
# Group D — extract_articles()
# ---------------------------------------------------------------------------

def test_extract_articles_skips_urls_in_skip_set(monkeypatch):
    calls: list[str] = []

    def fake_chat(*args, **kwargs):
        # Track which titles the SLM was actually invoked for via the user message.
        user_msg = kwargs["messages"][1]["content"]
        calls.append(user_msg)
        return _ollama_response(
            '{"state": "Sinaloa", "municipality": "Culiacán", "group": "CDS", '
            '"event_type": "homicidio", "confidence": 0.9}'
        )

    monkeypatch.setattr(extract.ollama, "chat", fake_chat)

    articles = [
        {"url": "https://a.com", "title": "a", "description": "d-a", "body": "body a"},
        {"url": "https://b.com", "title": "b", "description": "d-b", "body": "body b"},
        {"url": "https://c.com", "title": "c", "description": "d-c", "body": "body c"},
    ]
    skip = {"https://b.com"}

    results = extract.extract_articles(articles, skip_urls=skip)

    assert [r["url"] for r in results] == ["https://a.com", "https://c.com"]
    assert len(calls) == 2
    assert all("b" != line.split("Title: ")[1].split("\n")[0] for line in calls)


def test_extract_article_passes_format_schema_to_ollama(monkeypatch):
    """extract_article must forward _OUTPUT_SCHEMA as the format kwarg to ollama.chat."""
    captured: list[dict] = []

    def fake_chat(*args, **kwargs):
        captured.append(kwargs)
        return _ollama_response(
            '{"state": "Sinaloa", "municipality": "Culiacán", "group": "CDS", '
            '"event_type": "homicidio", "confidence": 0.9}'
        )

    monkeypatch.setattr(extract.ollama, "chat", fake_chat)
    extract.extract_article(_sample_article())

    assert captured, "ollama.chat was never called"
    assert "format" in captured[0], "format kwarg not passed to ollama.chat"
    assert captured[0]["format"] is extract._OUTPUT_SCHEMA


def test_extract_articles_defaults_skip_urls_to_empty_set(monkeypatch):
    call_count = {"n": 0}

    def fake_chat(*args, **kwargs):
        call_count["n"] += 1
        return _ollama_response(
            '{"state": "Sinaloa", "municipality": "Culiacán", "group": "CDS", '
            '"event_type": "homicidio", "confidence": 0.9}'
        )

    monkeypatch.setattr(extract.ollama, "chat", fake_chat)

    articles = [
        {"url": "https://a.com", "title": "a", "description": "d-a", "body": "x"},
        {"url": "https://b.com", "title": "b", "description": "d-b", "body": "y"},
    ]

    results = extract.extract_articles(articles)

    assert len(results) == 2
    assert call_count["n"] == 2
    assert [r["url"] for r in results] == ["https://a.com", "https://b.com"]


# ---------------------------------------------------------------------------
# Group E — VALID_STATES / VALID_EVENT_TYPES constants and _OUTPUT_SCHEMA
# ---------------------------------------------------------------------------

def test_valid_event_types_count():
    assert len(extract.VALID_EVENT_TYPES) == 14


def test_valid_states_count():
    # 31 states + "Ciudad de México" + "Internacional" + "Desconocido" = 34
    assert len(extract.VALID_STATES) == 34


def test_valid_states_includes_special_values():
    assert "Internacional" in extract._VALID_STATES_SET
    assert "Desconocido" in extract._VALID_STATES_SET


def test_valid_states_excludes_invalid_country_names():
    for bad in ("México", "EEUU", "EE.UU.", "Estados Unidos", "USA", "United States"):
        assert bad not in extract._VALID_STATES_SET, f"{bad!r} should not be a valid state"


def test_valid_event_types_set_matches_list():
    assert extract._VALID_EVENT_TYPES_SET == set(extract.VALID_EVENT_TYPES)


def test_valid_states_set_matches_list():
    assert extract._VALID_STATES_SET == set(extract.VALID_STATES)


def test_output_schema_enum_matches_valid_states():
    assert extract._OUTPUT_SCHEMA["properties"]["state"]["enum"] == extract.VALID_STATES


def test_output_schema_enum_matches_valid_event_types():
    assert extract._OUTPUT_SCHEMA["properties"]["event_type"]["enum"] == extract.VALID_EVENT_TYPES


def test_output_schema_has_all_required_fields():
    assert set(extract._OUTPUT_SCHEMA["required"]) == {
        "state", "municipality", "group", "event_type", "confidence"
    }


def test_output_schema_confidence_bounds():
    conf = extract._OUTPUT_SCHEMA["properties"]["confidence"]
    assert conf["minimum"] == 0.0
    assert conf["maximum"] == 1.0
