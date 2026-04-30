"""Tests for src/extract.py — SLM-based structured-data extraction."""

from __future__ import annotations

from datetime import datetime

import pytest

from src import extract


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
        "crime_type": "otro",
        "confidence": 0.0,
    }


def test_validate_fields_treats_none_empty_and_null_string_as_missing():
    result = extract._validate_fields({
        "state": None,
        "municipality": "",
        "group": "null",
        "crime_type": "homicidio",
        "confidence": 0.5,
    })
    assert result["state"] == "Desconocido"
    assert result["municipality"] == "Desconocido"
    assert result["group"] == "Desconocido"
    assert result["crime_type"] == "homicidio"
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


def test_extract_article_merges_slm_fields_and_adds_processed_at(monkeypatch):
    monkeypatch.setattr(
        extract.ollama,
        "chat",
        _make_fake_chat(
            '{"state": "Sinaloa", "municipality": "Culiacán", "group": "CDS", '
            '"crime_type": "homicidio", "confidence": 0.95}'
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
    assert result["crime_type"] == "homicidio"
    assert result["confidence"] == 0.95

    # processed_at is a valid ISO-format timestamp
    assert "processed_at" in result
    datetime.fromisoformat(result["processed_at"])


def test_extract_article_handles_markdown_fenced_response(monkeypatch):
    fenced = (
        '```json\n'
        '{"state": "Jalisco", "municipality": "Guadalajara", "group": "CJNG", '
        '"crime_type": "narcotráfico", "confidence": 0.8}\n'
        '```'
    )
    monkeypatch.setattr(extract.ollama, "chat", _make_fake_chat(fenced))

    result = extract.extract_article(_sample_article())

    assert result["state"] == "Jalisco"
    assert result["municipality"] == "Guadalajara"
    assert result["group"] == "CJNG"
    assert result["crime_type"] == "narcotráfico"
    assert result["confidence"] == 0.8


def test_extract_article_falls_back_on_ollama_exception(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("ollama is down")

    monkeypatch.setattr(extract.ollama, "chat", boom)

    article = _sample_article()
    result = extract.extract_article(article)
    out = capsys.readouterr().out

    # Original fields preserved
    assert result["url"] == article["url"]
    assert result["title"] == article["title"]

    # Fallback values applied
    assert result["state"] == "Desconocido"
    assert result["municipality"] == "Desconocido"
    assert result["group"] == "Desconocido"
    assert result["crime_type"] == "otro"
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

    article = {**_sample_article(), "body": ""}
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

    article = {**_sample_article(), "body": "  \n\t  "}
    result = extract.extract_article(article)

    assert calls == []
    assert result["group"] == "Desconocido"


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
            '"crime_type": "homicidio", "confidence": 0.9}'
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


def test_extract_articles_defaults_skip_urls_to_empty_set(monkeypatch):
    call_count = {"n": 0}

    def fake_chat(*args, **kwargs):
        call_count["n"] += 1
        return _ollama_response(
            '{"state": "Sinaloa", "municipality": "Culiacán", "group": "CDS", '
            '"crime_type": "homicidio", "confidence": 0.9}'
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
