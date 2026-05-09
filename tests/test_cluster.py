"""Tests for src/cluster.py — grouping keys, confidence, merging, and integration."""

from __future__ import annotations

import pandas as pd
import pytest

from src import cluster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_articles(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal articles DataFrame, filling in required columns with defaults."""
    defaults = {
        "url": "",
        "title": "Test headline sobre crimen organizado",
        "description": "",
        "body": "Cuerpo del artículo sobre el incidente.",
        "published_date": "2026-04-22T07:00:00+00:00",
        "source": "TestSource",
        "state": "Sinaloa",
        "municipality": "Desconocido",
        "group": "CJNG",
        "event_type": "homicidio",
        "confidence": 0.9,
        "processed_at": "",
        "event_id": "",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ---------------------------------------------------------------------------
# _cluster_key
# ---------------------------------------------------------------------------

def test_cluster_key_deduplicates_unextracted_articles_per_url():
    base = {
        "state": "Desconocido",
        "group": "Desconocido",
        "published_date": "Mon, 28 Apr 2026 12:00:00 GMT",
        "confidence": "0.0",
        "body": "",
        "title": "t",
    }
    r1 = pd.Series({**base, "url": "https://a.com/1"})
    r2 = pd.Series({**base, "url": "https://b.com/2"})
    assert cluster._cluster_key(r1) != cluster._cluster_key(r2)


def test_state_group_bucket_supports_four_tuple():
    assert cluster._state_group_bucket(("Desconocido", "CDS", 5, "https://a")) == (
        "Desconocido",
        "CDS",
        5,
    )


def test_cluster_key_extracted_rows_remain_three_tuple():
    row = pd.Series({
        "state": "Sinaloa",
        "group": "CJNG",
        "published_date": "Mon, 28 Apr 2026 12:00:00 GMT",
        "confidence": "0.5",
        "body": "some text",
        "url": "https://example.com",
    })
    key = cluster._cluster_key(row)
    assert len(key) == 3


# ---------------------------------------------------------------------------
# _compute_confidence
# ---------------------------------------------------------------------------

def test_compute_confidence_single_article():
    df = _make_articles([{"confidence": 0.8, "source": "Reforma"}])
    assert cluster._compute_confidence(df) == pytest.approx(0.8)


def test_compute_confidence_multi_source_boost():
    # mean=0.8, 3 unique sources → +0.10 bonus
    df = _make_articles([
        {"confidence": 0.8, "source": "Reforma"},
        {"confidence": 0.8, "source": "El Universal"},
        {"confidence": 0.8, "source": "Milenio"},
    ])
    assert cluster._compute_confidence(df) == pytest.approx(0.9)


def test_compute_confidence_duplicate_sources_do_not_boost():
    # Two rows from the same source → unique_sources=1 → no bonus
    df = _make_articles([
        {"confidence": 0.7, "source": "Reforma"},
        {"confidence": 0.7, "source": "Reforma"},
    ])
    assert cluster._compute_confidence(df) == pytest.approx(0.7)


def test_compute_confidence_capped_at_one():
    # 0.95 + 0.05*5 = 1.20 → capped at 1.0
    df = _make_articles([
        {"confidence": 0.95, "source": src}
        for src in ["A", "B", "C", "D", "E", "F"]
    ])
    assert cluster._compute_confidence(df) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _jaccard_words
# ---------------------------------------------------------------------------

def test_jaccard_words_identical_strings():
    assert cluster._jaccard_words(
        "matan exreina belleza mexicana",
        "matan exreina belleza mexicana",
    ) == pytest.approx(1.0)


def test_jaccard_words_no_overlap():
    assert cluster._jaccard_words(
        "capturan lider sinaloa operativo",
        "bloqueo jalisco policía municipal",
    ) == pytest.approx(0.0)


def test_jaccard_words_stopwords_excluded():
    # "de", "del", "la", "en" are stopwords; only content words count.
    # "homicidio familia" vs "homicidio menor" → intersection={homicidio}, union=3 → 1/3
    sim = cluster._jaccard_words(
        "homicidio de la familia",
        "homicidio del menor",
    )
    assert sim == pytest.approx(1 / 3)


def test_jaccard_words_belleza_headlines_below_threshold():
    # The two ex-reina-de-belleza headline styles share only "belleza",
    # giving Jaccard ≈ 0.056 — well below the 0.45 merge threshold.
    sim = cluster._jaccard_words(
        "Fiscalía de CDMX investiga asesinato de Carolina Flores ex reina de belleza en Polanco",
        "Escalofriante video matan exreina de belleza mexicana de balazo en cabeza sospechan de suegra",
    )
    assert sim < 0.10


# ---------------------------------------------------------------------------
# _absorb_desconocido
# ---------------------------------------------------------------------------

def test_absorb_desconocido_merges_into_known_state():
    key_to_event = {
        ("Nayarit", "CJNG", 100): "eid-nayarit",
        ("Desconocido", "CJNG", 100): "eid-desc",
    }
    result = cluster._absorb_desconocido(key_to_event)
    assert result[("Nayarit", "CJNG", 100)] == "eid-nayarit"
    assert result[("Desconocido", "CJNG", 100)] == "eid-nayarit"


def test_absorb_desconocido_no_specific_state_unchanged():
    # Only Desconocido clusters for this (group, bucket) — nothing to absorb into.
    key_to_event = {
        ("Desconocido", "CJNG", 100): "eid-a",
        ("Desconocido", "CDS", 100): "eid-b",
    }
    result = cluster._absorb_desconocido(key_to_event)
    assert result == key_to_event


def test_absorb_desconocido_two_real_states_stay_separate():
    # Jalisco and Nayarit may be genuinely different events, so they should
    # not be merged into each other.  Desconocido absorbs into whichever is larger.
    key_to_event = {
        ("Jalisco", "CJNG", 100): "eid-jalisco",
        ("Nayarit", "CJNG", 100): "eid-nayarit",
        ("Desconocido", "CJNG", 100): "eid-desc",
    }
    result = cluster._absorb_desconocido(key_to_event)
    assert result[("Jalisco", "CJNG", 100)] != result[("Nayarit", "CJNG", 100)]
    assert result[("Desconocido", "CJNG", 100)] in {"eid-jalisco", "eid-nayarit"}


def test_absorb_desconocido_internacional_not_treated_as_specific():
    # "Internacional" is in _NON_GEO, same as "Desconocido".
    # Desconocido should NOT absorb into Internacional.
    key_to_event = {
        ("Internacional", "CJNG", 100): "eid-intl",
        ("Desconocido", "CJNG", 100): "eid-desc",
    }
    result = cluster._absorb_desconocido(key_to_event)
    assert result[("Desconocido", "CJNG", 100)] == "eid-desc"
    assert result[("Internacional", "CJNG", 100)] == "eid-intl"


# ---------------------------------------------------------------------------
# _title_similarity_merge
# ---------------------------------------------------------------------------

def test_title_similarity_merge_high_overlap_merges():
    """Two cross-state clusters with near-identical titles merge into one event."""
    articles = _make_articles([
        {
            "state": "Jalisco", "group": "CJNG",
            "published_date": "2026-04-22T07:00:00+00:00",
            "title": "Detienen a líder del CJNG en Guadalajara",
            "confidence": 0.9, "url": "u1",
        },
        {
            "state": "Nayarit", "group": "CJNG",
            "published_date": "2026-04-22T07:00:00+00:00",
            "title": "Detienen al líder del CJNG en Guadalajara operativo",
            "confidence": 0.85, "url": "u2",
        },
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-jalisco",
        cluster._cluster_key(articles.iloc[1]): "eid-nayarit",
    }
    result = cluster._title_similarity_merge(articles, key_to_event)
    assert len(set(result.values())) == 1


def test_title_similarity_merge_low_overlap_stays_separate():
    """Two clusters with low title Jaccard (< 0.45) are not merged."""
    articles = _make_articles([
        {
            "state": "Ciudad de México", "group": "Desconocido",
            "published_date": "2026-04-22T07:00:00+00:00",
            "title": "Fiscalía de CDMX investiga asesinato de Carolina Flores ex reina de belleza en Polanco",
            "confidence": 0.96, "url": "u1",
        },
        {
            "state": "Baja California", "group": "Desconocido",
            "published_date": "2026-04-24T07:00:00+00:00",
            "title": "Escalofriante video matan exreina de belleza mexicana de balazo en cabeza sospechan de suegra",
            "confidence": 0.92, "url": "u2",
        },
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-a",
        cluster._cluster_key(articles.iloc[1]): "eid-b",
    }
    result = cluster._title_similarity_merge(articles, key_to_event)
    assert len(set(result.values())) == 2


def test_title_similarity_merge_different_groups_not_merged():
    """Clusters with different cartel groups are never compared, even with identical titles."""
    articles = _make_articles([
        {
            "state": "Jalisco", "group": "CJNG",
            "published_date": "2026-04-22T07:00:00+00:00",
            "title": "Detienen líderes cartel operativo nocturno Guadalajara decomisan armamento",
            "confidence": 0.9, "url": "u1",
        },
        {
            "state": "Sinaloa", "group": "CDS",
            "published_date": "2026-04-22T07:00:00+00:00",
            "title": "Detienen líderes cartel operativo nocturno Guadalajara decomisan armamento",
            "confidence": 0.9, "url": "u2",
        },
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-cjng",
        cluster._cluster_key(articles.iloc[1]): "eid-cds",
    }
    result = cluster._title_similarity_merge(articles, key_to_event)
    assert len(set(result.values())) == 2


# ---------------------------------------------------------------------------
# cluster_articles — integration
# ---------------------------------------------------------------------------

def test_cluster_articles_basic_grouping():
    """Articles with the same (state, group, date bucket) collapse into one event."""
    articles = _make_articles([
        {"state": "Sinaloa", "group": "CDS", "source": "A", "url": "u1"},
        {"state": "Sinaloa", "group": "CDS", "source": "B", "url": "u2"},
        {"state": "Sinaloa", "group": "CDS", "source": "C", "url": "u3"},
    ])
    _, events = cluster.cluster_articles(articles)
    assert len(events) == 1
    assert events.iloc[0]["article_count"] == 3


def test_cluster_articles_different_groups_separate_events():
    articles = _make_articles([
        {"state": "Jalisco", "group": "CJNG", "url": "u1"},
        {"state": "Jalisco", "group": "CDS",  "url": "u2"},
    ])
    _, events = cluster.cluster_articles(articles)
    assert len(events) == 2


def test_cluster_articles_bucket_boundary_splits_adjacent_days():
    """
    Known limitation: articles published one day apart can land in different
    5-day buckets when a boundary falls between them.

    Apr 21 → bucket 4112, Apr 22 → bucket 4113, so these two articles about
    the same homicide produce two events instead of one.

    When cross-bucket merging is implemented, change the assertion to == 1.
    """
    articles = _make_articles([
        {
            "state": "Ciudad de México", "group": "Desconocido",
            "published_date": "2026-04-21T07:00:00+00:00",
            "title": "Hallan sin vida a ex reina de belleza mexicana",
            "url": "u1",
        },
        {
            "state": "Ciudad de México", "group": "Desconocido",
            "published_date": "2026-04-22T07:00:00+00:00",
            "title": "Fiscalía de CDMX investiga asesinato de Carolina Flores ex reina de belleza",
            "url": "u2",
        },
    ])
    _, events = cluster.cluster_articles(articles)
    # Currently two events due to bucket boundary — update to 1 when fixed.
    assert len(events) == 2


def test_cluster_articles_same_case_different_headline_styles_stay_separate():
    """
    Known gap: sensationalist vs. factual headlines about the same incident
    share too few words to clear the 0.45 Jaccard threshold (actual: ~0.056).

    This is the ex-reina-de-belleza scenario: one article covers the murder
    (factual/legal framing), the other is a tabloid reconstruction.  No merge
    happens even though they're in the same bucket with the same group.

    When entity-aware or SLM-based merging is added, change assertion to == 1.
    """
    articles = _make_articles([
        {
            "state": "Ciudad de México", "group": "Desconocido",
            "published_date": "2026-04-22T07:00:00+00:00",
            "title": "Fiscalía de CDMX investiga asesinato de Carolina Flores ex reina de belleza en Polanco",
            "url": "u1",
        },
        {
            "state": "Baja California", "group": "Desconocido",
            "published_date": "2026-04-24T07:00:00+00:00",
            "title": "Escalofriante video matan exreina de belleza mexicana de balazo en cabeza sospechan de suegra",
            "url": "u2",
        },
    ])
    _, events = cluster.cluster_articles(articles)
    # Jaccard ≈ 0.056 → no title-similarity merge → two events. Update to 1 when fixed.
    assert len(events) == 2
