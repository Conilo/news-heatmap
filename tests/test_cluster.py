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


def test_jaccard_words_punctuation_stripped():
    # Without stripping, "belleza," ≠ "belleza" and the real CSV titles
    # (which include inline commas/colons and trailing " - Source Name")
    # produce Jaccard = 0.05 — below the adjacent-bucket threshold.
    # After stripping, "belleza," → "belleza" and the pair clears 0.10.
    sim = cluster._jaccard_words(
        "Hallan sin vida a ex reina de belleza mexicana: su suegra sería sospechosa - Univision",
        "Fiscalía de CDMX investiga asesinato de Carolina Flores, ex reina de belleza, en Polanco - La Jornada",
    )
    assert sim >= cluster._ADJACENT_BUCKET_JACCARD_THRESHOLD


def test_jaccard_words_belleza_cross_state_headlines_below_title_sim_threshold():
    # The two ex-reina-de-belleza headline styles (factual vs. tabloid) share
    # too few words to clear Stage 1c's 0.45 threshold — this gap remains
    # until entity-aware or SLM-based merging is added.
    sim = cluster._jaccard_words(
        "Fiscalía de CDMX investiga asesinato de Carolina Flores, ex reina de belleza, en Polanco - La Jornada",
        "Escalofriante video: matan a exreina de belleza mexicana de un balazo en la cabeza; sospechan de la suegra - Telemundo Las Vegas",
    )
    assert sim < cluster._TITLE_JACCARD_THRESHOLD


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
# _stage2_merge — trigger-based SLM disambiguation
# ---------------------------------------------------------------------------

def test_stage2_merge_no_candidates_zero_slm_calls(monkeypatch):
    """Stage 2 is a no-op (no SLM calls) when all pairs are outside the uncertain band."""
    slm_calls = []
    monkeypatch.setattr(cluster, "_slm_same_event", lambda a, b: slm_calls.append((a, b)) or (False, 0.0))

    articles = _make_articles([
        # Jaccard = 0.0 → below _SLM_MIN_JACCARD → skipped
        {"state": "Jalisco",  "group": "CJNG", "url": "u1",
         "title": "Detienen líder CJNG Guadalajara operativo nocturno armamento"},
        {"state": "Sinaloa",  "group": "CJNG", "url": "u2",
         "title": "Decomisan cocaína laboratorio clandestino Culiacán puerto marítimo"},
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-a",
        cluster._cluster_key(articles.iloc[1]): "eid-b",
    }
    cluster._stage2_merge(articles, key_to_event)
    assert slm_calls == []


def test_stage2_merge_uncertain_band_triggers_slm(monkeypatch):
    """Pairs with Jaccard in [_SLM_MIN_JACCARD, _TITLE_JACCARD_THRESHOLD) reach the SLM."""
    slm_calls = []
    monkeypatch.setattr(cluster, "_slm_same_event", lambda a, b: slm_calls.append((a, b)) or (False, 0.0))

    # These titles share "belleza" and "reina" across different states →
    # Jaccard ≈ 0.11 — inside the uncertain band.
    articles = _make_articles([
        {"state": "Ciudad de México", "group": "Desconocido", "url": "u1",
         "title": "Fiscalía investiga asesinato Carolina Flores ex reina belleza Polanco"},
        {"state": "Baja California",  "group": "Desconocido", "url": "u2",
         "title": "Matan exreina belleza mexicana balazo cabeza sospechan suegra crimen"},
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-cdmx",
        cluster._cluster_key(articles.iloc[1]): "eid-bc",
    }
    cluster._stage2_merge(articles, key_to_event)
    assert len(slm_calls) == 1


def test_stage2_merge_above_threshold_no_slm_calls(monkeypatch):
    """Pairs already above _TITLE_JACCARD_THRESHOLD are skipped (handled by Stage 1c)."""
    slm_calls = []
    monkeypatch.setattr(cluster, "_slm_same_event", lambda a, b: slm_calls.append((a, b)) or (False, 0.0))

    articles = _make_articles([
        {"state": "Jalisco", "group": "CJNG", "url": "u1",
         "title": "Detienen líder CJNG Guadalajara operativo nocturno armamento captura"},
        {"state": "Nayarit", "group": "CJNG", "url": "u2",
         "title": "Detienen líder CJNG Guadalajara operativo nocturno armamento captura"},
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-jalisco",
        cluster._cluster_key(articles.iloc[1]): "eid-nayarit",
    }
    cluster._stage2_merge(articles, key_to_event)
    assert slm_calls == []


def test_stage2_merge_slm_yes_merges_events(monkeypatch):
    """When the SLM returns same_event=True the two events are merged."""
    monkeypatch.setattr(cluster, "_slm_same_event", lambda a, b: (True, 0.9))

    articles = _make_articles([
        {"state": "Ciudad de México", "group": "Desconocido", "url": "u1",
         "title": "Fiscalía investiga asesinato Carolina Flores ex reina belleza Polanco"},
        {"state": "Baja California",  "group": "Desconocido", "url": "u2",
         "title": "Matan exreina belleza mexicana balazo cabeza sospechan suegra crimen"},
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-cdmx",
        cluster._cluster_key(articles.iloc[1]): "eid-bc",
    }
    result = cluster._stage2_merge(articles, key_to_event)
    assert len(set(result.values())) == 1


# ---------------------------------------------------------------------------
# _adjacent_bucket_merge
# ---------------------------------------------------------------------------

def test_adjacent_bucket_merge_same_state_group_merges():
    """Adjacent buckets with the same (state, group) and Jaccard >= 0.10 merge."""
    articles = _make_articles([
        {
            "state": "Ciudad de México", "group": "Desconocido",
            "published_date": "2026-04-21T07:00:00+00:00",  # bucket 4112
            "title": "Hallan sin vida a ex reina de belleza mexicana",
            "url": "u1",
        },
        {
            "state": "Ciudad de México", "group": "Desconocido",
            "published_date": "2026-04-22T07:00:00+00:00",  # bucket 4113
            "title": "Fiscalía de CDMX investiga asesinato de Carolina Flores ex reina de belleza",
            "url": "u2",
        },
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-4112",
        cluster._cluster_key(articles.iloc[1]): "eid-4113",
    }
    result = cluster._adjacent_bucket_merge(articles, key_to_event)
    assert len(set(result.values())) == 1


def test_adjacent_bucket_merge_low_overlap_stays_separate():
    """Adjacent buckets with unrelated titles (Jaccard < 0.10) stay separate."""
    articles = _make_articles([
        {
            "state": "Sinaloa", "group": "CDS",
            "published_date": "2026-04-21T07:00:00+00:00",  # bucket 4112
            "title": "Decomisan tonelada cocaína laboratorio clandestino Culiacán",
            "url": "u1",
        },
        {
            "state": "Sinaloa", "group": "CDS",
            "published_date": "2026-04-22T07:00:00+00:00",  # bucket 4113
            "title": "Arrestan líder financiero Cártel Sinaloa lavado dinero",
            "url": "u2",
        },
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-4112",
        cluster._cluster_key(articles.iloc[1]): "eid-4113",
    }
    result = cluster._adjacent_bucket_merge(articles, key_to_event)
    assert len(set(result.values())) == 2


def test_adjacent_bucket_merge_non_adjacent_buckets_not_merged():
    """Buckets more than one step apart are never compared, even with identical titles."""
    articles = _make_articles([
        {
            "state": "Jalisco", "group": "CJNG",
            "published_date": "2026-04-17T07:00:00+00:00",  # bucket 4112
            "title": "Detienen líder CJNG Guadalajara operativo nocturno",
            "url": "u1",
        },
        {
            "state": "Jalisco", "group": "CJNG",
            "published_date": "2026-04-27T07:00:00+00:00",  # bucket 4114 (two steps away)
            "title": "Detienen líder CJNG Guadalajara operativo nocturno",
            "url": "u2",
        },
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-4112",
        cluster._cluster_key(articles.iloc[1]): "eid-4114",
    }
    result = cluster._adjacent_bucket_merge(articles, key_to_event)
    assert len(set(result.values())) == 2


def test_adjacent_bucket_merge_different_states_not_merged():
    """Different states are never merged by Stage 1d, even with adjacent buckets."""
    articles = _make_articles([
        {
            "state": "Jalisco", "group": "CJNG",
            "published_date": "2026-04-21T07:00:00+00:00",
            "title": "Detienen líder CJNG Guadalajara operativo belleza reina",
            "url": "u1",
        },
        {
            "state": "Nayarit", "group": "CJNG",
            "published_date": "2026-04-22T07:00:00+00:00",
            "title": "Detienen líder CJNG Guadalajara operativo belleza reina",
            "url": "u2",
        },
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-jalisco",
        cluster._cluster_key(articles.iloc[1]): "eid-nayarit",
    }
    result = cluster._adjacent_bucket_merge(articles, key_to_event)
    assert len(set(result.values())) == 2


def test_adjacent_bucket_merge_desconocido_state_skipped():
    """Desconocido state is in _NON_GEO — adjacent-bucket merge never runs on it."""
    articles = _make_articles([
        {
            "state": "Desconocido", "group": "CJNG",
            "published_date": "2026-04-21T07:00:00+00:00",
            "title": "Detienen líder CJNG operativo belleza reina mexicana",
            "url": "u1",
        },
        {
            "state": "Desconocido", "group": "CJNG",
            "published_date": "2026-04-22T07:00:00+00:00",
            "title": "Detienen líder CJNG operativo belleza reina mexicana",
            "url": "u2",
        },
    ])
    key_to_event = {
        cluster._cluster_key(articles.iloc[0]): "eid-a",
        cluster._cluster_key(articles.iloc[1]): "eid-b",
    }
    result = cluster._adjacent_bucket_merge(articles, key_to_event)
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
    _, events = cluster.cluster_articles(articles, use_slm=False)
    assert len(events) == 1
    assert events.iloc[0]["article_count"] == 3


def test_cluster_articles_different_groups_separate_events():
    articles = _make_articles([
        {"state": "Jalisco", "group": "CJNG", "url": "u1"},
        {"state": "Jalisco", "group": "CDS",  "url": "u2"},
    ])
    _, events = cluster.cluster_articles(articles, use_slm=False)
    assert len(events) == 2


def test_cluster_articles_bucket_boundary_merges_same_state_group():
    """
    Stage 1d regression: articles published on either side of a 5-day bucket
    boundary (Apr 21 → bucket 4112, Apr 22 → bucket 4113) with the same state
    and group should merge when their title Jaccard >= 0.10.

    Uses the actual punctuated+sourced title format from the CSV to catch the
    "belleza," ≠ "belleza" punctuation bug that caused Jaccard to drop to 0.05.
    """
    articles = _make_articles([
        {
            "state": "Ciudad de México", "group": "Desconocido",
            "published_date": "2026-04-21T07:00:00+00:00",
            "title": "Hallan sin vida a ex reina de belleza mexicana: su suegra sería sospechosa de su muerte - Univision",
            "url": "u1",
        },
        {
            "state": "Ciudad de México", "group": "Desconocido",
            "published_date": "2026-04-22T07:00:00+00:00",
            "title": "Fiscalía de CDMX investiga asesinato de Carolina Flores, ex reina de belleza, en Polanco - La Jornada",
            "url": "u2",
        },
    ])
    _, events = cluster.cluster_articles(articles, use_slm=False)
    assert len(events) == 1
    assert events.iloc[0]["article_count"] == 2


def test_cluster_articles_same_case_different_headline_styles_stay_separate_without_slm():
    """
    Without SLM (use_slm=False), cross-state pairs with Jaccard in the uncertain
    band [0.02, 0.45) are not merged — Stage 2 is suppressed entirely.

    This is the ex-reina-de-belleza scenario: factual vs. tabloid headline styles
    share too few words for Stage 1c (Jaccard ≈ 0.056 < 0.45) and would normally
    reach Stage 2 for SLM disambiguation.  With use_slm=False, two events remain.
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
    _, events = cluster.cluster_articles(articles, use_slm=False)
    assert len(events) == 2


@pytest.mark.slm_live
def test_cluster_articles_slm_merges_uncertain_band_pair(monkeypatch):
    """
    With use_slm=True, a cross-state pair in the uncertain Jaccard band reaches
    Stage 2 and is merged when the SLM confirms it is the same event.

    Marked slm_live because it exercises the Stage 2 code path end-to-end
    (monkeypatching the Ollama call to keep it offline-safe).
    Run with: pytest --slm-live
    """
    monkeypatch.setattr(cluster, "_slm_same_event", lambda a, b: (True, 0.92))

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
    _, events = cluster.cluster_articles(articles, use_slm=True)
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Cross-group Stage 2 merge (Issue: Azcapotzalco homicide split across groups)
# ---------------------------------------------------------------------------

@pytest.mark.slm_live
def test_stage2_merges_cross_group_same_state_bucket_pair(monkeypatch):
    """
    Two articles about the same homicide in the same state+bucket but attributed
    to different cartel groups should be merged by Stage 2 when the SLM confirms
    they describe the same event.
    """
    monkeypatch.setattr(cluster, "_slm_same_event", lambda a, b: (True, 0.91))

    articles = _make_articles([
        {
            "state": "Ciudad de México", "group": "Los Julios",
            "published_date": "2026-04-30T03:00:00+00:00",
            "title": "Multihomicidio en Azcapotzalco detenido sostenía relación con una de las víctimas",
            "url": "u1",
        },
        {
            "state": "Ciudad de México", "group": "Unión Tepito",
            "published_date": "2026-05-01T04:00:00+00:00",
            "title": "La cacería de 24 horas para dar con los asesinos de Azcapotzalco",
            "url": "u2",
        },
    ])
    _, events = cluster.cluster_articles(articles, use_slm=True)
    assert len(events) == 1


# ---------------------------------------------------------------------------
# GROUP_ALIASES canonicalization for Unión Tepito variants
# ---------------------------------------------------------------------------

def test_normalize_group_union_tepito_variants():
    """'La Unión Tepito' and 'Unión Tepito' must resolve to the same canonical name."""
    from config import normalize_group
    assert normalize_group("Unión Tepito") == normalize_group("La Unión Tepito")
    assert normalize_group("union tepito") == normalize_group("Unión Tepito")
