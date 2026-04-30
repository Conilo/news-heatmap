"""Tests for src/cluster.py — grouping keys."""

from __future__ import annotations

import pandas as pd

from src import cluster


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
