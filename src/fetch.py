"""Fetch and filter Mexico cartel-related news from Google News."""

from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnews import GNews

import config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MEXICO_TERMS = [
    "mexico",
    "méxico",
    "mexicano",
    "mexicana",
]

MEXICO_STATES = [
    "aguascalientes", "baja california", "baja california sur", "campeche",
    "chiapas", "chihuahua", "coahuila", "colima", "durango", "guanajuato",
    "guerrero", "hidalgo", "jalisco", "michoacán", "michoacan", "morelos",
    "nayarit", "nuevo león", "nuevo leon", "oaxaca", "puebla", "querétaro",
    "queretaro", "quintana roo", "san luis potosí", "san luis potosi",
    "sinaloa", "sonora", "tabasco", "tamaulipas", "tlaxcala", "veracruz",
    "yucatán", "yucatan", "zacatecas", "ciudad de méxico", "cdmx",
    "estado de mexico", "estado de méxico",
]

_ALL_MEXICO_TERMS = set(MEXICO_TERMS + MEXICO_STATES)


def _is_mexico_related(article: dict[str, Any]) -> bool:
    """Return True if the article appears to be about Mexico."""
    text = " ".join([
        (article.get("title") or ""),
        (article.get("description") or ""),
    ]).lower()
    return any(term in text for term in _ALL_MEXICO_TERMS)


def _is_crime_related(article: dict[str, Any]) -> bool:
    """Return True if the article contains at least one crime keyword."""
    text = " ".join([
        (article.get("title") or ""),
        (article.get("description") or ""),
    ]).lower()
    return any(kw.lower() in text for kw in config.KEYWORDS)


def _normalize_article(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a gnews article dict into a consistent shape."""
    pub = raw.get("published date") or raw.get("published_date") or ""
    return {
        "url": raw.get("url", ""),
        "title": raw.get("title", ""),
        "description": raw.get("description", ""),
        "published_date": pub,
        "source": (raw.get("publisher") or {}).get("title", ""),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_articles(
    lookback_days: int = config.LOOKBACK_DAYS,
    max_results: int = config.MAX_ARTICLES,
) -> list[dict[str, Any]]:
    """
    Fetch Google News articles related to Mexico crime/cartels.

    Runs two passes:
      1. English query  ("mexico cartel narco")
      2. Spanish query  ("cartel mexico narco")

    Each article is deduplicated by URL, then filtered for Mexico content
    and cartel/crime keywords.

    Returns a list of normalized article dicts.
    """
    client = GNews(
        language="es",
        country="MX",
        period=f"{lookback_days}d",
        max_results=max_results,
    )

    queries = [
        "cartel narco mexico",
        "crimen organizado mexico",
        "narco drogas",
    ]

    seen_urls: set[str] = set()
    articles: list[dict[str, Any]] = []

    for query in queries:
        try:
            results = client.get_news(query)
        except Exception as exc:
            print(f"[fetch] Warning: query '{query}' failed — {exc}")
            continue

        for raw in results or []:
            normalized = _normalize_article(raw)
            url = normalized["url"]
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            if _is_mexico_related(normalized) and _is_crime_related(normalized):
                articles.append(normalized)

            if len(articles) >= max_results:
                break

        if len(articles) >= max_results:
            break

    print(f"[fetch] {len(articles)} articles after filtering.")
    return articles


if __name__ == "__main__":
    arts = fetch_articles()
    for a in arts[:5]:
        print(a["title"])
