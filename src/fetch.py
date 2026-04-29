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

def _build_query(terms: list[str], geo: str) -> str:
    """Build a Google News OR-query, quoting any multi-word terms."""
    quoted = [f'"{t}"' if " " in t else t for t in terms]
    return f"({' OR '.join(quoted)}) {geo}"


QUERY = _build_query(config.FETCH_QUERY_TERMS, config.FETCH_QUERY_GEO)


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

    Issues a single OR-combined Google News query (see `QUERY`, built from
    `config.FETCH_QUERY_TERMS`) and trusts Google's keyword matching against
    title/description plus the MX news edition (`country=MX, language=es`)
    plus the `mexico` token in the query for geographic filtering. No
    post-filter is applied — articles are only deduplicated by URL.

    Returns a list of normalized article dicts.
    """
    client = GNews(
        language="es",
        country="MX",
        period=f"{lookback_days}d",
        max_results=max_results,
    )

    try:
        results = client.get_news(QUERY)
    except Exception as exc:
        print(f"[fetch] Warning: query failed — {exc}")
        return []

    seen_urls: set[str] = set()
    articles: list[dict[str, Any]] = []

    for raw in results or []:
        normalized = _normalize_article(raw)
        url = normalized["url"]
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        articles.append(normalized)

        if len(articles) >= max_results:
            break

    print(f"[fetch] {len(articles)} articles fetched.")
    return articles


if __name__ == "__main__":
    arts = fetch_articles()
    for a in arts[:5]:
        print(a["title"])
