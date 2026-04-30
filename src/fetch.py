"""Fetch and filter Mexico cartel-related news from Google News."""

from __future__ import annotations

import sys
import os
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from googlenewsdecoder import gnewsdecoder
from gnews import GNews

import config

# Hosts where Google News often links but HTML article extraction is blocked or meaningless.
_NON_ARTICLE_HOST_SUFFIXES: tuple[str, ...] = (
    "facebook.com",
    "fb.com",
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "x.com",
    "tiktok.com",
)

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


def _non_article_host(url: str) -> bool:
    """True for social / video hosts we cannot treat as downloadable news HTML."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return True
    if not host:
        return True
    for suf in _NON_ARTICLE_HOST_SUFFIXES:
        if host == suf or host.endswith("." + suf):
            return True
    return False


def _publisher_url(url: str) -> str:
    """
    Turn Google News wrapper links into the publisher article URL.

    gnews' process_url() relies on ``requests.head`` + ``Location``, which
    often returns 400 for current ``/rss/articles/`` links, so the wrapper
    URL is kept and newspaper3k yields empty text.
    """
    if not url or "news.google.com" not in url:
        return url
    try:
        interval = config.GOOGLE_NEWS_DECODE_INTERVAL_SEC or None
        out = gnewsdecoder(url, interval=interval)
        if out.get("status") and out.get("decoded_url"):
            return str(out["decoded_url"])
        msg = out.get("message", out)
        print(f"[fetch] Warning: Google News URL not decoded — {msg}")
    except Exception as exc:
        print(f"[fetch] Warning: Google News decode failed — {exc}")
    return url


def _newspaper_body_from_url(url: str) -> str:
    """Download and parse article HTML via newspaper3k with a browser-like User-Agent."""
    from newspaper import Article, Config

    cfg = Config()
    cfg.browser_user_agent = config.ARTICLE_FETCH_USER_AGENT
    cfg.request_timeout = int(config.ARTICLE_FETCH_TIMEOUT_SEC)
    article = Article(url, language="es", config=cfg)
    article.download()
    article.parse()
    return (article.text or "").strip()


def _fetch_body(url: str) -> str:
    """
    Resolve Google News redirects, then download publisher HTML with newspaper3k.

    Returns empty string on failure or empty parse. Text is truncated to
    config.ARTICLE_BODY_MAX_CHARS_SLM for storage and downstream SLM use.
    """
    if not (url or "").strip():
        return ""
    target = _publisher_url(url.strip())
    if _non_article_host(target):
        print(f"[fetch] Skipping body fetch (non-article host): {target[:100]}")
        return ""

    try:
        text = _newspaper_body_from_url(target)
        if not text:
            print(f"[fetch] Warning: empty article text after parse for {target[:100]}")
            return ""
        limit = config.ARTICLE_BODY_MAX_CHARS_SLM
        if len(text) > limit:
            text = text[:limit]
        return text
    except Exception as exc:
        # 403 Forbidden = anti-bot / geo rules on many outlets (e.g. some EU publishers),
        # not necessarily paywalls. Paywalls often return 200 with a stub article.
        msg = str(exc)
        if "403" in msg or "Forbidden" in msg:
            print(f"[fetch] Warning: publisher blocked fetch (403) for {target[:100]} — {exc}")
        else:
            print(f"[fetch] Warning: full article failed for {target[:100]} — {exc}")
        return ""


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
    plus the `mexico` token in the query for geographic filtering.

    Returns **only** articles where full article text was retrieved (non-empty
    ``body``). RSS items without a downloadable body are skipped. Up to
    ``max_results`` such articles are returned, scanning up to
    ``config.GNEWS_RSS_MAX_ITEMS`` feed entries.
    """
    client = GNews(
        language="es",
        country="MX",
        period=f"{lookback_days}d",
        max_results=config.GNEWS_RSS_MAX_ITEMS,
    )

    try:
        results = client.get_news(QUERY)
    except Exception as exc:
        print(f"[fetch] Warning: query failed — {exc}")
        return []

    seen_urls: set[str] = set()
    articles: list[dict[str, Any]] = []
    candidates_tried = 0

    for raw in results or []:
        if len(articles) >= max_results:
            break
        normalized = _normalize_article(raw)
        url = normalized["url"]
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        candidates_tried += 1

        body = _fetch_body(url).strip()
        if not body:
            continue

        normalized["body"] = body
        articles.append(normalized)

    print(
        f"[fetch] {len(articles)} articles with full text "
        f"({candidates_tried} RSS URLs tried, cap {config.GNEWS_RSS_MAX_ITEMS})."
    )
    return articles


if __name__ == "__main__":
    arts = fetch_articles()
    for a in arts[:5]:
        print(a["title"])
