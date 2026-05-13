# Fetch design

`src/fetch.py` — queries Google News for Mexico crime/cartel articles, resolves
publisher URLs, downloads full article text, and returns a list of article dicts.

---

## Inputs and outputs

| | |
|---|---|
| Config inputs | `config.FETCH_QUERY_TERMS`, `config.FETCH_QUERY_GEO`, `config.LOOKBACK_DAYS`, `config.MAX_ARTICLES`, `config.GNEWS_RSS_MAX_ITEMS` |
| Output | `list[dict]` — each dict has `url`, `title`, `description`, `published_date`, `source`, `body` |

Entry point: `fetch_articles(lookback_days, max_results)`.  Called by the
pipeline runner; the result is merged into `data/articles.csv` by the caller.

---

## Pipeline

### Step 1 — Query construction

`_build_query()` produces a single Google News OR-query:

```
("term1" OR "term2" OR ...) mexico
```

Multi-word terms are quoted; single tokens are left bare.  The `mexico` token
in `config.FETCH_QUERY_GEO` provides geographic scoping on top of the
`country=MX, language=es` GNews client settings.

The resulting `QUERY` string is a module-level constant — it does not change
between calls, so the same query is reused across the session.

### Step 2 — RSS feed fetch

`GNews(language="es", country="MX", period="{lookback_days}d")` issues a
single RSS request to `news.google.com`.  Up to `config.GNEWS_RSS_MAX_ITEMS`
feed entries are returned (typically 100).  Each entry contains title,
description, published date, publisher, and a Google News wrapper URL.

### Step 3 — URL decoding (`_publisher_url`)

Google News wraps article links in `news.google.com/rss/articles/…` URLs.
`gnewsdecoder` resolves these to the actual publisher URL so that
`newspaper3k` can fetch article HTML.  On failure the wrapper URL is passed
through unchanged (newspaper3k will yield empty text for most wrapper URLs).

### Step 4 — Non-article host filtering (`_non_article_host`)

Social and video hosts (Facebook, Instagram, YouTube, Twitter/X, TikTok) are
skipped before any download attempt.  These hosts are listed in
`_NON_ARTICLE_HOST_SUFFIXES` and matched by suffix so subdomains are also
caught.

### Step 5 — Article body download (`_newspaper_body_from_url`)

`newspaper3k` downloads and parses publisher HTML using a browser-like
User-Agent (`config.ARTICLE_FETCH_USER_AGENT`) and a configurable timeout
(`config.ARTICLE_FETCH_TIMEOUT_SEC`).  Language is set to `"es"`.

Common failure modes:

| Error | Cause | Handling |
|---|---|---|
| 403 Forbidden | Anti-bot or geo-blocking rules | Logged and skipped |
| Empty text | Paywall stub (200 OK but no article text), JavaScript-rendered pages | Skipped |
| Decode exception | Network timeout, malformed HTML | Logged and skipped |

Articles with an empty body after parsing are discarded — the SLM extraction
step (`extract.py`) requires body text and falls back to no-op defaults
otherwise.

### Step 6 — Deduplication and cap

Duplicate URLs (same article from multiple RSS entries) are filtered by a
`seen_urls` set.  Collection stops once `max_results` articles with non-empty
bodies have been gathered, even if the RSS feed contained more entries.

---

## Configuration reference

| Key | Default | Description |
|---|---|---|
| `FETCH_QUERY_TERMS` | (list in `user_config.py`) | OR-combined search terms |
| `FETCH_QUERY_GEO` | `"mexico"` | Geographic token appended to query |
| `LOOKBACK_DAYS` | `14` | RSS period window |
| `MAX_ARTICLES` | `30` | Maximum articles with full text returned |
| `GNEWS_RSS_MAX_ITEMS` | (advanced config) | RSS entries fetched before applying the body cap |
| `GOOGLE_NEWS_DECODE_INTERVAL_SEC` | (advanced config) | Delay between gnewsdecoder calls to avoid rate limiting |
| `ARTICLE_FETCH_USER_AGENT` | (advanced config) | Browser UA string for newspaper3k |
| `ARTICLE_FETCH_TIMEOUT_SEC` | (advanced config) | Per-article download timeout |
| `ARTICLE_BODY_MAX_CHARS_SLM` | `8000` | Body text truncation limit before storage |

---

## Known limitations

| Limitation | Notes |
|---|---|
| JavaScript-rendered articles | `newspaper3k` parses static HTML only; SPA or JS-gated articles yield empty bodies and are discarded |
| Paywalled content | Returns a stub article (200 OK, short text); body passes the non-empty check but contains little signal for the SLM |
| Google News decode reliability | `gnewsdecoder` uses undocumented Google endpoints; wrapper URLs may stop resolving if Google changes its redirect scheme |
| RSS cap vs. body cap | `GNEWS_RSS_MAX_ITEMS` feeds are scanned to find `MAX_ARTICLES` with non-empty bodies; if most articles are blocked or paywalled, the effective yield can be much lower than `MAX_ARTICLES` |
| Single OR-query | All terms are combined in one query; relevance ranking is delegated entirely to Google News. A multi-query approach (one query per term, deduplication across results) could increase coverage |
