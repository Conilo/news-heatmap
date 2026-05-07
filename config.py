"""Stable import surface for the pipeline: ``import config`` everywhere.

Tune common behavior in ``user_config.py``. For RSS limits, HTTP, paths,
clustering buckets, CSV column lists, and group/state maps, see
``advanced_config.py``.
"""

from __future__ import annotations

from advanced_config import (
    ARTICLE_BODY_MAX_CHARS_SLM,
    ARTICLE_FETCH_TIMEOUT_SEC,
    ARTICLE_FETCH_USER_AGENT,
    ARTICLES_CSV,
    CLUSTER_WINDOW_DAYS,
    CSV_COLUMNS,
    DATA_DIR,
    EVENTS_CSV,
    EVENTS_CSV_COLUMNS,
    GEOJSON_FEATURE_KEY,
    GEOJSON_PATH,
    GOOGLE_NEWS_DECODE_INTERVAL_SEC,
    GNEWS_RSS_MAX_ITEMS,
    GROUP_ALIASES,
    LOCATION_TO_STATE,
    STATE_NAME_MAP,
    VALID_EVENT_TYPES,
    VALID_STATES,
    normalize_group,
    strip_accents,
)

from user_config import (
    FETCH_QUERY_GEO,
    FETCH_QUERY_TERMS,
    LOOKBACK_DAYS,
    MAX_ARTICLES,
    MODEL_NAME,
)

__all__ = [
    "ARTICLE_BODY_MAX_CHARS_SLM",
    "ARTICLE_FETCH_TIMEOUT_SEC",
    "ARTICLE_FETCH_USER_AGENT",
    "ARTICLES_CSV",
    "CLUSTER_WINDOW_DAYS",
    "CSV_COLUMNS",
    "DATA_DIR",
    "EVENTS_CSV",
    "EVENTS_CSV_COLUMNS",
    "FETCH_QUERY_GEO",
    "FETCH_QUERY_TERMS",
    "GEOJSON_FEATURE_KEY",
    "GEOJSON_PATH",
    "GOOGLE_NEWS_DECODE_INTERVAL_SEC",
    "GNEWS_RSS_MAX_ITEMS",
    "GROUP_ALIASES",
    "LOCATION_TO_STATE",
    "LOOKBACK_DAYS",
    "MAX_ARTICLES",
    "MODEL_NAME",
    "STATE_NAME_MAP",
    "VALID_EVENT_TYPES",
    "VALID_STATES",
    "normalize_group",
    "strip_accents",
]
