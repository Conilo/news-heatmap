"""CSV-based persistence layer for processed articles."""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=config.CSV_COLUMNS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    """Load the articles CSV into a DataFrame. Returns empty DataFrame if missing."""
    if not os.path.exists(config.ARTICLES_CSV):
        return _empty_df()
    try:
        df = pd.read_csv(config.ARTICLES_CSV, dtype=str)
        # Ensure all expected columns exist
        for col in config.CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[config.CSV_COLUMNS]
    except Exception as exc:
        print(f"[store] Warning: could not read CSV — {exc}")
        return _empty_df()


def save(df: pd.DataFrame) -> None:
    """Persist a DataFrame to the articles CSV, creating dirs as needed."""
    _ensure_data_dir()
    df[config.CSV_COLUMNS].to_csv(config.ARTICLES_CSV, index=False)


def append_new(new_articles: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Merge new articles into the existing CSV, deduplicating by URL.

    Returns the full updated DataFrame.
    """
    existing = load()
    existing_urls: set[str] = set(existing["url"].dropna().tolist())

    fresh = [a for a in new_articles if a.get("url", "") not in existing_urls]

    if not fresh:
        print("[store] No new articles to add.")
        return existing

    fresh_df = pd.DataFrame(fresh)
    # Align columns
    for col in config.CSV_COLUMNS:
        if col not in fresh_df.columns:
            fresh_df[col] = ""
    fresh_df = fresh_df[config.CSV_COLUMNS]

    combined = pd.concat([existing, fresh_df], ignore_index=True)
    # Drop any remaining duplicates (safety net)
    combined = combined.drop_duplicates(subset=["url"], keep="first")
    save(combined)
    print(f"[store] Added {len(fresh)} new articles (total: {len(combined)}).")
    return combined


def get_processed_urls() -> set[str]:
    """Return the set of URLs already in the CSV (already SLM-processed)."""
    df = load()
    return set(df["url"].dropna().tolist())
