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
        # keep_default_na=False keeps empty cells as "" instead of NaN —
        # downstream code does truthiness/string-equality checks (e.g. on
        # event_id, municipality) and NaN propagation has bitten us before.
        df = pd.read_csv(
            config.ARTICLES_CSV,
            dtype=str,
            keep_default_na=False,
            na_values=[],
        )
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


def load_events() -> pd.DataFrame:
    """Load the events CSV into a DataFrame. Returns empty DataFrame if missing."""
    if not os.path.exists(config.EVENTS_CSV):
        return pd.DataFrame(columns=config.EVENTS_CSV_COLUMNS)
    try:
        df = pd.read_csv(
            config.EVENTS_CSV,
            dtype=str,
            keep_default_na=False,
            na_values=[],
        )
        for col in config.EVENTS_CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[config.EVENTS_CSV_COLUMNS]
    except Exception as exc:
        print(f"[store] Warning: could not read events CSV — {exc}")
        return pd.DataFrame(columns=config.EVENTS_CSV_COLUMNS)


def save_events(df: pd.DataFrame) -> None:
    """Persist the events DataFrame to CSV."""
    _ensure_data_dir()
    df[config.EVENTS_CSV_COLUMNS].to_csv(config.EVENTS_CSV, index=False)


def update_rows(updated: list[dict]) -> pd.DataFrame:
    """
    Overwrite existing rows in the CSV by URL with new extracted field values.

    Only rows whose URL already exists in the CSV are updated.
    Returns the full updated DataFrame.
    """
    if not updated:
        return load()

    df = load()
    updated_df = pd.DataFrame(updated)

    # Align columns
    for col in config.CSV_COLUMNS:
        if col not in updated_df.columns:
            updated_df[col] = ""
    updated_df = updated_df[config.CSV_COLUMNS]

    df = df.set_index("url")
    df.update(updated_df.set_index("url"))
    result = df.reset_index()[config.CSV_COLUMNS]
    save(result)
    print(f"[store] Updated {len(updated)} rows in-place.")
    return result
