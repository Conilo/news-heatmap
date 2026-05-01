"""Shared geographic normalization (no Streamlit). Used by cluster, dashboard, and tests."""

from __future__ import annotations

import config


def normalize_state(state: str) -> str:
    """Map SLM/display variants to canonical GeoJSON estado names."""
    key = (state or "").strip().lower()
    return config.STATE_NAME_MAP.get(key, state)
