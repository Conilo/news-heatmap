"""Streamlit dashboard — Mexico Cartel Heatmap."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from fetch import fetch_articles
from extract import extract_articles
from store import append_new, get_processed_urls, load

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mexico Cartel Heatmap",
    page_icon="🗺️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_geojson() -> dict:
    with open(config.GEOJSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def _normalize_state(state: str) -> str:
    """Map SLM state output to exact GeoJSON name."""
    key = (state or "").strip().lower()
    return config.STATE_NAME_MAP.get(key, state)


def _run_pipeline() -> None:
    """Fetch, extract, and store new articles."""
    with st.spinner("Fetching articles from Google News..."):
        articles = fetch_articles(
            lookback_days=config.LOOKBACK_DAYS,
            max_results=config.MAX_ARTICLES,
        )
    if not articles:
        st.warning("No articles fetched. Check your internet connection or try again.")
        return

    already_processed = get_processed_urls()
    new_articles = [a for a in articles if a.get("url") not in already_processed]

    if not new_articles:
        st.info("No new articles to process — all already cached.")
        return

    progress = st.progress(0, text="Running SLM extraction...")
    extracted = []
    for i, article in enumerate(new_articles):
        from extract import extract_article
        extracted.append(extract_article(article))
        progress.progress((i + 1) / len(new_articles), text=f"SLM extraction {i+1}/{len(new_articles)}")

    progress.empty()
    append_new(extracted)
    st.success(f"Added {len(extracted)} new articles.")
    st.cache_data.clear()


def _build_map(df: pd.DataFrame, selected_groups: list[str]) -> go.Figure:
    """Build a Plotly choropleth of Mexico colored by incident count."""
    geojson = _load_geojson()

    # Normalize state names
    df = df.copy()
    df["state_norm"] = df["state"].apply(_normalize_state)

    # Filter to selected groups
    if selected_groups:
        df = df[df["group"].isin(selected_groups)]

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data for selected filters.")
        return fig

    # Aggregate: count incidents per state (total)
    state_counts = (
        df.groupby("state_norm")
        .size()
        .reset_index(name="incidents")
    )

    # Top group per state (for tooltip)
    top_group = (
        df.groupby(["state_norm", "group"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .drop_duplicates("state_norm")
        .rename(columns={"group": "top_group"})[["state_norm", "top_group"]]
    )

    state_counts = state_counts.merge(top_group, on="state_norm", how="left")

    fig = px.choropleth(
        state_counts,
        geojson=geojson,
        locations="state_norm",
        featureidkey=config.GEOJSON_FEATURE_KEY,
        color="incidents",
        color_continuous_scale="YlOrRd",
        hover_name="state_norm",
        hover_data={"incidents": True, "top_group": True},
        labels={"incidents": "Incidentes", "top_group": "Grupo principal"},
        title="Incidentes por estado",
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
    )
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar={"title": "Incidentes"},
        height=550,
    )
    return fig


def _build_group_map(df: pd.DataFrame, selected_group: str) -> go.Figure:
    """Build a choropleth for a single group."""
    geojson = _load_geojson()
    df = df.copy()
    df["state_norm"] = df["state"].apply(_normalize_state)
    subset = df[df["group"] == selected_group]

    if subset.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data for {selected_group}")
        return fig

    state_counts = (
        subset.groupby("state_norm")
        .size()
        .reset_index(name="incidents")
    )

    color = config.GROUP_COLORS.get(selected_group, "#1f77b4")

    fig = px.choropleth(
        state_counts,
        geojson=geojson,
        locations="state_norm",
        featureidkey=config.GEOJSON_FEATURE_KEY,
        color="incidents",
        color_continuous_scale=[[0, "#f0f0f0"], [1, color]],
        hover_name="state_norm",
        hover_data={"incidents": True},
        labels={"incidents": "Incidentes"},
        title=f"{selected_group} — incidentes por estado",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🗺️ Mexico Cartel Heatmap")
    st.caption(
        "Powered by Google News + Ollama SLM · "
        f"Model: `{config.MODEL_NAME}` · "
        f"Last {config.LOOKBACK_DAYS} days"
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Controls")

        if st.button("🔄 Refresh (fetch + extract)", use_container_width=True):
            _run_pipeline()
            st.rerun()

        st.divider()
        st.subheader("Filters")

        df_all = load()

        if df_all.empty:
            st.info("No data yet. Click Refresh to fetch articles.")
            return

        # Date filter
        df_all["published_date"] = pd.to_datetime(df_all["published_date"], errors="coerce", utc=True)
        min_date = df_all["published_date"].min()
        max_date = df_all["published_date"].max()

        if pd.isna(min_date):
            min_date = datetime.now(timezone.utc) - timedelta(days=config.LOOKBACK_DAYS)
        if pd.isna(max_date):
            max_date = datetime.now(timezone.utc)

        date_range = st.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

        # Group filter
        all_groups = sorted(df_all["group"].dropna().unique().tolist())
        selected_groups = st.multiselect(
            "Criminal groups",
            options=all_groups,
            default=all_groups,
        )

        # Crime type filter
        all_crimes = sorted(df_all["crime_type"].dropna().unique().tolist())
        selected_crimes = st.multiselect(
            "Crime type",
            options=all_crimes,
            default=all_crimes,
        )

        st.divider()
        st.caption(f"Total articles in cache: **{len(df_all)}**")

    # ---- Apply filters ----
    df = df_all.copy()

    if len(date_range) == 2:
        start, end = date_range
        df = df[
            (df["published_date"].dt.date >= start) &
            (df["published_date"].dt.date <= end)
        ]

    if selected_groups:
        df = df[df["group"].isin(selected_groups)]
    if selected_crimes:
        df = df[df["crime_type"].isin(selected_crimes)]

    if df.empty:
        st.warning("No articles match the current filters.")
        return

    # ---- Main choropleth ----
    st.subheader("Incidents by state")
    fig_main = _build_map(df, selected_groups=[])
    st.plotly_chart(fig_main, use_container_width=True)

    # ---- Per-group mini maps ----
    if len(all_groups) > 1:
        st.subheader("Breakdown by criminal group")
        groups_to_show = [g for g in selected_groups if g != "Desconocido"]
        if groups_to_show:
            cols = st.columns(min(len(groups_to_show), 2))
            for i, group in enumerate(groups_to_show):
                with cols[i % 2]:
                    st.plotly_chart(
                        _build_group_map(df, group),
                        use_container_width=True,
                    )

    # ---- Stats row ----
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Articles", len(df))
    col2.metric("States affected", df["state"].apply(_normalize_state).nunique())
    col3.metric("Groups identified", df[df["group"] != "Desconocido"]["group"].nunique())
    col4.metric("Crime types", df["crime_type"].nunique())

    # ---- Article table ----
    st.subheader("Article details")
    display_cols = ["published_date", "title", "state", "municipality", "group", "crime_type", "confidence", "source"]
    table_df = df[[c for c in display_cols if c in df.columns]].copy()
    table_df["published_date"] = table_df["published_date"].dt.strftime("%Y-%m-%d")
    table_df["confidence"] = pd.to_numeric(table_df["confidence"], errors="coerce").round(2)
    st.dataframe(
        table_df.sort_values("published_date", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "title": st.column_config.TextColumn("Title", width="large"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
        },
    )


if __name__ == "__main__":
    main()
