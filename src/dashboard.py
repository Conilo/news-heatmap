"""Streamlit dashboard — Mexico Cartel Heatmap (v2)."""

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
# Cached assets
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_geojson() -> dict:
    with open(config.GEOJSON_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _all_state_names() -> list[str]:
    geojson = _load_geojson()
    return sorted(f["properties"]["name"] for f in geojson["features"])


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_state(state: str) -> str:
    key = (state or "").strip().lower()
    return config.STATE_NAME_MAP.get(key, state)


def _apply_group_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the 'group' column in-place using GROUP_ALIASES."""
    df = df.copy()
    df["group"] = df["group"].apply(config.normalize_group)
    return df


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline() -> None:
    from fetch import fetch_articles
    from extract import extract_article

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
        extracted.append(extract_article(article))
        progress.progress(
            (i + 1) / len(new_articles),
            text=f"SLM extraction {i + 1}/{len(new_articles)}",
        )

    progress.empty()
    append_new(extracted)
    st.success(f"Added {len(extracted)} new articles.")
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def _build_map(df: pd.DataFrame) -> go.Figure:
    """
    Build a Plotly choropleth of all 32 Mexico states.
    States with zero incidents are shown in a neutral colour.
    """
    geojson = _load_geojson()

    df = df.copy()
    df["state_norm"] = df["state"].apply(_normalize_state)

    # Aggregate: count incidents per state
    state_counts = (
        df.groupby("state_norm")
        .size()
        .reset_index(name="incidents")
    )

    # Top group per state for tooltip
    top_group = (
        df.groupby(["state_norm", "group"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("state_norm")[["state_norm", "group"]]
        .rename(columns={"group": "top_group"})
    )

    # Merge onto all-32-states baseline so every state appears
    all_states = _all_state_names()
    base = pd.DataFrame({"state_norm": all_states})
    state_counts = (
        base
        .merge(state_counts, on="state_norm", how="left")
        .merge(top_group, on="state_norm", how="left")
        .fillna({"incidents": 0, "top_group": "—"})
    )
    state_counts["incidents"] = state_counts["incidents"].astype(int)

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
        title="Incidentes por estado — clic para detalle",
    )
    fig.update_geos(
        visible=False,
        showcoastlines=False,
        showland=True,
        landcolor="#f5f5f5",
        showcountries=True,
        countrycolor="#cccccc",
        lataxis_range=[14, 33],
        lonaxis_range=[-118, -86],
    )
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar={"title": "Incidentes"},
        height=560,
        clickmode="event+select",
    )
    return fig


# ---------------------------------------------------------------------------
# State detail panel
# ---------------------------------------------------------------------------

def _render_state_detail(df: pd.DataFrame, state_name: str) -> None:
    st.subheader(f"Detalle: {state_name}")

    state_df = df[df["state"].apply(_normalize_state) == state_name].copy()

    if state_df.empty:
        st.info("No hay incidentes registrados para este estado en el período seleccionado.")
        return

    # Metrics
    top_crime = (
        state_df["crime_type"].value_counts().index[0]
        if not state_df["crime_type"].dropna().empty
        else "—"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Incidentes", len(state_df))
    m2.metric("Grupos activos", state_df[state_df["group"] != "Desconocido"]["group"].nunique())
    m3.metric("Municipios", state_df["municipality"].nunique())
    m4.metric("Tipo principal", top_crime)

    col_left, col_right = st.columns(2)

    with col_left:
        # Incidents by group
        group_counts = (
            state_df.groupby("group")
            .size()
            .reset_index(name="incidentes")
            .sort_values("incidentes", ascending=True)
        )
        fig_g = px.bar(
            group_counts,
            x="incidentes",
            y="group",
            orientation="h",
            title="Incidentes por grupo",
            labels={"group": "Grupo", "incidentes": "Incidentes"},
        )
        fig_g.update_layout(height=300, margin={"t": 40, "b": 0, "l": 0, "r": 0})
        st.plotly_chart(fig_g, use_container_width=True)

        # Crime type pie
        crime_counts = state_df["crime_type"].value_counts().reset_index()
        crime_counts.columns = ["crime_type", "count"]
        fig_c = px.pie(
            crime_counts,
            names="crime_type",
            values="count",
            title="Tipos de crimen",
        )
        fig_c.update_layout(height=280, margin={"t": 40, "b": 0, "l": 0, "r": 0})
        st.plotly_chart(fig_c, use_container_width=True)

    with col_right:
        # Top municipalities
        muni_counts = (
            state_df[state_df["municipality"] != "Desconocido"]
            .groupby("municipality")
            .size()
            .reset_index(name="incidentes")
            .sort_values("incidentes", ascending=True)
            .tail(10)
        )
        if not muni_counts.empty:
            fig_m = px.bar(
                muni_counts,
                x="incidentes",
                y="municipality",
                orientation="h",
                title="Top municipios",
                labels={"municipality": "Municipio", "incidentes": "Incidentes"},
            )
            fig_m.update_layout(height=300, margin={"t": 40, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig_m, use_container_width=True)

        # Trend over time for this state
        if "published_date" in state_df.columns:
            trend = (
                state_df.copy()
                .assign(date=lambda d: pd.to_datetime(d["published_date"], errors="coerce", utc=True).dt.date)
                .groupby("date")
                .size()
                .reset_index(name="incidentes")
            )
            fig_t = px.line(
                trend,
                x="date",
                y="incidentes",
                title="Tendencia",
                markers=True,
                labels={"date": "Fecha", "incidentes": "Incidentes"},
            )
            fig_t.update_layout(height=270, margin={"t": 40, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig_t, use_container_width=True)

    # Recent articles
    st.markdown("**Artículos recientes**")
    art_cols = ["published_date", "title", "municipality", "group", "crime_type", "source"]
    art_df = state_df[[c for c in art_cols if c in state_df.columns]].copy()
    art_df["published_date"] = pd.to_datetime(
        art_df["published_date"], errors="coerce", utc=True
    ).dt.strftime("%Y-%m-%d")
    st.dataframe(
        art_df.sort_values("published_date", ascending=False).head(20),
        use_container_width=True,
        hide_index=True,
        column_config={"title": st.column_config.TextColumn("Título", width="large")},
    )


# ---------------------------------------------------------------------------
# Aggregate trend charts
# ---------------------------------------------------------------------------

def _render_trend_charts(df: pd.DataFrame) -> None:
    st.subheader("Tendencias generales")

    col_left, col_right = st.columns(2)

    with col_left:
        # Incidents over time (daily)
        if "published_date" in df.columns:
            trend = (
                df.copy()
                .assign(date=lambda d: pd.to_datetime(d["published_date"], errors="coerce", utc=True).dt.date)
                .groupby("date")
                .size()
                .reset_index(name="incidentes")
            )
            fig_trend = px.line(
                trend,
                x="date",
                y="incidentes",
                title="Incidentes por día",
                markers=True,
                labels={"date": "Fecha", "incidentes": "Incidentes"},
            )
            fig_trend.update_layout(height=320, margin={"t": 40, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        # Top 10 states
        top_states = (
            df.copy()
            .assign(state_norm=lambda d: d["state"].apply(_normalize_state))
            .groupby("state_norm")
            .size()
            .reset_index(name="incidentes")
            .sort_values("incidentes", ascending=True)
            .tail(10)
        )
        fig_states = px.bar(
            top_states,
            x="incidentes",
            y="state_norm",
            orientation="h",
            title="Top 10 estados",
            labels={"state_norm": "Estado", "incidentes": "Incidentes"},
        )
        fig_states.update_layout(height=320, margin={"t": 40, "b": 0, "l": 0, "r": 0})
        st.plotly_chart(fig_states, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        # Top groups
        top_groups = (
            df[df["group"] != "Desconocido"]
            .groupby("group")
            .size()
            .reset_index(name="incidentes")
            .sort_values("incidentes", ascending=True)
            .tail(8)
        )
        if not top_groups.empty:
            fig_groups = px.bar(
                top_groups,
                x="incidentes",
                y="group",
                orientation="h",
                title="Grupos más activos",
                labels={"group": "Grupo", "incidentes": "Incidentes"},
            )
            fig_groups.update_layout(height=300, margin={"t": 40, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig_groups, use_container_width=True)

    with col_right2:
        # Top municipalities
        top_munis = (
            df[df["municipality"] != "Desconocido"]
            .groupby("municipality")
            .size()
            .reset_index(name="incidentes")
            .sort_values("incidentes", ascending=True)
            .tail(8)
        )
        if not top_munis.empty:
            fig_munis = px.bar(
                top_munis,
                x="incidentes",
                y="municipality",
                orientation="h",
                title="Top municipios",
                labels={"municipality": "Municipio", "incidentes": "Incidentes"},
            )
            fig_munis.update_layout(height=300, margin={"t": 40, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig_munis, use_container_width=True)


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

        df_raw = load()

        if df_raw.empty:
            st.info("No data yet. Click Refresh to fetch articles.")
            return

        # Apply group name normalisation at display time
        df_all = _apply_group_normalization(df_raw)

        # Date filter
        df_all["published_date"] = pd.to_datetime(
            df_all["published_date"], errors="coerce", utc=True
        )
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

        # Group filter (uses normalised names)
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
            (df["published_date"].dt.date >= start)
            & (df["published_date"].dt.date <= end)
        ]

    if selected_groups:
        df = df[df["group"].isin(selected_groups)]
    if selected_crimes:
        df = df[df["crime_type"].isin(selected_crimes)]

    if df.empty:
        st.warning("No articles match the current filters.")
        return

    # ---- Top metrics ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Artículos", len(df))
    m2.metric(
        "Estados afectados",
        df["state"].apply(_normalize_state).nunique(),
    )
    m3.metric(
        "Grupos identificados",
        df[df["group"] != "Desconocido"]["group"].nunique(),
    )
    m4.metric("Tipos de crimen", df["crime_type"].nunique())

    st.divider()

    # ---- Main choropleth ----
    st.subheader("Mapa de incidentes")
    st.caption("Haz clic en un estado para ver su detalle.")

    fig_main = _build_map(df)
    map_event = st.plotly_chart(
        fig_main,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="main_map",
    )

    # ---- State detail panel ----
    selected_state: str | None = None
    if map_event and map_event.get("selection") and map_event["selection"].get("points"):
        point = map_event["selection"]["points"][0]
        # Plotly choropleth stores the location name in 'location'
        selected_state = point.get("location") or point.get("hovertext")

    if selected_state:
        with st.container(border=True):
            _render_state_detail(df, selected_state)
    else:
        st.info("Haz clic en un estado del mapa para ver el detalle.", icon="👆")

    st.divider()

    # ---- Aggregate trend charts ----
    _render_trend_charts(df)

    st.divider()

    # ---- Article table ----
    st.subheader("Artículos")
    display_cols = [
        "published_date", "title", "state", "municipality",
        "group", "crime_type", "confidence", "source",
    ]
    table_df = df[[c for c in display_cols if c in df.columns]].copy()
    table_df["published_date"] = table_df["published_date"].dt.strftime("%Y-%m-%d")
    table_df["confidence"] = pd.to_numeric(table_df["confidence"], errors="coerce").round(2)
    st.dataframe(
        table_df.sort_values("published_date", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "title": st.column_config.TextColumn("Título", width="large"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1
            ),
        },
    )


if __name__ == "__main__":
    main()
