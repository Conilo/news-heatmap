"""Streamlit dashboard — Mexico Cartel Heatmap (v3, event deduplication)."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from store import append_new, get_processed_urls, load, load_events

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crimen Organizado Heatmap",
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
    df = df.copy()
    df["group"] = df["group"].apply(config.normalize_group)
    return df


def _prep_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and normalise group names in events DataFrame."""
    df = events_df.copy()
    df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce", utc=True)
    df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce", utc=True)
    df["article_count"] = pd.to_numeric(df["article_count"], errors="coerce").fillna(1).astype(int)
    df["unique_sources"] = pd.to_numeric(df["unique_sources"], errors="coerce").fillna(1).astype(int)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["group"] = df["group"].apply(config.normalize_group)
    df["state"] = df["state"].apply(_normalize_state)
    return df


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline() -> None:
    from cluster import recompute_events
    from extract import extract_article
    from fetch import fetch_articles

    with st.spinner("Fetching articles from Google News..."):
        articles = fetch_articles(
            lookback_days=config.LOOKBACK_DAYS,
            max_results=config.MAX_ARTICLES,
        )
    if not articles:
        st.warning(
            "No articles with downloadable full text were found. "
            "Check your connection, publisher blocks (403), or try increasing "
            "`GNEWS_RSS_MAX_ITEMS` in config.py if the feed ran out of candidates."
        )
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

    with st.spinner("Clustering articles into events..."):
        recompute_events(use_slm=False)

    st.success(f"Added {len(extracted)} new articles and updated events.")
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Article card renderer
# ---------------------------------------------------------------------------

_ARTICLE_BODY_PREVIEW_CHARS = 600


def _shorten_article_body(text: str | None, max_chars: int = _ARTICLE_BODY_PREVIEW_CHARS) -> str:
    """Single-line-ish excerpt for expandable cards (whitespace collapsed, word-boundary cut)."""
    s = str(text or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    if len(s) <= max_chars:
        return s
    chunk = s[:max_chars]
    cut = chunk.rsplit(" ", 1)[0]
    if len(cut) < max_chars // 3:
        cut = chunk
    return cut.rstrip() + "…"


def _render_article_card(row: pd.Series) -> None:
    """Render a single article's metadata below its expander label."""
    url = str(row.get("url", "") or "")
    source = str(row.get("source", "—") or "—")
    pub_date = str(row.get("published_date", "—") or "—")
    state = str(row.get("state", "—") or "—")
    municipality = str(row.get("municipality", "—") or "—")
    group = str(row.get("group", "—") or "—")
    event_type = str(row.get("event_type", "—") or "—")
    conf = float(pd.to_numeric(row.get("confidence", 0), errors="coerce") or 0.0)

    muni_part = f" · {municipality}" if municipality not in {"—", "Desconocido"} else ""
    st.caption(
        f"📅 {pub_date} &nbsp;·&nbsp; 📰 {source} &nbsp;·&nbsp; "
        f"📍 {state}{muni_part} &nbsp;·&nbsp; "
        f"🔴 {group} &nbsp;·&nbsp; ⚖️ {event_type} &nbsp;·&nbsp; conf. {conf:.2f}"
    )
    preview = _shorten_article_body(row.get("body"))
    if preview:
        st.caption("Extracto del artículo")
        st.text(preview)
    if url:
        st.markdown(f"[Ver artículo original ↗]({url})")


# ---------------------------------------------------------------------------
# Map builder  (driven by events)
# ---------------------------------------------------------------------------

def _build_map(events_df: pd.DataFrame) -> go.Figure:
    """
    Build a choropleth of all 32 Mexico states coloured by event count.
    Events already have normalised state names.
    """
    geojson = _load_geojson()

    geo = events_df[~events_df["state"].isin({"Desconocido", "Internacional"})].copy()

    state_counts = (
        geo.groupby("state")
        .size()
        .reset_index(name="eventos")
    )

    top_group = (
        geo.groupby(["state", "group"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("state")[["state", "group"]]
        .rename(columns={"group": "top_group"})
    )

    all_states = _all_state_names()
    base = pd.DataFrame({"state": all_states})
    state_counts = (
        base
        .merge(state_counts, on="state", how="left")
        .merge(top_group, on="state", how="left")
        .fillna({"eventos": 0, "top_group": "—"})
    )
    state_counts["eventos"] = state_counts["eventos"].astype(int)

    fig = px.choropleth(
        state_counts,
        geojson=geojson,
        locations="state",
        featureidkey=config.GEOJSON_FEATURE_KEY,
        color="eventos",
        color_continuous_scale="YlOrRd",
        hover_name="state",
        hover_data={"eventos": True, "top_group": True},
        labels={"eventos": "Eventos", "top_group": "Grupo principal"},
        title="Eventos por estado — clic para detalle",
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
        coloraxis_colorbar={"title": "Eventos"},
        height=560,
        clickmode="event+select",
        dragmode=False,
    )
    return fig


# ---------------------------------------------------------------------------
# State detail panel  (events + article sub-list)
# ---------------------------------------------------------------------------

def _render_state_detail(
    events_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    state_name: str,
) -> None:
    st.subheader(f"Detalle: {state_name}")

    state_events = events_df[events_df["state"] == state_name].copy()

    if state_events.empty:
        st.info("No hay eventos registrados para este estado en el período seleccionado.")
        return

    top_event_type = (
        state_events["event_type"].value_counts().index[0]
        if not state_events["event_type"].dropna().empty
        else "—"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Eventos", len(state_events))
    m2.metric("Artículos vinculados", int(state_events["article_count"].sum()))
    m3.metric("Grupos activos", state_events[state_events["group"] != "Desconocido"]["group"].nunique())
    m4.metric("Tipo principal", top_event_type)

    col_left, col_right = st.columns(2)

    with col_left:
        group_counts = (
            state_events.groupby("group")
            .size()
            .reset_index(name="eventos")
            .sort_values("eventos", ascending=True)
        )
        fig_g = px.bar(
            group_counts,
            x="eventos", y="group", orientation="h",
            title="Eventos por grupo",
            labels={"group": "Grupo", "eventos": "Eventos"},
        )
        fig_g.update_layout(height=300, margin={"t": 40, "b": 0, "l": 0, "r": 0})
        st.plotly_chart(fig_g, use_container_width=True)

        type_counts = state_events["event_type"].value_counts().reset_index()
        type_counts.columns = ["event_type", "count"]
        fig_c = px.pie(
            type_counts, names="event_type", values="count",
            title="Tipos de evento",
        )
        fig_c.update_layout(height=280, margin={"t": 40, "b": 0, "l": 0, "r": 0})
        st.plotly_chart(fig_c, use_container_width=True)

    with col_right:
        muni_counts = (
            state_events[state_events["municipality"] != "Desconocido"]
            .groupby("municipality")
            .size()
            .reset_index(name="eventos")
            .sort_values("eventos", ascending=True)
            .tail(10)
        )
        if not muni_counts.empty:
            fig_m = px.bar(
                muni_counts,
                x="eventos", y="municipality", orientation="h",
                title="Top municipios",
                labels={"municipality": "Municipio", "eventos": "Eventos"},
            )
            fig_m.update_layout(height=300, margin={"t": 40, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig_m, use_container_width=True)

        trend = (
            state_events.copy()
            .assign(date=lambda d: d["first_seen"].dt.date)
            .groupby("date")
            .size()
            .reset_index(name="eventos")
        )
        if not trend.empty:
            fig_t = px.line(
                trend, x="date", y="eventos",
                title="Tendencia de eventos",
                markers=True,
                labels={"date": "Fecha", "eventos": "Eventos"},
            )
            fig_t.update_layout(height=270, margin={"t": 40, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig_t, use_container_width=True)

    # ---- Event list with expandable article sub-lists ----
    st.markdown("**Eventos registrados**")
    state_events_sorted = state_events.sort_values("first_seen", ascending=False)

    # Join articles
    state_articles = articles_df[
        articles_df["state"].apply(_normalize_state) == state_name
    ].copy() if not articles_df.empty else pd.DataFrame()

    for _, ev in state_events_sorted.iterrows():
        label = (
            f"[{ev['event_type'].upper()}] {ev['canonical_title'][:80]} "
            f"— {ev['group']} "
            f"({ev['article_count']} art., conf {float(ev['confidence']):.2f})"
        )
        with st.expander(label):
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Artículos", int(ev["article_count"]))
            ec2.metric("Fuentes únicas", int(ev["unique_sources"]))
            ec3.metric("Confianza", f"{float(ev['confidence']):.2f}")

            if not state_articles.empty and "event_id" in state_articles.columns:
                ev_articles = state_articles[
                    state_articles["event_id"] == ev["event_id"]
                ].copy()
                if not ev_articles.empty:
                    ev_articles["published_date"] = pd.to_datetime(
                        ev_articles["published_date"], errors="coerce", utc=True
                    ).dt.strftime("%Y-%m-%d")
                    ev_articles_sorted = ev_articles.sort_values(
                        "published_date", ascending=False
                    )
                    for _, art_row in ev_articles_sorted.iterrows():
                        art_title = str(art_row.get("title", "") or "Sin título")
                        art_source = str(art_row.get("source", "—") or "—")
                        art_date = str(art_row.get("published_date", "—") or "—")
                        lbl = f"{art_title[:100]}{'…' if len(art_title) > 100 else ''}"
                        with st.expander(lbl):
                            _render_article_card(art_row)


# ---------------------------------------------------------------------------
# Aggregate trend charts  (driven by events)
# ---------------------------------------------------------------------------

def _render_trend_charts(events_df: pd.DataFrame) -> None:
    st.subheader("Tendencias generales")

    _NON_GEO = {"Desconocido", "Internacional"}
    geo_events = events_df[~events_df["state"].isin(_NON_GEO)]

    col_left, col_right = st.columns(2)

    with col_left:
        trend = (
            events_df.copy()
            .assign(date=lambda d: d["first_seen"].dt.date)
            .groupby("date")
            .size()
            .reset_index(name="eventos")
        )
        if not trend.empty:
            fig_trend = px.line(
                trend, x="date", y="eventos",
                title="Eventos por día (total)",
                markers=True,
                labels={"date": "Fecha", "eventos": "Eventos"},
            )
            fig_trend.update_layout(height=320, margin={"t": 40, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        if geo_events.empty:
            st.info("No hay datos geolocalizados para mostrar estados.")
        else:
            top_states = (
                geo_events.groupby("state")
                .size()
                .reset_index(name="eventos")
                .sort_values("eventos", ascending=True)
                .tail(10)
            )
            fig_states = px.bar(
                top_states,
                x="eventos", y="state", orientation="h",
                title="Top 10 estados (geolocalizados)",
                labels={"state": "Estado", "eventos": "Eventos"},
            )
            fig_states.update_layout(height=320, margin={"t": 40, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig_states, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        top_groups = (
            events_df[events_df["group"] != "Desconocido"]
            .groupby("group")
            .size()
            .reset_index(name="eventos")
            .sort_values("eventos", ascending=True)
            .tail(8)
        )
        if not top_groups.empty:
            fig_groups = px.bar(
                top_groups,
                x="eventos", y="group", orientation="h",
                title="Grupos más activos",
                labels={"group": "Grupo", "eventos": "Eventos"},
            )
            fig_groups.update_layout(height=300, margin={"t": 40, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig_groups, use_container_width=True)

    with col_right2:
        top_munis = (
            geo_events[geo_events["municipality"] != "Desconocido"]
            .groupby("municipality")
            .size()
            .reset_index(name="eventos")
            .sort_values("eventos", ascending=True)
            .tail(8)
        )
        if not top_munis.empty:
            fig_munis = px.bar(
                top_munis,
                x="eventos", y="municipality", orientation="h",
                title="Top municipios (geolocalizados)",
                labels={"municipality": "Municipio", "eventos": "Eventos"},
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

        # Load raw articles (for article table + filter population)
        df_raw = load()
        events_raw = load_events()

        if df_raw.empty or events_raw.empty:
            st.info("No data yet. Click Refresh to fetch articles.")
            return

        articles_df = _apply_group_normalization(df_raw)
        events_df = _prep_events(events_raw)

        # Date filter — based on event first_seen
        min_date = events_df["first_seen"].min()
        max_date = events_df["first_seen"].max()

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

        # Group filter — from events
        all_groups = sorted(events_df["group"].dropna().unique().tolist())
        selected_groups = st.multiselect("Criminal groups", options=all_groups, default=all_groups)

        # Event type filter — from events
        all_event_types = sorted(events_df["event_type"].dropna().unique().tolist())
        selected_event_types = st.multiselect("Event type", options=all_event_types, default=all_event_types)

        st.divider()
        st.caption(
            f"Articles in cache: **{len(df_raw)}**  \n"
            f"Events: **{len(events_raw)}**"
        )

    # ---- Apply filters to events ----
    ev = events_df.copy()

    if len(date_range) == 2:
        start, end = date_range
        ev = ev[
            (ev["first_seen"].dt.date >= start)
            & (ev["first_seen"].dt.date <= end)
        ]

    if selected_groups:
        ev = ev[ev["group"].isin(selected_groups)]
    if selected_event_types:
        ev = ev[ev["event_type"].isin(selected_event_types)]

    # Unassociated articles: not linked to any event (across ALL articles, unfiltered)
    if "event_id" in articles_df.columns:
        unassociated = articles_df[
            articles_df["event_id"].isna() | (articles_df["event_id"] == "")
        ].copy()
    else:
        unassociated = pd.DataFrame()

    # Also filter articles to match same event_ids (for article table)
    active_event_ids = set(ev["event_id"].dropna().tolist())
    art = articles_df.copy()
    if "event_id" in art.columns and active_event_ids:
        art = art[art["event_id"].isin(active_event_ids)]

    if ev.empty:
        st.warning("No events match the current filters.")
        return

    # ---- Top metrics ----
    n_geo = int((~ev["state"].isin({"Desconocido", "Internacional"})).sum())
    n_intl = int((ev["state"] == "Internacional").sum())
    n_arts = int(ev["article_count"].sum())
    n_unassoc = len(unassociated)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Eventos", len(ev))
    m2.metric("Artículos totales", n_arts)
    m3.metric("Geolocalizados", n_geo)
    m4.metric("Internacional", n_intl)
    m5.metric(
        "Grupos identificados",
        ev[ev["group"] != "Desconocido"]["group"].nunique(),
    )
    m6.metric("Sin evento", n_unassoc)

    st.divider()

    # ---- Main choropleth ----
    st.subheader("Mapa de eventos")
    st.caption("Haz clic en un estado para ver su detalle.")

    fig_main = _build_map(ev)
    map_event = st.plotly_chart(
        fig_main,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="main_map",
        config={"scrollZoom": False, "displayModeBar": False},
    )

    # ---- State detail panel ----
    selected_state: str | None = None
    if map_event and map_event.get("selection") and map_event["selection"].get("points"):
        point = map_event["selection"]["points"][0]
        selected_state = point.get("location") or point.get("hovertext")

    if selected_state:
        with st.container(border=True):
            _render_state_detail(ev, art, selected_state)
    else:
        st.info("Haz clic en un estado del mapa para ver el detalle.", icon="👆")

    st.divider()

    # ---- Aggregate trend charts ----
    _render_trend_charts(ev)

    st.divider()

    # ---- Article list (linked to events) ----
    st.subheader(f"Artículos ({len(art)})")
    if art.empty:
        st.info("No hay artículos vinculados a eventos en el período seleccionado.")
    else:
        art_sorted = art.copy()
        art_sorted["published_date"] = pd.to_datetime(
            art_sorted["published_date"], errors="coerce", utc=True
        ).dt.strftime("%Y-%m-%d")
        art_sorted = art_sorted.sort_values("published_date", ascending=False)
        _MAX_ARTICLES_SHOWN = 100
        shown = art_sorted.head(_MAX_ARTICLES_SHOWN)
        for _, row in shown.iterrows():
            art_title = str(row.get("title", "") or "Sin título")
            art_source = str(row.get("source", "—") or "—")
            art_date = str(row.get("published_date", "—") or "—")
            lbl = f"{art_title[:100]}{'…' if len(art_title) > 100 else ''}"
            with st.expander(lbl):
                _render_article_card(row)
        if len(art_sorted) > _MAX_ARTICLES_SHOWN:
            st.caption(
                f"Mostrando {_MAX_ARTICLES_SHOWN} de {len(art_sorted)} artículos. "
                "Aplica un filtro para reducir la lista."
            )

    st.divider()

    # ---- Unassociated articles ----
    st.subheader(f"Artículos sin evento ({n_unassoc})")
    st.caption(
        "Artículos capturados que el motor de clustering no pudo asociar a ningún evento."
    )
    if unassociated.empty:
        st.success("Todos los artículos están vinculados a un evento.")
    else:
        unassoc_sorted = unassociated.copy()
        unassoc_sorted["published_date"] = pd.to_datetime(
            unassoc_sorted["published_date"], errors="coerce", utc=True
        ).dt.strftime("%Y-%m-%d")
        unassoc_sorted = unassoc_sorted.sort_values("published_date", ascending=False)

        # Quick breakdown stats
        uc1, uc2, uc3 = st.columns(3)
        top_unassoc_state = (
            unassoc_sorted[~unassoc_sorted["state"].isin({"Desconocido", "Internacional"})]
            ["state"].value_counts()
        )
        top_unassoc_group = (
            unassoc_sorted[unassoc_sorted["group"] != "Desconocido"]
            ["group"].value_counts()
        )
        uc1.metric(
            "Estado más frecuente",
            top_unassoc_state.index[0] if not top_unassoc_state.empty else "—",
        )
        uc2.metric(
            "Grupo más frecuente",
            top_unassoc_group.index[0] if not top_unassoc_group.empty else "—",
        )
        uc3.metric(
            "Sin geolocalización",
            int((unassoc_sorted["state"] == "Desconocido").sum()),
        )

        for _, row in unassoc_sorted.iterrows():
            art_title = str(row.get("title", "") or "Sin título")
            art_source = str(row.get("source", "—") or "—")
            art_date = str(row.get("published_date", "—") or "—")
            lbl = f"{art_title[:100]}{'…' if len(art_title) > 100 else ''}"
            with st.expander(lbl):
                _render_article_card(row)


if __name__ == "__main__":
    main()
