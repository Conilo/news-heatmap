"""
Cluster articles into deduplicated events.

Stage 1 (always runs): group articles by
    (normalized_state, normalized_group, crime_type, date_bucket)
where date_bucket = floor(days_since_epoch / CLUSTER_WINDOW_DAYS).

Stage 2 (optional, SLM): for cross-key candidate pairs that share the same
group + date_bucket but differ on state or crime_type, ask the SLM whether
they describe the same specific incident and merge if confirmed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import config
from config import normalize_group
from store import load, load_events, save, save_events

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
_NON_GEO = {"Desconocido", "Internacional"}


def _date_bucket(date_val: Any) -> int:
    """Convert a date/timestamp to a CLUSTER_WINDOW_DAYS bucket integer."""
    try:
        ts = pd.to_datetime(date_val, utc=True)
        if pd.isna(ts):
            return -1
        delta = ts - _EPOCH
        return int(delta.days // config.CLUSTER_WINDOW_DAYS)
    except Exception:
        return -1


def _normalize_state(state: str) -> str:
    key = (state or "").strip().lower()
    return config.STATE_NAME_MAP.get(key, state)


def _cluster_key(row: pd.Series) -> tuple:
    """
    Return the Stage-1 grouping key for a single article row.

    crime_type is intentionally excluded: the SLM often labels the same
    incident as "narcotráfico", "homicidio", "enfrentamiento", or "otro"
    depending on headline framing, which would split one real event into
    multiple clusters.
    """
    return (
        _normalize_state(str(row.get("state", ""))),
        normalize_group(str(row.get("group", ""))),
        _date_bucket(row.get("published_date")),
    )


def _most_common(series: pd.Series, exclude: set[str] | None = None) -> str:
    """Return the most common non-excluded value, or the overall most common."""
    s = series.dropna().astype(str)
    if exclude:
        filtered = s[~s.isin(exclude)]
        if not filtered.empty:
            return filtered.mode().iloc[0]
    if s.empty:
        return "Desconocido"
    return s.mode().iloc[0]


def _compute_confidence(group_df: pd.DataFrame) -> float:
    """
    confidence = min(1.0, mean_article_confidence + 0.05 * (unique_sources - 1))
    """
    confs = pd.to_numeric(group_df["confidence"], errors="coerce").dropna()
    mean_conf = float(confs.mean()) if not confs.empty else 0.0
    unique_sources = group_df["source"].dropna().nunique()
    score = mean_conf + 0.05 * max(0, unique_sources - 1)
    return round(min(1.0, score), 4)


def _build_event_row(event_id: str, group_df: pd.DataFrame) -> dict:
    """Aggregate a group of articles into a single event dict."""
    dates = pd.to_datetime(group_df["published_date"], errors="coerce", utc=True)
    confs = pd.to_numeric(group_df["confidence"], errors="coerce")

    # Pick canonical title from the highest-confidence article
    best_idx = confs.idxmax() if not confs.dropna().empty else group_df.index[0]
    canonical_title = str(group_df.loc[best_idx, "title"])

    return {
        "event_id": event_id,
        "state": _most_common(group_df["state"].apply(_normalize_state)),
        "municipality": _most_common(group_df["municipality"], exclude={"Desconocido"}),
        "group": _most_common(group_df["group"].apply(normalize_group), exclude={"Desconocido"}),
        "crime_type": _most_common(group_df["crime_type"]),
        "first_seen": dates.min().isoformat() if not dates.dropna().empty else "",
        "last_seen": dates.max().isoformat() if not dates.dropna().empty else "",
        "article_count": len(group_df),
        "unique_sources": int(group_df["source"].dropna().nunique()),
        "confidence": _compute_confidence(group_df),
        "canonical_title": canonical_title,
    }


# ---------------------------------------------------------------------------
# Stage 1b — Desconocido-absorb merge
# ---------------------------------------------------------------------------

def _absorb_desconocido(key_to_event: dict[tuple, str]) -> dict[tuple, str]:
    """
    Merge Desconocido-state clusters into state-specific clusters when they
    share the same (group, date_bucket).

    Example: (Desconocido, CJNG, bucket_5) merges into (Nayarit, CJNG, bucket_5)
    because a Desconocido article is likely the same event, just without an
    explicit state mention.

    When multiple specific states exist for the same (group, bucket), we keep
    them separate (they might genuinely be different events) and only absorb
    the Desconocido cluster.
    """
    # Build reverse map: (group, bucket) → list of (state, event_id)
    from collections import defaultdict
    group_bucket: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for (state, group, bucket), eid in key_to_event.items():
        group_bucket[(group, bucket)].append((state, eid))

    merge_map: dict[str, str] = {}  # old_event_id → canonical_event_id

    # Build article counts per event_id for "largest cluster wins"
    eid_counts: dict[str, int] = {}
    for eid in key_to_event.values():
        eid_counts[eid] = eid_counts.get(eid, 0) + 1

    for (group, bucket), entries in group_bucket.items():
        states = {s for s, _ in entries}
        if "Desconocido" not in states:
            continue
        specific = [s for s in states if s not in _NON_GEO]
        if not specific:
            continue
        # Find the specific-state event with the most articles to absorb into
        target_eid = max(
            (eid for s, eid in entries if s in specific),
            key=lambda e: eid_counts.get(e, 0),
        )
        for s, eid in entries:
            if s == "Desconocido" and eid != target_eid:
                merge_map[eid] = target_eid

    if not merge_map:
        return key_to_event

    def _resolve(eid: str) -> str:
        seen: set[str] = set()
        while eid in merge_map and eid not in seen:
            seen.add(eid)
            eid = merge_map[eid]
        return eid

    return {k: _resolve(v) for k, v in key_to_event.items()}


# ---------------------------------------------------------------------------
# Stage 1c — Title-similarity merge for cross-state same-group clusters
# ---------------------------------------------------------------------------

def _jaccard_words(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    sw = {"de", "del", "la", "el", "en", "al", "los", "las", "y", "a", "que",
          "un", "una", "con", "por", "su", "se", "es", "fue"}
    wa = {w.lower() for w in a.split() if len(w) > 2 and w.lower() not in sw}
    wb = {w.lower() for w in b.split() if len(w) > 2 and w.lower() not in sw}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


_TITLE_JACCARD_THRESHOLD = 0.45


def _title_similarity_merge(
    articles: pd.DataFrame,
    key_to_event: dict[tuple, str],
) -> dict[tuple, str]:
    """
    For events that share the same (group, date_bucket) but differ on state,
    compute pairwise title Jaccard similarity between their representative
    titles.  If similarity > threshold, merge the smaller cluster into the
    larger one (more articles wins).
    """
    from collections import defaultdict

    # Map event_id → articles belonging to it
    articles = articles.copy()
    articles["_key"] = articles.apply(_cluster_key, axis=1)
    articles["_event_id"] = articles["_key"].map(key_to_event)
    articles["_group"] = articles["group"].apply(normalize_group)
    articles["_bucket"] = articles["published_date"].apply(_date_bucket)

    # Representative title per event: first (highest-confidence) article
    rep_title: dict[str, str] = {}
    event_size: dict[str, int] = {}
    for eid, grp in articles.groupby("_event_id"):
        confs = pd.to_numeric(grp["confidence"], errors="coerce")
        best = confs.idxmax() if not confs.dropna().empty else grp.index[0]
        rep_title[str(eid)] = str(grp.loc[best, "title"])
        event_size[str(eid)] = len(grp)

    merge_map: dict[str, str] = {}

    group_bucket_events: dict[tuple, list[str]] = defaultdict(list)
    for (state, group, bucket), eid in key_to_event.items():
        if state not in _NON_GEO:
            group_bucket_events[(group, bucket)].append(eid)

    for (group, bucket), eids in group_bucket_events.items():
        unique_eids = list(set(eids))
        if len(unique_eids) <= 1:
            continue
        for i, eid_a in enumerate(unique_eids):
            for eid_b in unique_eids[i + 1:]:
                canon_a = merge_map.get(eid_a, eid_a)
                canon_b = merge_map.get(eid_b, eid_b)
                if canon_a == canon_b:
                    continue
                sim = _jaccard_words(
                    rep_title.get(eid_a, ""),
                    rep_title.get(eid_b, ""),
                )
                if sim >= _TITLE_JACCARD_THRESHOLD:
                    # Keep the event with more articles
                    keep = canon_a if event_size.get(canon_a, 0) >= event_size.get(canon_b, 0) else canon_b
                    drop = canon_b if keep == canon_a else canon_a
                    merge_map[drop] = keep
                    print(
                        f"[cluster] Title-sim merge ({sim:.2f}): "
                        f"{rep_title.get(drop,'')[:50]} → "
                        f"{rep_title.get(keep,'')[:50]}"
                    )

    if not merge_map:
        return key_to_event

    def _resolve(eid: str) -> str:
        seen: set[str] = set()
        while eid in merge_map and eid not in seen:
            seen.add(eid)
            eid = merge_map[eid]
        return eid

    return {k: _resolve(v) for k, v in key_to_event.items()}


# ---------------------------------------------------------------------------
# Stage 2 — SLM disambiguation (optional)
# ---------------------------------------------------------------------------

_SLM_DEDUP_PROMPT = """\
Do the following two news headlines describe the exact same specific incident
(same event, same location, same people involved)?

Headline A: {a}
Headline B: {b}

Respond ONLY with valid JSON: {{"same_event": true/false, "confidence": 0.0-1.0}}
"""


def _slm_same_event(title_a: str, title_b: str) -> tuple[bool, float]:
    """Ask the SLM whether two headlines describe the same event."""
    try:
        import ollama
        prompt = _SLM_DEDUP_PROMPT.format(a=title_a, b=title_b)
        response = ollama.chat(
            model=config.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        text = response["message"]["content"]
        text = re.sub(r"```(?:json)?", "", text).strip()
        data = json.loads(text)
        return bool(data.get("same_event", False)), float(data.get("confidence", 0.5))
    except Exception as exc:
        print(f"[cluster] SLM disambiguation failed: {exc}")
        return False, 0.0


def _stage2_merge(
    articles: pd.DataFrame,
    key_to_event: dict[tuple, str],
    use_slm: bool = True,
) -> dict[tuple, str]:
    """
    For articles whose Stage-1 key differs only in state or crime_type but share
    the same (group, date_bucket), optionally ask the SLM to merge them.

    Returns updated key_to_event mapping.
    """
    if not use_slm:
        return key_to_event

    # Index articles by their Stage-1 key
    articles = articles.copy()
    articles["_key"] = articles.apply(_cluster_key, axis=1)
    articles["_event_id"] = articles["_key"].map(key_to_event)

    # Find candidate pairs: same group + date_bucket, different full key
    articles["_group_norm"] = articles["group"].apply(normalize_group)
    articles["_bucket"] = articles["published_date"].apply(_date_bucket)

    # Group by (group, bucket) to find cross-key candidates
    merge_map: dict[str, str] = {}  # old_event_id → new_event_id (canonical)

    for _, bucket_group in articles.groupby(["_group_norm", "_bucket"]):
        event_ids = bucket_group["_event_id"].unique()
        if len(event_ids) <= 1:
            continue

        # Compare each pair of distinct event_ids
        for i, eid_a in enumerate(event_ids):
            for eid_b in event_ids[i + 1:]:
                # Resolve through any already-established merges
                canon_a = merge_map.get(eid_a, eid_a)
                canon_b = merge_map.get(eid_b, eid_b)
                if canon_a == canon_b:
                    continue

                title_a = bucket_group[bucket_group["_event_id"] == eid_a]["title"].iloc[0]
                title_b = bucket_group[bucket_group["_event_id"] == eid_b]["title"].iloc[0]

                same, _conf = _slm_same_event(str(title_a), str(title_b))
                if same:
                    # Merge eid_b → eid_a (keep alphabetically smaller for stability)
                    keep = min(canon_a, canon_b)
                    drop = max(canon_a, canon_b)
                    merge_map[drop] = keep
                    print(f"[cluster] Stage-2 merge: {drop[:8]}… → {keep[:8]}…")

    if not merge_map:
        return key_to_event

    # Apply merge_map transitively to key_to_event
    def _resolve(eid: str) -> str:
        seen = set()
        while eid in merge_map and eid not in seen:
            seen.add(eid)
            eid = merge_map[eid]
        return eid

    return {k: _resolve(v) for k, v in key_to_event.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_articles(
    articles: pd.DataFrame,
    use_slm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster a DataFrame of articles into events.

    Returns:
        articles_df  — original columns + 'event_id'
        events_df    — one row per deduplicated event
    """
    if articles.empty:
        return articles.copy(), pd.DataFrame(columns=config.EVENTS_CSV_COLUMNS)

    articles = articles.copy()

    # Ensure event_id column exists
    if "event_id" not in articles.columns:
        articles["event_id"] = ""

    # Stage 1: assign a UUID to each unique (state, group, date_bucket)
    key_to_event: dict[tuple, str] = {}
    for idx, row in articles.iterrows():
        key = _cluster_key(row)
        if key not in key_to_event:
            existing = articles.loc[idx, "event_id"]
            # NaN is truthy in Python (it's a non-zero float), so a plain
            # truthy check would propagate NaN here — guard explicitly.
            if pd.isna(existing) or not str(existing).strip():
                key_to_event[key] = str(uuid.uuid4())
            else:
                key_to_event[key] = str(existing)

    # Stage 1b: absorb Desconocido-state clusters into unambiguous state clusters
    key_to_event = _absorb_desconocido(key_to_event)

    # Stage 1c: title-similarity merge for cross-state clusters (same group+bucket)
    key_to_event = _title_similarity_merge(articles, key_to_event)

    # Stage 2 (optional SLM disambiguation)
    if use_slm:
        key_to_event = _stage2_merge(articles, key_to_event, use_slm=True)

    # Assign event_id back to articles
    articles["event_id"] = articles.apply(
        lambda r: key_to_event[_cluster_key(r)], axis=1
    )

    # Build events_df: one row per event_id
    event_rows = []
    for event_id, group_df in articles.groupby("event_id"):
        event_rows.append(_build_event_row(str(event_id), group_df))

    events_df = pd.DataFrame(event_rows, columns=config.EVENTS_CSV_COLUMNS)

    return articles[config.CSV_COLUMNS], events_df


def recompute_events(use_slm: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reload articles.csv, re-run clustering from scratch, save both CSVs.

    Returns (articles_df, events_df).
    """
    articles = load()
    if articles.empty:
        print("[cluster] No articles to cluster.")
        return articles, pd.DataFrame(columns=config.EVENTS_CSV_COLUMNS)

    print(f"[cluster] Clustering {len(articles)} articles (SLM={use_slm})…")
    articles_out, events_out = cluster_articles(articles, use_slm=use_slm)

    save(articles_out)
    save_events(events_out)

    n_events = len(events_out)
    n_articles = len(articles_out)
    ratio = n_articles / n_events if n_events else 0
    print(
        f"[cluster] Done: {n_articles} articles → {n_events} events "
        f"(avg {ratio:.1f} articles/event)"
    )
    return articles_out, events_out


if __name__ == "__main__":
    arts, evts = recompute_events(use_slm=False)
    print(evts[["state", "group", "crime_type", "article_count", "confidence", "canonical_title"]].to_string())
