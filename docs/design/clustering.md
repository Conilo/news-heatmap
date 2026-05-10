# Clustering design

`src/cluster.py` — groups raw articles into deduplicated events and writes
`data/events.csv`.

---

## Inputs and outputs

| | |
|---|---|
| Input | `data/articles.csv` — one row per fetched article, with SLM-extracted fields (`state`, `group`, `event_type`, `confidence`, …) |
| Output | `data/articles.csv` — same rows, `event_id` column populated |
| Output | `data/events.csv` — one row per deduplicated event |

Entry point: `recompute_events()` reloads articles from disk, runs the full
pipeline, and saves both files.  `cluster_articles(df)` is the pure function
for callers that already hold a DataFrame.

Both functions accept a `use_slm: bool = True` flag.  Pass `use_slm=False` to
suppress Stage 2 Ollama calls — useful in CI or when running the full test
suite without a local model, since Stages 1b–1d handle the common cases
deterministically.  The `--slm-live` pytest marker gates any test that requires
a real Ollama endpoint.

---

## Pipeline

All stages run on every call, in order.

### Stage 1 — Key assignment

Every article is mapped to a 3-tuple key:

```
key = (normalize_state(state), normalize_group(group), date_bucket)
date_bucket = floor(days_since_epoch / CLUSTER_WINDOW_DAYS)   # CLUSTER_WINDOW_DAYS = 5
```

Articles that share the same key are placed in the same cluster and receive
the same `event_id`.  `event_type` is intentionally excluded: the SLM often
labels the same incident differently depending on headline framing, and
including it would split one real event into multiple clusters.

**Unextracted articles** (no body, confidence = 0, state and group both
Desconocido) get a 4-tuple key with a URL suffix so they never collapse into
a single cluster.

**Event-id stability** — existing `event_id` values from the CSV are reused
so downstream consumers (dashboard, future deduplication logic) can track
events across re-runs.  The one exception is a *contested* id: if the same
`event_id` was previously written for two different keys (sign of an
over-merge in an earlier run), both keys are issued fresh UUIDs instead.

### Stage 1b — Desconocido absorb

A cluster whose `state = Desconocido` likely covers the same incident as a
known-state cluster in the same `(group, date_bucket)`.  This stage folds it
in.

Rules:

- If exactly one specific state exists for a `(group, bucket)` pair, the
  Desconocido cluster is merged into it.
- If two or more specific states exist, the Desconocido cluster is merged into
  the largest one (by article count).  The specific-state clusters are left
  separate — they may represent genuinely different events.
- `Internacional` is treated as non-geographic (same as Desconocido) and is
  never a merge target.

### Stage 1c — Cross-state title-similarity merge

For clusters sharing the same `(group, date_bucket)` but different states,
computes word-level Jaccard similarity between representative titles.  Merges
when `sim >= _TITLE_JACCARD_THRESHOLD` (0.45).

A *representative title* is the title of the highest-confidence article in
the cluster.

**Jaccard implementation** — punctuation is stripped from each token before
hashing (`"belleza,"` → `"belleza"`), and a Spanish stopword list filters
short function words.  This matters because CSV titles include inline
punctuation and trailing source names (e.g. `"… en Polanco - La Jornada"`).

The 0.45 threshold is conservative: requiring nearly half the content words to
overlap limits false merges when different events happen to involve the same
cartel and time window.

### Stage 1d — Adjacent-bucket merge

Stage 1 never compares across bucket boundaries, so a story published on day
`N` and a follow-up on day `N+1` can land in different buckets and produce two
events.  This stage catches that case.

For clusters sharing the same `(state, group)` but in consecutive buckets
(`bucket_n` and `bucket_n+1`), merges when `sim >= _ADJACENT_BUCKET_JACCARD_THRESHOLD`
(0.10).

The lower threshold (0.10 vs 1c's 0.45) is safe because the same-state
requirement is a stricter pre-condition than Stage 1c's cross-state search.
Only non-NON_GEO states are considered: Desconocido/Internacional clusters are
too ambiguous to merge across bucket boundaries on title similarity alone.

Buckets more than one step apart are never compared.

### Stage 2 — SLM disambiguation

After the rule-based stages, some cross-state same-`(group, bucket)` pairs
remain unresolved: their Jaccard is below Stage 1c's threshold (0.45) but
above near-zero.  Word overlap is too low for a deterministic decision, but
the SLM can read the full headline and judge whether the two articles describe
the same specific incident.

**Trigger band** — only pairs with Jaccard in `[_SLM_MIN_JACCARD, _TITLE_JACCARD_THRESHOLD)`
(i.e. `[0.02, 0.45)`) are sent to Ollama:

- Below 0.02: essentially no shared vocabulary → almost certainly unrelated,
  skip to avoid noise.
- Above 0.45: already handled by Stage 1c → skip.

This makes Stage 2 a **no-op with zero Ollama calls** when all remaining pairs
fall outside the uncertain band, which is the common case after Stages 1b–1d
run.

The SLM is queried with the prompt in `_SLM_DEDUP_PROMPT` and returns
`{"same_event": bool, "confidence": float}`.

---

## Confidence scoring

Each event's `confidence` field aggregates the article-level SLM extraction
confidences:

```
confidence = min(1.0, mean(article_confidence) + 0.05 × (unique_sources − 1))
```

The base is the mean of the SLM's per-article confidence scores.  Each
additional unique news source beyond the first adds a 5% corroboration bonus,
capped at 1.0.

---

## Time complexity

Let **N** = number of articles, **K** = number of unique cluster keys
(K ≤ N, typically K ≪ N).

| Stage | Complexity | Dominant cost |
|---|---|---|
| Stage 1 (key assignment) | O(N) | Two linear passes over articles |
| Stage 1b (Desconocido absorb) | O(K) | One pass over the key dict |
| Stage 1c (cross-state title-sim) | O(N + K²) | All-pairs comparison per (group, bucket); O(N) to build rep titles |
| Stage 1d (adjacent-bucket) | O(N + K) | O(N) to build rep titles; at most K adjacent pairs |
| Stage 2 (SLM) | O(K²) worst-case Jaccard scans + O(P) Ollama calls | P = pairs in uncertain band, typically P ≪ K² |
| **Total** | **O(N + K²)** | Stage 1c dominates |

At current scale (N ≈ 120 articles, K ≈ 40–50 events) Stage 1c produces at
most ~1,200 Jaccard comparisons and Stage 1d adds at most ~50.  Stage 2 has
made zero Ollama calls in practice since Stages 1b–1d leave no pairs in the
uncertain band.

---

## Known limitations

| Limitation | Planned fix |
|---|---|
| Cross-state, same-bucket pairs with very low Jaccard (e.g. factual vs. tabloid headline styles) are not merged by Stage 1c and only reach Stage 2 if Jaccard ≥ 0.02 | Stage 2 SLM handles this when triggered; entity-extraction could broaden coverage |
| SLM misclassifies some `detención` articles as `disturbio` when the body mentions fires triggered by a capture | Python-side headline priority enforcement in `extract.py` (planned) |
| `event_type` is excluded from the cluster key, so a genuine follow-up disturbio and the capture that triggered it land in the same cluster | Acceptable trade-off: prevents splitting one real event across multiple clusters due to SLM label variance |
