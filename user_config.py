"""Settings you are most likely to change day to day.

Edit this file for model choice, how far back to look, how many articles to
pull, and which search terms anchor the Google News query.
"""

# ---------------------------------------------------------------------------
# Small language model
# ---------------------------------------------------------------------------
MODEL_NAME = "llama3.2:3b"  # e.g. "llama3.2:3b" or "gemma3:4b"

# ---------------------------------------------------------------------------
# News fetching
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 14
MAX_ARTICLES = 30  # target count of articles with full text per run (SLM + storage)

# OR-joined terms for the Google News query. Keep ~8–10 broad terms; very long
# OR-chains weaken relevance. Multi-word terms are auto-quoted in fetch.
FETCH_QUERY_TERMS = [
    "narcotráfico",
    "cartel",
    "sicario",
    "crimen organizado",
    "fentanilo",
    "homicidio",
    "ejecutado",
    "grupo delictivo",
]

# Geographic anchor implicitly AND-ed with the OR block above.
FETCH_QUERY_GEO = "mexico"
