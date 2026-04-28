# News Analyzer — Mexico Cartel Heatmap

A local pipeline that fetches Google News, filters cartel/narco-related articles from Mexico, uses a Small Language Model to extract structured data, and renders an interactive state-level heatmap in Streamlit.

## Architecture

```
gnews fetch → keyword filter → SLM extraction (Ollama) → CSV cache → Streamlit dashboard
```

## Prerequisites

1. **Ollama** — install from https://ollama.com, then pull the default model:
   ```bash
   ollama pull llama3.2:3b
   ```

2. **Conda environment** (recommended — works with Miniconda/Anaconda):
   ```bash
   conda create -n news-analyzer python=3.11 -y
   conda activate news-analyzer
   pip install -r requirements.txt
   ```

   Or with plain venv:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Running

```bash
conda activate news-analyzer
streamlit run src/dashboard.py
```

The dashboard has a **Refresh** button that fetches new articles, runs SLM extraction, and updates the map.

## Configuration

Edit `config.py` to change:
- `MODEL_NAME` — Ollama model (e.g. `"llama3.2:3b"`, `"phi3:mini"`, `"gemma3:4b"`)
- `LOOKBACK_DAYS` — how many days back to fetch news
- `MAX_ARTICLES` — cap on articles fetched per run
- `KEYWORDS` — cartel/crime terms used to filter articles

## Output

The dashboard shows:
- **Choropleth map** of Mexico colored by incident count per state, with one color per criminal group
- **Sidebar filters**: date range, group, crime type
- **Article table** with title, extracted state/municipality, group, and crime type
