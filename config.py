"""Central configuration for the news analyzer pipeline."""

# ---------------------------------------------------------------------------
# SLM / Ollama
# ---------------------------------------------------------------------------
MODEL_NAME = "llama3.2:3b"   # swap to e.g. "phi3:mini" or "gemma3:4b"
OLLAMA_HOST = "http://localhost:11434"

# ---------------------------------------------------------------------------
# News fetching
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 7
MAX_ARTICLES = 100   # cap per run to keep SLM calls manageable

# Keywords used to filter cartel/crime-related articles.
# Articles must contain at least one of these (case-insensitive).
KEYWORDS = [
    "cartel",
    "cártel",
    "narco",
    "narcotrafico",
    "narcotráfico",
    "sicario",
    "crimen organizado",
    "grupo delictivo",
    "fentanilo",
    "plaza",       # as in "disputed plaza"
    "halcon",
    "halcón",
    "capo",
    "homicidio",
    "ejecución",
    "ejecutado",
    "levantón",
    "levanton",
    "desaparecido",
    "cjng",
    "cartel jalisco nueva generación",
    "cartel jalisco nueva generacion",
    "cartel de sinaloa",
    "cartel del golfo",
    "zetas",
    "beltrán leyva",
    "beltran leyva",
    "familia michoacana",
    "caballeros templarios",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data"
ARTICLES_CSV = f"{DATA_DIR}/articles.csv"
EVENTS_CSV = f"{DATA_DIR}/events.csv"
GEOJSON_PATH = "assets/mexico.geojson"

# ---------------------------------------------------------------------------
# Event clustering
# ---------------------------------------------------------------------------
CLUSTER_WINDOW_DAYS = 5   # articles within this many days share a date bucket

EVENTS_CSV_COLUMNS = [
    "event_id",
    "state",
    "municipality",
    "group",
    "crime_type",
    "first_seen",
    "last_seen",
    "article_count",
    "unique_sources",
    "confidence",
    "canonical_title",
]

# ---------------------------------------------------------------------------
# CSV schema — column names kept in one place
# ---------------------------------------------------------------------------
# GeoJSON featureidkey for Plotly choropleth
GEOJSON_FEATURE_KEY = "properties.name"

# Normalize common SLM state name variants → exact GeoJSON names
STATE_NAME_MAP: dict[str, str] = {
    "aguascalientes": "Aguascalientes",
    "baja california": "Baja California",
    "baja california norte": "Baja California",
    "baja california sur": "Baja California Sur",
    "campeche": "Campeche",
    "chiapas": "Chiapas",
    "chihuahua": "Chihuahua",
    "ciudad de mexico": "Ciudad de México",
    "ciudad de méxico": "Ciudad de México",
    "cdmx": "Ciudad de México",
    "coahuila": "Coahuila",
    "coahuila de zaragoza": "Coahuila",
    "colima": "Colima",
    "durango": "Durango",
    "guanajuato": "Guanajuato",
    "guerrero": "Guerrero",
    "hidalgo": "Hidalgo",
    "jalisco": "Jalisco",
    "mexico": "México",
    "méxico": "México",
    "estado de mexico": "México",
    "estado de méxico": "México",
    "michoacan": "Michoacán",
    "michoacán": "Michoacán",
    "morelos": "Morelos",
    "nayarit": "Nayarit",
    "nuevo leon": "Nuevo León",
    "nuevo león": "Nuevo León",
    "oaxaca": "Oaxaca",
    "puebla": "Puebla",
    "queretaro": "Querétaro",
    "querétaro": "Querétaro",
    "quintana roo": "Quintana Roo",
    "san luis potosi": "San Luis Potosí",
    "san luis potosí": "San Luis Potosí",
    "sinaloa": "Sinaloa",
    "sonora": "Sonora",
    "tabasco": "Tabasco",
    "tamaulipas": "Tamaulipas",
    "tlaxcala": "Tlaxcala",
    "veracruz": "Veracruz",
    "yucatan": "Yucatán",
    "yucatán": "Yucatán",
    "zacatecas": "Zacatecas",
}

# ---------------------------------------------------------------------------
# Group name canonicalization
# ---------------------------------------------------------------------------
# Maps any lowercase variant the SLM might produce → canonical display name.
GROUP_ALIASES: dict[str, str] = {
    # CJNG
    "cjng": "CJNG",
    "cartel jalisco nueva generacion": "CJNG",
    "cartel jalisco nueva generación": "CJNG",
    "cártel jalisco nueva generacion": "CJNG",
    "cártel jalisco nueva generación": "CJNG",
    "jalisco nueva generacion": "CJNG",
    "jalisco nueva generación": "CJNG",
    "cartel de jalisco": "CJNG",
    "cártel de jalisco": "CJNG",
    "cartel de jalisco nueva generacion": "CJNG",
    "cartel de jalisco nueva generación": "CJNG",
    "cártel de jalisco nueva generacion": "CJNG",
    "cártel de jalisco nueva generación": "CJNG",
    "el mencho": "CJNG",              # Nemesio Oseguera Cervantes, CJNG leader
    "nemesio oseguera": "CJNG",
    "cng": "CJNG",                    # common SLM typo for CJNG
    # Cártel de Sinaloa
    "cartel de sinaloa": "Cártel de Sinaloa",
    "cártel de sinaloa": "Cártel de Sinaloa",
    "sinaloa cartel": "Cártel de Sinaloa",
    # NOTE: bare "sinaloa" removed — ambiguous with the state name
    "los chapitos": "Cártel de Sinaloa",
    "chapitos": "Cártel de Sinaloa",
    "ismael zambada": "Cártel de Sinaloa",
    "el mayo": "Cártel de Sinaloa",
    "los salazar": "Los Salazar",     # Sinaloa-aligned faction, keep distinct
    # Cártel del Pacífico  (keep distinct per user request)
    "cartel del pacifico": "Cártel del Pacífico",
    "cártel del pacifico": "Cártel del Pacífico",
    "cartel del pacífico": "Cártel del Pacífico",
    "cártel del pacífico": "Cártel del Pacífico",
    # Cártel del Golfo
    "cartel del golfo": "Cártel del Golfo",
    "cártel del golfo": "Cártel del Golfo",
    "gulf cartel": "Cártel del Golfo",
    "cdg": "Cártel del Golfo",
    # Los Zetas
    "los zetas": "Los Zetas",
    "zetas": "Los Zetas",
    # Cártel del Noreste
    "cartel del noreste": "Cártel del Noreste",
    "cártel del noreste": "Cártel del Noreste",
    "cdn": "Cártel del Noreste",
    # Beltrán Leyva
    "beltran leyva": "Beltrán Leyva",
    "beltrán leyva": "Beltrán Leyva",
    "organizacion beltran leyva": "Beltrán Leyva",
    "organización beltrán leyva": "Beltrán Leyva",
    "obl": "Beltrán Leyva",
    # La Familia Michoacana
    "familia michoacana": "La Familia Michoacana",
    "la familia michoacana": "La Familia Michoacana",
    "la familia": "La Familia Michoacana",
    # Nueva Familia Michoacana (distinct from La Familia Michoacana)
    "nueva familia michoacana": "Nueva Familia Michoacana",
    "nueva familia": "Nueva Familia Michoacana",
    # Caballeros Templarios
    "caballeros templarios": "Caballeros Templarios",
    "los caballeros templarios": "Caballeros Templarios",
    "knights templar": "Caballeros Templarios",
    # Guerreros Unidos
    "guerreros unidos": "Guerreros Unidos",
    # Los Viagras
    "los viagras": "Los Viagras",
    "viagras": "Los Viagras",
}


import re as _re
import unicodedata as _unicodedata


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in _unicodedata.normalize("NFD", s)
        if _unicodedata.category(c) != "Mn"
    )


def normalize_group(name: str) -> str:
    """
    Return the canonical group name for any known alias.

    Three-step pipeline:
      1. If the SLM returned comma-separated groups, take the first one.
      2. Strip parenthetical abbreviations such as "(CJNG)" or "(CDG)".
      3. Exact lowercase lookup, then accent-stripped fallback lookup.
    """
    if not name or name.strip().lower() in ("desconocido", "unknown", ""):
        return "Desconocido"

    # Step 1: take only the first group when multiple are comma-separated
    first = name.split(",")[0].strip()

    # Step 2: remove parenthetical content e.g. "(CJNG)" / "(Cártel del Golfo)"
    cleaned = _re.sub(r"\s*\(.*?\)", "", first).strip()

    # Step 3a: exact lowercase lookup
    key = cleaned.lower()
    if key in GROUP_ALIASES:
        return GROUP_ALIASES[key]

    # Step 3b: accent-stripped fallback (handles é/e, á/a mismatches)
    key_no_accent = _strip_accents(key)
    for alias_key, canonical in GROUP_ALIASES.items():
        if _strip_accents(alias_key) == key_no_accent:
            return canonical

    # Return the cleaned string so parentheticals are at least removed
    return cleaned


# Cartel colors for the map (add more as needed)
GROUP_COLORS: dict[str, str] = {
    "Cártel de Sinaloa": "#1f77b4",
    "CJNG": "#d62728",
    "Cártel del Golfo": "#2ca02c",
    "Los Zetas": "#9467bd",
    "Cártel del Pacífico": "#17becf",
    "Beltrán Leyva": "#8c564b",
    "La Familia Michoacana": "#e377c2",
    "Nueva Familia Michoacana": "#f7b6d2",
    "Caballeros Templarios": "#7f7f7f",
    "Cártel del Noreste": "#bcbd22",
    "Los Salazar": "#ffbb78",
    "Guerreros Unidos": "#98df8a",
    "Los Viagras": "#c5b0d5",
    "Desconocido": "#aec7e8",
}

# ---------------------------------------------------------------------------
# CSV schema — column names kept in one place
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "url",
    "title",
    "description",
    "published_date",
    "source",
    "state",
    "municipality",
    "group",
    "crime_type",
    "confidence",
    "processed_at",
    "event_id",
]
