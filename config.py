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
    "fentanyl",
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
    "jalisco",     # Cártel Jalisco Nueva Generación shorthand context
    "sinaloa cartel",
    "gulf cartel",
    "cartel del golfo",
    "los zetas",
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
GEOJSON_PATH = "assets/mexico.geojson"

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

# Cartel colors for the map (add more as needed)
GROUP_COLORS: dict[str, str] = {
    "Cártel de Sinaloa": "#1f77b4",
    "CJNG": "#d62728",
    "Cártel del Golfo": "#2ca02c",
    "Los Zetas": "#9467bd",
    "Cártel Jalisco Nueva Generación": "#d62728",
    "Beltrán Leyva": "#8c564b",
    "La Familia Michoacana": "#e377c2",
    "Caballeros Templarios": "#7f7f7f",
    "Cártel del Noreste": "#bcbd22",
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
]
