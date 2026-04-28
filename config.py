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
    "cartel jalisco nueva generación",
    "cartel jalisco nueva generacion",
    "cartel de sinaloa",
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
    # Cártel de Sinaloa
    "cartel de sinaloa": "Cártel de Sinaloa",
    "cártel de sinaloa": "Cártel de Sinaloa",
    "sinaloa cartel": "Cártel de Sinaloa",
    "sinaloa": "Cártel de Sinaloa",   # only when used as group name
    "los chapitos": "Cártel de Sinaloa",
    "chapitos": "Cártel de Sinaloa",
    "ismael zambada": "Cártel de Sinaloa",
    "el mayo": "Cártel de Sinaloa",
    # Cártel del Golfo
    "cartel del golfo": "Cártel del Golfo",
    "cártel del golfo": "Cártel del Golfo",
    "gulf cartel": "Cártel del Golfo",
    "cdg": "Cártel del Golfo",
    # Los Zetas
    "los zetas": "Los Zetas",
    "zetas": "Los Zetas",
    "z": "Los Zetas",
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
    # Caballeros Templarios
    "caballeros templarios": "Caballeros Templarios",
    "los caballeros templarios": "Caballeros Templarios",
    "knights templar": "Caballeros Templarios",
    # Guerreros Unidos
    "guerreros unidos": "Guerreros Unidos",
    # Viagra / Los Viagras
    "los viagras": "Los Viagras",
    "viagras": "Los Viagras",
}


def normalize_group(name: str) -> str:
    """Return the canonical group name for any known alias, else title-case the input."""
    if not name:
        return "Desconocido"
    canonical = GROUP_ALIASES.get(name.strip().lower())
    return canonical if canonical else name


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
