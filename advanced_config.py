"""Power-user settings: RSS limits, HTTP, paths, clustering, CSV schemas, aliases."""

from __future__ import annotations

import re as _re
import unicodedata as _unicodedata

from user_config import MAX_ARTICLES

# ---------------------------------------------------------------------------
# News fetching (beyond user-facing knobs)
# ---------------------------------------------------------------------------
GNEWS_RSS_MAX_ITEMS = max(MAX_ARTICLES * 4, 100)

# Truncate article text for CSV and SLM context (newspaper3k download).
ARTICLE_BODY_MAX_CHARS_SLM = 8000

# Delay between Google News URL decodes; 0 disables.
GOOGLE_NEWS_DECODE_INTERVAL_SEC = 0.0

# newspaper3k (some publishers 403 bare bots; not a paywall workaround).
ARTICLE_FETCH_TIMEOUT_SEC = 25
ARTICLE_FETCH_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data"
ARTICLES_CSV = f"{DATA_DIR}/articles.csv"
EVENTS_CSV = f"{DATA_DIR}/events.csv"
GEOJSON_PATH = "assets/mexico.geojson"

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
CLUSTER_WINDOW_DAYS = 5  # articles within this many days share a date bucket

# ---------------------------------------------------------------------------
# Schemas & map wiring
# ---------------------------------------------------------------------------
EVENTS_CSV_COLUMNS = [
    "event_id",
    "state",
    "municipality",
    "group",
    "event_type",
    "first_seen",
    "last_seen",
    "article_count",
    "unique_sources",
    "confidence",
    "canonical_title",
]

GEOJSON_FEATURE_KEY = "properties.name"

CSV_COLUMNS = [
    "url",
    "title",
    "description",
    "body",
    "published_date",
    "source",
    "state",
    "municipality",
    "group",
    "event_type",
    "confidence",
    "processed_at",
    "event_id",
]

# ---------------------------------------------------------------------------
# State names: SLM / text variants → exact GeoJSON feature names
# ---------------------------------------------------------------------------
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
# Municipality / locality → canonical GeoJSON state (accent-folded substring match)
# ---------------------------------------------------------------------------
_LOCATION_TO_STATE_ENTRIES: tuple[tuple[str, str], ...] = (
    ("Culiacán", "Sinaloa"),
    ("Mazatlán", "Sinaloa"),
    ("Los Mochis", "Sinaloa"),
    ("Guasave", "Sinaloa"),
    ("Guadalajara", "Jalisco"),
    ("Zapopan", "Jalisco"),
    ("Puerto Vallarta", "Jalisco"),
    ("Ciudad Juárez", "Chihuahua"),
    ("Parral", "Chihuahua"),
    ("Monterrey", "Nuevo León"),
    ("San Pedro Garza García", "Nuevo León"),
    ("San Pedro Garza", "Nuevo León"),
    ("Linares", "Nuevo León"),
    ("Tijuana", "Baja California"),
    ("Ensenada", "Baja California"),
    ("Mexicali", "Baja California"),
    ("Acapulco", "Guerrero"),
    ("Chilpancingo", "Guerrero"),
    ("Iguala", "Guerrero"),
    ("Morelia", "Michoacán"),
    ("Uruapan", "Michoacán"),
    ("Apatzingán", "Michoacán"),
    ("Reynosa", "Tamaulipas"),
    ("Matamoros", "Tamaulipas"),
    ("Nuevo Laredo", "Tamaulipas"),
    ("Tampico", "Tamaulipas"),
    ("Torreón", "Coahuila"),
    ("Saltillo", "Coahuila"),
    ("Piedras Negras", "Coahuila"),
    ("Cancún", "Quintana Roo"),
    ("Playa del Carmen", "Quintana Roo"),
    ("Chetumal", "Quintana Roo"),
    ("Mérida", "Yucatán"),
    ("Valladolid", "Yucatán"),
    ("Salina Cruz", "Oaxaca"),
    ("Tuxtepec", "Oaxaca"),
    ("Coatzacoalcos", "Veracruz"),
    ("Poza Rica", "Veracruz"),
    ("Xalapa", "Veracruz"),
    ("Veracruz", "Veracruz"),
    ("Villahermosa", "Tabasco"),
    ("Cárdenas", "Tabasco"),
    ("Tuxtla Gutiérrez", "Chiapas"),
    ("San Cristóbal", "Chiapas"),
    ("Tapachula", "Chiapas"),
    ("Hermosillo", "Sonora"),
    ("Nogales", "Sonora"),
    ("Caborca", "Sonora"),
    ("Cajeme", "Sonora"),
    ("Tepic", "Nayarit"),
    ("Bahía de Banderas", "Nayarit"),
    ("Gómez Palacio", "Durango"),
    ("Durango", "Durango"),
    ("Fresnillo", "Zacatecas"),
    ("Manzanillo", "Colima"),
    ("Colima", "Colima"),
    ("Irapuato", "Guanajuato"),
    ("Celaya", "Guanajuato"),
    ("Salamanca", "Guanajuato"),
    ("León", "Guanajuato"),
    ("Tehuacán", "Puebla"),
    ("Puebla", "Puebla"),
    ("San Juan del Río", "Querétaro"),
    ("Querétaro", "Querétaro"),
    ("Aguascalientes", "Aguascalientes"),
    ("Ciudad Valles", "San Luis Potosí"),
    ("San Luis Potosí", "San Luis Potosí"),
    ("Los Cabos", "Baja California Sur"),
    ("La Paz", "Baja California Sur"),
    ("Campeche", "Campeche"),
    ("Tlaxcala", "Tlaxcala"),
    ("Pachuca", "Hidalgo"),
    ("Tula", "Hidalgo"),
    ("Naucalpan", "México"),
    ("Ecatepec", "México"),
    ("Texcoco", "México"),
    ("Toluca", "México"),
    ("Iztapalapa", "Ciudad de México"),
    ("Tepito", "Ciudad de México"),
    ("Cuernavaca", "Morelos"),
    ("Cuautla", "Morelos"),
    ("Chihuahua", "Chihuahua"),
    ("Zacatecas", "Zacatecas"),
    ("Oaxaca", "Oaxaca"),
)

# ---------------------------------------------------------------------------
# Group name canonicalization (lowercase alias → display name)
# ---------------------------------------------------------------------------
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
    "el mencho": "CJNG",
    "nemesio oseguera": "CJNG",
    "cng": "CJNG",
    # Cártel de Sinaloa
    "cartel de sinaloa": "Cártel de Sinaloa",
    "cártel de sinaloa": "Cártel de Sinaloa",
    "sinaloa cartel": "Cártel de Sinaloa",
    "los chapitos": "Cártel de Sinaloa",
    "chapitos": "Cártel de Sinaloa",
    "ismael zambada": "Cártel de Sinaloa",
    "el mayo": "Cártel de Sinaloa",
    "los salazar": "Los Salazar",
    # Cártel del Pacífico
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
    # Nueva Familia Michoacana
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


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in _unicodedata.normalize("NFD", s)
        if _unicodedata.category(c) != "Mn"
    )


def _build_location_to_state() -> dict[str, str]:
    out: dict[str, str] = {}
    for place, est in _LOCATION_TO_STATE_ENTRIES:
        k = _strip_accents(place).casefold().strip()
        out[k] = est
    return out


LOCATION_TO_STATE: dict[str, str] = _build_location_to_state()


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

    first = name.split(",")[0].strip()
    cleaned = _re.sub(r"\s*\(.*?\)", "", first).strip()

    key = cleaned.lower()
    if key in GROUP_ALIASES:
        return GROUP_ALIASES[key]

    key_no_accent = _strip_accents(key)
    for alias_key, canonical in GROUP_ALIASES.items():
        if _strip_accents(alias_key) == key_no_accent:
            return canonical

    return cleaned
