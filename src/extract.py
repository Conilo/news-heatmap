"""Use a local Ollama SLM to extract structured data from each article."""

from __future__ import annotations

import json
import re
import sys
import os
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama

import config

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a structured data extractor specialized in Mexican crime and cartel news.
Given a news article title, a short Google News description snippet, and the article body text, extract the following fields as valid JSON.
Respond ONLY with valid JSON — no markdown fences, no explanation, nothing else.

=== FIELDS ===

state:
  One of the 32 Mexican state names in Spanish (e.g. "Sinaloa", "Jalisco", "Tamaulipas"), OR:
  - "Internacional"  — use this when the event CLEARLY happened outside Mexico
    (e.g. arrests in the USA, Europe, Asia, Africa; court cases in the US; seizures abroad).
  - "Desconocido"   — use this ONLY when the location is genuinely unknown and
    the article does not provide enough clues, even indirect ones.

  IMPORTANT — infer state from city/municipality names and indirect clues:
    Culiacán, Mazatlán, Los Mochis, Guasave         → Sinaloa
    Guadalajara, Zapopan, Puerto Vallarta, Tepic*    → Jalisco  (*Tepic = Nayarit)
    Ciudad Juárez, Chihuahua, Parral                 → Chihuahua
    Monterrey, San Pedro Garza, Linares              → Nuevo León
    Tijuana, Ensenada, Mexicali                      → Baja California
    Acapulco, Chilpancingo, Iguala                   → Guerrero
    Morelia, Uruapan, Apatzingán                     → Michoacán
    Reynosa, Matamoros, Nuevo Laredo, Tampico        → Tamaulipas
    Torreón, Saltillo, Piedras Negras                → Coahuila
    Cancún, Playa del Carmen, Chetumal               → Quintana Roo
    Mérida, Valladolid                               → Yucatán
    Oaxaca, Salina Cruz, Tuxtepec                    → Oaxaca
    Veracruz, Xalapa, Coatzacoalcos, Poza Rica       → Veracruz
    Villahermosa, Cárdenas                           → Tabasco
    Tuxtla Gutiérrez, San Cristóbal, Tapachula       → Chiapas
    Hermosillo, Nogales, Caborca, Cajeme             → Sonora
    Tepic, Bahía de Banderas                         → Nayarit
    Durango, Gómez Palacio                           → Durango
    Zacatecas, Fresnillo                             → Zacatecas
    Colima, Manzanillo                               → Colima
    León, Irapuato, Celaya, Salamanca                → Guanajuato
    Puebla, Tehuacán                                 → Puebla
    Querétaro, San Juan del Río                      → Querétaro
    Aguascalientes                                   → Aguascalientes
    San Luis Potosí, Ciudad Valles                   → San Luis Potosí
    La Paz, Los Cabos                                → Baja California Sur
    Campeche                                         → Campeche
    Chetumal                                         → Quintana Roo
    Tlaxcala                                         → Tlaxcala
    Pachuca, Tula                                    → Hidalgo
    Toluca, Naucalpan, Ecatepec, Texcoco             → México (Estado de México)
    Ciudad de México, CDMX, Iztapalapa, Tepito       → Ciudad de México
    Cuernavaca, Cuautla                              → Morelos
    Chilpancingo, Acapulco, Iguala                   → Guerrero

  IMPORTANT: Do NOT infer state from the cartel's home territory alone.
  A CJNG story with no explicit location is "Desconocido", NOT "Jalisco".
  Cartels operate across many states; only use explicit geographic clues.

municipality:
  City or municipality name in Spanish, or "Desconocido".

group:
  Name of the criminal organization (e.g. "CJNG", "Cártel de Sinaloa", "Los Zetas"), or "Desconocido".

crime_type:
  One of: homicidio, desaparición, extorsión, narcotráfico, enfrentamiento,
          secuestro, robo, amenaza, corrupción, disturbio, otro

  Use `disturbio` for narcobloqueos, quema de vehículos o negocios, motines,
  y disturbios públicos relacionados con grupos criminales (típicamente tras
  la captura de un líder).

confidence:
  Float 0.0–1.0 reflecting your certainty across all fields.

=== EXAMPLES ===

Input:
  Title: La Marina detiene en Nayarit al Jardinero, líder del CJNG
  Description: La Marina detiene en Nayarit al Jardinero, líder del CJNG
Output:
{"state": "Nayarit", "municipality": "Desconocido", "group": "CJNG", "crime_type": "otro", "confidence": 0.95}

Input:
  Title: Caen 37 integrantes de la Mafia Mexicana ligados al Cártel de Sinaloa tras redada en California
  Description: Caen 37 integrantes de la Mafia Mexicana ligados al Cártel de Sinaloa tras redada en California
Output:
{"state": "Internacional", "municipality": "California", "group": "Cártel de Sinaloa", "crime_type": "narcotráfico", "confidence": 0.97}

Input:
  Title: Detienen en México a narco líder del Cártel Jalisco Nueva Generación
  Description: Detienen en México a narco líder del Cártel Jalisco Nueva Generación
Output:
{"state": "Desconocido", "municipality": "Desconocido", "group": "CJNG", "crime_type": "otro", "confidence": 0.70}

Input:
  Title: Incendian autos y negocios en Nayarit tras captura de "El Jardinero", posible sucesor del "Mencho" en el CJNG
  Description: Tras la detención de "El Jardinero", presuntos integrantes del CJNG quemaron vehículos y comercios en varios municipios de Nayarit, generando narcobloqueos.
Output:
{"state": "Nayarit", "municipality": "Desconocido", "group": "CJNG", "crime_type": "disturbio", "confidence": 0.92}
"""

_FALLBACK: dict[str, Any] = {
    "state": "Desconocido",
    "municipality": "Desconocido",
    "group": "Desconocido",
    "crime_type": "otro",
    "confidence": 0.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract the first JSON object from the model response."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find a {...} block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _validate_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure all expected keys exist, filling missing ones from FALLBACK."""
    result = dict(_FALLBACK)
    for key in _FALLBACK:
        if key in data and data[key] not in (None, "", "null"):
            result[key] = data[key]
    # Clamp confidence
    try:
        result["confidence"] = float(result["confidence"])
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))
    except (TypeError, ValueError):
        result["confidence"] = 0.0
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_article(article: dict[str, Any]) -> dict[str, Any]:
    """
    Run SLM extraction on a single article when ``article["body"]`` is non-empty.

    Merges the returned structured fields into the article dict and adds
    a `processed_at` timestamp. If there is no body text, skips the SLM and
    uses fallback structured values.
    """
    body = (article.get("body") or "").strip()

    extracted: dict[str, Any] = dict(_FALLBACK)
    if body:
        user_text = (
            f"Title: {article.get('title', '')}\n"
            f"Description: {article.get('description', '')}\n"
            f"Article: {body}"
        )
        try:
            response = ollama.chat(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                options={"temperature": 0.0},
            )
            raw_text = response["message"]["content"]
            parsed = _parse_json_response(raw_text)
            extracted = _validate_fields(parsed)
        except Exception as exc:
            print(f"[extract] Warning: SLM call failed for '{article.get('title', '')}' — {exc}")
    else:
        print(f"[extract] Skipping SLM (no body): {article.get('title', '')[:60]}")

    return {
        **article,
        **extracted,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_articles(
    articles: list[dict[str, Any]],
    skip_urls: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Batch-extract structured data for a list of articles.

    `skip_urls` is a set of already-processed URLs to avoid re-running the SLM.
    """
    skip_urls = skip_urls or set()
    results: list[dict[str, Any]] = []

    for i, article in enumerate(articles, 1):
        url = article.get("url", "")
        if url in skip_urls:
            continue
        print(f"[extract] Processing {i}/{len(articles)}: {article.get('title', '')[:60]}")
        results.append(extract_article(article))

    return results


if __name__ == "__main__":
    sample = {
        "url": "https://example.com/1",
        "title": "Cártel de Sinaloa ejecuta a 3 personas en Culiacán",
        "description": "Sicarios del Cártel de Sinaloa abrieron fuego contra un grupo de personas en el centro de Culiacán, Sinaloa, dejando tres muertos.",
        "body": "En el centro de Culiacán, testigos reportaron disparos. Tres personas murieron. Las autoridades atribuyen el hecho al Cártel de Sinaloa.",
        "published_date": "2024-01-01",
        "source": "El Universal",
    }
    result = extract_article(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))
