"""Use a local Ollama SLM to extract structured data from each article."""

from __future__ import annotations

import json
import re
import sys
import os
import unicodedata
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama

import config
from advanced_config import VALID_EVENT_TYPES, VALID_STATES

_VALID_STATES_SET = frozenset(VALID_STATES)
_VALID_EVENT_TYPES_SET = frozenset(VALID_EVENT_TYPES)

# JSON schema for Ollama's constrained-decoding format parameter.
# Enum fields constrain token sampling at generation time — the model
# physically cannot produce a value outside these lists.
_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "state":        {"type": "string", "enum": VALID_STATES},
        "municipality": {"type": "string"},
        "group":        {"type": "string"},
        "event_type":   {"type": "string", "enum": VALID_EVENT_TYPES},
        "confidence":   {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["state", "municipality", "group", "event_type", "confidence"],
}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a structured data extractor specialized in Mexican crime and cartel news.
Given a news article title and the article body text, analyze the topic including
"what", "where", "when" and "who" and then extract the following fields
as valid JSON — no markdown fences, no explanation, nothing else.

=== FIELDS ===

state:
  One of the 32 Mexican state names in which the event happened:
  "Sinaloa", "Jalisco", "Tamaulipas", "Chihuahua", "Durango", "Guanajuato",
  "Guerrero", "Hidalgo", "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca",
  "Puebla", "Querétaro", "Quintana Roo", "San Luis Potosí", "Sonora", "Tabasco",
  "Tlaxcala", "Veracruz", "Yucatán", "Zacatecas", "Ciudad de México", "Estado de México",
  "Campeche", "Baja California", "Baja California Sur", "Aguascalientes", "Chiapas",
  "Colima", or:
  - "Internacional" when the event clearly happened outside Mexico (e.g. arrests or
    trials abroad).
  - "Desconocido" — no usable geographic clue in the article (even indirect).

  Infer the state from explicit place names when you can. If your best answer for state
  is "Desconocido" but you did extract a municipality, put that city/municipality in
  the municipality field.

  Do NOT infer state from the cartel's home territory alone (e.g. CJNG with no location
  stays "Desconocido", not "Jalisco") or from the article's publication place.

  For biographical profiles, retrospectives, and opinion pieces (event_type "otro"),
  use "Desconocido" for state — even when a Mexican state is mentioned in the body.
  The state field captures where a specific criminal event occurred, not where a person
  is from, generally operates, or has historical ties.

  INVALID values: "México" (the country), "EEUU", "USA", "Estados Unidos", 
  "EE.UU.", "United States". Events in the US or abroad → "Internacional".
  Do not confuse "México" (the country) with "Estado de México" (the state).

municipality:
  City or municipality name or "Desconocido". 

  INVALID values: State names

group:
  Name of the criminal organization (e.g. "CJNG", "Cártel de Sinaloa", "Los Zetas"),
  or "Desconocido".

event_type:
  One of: "homicidio", "desaparición", "extorsión", "narcotráfico", "enfrentamiento", "muerte",
          "secuestro", "robo", "incautación", "redada", "corrupción", "disturbio", "detención",
          or "otro".

  PRIORITY RULES (apply in order):

  1. "disturbio" — headline contains "incendian", "queman", "bloqueos", "narcobloqueo",
     "motín": use "disturbio" regardles of what the body says. The body may describe
     a prior arrest that triggered the disturbio — that does not change the classification.

  2. "detención" — headline contains any of: "detienen", "detuvo", "detenidos", "detención",
     "capturan", "capturó", "captura", "capturado", "arrestan", "arrestados", "arresta",
     "arrestó", "aprehenden", "vinculan a proceso". Use "detención" even when:
     - the capture is for a past crime (e.g. "Cuatro detenidos por homicidio" → "detención", NOT "homicidio")
     - the body describes drug trafficking, money laundering, or cartel operations (→ still "detención", NOT "narcotráfico")
     - the arrest occurs abroad (state="Internacional" but event_type still "detención")
     The headline arrest verb always wins over the body's description of the underlying crime.

  3. "incautación" — headline contains "decomisan", "aseguran", "incautan" and a drug/weapon
     quantity. Use "incautación" even when arrests are also mentioned.

  4. "narcotráfico" — article describes drug trafficking operations, networks, or supply
     chains; or reports a government indictment, formal accusation, sanction, or extradition
     request specifically for drug trafficking crimes (including officials or politicians
     accused of coordinating with cartels to traffic drugs).
     Use "narcotráfico" even when the accused is a public official — the trafficking charge
     is the event, not the corruption.
     Do NOT use for: diplomatic or political analysis about the *impact* of narco accusations
     on a government or politician (→ "otro"); public opinion surveys about security or drugs
     (→ "otro"); or articles that only mention cartels in passing without describing a
     specific trafficking operation or legal action.

  5. "homicidio" — headline says "asesinado", "asesinada", "ejecutado", "ejecutada",
     "balaceado", "matan", "mató", "ultimaron" or describes a deliberate killing.
     Use "homicidio" when the death is clearly intentional/violent, even if the word
     "muerto" also appears.

  6. "muerte" — headline says "fallece", "muere" or "muerto" AND the headline itself does
     NOT use a deliberate-killing verb ("asesinado", "ejecutado", "matan", "mató", etc.).
     Use "muerte" even when the body describes violence, wounds, or a shooting — only the
     headline's own wording matters here. Do NOT let body context flip "muerte" to "homicidio".

  7. "otro" — article is a biographical profile, an InSight Crime entry,
     a narcocorrido or rap/music cultural analysis, a political party statement, a
     retrospective with no specific current criminal event, a public opinion survey about
     security or drugs, or editorial/political analysis about the diplomatic impact of
     narco policy. Use "otro" even when the word "narcotráfico" appears in the headline,
     if the article does not report a specific ongoing operation or legal action.

confidence:
  Float 0.0–1.0 reflecting your certainty across all fields.

=== EXAMPLES ===

Input:
  Title: Incendian autos y negocios en Nayarit tras captura de "El Jardinero", posible sucesor del CJNG
  Article: Fuerzas de la Marina detuvieron a Audias Flores Silva en Nayarit. Tras la captura, grupos criminales quemaron vehículos y negocios en varios municipios del estado.
Output:
{"state": "Nayarit", "municipality": "Desconocido", "group": "CJNG", "event_type": "disturbio", "confidence": 0.93}

Input:
  Title: Golpe al Narco en Chiapas: Decomisan casi Una Tonelada de Cocaína y Detienen a 6
  Article: Miembros de las Fuerzas de Seguridad aseguraron cerca de una tonelada de cocaína y detuvieron a seis extranjeros en Chiapas.
Output:
{"state": "Chiapas", "municipality": "Desconocido", "group": "Desconocido", "event_type": "incautación", "confidence": 0.92}

Input:
  Title: Alejandro Treviño Morales, alias 'El Z42' - InSight Crime
  Article: Alejandro Treviño Morales era miembro de Los Zetas y hermano del exlíder del cártel. Este perfil resume su trayectoria criminal en Tamaulipas y Nuevo León.
Output:
{"state": "Desconocido", "municipality": "Desconocido", "group": "Los Zetas", "event_type": "otro", "confidence": 0.95}

Input:
  Title: Cuatro detenidos por brutal homicidio de una familia en la Ciudad de México
  Article: Autoridades de Ciudad de México confirmaron la detención de cuatro presuntos responsables del asesinato de cuatro integrantes de una familia en el norte de la capital.
Output:
{"state": "Ciudad de México", "municipality": "Desconocido", "group": "Desconocido", "event_type": "detención", "confidence": 0.91}

Input:
  Title: Escalofriante video: matan a exreina de belleza mexicana de un balazo en la cabeza
  Article: Una mujer identificada como ex reina de belleza fue asesinada a balazos en Culiacán. Las autoridades investigan presuntos vínculos con el crimen organizado.
Output:
{"state": "Sinaloa", "municipality": "Culiacán", "group": "Desconocido", "event_type": "homicidio", "confidence": 0.94}

Input:
  Title: Fiscalía de Nueva York acusa formalmente al gobernador de Sinaloa y a otros nueve funcionarios por vínculos con el narco
  Article: El gobernador de Sinaloa, Rubén Rocha Moya, fue acusado formalmente por la fiscalía federal de Nueva York de facilitar el tráfico de drogas del Cártel de Sinaloa hacia Estados Unidos junto con otros nueve funcionarios estatales.
Output:
{"state": "Sinaloa", "municipality": "Desconocido", "group": "Cártel de Sinaloa", "event_type": "narcotráfico", "confidence": 0.92}

Input:
  Title: EE.UU. y México sancionan a personas y empresas que proveen precursores químicos al Cartel de Sinaloa
  Article: El Departamento del Tesoro de EE.UU. sancionó a una red global de proveedores de precursores químicos que abastecen laboratorios de fentanilo del Cártel de Sinaloa en México.
Output:
{"state": "Internacional", "municipality": "Desconocido", "group": "Cártel de Sinaloa", "event_type": "narcotráfico", "confidence": 0.90}

Input:
  Title: Inseguridad y narco preocupan a mexicanos a 50 días del mundial
  Article: El 56% de los mexicanos se muestra preocupado por el desarrollo del torneo debido al narcotráfico, según una encuesta.
Output:
{"state": "Desconocido", "municipality": "Desconocido", "group": "Desconocido", "event_type": "otro", "confidence": 0.88}

Input:
  Title: Capturan en Culiacán a operador del Cártel de Sinaloa que coordinaba envíos de fentanilo
  Article: Elementos de la Guardia Nacional arrestaron a un hombre identificado como coordinador de rutas de tráfico de fentanilo del Cártel de Sinaloa en Culiacán. El detenido es señalado de organizar envíos de drogas hacia la frontera norte. Será puesto a disposición del Ministerio Público Federal.
Output:
{"state": "Sinaloa", "municipality": "Culiacán", "group": "Cártel de Sinaloa", "event_type": "detención", "confidence": 0.93}

Input:
  Title: Fallece presunto delincuente herido durante persecución en Tijuana, Baja California
  Article: Un hombre herido durante una persecución policiaca en Tijuana falleció en el Hospital General horas después. Las circunstancias en que resultó herido no fueron detalladas por las autoridades. No se descarta que haya recibido un impacto de bala durante el operativo.
Output:
{"state": "Baja California", "municipality": "Tijuana", "group": "Desconocido", "event_type": "muerte", "confidence": 0.87}

Input:
  Title: DEA arresta en Chicago a operador financiero del Cártel de Sinaloa
  Article: Agentes de la DEA detuvieron en Chicago a un ciudadano mexicano identificado como lavador de dinero del Cártel de Sinaloa. El detenido es acusado de mover decenas de millones de dólares de ganancias del narcotráfico a través de negocios fachada en Estados Unidos.
Output:
{"state": "Internacional", "municipality": "Desconocido", "group": "Cártel de Sinaloa", "event_type": "detención", "confidence": 0.91}
"""

_FALLBACK: dict[str, Any] = {
    "state": "Desconocido",
    "municipality": "Desconocido",
    "group": "Desconocido",
    "event_type": "otro",
    "confidence": 0.0,
}


def _geo_normalize(text: str) -> str:
    folded = unicodedata.normalize("NFD", text)
    no_marks = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    return no_marks.casefold()


_LOCATION_MATCH_ORDER: tuple[tuple[str, str], ...] = tuple(
    sorted(config.LOCATION_TO_STATE.items(), key=lambda kv: (-len(kv[0]), kv[0]))
)
_STATE_MATCH_ORDER: tuple[tuple[str, str], ...] = tuple(
    sorted(config.STATE_NAME_MAP.items(), key=lambda kv: (-len(kv[0]), kv[0]))
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_state_from_municipality(municipality: str) -> str | None:
    """
    Map an SLM ``municipality`` string to estado using ``LOCATION_TO_STATE`` and
    ``STATE_NAME_MAP`` (accent-folded; substring match inside the municipality only).
    """
    raw = (municipality or "").strip()
    municipality_normalized = _geo_normalize(raw)
    if not raw or municipality_normalized == _geo_normalize("Desconocido"):
        return None

    if municipality_normalized in config.LOCATION_TO_STATE:
        return config.LOCATION_TO_STATE[municipality_normalized]

    for loc_key, est in _LOCATION_MATCH_ORDER:
        if loc_key in municipality_normalized:
            return est

    for phrase, est in _STATE_MATCH_ORDER:
        pn = _geo_normalize(phrase)
        if pn in municipality_normalized:
            return est

    return None


def _maybe_fill_state_from_municipality(extracted: dict[str, Any]) -> None:
    """If state is still Desconocido, derive it only from ``municipality`` when lookup hits."""
    if extracted.get("state") != "Desconocido":
        return
    mun = extracted.get("municipality")
    inferred = _infer_state_from_municipality("" if mun is None else str(mun))
    if inferred:
        extracted["state"] = inferred


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
    """Ensure all expected keys exist, filling missing ones from FALLBACK.

    Also guards enum fields against out-of-vocabulary values that can slip
    through even with constrained decoding (e.g. accent/case mismatches).
    """
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
    # Reject any value outside the allowed enums
    if result["event_type"] not in _VALID_EVENT_TYPES_SET:
        print(f"[extract] Invalid event_type {result['event_type']!r} — falling back to 'otro'")
        result["event_type"] = "otro"
    if result["state"] not in _VALID_STATES_SET:
        print(f"[extract] Invalid state {result['state']!r} — falling back to 'Desconocido'")
        result["state"] = "Desconocido"
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
        body_truncated = body[:config.ARTICLE_BODY_MAX_CHARS_SLM]
        user_text = (
            f"Title: {article.get('title', '')}\n"
            f"Article: {body_truncated}"
        )
        try:
            response = ollama.chat(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                format=_OUTPUT_SCHEMA,
                options={"temperature": 0.0},
            )
            raw_text = response["message"]["content"]
            parsed = _parse_json_response(raw_text)
            extracted = _validate_fields(parsed)
        except Exception as exc:
            print(f"[extract] Warning: SLM call failed for '{article.get('title', '')}' — {exc}")
    else:
        print(f"[extract] Skipping SLM (no body): {article.get('title', '')[:60]}")

    _maybe_fill_state_from_municipality(extracted)

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
