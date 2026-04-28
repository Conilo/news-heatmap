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
You are a structured data extractor specialized in Mexican crime news.
Given a news article title and description, extract the following fields in JSON.
Respond ONLY with valid JSON — no markdown, no explanation.

Fields:
- state: Mexican state name in Spanish (e.g. "Sinaloa"), or "Desconocido" if not determinable
- municipality: City or municipality name, or "Desconocido"
- group: Name of the criminal organization involved (e.g. "Cártel de Sinaloa", "CJNG",
  "Los Zetas", "Cártel del Golfo", etc.), or "Desconocido"
- crime_type: Short label for the crime type in Spanish, one of:
  [homicidio, desaparición, extorsión, narcotráfico, enfrentamiento, secuestro,
   robo, amenaza, corrupción, otro]
- confidence: float 0.0-1.0 indicating your confidence in the extraction

Example output:
{"state": "Sinaloa", "municipality": "Culiacán", "group": "Cártel de Sinaloa",
 "crime_type": "homicidio", "confidence": 0.92}
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
    Run SLM extraction on a single article.

    Merges the returned structured fields into the article dict and adds
    a `processed_at` timestamp.  Returns the enriched article dict.
    """
    user_text = (
        f"Title: {article.get('title', '')}\n"
        f"Description: {article.get('description', '')}"
    )

    extracted: dict[str, Any] = dict(_FALLBACK)
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
        "published_date": "2024-01-01",
        "source": "El Universal",
    }
    result = extract_article(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))
