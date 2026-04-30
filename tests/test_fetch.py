"""Tests for src/fetch.py — pure-logic + opt-in live statistical test."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

import config
from src import fetch
from src.fetch import _build_query, _normalize_article, _non_article_host, _publisher_url, fetch_articles


# ---------------------------------------------------------------------------
# Group A — _build_query
# ---------------------------------------------------------------------------

def test_build_query_quotes_multi_word_terms():
    q = _build_query(["narco", "crimen organizado"], "mexico")
    assert "narco" in q
    assert '"crimen organizado"' in q
    assert "narco organizado" not in q


def test_build_query_format():
    q = _build_query(["a", "b", "c"], "mexico")
    assert re.match(r"^\(.+\) mexico$", q)
    assert "a OR b OR c" in q


def test_build_query_uses_config():
    """Module-level QUERY must be wired to config.FETCH_QUERY_TERMS / _GEO."""
    expected = _build_query(config.FETCH_QUERY_TERMS, config.FETCH_QUERY_GEO)
    assert fetch.QUERY == expected


# ---------------------------------------------------------------------------
# Group B — _normalize_article
# ---------------------------------------------------------------------------

def test_normalize_full_article():
    raw = {
        "url": "https://example.com/x",
        "title": "Hello",
        "description": "World",
        "published date": "Mon, 28 Apr 2026 12:00:00 GMT",
        "publisher": {"title": "BBC"},
    }
    assert _normalize_article(raw) == {
        "url": "https://example.com/x",
        "title": "Hello",
        "description": "World",
        "published_date": "Mon, 28 Apr 2026 12:00:00 GMT",
        "source": "BBC",
    }


def test_normalize_handles_missing_fields():
    out = _normalize_article({})
    assert out == {
        "url": "",
        "title": "",
        "description": "",
        "published_date": "",
        "source": "",
    }


def test_normalize_handles_missing_publisher():
    out = _normalize_article({"url": "u", "publisher": None})
    assert out["source"] == ""
    assert out["url"] == "u"


def test_normalize_accepts_underscore_published_date():
    out = _normalize_article({"published_date": "2026-04-28"})
    assert out["published_date"] == "2026-04-28"


# ---------------------------------------------------------------------------
# Group B2 — _publisher_url
# ---------------------------------------------------------------------------

def test_publisher_url_passes_through_non_google():
    assert _publisher_url("https://example.com/x") == "https://example.com/x"


def test_publisher_url_decodes_google_news(monkeypatch):
    def fake_gnewsdecoder(u, interval=None, proxy=None):
        assert "news.google.com" in u
        return {"status": True, "decoded_url": "https://publisher.example/a.html"}

    monkeypatch.setattr(fetch, "gnewsdecoder", fake_gnewsdecoder)
    assert (
        _publisher_url("https://news.google.com/rss/articles/ABCTOKEN?oc=5")
        == "https://publisher.example/a.html"
    )


def test_non_article_host_social_domains():
    assert _non_article_host("https://www.facebook.com/BBC/posts/x")
    assert _non_article_host("https://m.facebook.com/videos/x")
    assert not _non_article_host("https://www.elpais.com/noticia.html")


# ---------------------------------------------------------------------------
# Group C — fetch_articles with mocked GNews
# ---------------------------------------------------------------------------

def _raw(url: str, title: str = "t") -> dict:
    """Build a minimal gnews-style raw result."""
    return {
        "url": url,
        "title": title,
        "description": "d",
        "published date": "Mon, 28 Apr 2026 12:00:00 GMT",
        "publisher": {"title": "src"},
    }


def _patch_gnews(return_value=None, side_effect=None, newspaper_text="Cuerpo completo simulado para la prueba unitaria."):
    """Patch GNews client and newspaper extraction for unit tests."""
    patchers: list = []

    p = patch("src.fetch.GNews")
    mock_class = p.start()
    patchers.append(p)
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.get_news.side_effect = side_effect
    else:
        mock_client.get_news.return_value = return_value
    mock_class.return_value = mock_client

    p2 = patch("src.fetch._newspaper_body_from_url", return_value=newspaper_text)
    p2.start()
    patchers.append(p2)

    class _MultiPatcher:
        @staticmethod
        def stop() -> None:
            for x in reversed(patchers):
                x.stop()

    return _MultiPatcher()


def test_fetch_skips_newspaper_for_social_hosts(monkeypatch):
    calls: list[str] = []

    def capture(u: str) -> str:
        calls.append(u)
        return "text"

    monkeypatch.setattr(fetch, "_newspaper_body_from_url", capture)
    monkeypatch.setattr(
        fetch,
        "gnewsdecoder",
        lambda u, interval=None, proxy=None: {
            "status": True,
            "decoded_url": "https://www.facebook.com/page/videos/1",
        },
    )
    p = patch("src.fetch.GNews")
    mock_class = p.start()
    mock_client = MagicMock()
    mock_client.get_news.return_value = [_raw("https://news.google.com/rss/articles/XX")]
    mock_class.return_value = mock_client
    try:
        arts = fetch_articles(max_results=5)
    finally:
        p.stop()

    assert arts == []
    assert calls == []


def test_fetch_articles_deduplicates_by_url():
    patcher = _patch_gnews(return_value=[
        _raw("https://a.com", "first"),
        _raw("https://b.com", "second"),
        _raw("https://a.com", "first-dup"),
    ])
    try:
        arts = fetch_articles(max_results=10)
    finally:
        patcher.stop()

    urls = [a["url"] for a in arts]
    assert urls == ["https://a.com", "https://b.com"]
    assert all(a["body"] for a in arts)


def test_fetch_passes_decoded_url_to_newspaper(monkeypatch):
    """Google News wrappers must be decoded before newspaper3k fetch."""
    mock_np = MagicMock(return_value="cuerpo")
    monkeypatch.setattr(fetch, "gnewsdecoder", lambda u, interval=None, proxy=None: {
        "status": True,
        "decoded_url": "https://publisher.test/noticia",
    })
    patcher = patch("src.fetch.GNews")
    mock_class = patcher.start()
    mock_client = MagicMock()
    mock_client.get_news.return_value = [_raw("https://news.google.com/rss/articles/XXTOKEN")]
    mock_class.return_value = mock_client
    p_np = patch("src.fetch._newspaper_body_from_url", mock_np)
    p_np.start()
    try:
        arts = fetch_articles(max_results=5)
    finally:
        p_np.stop()
        patcher.stop()

    mock_np.assert_called_once_with("https://publisher.test/noticia")
    assert arts[0]["body"] == "cuerpo"


def test_fetch_articles_respects_max_results():
    patcher = _patch_gnews(return_value=[_raw(f"https://x{i}.com") for i in range(10)])
    try:
        arts = fetch_articles(max_results=3)
    finally:
        patcher.stop()

    assert len(arts) == 3


def test_fetch_articles_skips_empty_urls():
    patcher = _patch_gnews(return_value=[
        _raw(""),
        _raw("https://ok.com"),
    ])
    try:
        arts = fetch_articles(max_results=10)
    finally:
        patcher.stop()

    assert [a["url"] for a in arts] == ["https://ok.com"]
    assert arts[0]["body"]


def test_fetch_articles_returns_empty_on_exception():
    patcher = _patch_gnews(side_effect=RuntimeError("boom"))
    try:
        arts = fetch_articles(max_results=10)
    finally:
        patcher.stop()

    assert arts == []


def test_fetch_articles_body_empty_when_newspaper_returns_empty():
    patcher = patch("src.fetch.GNews")
    mock_class = patcher.start()
    mock_client = MagicMock()
    mock_client.get_news.return_value = [_raw("https://broken.com")]
    mock_class.return_value = mock_client
    p2 = patch("src.fetch._newspaper_body_from_url", return_value="")
    p2.start()
    try:
        arts = fetch_articles(max_results=10)
    finally:
        p2.stop()
        patcher.stop()

    assert arts == []


def test_fetch_continues_past_empty_body_until_enough_with_text(monkeypatch):
    """Skipped RSS rows without body do not count toward max_results."""
    seq = iter(["", "", "third body"])

    def fake_np(u: str) -> str:
        return next(seq)

    monkeypatch.setattr(fetch, "_newspaper_body_from_url", fake_np)
    p = patch("src.fetch.GNews")
    mock_class = p.start()
    mock_client = MagicMock()
    mock_client.get_news.return_value = [
        _raw("https://fail1.com"),
        _raw("https://fail2.com"),
        _raw("https://ok.com", "winner"),
    ]
    mock_class.return_value = mock_client
    try:
        arts = fetch_articles(max_results=1)
    finally:
        p.stop()

    assert len(arts) == 1
    assert arts[0]["url"] == "https://ok.com"
    assert arts[0]["body"] == "third body"


def test_gnews_client_uses_rss_cap_from_config(monkeypatch):
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.get_news.return_value = []
    mock_cls.return_value = mock_instance
    monkeypatch.setattr(fetch, "_newspaper_body_from_url", lambda u: "x")
    with patch("src.fetch.GNews", mock_cls):
        fetch_articles(max_results=2)
    mock_cls.assert_called_once()
    call_kw = mock_cls.call_args.kwargs
    assert call_kw["max_results"] == config.GNEWS_RSS_MAX_ITEMS


    patcher = _patch_gnews(return_value=None)
    try:
        arts = fetch_articles(max_results=10)
    finally:
        patcher.stop()

    assert arts == []


# ---------------------------------------------------------------------------
# Group D — Live statistical test (opt-in via --live)
# ---------------------------------------------------------------------------

# Broad superset of the old config.KEYWORDS list, plus common verb conjugations
# (detiene/detienen, asesinada, etc.) that surfaced as false negatives during
# calibration.
_CRIME_TERMS: set[str] = {
    "cartel", "cártel", "narco", "narcotráfico", "narcotrafico",
    "sicario", "crimen organizado", "crimen", "criminal", "criminales",
    "grupo delictivo", "delincuencia", "delictivo",
    "fentanilo", "droga", "drogas",
    "homicidio", "feminicidio", "asesinato", "asesinada", "asesinado",
    "ejecución", "ejecutado", "víctima", "víctimas",
    "detención", "detenido", "detenidos", "detiene", "detienen",
    "captura", "capturado", "capturan",
    "violencia", "ataque", "armado",
    # Source-level crime signal: InSight Crime is a publication that
    # exclusively covers organized crime, so a profile published there is
    # by-definition crime-relevant even when the title is just a person's name.
    "insight crime",
}

# Mexico identifiers — country/demonym terms plus all 32 state names
# (resurrected from the deleted MEXICO_STATES list in fetch.py).
_MEXICO_TERMS: set[str] = {
    "mexico", "méxico", "mexicano", "mexicana", "mexicanos", "mexicanas",
    "aguascalientes", "baja california", "baja california sur", "campeche",
    "chiapas", "chihuahua", "coahuila", "colima", "durango", "guanajuato",
    "guerrero", "hidalgo", "jalisco", "michoacán", "michoacan", "morelos",
    "nayarit", "nuevo león", "nuevo leon", "oaxaca", "puebla", "querétaro",
    "queretaro", "quintana roo", "san luis potosí", "san luis potosi",
    "sinaloa", "sonora", "tabasco", "tamaulipas", "tlaxcala", "veracruz",
    "yucatán", "yucatan", "zacatecas", "ciudad de méxico", "cdmx",
    "estado de mexico", "estado de méxico",
    # Major Mexican cities/regions that often appear without the state name
    "tapachula", "azcapotzalco", "atizapán", "ciudad obregón", "juárez",
}

# Mexican cartel aliases imply both Mexican context and crime context.
_MEXICAN_CARTEL_NAMES: set[str] = {k.lower() for k in config.GROUP_ALIASES.keys()}

# Well-known Mexican news outlets — used as a Mexico-relevance signal when an
# article body doesn't mention Mexico/state names explicitly. Lowercase
# substring match against the `source` field.
_MEXICAN_SOURCES: set[str] = {
    "milenio", "proceso", "el universal", "la jornada", "el financiero",
    "infobae", "excélsior", "excelsior", "aristegui", "sinembargo",
    "el sol de méxico", "el sol de mexico", "la silla rota", "el heraldo",
    "noroeste", "serpientesyescaleras", "plana mayor", ".mx",
}


def _haystack(article: dict) -> str:
    return " ".join([
        article.get("title", "") or "",
        article.get("description", "") or "",
        article.get("source", "") or "",
    ]).lower()


def _is_crime_relevant(article: dict) -> bool:
    text = _haystack(article)
    return (
        any(t in text for t in _CRIME_TERMS)
        or any(t in text for t in _MEXICAN_CARTEL_NAMES)
    )


def _is_mexico_relevant(article: dict) -> bool:
    text = _haystack(article)
    source = (article.get("source", "") or "").lower()
    return (
        any(t in text for t in _MEXICO_TERMS)
        or any(t in text for t in _MEXICAN_CARTEL_NAMES)
        or any(s in source for s in _MEXICAN_SOURCES)
    )


@pytest.mark.live
@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_live_results_are_relevant(monkeypatch):
    """At least 95% of fetched articles must be crime-relevant AND Mexico-relevant."""
    # Avoid N sequential newspaper downloads against live publishers during this test.
    monkeypatch.setattr(
        fetch,
        "_fetch_body",
        lambda url: "stub" if url else "",
    )
    articles = fetch_articles()
    n = len(articles)
    assert n >= 20, f"only {n} articles fetched — too few for a statistical claim"

    crime_hits = sum(1 for a in articles if _is_crime_relevant(a))
    mexico_hits = sum(1 for a in articles if _is_mexico_relevant(a))

    assert crime_hits / n >= 0.90, (
        f"crime relevance {crime_hits}/{n} = {crime_hits / n:.0%} below 90% threshold"
    )
    assert mexico_hits / n >= 0.90, (
        f"mexico relevance {mexico_hits}/{n} = {mexico_hits / n:.0%} below 90% threshold"
    )
