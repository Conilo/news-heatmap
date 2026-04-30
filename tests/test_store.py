"""Tests for src/store.py — CSV-based persistence layer."""

from __future__ import annotations

import os
import shutil

import pandas as pd
import pytest

import config
from src import store


# ---------------------------------------------------------------------------
# Fixture — redirect all data paths into a per-test tmp directory
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Point config.DATA_DIR / ARTICLES_CSV / EVENTS_CSV at tmp_path."""
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ARTICLES_CSV", str(tmp_path / "articles.csv"))
    monkeypatch.setattr(config, "EVENTS_CSV", str(tmp_path / "events.csv"))
    return tmp_path


def _full_article(url: str, **overrides) -> dict:
    """Build an article dict populated for every column in config.CSV_COLUMNS."""
    base = {
        "url": url,
        "title": "t",
        "description": "d",
        "body": "article body text",
        "published_date": "Mon, 28 Apr 2026 12:00:00 GMT",
        "source": "src",
        "state": "Sinaloa",
        "municipality": "Culiacán",
        "group": "CDS",
        "event_type": "homicidio",
        "confidence": "0.9",
        "processed_at": "2026-04-28T12:00:00",
        "event_id": "evt-1",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Group A — load()
# ---------------------------------------------------------------------------

def test_load_returns_empty_df_when_file_missing():
    df = store.load()
    assert df.empty
    assert list(df.columns) == config.CSV_COLUMNS


def test_load_round_trips_written_csv():
    rows = [_full_article("https://a.com"), _full_article("https://b.com", title="t2")]
    store.save(pd.DataFrame(rows))

    df = store.load()
    assert list(df.columns) == config.CSV_COLUMNS
    assert df["url"].tolist() == ["https://a.com", "https://b.com"]
    assert df.loc[1, "title"] == "t2"


def test_load_backfills_missing_columns_for_older_schemas():
    # Simulate an older CSV missing some of today's columns.
    legacy = pd.DataFrame(
        [{"url": "https://a.com", "title": "t", "source": "src"}]
    )
    legacy.to_csv(config.ARTICLES_CSV, index=False)

    df = store.load()
    assert list(df.columns) == config.CSV_COLUMNS
    for col in config.CSV_COLUMNS:
        if col not in {"url", "title", "source"}:
            assert df.loc[0, col] == ""


def test_load_keeps_empty_cells_as_empty_string_not_nan():
    """Guards the keep_default_na=False invariant called out in store.py."""
    rows = [_full_article("https://a.com", municipality="", event_id="")]
    store.save(pd.DataFrame(rows))

    df = store.load()
    assert df.loc[0, "municipality"] == ""
    assert df.loc[0, "event_id"] == ""
    assert not df.isna().any().any()


def test_load_returns_empty_and_warns_on_corrupt_csv(capsys):
    with open(config.ARTICLES_CSV, "wb") as f:
        f.write(b"\x00\x01\x02 not,a,valid\ncsv\xff\xfe")

    # pandas can sometimes still parse garbage; force a real failure by making
    # the path a directory instead of a file.
    os.remove(config.ARTICLES_CSV)
    os.makedirs(config.ARTICLES_CSV, exist_ok=True)

    df = store.load()
    out = capsys.readouterr().out
    assert df.empty
    assert list(df.columns) == config.CSV_COLUMNS
    assert "[store] Warning" in out


# ---------------------------------------------------------------------------
# Group B — save()
# ---------------------------------------------------------------------------

def test_save_creates_data_dir_when_missing(tmp_path):
    shutil.rmtree(tmp_path)
    assert not os.path.exists(tmp_path)

    store.save(pd.DataFrame([_full_article("https://a.com")]))

    assert os.path.isdir(tmp_path)
    assert os.path.exists(config.ARTICLES_CSV)


def test_save_writes_only_csv_columns_dropping_extras():
    row = _full_article("https://a.com")
    row["extra_col"] = "should not appear"
    store.save(pd.DataFrame([row]))

    raw = pd.read_csv(config.ARTICLES_CSV, dtype=str, keep_default_na=False)
    assert list(raw.columns) == config.CSV_COLUMNS
    assert "extra_col" not in raw.columns


# ---------------------------------------------------------------------------
# Group C — append_new()
# ---------------------------------------------------------------------------

def test_append_new_writes_all_rows_when_store_empty():
    arts = [_full_article("https://a.com"), _full_article("https://b.com")]
    result = store.append_new(arts)

    assert result["url"].tolist() == ["https://a.com", "https://b.com"]
    assert store.load()["url"].tolist() == ["https://a.com", "https://b.com"]


def test_append_new_dedupes_against_existing_urls():
    store.save(pd.DataFrame([_full_article("https://a.com", title="original")]))

    result = store.append_new([
        _full_article("https://a.com", title="dup-attempt"),
        _full_article("https://b.com", title="new"),
    ])

    assert result["url"].tolist() == ["https://a.com", "https://b.com"]
    # Existing row preserved, not overwritten.
    assert result.loc[result["url"] == "https://a.com", "title"].iloc[0] == "original"


def test_append_new_is_noop_when_all_urls_already_known(capsys):
    store.save(pd.DataFrame([
        _full_article("https://a.com"),
        _full_article("https://b.com"),
    ]))

    before = store.load()
    result = store.append_new([_full_article("https://a.com", title="dup")])
    after = store.load()
    out = capsys.readouterr().out

    assert "No new articles to add." in out
    pd.testing.assert_frame_equal(before, result.reset_index(drop=True))
    pd.testing.assert_frame_equal(before, after)


def test_append_new_aligns_missing_columns_to_empty_string():
    store.append_new([{"url": "https://a.com", "title": "sparse"}])

    df = store.load()
    assert list(df.columns) == config.CSV_COLUMNS
    assert df.loc[0, "title"] == "sparse"
    assert df.loc[0, "state"] == ""
    assert df.loc[0, "event_id"] == ""


def test_append_new_dedupes_within_input_batch():
    """Safety-net drop_duplicates inside append_new should collapse intra-batch dups."""
    result = store.append_new([
        _full_article("https://a.com", title="first"),
        _full_article("https://a.com", title="dup"),
        _full_article("https://b.com"),
    ])

    assert result["url"].tolist() == ["https://a.com", "https://b.com"]
    assert result.loc[result["url"] == "https://a.com", "title"].iloc[0] == "first"


# ---------------------------------------------------------------------------
# Group D — get_processed_urls()
# ---------------------------------------------------------------------------

def test_get_processed_urls_empty_when_csv_missing():
    assert store.get_processed_urls() == set()


def test_get_processed_urls_returns_url_set():
    store.save(pd.DataFrame([
        _full_article("https://a.com"),
        _full_article("https://b.com"),
        _full_article("https://c.com"),
    ]))

    assert store.get_processed_urls() == {
        "https://a.com",
        "https://b.com",
        "https://c.com",
    }


# ---------------------------------------------------------------------------
# Group E — load_events() / save_events()
# ---------------------------------------------------------------------------

def _full_event(event_id: str, **overrides) -> dict:
    base = {
        "event_id": event_id,
        "state": "Sinaloa",
        "municipality": "Culiacán",
        "group": "CDS",
        "event_type": "homicidio",
        "first_seen": "2026-04-25",
        "last_seen": "2026-04-28",
        "article_count": "3",
        "unique_sources": "2",
        "confidence": "0.9",
        "canonical_title": "title",
    }
    base.update(overrides)
    return base


def test_load_events_returns_empty_df_when_file_missing():
    df = store.load_events()
    assert df.empty
    assert list(df.columns) == config.EVENTS_CSV_COLUMNS


def test_save_events_round_trips_through_load_events():
    events = [_full_event("evt-1"), _full_event("evt-2", state="Jalisco")]
    store.save_events(pd.DataFrame(events))

    df = store.load_events()
    assert list(df.columns) == config.EVENTS_CSV_COLUMNS
    assert df["event_id"].tolist() == ["evt-1", "evt-2"]
    assert df.loc[1, "state"] == "Jalisco"


def test_load_events_backfills_missing_columns_for_older_schemas():
    legacy = pd.DataFrame([{"event_id": "evt-1", "state": "Sinaloa"}])
    legacy.to_csv(config.EVENTS_CSV, index=False)

    df = store.load_events()
    assert list(df.columns) == config.EVENTS_CSV_COLUMNS
    for col in config.EVENTS_CSV_COLUMNS:
        if col not in {"event_id", "state"}:
            assert df.loc[0, col] == ""


# ---------------------------------------------------------------------------
# Group F — update_rows()
# ---------------------------------------------------------------------------

def test_update_rows_empty_input_is_noop():
    store.save(pd.DataFrame([_full_article("https://a.com", title="orig")]))
    before = store.load()

    result = store.update_rows([])

    pd.testing.assert_frame_equal(before, result)
    pd.testing.assert_frame_equal(before, store.load())


def test_update_rows_overwrites_only_existing_urls():
    """df.update inserts nothing for unknown URLs — only matching rows change."""
    store.save(pd.DataFrame([
        _full_article("https://a.com", title="orig-a", state="Sinaloa"),
        _full_article("https://b.com", title="orig-b", state="Jalisco"),
    ]))

    result = store.update_rows([
        {"url": "https://a.com", "title": "new-a", "state": "Sonora"},
        {"url": "https://nope.com", "title": "ghost"},
    ])

    assert result["url"].tolist() == ["https://a.com", "https://b.com"]
    assert "https://nope.com" not in result["url"].tolist()
    assert result.loc[result["url"] == "https://a.com", "title"].iloc[0] == "new-a"
    assert result.loc[result["url"] == "https://a.com", "state"].iloc[0] == "Sonora"


def test_update_rows_leaves_untouched_rows_unchanged():
    store.save(pd.DataFrame([
        _full_article("https://a.com", title="orig-a"),
        _full_article("https://b.com", title="orig-b"),
    ]))

    store.update_rows([{"url": "https://a.com", "title": "new-a"}])

    df = store.load()
    assert df.loc[df["url"] == "https://b.com", "title"].iloc[0] == "orig-b"


def test_update_rows_aligns_sparse_dicts_to_full_schema():
    """Sparse update dicts must not blow up the column-alignment step."""
    store.save(pd.DataFrame([_full_article("https://a.com", state="Sinaloa")]))

    # df.update only overwrites with non-NaN values, so empty-string fills from
    # the alignment step shouldn't clobber existing populated fields. We only
    # assert the call succeeds and the targeted field changes.
    result = store.update_rows([{"url": "https://a.com", "title": "new-title"}])

    assert list(result.columns) == config.CSV_COLUMNS
    assert result.loc[0, "title"] == "new-title"
