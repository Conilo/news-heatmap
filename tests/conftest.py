"""Pytest configuration: opt-in flags for network and local Ollama tests."""

from __future__ import annotations

import os
import sys

import pytest

# Make `src.fetch` and `config` importable from the project root regardless of
# where pytest is invoked from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Match `streamlit run src/dashboard.py`: `src/*.py` use `from store import …`
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run tests that hit the live Google News API",
    )
    parser.addoption(
        "--slm-live",
        action="store_true",
        default=False,
        help="run tests that call the local Ollama extractor (slow)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "live: hits the live Google News API (skipped unless --live)"
    )
    config.addinivalue_line(
        "markers", "flaky: retry on failure (handled by pytest-rerunfailures)"
    )
    config.addinivalue_line(
        "markers",
        "slm_live: requires local Ollama and config.MODEL_NAME (skipped unless --slm-live)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skip_live = pytest.mark.skip(reason="needs --live")
    if not config.getoption("--live"):
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)

    skip_slm_live = pytest.mark.skip(reason="needs --slm-live")
    if not config.getoption("--slm-live"):
        for item in items:
            if "slm_live" in item.keywords:
                item.add_marker(skip_slm_live)
