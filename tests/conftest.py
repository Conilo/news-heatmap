"""Pytest configuration: opt-in `--live` flag for tests that hit the network."""

from __future__ import annotations

import os
import sys

import pytest

# Make `src.fetch` and `config` importable from the project root regardless of
# where pytest is invoked from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run tests that hit the live Google News API",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "live: hits the live Google News API (skipped unless --live)"
    )
    config.addinivalue_line(
        "markers", "flaky: retry on failure (handled by pytest-rerunfailures)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="needs --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
