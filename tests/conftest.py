"""Pytest configuration for optional AIMMD test groups.

The repository already had a `--runslow` opt-in for long integration tests.
This file now also exposes a separate `--rungraph` opt-in for the graph/GNN
utility tests, because those depend on a larger external stack and exercise
functionality that is not part of the default unit-test surface.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--rungraph", action="store_true", default=False, help="run optional graph utility tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "graph: mark optional graph utility tests")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_graph = pytest.mark.skip(reason="need --rungraph option to run")
    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        if "graph" in item.keywords and not config.getoption("--rungraph"):
            item.add_marker(skip_graph)
