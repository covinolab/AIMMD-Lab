"""Pytest configuration: test groups, the markers that select them, shared fixtures.

Markers that need an opt-in flag (the test is skipped unless the flag is given):

- ``slow`` (``--runslow``): long end-to-end runs, e.g. GROMACS campaigns.
- ``graph`` (``--rungraph``): the graph/GNN utilities. They need
  ``torch_geometric`` and ``torch_cluster``, a larger external stack that is
  not part of the default unit-test surface.
- ``bioemu2_weights`` (``--runweights``): the real BioEmu2 checkpoint from the
  Hugging Face cache (several GB, usually a GPU). Never run in GitHub CI.

Markers that skip themselves when what they need is missing (no flag):

- ``gromacs``: the test starts ``gmx``. Skipped when neither ``gmx`` nor
  ``gmx_mpi`` is on ``PATH``. Take the executable from the ``gmx_executable``
  fixture rather than looking it up again in the test module.
- ``bioemu2``: the test needs the ``bioemu2`` package (it uses a tiny
  random-weight model on the CPU, no download). Skipped when the package is
  not installed.
- ``gpu``: the test needs CUDA. Skipped when ``torch.cuda.is_available()`` is
  False.

A marker can be combined with others: a long GROMACS run is ``slow`` and
``gromacs``, and runs only with ``--runslow`` on a machine with GROMACS.

Every test also has a time limit from pytest-timeout (``timeout`` under
``[tool.pytest.ini_options]`` in ``pyproject.toml``), so that a hung test
fails instead of holding a CI runner for hours.
"""

import importlib.util
import shutil

import pytest

#: marker name -> what it means (registered in `pytest_configure`, so that
#: ``--strict-markers`` accepts them and ``pytest --markers`` lists them)
MARKERS = {
    'slow': 'long end-to-end run; skipped unless --runslow',
    'graph': 'needs torch_geometric and torch_cluster; skipped unless --rungraph',
    'gromacs': 'starts gmx; skipped when no gmx or gmx_mpi is on PATH '
               '(use the gmx_executable fixture)',
    'bioemu2': 'needs the bioemu2 package (tiny random-weight model on the CPU); '
               'skipped when bioemu2 is not installed',
    'bioemu2_weights': 'needs the real BioEmu2 checkpoint in the Hugging Face '
                       'cache; skipped unless --runweights',
    'gpu': 'needs CUDA; skipped when torch.cuda.is_available() is False',
}

#: markers that run only when their command-line flag is given
OPT_IN_FLAGS = {
    'slow': '--runslow',
    'graph': '--rungraph',
    'bioemu2_weights': '--runweights',
}


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--rungraph", action="store_true", default=False, help="run optional graph utility tests"
    )
    parser.addoption(
        "--runweights", action="store_true", default=False,
        help="run the tests that need the real BioEmu2 checkpoint (marker bioemu2_weights)"
    )


def pytest_configure(config):
    for name, meaning in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {meaning}")


def find_gromacs():
    """The GROMACS executable on PATH (``gmx``, else ``gmx_mpi``), or None.

    The same lookup as `aimmd._init._resolve_gromacs`, without importing aimmd.
    """
    return shutil.which('gmx') or shutil.which('gmx_mpi')


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _missing_requirements(items):
    """Skip reason for each self-skipping marker whose requirement is missing.

    Only the markers that some collected test carries are checked, so a run
    without ``gpu`` tests never imports torch here.
    """
    used = {name for item in items for name in ('gromacs', 'bioemu2', 'gpu')
            if item.get_closest_marker(name) is not None}
    reasons = {}
    if 'gromacs' in used and find_gromacs() is None:
        reasons['gromacs'] = 'GROMACS not found (no gmx or gmx_mpi on PATH)'
    if 'bioemu2' in used and importlib.util.find_spec('bioemu2') is None:
        reasons['bioemu2'] = 'bioemu2 is not installed'
    if 'gpu' in used and not _cuda_available():
        reasons['gpu'] = 'CUDA is not available'
    return reasons


def pytest_collection_modifyitems(config, items):
    reasons = {name: f"need {flag} option to run"
               for name, flag in OPT_IN_FLAGS.items() if not config.getoption(flag)}
    reasons.update(_missing_requirements(items))
    for item in items:
        for name, reason in reasons.items():
            if item.get_closest_marker(name) is not None:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def gmx_executable():
    """Path of the GROMACS executable for tests that start ``gmx``.

    It is the executable AIMMD resolved at import (``aimmd._config.GROMACS``),
    else the one on PATH; the test is skipped when there is none. Tests that
    take this fixture should also carry the ``gromacs`` marker.
    """
    import aimmd

    gmx = getattr(aimmd._config, 'GROMACS', None) or find_gromacs()
    if gmx is None:
        pytest.skip('GROMACS not found (no gmx or gmx_mpi on PATH)')
    return gmx


@pytest.fixture(autouse=True)
def _reset_graph_cache_reader_role():
    """Keep the trainer's reader-role flag from leaking between tests.

    ``shm_cache.set_reader_role()`` is process-global and sticky by design (a
    process is a trainer or it is not). Under pytest, though, the multi-system
    trainer tests set it and every later test inherits it -- which would quietly
    turn any subsequent `store_*_in_sqlite` into a no-op and make results depend
    on collection order. shm_cache is stdlib-only, so importing it here is free.
    """
    try:
        from aimmd.network import shm_cache
    except Exception:                                           # noqa: BLE001
        yield
        return
    shm_cache._READER_ROLE = False
    yield
    shm_cache._READER_ROLE = False


@pytest.fixture(autouse=True)
def _restore_series_registry():
    """Keep the descriptor-series registry from leaking between tests.

    Every node-table featurizer registers its series in the process-global
    registry of `aimmd.core.series` when it is built, as a params file does
    in a job. Under pytest the featurizers of one test would stay registered
    for every later test, and a params whose computed series is not
    registered while another one is counts as written for other settings
    (`SeriesMismatchError`): restore the registry after each test.
    """
    try:
        from aimmd.core import series
    except Exception:                                           # noqa: BLE001
        yield
        return
    saved = dict(series._POLICIES)
    yield
    series._POLICIES.clear()
    series._POLICIES.update(saved)
