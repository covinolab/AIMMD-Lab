"""The opt-in descriptor series of C-GAIMMD: ``GeneratorDescriptors.series``.

``descriptors_series = DESCRIPTORS.series`` gives the C-GAIMMD descriptors a
series of their own, ``'descriptors-cg'`` and 10 hex characters of the
SHA-256 of everything the rows depend on, the water placement of a cluster
included. The tests cover

- the name: stable for the same settings and structure, independent of the
  structure's path, different for other settings, for a cluster under another
  water placement (so a cache of the old placement, from before the imaging
  fix, is never resumed under the new name);
- the registration with :mod:`aimmd.core.series` (a later ``refill=False``
  never undoes a ``refill=True``), the refill callable (its ``claims``, a
  refill of a run's trajectories through ``ensure_series_coverage``, a stop),
  and the mismatch error for a params file pinned to a name of other settings;
- finding the object behind a module-level wrapper (the params-file pattern),
  for the atom-count check of the shooting worker and for the restart's
  descriptor record; bound methods are refused as params functions;
- an RFPS run continued as C-GAIMMD with a series of its own, in process: the
  trainer's start refills the series (``refill=True``), the RFPS caches stay as
  they are, no migration.
"""

import os
import sys
import shutil
from pathlib import Path

import numpy as np
import pytest

import aimmd
from aimmd.cache.npy import save_npy
from aimmd.cgaimmd.preparation import descriptors as descriptors_module
from aimmd.cgaimmd.preparation.descriptors import (
    SERIES_PREFIX, ClusterSelection, GeneratorAtoms, GeneratorDescriptors,
    GeneratorDescriptorsRefiller, find_generator_descriptors)
from aimmd.core import series as core_series
from aimmd.core.series import (RefillStopped, SeriesCoverageError, SeriesMismatchError,
                               check_series_coverage, ensure_series_coverage, series_coverage,
                               series_policy)
from aimmd.path.utils import get_cache_fname
from tests._cgaimmd_helpers import build_cgaimmd_run_dir, cgaimmd_params_namespace


TESTS = Path(__file__).resolve().parent
NANMA = TESTS / 'cgaimmd_nanma' / 'conf.gro'
NANMA_WATER = TESTS / 'cgaimmd_nanma_water' / 'conf.gro'


@pytest.fixture(autouse=True)
def _restore_sys_path(monkeypatch):
    """``Params.load`` leaves the params file's folder in ``sys.path``; take it out after the test.

    Otherwise a later ``import params`` (``tests/test_retinal.py`` imports its
    ``params.py`` by name) finds a ``params.py`` of this module's runs first.
    """
    monkeypatch.setattr(sys, 'path', list(sys.path))

_DESCRIPTORS = None
"""The module-level object the wrapper below calls (the params-file pattern)."""


def wrapped_descriptors(trajectory):
    """A params file's descriptors_function: a wrapper around a GeneratorDescriptors."""
    return _DESCRIPTORS(trajectory)


def one_atom_structure(folder, name='one_atom.gro'):
    """A one-atom structure for the frames of build_cgaimmd_run_dir."""
    fname = Path(folder) / name
    fname.write_text('one atom\n    1\n    1MOL     C1    1   0.000   0.000   0.000\n'
                     '   5.00000   5.00000   5.00000\n')
    return str(fname)


def cluster_atoms(n_waters=4, **cluster):
    return GeneratorAtoms(n_solute=22, cluster=ClusterSelection(n_waters=n_waters, **cluster))


# ---------------------------------------------------------------------------
# the name
# ---------------------------------------------------------------------------

def test_series_name_fingerprints_the_settings(tmp_path):
    vacuum = GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22))
    assert vacuum.series.startswith(SERIES_PREFIX)
    assert len(vacuum.series) == len(SERIES_PREFIX) + 10
    assert vacuum.series == SERIES_PREFIX + vacuum.fingerprint[:10]
    # the same settings and structure: the same series, wherever the structure lies
    moved = tmp_path / 'elsewhere.gro'
    shutil.copy(NANMA, moved)
    assert GeneratorDescriptors(str(moved), GeneratorAtoms(n_solute=22)).series == vacuum.series
    # other settings: another series
    names = {vacuum.series,
             GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22), whole=False).series,
             GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=21)).series,
             GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22),
                                  masses=np.ones(22)).series}
    assert len(names) == 4
    spec = vacuum.spec()
    assert spec['generator_atoms'] == {'n_solute': 22, 'cluster': None}
    assert spec['layout'] == descriptors_module.DESCRIPTOR_LAYOUT
    assert 'water_placement' not in str(spec)                   # no cluster, no placement
    spec['whole'] = 'changed'                                     # a copy
    assert vacuum.spec()['whole'] is True


def test_cluster_series_changes_with_the_water_placement(monkeypatch):
    """A cluster's series names its water placement: the rows of the placement before the
    imaging fix (version 1) can never be read under the series of the current one."""
    current = GeneratorDescriptors(str(NANMA_WATER), cluster_atoms())
    assert current.spec()['generator_atoms']['cluster']['water_placement'] == 2
    assert current.series != GeneratorDescriptors(str(NANMA_WATER), cluster_atoms(5)).series
    ranked = cluster_atoms(ranking=({'reference': 'index 0:21', 'mode': 'com', 'n': -1},))
    assert current.series != GeneratorDescriptors(str(NANMA_WATER), ranked).series
    monkeypatch.setattr(descriptors_module, 'WATER_PLACEMENT', 1)
    assert GeneratorDescriptors(str(NANMA_WATER), cluster_atoms()).series != current.series
    # the solute alone does not depend on the water placement
    vacuum = GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22)).series
    monkeypatch.setattr(descriptors_module, 'WATER_PLACEMENT', 2)
    assert GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22)).series == vacuum


# ---------------------------------------------------------------------------
# registration and the refill callable
# ---------------------------------------------------------------------------

def test_series_is_registered_and_refill_is_never_undone():
    first = GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22))
    policy = series_policy(first.series)
    assert policy.refill is False and isinstance(policy.refiller, GeneratorDescriptorsRefiller)
    refilling = GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22), refill=True)
    assert series_policy(first.series).refill is True
    # e.g. the descriptors a params file builds for its validity check, after the run's
    GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22))
    policy = series_policy(first.series)
    assert policy.refill is True and policy.refiller.descriptors is refilling


def test_refiller_claims_only_cgaimmd_series():
    refiller = GeneratorDescriptorsRefiller(
        GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22)))
    assert refiller.claims(SERIES_PREFIX + '0123456789')
    for other in ('descriptors', 'descriptors-gn0123456789', SERIES_PREFIX + '0123',
                  SERIES_PREFIX + '0123456789a', SERIES_PREFIX + 'XYZ3456789'):
        assert not refiller.claims(other), other


def _series_params(tmp_path, descriptors, series_name):
    global _DESCRIPTORS
    _DESCRIPTORS = descriptors
    return cgaimmd_params_namespace(descriptors_function=wrapped_descriptors,
                                    descriptors_series=series_name,
                                    path=tmp_path / 'params1.py', parent=tmp_path)


@pytest.fixture
def restore_module_descriptors():
    yield
    global _DESCRIPTORS
    _DESCRIPTORS = None


def test_a_run_without_the_series_is_refused_or_refilled(tmp_path, restore_module_descriptors):
    """A run whose trajectories lack the series: refused with refill=False, refilled with
    refill=True (every trajectory but the initial paths, the rows those of the descriptors)."""
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2)
    structure = one_atom_structure(tmp_path)
    descriptors = GeneratorDescriptors(structure, GeneratorAtoms(n_solute=1))
    params = _series_params(tmp_path, descriptors, descriptors.series)
    with pytest.raises(SeriesCoverageError, match=descriptors.series) as caught:
        check_series_coverage(params, layout.run)
    assert 'C-GAIMMD descriptor' in str(caught.value)
    refilling = GeneratorDescriptors(structure, GeneratorAtoms(n_solute=1), refill=True)
    params = _series_params(tmp_path, refilling, refilling.series)
    lines = []
    assert ensure_series_coverage(params, layout.run, log=lines.append, jobs=1)
    assert series_coverage(layout.run, refilling.series).complete
    for path in layout.shots + layout.free:
        rows = np.load(get_cache_fname(path.fname, refilling.series))
        np.testing.assert_allclose(rows, np.load(get_cache_fname(path.fname, 'descriptors')),
                                   atol=1e-5)
    assert any('featurized' in line for line in lines), lines
    assert all(line.startswith(core_series.LOG_PREFIX) for line in lines)


def test_refill_stops_between_trajectories(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2)
    descriptors = GeneratorDescriptors(one_atom_structure(tmp_path), GeneratorAtoms(n_solute=1))
    coverage = series_coverage(layout.run, descriptors.series)
    polls = []

    def stop():
        polls.append(1)
        return len(polls) > 1

    with pytest.raises(RefillStopped, match='after 1 of'):
        GeneratorDescriptorsRefiller(descriptors)(coverage, log=lambda line: None, stop=stop)
    assert len(series_coverage(layout.run, descriptors.series).missing) == \
        len(coverage.missing) - 1


def test_params_pinned_to_other_settings_are_refused(tmp_path, restore_module_descriptors):
    """A params file that pins the series of other settings (e.g. of the old water placement)
    stops before any work, whether or not the run holds files of that series."""
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    descriptors = GeneratorDescriptors(one_atom_structure(tmp_path), GeneratorAtoms(n_solute=1))
    pinned = SERIES_PREFIX + '0123456789'
    for path in layout.shots + layout.free:                       # rows of the pinned series
        save_npy(get_cache_fname(path.fname, pinned),
                 np.load(get_cache_fname(path.fname, 'descriptors')))
    params = _series_params(tmp_path, descriptors, pinned)
    for check in (check_series_coverage, ensure_series_coverage):
        with pytest.raises(SeriesMismatchError, match=descriptors.series):
            check(params, layout.run)


# ---------------------------------------------------------------------------
# the object behind a descriptors function
# ---------------------------------------------------------------------------

def test_find_generator_descriptors(restore_module_descriptors):
    global _DESCRIPTORS
    descriptors = GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22))
    assert find_generator_descriptors(descriptors) is descriptors
    assert find_generator_descriptors(descriptors.__call__) is descriptors
    _DESCRIPTORS = descriptors
    assert find_generator_descriptors(wrapped_descriptors) is descriptors
    assert find_generator_descriptors(None) is None
    assert find_generator_descriptors(lambda trajectory: None) is None
    assert find_generator_descriptors(np.sum) is None

    other = GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=21))

    def two(trajectory):
        return descriptors(trajectory) if trajectory else other(trajectory)

    assert find_generator_descriptors(two) is None                # ambiguous: none


def _helper_descriptors(trajectory):
    return _DESCRIPTORS(trajectory)


def _indirect_descriptors(trajectory):                    # wrapper -> helper -> DESCRIPTORS
    return _helper_descriptors(trajectory)


def _two_helpers_down(trajectory):                        # wrapper -> helper -> helper
    return _indirect_descriptors(trajectory)


def _three_helpers_down(trajectory):                      # one level more than followed
    return _two_helpers_down(trajectory)


def test_find_generator_descriptors_follows_helpers_closures_and_partials(
        restore_module_descriptors):
    """The object is found behind module-level helpers (two levels below the wrapper), a
    closure and a functools.partial; a third helper level is not followed."""
    import functools
    global _DESCRIPTORS
    descriptors = GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22))
    _DESCRIPTORS = descriptors
    assert find_generator_descriptors(_helper_descriptors) is descriptors
    assert find_generator_descriptors(_indirect_descriptors) is descriptors
    assert descriptors_module.FOLLOWED_CALL_LEVELS == 2
    assert find_generator_descriptors(_two_helpers_down) is descriptors
    assert find_generator_descriptors(_three_helpers_down) is None

    def closure(trajectory):
        return descriptors(trajectory)

    def keyword(trajectory, descriptors=None):
        return descriptors(trajectory)

    assert find_generator_descriptors(closure) is descriptors
    assert find_generator_descriptors(functools.partial(keyword, descriptors=descriptors)) \
        is descriptors
    assert find_generator_descriptors(functools.partial(_indirect_descriptors)) is descriptors


def test_atom_count_checked_through_a_wrapper(tmp_path, restore_module_descriptors):
    """The shooting worker's atom-count check finds the descriptors behind the params-file
    wrapper (the production pattern), not only a bare GeneratorDescriptors."""
    from aimmd.cgaimmd.source import check_generator_atoms
    params = _series_params(tmp_path, GeneratorDescriptors(str(NANMA),
                                                           GeneratorAtoms(n_solute=22)),
                            'descriptors')
    check_generator_atoms(params, 22)
    with pytest.raises(ValueError, match='generates 1 atoms but the descriptors function '
                                         r'\(GeneratorDescriptors\) selects 22'):
        check_generator_atoms(params, 1)
    check_generator_atoms(cgaimmd_params_namespace(), 5)          # a plain function: not checked


def test_bound_methods_are_refused_as_params_functions():
    params = aimmd.Params.placeholder.copy()
    descriptors = GeneratorDescriptors(str(NANMA), GeneratorAtoms(n_solute=22))
    with pytest.raises(TypeError, match='wrapper'):
        params.descriptors_function = descriptors.__call__


# ---------------------------------------------------------------------------
# an RFPS run continued as C-GAIMMD with a series of its own, in process
# ---------------------------------------------------------------------------

SERIES_RECIPE = '''
import numpy as np
import torch
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors

engine = 'toy'
states = 'ARB'
chain_type = 'cgaimmd'
initial_paths = ['seeds_A.xtc', 'seeds_B.xtc']
initial_network = 'pretrained.h5'
round_shooting_index = True
nbins = 3
cgaimmd_parameters = {'placement': 'none', 'lambda_clip': 10.0, 'window_after_trained_shots': 5,
                      'training_device': 'cpu', 'values_device': 'cpu'}

DESCRIPTORS = GeneratorDescriptors('one_atom.gro', GeneratorAtoms(n_solute=1), refill=True)
descriptors_series = DESCRIPTORS.series

def toy_mdrun(ts):
    ts.positions = ts.positions + 0.01

def states_function(trajectory):
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    return np.where(x <= -0.5, 'A', np.where(x >= 0.5, 'B', 'R'))

def descriptors_function(trajectory):
    return DESCRIPTORS(trajectory)

class Network(TinyFlowNetwork):
    pass

network = Network()

FIT_SERIES = []

def fit(params, pathensemble, verbose=False, worker=None):
    FIT_SERIES.append(params.descriptors_series)
    return [0.5], {'status': 'trained', 'reason': '', 'round': 1, 'record': {},
                   'network_metadata': {'round': 1, 'n_trained_shots': 6, 'fit': 'tests.fake'}}
'''


@pytest.fixture
def restore_signals():
    import signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def test_rfps_run_continued_with_a_series_of_its_own(tmp_path, restore_signals):
    """No migration: the trainer's start refills the C-GAIMMD series of the run's trajectories,
    the RFPS caches ('descriptors', another network's) stay untouched, nothing is refused."""
    import torch
    from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
    from aimmd.cgaimmd.io import file_sha256
    from tests._helpers_unit import write_trajectory
    layout = build_cgaimmd_run_dir(tmp_path / 'source')
    rfps_caches = {}
    for path in layout.shots + layout.free:                       # another network's (x, x^2)
        fname = get_cache_fname(path.fname, 'descriptors')
        x = np.load(fname)[:, 0]
        save_npy(fname, np.stack([x, x ** 2], axis=1).astype(np.float32))
        rfps_caches[fname] = file_sha256(fname)
    folder = tmp_path / 'params'
    folder.mkdir()
    one_atom_structure(folder)
    for kind, x in (('A', [-1.0, -1.1, -0.9]), ('B', [1.0, 1.1, 0.9])):
        positions = np.zeros((len(x), 1, 3), dtype=np.float32)
        positions[:, 0, 0] = x
        write_trajectory(folder, stem=f'seeds_{kind}', positions=positions)
    torch.save(TinyFlowNetwork().state_dict(), folder / 'pretrained.h5')
    (folder / 'params.py').write_text(SERIES_RECIPE)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    series_name = params.descriptors_series
    assert series_name.startswith(SERIES_PREFIX)
    aimmd.Worker(params, layout.run, walltime=120).train(nrounds=1)
    assert params.fit.__globals__['FIT_SERIES'] == [series_name]
    assert series_coverage(layout.run, series_name).complete
    for fname, sha in rfps_caches.items():
        assert file_sha256(fname) == sha                          # the RFPS history is untouched
    for path in layout.shots:
        rows = np.load(get_cache_fname(path.fname, series_name))
        assert rows.shape == (len(path), 3)
    assert not os.path.exists(os.path.join(layout.run, 'cgaimmd_descriptors.json'))
