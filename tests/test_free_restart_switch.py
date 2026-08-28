"""
Unit tests for `params.restart_free_simulations_from`, the string switch that
selects where a free simulation's restart configuration comes from.

Background
----------
`aimmd.worker._free` historically had exactly two sources, selected by a
state-selector string called `restart_free_simulations_with_transitions`:

  ''    -> the last frame the previous free trajectory spent in the target
           state, i.e. the configuration it escaped from, which lies ON the
           state boundary;
  'A' / 'AB' / 'all'
        -> the end frames of a randomly sampled AIMMD transition path.

Boundary seeding makes every first passage start from the boundary-entry
distribution rather than from the equilibrium distribution inside the state.
`restart_free_simulations_from` generalises the choice to four named sources and
deprecates the old flag, which is still read (with a DeprecationWarning) so that
existing params files keep working unchanged.

Tests
-----
test_default_is_crossing                     — default reproduces today exactly
test_bare_mode_applies_to_every_state        — 'equilibrium' means everywhere
test_per_state_entry                         — 'A:equilibrium' is A only
test_per_state_entries_and_bare_default      — mixed spec
test_separators_and_case_are_normalised      — commas, spaces, case
test_canonical_form_is_stored                — round-trips through save/str
test_unknown_mode_raises                     — typos are caught at load
test_malformed_entry_raises                  — 'A:' , ':x' , 'A:b:c'
test_duplicate_state_raises                  — 'A:basin A:crossing'
test_two_bare_modes_raise                    — ambiguous default
test_unknown_state_letter_raises             — 'Z:basin' with states='ARB'
test_basin_on_the_reactive_state_raises      — meaningless, fail fast
test_crossing_on_the_reactive_state_is_fine  — the free-R worker still works

Deprecation of `restart_free_simulations_with_transitions`
test_deprecated_all_reads_as_transitions
test_deprecated_letters_read_as_per_state_transitions
test_deprecated_empty_is_silent
test_deprecated_emits_a_deprecation_warning
test_deprecation_message_names_the_replacement
test_setting_both_raises

Worker wiring
test_worker_maps_equilibrium_to_unbiased_weighting
test_worker_maps_basin_to_occupancy_weighting
test_worker_crossing_never_calls_the_basin_helper
"""

import warnings

import numpy as np
import pytest

import aimmd
from aimmd.params.utils import (FREE_RESTART_MODES, canonical_free_restart_from,
                                parse_free_restart_from)


def _params(**kwargs):
    p = aimmd.Params.placeholder.copy()
    p.__dict__.setdefault('states', 'ARB')
    p.__dict__.update(kwargs)
    return p


# ── the switch itself ──────────────────────────────────────────────────────

def test_default_is_crossing():
    p = _params()
    assert p.restart_free_simulations_from == 'crossing'
    for s in 'ARB':
        assert p.free_restart_mode(s) == 'crossing'


def test_bare_mode_applies_to_every_state():
    p = _params()
    p.restart_free_simulations_from = 'equilibrium'
    assert p.free_restart_mode('A') == 'equilibrium'
    assert p.free_restart_mode('B') == 'equilibrium'


def test_per_state_entry():
    p = _params()
    p.restart_free_simulations_from = 'A:equilibrium'
    assert p.free_restart_mode('A') == 'equilibrium'
    assert p.free_restart_mode('B') == 'crossing'


def test_per_state_entries_and_bare_default():
    p = _params()
    p.restart_free_simulations_from = 'transitions A:equilibrium'
    assert p.free_restart_mode('A') == 'equilibrium'
    assert p.free_restart_mode('B') == 'transitions'


def test_separators_and_case_are_normalised():
    p = _params()
    p.restart_free_simulations_from = ' a : EQUILIBRIUM ,  b:Basin '
    assert p.free_restart_mode('A') == 'equilibrium'
    assert p.free_restart_mode('B') == 'basin'


def test_canonical_form_is_stored():
    p = _params()
    p.restart_free_simulations_from = ' a:equilibrium ,b:basin'
    assert p.restart_free_simulations_from == 'A:equilibrium B:basin'
    # and the canonical form parses back to the same thing
    assert (parse_free_restart_from(p.restart_free_simulations_from)
            == parse_free_restart_from(' a:equilibrium ,b:basin'))


@pytest.mark.parametrize('bad', ['equilibirum', 'A:crosing', 'boundary'])
def test_unknown_mode_raises(bad):
    p = _params()
    with pytest.raises(TypeError) as e:
        p.restart_free_simulations_from = bad
    assert 'restart_free_simulations_from' in str(e.value)
    for mode in FREE_RESTART_MODES:
        assert mode in str(e.value), 'the error must list the valid modes'


@pytest.mark.parametrize('bad', ['A:', ':basin', 'A:basin:crossing'])
def test_malformed_entry_raises(bad):
    p = _params()
    with pytest.raises(TypeError):
        p.restart_free_simulations_from = bad


def test_duplicate_state_raises():
    p = _params()
    with pytest.raises(TypeError):
        p.restart_free_simulations_from = 'A:basin A:crossing'


def test_two_bare_modes_raise():
    p = _params()
    with pytest.raises(TypeError):
        p.restart_free_simulations_from = 'basin crossing'


def test_unknown_state_letter_raises():
    p = _params(states='ARB')
    with pytest.raises(TypeError):
        p.restart_free_simulations_from = 'Z:basin'


def test_basin_on_the_reactive_state_raises():
    """'inside the state' is the barrier region for R; there is no such draw."""
    p = _params(states='ARB')
    with pytest.raises(TypeError) as e:
        p.restart_free_simulations_from = 'R:equilibrium'
    assert 'reactive' in str(e.value).lower()


def test_crossing_on_the_reactive_state_is_fine():
    p = _params(states='ARB')
    p.restart_free_simulations_from = 'R:crossing A:equilibrium'
    assert p.free_restart_mode('R') == 'crossing'


def test_bare_mode_does_not_impose_basin_on_the_reactive_state():
    """A bare 'equilibrium' means 'wherever it is meaningful'."""
    p = _params(states='ARB')
    p.restart_free_simulations_from = 'equilibrium'
    assert p.free_restart_mode('R') == 'crossing'
    assert p.free_restart_mode('A') == 'equilibrium'


# ── deprecation ────────────────────────────────────────────────────────────

def test_deprecated_all_reads_as_transitions():
    p = _params()
    with pytest.warns(DeprecationWarning):
        p.restart_free_simulations_with_transitions = 'all'
    assert p.free_restart_mode('A') == 'transitions'
    assert p.free_restart_mode('B') == 'transitions'


def test_deprecated_letters_read_as_per_state_transitions():
    p = _params()
    with pytest.warns(DeprecationWarning):
        p.restart_free_simulations_with_transitions = 'A'
    assert p.free_restart_mode('A') == 'transitions'
    assert p.free_restart_mode('B') == 'crossing'


def test_deprecated_empty_is_silent():
    p = _params()
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        p.restart_free_simulations_with_transitions = ''
    assert p.free_restart_mode('A') == 'crossing'


def test_deprecation_message_names_the_replacement():
    p = _params()
    with pytest.warns(DeprecationWarning) as rec:
        p.restart_free_simulations_with_transitions = 'AB'
    msg = str(rec[0].message)
    assert 'restart_free_simulations_from' in msg
    assert "'AB:transitions'" in msg, 'must give the exact replacement value'


def test_setting_both_raises():
    """Two fields asking for different restart sources is a config error.

    Whether it surfaces at assignment (`_process_and_check`) or at first use
    (`free_restart_mode`) depends on how the params object was built, so accept
    either — what matters is that it is never resolved silently.
    """
    p = _params()
    p.restart_free_simulations_from = 'A:equilibrium'
    with pytest.raises(TypeError) as e:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            p.restart_free_simulations_with_transitions = 'all'
        p.free_restart_mode('A')
    assert 'restart_free_simulations_from' in str(e.value)
    assert 'restart_free_simulations_with_transitions' in str(e.value)


# ── worker wiring ──────────────────────────────────────────────────────────

def test_worker_maps_equilibrium_to_unbiased_weighting():
    from aimmd.worker._free import _basin_weighting_for_mode
    assert _basin_weighting_for_mode('equilibrium') == 'unbiased'


def test_worker_maps_basin_to_occupancy_weighting():
    from aimmd.worker._free import _basin_weighting_for_mode
    assert _basin_weighting_for_mode('basin') == 'occupancy'


def test_worker_crossing_never_calls_the_basin_helper():
    from aimmd.worker._free import _basin_weighting_for_mode
    assert _basin_weighting_for_mode('crossing') is None
    assert _basin_weighting_for_mode('transitions') is None


# ── integration: the free worker actually dispatches on the switch ─────────

def _run_free_once(monkeypatch, tmp_path, spec, states='ARB'):
    """Drive `_free` through one completed segment and record what it asked for.

    Returns the keyword arguments `_free` passed to
    `get_basin_frames_for_free_restart`, or None if it never called it.
    """
    from tests._helpers_unit import build_path
    from tests.test_worker_runtime_unit import TinyFreeWorker

    initial = build_path(
        tmp_path, stem='free_initial',
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]],
                           dtype=np.float32))

    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        states=states,
        nbins=1,
        trajectory_extension='.xtc',
        trajectory_update_batch_size=2,
        pipeline=['states', 'descriptors', 'values'],
        extra_free_frames=0,
        restart_free_simulations_with_transitions='',
        restart_free_simulations_from=spec,
        free_restart_basin_min_frames=0,
        check_if_initialized=lambda deffnm: False,
        shot_chains=lambda directory, r, old=None: [],
        free_trajectories=lambda directory, t=None: [],
        initialize_simulation=lambda frames, deffnm: None,
    )

    worker = TinyFreeWorker(params, aimmd.PathEnsemble(initial), tmp_path)
    monkeypatch.setattr(
        'aimmd.worker._free.Path',
        lambda *a, **k: initial.copy() if not a else aimmd.Path(*a, **k))

    calls = []

    def fake_basin(pool, target, reactive, **kwargs):
        calls.append(dict(target=target, reactive=reactive, **kwargs))
        return None, None            # force the documented crossing fallback

    monkeypatch.setattr('aimmd.worker._free.get_basin_frames_for_free_restart',
                        fake_basin)
    monkeypatch.setattr('aimmd.worker._free.remove', lambda *a, **k: None)

    results = iter([(None, 0, '', 0), (0, 2, 'A', 2), (None, 0, '', 0)])

    def fake_simulate(*a, **k):
        out = next(results)
        if worker.total_steps >= 1:
            worker.must_stop = True
        return out

    monkeypatch.setattr(worker, '_simulate', fake_simulate, raising=False)
    worker._free(target_state='A', k=0, total=1, wait=False)
    return calls[0] if calls else None


def test_free_worker_asks_for_unbiased_weighting_under_equilibrium(monkeypatch, tmp_path):
    call = _run_free_once(monkeypatch, tmp_path, 'A:equilibrium')
    assert call is not None, '_free must consult the in-basin helper'
    assert call['target'] == 'A' and call['reactive'] == 'R'
    assert call['weighting'] == 'unbiased'


def test_free_worker_asks_for_occupancy_weighting_under_basin(monkeypatch, tmp_path):
    call = _run_free_once(monkeypatch, tmp_path, 'A:basin')
    assert call is not None
    assert call['weighting'] == 'occupancy'


def test_free_worker_never_consults_the_helper_under_crossing(monkeypatch, tmp_path):
    """The default must not change behaviour at all."""
    assert _run_free_once(monkeypatch, tmp_path, 'crossing') is None


def test_free_worker_never_consults_the_helper_under_transitions(monkeypatch, tmp_path):
    assert _run_free_once(monkeypatch, tmp_path, 'transitions') is None


def test_free_worker_honours_the_deprecated_flag(monkeypatch, tmp_path):
    """`restart_free_simulations_with_transitions` must keep working verbatim."""
    from tests._helpers_unit import build_path
    from tests.test_worker_runtime_unit import TinyFreeWorker

    initial = build_path(
        tmp_path, stem='legacy_initial',
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]],
                           dtype=np.float32))
    chain_calls = []
    params = aimmd.Params.placeholder.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        params.__dict__.update(
            states='ARB', nbins=1, trajectory_extension='.xtc',
            trajectory_update_batch_size=2,
            pipeline=['states', 'descriptors', 'values'],
            extra_free_frames=0,
            restart_free_simulations_with_transitions='all',
            check_if_initialized=lambda deffnm: False,
            shot_chains=lambda directory, r, old=None: (chain_calls.append(r), [])[1],
            initialize_simulation=lambda frames, deffnm: None,
        )

    assert params.free_restart_mode('A') == 'transitions'

    worker = TinyFreeWorker(params, aimmd.PathEnsemble(initial), tmp_path)
    monkeypatch.setattr(
        'aimmd.worker._free.Path',
        lambda *a, **k: initial.copy() if not a else aimmd.Path(*a, **k))
    monkeypatch.setattr('aimmd.worker._free.remove', lambda *a, **k: None)
    results = iter([(None, 0, '', 0), (0, 2, 'A', 2), (None, 0, '', 0)])

    def fake_simulate(*a, **k):
        out = next(results)
        if worker.total_steps >= 1:
            worker.must_stop = True
        return out

    monkeypatch.setattr(worker, '_simulate', fake_simulate, raising=False)
    worker._free(target_state='A', k=0, total=1, wait=False)
    assert chain_calls, 'the transitions source must go looking for shot chains'
