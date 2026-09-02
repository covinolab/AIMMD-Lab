"""
Unit tests for the two free-simulation seeding params.

`params.free_seeding_position` says where inside the state the FIRST free
trajectory of that state starts, picked out of the initial path by fractional
position over the state's own run of frames (0.0 = furthest from the reactive
region, 1.0 = adjacent to it, which is the historical behaviour and the
default).

`params.free_restart_source` says where every LATER free trajectory of that
state restarts from ('crossing' = the frame the previous one escaped from,
which is the historical behaviour and the default; 'seed', 'basin',
'equilibrium', 'transitions').

The two are independent and both are settable per state.

Tests
-----
Default fidelity (the property everything else must not break)
test_defaults_are_boundary_and_crossing
test_boundary_reproduces_the_historical_couples_for_a_leading_state
test_boundary_reproduces_the_historical_couples_for_a_trailing_state
test_boundary_never_touches_the_untrimmed_path

Position grammar
test_position_aliases
test_position_accepts_floats_and_float_strings
test_position_accepts_random
test_position_per_state_dict
test_position_dict_leaves_unnamed_states_on_the_default
test_position_unknown_name_raises
test_position_out_of_range_raises
test_position_unknown_state_letter_raises
test_position_for_the_reactive_state_is_ignored
test_position_canonical_form_round_trips

Index rule over the state's run
test_index_rule_named_values
test_index_rule_fraction
test_index_rule_single_frame_run
test_index_rule_two_frame_run_ties_to_the_boundary
test_index_rule_is_mirrored_for_a_trailing_state

Frame selection on real paths
test_deepest_picks_the_first_frame_of_a_leading_run
test_deepest_picks_the_last_frame_of_a_trailing_run
test_middle_picks_the_middle_of_the_run
test_the_run_is_the_one_containing_the_transition_boundary
test_couple_is_always_two_frames_with_the_prefix_on_the_boundary_side
test_random_is_reproducible_for_a_given_worker

Restart sources
test_restart_source_default_and_bare
test_restart_source_per_state_dict
test_restart_source_unknown_raises
test_in_state_sources_degrade_on_the_reactive_state
test_restart_source_canonical_form_round_trips

Legacy
test_deprecated_transitions_flag_maps_to_transitions
test_deprecated_transitions_flag_all
test_deprecated_transitions_flag_empty_is_silent
test_deprecated_transitions_flag_warns
test_deprecated_flag_plus_new_field_raises
test_round_one_field_is_gone
"""

import warnings

import numpy as np
import pytest

import aimmd
from aimmd.params.utils import (
    FREE_RESTART_SOURCES,
    FREE_RESTART_IN_STATE_SOURCES,
    SEEDING_POSITION_ALIASES,
    canonical_restart_source,
    canonical_seeding_position,
    legacy_transitions_replacement,
    parse_restart_source,
    parse_seeding_position,
)
from aimmd.worker.utils import (
    get_initial_frames_for_free_simulations,
    seed_index_in_run,
    state_run_locs,
)

from ._helpers_unit import build_path


def _params(**kwargs):
    p = aimmd.Params.placeholder.copy()
    p.__dict__.setdefault('states', 'ARB')
    p.__dict__.update(kwargs)
    return p


def _path(tmp_path, state_string, stem='traj'):
    """A synthetic Path whose cached states spell `state_string`."""
    n = len(state_string)
    positions = np.zeros((n, 2, 3), dtype=np.float32)
    positions[:, 0, 0] = np.arange(n, dtype=np.float32)
    positions[:, 1, 0] = np.arange(n, dtype=np.float32) + 1.0
    return build_path(tmp_path, stem=stem, positions=positions,
                      states=list(state_string))


def _historical_couple(path, target_state):
    """The couple the pre-fix implementation built, transcribed verbatim."""
    if path.initial('states') == target_state:
        return path[1::-1]
    return path[-2:]


# ── default fidelity ───────────────────────────────────────────────────────

def test_defaults_are_boundary_and_crossing():
    p = _params()
    assert p.free_seeding_position == 'boundary'
    assert p.free_restart_source == 'crossing'
    assert p.free_seeding_position_for('A') == 1.0
    assert p.free_seeding_position_for('B') == 1.0
    assert p.free_restart_source_for('A') == 'crossing'
    assert p.free_restart_source_for('B') == 'crossing'


def test_boundary_reproduces_the_historical_couples_for_a_leading_state(tmp_path):
    path = _path(tmp_path, 'ARRRRB')      # a trimmed transition block
    got = get_initial_frames_for_free_simulations([path], 'A', 'R')
    want = _historical_couple(path, 'A')
    assert len(got) == 1
    assert list(got[0].locs) == list(want.locs)
    assert list(got[0].states) == list(want.states) == ['R', 'A']


def test_boundary_reproduces_the_historical_couples_for_a_trailing_state(tmp_path):
    path = _path(tmp_path, 'ARRRRB')      # a trimmed transition block
    got = get_initial_frames_for_free_simulations([path], 'B', 'R')
    want = _historical_couple(path, 'B')
    assert list(got[0].locs) == list(want.locs)
    assert list(got[0].states) == list(want.states) == ['R', 'B']


def test_boundary_never_touches_the_untrimmed_path(tmp_path):
    """position 1.0 must take the historical code path and ignore the extras."""
    path = _path(tmp_path, 'ARRRRB')
    sentinel = object()          # would explode if it were indexed
    got = get_initial_frames_for_free_simulations(
        [path], 'A', 'R', position=1.0, untrimmed_paths=[sentinel])
    assert list(got[0].locs) == [1, 0]


# ── position grammar ──────────────────────────────────────────────────────

def test_position_aliases():
    assert SEEDING_POSITION_ALIASES == {'boundary': 1.0, 'middle': 0.5,
                                        'deepest': 0.0}
    for name, value in SEEDING_POSITION_ALIASES.items():
        default, per_state = parse_seeding_position(name, states='ARB')
        assert default == value and per_state == {}


def test_position_accepts_floats_and_float_strings():
    assert parse_seeding_position(0.25, states='ARB')[0] == 0.25
    assert parse_seeding_position('0.25', states='ARB')[0] == 0.25
    assert parse_seeding_position(0, states='ARB')[0] == 0.0
    assert parse_seeding_position(1, states='ARB')[0] == 1.0


def test_position_accepts_random():
    assert parse_seeding_position('random', states='ARB')[0] == 'random'
    assert parse_seeding_position('RANDOM', states='ARB')[0] == 'random'


def test_position_per_state_dict():
    default, per_state = parse_seeding_position(
        {'A': 'deepest', 'B': 0.5}, states='ARB')
    assert default == 1.0
    assert per_state == {'A': 0.0, 'B': 0.5}


def test_position_dict_leaves_unnamed_states_on_the_default():
    p = _params(free_seeding_position={'A': 'deepest'})
    assert p.free_seeding_position_for('A') == 0.0
    assert p.free_seeding_position_for('B') == 1.0


def test_position_unknown_name_raises():
    with pytest.raises(TypeError, match='free_seeding_position'):
        parse_seeding_position('shallowest', states='ARB')


def test_position_out_of_range_raises():
    for bad in (-0.1, 1.1, 2, '-1'):
        with pytest.raises(TypeError):
            parse_seeding_position(bad, states='ARB')


def test_position_unknown_state_letter_raises():
    with pytest.raises(TypeError, match='Z'):
        parse_seeding_position({'Z': 'deepest'}, states='ARB')


def test_position_for_the_reactive_state_is_ignored():
    p = _params(free_seeding_position={'R': 'deepest'})
    assert p.free_seeding_position_for('R') == 1.0


def test_position_canonical_form_round_trips():
    for value in ('boundary', 'middle', 'deepest', 'random', 0.25,
                  {'A': 'deepest'}, {'A': 'deepest', 'B': 0.5}):
        once = canonical_seeding_position(value, states='ARB')
        assert canonical_seeding_position(once, states='ARB') == once


# ── the index rule ────────────────────────────────────────────────────────

def test_index_rule_named_values():
    n = 5
    assert seed_index_in_run(n, 1.0) == 4
    assert seed_index_in_run(n, 0.5) == 2
    assert seed_index_in_run(n, 0.0) == 0


def test_index_rule_fraction():
    assert seed_index_in_run(5, 0.25) == 1
    assert seed_index_in_run(9, 0.25) == 2


def test_index_rule_single_frame_run():
    for p in (0.0, 0.5, 1.0):
        assert seed_index_in_run(1, p) == 0


def test_index_rule_two_frame_run_ties_to_the_boundary():
    """round(0.5) == 0 in Python; the rule must not inherit that."""
    assert seed_index_in_run(2, 0.5) == 1
    assert seed_index_in_run(2, 0.0) == 0
    assert seed_index_in_run(2, 1.0) == 1


def test_index_rule_is_mirrored_for_a_trailing_state():
    """The run is ordered far-side-first, so the rule itself never mirrors."""
    states = np.array(list('AAARRRRBB'), dtype='<U1')
    leading = state_run_locs(states, boundary_loc=2, at_start=True)
    trailing = state_run_locs(states, boundary_loc=7, at_start=False)
    assert leading == [0, 1, 2]
    assert trailing == [8, 7]


# ── frame selection ───────────────────────────────────────────────────────

def test_deepest_picks_the_first_frame_of_a_leading_run(tmp_path):
    path = _path(tmp_path, 'AAARRRRBB')
    trimmed = path[2:]
    got = get_initial_frames_for_free_simulations(
        [trimmed], 'A', 'R', position=0.0, untrimmed_paths=[path])
    assert list(got[0].locs) == [1, 0]
    assert list(got[0].states) == ['A', 'A']


def test_deepest_picks_the_last_frame_of_a_trailing_run(tmp_path):
    path = _path(tmp_path, 'AAARRRRBB')
    trimmed = path[:8]
    got = get_initial_frames_for_free_simulations(
        [trimmed], 'B', 'R', position=0.0, untrimmed_paths=[path])
    assert list(got[0].locs) == [7, 8]
    assert list(got[0].states) == ['B', 'B']


def test_middle_picks_the_middle_of_the_run(tmp_path):
    path = _path(tmp_path, 'AAAAARRRRBBBB')
    trimmed = path[4:]
    got = get_initial_frames_for_free_simulations(
        [trimmed], 'A', 'R', position=0.5, untrimmed_paths=[path])
    assert got[0].locs[-1] == 2


def test_the_run_is_the_one_containing_the_transition_boundary(tmp_path):
    """A leading dip out of the state must not be swept into the run."""
    path = _path(tmp_path, 'ARAAARRRB')
    trimmed = path[4:]           # transition block starts at the last A
    got = get_initial_frames_for_free_simulations(
        [trimmed], 'A', 'R', position=0.0, untrimmed_paths=[path])
    assert got[0].locs[-1] == 2  # the run is locs 2..4, not 0..4


def test_couple_is_always_two_frames_with_the_prefix_on_the_boundary_side(tmp_path):
    path = _path(tmp_path, 'AAAAARRRRBBBB')
    trimmed = path[4:]
    for position in (0.0, 0.25, 0.5, 1.0):
        got = get_initial_frames_for_free_simulations(
            [trimmed], 'A', 'R', position=position, untrimmed_paths=[path])
        couple = got[0]
        assert len(couple) == 2
        assert couple.locs[0] == couple.locs[-1] + 1   # prefix is boundary-side
        assert couple.states[-1] == 'A'


def test_random_is_reproducible_for_a_given_worker(tmp_path):
    path = _path(tmp_path, 'AAAAARRRRBBBB')
    trimmed = path[4:]
    picks = []
    for _ in range(2):
        got = get_initial_frames_for_free_simulations(
            [trimmed], 'A', 'R', position='random',
            untrimmed_paths=[path], rng=np.random.default_rng(7))
        picks.append(got[0].locs[-1])
    assert picks[0] == picks[1]
    other = get_initial_frames_for_free_simulations(
        [trimmed], 'A', 'R', position='random',
        untrimmed_paths=[path], rng=np.random.default_rng(8))
    assert 0 <= other[0].locs[-1] <= 4


# ── restart sources ───────────────────────────────────────────────────────

def test_restart_source_default_and_bare():
    assert FREE_RESTART_SOURCES == ('crossing', 'seed', 'basin',
                                    'equilibrium', 'transitions')
    assert parse_restart_source('', states='ARB') == ('crossing', {})
    assert parse_restart_source('equilibrium', states='ARB') == \
        ('equilibrium', {})


def test_restart_source_per_state_dict():
    default, per_state = parse_restart_source(
        {'A': 'equilibrium', 'B': 'crossing'}, states='ARB')
    assert default == 'crossing'
    assert per_state == {'A': 'equilibrium', 'B': 'crossing'}


def test_restart_source_unknown_raises():
    with pytest.raises(TypeError, match='free_restart_source'):
        parse_restart_source('boundary', states='ARB')


def test_in_state_sources_degrade_on_the_reactive_state():
    for source in FREE_RESTART_IN_STATE_SOURCES:
        p = _params(free_restart_source=source)
        assert p.free_restart_source_for('R') == 'crossing'
        assert p.free_restart_source_for('A') == source


def test_restart_source_canonical_form_round_trips():
    for value in ('crossing', 'equilibrium', {'A': 'equilibrium'},
                  {'A': 'basin', 'B': 'transitions'}):
        once = canonical_restart_source(value, states='ARB')
        assert canonical_restart_source(once, states='ARB') == once


# ── legacy ────────────────────────────────────────────────────────────────

def test_deprecated_transitions_flag_maps_to_transitions():
    assert legacy_transitions_replacement('A') == {'A': 'transitions'}
    assert legacy_transitions_replacement('AB') == {'A': 'transitions',
                                                    'B': 'transitions'}


def test_deprecated_transitions_flag_all():
    assert legacy_transitions_replacement('all') == 'transitions'


def test_deprecated_transitions_flag_empty_is_silent():
    assert legacy_transitions_replacement('') == ''
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _params(restart_free_simulations_with_transitions='')


def test_deprecated_transitions_flag_warns():
    with pytest.warns(DeprecationWarning, match='free_restart_source'):
        p = _params()
        p.restart_free_simulations_with_transitions = 'A'
    assert p.free_restart_source_for('A') == 'transitions'
    assert p.free_restart_source_for('B') == 'crossing'


def test_deprecated_flag_plus_new_field_raises():
    p = _params()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        p.__dict__['restart_free_simulations_with_transitions'] = 'A'
    p.__dict__['free_restart_source'] = {'A': 'equilibrium'}
    with pytest.raises(TypeError, match='both'):
        p._free_restart_spec()


def test_round_one_field_is_gone():
    """The round-1 trial switch is removed outright, not deprecated."""
    assert 'restart_free_simulations_from' not in \
        aimmd.Params.placeholder.__dataclass_fields__
    with pytest.raises(TypeError):
        _params().restart_free_simulations_from = 'A:equilibrium'


# ── worker wiring ─────────────────────────────────────────────────────────

def test_worker_maps_sources_to_basin_weighting():
    from aimmd.worker._free import _basin_weighting_for_source
    assert _basin_weighting_for_source('equilibrium') == 'unbiased'
    assert _basin_weighting_for_source('basin') == 'occupancy'
    for source in ('crossing', 'seed', 'transitions'):
        assert _basin_weighting_for_source(source) is None


def test_min_frames_field_is_renamed():
    p = _params()
    assert p.free_restart_min_frames == 0
    assert 'free_restart_basin_min_frames' not in \
        aimmd.Params.placeholder.__dataclass_fields__
