"""Seeding free simulations away from the boundary with more than one initial path.

``get_initial_frames_for_free_simulations`` checks each extracted couple with
the local name ``states``, which is also its ``states`` argument
(``params.states``). The check overwrote the argument with the couple's
per-frame labels, so from the second initial path on the untrimmed-path
lookup and the seed placement received an array instead of ``'ARB'`` and the
free worker died with "The truth value of an array ... is ambiguous" for every
``free_seeding_position`` other than ``'boundary'``.
"""

import numpy as np
import pytest

import aimmd
from aimmd.worker.utils import get_initial_frames_for_free_simulations
from tests._helpers_unit import build_path


def one_atom(x):
    positions = np.zeros((len(x), 1, 3), dtype=np.float32)
    positions[:, 0, 0] = x
    return positions


X = [-1.4, -1.2, -1.0, -0.7, -0.2, 0.2, 0.7, 1.0, 1.2]         # A A A A R R B B B


@pytest.mark.parametrize('position', [0.0, 0.5, 'random'])
def test_every_initial_path_seeds_at_the_position(tmp_path, position):
    first = build_path(tmp_path, stem='first', positions=one_atom(X))
    second = build_path(tmp_path, stem='second', positions=one_atom(X[::-1]))
    untrimmed = {'first.xtc': first, 'second.xtc': second}
    couples = get_initial_frames_for_free_simulations(
        aimmd.PathEnsemble([first, second]), 'A', 'R', position=position,
        untrimmed_paths=untrimmed, states='ARB', rng=np.random.default_rng(0))
    assert len(couples) == 2
    for couple in couples:
        assert couple.states[-1] == 'A'                        # the seed is in the target state
    if position == 0.0:                                         # the deepest A frame of each path
        np.testing.assert_allclose([c.positions[-1, 0, 0] for c in couples], [-1.4, -1.4])


def test_boundary_unchanged_with_two_paths(tmp_path):
    first = build_path(tmp_path, stem='first', positions=one_atom(X))
    second = build_path(tmp_path, stem='second', positions=one_atom(X[::-1]))
    couples = get_initial_frames_for_free_simulations(
        aimmd.PathEnsemble([first, second]), 'B', 'R', position=1.0, states='ARB')
    assert [list(c.states) for c in couples] == [['B', 'B'], ['B', 'B']]
