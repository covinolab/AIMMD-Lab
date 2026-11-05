"""Reclassifying a shot as free must not touch the other excursions' entries.

`PathEnsemble.reweight(sp_cutoff_min=..., sp_cutoff_max=...)` treats shots
whose shooting value lies outside the cutoffs as free excursions. The loop
over those shots, ``for i, k in enumerate(mask)``, reads the k-th excursion's
values, but indexed ``to_o`` and ``extremes`` with the position ``i`` in the
mask instead of ``k``: the extreme of the reclassified shot was written into
another excursion's slot.

In the directional reweighting (``states='ARB'``) every such slot is written
again later (free and shot excursions alike), so the weights are unaffected;
in the diagnostic mode (fewer than three states, ``states='R'``) the returned
extremes of the other shots are wrong.
"""
import numpy as np
import pytest

from aimmd.pathensemble import PathEnsemble
from tests._helpers_unit import build_path


def _shot(tmp_path, stem, x, shooting_index):
    positions = np.zeros((len(x), 1, 3), dtype=np.float32)
    positions[:, 0, 0] = x
    return build_path(tmp_path, stem=stem, positions=positions,
                      shooting_index=shooting_index)


@pytest.fixture
def ensemble(tmp_path):
    # an A->R->A shot inside the cutoffs, then one below sp_cutoff_min = -0.3
    kept = _shot(tmp_path, 'kept', [-1.0, -0.2, 0.1, -1.0], 2)
    low = _shot(tmp_path, 'low', [-1.0, -0.4, -0.35, -1.0], 1)
    high = _shot(tmp_path, 'high', [-1.0, 0.35, 0.4, -1.0], 1)
    return PathEnsemble(kept, low, high)


@pytest.mark.parametrize('cutoffs', [
    {'sp_cutoff_min': -0.3}, {'sp_cutoff_max': 0.3},
    {'sp_cutoff_min': -0.3, 'sp_cutoff_max': 0.3}])
def test_diagnostic_extremes_of_other_shots_are_untouched(ensemble, cutoffs):
    reference = ensemble.reweight('R')
    result = ensemble.reweight('R', **cutoffs)
    extremes = result[4]

    # the extreme of the shot inside the cutoffs is not affected by the
    # reclassification (its factor is: shot factors are density corrected
    # among the remaining shots)
    assert extremes[0] == reference[4][0]
    # a reclassified shot's extreme is that of its own internal values
    # (diagnostic mode: -min)
    if 'sp_cutoff_min' in cutoffs:
        assert extremes[1] == pytest.approx(0.4)
    if 'sp_cutoff_max' in cutoffs:
        assert extremes[2] == pytest.approx(-0.35)
