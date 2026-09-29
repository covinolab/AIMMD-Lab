"""`chain_type` is normalised to exactly 'tps' or 'rfps'.

The validator in `aimmd/params/_helpers.py` accepts any case and any value
starting with 'tp' or 'rf', and meant to store the canonical name. Every
consumer then compares against the canonical string (`chain_type == 'tps'` in
the shoot worker, the trainer, `shot_paths` and the TPS pool-size check), so a
value that is accepted but not normalised -- 'tp' -- runs the chain as RFPS in
those places while `select_shooting_point` (which tests `startswith('tp')`)
treats it as TPS, and the TPS-only check `selection_pool_size == 1` is skipped.
"""
import pytest

import aimmd


@pytest.fixture
def params():
    params = aimmd.Params.placeholder.copy()
    # the TPS-only consistency check needs a pool of one
    params.update(selection_pool_size=1, save=False)
    return params


@pytest.mark.parametrize('value, expected', [
    ('tps', 'tps'), ('TPS', 'tps'), ('tp', 'tps'), ('Tp', 'tps'),
    ('rfps', 'rfps'), ('RFPS', 'rfps'), ('rf', 'rfps'), ('Rf', 'rfps'),
])
def test_chain_type_is_normalised(params, value, expected):
    params.update(chain_type=value, save=False)
    assert params.chain_type == expected


def test_tps_prefix_triggers_the_tps_pool_size_check(params):
    """'tp' must be held to the same `selection_pool_size == 1` rule as 'tps'."""
    params.update(selection_pool_size=3, save=False)
    with pytest.raises(TypeError, match='selection pool size must be 1'):
        params.update(chain_type='tp', save=False)


def test_unknown_chain_type_still_raises(params):
    with pytest.raises(TypeError, match='chain_type'):
        params.update(chain_type='rejection-free', save=False)
