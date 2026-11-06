"""The fit keywords checked without training (``aimmd.cgaimmd.fitting.settings_only``).

The preflight calls a params file's ``fit`` inside ``settings_only()``: the
C-GAIMMD fits check their keywords and start and raise ``FitSettings``
instead of reading data or training. ``cgaimmd_fits_behind`` tells whether a
``fit`` reaches one of them (so that no other fit is ever called by the
preflight).
"""

import functools
import threading

import pytest

from aimmd.cgaimmd import fitting
from aimmd.cgaimmd.fitting import FitSettings, cgaimmd_fits_behind, settings_only
from aimmd.cgaimmd.flow_matching.fit import flow_matching_fit, labels_from_start_problem

FLOW = 'aimmd.cgaimmd.flow_matching.fit.flow_matching_fit'


def _helper(params, pathensemble, **kwargs):
    return flow_matching_fit(params, pathensemble, steps_per_round=8, **kwargs)


def test_fits_behind_a_params_file_wrapper():
    def fit(params, pathensemble, verbose=False, worker=None):         # the params-file pattern
        return flow_matching_fit(params, pathensemble, steps_per_round=4000)

    def through_a_helper(params, pathensemble, verbose=False, worker=None):
        return _helper(params, pathensemble)

    def other(params, pathensemble, verbose=False, worker=None):
        return [], {}

    assert cgaimmd_fits_behind(flow_matching_fit) == [FLOW]
    assert cgaimmd_fits_behind(fit) == [FLOW]
    assert cgaimmd_fits_behind(through_a_helper) == [FLOW]
    assert cgaimmd_fits_behind(functools.partial(flow_matching_fit, steps_per_round=8)) == [FLOW]
    assert cgaimmd_fits_behind(other) == [] and cgaimmd_fits_behind(None) == []
    from aimmd.network import fit as aimmd_fit
    assert cgaimmd_fits_behind(aimmd_fit) == []


def test_settings_only_is_per_thread_and_nests():
    seen = []
    with settings_only():
        assert fitting.settings_only_active()
        with settings_only():
            assert fitting.settings_only_active()
        assert fitting.settings_only_active()
        thread = threading.Thread(target=lambda: seen.append(fitting.settings_only_active()))
        thread.start()
        thread.join()
    assert seen == [False] and not fitting.settings_only_active()


def test_flow_matching_fit_checks_without_reading_data(tmp_path):
    """Inside settings_only the fit raises before it touches the path ensemble (None here)."""
    import torch
    from tests._cgaimmd_helpers import cgaimmd_params_namespace
    from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
    network = TinyFlowNetwork()
    torch.save(network.state_dict(), tmp_path / 'initial_network.h5')
    params = cgaimmd_params_namespace(network=network,
                                      initial_network=str(tmp_path / 'initial_network.h5'))
    with settings_only(), pytest.raises(FitSettings) as raised:
        flow_matching_fit(params, None, steps_per_round=5, relabel_every=2)
    checked = raised.value
    assert checked.fit == 'aimmd.cgaimmd.flow_matching.fit'
    assert checked.keywords == {'steps_per_round': 5, 'relabel_every': 2}
    assert checked.settings['steps_per_round'] == 5 and checked.settings['lr'] == 3e-4
    assert checked.problems == [] and checked.notes == ['every round starts from initial_network']
    with settings_only(), pytest.raises(TypeError, match='unexpected keyword'):
        flow_matching_fit(params, None, steps_per_rnd=5)


def test_labels_from_start_problem():
    from aimmd.cgaimmd.flow_matching.fit import FLOW_FIT_DEFAULTS
    assert labels_from_start_problem(FLOW_FIT_DEFAULTS) is None                    # 1000 of 4000
    for change in ({'relabel_every': 0}, {'relabel_every': 4000},
                   {'relabel_every': 10, 'steps_per_round': 10}):
        assert 'relabel_every' in labels_from_start_problem({**FLOW_FIT_DEFAULTS, **change})
    for unused in ({'fm_weight': 0.0}, {'n_fm': 0}, {'instate_fm_share': 1.0}):
        assert labels_from_start_problem({**FLOW_FIT_DEFAULTS, 'relabel_every': 0,
                                          **unused}) is None


@pytest.mark.bioemu2
def test_bioemu2_sequential_fit_checks_its_round_settings():
    pytest.importorskip('bioemu2')
    from aimmd.cgaimmd.bioemu2.fit import sequential_fit
    assert cgaimmd_fits_behind(sequential_fit) == ['aimmd.cgaimmd.bioemu2.fit.sequential_fit']
    with settings_only(), pytest.raises(FitSettings) as raised:
        sequential_fit(None, None, adapter_steps=3000, ligand_weight=5.0, shot_stride=2)
    assert raised.value.keywords == {'adapter_steps': 3000, 'ligand_weight': 5.0, 'shot_stride': 2}
    assert raised.value.problems == []
    with settings_only(), pytest.raises(TypeError, match="'adapter_stepz'"):
        sequential_fit(None, None, adapter_stepz=3000)
    with settings_only(), pytest.raises(ValueError, match='p_uncond must be in'):
        sequential_fit(None, None, p_uncond=1.5)
