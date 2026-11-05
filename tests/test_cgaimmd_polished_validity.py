"""The polish of generated samples and the validity check around it.

``polish`` minimizes each raw structure alone; ``PolishedValidity`` checks the
raw sample, polishes it and checks the polished sample. The BioEmu2
production params wrote that check out by hand around the private
``aimmd.flow.validate._polish``; the recorded sample of
``tests/_cgaimmd_polish_golden.py`` pins ``PolishedValidity`` to what that
hand-written hook returned on the flow branch, and the hook itself still
runs through the old names. No test here starts GROMACS: the minimizer is a
recorded stand-in.
"""

import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('networkx')
mda = pytest.importorskip('MDAnalysis')

from aimmd.cgaimmd import _legacy                                            # noqa: E402
from aimmd.cgaimmd.preparation.minimize import (GromacsMinimizer, PolishResult,  # noqa: E402
                                                polish)
from aimmd.cgaimmd.preparation.validity import PolishedValidity              # noqa: E402
from aimmd.cgaimmd.types import ValidityResult                               # noqa: E402
from tests import _cgaimmd_polish_golden as golden                           # noqa: E402


def _deprecations(record, text=''):
    return [str(w.message) for w in record
            if issubclass(w.category, DeprecationWarning) and text in str(w.message)]


@pytest.fixture(scope='module')
def recorded():
    """The recorded sample: inputs, the stand-in minimizer and the archived hook's output."""
    data = np.load(golden.GOLDEN)
    minimizer = golden.RecordedMinimizer(data['displacements'], data['fails'],
                                         golden.heavy_atoms())
    checker, _ = golden._checker('aimmd.cgaimmd.preparation')
    return {'sample': data['sample'], 'minimizer': minimizer, 'checker': checker,
            'positions': data['positions'], 'ok': data['ok'], 'reason': data['reason'],
            'info': json.loads(str(data['info'])), 'names': [str(n) for n in data['names']]}


def _assert_recorded(result, recorded):
    positions, ok, reason, info = golden.as_arrays(result)
    np.testing.assert_array_equal(positions, recorded['positions'])
    assert ok.tolist() == recorded['ok'].tolist()
    assert reason.tolist() == recorded['reason'].tolist()
    assert json.loads(info) == recorded['info']


def test_the_recorded_sample_covers_every_branch(recorded):
    """The sample has raw failures the polish repairs, polish failures and polish damage."""
    by_name = dict(zip(recorded['names'], zip(recorded['ok'], recorded['reason'],
                                               recorded['info'])))
    assert by_name['hydrogen_bonded_twice'][2]['raw_reason'] == 'bond_count'
    assert by_name['hydrogen_bonded_twice'][0]                       # repaired by the polish
    assert by_name['clash'][2]['raw_reason'] == 'clash' and by_name['clash'][0]
    assert by_name['not_finite'][1] == 'polish: not finite'
    assert 'polish_ok' not in by_name['not_finite'][2]
    assert by_name['polish_fails'][1].startswith('polish: minimize_failed: ')
    assert by_name['polish_fails'][2]['heavy_rms_displacement_A'] is None
    assert by_name['polish_damages'][2]['raw_ok'] and by_name['polish_damages'][1] == 'bond_count'
    assert by_name['mirror_image'][1] == 'chirality'
    # the exchanged hydrogens are relabelled back into topology order
    i = recorded['names'].index('hydrogens_exchanged')
    assert not np.allclose(recorded['positions'][i], recorded['sample'][i], atol=0.1)
    # the far structure is polished where it is (its centroid kept)
    i = recorded['names'].index('far_from_origin')
    np.testing.assert_allclose(recorded['positions'][i].mean(0), recorded['sample'][i].mean(0),
                               atol=1e-12)


@pytest.mark.parametrize('workers', [1, 2])
def test_polished_validity_matches_the_archived_hook(tmp_path, recorded, workers):
    """PolishedValidity returns, bit for bit, what the archived hand-written hook returned."""
    system = golden.system_folder(tmp_path / 'system')
    check = PolishedValidity(recorded['checker'], recorded['minimizer'],
                             os.path.join(system, 'conf_A_box.gro'), workers=workers)
    result = check(recorded['sample'], golden.context(tmp_path / 'scratch'))
    assert isinstance(result, ValidityResult)
    _assert_recorded(result, recorded)
    assert (tmp_path / 'scratch' / 'polish').is_dir()
    assert check.settings['function'] == 'aimmd.cgaimmd.preparation.validity.PolishedValidity'
    assert check.settings['minimizer'] == recorded['minimizer'].settings
    assert check.settings['validity'] == recorded['checker'].settings
    json.dumps(check.settings)


def test_archived_hook_runs_through_the_old_names(tmp_path, recorded):
    """The hand-written hook, verbatim, on this checkout: same result, one warning per old name."""
    _legacy._reset()
    hook = golden.load_archived_hook(recorded['checker'], recorded['minimizer'],
                                     golden.system_folder(tmp_path / 'system'))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        result = hook.flow_validity_function(recorded['sample'],
                                             golden.context(tmp_path / 'scratch'))
        _assert_recorded(result, recorded)
        again = hook.flow_validity_function(recorded['sample'],
                                            golden.context(tmp_path / 'scratch_again'))
        _assert_recorded(again, recorded)
    polish_warnings = _deprecations(record, 'aimmd.flow.validate._polish')
    assert polish_warnings == ['aimmd.flow.validate._polish is deprecated; use '
                               'aimmd.cgaimmd.preparation.minimize.polish (the old names of '
                               'C-GAIMMD are read for one release)']
    assert len(_deprecations(record, 'aimmd.flow.types.ValidityResult')) == 1
    assert len(_deprecations(record)) == 2


def test_old_private_name_of_the_polish(tmp_path, recorded):
    """``aimmd.flow.validate._polish`` with the archived call: ``polish``, one DeprecationWarning."""
    _legacy._reset()
    import aimmd.flow.validate as validate
    system = golden.system_folder(tmp_path / 'system')
    universe = mda.Universe(os.path.join(system, 'conf_A_box.gro'))
    params = SimpleNamespace(universe=universe, parent=system)
    finite = np.isfinite(recorded['sample']).reshape(len(recorded['sample']), -1).all(1)
    atom_count = SimpleNamespace(n_atoms=golden.N_ATOMS)
    with pytest.warns(DeprecationWarning, match='aimmd.flow.validate._polish is deprecated; use '
                                                'aimmd.cgaimmd.preparation.minimize.polish'
                      ) as record:
        old = validate._polish(recorded['sample'][finite], recorded['minimizer'], atom_count,
                               params, universe.dimensions, str(tmp_path / 'old'),
                               golden.WORKERS, golden.SEED)
    assert len(_deprecations(record)) == 1
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)          # warned before: no repeat
        assert validate._polish is polish
    new = polish(recorded['sample'][finite], recorded['minimizer'], atom_count, params,
                 universe.dimensions, str(tmp_path / 'new'), golden.WORKERS, golden.SEED)
    assert len(old) == len(new) == int(finite.sum())
    for a, b in zip(old, new):
        assert a.ok == b.ok and a.reason == b.reason and a.record == b.record
        if a.ok:
            np.testing.assert_array_equal(a.positions, b.positions)


class _Recording:
    """A minimizer that records its contexts and moves every atom by +1 A in x.

    It returns one atom more than it was given (as a solvated frame would), and
    fails for the structures whose context seed is ``10 + k`` for ``k`` in `fail`.
    """

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.contexts = []

    def __call__(self, frame, context):
        self.contexts.append(context)
        x = np.asarray(frame.positions, dtype=float)
        if context.seed - 10 in self.fail:
            return SimpleNamespace(positions=x, ok=False, reason='minimize_failed: no',
                                   record={'settings': {'a': 1}, 'error': 'no'})
        return SimpleNamespace(positions=np.concatenate([x + [1.0, 0.0, 0.0], x[:1]]), ok=True,
                               reason='', record={'settings': {'a': 1}, 'steps': 3})


def test_polish(tmp_path):
    """Centred in the box, minimized, back at the input's centroid; failures and records."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4, 3)) * 2.0 + np.array([[[50.0, -20.0, 7.0]]])
    box = np.array([30.0, 30.0, 30.0, 90.0, 90.0, 90.0])
    minimizer = _Recording(fail={1})
    params = SimpleNamespace(universe=None, parent=str(tmp_path))
    results = polish(x, minimizer, SimpleNamespace(n_atoms=4), params, box, str(tmp_path), 1, 10)
    assert [type(r) for r in results] == [PolishResult] * 3
    assert [r.ok for r in results] == [True, False, True]
    for k in (0, 2):
        # the first M atoms of the result (the extra atom is dropped), translated back
        np.testing.assert_allclose(results[k].positions.mean(0), x[k].mean(0), atol=1e-12)
        np.testing.assert_allclose(results[k].positions - results[k].positions.mean(0),
                                   x[k] - x[k].mean(0), atol=1e-12)
        assert results[k].record == {'steps': 3} and results[k].reason == ''
    assert results[1] == (None, False, 'minimize_failed: no', {'error': 'no'})
    # the context of each minimization
    contexts = minimizer.contexts
    assert [c.seed for c in contexts] == [10, 11, 12]
    assert [c.path_name for c in contexts] == ['polish000000', 'polish000001', 'polish000002']
    assert all(c.scratch == str(tmp_path) and c.params is params for c in contexts)
    assert all(c.settings == {'keep_scratch': False} and not c.must_stop() for c in contexts)
    # threads give the same results in the same order
    threaded = polish(x, _Recording(fail={1}), SimpleNamespace(n_atoms=4), params, box,
                      str(tmp_path), workers=3, seed=10, keep_scratch=True,
                      must_stop=lambda: True)
    assert [r.ok for r in threaded] == [True, False, True]
    np.testing.assert_array_equal(threaded[2].positions, results[2].positions)
    with pytest.raises(ValueError, match='6 numbers'):
        polish(x, minimizer, SimpleNamespace(n_atoms=4), params, box[:3], str(tmp_path))


def test_polish_hands_the_stop_request_and_scratch_setting_on(tmp_path):
    """must_stop and keep_scratch reach the minimizer's context."""
    minimizer = _Recording()
    polish(np.zeros((1, 2, 3)), minimizer, SimpleNamespace(n_atoms=2), None,
           [10.0, 10.0, 10.0, 90.0, 90.0, 90.0], str(tmp_path), must_stop=lambda: True,
           keep_scratch=True)
    assert minimizer.contexts[0].must_stop() is True
    assert minimizer.contexts[0].settings == {'keep_scratch': True}


def _two_atoms(tmp_path, box=True):
    lines = ['two atoms', '    2', '    1MOL     C1    1   0.100   0.100   0.100',
             '    1MOL     C2    2   0.250   0.100   0.100']
    lines.append('   3.00000   3.00000   3.00000' if box else '   0.00000   0.00000   0.00000')
    path = tmp_path / ('two.gro' if box else 'two_nobox.gro')
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


def test_polished_validity_info_keys_and_context(tmp_path):
    """info_keys copy the checker's own info; the polish gets the run's stop request and setting."""
    def checker(positions, context):
        x = np.asarray(positions, dtype=float)
        spread = [float(np.ptp(s[:, 0])) for s in x]
        return ValidityResult(positions=x, ok=np.array([s < 2.0 for s in spread]),
                              reason=np.array(['' if s < 2.0 else 'wide' for s in spread],
                                              dtype=object),
                              info=[{'spread_A': s, 'other': 1} for s in spread])

    minimizer = _Recording()
    check = PolishedValidity(checker, minimizer, _two_atoms(tmp_path), info_keys=('spread_A',))
    assert check.n_atoms == 2 and check.box.tolist() == [30.0, 30.0, 30.0, 90.0, 90.0, 90.0]
    x = np.array([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    stop = {'asked': False}
    context = SimpleNamespace(scratch=str(tmp_path / 'scratch'), seed=10, params=None,
                              must_stop=lambda: stop['asked'], settings={'keep_scratch': True})
    result = check(x, context)
    # the polish moved every atom by the same 1 A: the spread, and so the verdict, are the same
    assert result.ok.tolist() == [True, False] and result.reason.tolist() == ['', 'wide']
    assert result.info[0] == {'raw_ok': True, 'raw_reason': '', 'raw_spread_A': 1.5,
                              'polish_ok': True, 'heavy_rms_displacement_A': None,
                              'spread_A': 1.5}
    assert result.info[1]['raw_spread_A'] == 3.0 and result.info[1]['spread_A'] == 3.0
    assert minimizer.contexts[0].settings == {'keep_scratch': True}
    stop['asked'] = True
    assert minimizer.contexts[0].must_stop() is True
    # the polish system: the structure's atoms, its folder
    polish_params = minimizer.contexts[0].params
    assert len(polish_params.universe.atoms) == 2
    assert polish_params.parent == str(tmp_path)
    # one (M, 3) sample is a batch of one
    assert check(x[0], context).ok.tolist() == [True]
    with pytest.raises(ValueError, match=r'expected \(n, 2, 3\)'):
        check(np.zeros((1, 3, 3)), context)


def test_polished_validity_uses_the_runs_gromacs_commands(tmp_path):
    """A minimizer without its own commands takes the run's grompp and mdrun, without -p and -n."""
    minimizer = GromacsMinimizer(topology=str(tmp_path / 'solute.top'))
    check = PolishedValidity(lambda x, c: None, minimizer, _two_atoms(tmp_path))
    run = SimpleNamespace(gmx_grompp='/g/gmx grompp -maxwarn 5 -p topol.top -n index.ndx',
                          gmx_mdrun='/g/gmx mdrun -ntomp 2 -plumed plumed.dat', parent='/run')
    polish_params = check._polish_params(run)
    assert polish_params.gmx_grompp == '/g/gmx grompp -maxwarn 5'
    grompp, mdrun, topology, _ = minimizer._commands(polish_params)
    assert grompp == '/g/gmx grompp -maxwarn 5' and mdrun == '/g/gmx mdrun -ntomp 2'
    assert topology == str(tmp_path / 'solute.top')
    # params without GROMACS commands (a toy run, or none at all)
    assert not hasattr(check._polish_params(None), 'gmx_grompp')


def test_polished_validity_configuration_errors(tmp_path):
    structure = _two_atoms(tmp_path)
    with pytest.raises(ValueError, match='topology='):
        PolishedValidity(lambda x, c: None, GromacsMinimizer(), structure)
    with pytest.raises(TypeError, match='validity'):
        PolishedValidity(None, _Recording(), structure)
    with pytest.raises(TypeError, match='minimizer'):
        PolishedValidity(lambda x, c: None, 'gmx mdrun', structure)
    with pytest.raises(ValueError, match='box'):
        PolishedValidity(lambda x, c: None, _Recording(), _two_atoms(tmp_path, box=False))
    with pytest.raises(ValueError, match='workers'):
        PolishedValidity(lambda x, c: None, _Recording(), structure, workers=0)
    check = PolishedValidity(lambda x, c: None, _Recording(), _two_atoms(tmp_path, box=False),
                             box=[20.0, 20.0, 20.0, 90.0, 90.0, 90.0])
    assert check.box.tolist() == [20.0, 20.0, 20.0, 90.0, 90.0, 90.0]


def test_the_package_names_the_polish_tools():
    import aimmd.cgaimmd.preparation as preparation
    assert preparation.PolishedValidity is PolishedValidity and preparation.polish is polish
    assert {'polish', 'PolishedValidity', 'GromacsMinimizer'} <= set(preparation.__all__)
    assert Path(golden.HOOK).is_file()
