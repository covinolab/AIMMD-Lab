"""The archived chignolin production params on this package: its preparation objects and hooks.

``tests/data/legacy_cgaimmd/params_chignolin_faimmd_equilibrium_free.py`` is the
params file of the online BioEmu2 chignolin campaign of 27 Oct 2026, copied
verbatim (md5 07b257c75fee8c3d869ff2d0bf1973c8). It imports its preparation
tools from ``aimmd.flow`` and builds them at import, with
``restrain='flow_heavy'``, and its validity hook calls
``aimmd.flow.validate._polish``. Here it runs on this checkout, through the
old names, with stand-ins for what it reads from its folder and from the
system folder on JUPITER:

- the chignolin system becomes NANMA: its solvated system
  (``tests/cgaimmd_nanma_water``) is ``run.gro``/``topol.top``, NANMA in
  vacuum (``tests/cgaimmd_nanma``) the system folder of the validity check
  and of the polish (``BIOEMU2_COMMITTOR_CHIGNOLIN``, read by the params file
  itself), NANMA's free MD the clash reference;
- ``chignolin_states`` (``N_PROTEIN = 22``) and ``bioemu2_committor`` (the
  network class and ``values_function_for``) are small stubs.

The hook then runs with GROMACS (the polish is a real restrained EM), and
``PolishedValidity`` built from the same objects gives the same result. Loading
the whole params file through ``Params.load`` (``chain_type='flow'``, the
``flow_*`` fields, the network) and reading old sidecars are tested where those
parts live.
"""

import importlib.util
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('networkx')
mda = pytest.importorskip('MDAnalysis')

from aimmd.cgaimmd import _legacy                                            # noqa: E402
from tests._cgaimmd_helpers import gromacs_has_thread_mpi                    # noqa: E402

pytestmark = pytest.mark.gromacs          # read_masses of the water needs GROMACS' share folder

HERE = Path(__file__).resolve().parent
ARCHIVED = HERE / 'data' / 'legacy_cgaimmd' / 'params_chignolin_faimmd_equilibrium_free.py'
NANMA = HERE / 'cgaimmd_nanma'
WATER = HERE / 'cgaimmd_nanma_water'

STATES_STUB = '''\
"""Stand-in for the chignolin state definition: NANMA has 22 atoms."""
import numpy as np

N_PROTEIN = 22


def states_function(trajectory):
    return np.array(['R'] * len(trajectory))
'''

NETWORK_STUB = '''\
"""Stand-in for bioemu2_committor.aimmd_flow: the members of the archived network class."""
import numpy as np
import torch


class BioEmu2FlowNetwork(torch.nn.Module):
    n_atoms = 22
    lambda_clip = 10.0

    def __init__(self, *, shared=False, cfg_weight=1.0, **kwargs):
        super().__init__()
        self.head = torch.nn.Linear(66, 1)

    def forward(self, descriptors):
        return self.head(descriptors.reshape(len(descriptors), -1))

    def generate(self, lam, n=None, **kwargs):
        raise NotImplementedError


def values_function_for(network, batchsize=256):
    def values(descriptors):
        return np.zeros(len(descriptors))
    return values
'''


@pytest.fixture
def archived(tmp_path, monkeypatch, gmx_executable):
    """The archived params module, imported from a folder of stand-ins, and the old-name warnings.

    The file calls GROMACS by the bare name ``gmx``; a ``gmx`` link to the tests' GROMACS
    goes first on PATH, so that it runs where GROMACS is installed as ``gmx_mpi`` only.
    """
    run = tmp_path / 'run'
    gmx_bin = tmp_path / 'gmx_bin'
    gmx_bin.mkdir()
    (gmx_bin / 'gmx').symlink_to(gmx_executable)
    monkeypatch.setenv('PATH', f'{gmx_bin}{os.pathsep}{os.environ.get("PATH", "")}')
    system = tmp_path / 'system'
    (run / 'bioemu2_committor').mkdir(parents=True)
    system.mkdir()
    shutil.copy(ARCHIVED, run / ARCHIVED.name)
    shutil.copy(WATER / 'conf.gro', run / 'run.gro')
    shutil.copy(WATER / 'topol.top', run / 'topol.top')
    shutil.copy(WATER / 'em.mdp', run / 'em_flow.mdp')
    shutil.copy(WATER / 'nvt.mdp', run / 'relax_flow.mdp')
    shutil.copy(NANMA / 'free_md_A.xtc', run / 'clash_reference_every50th_free_md_frame.xtc')
    np.save(run / 'validity_node_labels.npy', np.arange(22))
    (run / 'chignolin_states.py').write_text(STATES_STUB)
    (run / 'bioemu2_committor' / '__init__.py').write_text('')
    (run / 'bioemu2_committor' / 'aimmd_flow.py').write_text(NETWORK_STUB)
    shutil.copy(NANMA / 'conf.gro', system / 'conf_A_box.gro')
    shutil.copy(NANMA / 'topol.top', system / 'topol.top')
    shutil.copy(NANMA / 'em.mdp', system / 'em.mdp')
    monkeypatch.setenv('BIOEMU2_COMMITTOR_CHIGNOLIN', str(system))
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    _legacy._reset()
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            spec = importlib.util.spec_from_file_location('archived_chignolin_params',
                                                          run / ARCHIVED.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        yield SimpleNamespace(module=module, run=run, system=system, record=record)
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - saved_modules:
            if name in ('chignolin_states', 'archived_chignolin_params') or \
                    name.startswith('bioemu2_committor'):
                del sys.modules[name]


def _deprecations(record):
    """The old names of C-GAIMMD among the warnings (other libraries may warn as well)."""
    return sorted(str(w.message).split(' is deprecated')[0] for w in record
                  if issubclass(w.category, DeprecationWarning)
                  and 'the old names of C-GAIMMD are read for one release' in str(w.message))


def test_the_archived_params_build_their_tools_through_the_old_names(archived):
    """Every old name of the file resolves to the new tool, with one DeprecationWarning each."""
    from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors
    from aimmd.cgaimmd.preparation.minimize import GromacsMinimizer
    from aimmd.cgaimmd.preparation.solvate import GromacsSolvator
    from aimmd.cgaimmd.preparation.validity import TopologyValidity
    module = archived.module
    assert _deprecations(archived.record) == sorted([
        'aimmd.flow.descriptors.FlowAtoms', 'aimmd.flow.descriptors.FlowDescriptors',
        'aimmd.flow.minimize.GromacsMinimizer', 'aimmd.flow.solvate.GromacsSolvator',
        'aimmd.flow.solvate.NVTStage', 'aimmd.flow.topology.read_masses',
        'aimmd.flow.validity.TopologyValidity', "restrain 'flow_heavy'"])
    assert isinstance(module._descriptors, GeneratorDescriptors)
    assert module._descriptors.generator_atoms == GeneratorAtoms(22)
    assert isinstance(module._validity, TopologyValidity)
    assert isinstance(module._solvator, GromacsSolvator)
    for minimizer in (module._polisher, module._minimizer):
        assert isinstance(minimizer, GromacsMinimizer)
        assert minimizer.restrain == minimizer.settings['restrain'] == 'generator_heavy'
    assert module._solvator.settings['restrain'] == 'generator_heavy'
    # the descriptors of a frame of the run: NANMA whole, 22 x 3
    universe = mda.Universe(str(archived.run / 'run.gro'))
    rows = module.descriptors_function(universe.trajectory)
    assert rows.shape == (1, 66) and rows.dtype == np.float32


def _samples():
    """Three generated stand-ins: a free-MD frame, one with a hydrogen bonded twice, one not finite."""
    from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors
    descriptors = GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(22))
    universe = mda.Universe(str(NANMA / 'conf.gro'), str(NANMA / 'free_md_A.xtc'))
    frames = np.stack([descriptors.select(ts.positions, ts.dimensions)
                       for ts in universe.trajectory[:3]])
    frames[1, 7] = (frames[1, 6] + frames[1, 8]) / 2          # HL between NL and CA
    frames[2, 10] = np.nan
    return frames


def test_archived_hook_equals_polished_validity_with_gromacs(archived, tmp_path):
    """The hand-written hook (real restrained EM) and PolishedValidity of its objects agree."""
    from aimmd.cgaimmd.preparation.validity import PolishedValidity
    if not gromacs_has_thread_mpi(shutil.which('gmx')):
        pytest.skip('the archived params run mdrun -ntmpi 1, which an MPI build of GROMACS refuses')
    module = archived.module
    samples = _samples()

    def context(name):
        scratch = tmp_path / name
        scratch.mkdir()
        return SimpleNamespace(scratch=str(scratch), seed=4321, log=print, params=None,
                               must_stop=lambda: False, settings={})

    _legacy._reset()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        old = module.flow_validity_function(samples, context('hook'))
    assert _deprecations(record) == ['aimmd.flow.types.ValidityResult',
                                     'aimmd.flow.validate._polish']
    check = PolishedValidity(module._validity, module._polisher,
                             str(archived.system / 'conf_A_box.gro'),
                             workers=module._POLISH_WORKERS)
    new = check(samples, context('polished_validity'))
    assert np.asarray(old.ok).tolist() == np.asarray(new.ok).tolist()
    assert [str(r) for r in old.reason] == [str(r) for r in new.reason]
    np.testing.assert_allclose(np.asarray(new.positions)[:2], np.asarray(old.positions)[:2],
                               atol=1e-6)
    assert json.dumps(old.info, sort_keys=True) == json.dumps(new.info, sort_keys=True)
    # what happened: the raw frame is valid, the doubly bonded hydrogen is repaired by the polish,
    # the non-finite sample is not polished
    assert [i['raw_reason'] for i in new.info] == ['', 'bond_count', 'non_finite']
    assert np.asarray(new.ok).tolist() == [True, True, False]
    assert new.reason[2] == 'polish: not finite'
    assert all(i['polish_ok'] for i in new.info[:2]) and 'polish_ok' not in new.info[2]
    # the polish moved the heavy atoms little (restrained EM) and the hydrogen back to its bond
    assert new.info[0]['heavy_rms_displacement_A'] < 0.1
    assert 0.9 < np.linalg.norm(new.positions[1][7] - new.positions[1][6]) < 1.1
