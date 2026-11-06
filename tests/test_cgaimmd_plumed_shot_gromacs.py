"""One C-GAIMMD shot of vacuum NANMA under a static PLUMED bias, with GROMACS and PLUMED.

The shooting worker runs in this process with the real GROMACS engine: the
shot's halves run with the run's ``gmx_mdrun``, which carries ``-plumed``
(a lower wall on phi that acts deep in A only), while the candidate's
preparation (a ``GromacsMinimizer`` that uses the same mdrun command) runs
without it. The generator is registered by the params file and returns a
fixed configuration in R (a frame of unbiased free MD; here A is phi below
-70 degrees, B phi above 50 degrees, so that this frame at -60 degrees lies in
R), so that no network has to be trained.

Checked: the shot is registered with its PLUMED output (``path000001_COLVAR``)
and its bias cache (``path000001.xtc.bias.npy``, one value per frame, zero in
R, the wall's in A), the sidecar records the bias (the PLUMED file and its
sha256), and the minimization ran unbiased (no ``-plumed`` in its command, no
PLUMED output in its folder). Local lane only: GROMACS must be patched with
PLUMED (the conda GROMACS of CI is not); the test skips otherwise.
"""

import glob
import json
import os
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.io import file_sha256
from aimmd.worker.utils import write_initial_network
from tests._cgaimmd_helpers import mdrun_thread_options


pytestmark = pytest.mark.gromacs

NANMA = Path(__file__).resolve().parent / 'cgaimmd_nanma'


@pytest.fixture(autouse=True)
def _restore_sys_path(monkeypatch):
    """``Params.load`` leaves the params file's folder in ``sys.path``; take it out after the test.

    Otherwise a later ``import params`` (``tests/test_retinal.py`` imports its
    ``params.py`` by name) finds a ``params.py`` of this module's runs first.
    """
    monkeypatch.setattr(sys, 'path', list(sys.path))

PLUMED = '''phi: TORSION ATOMS=5,7,9,15
psi: TORSION ATOMS=7,9,15,17
wall: LOWER_WALLS ARG=phi AT=-1.40 KAPPA=200.0
PRINT ARG=phi,psi,wall.bias FILE=COLVAR STRIDE=1
'''
"""A static wall below phi = -1.40 rad (-80 degrees): it acts inside A (phi < -70) only."""

PARAMS = '''
import MDAnalysis as mda
import numpy as np
import torch
from MDAnalysis.lib.distances import calc_dihedrals
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.generation import register_generator
from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors
from aimmd.cgaimmd.preparation.minimize import GromacsMinimizer

engine = 'gromacs'
chain_type = 'cgaimmd'
states = 'ARB'
topology = 'conf.gro'
gmx_mdp = 'run.mdp'
gmx_grompp = '{gmx} grompp -maxwarn 5 -p topol.top'
gmx_mdrun = '{gmx} mdrun {threads} -nb cpu -pin off -plumed {plumed}'
gmx_eneconv = 'printf "c\\nc\\n" | {gmx} -nobackup eneconv -settime'
initial_paths = ['free_md_A.xtc', 'free_md_B.xtc']
initial_network = 'initial_network.h5'
masses_from_topology = True
round_shooting_index = True
max_length = 2000
record_bias = True
bias_source = 'file'
bias_reactive_threshold = 0.5
cgaimmd_generator = 'test_nanma_frame'
cgaimmd_parameters = {{'placement': 'box_center', 'lambda_clip': 10.0,
                      'window_after_trained_shots': 5, 'n_targets': 2,
                      'generation_device': 'cpu', 'values_device': 'cpu',
                      'training_device': 'cpu'}}
_MINIMIZER = GromacsMinimizer(em_mdp='em.mdp', nsteps=50)

def cgaimmd_minimize_function(frame, context):
    return _MINIMIZER(frame, context)

DESCRIPTORS = GeneratorDescriptors('conf.gro', GeneratorAtoms(n_solute=22))

def descriptors_function(trajectory):
    return DESCRIPTORS(trajectory)

def states_function(trajectory):
    """Half-planes in phi: A below -70 degrees, B above 50 degrees, R between."""
    out = []
    for frame in trajectory:
        p = frame.positions.astype(np.float64)
        box = frame.dimensions
        box = box if box is not None and np.all(box[:3] > 0) else None
        phi = np.degrees(calc_dihedrals(p[4], p[6], p[8], p[14], box=box))
        out.append('A' if phi < -70 else 'B' if phi > 50 else 'R')
    return np.array(out)

def bias_function(fname):
    """The wall's bias from <traj>_COLVAR (column 4), in kT at 300 K."""
    data = np.loadtxt(fname[:-len('.xtc')] + '_COLVAR', comments='#', ndmin=2)
    return data[:, 3] / 2.494339

class Network(TinyFlowNetwork):
    def __init__(self):
        super().__init__(n_atoms=22)

network = Network()

_universe = mda.Universe('conf.gro', 'free_md_A.xtc')
_universe.trajectory[0]                           # phi -59.5 degrees: in R here
R_FRAME = DESCRIPTORS.select(_universe.atoms.positions, _universe.dimensions)

class NanmaFrame:
    """A generator that returns the R frame for every request (no training needed)."""

    method = 'test_nanma_frame'

    def check(self, network):
        return []

    def settings(self):
        return {{}}

    def generate(self, network, lam, *, generator=None):
        x = torch.as_tensor(R_FRAME, dtype=torch.float32)
        return x.expand(len(lam), -1, -1).clone()

register_generator('test_nanma_frame', NanmaFrame)
'''


@pytest.fixture
def restore_signals():
    import signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def test_cgaimmd_shot_under_a_static_plumed_bias(tmp_path, monkeypatch, gmx_executable,
                                                 restore_signals):
    version = subprocess.run([gmx_executable, '-version'], capture_output=True, text=True,
                             timeout=60).stdout
    if 'plumed' not in version.lower():
        pytest.skip('this GROMACS is not patched with PLUMED')
    folder = tmp_path / 'nanma'
    folder.mkdir()
    for name in ('conf.gro', 'topol.top', 'run.mdp', 'em.mdp', 'free_md_A.xtc', 'free_md_B.xtc'):
        shutil.copy(NANMA / name, folder / name)
    plumed = folder / 'wall.dat'
    plumed.write_text(PLUMED)
    (folder / 'params.py').write_text(PARAMS.format(
        gmx=gmx_executable, threads=mdrun_thread_options(gmx_executable, 1), plumed=plumed))
    torch.manual_seed(0)
    torch.save(TinyFlowNetwork(n_atoms=22).state_dict(), folder / 'initial_network.h5')
    monkeypatch.chdir(folder)
    np.random.seed(0)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    run = str(folder / 'run')
    assert write_initial_network(params, run) == 'written'
    aimmd.Worker(params, run, nsteps=1, walltime=600).shoot(1, 0)

    chain = Path(run, 'chainR0')
    path_file = chain / 'path000001.xtc'
    assert path_file.is_file()
    path = params.shot_chains(run, 'R', 0)[0]
    # the shot ran under the bias: its PLUMED output and its bias cache, one value per frame
    assert (chain / 'path000001_COLVAR').is_file()
    bias = np.load(chain / 'path000001.xtc.bias.npy')
    assert len(bias) == len(path)
    states = np.asarray(path.states)
    assert np.all(np.abs(bias[states == 'R']) < 1e-6)        # zero where it is not applied
    assert np.all(bias >= 0) and np.isfinite(bias).all()
    # the sidecar records the bias
    record = json.loads(Path(f'{path_file}.cgaimmd.json').read_text())
    assert record['bias']['record_bias'] is True and record['bias']['bias_source'] == 'file'
    assert record['bias']['plumed_file'] == str(plumed)
    assert record['bias']['plumed_sha256'] == file_sha256(plumed)
    assert record['bias']['preparation_unbiased'] is True       # the wrapper of a GromacsMinimizer
    assert record['bias']['preparation_unverified_hooks'] == {}
    # the preparation ran without the bias: its mdrun had no -plumed, it wrote no PLUMED output
    minimize = record['minimize']
    assert minimize['ok'] is True
    assert '-plumed' not in json.dumps(minimize['record'])
    scratch = chain / 'cgaimmd_scratch'
    assert not glob.glob(str(scratch / '**' / 'COLVAR*'), recursive=True)
    assert not glob.glob(str(scratch / '**' / 'PLUMED.OUT'), recursive=True)
