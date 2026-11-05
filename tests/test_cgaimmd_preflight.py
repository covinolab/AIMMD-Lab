"""``python -m aimmd.cgaimmd.preflight params.py``: the checks before a C-GAIMMD job.

- the toy 1-D C-GAIMMD params pass (exit status 0), through ``main`` and as a
  module run in a subprocess; a params file that does not load, an
  ``initial_network`` that does not load into the network and a zero-point
  violation fail (exit status 1) and say why;
- the PLUMED checks on stand-in params: a missing or relative ``-plumed``
  file, a file the input reads that is missing or relative, no ``PRINT ...
  FILE=COLVAR`` for a file-mode bias, ``record_bias`` without ``-plumed`` and
  ``-plumed`` without ``record_bias``;
- the bias zero point of a reader-mode bias (the toy, on the prepared
  candidate);
- with GROMACS and PLUMED (marker ``gromacs``): vacuum NANMA params whose
  shots run under a static PLUMED wall; the zero point is evaluated as the
  shots evaluate it (grompp, step 0 of ``mdrun -plumed``, ``bias_function`` on
  the ``COLVAR``) and a shifted bias fails; a missing PLUMED file fails.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aimmd.cgaimmd import preflight
from aimmd.cgaimmd.preflight import FAIL, OK, SKIPPED, WARNING, PreflightReport, run_preflight
from tests._cgaimmd_helpers import cgaimmd_params_namespace
from tests.test_toy_1d_cgaimmd import make_campaign


TESTS = Path(__file__).resolve().parent
NANMA = TESTS / 'cgaimmd_nanma'


def statuses(report):
    return {check.name: check.status for check in report.checks}


def detail(report, name):
    return ' '.join(check.detail for check in report[name])


# ---------------------------------------------------------------------------
# the toy 1-D C-GAIMMD params
# ---------------------------------------------------------------------------

def test_toy_params_pass(tmp_path, capsys):
    folder = make_campaign(tmp_path / 'toy')
    assert preflight.main([str(folder / 'params.py'), '--candidates', '8']) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1].startswith('preflight OK: ')
    for name in ('params', 'files', 'initial_network', 'generator', 'numerics',
                 'configuration', 'candidate'):
        assert f'OK       {name}: ' in out, name
    assert "generator: 'ode'" in out and 'final check passed (state R' in out
    assert 'SKIPPED  bias: no bias' in out
    # nothing written next to the params (MDAnalysis keeps its hidden offsets files there)
    assert sorted(p.name for p in folder.iterdir() if not p.name.startswith('.')) == \
        ['initial_A.xtc', 'initial_B.xtc', 'initial_network.h5', 'params.py']


def test_module_runs_as_a_command(tmp_path):
    """The command of a job script: ``python -m aimmd.cgaimmd.preflight params.py``."""
    folder = make_campaign(tmp_path / 'toy')
    done = subprocess.run([sys.executable, '-m', 'aimmd.cgaimmd.preflight',
                           str(folder / 'params.py'), '--no-generate'],
                          capture_output=True, text=True, timeout=300,
                          env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'SKIPPED  candidate: not asked for (--no-generate)' in done.stdout
    bad = subprocess.run([sys.executable, '-m', 'aimmd.cgaimmd.preflight',
                          str(folder / 'missing.py')], capture_output=True, text=True,
                         timeout=300)
    assert bad.returncode == 1 and 'FAIL     params:' in bad.stdout


def test_params_that_do_not_load_fail(tmp_path):
    (tmp_path / 'params.py').write_text("chain_type = 'cgaimmd'\nimport not_a_module_here\n")
    report = run_preflight(str(tmp_path / 'params.py'))
    assert not report.ok and statuses(report) == {'params': FAIL}
    assert 'not_a_module_here' in detail(report, 'params')
    assert report.text().splitlines()[-1] == 'preflight FAILED: 1 of 1 checks failed'


def test_initial_network_that_does_not_load_fails(tmp_path):
    from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
    folder = make_campaign(tmp_path / 'toy')
    torch.save(TinyFlowNetwork(hidden=4).state_dict(), folder / 'initial_network.h5')
    report = run_preflight(str(folder / 'params.py'), generate=False)
    assert statuses(report)['initial_network'] == FAIL
    assert 'does not load into params.network' in detail(report, 'initial_network')
    assert not report.ok


TOY_BIAS = '''

record_bias = True
bias_source = 'reader'
BIAS_OFFSET = {offset!r}


def bias_function(trajectory):
    """A wall in A (x < 1), zero elsewhere, plus BIAS_OFFSET (in kT)."""
    x = np.array([ts.positions[0, 0] for ts in trajectory], dtype=float)
    return np.where(x < 1.0, 2.0 * (1.0 - x), 0.0) + BIAS_OFFSET
'''


@pytest.mark.parametrize('offset, status', [(0.0, OK), (0.5, FAIL)])
def test_reader_bias_zero_point(tmp_path, offset, status):
    """A reader-mode bias is evaluated on the frames in R (here the prepared candidate)."""
    folder = make_campaign(tmp_path / 'toy')
    with open(folder / 'params.py', 'a') as handle:
        handle.write(TOY_BIAS.format(offset=offset))
    report = run_preflight(str(folder / 'params.py'), candidates=8)
    assert statuses(report)['candidate'] == OK
    assert statuses(report)['bias zero point'] == status, report.text()
    assert report.ok == (status == OK)
    if status == FAIL:
        assert 'the bias in the reactive region is not zero: 0.5 kT' in detail(report,
                                                                            'bias zero point')


# ---------------------------------------------------------------------------
# the PLUMED checks on stand-in params
# ---------------------------------------------------------------------------

PLUMED_INPUT = '''# a frozen OPES bias, as in the trypsin campaign
d: DISTANCE ATOMS=1,2
opes: OPES_METAD ...
  ARG=d PACE=1000000000 BARRIER=25 RESTART=YES
  STATE_RFILE={state} STATE_WFILE=STATE_OUT
...
PRINT ARG=d,opes.bias FILE={print_file} STRIDE=1
'''


def plumed_params(tmp_path, *, plumed='opes.dat', state='STATE_FROZEN', print_file='COLVAR',
                  write_plumed=True, write_state=True, record_bias=True, mdrun_plumed=True):
    folder = tmp_path / 'campaign'
    folder.mkdir(exist_ok=True)
    state_path = folder / 'STATE_FROZEN'
    if write_state:
        state_path.write_text('#! FIELDS time sigma_0 height\n0 0.1 1\n')
    state_value = str(state_path) if state == 'STATE_FROZEN' else state
    plumed_path = folder / 'opes.dat'
    if write_plumed:
        plumed_path.write_text(PLUMED_INPUT.format(state=state_value, print_file=print_file))
    plumed_value = str(plumed_path) if plumed == 'opes.dat' else plumed
    mdrun = 'gmx mdrun -ntomp 1' + (f' -plumed {plumed_value}' if mdrun_plumed else '')
    return cgaimmd_params_namespace(engine='gromacs', gmx_mdrun=mdrun, record_bias=record_bias,
                                    bias_source='file', bias_function=lambda fname: np.zeros(1),
                                    path=folder / 'params.py')


@pytest.fixture
def no_mdrun(monkeypatch):
    """The executable check passes without running GROMACS."""
    monkeypatch.setattr(preflight, '_plumed_executable', lambda params: (True, 'stand-in'))


def bias_report(params, frames=(), scratch=None):
    report = PreflightReport('params.py')
    preflight._check_bias(report, params, list(frames), scratch)
    return report


def test_plumed_setup_ok(tmp_path, no_mdrun):
    report = bias_report(plumed_params(tmp_path))
    assert statuses(report) == {'plumed': OK, 'plumed executable': OK,
                                'bias zero point': WARNING}
    assert 'STATE_RFILE=' in detail(report, 'plumed')
    assert 'no frame in the reactive region' in detail(report, 'bias zero point')


@pytest.mark.parametrize('case, expected', [
    (dict(write_plumed=False), 'does not exist'),
    (dict(plumed='opes_relative.dat'), 'is relative: mdrun runs in each chain folder'),
    (dict(write_state=False), 'STATE_RFILE=/'),
    (dict(state='STATE_FROZEN_relative'), 'STATE_RFILE=STATE_FROZEN_relative is relative'),
    (dict(print_file='COLVAR_other'), 'no PRINT ... FILE=COLVAR'),
], ids=['missing-file', 'relative-file', 'missing-state', 'relative-state', 'no-colvar'])
def test_plumed_problems_fail(tmp_path, no_mdrun, case, expected):
    report = bias_report(plumed_params(tmp_path, **case))
    assert statuses(report)['plumed'] == FAIL, report.text()
    assert expected in detail(report, 'plumed')
    assert statuses(report)['bias zero point'] == SKIPPED
    assert not report.ok


def test_record_bias_and_plumed_go_together(tmp_path, no_mdrun):
    report = bias_report(plumed_params(tmp_path, mdrun_plumed=False))
    assert statuses(report) == {'plumed': FAIL}
    assert "but gmx_mdrun has no -plumed" in detail(report, 'plumed')
    report = bias_report(plumed_params(tmp_path, record_bias=False))
    assert statuses(report)['bias'] == WARNING and report.ok
    assert 'record_bias is False' in detail(report, 'bias')
    params = cgaimmd_params_namespace(engine='gromacs', gmx_mdrun='gmx mdrun -ntomp 1')
    assert statuses(bias_report(params)) == {'bias': SKIPPED}


def test_mdrun_without_plumed_fails(tmp_path, monkeypatch):
    """The executable check reads ``mdrun -h``: an mdrun without -plumed is not PLUMED-patched."""
    fake = tmp_path / 'gmx_fake'
    fake.write_text('#!/bin/sh\necho "SYNOPSIS gmx mdrun [-s [<.tpr>]] [-deffnm <string>]"\n')
    fake.chmod(0o755)
    params = cgaimmd_params_namespace(engine='gromacs', gmx_mdrun=f'{fake} mdrun -plumed x.dat')
    ok, text = preflight._plumed_executable(params)
    assert not ok and 'has no -plumed option' in text
    fake.write_text('#!/bin/sh\necho " -plumed [<.dat>]  (plumed.dat)  (Opt.)"\n')
    assert preflight._plumed_executable(params)[0]


# ---------------------------------------------------------------------------
# GROMACS and PLUMED: vacuum NANMA under a static wall
# ---------------------------------------------------------------------------

NANMA_PARAMS = '''
import numpy as np
from MDAnalysis.lib.distances import calc_dihedrals
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors

engine = 'gromacs'
chain_type = 'cgaimmd'
states = 'ARB'
topology = 'conf.gro'
gmx_mdp = 'run.mdp'
gmx_grompp = '{gmx} grompp -maxwarn 5 -p topol.top'
gmx_mdrun = '{gmx} mdrun {threads} -nb cpu -pin off -plumed {plumed}'
initial_paths = ['free_md_A.xtc', 'free_md_B.xtc']
initial_network = 'initial_network.h5'
record_bias = True
bias_source = 'file'
cgaimmd_generator = {{'method': 'ode', 'sampler': 'euler', 'n_steps': 2}}
cgaimmd_parameters = {{'placement': 'none', 'lambda_clip': 10.0,
                      'window_after_trained_shots': 5, 'generation_device': 'cpu',
                      'values_device': 'cpu', 'training_device': 'cpu'}}

DESCRIPTORS = GeneratorDescriptors('conf.gro', GeneratorAtoms(n_solute=22))

def descriptors_function(trajectory):
    return DESCRIPTORS(trajectory)

PHI, PSI = [4, 6, 8, 14], [6, 8, 14, 16]

def states_function(trajectory):
    """NANMA's discs, A narrowed to 10 degrees so that some frames of free_md_A are in R."""
    out = []
    for frame in trajectory:
        p = frame.positions.astype(np.float64)
        phi, psi = np.degrees([calc_dihedrals(p[a], p[b], p[c], p[d])
                               for a, b, c, d in (PHI, PSI)])
        da = np.hypot((phi + 83 + 180) % 360 - 180, (psi - 73 + 180) % 360 - 180)
        db = np.hypot((phi - 71 + 180) % 360 - 180, (psi + 71 + 180) % 360 - 180)
        out.append('A' if da <= 10 else 'B' if db <= 20 else 'R')
    return np.array(out)

class Network(TinyFlowNetwork):
    def __init__(self):
        super().__init__(n_atoms=22)

network = Network()

BIAS_OFFSET = {offset!r}         # kJ/mol

def bias_function(fname):
    """The wall's bias from <traj>_COLVAR (column 4), in kT at 300 K."""
    data = np.loadtxt(fname[:-len('.xtc')] + '_COLVAR', comments='#', ndmin=2)
    return (data[:, 3] + BIAS_OFFSET) / 2.494339
'''

NANMA_PLUMED = '''phi: TORSION ATOMS=5,7,9,15
psi: TORSION ATOMS=7,9,15,17
wall: UPPER_WALLS ARG=psi AT=3.0 KAPPA=50.0
PRINT ARG=phi,psi,wall.bias FILE=COLVAR STRIDE=1
'''


def nanma_campaign(folder, gmx, *, offset=0.0, write_plumed=True):
    """Vacuum NANMA C-GAIMMD params whose shots run under a static PLUMED wall (psi > 3 rad)."""
    import shutil
    from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
    from tests._cgaimmd_helpers import mdrun_thread_options
    folder = Path(folder)
    folder.mkdir(parents=True)
    for name in ('conf.gro', 'topol.top', 'run.mdp', 'free_md_A.xtc', 'free_md_B.xtc'):
        shutil.copy(NANMA / name, folder / name)
    plumed = folder / 'wall.dat'
    if write_plumed:
        plumed.write_text(NANMA_PLUMED)
    (folder / 'params.py').write_text(NANMA_PARAMS.format(
        gmx=gmx, threads=mdrun_thread_options(gmx, 1), plumed=plumed, offset=offset))
    torch.manual_seed(0)
    torch.save(TinyFlowNetwork(n_atoms=22).state_dict(), folder / 'initial_network.h5')
    return folder


@pytest.mark.gromacs
@pytest.mark.parametrize('offset, status', [(0.0, OK), (25.0, FAIL)])
def test_nanma_plumed_zero_point(tmp_path, gmx_executable, offset, status):
    """The zero point as the shots see it: the initial paths' R frames are made .tpr files
    and run for zero steps with -plumed; a bias shifted by 25 kJ/mol (a forgotten OPES
    BARRIER) fails."""
    folder = nanma_campaign(tmp_path / 'nanma', gmx_executable, offset=offset)
    scratch = tmp_path / 'scratch'
    report = run_preflight(str(folder / 'params.py'), generate=False, scratch=str(scratch))
    found = statuses(report)
    for name in ('params', 'files', 'initial_network', 'generator', 'numerics',
                 'configuration', 'gromacs', 'plumed', 'plumed executable'):
        assert found[name] == OK, report.text()
    assert found['bias zero point'] == status, report.text()
    text = detail(report, 'bias zero point')
    if status == OK:
        assert 'step 0 of mdrun -plumed' in text
        # PLUMED evaluated the frames: their COLVAR next to each frame's run
        assert sorted(p.name for p in (scratch / 'bias').glob('*_COLVAR'))
    else:
        assert 'shift it in bias_function' in text and '10' in text
    assert report.ok == (status == OK)
    assert not (folder / 'COLVAR').exists()                         # nothing next to the params


@pytest.mark.gromacs
def test_nanma_missing_plumed_file_fails(tmp_path, gmx_executable, capsys):
    folder = nanma_campaign(tmp_path / 'nanma', gmx_executable, write_plumed=False)
    assert preflight.main([str(folder / 'params.py'), '--no-generate']) == 1
    out = capsys.readouterr().out
    assert f"FAIL     plumed: the -plumed file '{folder / 'wall.dat'}' does not exist" in out
    assert 'SKIPPED  bias zero point: not checked' in out
    assert out.strip().splitlines()[-1].startswith('preflight FAILED: 1 of ')


def test_plumed_input_files_follow_includes(tmp_path):
    """The files a PLUMED input reads, followed into INCLUDE files; comments and outputs ignored."""
    from aimmd.cgaimmd.plumed import (mdrun_executable, plumed_file_of, plumed_input_files,
                                      print_files)
    (tmp_path / 'grid.dat').write_text('grid')
    (tmp_path / 'inner.dat').write_text('m: METAD ARG=d GRID_RFILE=grid.dat FILE=HILLS '
                                        '# STATE_RFILE=commented\n')
    (tmp_path / 'main.dat').write_text('# INCLUDE FILE=commented.dat\nINCLUDE FILE=inner.dat\n'
                                       'PRINT ARG=d FILE=OUT\nPRINT ARG=d\n')
    files = plumed_input_files(str(tmp_path / 'main.dat'))
    assert [(entry['keyword'], entry['written'], entry['exists'], entry['relative'])
            for entry in files] == [('INCLUDE FILE', 'inner.dat', True, True),
                                    ('GRID_RFILE', 'grid.dat', True, True)]
    assert files[1]['path'] == str(tmp_path / 'grid.dat') and files[1]['sha256']
    assert print_files(str(tmp_path / 'main.dat')) == ['OUT', 'COLVAR']
    assert plumed_file_of('srun -p queue gmx_mpi mdrun -ntomp 2 -plumed /a/b.dat') == '/a/b.dat'
    assert plumed_file_of('gmx mdrun -ntomp 2') is None
    assert mdrun_executable('srun -n 1 gmx_mpi mdrun -ntomp 2') == 'srun -n 1 gmx_mpi mdrun'
    assert mdrun_executable('mdrun_mpi -ntomp 2') == 'mdrun_mpi'
    assert mdrun_executable('python run.py') is None
