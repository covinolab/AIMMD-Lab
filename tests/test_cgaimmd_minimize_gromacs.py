"""GromacsMinimizer with GROMACS on NANMA in vacuum.

Also verifies the GROMACS behaviour the minimizer relies on: with
``nstxout = 0`` the ``.trr`` of an EM holds exactly the final minimized frame.
GROMACS is the ``gmx_executable`` of ``tests/conftest.py`` (marker
``gromacs``): skipped without GROMACS.
"""

import glob
import os
import time
from pathlib import Path, PosixPath

import numpy as np
import pytest

mda = pytest.importorskip('MDAnalysis')

import aimmd                                                                 # noqa: E402
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork                  # noqa: E402
from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors  # noqa: E402
from aimmd.cgaimmd.preparation.gromacs import _read_mdp, mdp_text, read_gro  # noqa: E402
from aimmd.cgaimmd.preparation.minimize import GromacsMinimizer              # noqa: E402
from aimmd.cgaimmd.preparation.topology import read_masses                   # noqa: E402
from aimmd.cgaimmd.types import PreparedFrame                                # noqa: E402
from tests._cgaimmd_helpers import (cgaimmd_params_namespace, make_candidate_context,  # noqa: E402
                                    mdrun_thread_options)

pytestmark = pytest.mark.gromacs

NANMA = Path(__file__).resolve().parent / 'cgaimmd_nanma'


def _params(tmp_path, gmx):
    universe = mda.Universe(str(NANMA / 'conf.gro'))
    return cgaimmd_params_namespace(
        engine='gromacs', topology=str(NANMA / 'conf.gro'), gmx_mdp=str(NANMA / 'run.mdp'),
        gmx_grompp=f'{gmx} grompp -maxwarn 5 -p {NANMA / "topol.top"}',
        gmx_mdrun=f'{gmx} mdrun {mdrun_thread_options(gmx, 1)} -nb cpu -pin off',
        network=TinyFlowNetwork(n_atoms=22), _universe=universe, path=PosixPath(tmp_path))


def _candidate(index=3):
    """An unbiased free-MD frame (a stand-in generated sample), whole, centered in the box."""
    universe = mda.Universe(str(NANMA / 'conf.gro'), str(NANMA / 'free_md_A.xtc'))
    universe.trajectory[index]
    descriptors = GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(22))
    x = descriptors.select(universe.atoms.positions, universe.dimensions)
    box = np.asarray(universe.dimensions, dtype=float)
    return PreparedFrame(positions=x - x.mean(0) + box[:3] / 2, dimensions=box)


def _context(tmp_path, params, **settings):
    return make_candidate_context(tmp_path, params=params, **settings)


def _restrained_atoms(text):
    """1-based atoms of the AIMMD_CGAIMMD_POSRES position restraints of a derived topology."""
    atoms, inside = [], False
    for line in text.splitlines():
        body = line.split(';')[0].strip()
        if body == '#ifdef AIMMD_CGAIMMD_POSRES':
            inside = True
        elif body == '#endif':
            inside = False
        elif inside and body and body.split()[0].isdigit():
            atoms.append(int(body.split()[0]))
    return atoms


def test_restrained_em_vacuum(tmp_path, gmx_executable):
    """Restrained EM lowers the energy and keeps the heavy atoms close to the candidate."""
    params = _params(tmp_path, gmx_executable)
    context = _context(tmp_path, params)
    frame = _candidate()
    result = GromacsMinimizer(em_mdp=str(NANMA / 'em.mdp'))(frame, context)
    assert result.ok, result.reason
    assert result.reason == '' and result.positions.shape == (22, 3)
    record = result.record
    for key in ('settings', 'epot_before', 'epot_after', 'fmax', 'steps', 'converged',
                'max_displacement_A', 'heavy_rms_displacement_A', 'seconds'):
        assert key in record, key
    assert record['epot_after'] < record['epot_before']
    assert record['max_displacement_A'] < 0.3 and record['heavy_rms_displacement_A'] < 0.1
    assert np.abs(result.positions - frame.positions).max() > 1e-3        # it did minimize
    np.testing.assert_array_equal(result.dimensions, frame.dimensions)
    assert os.listdir(context.scratch) == []                              # scratch removed
    # unrestrained EM moves the heavy atoms further
    free = GromacsMinimizer(em_mdp=str(NANMA / 'em.mdp'), restrain='none')(frame, context)
    assert free.ok and free.record['max_displacement_A'] > record['max_displacement_A']
    # the displacement limit rejects
    strict = GromacsMinimizer(em_mdp=str(NANMA / 'em.mdp'), restrain='none',
                              max_displacement=1e-4)(frame, context)
    assert not strict.ok and strict.reason.startswith(
        'minimize_failed: a generator heavy atom moved')


def test_final_frame_read(tmp_path, gmx_executable):
    """The .trr of the EM holds one frame, the final minimized one (= em.gro)."""
    from MDAnalysis.lib.formats.libmdaxdr import TRRFile
    params = _params(tmp_path, gmx_executable)
    context = _context(tmp_path, params, keep_scratch=True)
    frame = _candidate(5)
    result = GromacsMinimizer(em_mdp=str(NANMA / 'em.mdp'))(frame, context)
    assert result.ok, result.reason
    workdir = Path(result.record['directory'])
    with TRRFile(str(workdir / 'em.trr')) as handle:
        frames = [f for f in handle]
    assert len(frames) == 1
    final, _ = read_gro(workdir / 'em.gro')
    assert np.abs(result.positions - final).max() < 0.006                # em.gro: 3 decimals
    start, _ = read_gro(workdir / 'start.gro')
    assert np.abs(start - frame.positions).max() < 1e-4                  # 6 decimals in
    assert (workdir / 'restrained.top').exists() and (workdir / 'ref.gro').exists()
    assert '-DAIMMD_CGAIMMD_POSRES' in (workdir / 'em.mdp').read_text()
    # 'generator_heavy' restrains exactly the 10 heavy atoms of NANMA, 'generator_all' all 22
    heavy = [int(i) + 1 for i in np.flatnonzero(read_masses(NANMA / 'topol.top') > 1.5)]
    assert len(heavy) == 10
    assert _restrained_atoms((workdir / 'restrained.top').read_text()) == heavy
    every = GromacsMinimizer(em_mdp=str(NANMA / 'em.mdp'), restrain='generator_all')(frame,
                                                                                      context)
    assert every.ok, every.reason
    text = (Path(every.record['directory']) / 'restrained.top').read_text()
    assert _restrained_atoms(text) == list(range(1, 23))


def test_nstxout_override(tmp_path, gmx_executable):
    """nstxout = 0 is set whatever the template says, so the .trr holds the final frame only."""
    from MDAnalysis.lib.formats.libmdaxdr import TRRFile
    template = tmp_path / 'em_every_step.mdp'
    template.write_text((NANMA / 'em.mdp').read_text() + 'nstxout = 1\n')
    context = _context(tmp_path, _params(tmp_path, gmx_executable), keep_scratch=True)
    result = GromacsMinimizer(em_mdp=str(template))(_candidate(2), context)
    assert result.ok, result.reason
    assert result.record['steps'] >= 2                  # the template alone would write frames
    workdir = Path(result.record['directory'])
    assert _read_mdp(workdir / 'em.mdp')['nstxout'] == '0'
    with TRRFile(str(workdir / 'em.trr')) as handle:
        assert len([f for f in handle]) == 1


def test_explicit_mdrun_with_plumed_minimizes_unbiased(tmp_path, gmx_executable):
    """An explicit mdrun command with -plumed: the EM runs without the bias.

    The PLUMED input restrains the CL-CA distance far from its value and prints it every
    step; the EM with that command must write no PLUMED output and give the frame of the
    EM without -plumed. (A GROMACS without PLUMED would fail on -plumed.)
    """
    plumed = tmp_path / 'bias.dat'
    plumed.write_text('d: DISTANCE ATOMS=1,9\n'
                      'RESTRAINT ARG=d AT=0.6 KAPPA=100000.0\n'
                      'PRINT ARG=d FILE=COLVAR_preparation_bias STRIDE=1\n')
    params = _params(tmp_path, gmx_executable)
    context = _context(tmp_path, params, keep_scratch=True)
    frame = _candidate(4)
    plain = GromacsMinimizer(em_mdp=str(NANMA / 'em.mdp'), mdrun=params.gmx_mdrun)(frame, context)
    biased = GromacsMinimizer(em_mdp=str(NANMA / 'em.mdp'),
                              mdrun=f'{params.gmx_mdrun} -plumed {plumed}')(frame, context)
    assert plain.ok and biased.ok, (plain.reason, biased.reason)
    assert list(tmp_path.rglob('COLVAR_preparation_bias*')) == []
    np.testing.assert_allclose(biased.positions, plain.positions, atol=1e-5)


def test_must_stop_kills_mdrun(tmp_path, gmx_executable):
    """must_stop terminates a running mdrun (not grompp): a failed candidate, returned promptly.

    must_stop fires 1 s after mdrun has written its log, and the "minimization" is a 10^8-step
    MD run (the production mdp as template), so it is mdrun that is stopped; an EM with emtol = 0
    is not enough, NANMA reaches machine precision in under a second.
    """
    params = _params(tmp_path, gmx_executable)
    context = _context(tmp_path, params)
    seen = {}

    def must_stop():
        if not glob.glob(os.path.join(context.scratch, 'em_*', 'em.log')):
            return False
        seen.setdefault('log', time.monotonic())
        return time.monotonic() - seen['log'] > 1.0

    context.must_stop = must_stop
    slow = GromacsMinimizer(em_mdp=str(NANMA / 'run.mdp'), restrain='none', nsteps=10 ** 8,
                            timeout=60.0)
    start = time.monotonic()
    result = slow(_candidate(), context)
    elapsed = time.monotonic() - start
    assert not result.ok and result.reason.startswith('minimize_failed: ')
    # the reason names the executable as the command does (gmx, gmx_mpi)
    assert f'{os.path.basename(gmx_executable)} mdrun stopped by must_stop' in result.reason, (
        result.reason)
    assert 'log' in seen and 1.0 < elapsed < 30.0, elapsed


def test_unrestrained_equals_params_minimize_energy(tmp_path, gmx_executable, monkeypatch):
    """GromacsMinimizer(restrain='none') and Params.minimize_energy give the same frame."""
    from MDAnalysis.coordinates.TRR import TRRWriter
    monkeypatch.chdir(tmp_path)                 # minimize_energy's grompp writes mdout.mdp here
    params = _params(tmp_path, gmx_executable)
    em = tmp_path / 'em_same.mdp'
    em.write_text(mdp_text(NANMA / 'em.mdp', {'emtol': 100.0, 'nsteps': 1000, 'nstxout': 0}))
    frame = _candidate(7)
    positions = frame.positions.astype(np.float32).astype(np.float64)   # what a trr holds
    universe = mda.Universe.empty(22, trajectory=True)
    universe.trajectory.ts.positions = positions
    universe.trajectory.ts.dimensions = frame.dimensions
    with TRRWriter(str(tmp_path / 'frame.trr'), 22) as writer:
        writer.write(universe.atoms)
    params.minimize_energy(str(tmp_path / 'frame.trr'), str(tmp_path / 'minimized.trr'),
                           em_mdp=str(em))
    reference = aimmd.Path(str(tmp_path / 'minimized.trr')).positions[0]
    context = _context(tmp_path, params)
    result = GromacsMinimizer(em_mdp=str(em), restrain='none', emtol=100.0, nsteps=1000)(
        PreparedFrame(positions=positions, dimensions=frame.dimensions), context)
    assert result.ok, result.reason
    rms = np.sqrt(((result.positions - reference) ** 2).sum(-1).mean())
    assert rms < 0.01, rms
    assert np.abs(result.positions - positions).max() > 0.01              # both minimized
