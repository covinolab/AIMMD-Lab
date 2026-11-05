"""GROMACS plumbing, minimizer and solvator logic without running GROMACS.

Everything that needs no GROMACS executable: command surgery, mdp files,
``.gro``/``.trr``/log files, ``run_gmx`` on shell commands, and the steps of
the solvator that are pure geometry (plan, selection consistency, ion
candidates). The GROMACS runs are in ``test_cgaimmd_minimize_gromacs.py`` and
``test_cgaimmd_solvate_gromacs.py`` (marker ``gromacs``).
"""

import inspect
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mda = pytest.importorskip('MDAnalysis')

from aimmd.cgaimmd.preparation import gromacs as G                                     # noqa: E402
from aimmd.cgaimmd.preparation.descriptors import ClusterSelection, GeneratorAtoms, GeneratorDescriptors  # noqa: E402
from aimmd.cgaimmd.preparation.minimize import GromacsMinimizer                        # noqa: E402
from aimmd.cgaimmd.preparation.solvate import ExclusionSphere, GromacsSolvator, NVTStage  # noqa: E402

L = 30.0
BOX = np.array([L, L, L, 90.0, 90.0, 90.0])


def _min_image(v):
    return v - L * np.round(v / L)


TOP = """\
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
  CT   6  12.011  0.0  A  0.35  0.27
  HC   1   1.008  0.0  A  0.25  0.12
  OW   8  15.999  0.0  A  0.315 0.636
  HW   1   1.008  0.0  A  0.0   0.0
  NA   11 22.990  0.0  A  0.24  0.20
  CLA  17 35.450  0.0  A  0.40  0.60

[ moleculetype ]
LIG 3
[ atoms ]
 1  CT  1  LIG  C1   1  -0.20  12.011
 2  CT  1  LIG  C2   2   0.10  12.011
 3  HC  1  LIG  H3   3   0.10   1.008
[ bonds ]
 1 2 1
 2 3 1

[ moleculetype ]
SOL 2
[ atoms ]
 1  OW  1  SOL  OW   1  -0.834
 2  HW  1  SOL  HW1  1   0.417
 3  HW  1  SOL  HW2  1   0.417
[ settles ]
 1  1  0.09572  0.15139

[ moleculetype ]
NA 1
[ atoms ]
 1  NA  1  NA  NA  1  1.0

[ moleculetype ]
CL 1
[ atoms ]
 1  CLA 1  CL  CL  1  -1.0

[ moleculetype ]
K 1
[ atoms ]
 1  NA  1  K   K   1  1.0

[ system ]
test
[ molecules ]
"""


def _system(tmp_path, n_waters=40, ions=(('NA', 1), ('CL', 1)), seed=0, extra_molecules=''):
    """topol.top + sys.gro: LIG, SOL waters, then the ions (positions random)."""
    rng = np.random.default_rng(seed)
    molecules = f'LIG 1\nSOL {n_waters}\n' + ''.join(f'{n} {c}\n' for n, c in ions if c)
    (tmp_path / 'topol.top').write_text(TOP + molecules + extra_molecules)
    names, resnames, resids = ['C1', 'C2', 'H3'], ['LIG'] * 3, [1] * 3
    positions = [[15.0, 15.0, 15.0], [16.5, 15.0, 15.0], [17.0, 15.9, 15.0]]
    for k, o in enumerate(rng.uniform(0.0, L, size=(n_waters, 3))):
        positions += [o, o + [0.96, 0, 0], o + [-0.24, 0.93, 0]]
        names += ['OW', 'HW1', 'HW2']
        resnames += ['SOL'] * 3
        resids += [2 + k] * 3
    for name, count in ions:
        for _ in range(count):
            positions.append(rng.uniform(0.0, L, size=3))
            names.append(name)
            resnames.append(name)
            resids.append(resids[-1] + 1)
    atoms = SimpleNamespace(names=names, resnames=resnames, resids=resids)
    G.write_gro(tmp_path / 'sys.gro', atoms, np.asarray(positions) % L, BOX)
    return tmp_path / 'topol.top', tmp_path / 'sys.gro'


def test_split_grompp():
    """The topology option is removed and returned; the tools' own options are dropped."""
    assert G.split_grompp('/opt/gmx grompp -maxwarn 5 -p topol.top') == (
        '/opt/gmx grompp -maxwarn 5', 'topol.top')
    assert G.split_grompp('gmx grompp -maxwarn 5') == ('gmx grompp -maxwarn 5', None)
    assert G.split_grompp("gmx grompp -p '/a dir/t.top' -po x.mdp") == (
        'gmx grompp -po x.mdp', '/a dir/t.top')
    assert G._grompp_command('gmx grompp -maxwarn 5 -po /tmp/x.mdp -nobackup -f a.mdp') == (
        'gmx grompp -maxwarn 5')
    assert G._grompp_command('gmx grompp -maxwarn 5 -f a.mdp', maxwarn=2) == 'gmx grompp'
    # a launcher prefix keeps its options: srun's -p is its partition, -n its task count
    assert G.split_grompp('srun -p short -n 1 gmx_mpi grompp -p topol.top -maxwarn 2') == (
        'srun -p short -n 1 gmx_mpi grompp -maxwarn 2', 'topol.top')
    # an index file is kept (every grompp of the full system can use it), except on request
    assert G._grompp_command('srun -n 1 gmx_mpi grompp -n index.ndx -f a.mdp') == (
        'srun -n 1 gmx_mpi grompp -n index.ndx')
    assert G._grompp_command('gmx grompp -n index.ndx -maxwarn 2', index=False) == (
        'gmx grompp -maxwarn 2')
    assert G._with_absolute_path("gmx grompp -n idx.ndx -maxwarn 2", '-n', '/a dir') == (
        "gmx grompp -n '/a dir/idx.ndx' -maxwarn 2")
    assert G._with_absolute_path('gmx grompp -n /x/i.ndx', '-n', '/y') == 'gmx grompp -n /x/i.ndx'
    assert G._params_grompp(SimpleNamespace(gmx_grompp='gmx grompp -p t.top -n i.ndx',
                                            parent='/run')) == 'gmx grompp -n /run/i.ndx'
    # a file option given without its file takes no value; -1 is a value
    assert G.split_grompp('gmx grompp -p -maxwarn 2') == ('gmx grompp -maxwarn 2', None)
    assert G._grompp_command('gmx grompp -t -n -maxwarn 1 -r -nobackup') == (
        'gmx grompp -n -maxwarn 1')


def test_sanitize_mdrun():
    """Run-control, checkpoint and PLUMED options go; EM also drops the unsupported offloads."""
    command = ('/opt/gmx mdrun -ntmpi 1 -ntomp 2 -plumed plumed.dat -cpi state.cpt -cpt 5 -maxh 2 '
               '-deffnm md -noappend -pin off -nb gpu -pme gpu')
    assert G.sanitize_mdrun(command) == '/opt/gmx mdrun -ntmpi 1 -ntomp 2 -pin off -nb gpu -pme gpu'
    assert G._mdrun_command(G.sanitize_mdrun(command), minimization=True) == (
        '/opt/gmx mdrun -ntmpi 1 -ntomp 2 -pin off -nb gpu')
    assert G._mdrun_command('gmx mdrun -s a.tpr -nobackup -c x.gro -update gpu') == (
        'gmx mdrun -update gpu')
    assert G.sanitize_mdrun('bash mdrun.sh') == 'bash mdrun.sh'
    # shell syntax and quoting survive
    assert G._mdrun_command("sleep 1 && OMP_NUM_THREADS=2 gmx mdrun -g 'my log.log' -ntomp 2") == (
        'sleep 1 && OMP_NUM_THREADS=2 gmx mdrun -ntomp 2')
    assert G.sanitize_mdrun("bash '/my dir/mdrun.sh' -maxh 3 -ntomp 2") == (
        "bash '/my dir/mdrun.sh' -ntomp 2")
    # launcher prefixes keep their options (srun -c is cpus-per-task, mpirun -x an env var)
    assert G._mdrun_command(G.sanitize_mdrun('srun -n 1 -c 8 gmx_mpi mdrun -ntomp 8 -maxh 2'),
                            minimization=True) == 'srun -n 1 -c 8 gmx_mpi mdrun -ntomp 8'
    assert G._mdrun_command('mpirun -x OMP_NUM_THREADS -np 4 gmx_mpi mdrun -s a.tpr -x t.xtc') == (
        'mpirun -x OMP_NUM_THREADS -np 4 gmx_mpi mdrun')
    # every shell segment is judged by its own GROMACS word; no such word: unchanged
    assert G.sanitize_mdrun('cd /x && gmx mdrun -cpi s.cpt -ntomp 2 | tee -a log') == (
        'cd /x && gmx mdrun -ntomp 2 | tee -a log')
    assert G.sanitize_mdrun("bash -c 'x y' -maxh 3") == "bash -c 'x y' -maxh 3"
    assert G.sanitize_mdrun('gmx mdrun -ntomp 2 && cp -r a b') == 'gmx mdrun -ntomp 2 && cp -r a b'
    # a file option without its file, and -nsteps (it would override the mdp's nsteps)
    assert G.sanitize_mdrun('gmx mdrun -cpi -ntomp 4 -pin on') == 'gmx mdrun -ntomp 4 -pin on'
    assert G._mdrun_command('gmx mdrun -nsteps -1 -ntomp 2 -nsteps 500') == 'gmx mdrun -ntomp 2'


def test_mdp_text(tmp_path):
    """Overridden keys replaced, missing ones appended, define/include appended, duplicates superseded."""
    template = tmp_path / 'run.mdp'
    template.write_text(textwrap.dedent("""\
        ; production
        integrator          = md
        define              = -DPOSRES
        nsteps              = 100      ; comment kept
        nstxout-compressed  = 50
        tc-grps             = System
        nsteps              = 200      ; a duplicate (GROMACS would take this one)
        """))
    text = G.mdp_text(template, {'define': '-DAIMMD_CGAIMMD_POSRES', 'NSTEPS': 5,
                                 'nstxout_compressed': 0, 'include': '-I/a/b', 'gen_vel': True,
                                 'ref_t': (300, 310), 'dt': 0.002},
                      header='a derived file\nsecond line')
    lines = text.splitlines()
    assert lines[0] == '; a derived file' and lines[1] == '; second line'
    values = {}
    for line in lines:
        body = line.split(';', 1)[0]
        if '=' in body:
            key, value = body.split('=', 1)
            values[key.strip()] = value.strip()
    assert values['define'] == '-DPOSRES -DAIMMD_CGAIMMD_POSRES'
    assert values['NSTEPS'] == '5' and 'nsteps' not in values
    assert values['nstxout_compressed'] == '0'
    assert values['integrator'] == 'md' and values['tc-grps'] == 'System'
    assert values['include'] == '-I/a/b' and values['gen_vel'] == 'yes'
    assert values['ref_t'] == '300 310' and values['dt'] == '0.002'
    set_here = [line for line in lines if '; set here' in line]
    assert len(set_here) == 7
    superseded = [line for line in lines if 'superseded' in line]
    assert len(superseded) == 1 and superseded[0].startswith('; nsteps') and '200' in superseded[0]
    assert G._read_mdp(template)['nstxout_compressed'] == '50'
    assert G._read_mdp(template)['nsteps'] == '200'                     # the last entry wins
    # an include of the template is kept and extended, like a define
    template.write_text('integrator = steep\ninclude = -I/x\n')
    assert _mdp_values(G.mdp_text(template, {'include': '-I/a/b'}))['include'] == '-I/x -I/a/b'


def test_gmx_executable_and_topology_file(tmp_path, monkeypatch):
    """The executable (path, PATH lookup or AIMMD's), the topology, seeds and work directories."""
    params = SimpleNamespace(gmx_grompp='/opt/gromacs/bin/gmx grompp -p sys/topol.top',
                             parent=tmp_path)
    assert G.gmx_executable(params) == '/opt/gromacs/bin/gmx'
    assert G.topology_file(params) == str(tmp_path / 'sys' / 'topol.top')
    assert G.topology_file(SimpleNamespace(gmx_grompp='gmx grompp', parent=tmp_path)) == str(
        tmp_path / 'topol.top')
    absolute = SimpleNamespace(gmx_grompp='gmx grompp -p /x/t.top', parent=tmp_path)
    assert G.topology_file(absolute) == '/x/t.top'
    import aimmd._config as config
    monkeypatch.setattr(config, 'GROMACS', '/from/config/gmx', raising=False)
    assert G.gmx_executable(SimpleNamespace(gmx_grompp='bash grompp.sh')) == '/from/config/gmx'
    # a bare name is resolved on PATH; one that is not on PATH is never returned bare
    monkeypatch.setattr(G.shutil, 'which',
                        lambda name: '/on/path/gmx_mpi' if name == 'gmx_mpi' else None)
    assert G.gmx_executable(SimpleNamespace(gmx_grompp='gmx_mpi grompp')) == '/on/path/gmx_mpi'
    assert G.gmx_executable(SimpleNamespace(gmx_grompp='gmx_nonexistent grompp')) == (
        '/from/config/gmx')
    monkeypatch.setattr(config, 'GROMACS', None)
    with pytest.raises(RuntimeError):
        G.gmx_executable(SimpleNamespace(gmx_grompp='bash grompp.sh'))
    with pytest.raises(RuntimeError, match='PATH'):
        G.gmx_executable(SimpleNamespace(gmx_grompp='gmx_nonexistent grompp'))
    assert G._gromacs_seed(0) == 1 and G._gromacs_seed(2 ** 31 - 2) == 2 ** 31 - 1
    assert 1 <= G._gromacs_seed(2 ** 62 + 17) < 2 ** 31
    first, second = G._work_dir(tmp_path / 'scratch', 'em'), G._work_dir(tmp_path / 'scratch', 'em')
    assert first.name == 'em_0000' and second.name == 'em_0001' and first.is_dir()


def test_write_read_gro(tmp_path):
    """.gro files at 3 and 6 decimals, orthorhombic, triclinic and zero boxes, read back."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 30, size=(5, 3))
    atoms = SimpleNamespace(names=['C', 'N', 'O', 'H', 'S'], resnames=['R'] * 5, resids=[1] * 5)
    G.write_gro(tmp_path / 'a.gro', atoms, x, BOX)
    y, dims = G.read_gro(tmp_path / 'a.gro')
    assert np.abs(y - x).max() <= 0.005 + 1e-9
    np.testing.assert_allclose(dims, BOX)
    universe = mda.Universe(str(tmp_path / 'a.gro'))                # standard precision
    np.testing.assert_allclose(universe.atoms.positions, y, atol=1e-4)
    assert list(universe.atoms.names) == atoms.names
    G.write_gro(tmp_path / 'b.gro', atoms, x, BOX, decimals=6)
    z, _ = G.read_gro(tmp_path / 'b.gro')
    assert np.abs(z - x).max() <= 5e-6 + 1e-12
    resnames, names, _, _ = G._gro_records(tmp_path / 'b.gro')
    assert names == atoms.names and resnames == ['R'] * 5
    G.write_gro(tmp_path / 'c.gro', atoms, x, None)
    assert G.read_gro(tmp_path / 'c.gro')[1][:3].tolist() == [0.0, 0.0, 0.0]
    # a triclinic box: nine box numbers, the same dimensions for read_gro and MDAnalysis
    triclinic = np.array([30.0, 32.0, 34.0, 70.0, 80.0, 60.0])
    G.write_gro(tmp_path / 't.gro', atoms, x, triclinic)
    assert len((tmp_path / 't.gro').read_text().splitlines()[-1].split()) == 9
    np.testing.assert_allclose(G.read_gro(tmp_path / 't.gro')[1], triclinic, atol=1e-3)
    np.testing.assert_allclose(mda.Universe(str(tmp_path / 't.gro')).dimensions, triclinic,
                               atol=1e-3)
    with pytest.raises(ValueError):
        G.write_gro(tmp_path / 'd.gro', atoms, x[:4], BOX)


def test_read_trr_last_frame(tmp_path):
    """The last frame with coordinates and its box; a missing file raises."""
    from MDAnalysis.coordinates.TRR import TRRWriter
    universe = mda.Universe.empty(4, trajectory=True)
    frames = [np.full((4, 3), float(k)) for k in range(3)]
    with TRRWriter(str(tmp_path / 'x.trr'), 4) as writer:
        for k, frame in enumerate(frames):
            universe.trajectory.ts.positions = frame
            universe.trajectory.ts.dimensions = [20.0 + k, 20.0, 20.0, 90.0, 90.0, 90.0]
            writer.write(universe.atoms)
    x, dims = G.read_trr_last_frame(tmp_path / 'x.trr')
    np.testing.assert_allclose(x, frames[-1], atol=1e-5)
    np.testing.assert_allclose(dims, [22.0, 20.0, 20.0, 90.0, 90.0, 90.0], atol=1e-4)
    with pytest.raises(OSError):
        G.read_trr_last_frame(tmp_path / 'missing.trr')


EM_LOG = """\
Steepest Descents:
   Tolerance (Fmax)   =  1.00000e+02
   Number of steps    =          200
           Step           Time
              0        0.00000

   Energies (kJ/mol)
           Bond            U-B    Proper Dih.  Improper Dih.          LJ-14
    1.89699e+01    4.09036e+01    3.30869e+01    6.79229e+00    6.52644e+00
     Coulomb-14        LJ (SR)   Coulomb (SR)      Potential Pressure (bar)
    2.22879e+02   -3.05330e-01   -3.18562e+02    1.02911e+01    4.28607e+00

           Step           Time
              1        1.00000

   Energies (kJ/mol)
           Bond            U-B    Proper Dih.  Improper Dih.          LJ-14
    1.00000e+01    4.00000e+01    3.00000e+01    6.00000e+00    6.00000e+00
     Coulomb-14        LJ (SR)   Coulomb (SR)      Potential Pressure (bar)
    2.22000e+02   -3.00000e-01   -3.18000e+02    5.00000e+00    4.00000e+00

Steepest Descents converged to Fmax < 100 in 44 steps
Potential Energy  = -4.4491333e+01
Maximum force     =  9.2552406e+01 on atom 13
Norm of force     =  4.8123295e+01
"""


def test_parse_em_log(tmp_path):
    """Final and first potential, Fmax, steps and the three convergence outcomes."""
    log = tmp_path / 'em.log'
    log.write_text(EM_LOG)
    assert G.parse_em_log(log) == {'epot': -44.491333, 'fmax': 92.552406, 'steps': 44,
                                   'converged': True, 'epot_start': 10.2911}
    log.write_text(EM_LOG.replace('converged to Fmax < 100 in 44 steps',
                                  'converged to machine precision in 36 steps,\n'
                                  'but did not reach the requested Fmax < 10.'))
    assert G.parse_em_log(log)['converged'] is False and G.parse_em_log(log)['steps'] == 36
    log.write_text(EM_LOG.replace('converged to Fmax < 100 in 44 steps',
                                  'did not converge to Fmax < 100 in 201 steps.'))
    assert G.parse_em_log(log)['converged'] is False and G.parse_em_log(log)['steps'] == 201
    log.write_text('nothing\n')
    assert G.parse_em_log(log) == {'epot': None, 'fmax': None, 'steps': None, 'converged': None,
                                   'epot_start': None}


# execute_command leaves the pipe of a stopped process open (a ResourceWarning at garbage
# collection); a pre-existing issue of aimmd.execute, to be fixed in the bug-fix stream
@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
def test_run_gmx(tmp_path):
    """Output and log, stdin, the working directory, failures, must_stop and the timeout."""
    log = tmp_path / 'run.log'
    out = G.run_gmx('echo hello', tmp_path, log)
    assert 'hello' in out and 'hello' in log.read_text()
    assert 'piped' in G.run_gmx('cat', tmp_path, log, stdin='piped\n')
    G.run_gmx('pwd > where.txt', tmp_path, log)
    assert (tmp_path / 'where.txt').read_text().strip() == str(tmp_path)
    with pytest.raises(RuntimeError, match='failed') as info:
        G.run_gmx(f'{sys.executable} -c "print(1); print(2); raise SystemExit(3)"', tmp_path, log)
    assert '2' in str(info.value)
    start = time.monotonic()
    flag = {'stop': False}
    threading.Timer(0.5, lambda: flag.__setitem__('stop', True)).start()
    with pytest.raises(RuntimeError, match='^sleep stopped by must_stop'):
        G.run_gmx('sleep 30', tmp_path, log, must_stop=lambda: flag['stop'])
    assert G._command_name('sleep 1 && OMP_NUM_THREADS=2 /x/gmx_mpi mdrun -ntomp 2') == (
        'gmx_mpi mdrun')
    assert G._command_name("'/a b/gmx' -quiet grompp") == 'gmx'
    assert G._command_name('echo hi') == 'echo' and G._command_name('gmx') == 'gmx'
    with pytest.raises(RuntimeError, match='timed out'):
        G.run_gmx('sleep 30', tmp_path, log, timeout=0.5)
    assert time.monotonic() - start < 20


def test_minimizer_commands(tmp_path):
    """The minimizer's resolved commands, topology and EM template; every argument recorded."""
    minimizer = GromacsMinimizer(em_mdp='em.mdp', maxwarn=2, topology='sys/topol.top')
    assert minimizer.em_mdp == str(Path('em.mdp').resolve())
    params = SimpleNamespace(gmx_grompp='/g/gmx grompp -maxwarn 5 -p topol.top -po /x.mdp',
                             gmx_mdrun='/g/gmx mdrun -ntomp 2 -plumed p.dat -pme gpu -nb gpu',
                             parent=tmp_path)
    grompp, mdrun, topology, em_mdp = minimizer._commands(params)
    assert grompp == '/g/gmx grompp -maxwarn 2'
    assert mdrun == '/g/gmx mdrun -ntomp 2 -nb gpu'
    assert topology == str(Path('sys/topol.top').resolve()) and em_mdp == minimizer.em_mdp
    default = GromacsMinimizer()._commands(params)
    assert default[0] == '/g/gmx grompp -maxwarn 5'
    # the run's index file is kept, relative to params.parent; an explicit one to the cwd
    indexed = SimpleNamespace(gmx_grompp='/g/gmx grompp -n sys/index.ndx -p topol.top',
                              gmx_mdrun='/g/gmx mdrun', parent=tmp_path)
    assert GromacsMinimizer()._commands(indexed)[0] == (
        f'/g/gmx grompp -n {tmp_path / "sys" / "index.ndx"}')
    explicit = GromacsMinimizer(grompp='/g/gmx grompp -n my.ndx -f x.mdp', maxwarn=1)
    assert explicit._commands(indexed)[0] == (
        f'/g/gmx grompp -n {Path("my.ndx").resolve()} -maxwarn 1')
    assert default[2] == str(tmp_path / 'topol.top')
    assert default[3].endswith('aimmd/engines/em.mdp') and Path(default[3]).is_file()
    assert set(minimizer.settings) >= set(inspect.signature(GromacsMinimizer).parameters)
    with pytest.raises(ValueError):
        GromacsMinimizer(restrain='heavy')


def test_solvator_commands(tmp_path):
    """The solvator's commands: EM without the unsupported offloads, genion without the index."""
    descriptors = _cluster_descriptors(tmp_path)
    params = SimpleNamespace(gmx_grompp='/g/gmx grompp -maxwarn 5 -n index.ndx -p topol.top',
                             gmx_mdrun='/g/gmx mdrun -ntomp 2 -maxh 1 -pme gpu -nb gpu -nsteps 9',
                             parent=tmp_path)
    solvator = GromacsSolvator(descriptors, topology=str(tmp_path / 'topol.top'), maxwarn=2)
    commands = solvator._commands(params)
    assert commands == {
        'grompp': f'/g/gmx grompp -n {tmp_path / "index.ndx"} -maxwarn 2',
        'grompp_genion': '/g/gmx grompp -maxwarn 2',
        'mdrun': '/g/gmx mdrun -ntomp 2 -pme gpu -nb gpu',
        'mdrun_em': '/g/gmx mdrun -ntomp 2 -nb gpu', 'gmx': '/g/gmx'}
    # an explicit mdrun (a wrapper) is used for EM as it is
    wrapped = GromacsSolvator(descriptors, topology=str(tmp_path / 'topol.top'),
                              mdrun='bash mdrun_gpu.sh -pme gpu')._commands(params)
    assert wrapped['mdrun'] == wrapped['mdrun_em'] == 'bash mdrun_gpu.sh -pme gpu'


def test_explicit_mdrun_commands_lose_plumed(tmp_path):
    """An explicit mdrun= loses -plumed and the checkpoint options too: preparation is unbiased.

    A run under a static PLUMED bias hands its shooting command to the minimizer and the
    solvator; the launcher prefix keeps its own options, the explicit offloads stay.
    """
    params = SimpleNamespace(gmx_grompp='/g/gmx grompp -p topol.top',
                             gmx_mdrun='/g/gmx mdrun -plumed run_bias.dat', parent=tmp_path)
    explicit = ('srun -c 4 /g/gmx_mpi mdrun -plumed /abs/opes.dat -ntomp 4 -cpi state.cpt '
                '-cpt 5 -maxh 2 -noappend -pme gpu -nb gpu')
    unbiased = 'srun -c 4 /g/gmx_mpi mdrun -ntomp 4 -pme gpu -nb gpu'
    assert GromacsMinimizer(mdrun=explicit)._commands(params)[1] == unbiased
    assert GromacsMinimizer(mdrun=explicit).settings['mdrun'] == explicit      # recorded as given
    commands = GromacsSolvator(_cluster_descriptors(tmp_path), topology=str(tmp_path / 'topol.top'),
                               mdrun=explicit)._commands(params)
    assert commands['mdrun'] == commands['mdrun_em'] == unbiased
    # the run's own command, as before
    assert GromacsMinimizer()._commands(params)[1] == '/g/gmx mdrun'


def _cluster_descriptors(tmp_path, n_waters=4, **kwargs):
    _, structure = _system(tmp_path, **kwargs)
    return GeneratorDescriptors(str(structure), GeneratorAtoms(3, ClusterSelection(n_waters=n_waters)))


def test_solvator_plan(tmp_path):
    """The composition, the box and the checks of the plan."""
    descriptors = _cluster_descriptors(tmp_path)
    solvator = GromacsSolvator(descriptors, topology=str(tmp_path / 'topol.top'))
    plan = solvator._plan(str(tmp_path / 'topol.top'))
    assert (plan.S, plan.K, plan.apw, plan.n_atoms) == (3, 4, 3, 3 + 120 + 2)
    assert plan.water == 'SOL' and plan.n_water == 40
    assert plan.cation == ('NA', 'NA', 1) and plan.anion == ('CL', 'CL', -1)
    assert (plan.n_cation, plan.n_anion, plan.n_ions) == (1, 1, 2)
    assert plan.layout == [('water', 'SOL', 40), ('ion', 'NA', 1), ('ion', 'CL', 1)]
    assert plan.solute_entries == [('LIG', 1)]
    np.testing.assert_allclose(plan.box, [L, L, L])
    assert plan.water_starts.tolist() == list(range(3, 123, 3))
    assert plan.solute_heavy.tolist() == [0, 1] and plan.water_heavy.tolist() == [0]
    text = solvator._molecules_text(plan, 10)
    assert text.rstrip().endswith('LIG              1\nSOL              10')
    # two cation species: genion places one
    sub = tmp_path / 'two'
    sub.mkdir()
    top, structure = _system(sub, ions=(('NA', 1), ('K', 1), ('CL', 1)))
    with pytest.raises(NotImplementedError, match='two cation'):
        GromacsSolvator(GeneratorDescriptors(str(structure), GeneratorAtoms(3)), topology=str(top))
    # name mismatch between topology and structure
    other = tmp_path / 'other'
    other.mkdir()
    top, structure = _system(other)
    top.write_text(top.read_text().replace(' HW1 ', ' HX1 '))
    with pytest.raises(ValueError, match='disagree'):
        GromacsSolvator(GeneratorDescriptors(str(structure), GeneratorAtoms(3)), topology=str(top))
    # the K cluster waters must be the first K of the first water entry
    few = tmp_path / 'few'
    few.mkdir()
    top, structure = _system(few, n_waters=3)
    top.write_text(top.read_text().replace('SOL 3\n', 'SOL 1\nSOL 2\n'))
    descriptors_few = GeneratorDescriptors(str(structure), GeneratorAtoms(3, ClusterSelection(n_waters=2)))
    with pytest.raises(NotImplementedError, match='first water entry'):
        GromacsSolvator(descriptors_few, topology=str(top))
    # a solute-only system: no cluster, the water found by its [ settles ]
    only = GromacsSolvator(GeneratorDescriptors(str(tmp_path / 'sys.gro'), GeneratorAtoms(3)),
                           topology=str(tmp_path / 'topol.top'))
    assert only._plan(str(tmp_path / 'topol.top')).K == 0


def test_water_copy_name(tmp_path):
    """The restrained water copy takes the first free name: SOLR, SOLR1, SOLR2, ..."""
    from aimmd.cgaimmd.preparation.topology import read_topology, restrained_topology
    descriptors = _cluster_descriptors(tmp_path, n_waters=4)
    top = tmp_path / 'topol.top'
    assert GromacsSolvator(descriptors, topology=str(top))._plan(str(top)).water_copy == 'SOLR'
    unused = ''.join(f'[ moleculetype ]\n{name} 1\n[ atoms ]\n 1  NA  1  {name}  X  1  0.0\n\n'
                     for name in ('SOLR', 'SOLR2'))
    top.write_text(top.read_text().replace('[ system ]', unused + '[ system ]'))
    plan = GromacsSolvator(descriptors, topology=str(top))._plan(str(top))
    assert (plan.water_copy, plan.water_copy_suffix) == ('SOLR1', 'R1')
    text = restrained_topology(str(top), [0, 1, 3, 6, 9, 12], 1e4,
                               water_copy_suffix=plan.water_copy_suffix, n_restrained_waters=4)
    derived = tmp_path / 'derived.top'
    derived.write_text(text)
    parsed = read_topology(derived)
    assert parsed.molecules[:3] == [('LIG', 1), ('SOLR1', 4), ('SOL', 36)]
    assert 'SOLR' in parsed.moleculetypes and 'SOLR2' in parsed.moleculetypes


def test_consistency_deletions(tmp_path):
    """Added waters the ranking would have chosen are deleted; the others are kept."""
    descriptors = _cluster_descriptors(tmp_path, n_waters=4)
    universe = mda.Universe(descriptors.structure)
    selected = descriptors.select(universe.atoms.positions, universe.dimensions)
    solvator = GromacsSolvator(descriptors, topology=str(tmp_path / 'topol.top'))
    radii = descriptors.cluster_radii(selected)
    heavy = selected[:2]
    direction = np.array([0.0, 0.0, 1.0])
    inside = heavy[0] + direction * (radii[0] - 0.2)
    outside = heavy[0] + direction * (radii[0] + 3.0)
    oxygens = np.stack([inside, outside])
    mask = solvator._consistency_violations(selected, oxygens, np.array([L, L, L]), radii)
    distances = np.linalg.norm(oxygens[:, None] - heavy[None], axis=-1).min(1)
    assert mask.tolist() == (distances < radii[0]).tolist() == [True, False]
    assert not solvator._consistency_violations(selected, oxygens, np.array([L] * 3), None).any()


def test_consistency_any_water_order(tmp_path):
    """Two rules: a cluster with its waters in another order deletes the same added waters."""
    _, structure = _system(tmp_path)
    ranking = ({'reference': 'name H3', 'n': 2}, {'reference': 'name C1', 'mode': 'com', 'n': -1})
    descriptors = GeneratorDescriptors(str(structure), GeneratorAtoms(3, ClusterSelection(
        n_waters=6, ranking=ranking)))
    universe = mda.Universe(descriptors.structure)
    selected = descriptors.select(universe.atoms.positions, universe.dimensions)
    blocks = selected[3:].reshape(-1, 3, 3)
    shuffled = np.concatenate([selected[:3], blocks[[5, 3, 0, 4, 1, 2]].reshape(-1, 3)])
    solvator = GromacsSolvator(descriptors, topology=str(tmp_path / 'topol.top'))
    oxygens = np.random.default_rng(2).uniform(0.0, L, size=(400, 3))
    lengths = np.array([L, L, L])
    ordered = solvator._consistency_violations(selected, oxygens, lengths,
                                               descriptors.cluster_radii(selected))
    assert 0 < ordered.sum() < len(oxygens)
    again = solvator._consistency_violations(shuffled, oxygens, lengths,
                                             descriptors.cluster_radii(shuffled))
    assert again.tolist() == ordered.tolist()


def test_ion_candidates_contiguous(tmp_path):
    """Replaceable waters: far from the solute, outside the exclusion; ordered last."""
    descriptors = _cluster_descriptors(tmp_path)
    sphere = ExclusionSphere(center=lambda selected: selected[0], radius=lambda selected: 8.0)
    solvator = GromacsSolvator(descriptors, topology=str(tmp_path / 'topol.top'),
                               ion_min_solute_distance=5.0, ion_exclusion=sphere)
    universe = mda.Universe(descriptors.structure)
    selected = descriptors.select(universe.atoms.positions, universe.dimensions)
    rng = np.random.default_rng(4)
    bulk = rng.uniform(0, L, size=(60, 1, 3))
    bulk = np.concatenate([bulk, bulk + [0.96, 0, 0], bulk + [-0.24, 0.93, 0]], axis=1)
    replaceable = solvator._replaceable(selected, bulk[:, 0], np.array([L] * 3))
    d_solute = np.linalg.norm(_min_image(bulk[:, 0, None] - selected[None, :3]), axis=-1).min(1)
    d_center = np.linalg.norm(_min_image(bulk[:, 0] - selected[0]), axis=-1)
    assert replaceable.tolist() == ((d_solute > 5.0) & (d_center > 8.0)).tolist()
    assert 0 < replaceable.sum() < len(bulk)
    ordered, n_fixed = solvator._order_for_ions(bulk, replaceable)
    assert n_fixed == int((~replaceable).sum())
    np.testing.assert_array_equal(ordered[n_fixed:], bulk[replaceable])
    np.testing.assert_array_equal(ordered[:n_fixed], bulk[~replaceable])
    assert sphere.resolve(selected)[1] == 8.0 and np.allclose(sphere.resolve(selected)[0], selected[0])
    # without an exclusion sphere only the distance to the solute counts
    plain = GromacsSolvator(descriptors, topology=str(tmp_path / 'topol.top'),
                            ion_min_solute_distance=7.0)
    assert plain._replaceable(selected, bulk[:, 0], np.array([L] * 3)).tolist() == (
        d_solute > 7.0).tolist()


def test_solvator_settings_recorded(tmp_path):
    """Every constructor argument is in the settings (the three tolerances included)."""
    descriptors = _cluster_descriptors(tmp_path)
    (tmp_path / 'nvt.mdp').write_text('integrator = md\ndt = 0.002\nref_t = 300\n')
    stage = NVTStage(str(tmp_path / 'nvt.mdp'), 2.0, {'tau_t': 0.1}, gen_vel=True)
    solvator = GromacsSolvator(descriptors, topology=str(tmp_path / 'topol.top'),
                               nvt_stages=[stage], water_exclusion=ExclusionSphere(
                                   center=lambda selected: selected[0], radius=6.0))
    names = set(inspect.signature(GromacsSolvator).parameters) - {'descriptors'}
    assert names <= set(solvator.settings)
    assert solvator.settings['moved_atom_tolerance'] == 0.01
    assert solvator.settings['box_margin'] == 0.5 and solvator.settings['ion_rmin_nm'] == 0.6
    assert solvator.settings['nvt_stages'][0] == {'mdp': str(tmp_path / 'nvt.mdp'), 'ps': 2.0,
                                                  'overrides': {'tau_t': 0.1}, 'gen_vel': True}
    assert solvator.settings['water_exclusion']['radius'] == 6.0
    assert solvator.settings['descriptors']['n_waters'] == 4
    import json
    json.dumps(solvator.settings)
    with pytest.raises(ValueError):
        GromacsSolvator(descriptors, restrain='all')
    with pytest.raises(TypeError):
        GromacsSolvator(descriptors, nvt_stages=['nvt.mdp'])


def _mdp_values(text):
    values = {}
    for line in text.splitlines():
        body = line.split(';', 1)[0]
        if '=' in body:
            key, value = body.split('=', 1)
            values[G._mdp_key(key)] = value.strip()
    return values


def test_nvt_stage_mdp(tmp_path):
    """NVT stages are NVT; the solvator's keys win, define/include merge, dt from the overrides."""
    descriptors = _cluster_descriptors(tmp_path)
    npt = tmp_path / 'npt.mdp'
    npt.write_text(textwrap.dedent("""\
        integrator = md
        dt         = 0.002
        define     = -DPOSRES
        tcoupl     = V-rescale
        ref_t      = 300 310
        pcoupl     = Parrinello-Rahman
        tau_p      = 2.0
        """))
    hot = tmp_path / 'hot.mdp'
    hot.write_text(npt.read_text() + 'gen_temp = 350\n')
    stages = [NVTStage(str(npt), 1.0, {'dt': 0.0005, 'define': '-DEXTRA', 'include': '-I/extra',
                                       'tau_t': 0.1}, gen_vel=True),
              NVTStage(str(hot), 0.2),
              NVTStage(str(hot), 0.1, gen_vel=True)]
    solvator = GromacsSolvator(descriptors, topology=str(tmp_path / 'topol.top'),
                               nvt_stages=stages)
    first, record = solvator._nvt_stage(0, True, '/top/dir', 41)
    values = _mdp_values(G.mdp_text(npt, first))
    assert values['pcoupl'] == 'no' and values['tau_p'] == '2.0'
    assert values['define'] == '-DPOSRES -DAIMMD_CGAIMMD_POSRES -DEXTRA'
    assert values['include'] == '-I/top/dir -I/extra'
    assert values['dt'] == '0.0005' and values['nsteps'] == '2000' and values['tau_t'] == '0.1'
    assert values['nstxout'] == '0' and values['nstxout_compressed'] == '0'
    assert values['gen_vel'] == 'yes' and values['continuation'] == 'no'
    assert values['gen_temp'] == '300'                                   # the first ref_t
    seed = G._gromacs_seed(41 + 1)
    assert values['gen_seed'] == values['ld_seed'] == str(seed)
    assert record == {'mdp': str(npt), 'ps': 1.0, 'dt': 0.0005, 'nsteps': 2000, 'ld_seed': seed,
                      'start': 'em.trr', 'gen_seed': seed, 'gen_temp': 300.0}
    second, record = solvator._nvt_stage(1, False, '/top/dir', 41)
    values = _mdp_values(G.mdp_text(hot, second))
    assert values['define'] == '-DPOSRES'                                # no restraints: as is
    assert values['gen_vel'] == 'no' and values['continuation'] == 'yes'
    assert values['nsteps'] == '100' and values['ld_seed'] == str(G._gromacs_seed(43))
    assert record['start'] == 'nvt0.cpt' and 'gen_seed' not in record
    third, record = solvator._nvt_stage(2, True, '/top/dir', 41)
    values = _mdp_values(G.mdp_text(hot, third))
    assert record['gen_temp'] == 350.0 and values['gen_temp'] == '350'  # the template's gen_temp
    assert values['nstxout'] == '50' and values['nstvout'] == values['nstfout'] == '0'   # last
    assert record['start'] == 'nvt1.cpt' and record['gen_seed'] == G._gromacs_seed(44)
    # the keys the solvator sets itself, and pressure coupling, are refused
    for overrides in ({'nsteps': 5}, {'Gen-Seed': 3}, {'ld_seed': 1}, {'nstxout': 10},
                      {'continuation': True}, {'pcoupl': 'C-rescale'}):
        with pytest.raises(ValueError):
            NVTStage(str(npt), 1.0, overrides)
    NVTStage(str(npt), 1.0, {'pcoupl': 'no'})
    NVTStage(str(npt), 1.0, {'Pcoupl': False})
    # a stage of no time, or of less than one step, and gen_vel without a temperature raise
    # when the stage or the solvator is made, not once per candidate
    for ps in (0.0, -1.0, float('nan'), 'long'):
        with pytest.raises(ValueError, match='ps'):
            NVTStage(str(npt), ps)
    top = str(tmp_path / 'topol.top')
    with pytest.raises(ValueError, match='0 steps'):
        GromacsSolvator(descriptors, topology=top, nvt_stages=[NVTStage(str(npt), 0.0009)])
    GromacsSolvator(descriptors, topology=top, nvt_stages=[NVTStage(str(npt), 0.0011)])
    cold = tmp_path / 'cold.mdp'
    cold.write_text('integrator = md\ndt = 0.002\n')
    with pytest.raises(ValueError, match='gen_temp or ref_t'):
        GromacsSolvator(descriptors, topology=top, nvt_stages=[NVTStage(str(cold), 1.0,
                                                                        gen_vel=True)])
    GromacsSolvator(descriptors, topology=top, nvt_stages=[NVTStage(str(cold), 1.0,
                                                                    {'gen_temp': 310},
                                                                    gen_vel=True)])
    with pytest.raises(ValueError, match='does not exist'):
        GromacsSolvator(descriptors, nvt_stages=[NVTStage(str(tmp_path / 'missing.mdp'), 1.0)])


def _solvator_context(tmp_path, descriptors):
    """A candidate context for the solvator without GROMACS (commands are strings only)."""
    from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
    from tests._cgaimmd_helpers import cgaimmd_params_namespace, make_candidate_context
    params = cgaimmd_params_namespace(
        gmx_grompp=f'/g/gmx grompp -p {tmp_path / "topol.top"}', gmx_mdrun='/g/gmx mdrun',
        network=TinyFlowNetwork(n_atoms=descriptors.n_atoms),
        _universe=mda.Universe(descriptors.structure), parent=tmp_path)
    return make_candidate_context(tmp_path, params=params)


def test_water_box_checked_by_names(tmp_path, monkeypatch):
    """A 4-site water box is refused although 12 added atoms are a multiple of 3."""
    from aimmd.cgaimmd.preparation import solvate
    descriptors = _cluster_descriptors(tmp_path, n_waters=4)
    universe = mda.Universe(descriptors.structure)
    selected = descriptors.select(universe.atoms.positions, universe.dimensions)

    def fake_solvate(command, cwd, log, **kwargs):
        assert ' solvate ' in command
        x, dims = G.read_gro(Path(cwd) / 'cluster.gro')
        resnames, names, _, _ = G._gro_records(Path(cwd) / 'cluster.gro')
        o = np.array([2.0, 2.0, 2.0])
        added = [o + [3.0 * k, 0, 0] + d for k in range(3)
                 for d in ([0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [0.1, 0.1, 0])]
        atoms = SimpleNamespace(names=names + ['OW', 'HW1', 'HW2', 'MW'] * 3,
                                resnames=resnames + ['SOL'] * 12, resids=[1] * (len(x) + 12))
        G.write_gro(Path(cwd) / 'solvated.gro', atoms, np.concatenate([x, added]), dims)

    monkeypatch.setattr(solvate, 'run_gmx', fake_solvate)
    solvator = GromacsSolvator(descriptors, topology=str(tmp_path / 'topol.top'))
    result = solvator(selected, _solvator_context(tmp_path, descriptors))
    assert not result.ok
    assert result.reason.startswith('prepare_failed: the water box spc216.gro does not have '
                                    '3-atom molecules'), result.reason


def _aimmd_posres_atoms(text):
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


def test_minimizer_restraints_and_displacement(tmp_path, monkeypatch):
    """Restrained atoms, minimum-image heavy-atom displacements and energies, without GROMACS.

    run_gmx is replaced: 'mdrun' writes em.trr (a hydrogen moved by 1 A, a heavy atom by 0.2 A
    across the x = 0 face of the box) and em.log (EM_LOG).
    """
    from MDAnalysis.coordinates.TRR import TRRWriter

    from aimmd.cgaimmd.preparation import minimize
    from aimmd.cgaimmd.preparation.topology import read_masses
    from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
    from aimmd.cgaimmd.types import PreparedFrame
    from tests._cgaimmd_helpers import cgaimmd_params_namespace, make_candidate_context
    nanma = Path(__file__).resolve().parent / 'cgaimmd_nanma'
    universe = mda.Universe(str(nanma / 'conf.gro'))
    box = np.asarray(universe.dimensions, dtype=float)
    masses = read_masses(nanma / 'topol.top')
    heavy, hydrogen = np.flatnonzero(masses > 1.5), np.flatnonzero(masses < 1.5)
    x = universe.atoms.positions.astype(float)
    x = x - x[heavy[0]] + [0.1, box[1] / 2, box[2] / 2]      # a heavy atom 0.1 A from a face
    out = x.copy()
    out[heavy[0], 0] += box[0] - 0.2                         # 0.2 A across it, written wrapped
    out[hydrogen[0], 2] += 1.0
    commands = []

    def fake_run_gmx(command, cwd, log, **kwargs):
        commands.append(command)
        if ' mdrun ' in command:
            empty = mda.Universe.empty(22, trajectory=True)
            empty.trajectory.ts.positions = out
            empty.trajectory.ts.dimensions = box
            with TRRWriter(str(Path(cwd) / 'em.trr'), 22) as writer:
                writer.write(empty.atoms)
            (Path(cwd) / 'em.log').write_text(EM_LOG)
        return ''

    monkeypatch.setattr(minimize, 'run_gmx', fake_run_gmx)
    params = cgaimmd_params_namespace(
        gmx_grompp=f'/g/gmx grompp -p {nanma / "topol.top"}', gmx_mdrun='/g/gmx mdrun -ntomp 1',
        network=TinyFlowNetwork(n_atoms=22), _universe=universe, parent=tmp_path)
    context = make_candidate_context(tmp_path, params=params, keep_scratch=True)
    frame = PreparedFrame(positions=x, dimensions=box)
    # heavy atoms only (the hydrogen's 1 A does not count), minimum image (not 79.8 A)
    result = GromacsMinimizer(max_displacement=0.5)(frame, context)
    assert result.ok, result.reason
    record = result.record
    assert record['max_displacement_A'] == pytest.approx(0.2, abs=1e-4)
    assert record['heavy_rms_displacement_A'] == pytest.approx(0.2 / np.sqrt(len(heavy)), abs=1e-4)
    assert (record['epot_before'], record['epot_after']) == (10.2911, -44.491333)
    assert (record['steps'], record['converged'], record['fmax']) == (44, True, 92.552406)
    np.testing.assert_allclose(result.positions, out, atol=1e-4)
    workdir = Path(record['directory'])
    assert G._read_mdp(workdir / 'em.mdp')['nstxout'] == '0'
    assert G._read_mdp(workdir / 'em.mdp')['define'].endswith('-DAIMMD_CGAIMMD_POSRES')
    assert _aimmd_posres_atoms((workdir / 'restrained.top').read_text()) == [
        int(i) + 1 for i in heavy]
    assert ' -r ref.gro -p restrained.top ' in commands[0] and ' mdrun ' in commands[1]
    # the limit applies to the heavy atoms: 0.1 A rejects the 0.2 A move
    strict = GromacsMinimizer(max_displacement=0.1)(frame, context)
    assert not strict.ok and strict.reason.startswith(
        'minimize_failed: a generator heavy atom moved 0.200 A')
    # 'generator_all' restrains every generator atom; 'none' uses the topology itself and no -r
    every = GromacsMinimizer(restrain='generator_all')(frame, context)
    text = (Path(every.record['directory']) / 'restrained.top').read_text()
    assert _aimmd_posres_atoms(text) == list(range(1, 23))
    commands.clear()
    free = GromacsMinimizer(restrain='none')(frame, context)
    assert free.ok and not (Path(free.record['directory']) / 'restrained.top').exists()
    assert ' -r ' not in commands[0] and f'-p {nanma / "topol.top"}' in commands[0]
    assert 'define' not in G._read_mdp(Path(free.record['directory']) / 'em.mdp')


def test_old_restraint_names(tmp_path, recwarn):
    """'flow_heavy' and 'flow_all' mean the new modes, with one DeprecationWarning each.

    The archived production params build their minimizers through the old
    module name with ``restrain='flow_heavy'``; only the new names are stored.
    """
    from aimmd.cgaimmd import _legacy
    _legacy._reset()
    with pytest.warns(DeprecationWarning, match="restrain 'flow_heavy' is deprecated; use "
                                                "restrain 'generator_heavy'"):
        from aimmd.flow.minimize import GromacsMinimizer as OldName
        minimizer = OldName(em_mdp='em.mdp', restrain='flow_heavy', restraint_k=1e4)
    assert OldName is GromacsMinimizer
    assert minimizer.restrain == minimizer.settings['restrain'] == 'generator_heavy'
    recwarn.clear()
    assert GromacsMinimizer(restrain='flow_heavy').restrain == 'generator_heavy'
    assert [w for w in recwarn if issubclass(w.category, DeprecationWarning)] == []
    descriptors = _cluster_descriptors(tmp_path)
    with pytest.warns(DeprecationWarning, match="restrain 'flow_all' is deprecated"):
        solvator = GromacsSolvator(descriptors, topology=str(tmp_path / 'topol.top'),
                                   restrain='flow_all')
    assert solvator.restrain == solvator.settings['restrain'] == 'generator_all'
    assert G.RESTRAINT_MODES == ('generator_heavy', 'generator_all', 'none')
    assert G.restraint_mode('none') == 'none'
    for bad in ('heavy', 'FLOW_HEAVY', ['generator_heavy'], None):
        with pytest.raises(ValueError, match='restrain must be one of'):
            G.restraint_mode(bad)
