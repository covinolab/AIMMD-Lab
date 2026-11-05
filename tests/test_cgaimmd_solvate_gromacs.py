"""GromacsSolvator with GROMACS on NANMA in TIP3P water.

``tests/cgaimmd_nanma_water`` (built once by its ``build.sh``) is NANMA in a
2.8 nm box of 695 TIP3P waters with one Na+ and one Cl-. The candidate is a
cluster of NANMA and its 6 nearest waters cut out of that system by
``GeneratorDescriptors``, i.e. a stand-in for a generated cluster. GROMACS is
the ``gmx_executable`` of ``tests/conftest.py`` (marker ``gromacs``): skipped
without GROMACS.
"""

import subprocess
from pathlib import Path, PosixPath

import numpy as np
import pytest

mda = pytest.importorskip('MDAnalysis')

from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork                        # noqa: E402
from aimmd.cgaimmd.preparation import solvate                                       # noqa: E402
from aimmd.cgaimmd.preparation.descriptors import (ClusterSelection, GeneratorAtoms,  # noqa: E402
                                                   GeneratorDescriptors)
from aimmd.cgaimmd.preparation.gromacs import (_gro_records, _gromacs_seed, _read_mdp,  # noqa: E402
                                               read_gro, read_trr_last_frame, write_gro)
from aimmd.cgaimmd.preparation.solvate import (ExclusionSphere, GromacsSolvator,  # noqa: E402
                                               NVTStage)
from tests._cgaimmd_helpers import cgaimmd_params_namespace, make_candidate_context  # noqa: E402

pytestmark = pytest.mark.gromacs

WATER = Path(__file__).resolve().parent / 'cgaimmd_nanma_water'
K = 6
N_ATOMS = 22 + 3 * 695 + 2


def _solute_center(selected):
    return selected[:22].mean(axis=0)


def _setup(tmp_path, gmx):
    universe = mda.Universe(str(WATER / 'conf.gro'))
    descriptors = GeneratorDescriptors(str(WATER / 'conf.gro'),
                                       GeneratorAtoms(22, ClusterSelection(n_waters=K)))
    params = cgaimmd_params_namespace(
        engine='gromacs', topology=str(WATER / 'conf.gro'), gmx_mdp=str(WATER / 'nvt.mdp'),
        gmx_grompp=f'{gmx} grompp -maxwarn 5 -p {WATER / "topol.top"}',
        gmx_mdrun=f'{gmx} mdrun -ntmpi 1 -ntomp 2 -nb cpu -pin off',
        network=TinyFlowNetwork(n_atoms=descriptors.n_atoms), _universe=universe,
        path=PosixPath(tmp_path))
    selected = descriptors.select(universe.atoms.positions, universe.dimensions)
    return descriptors, params, selected, universe


def _solvator(descriptors, **kwargs):
    kwargs = {'em_mdp': str(WATER / 'em.mdp'), 'em_nsteps': 500,
              'ion_exclusion': ExclusionSphere(center=_solute_center, radius=9.0), **kwargs}
    return GromacsSolvator(descriptors, **kwargs)


def _recording(commands):
    """solvate.run_gmx that also appends every command to `commands`."""
    real = solvate.run_gmx

    def run_gmx(command, *args, **kwargs):
        commands.append(command)
        return real(command, *args, **kwargs)
    return run_gmx


def _posres_blocks(text):
    """{moleculetype or '#include <file>': [(atom, funct), ...]} of the restraint blocks.

    A block belongs to the moleculetype defined before it, or to the unconditional #include
    line before it (a moleculetype defined in that file).
    """
    out, where, inside, expecting, depth = {}, None, False, False, 0
    for line in text.splitlines():
        body = line.split(';')[0].strip()
        if not body:
            continue
        if body == '#ifdef AIMMD_CGAIMMD_POSRES':
            inside = True
        elif inside and body == '#endif':
            inside = False
        elif inside and body.split()[0].isdigit():
            fields = body.split()
            out.setdefault(where, []).append((int(fields[0]), int(fields[1])))
        elif body.startswith(('#ifdef', '#ifndef')):
            depth += 1
        elif body.startswith('#endif'):
            depth -= 1
        elif body.startswith('#include') and depth == 0:
            where = body
        elif body.startswith('['):
            expecting = body.strip('[] ').lower() == 'moleculetype'
        elif expecting:
            where, expecting = body.split()[0], False
    return out


def _nanma_heavy():
    """1-based heavy atoms of NANMA (topology masses)."""
    from aimmd.cgaimmd.preparation.topology import read_masses
    return [int(i) + 1 for i in np.flatnonzero(read_masses(WATER / 'topol.top')[:22] > 1.5)]


@pytest.fixture(scope='module')
def solvated(tmp_path_factory, gmx_executable):
    tmp_path = tmp_path_factory.mktemp('solvated')
    descriptors, params, selected, universe = _setup(tmp_path, gmx_executable)
    context = make_candidate_context(tmp_path, params=params, keep_scratch=True)
    commands = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(solvate, 'run_gmx', _recording(commands))
        result = _solvator(descriptors)(selected, context)
    return {'result': result, 'descriptors': descriptors, 'params': params, 'selected': selected,
            'universe': universe, 'workdir': Path(result.record['directory']), 'tmp': tmp_path,
            'context': context, 'commands': commands, 'gmx': gmx_executable}


def _placed(selected, box):
    return selected - selected[:22].mean(axis=0) + np.asarray(box[:3]) / 2.0


def test_composition_matches_topology(solvated):
    """The output has the topology's composition and atom order: grompp takes it unchanged."""
    result = solvated['result']
    assert result.ok, result.reason
    assert result.positions.shape == (N_ATOMS, 3) and np.isfinite(result.positions).all()
    np.testing.assert_allclose(result.dimensions, solvated['universe'].dimensions, atol=1e-4)
    build = result.record['build']
    for key in ('solvate_scale', 'waters_added', 'deleted_by_consistency', 'deleted_by_trim',
                'ion_seed', 'ion_min_solute_distance_A'):
        assert key in build, key
    assert set(result.record) >= {'settings', 'build', 'em', 'nvt', 'displacement', 'seconds'}
    workdir = solvated['workdir']
    write_gro(workdir / 'result.gro', solvated['universe'].atoms, result.positions,
              result.dimensions, decimals=6)
    (workdir / 'check.mdp').write_text('integrator = steep\nnsteps = 0\ncutoff-scheme = Verlet\n'
                                       'coulombtype = PME\nrcoulomb = 1.0\nrvdw = 1.0\n')
    proc = subprocess.run([solvated['gmx'], 'grompp', '-f', 'check.mdp', '-c', 'result.gro', '-p',
                           str(WATER / 'topol.top'), '-o', 'check.tpr', '-po', 'check_out.mdp'],
                          cwd=workdir, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert 'do not match' not in proc.stderr                            # atom names agree


def test_consistency_deletions_cluster(solvated):
    """The rebuilt system's own cluster selection is the candidate's cluster."""
    result, descriptors = solvated['result'], solvated['descriptors']
    initial, dims = read_gro(solvated['workdir'] / 'full_initial.gro')
    placed = _placed(solvated['selected'], dims)
    # the K cluster waters are the first waters of the system, in the candidate's order
    np.testing.assert_allclose(initial[:22 + 3 * K], placed, atol=1e-5)
    again = descriptors.select(initial, dims)
    np.testing.assert_allclose(again, placed, atol=1e-3)
    # and the minimized system still selects the same waters (a set: the order may change)
    after = descriptors.select(result.positions, result.dimensions)
    assert np.abs(after[:22] - placed[:22]).max() < 0.5
    oxygens, before = after[22::3], placed[22::3]
    nearest = np.linalg.norm(oxygens[:, None] - before[None], axis=-1).min(axis=1)
    assert nearest.max() < 0.5, nearest
    assert result.record['build']['deleted_by_consistency'] >= 0
    assert len(result.record['build']['cluster_radii_A']) == 1
    assert result.record['build']['water_copy'] == 'SOLR'


def test_ions_outside_exclusion(solvated):
    """Ions replace only bulk waters far from the solute and outside the exclusion sphere."""
    initial, dims = read_gro(solvated['workdir'] / 'full_initial.gro')
    ions = initial[-2:]
    solute = initial[:22]
    box = dims[:3]

    def distance(a, b):
        d = a - b
        return np.linalg.norm(d - box * np.round(d / box), axis=-1)

    assert min(distance(ion, solute).min() for ion in ions) > 5.0
    center = _solute_center(initial)
    assert all(distance(ion, center) > 9.0 for ion in ions)
    _, names, _, _ = _gro_records(solvated['workdir'] / 'full_initial.gro')
    assert names[-2:] == ['NA', 'CL']
    # genion's group: exactly the trailing block of replaceable bulk waters of pre_ion.gro
    pre, _ = read_gro(solvated['workdir'] / 'pre_ion.gro')
    indices = np.array([int(v) for line in (solvated['workdir'] / 'ions.ndx').read_text()
                        .splitlines()[1:] for v in line.split()])
    assert indices[-1] == len(pre) and (np.diff(indices) == 1).all() and len(indices) % 3 == 0
    oxygens = pre[22 + 3 * K::3]
    replaceable = ((np.array([distance(o, pre[:22]).min() for o in oxygens]) > 5.0)
                   & (distance(oxygens, _solute_center(pre)) > 9.0))
    first = (indices[0] - 1 - 22 - 3 * K) // 3
    assert replaceable[first:].all() and not replaceable[:first].any()
    # the ion seed is derived from the candidate's seed, and genion gets it
    seed = solvated['result'].record['build']['ion_seed']
    assert seed == _gromacs_seed(solvated['context'].seed)
    genion = [c for c in solvated['commands'] if ' genion ' in c]
    assert len(genion) == 1 and f'-seed {seed}' in genion[0]
    assert '-n ions.ndx' in genion[0]


def test_restrained_em_keeps_generator_atoms(solvated):
    """Restrained EM moves the selected heavy atoms by less than 0.3 A."""
    result = solvated['result']
    # 'generator_heavy': the solute's heavy atoms and the oxygen of every cluster water (the copy)
    text = (solvated['workdir'] / 'topol_restrained.top').read_text()
    assert _posres_blocks(text) == {'Other': [(i, 1) for i in _nanma_heavy()], 'SOLR': [(1, 1)]}
    molecules = [line.split() for line in text.split('[ molecules ]')[-1].splitlines()
                 if line.split() and not line.strip().startswith(';')]
    assert molecules == [['Other', '1'], ['SOLR', str(K)], ['SOL', str(695 - K)], ['NA', '1'],
                         ['CL', '1']]
    displacement = result.record['displacement']
    assert displacement['heavy_max_A'] < 0.3, displacement
    assert result.record['em']['steps'] is not None
    assert result.record['em']['epot'] < result.record['em']['epot_start']


def test_nvt_stage(tmp_path, gmx_executable):
    """A restrained NVT stage from fresh velocities; the restraints hold the generator atoms."""
    descriptors, params, selected, universe = _setup(tmp_path, gmx_executable)
    context = make_candidate_context(tmp_path, params=params)
    stage = NVTStage(str(WATER / 'nvt.mdp'), 0.2, {'tau_t': 0.1}, gen_vel=True)
    result = _solvator(descriptors, nvt_stages=[stage])(selected, context)
    assert result.ok, result.reason
    nvt = result.record['nvt']
    assert len(nvt) == 1 and nvt[0]['nsteps'] == 100 and nvt[0]['gen_temp'] == 300.0
    assert 1 <= nvt[0]['gen_seed'] < 2 ** 31
    # thermal motion in the restraint: kT / k = 2.5e-4 nm^2 per dimension at 300 K and
    # k = 1e4 kJ mol^-1 nm^-2, i.e. 0.27 A rms in 3-D
    displacement = result.record['displacement']
    assert displacement['heavy_rms_A'] < 0.45 and displacement['heavy_max_A'] < 1.0, displacement
    assert list(Path(context.scratch).iterdir()) == []                   # scratch removed
    # the same candidate with the same seeds builds the same system
    again = _solvator(descriptors, nvt_stages=[stage])(selected, make_candidate_context(
        tmp_path / 'again', params=params))
    assert again.ok and again.record['build'] == result.record['build']


def test_nvt_stage_is_nvt(tmp_path, monkeypatch, gmx_executable):
    """A pressure-coupled template runs NVT: pcoupl = no, the last frame keeps the box."""
    descriptors, params, selected, universe = _setup(tmp_path, gmx_executable)
    npt = tmp_path / 'npt.mdp'
    npt.write_text((WATER / 'nvt.mdp').read_text().replace(
        'pcoupl              = no', 'pcoupl = C-rescale\ntau_p = 1.0\ncompressibility = 4.5e-5\n'
        'ref_p = 1.0'))
    context = make_candidate_context(tmp_path, params=params, keep_scratch=True)
    stage = NVTStage(str(npt), 0.1, gen_vel=True)
    commands = []
    with monkeypatch.context() as patch:
        patch.setattr(solvate, 'run_gmx', _recording(commands))
        result = _solvator(descriptors, nvt_stages=[stage])(selected, context)
    assert result.ok, result.reason
    workdir = Path(result.record['directory'])
    # the first stage starts from the full-precision EM frame and restrains to the reference
    nvt_grompp = [c for c in commands if ' -f nvt0.mdp ' in c]
    assert len(nvt_grompp) == 1 and ' -t em.trr ' in nvt_grompp[0]
    assert ' -r restraint_reference.gro ' in nvt_grompp[0]
    assert _read_mdp(workdir / 'nvt0.mdp')['pcoupl'] == 'no'
    assert _read_mdp(workdir / 'nvt0.mdp')['ld_seed'] == str(result.record['nvt'][0]['ld_seed'])
    assert 'absolute position restraints' not in (workdir / 'gromacs.log').read_text()
    _, box = read_trr_last_frame(workdir / 'nvt0.trr')
    np.testing.assert_allclose(box, universe.dimensions, atol=1e-3)
    # a box that changed all the same fails the candidate
    real = solvate.read_trr_last_frame

    def grown(path):
        x, dims = real(path)
        return x, dims * np.array([1.01, 1.0, 1.0, 1.0, 1.0, 1.0])

    monkeypatch.setattr(solvate, 'read_trr_last_frame', grown)
    changed = _solvator(descriptors)(selected, make_candidate_context(tmp_path / 'box',
                                                                      params=params))
    assert not changed.ok and changed.reason.startswith('prepare_failed: the box changed')


def test_does_not_fit(tmp_path, gmx_executable):
    """A candidate larger than the box is a rejection, not an error."""
    descriptors, params, selected, _ = _setup(tmp_path, gmx_executable)
    context = make_candidate_context(tmp_path, params=params)
    stretched = selected.copy()
    stretched[:22] = selected[:22].mean(0) + 3.0 * (selected[:22] - selected[:22].mean(0))
    stretched[22:] += 40.0
    result = _solvator(descriptors)(stretched, context)
    assert not result.ok and result.reason.startswith('prepare_failed: does not fit')


def test_solute_only_with_water_exclusion(tmp_path, gmx_executable):
    """No cluster: the solute alone is re-solvated; a flat-bottom sphere keeps bulk water out."""
    _, params, _, universe = _setup(tmp_path, gmx_executable)
    descriptors = GeneratorDescriptors(str(WATER / 'conf.gro'), GeneratorAtoms(22))
    params.__dict__['network'] = TinyFlowNetwork(n_atoms=22)
    solute = descriptors.select(universe.atoms.positions, universe.dimensions)
    radius = 4.5
    sphere = ExclusionSphere(center=_solute_center, radius=radius)
    context = make_candidate_context(tmp_path, params=params, keep_scratch=True)
    result = GromacsSolvator(descriptors, em_mdp=str(WATER / 'em.mdp'), em_nsteps=500,
                             water_exclusion=sphere, flat_bottom_k=1e5)(solute, context)
    assert result.ok, result.reason
    assert result.record['build']['water_exclusion']['radius_A'] == radius
    assert 'cluster_radii_A' not in result.record['build']
    workdir = Path(result.record['directory'])
    text = (workdir / 'topol_restrained.top').read_text()
    lines = text.splitlines()
    assert lines[lines.index('#include "charmm27.ff/tip3p.itp"') + 1] == (
        '#ifdef AIMMD_CGAIMMD_POSRES')
    x, box = result.positions, result.dimensions[:3]
    oxygens = x[22:22 + 3 * 695:3]
    d = oxygens - _solute_center(x)
    d = np.linalg.norm(d - box * np.round(d / box), axis=-1)
    # EM stops below emtol = 1000 kJ mol^-1 nm^-1, which the wall balances at emtol / k =
    # 0.01 nm = 0.1 A inside the sphere (without the wall the nearest O stays at 3.7 A)
    assert d.min() > radius - 0.15, d.min()
    assert result.record['displacement']['heavy_max_A'] < 0.3


def test_index_file_kept(tmp_path, gmx_executable):
    """The run's index file reaches the full-system grompp: an NVT stage coupling its own groups."""
    descriptors, params, selected, _ = _setup(tmp_path, gmx_executable)
    groups = {'Solute': np.arange(1, 23), 'Rest': np.arange(23, N_ATOMS + 1)}
    with open(tmp_path / 'index.ndx', 'w') as handle:
        for name, indices in groups.items():
            handle.write(f'[ {name} ]\n' + '\n'.join(
                ' '.join(str(i) for i in indices[k:k + 15]) for k in range(0, len(indices), 15))
                + '\n')
    params.__dict__['gmx_grompp'] = (f'{gmx_executable} grompp -maxwarn 5 -n {tmp_path / "index.ndx"} '
                                     f'-p {WATER / "topol.top"}')
    stage = NVTStage(str(WATER / 'nvt.mdp'), 0.02, {'tc_grps': 'Solute Rest', 'tau_t': '0.1 0.1',
                                                    'ref_t': '300 300'}, gen_vel=True)
    context = make_candidate_context(tmp_path, params=params)
    result = _solvator(descriptors, nvt_stages=[stage])(selected, context)
    assert result.ok, result.reason
    assert f'-n {tmp_path / "index.ndx"}' in result.record['commands']['grompp']
    assert '-n' not in result.record['commands']['grompp_genion'].split()


def test_restraint_modes(tmp_path, gmx_executable):
    """'generator_all' with a water exclusion on a cluster, and 'none': restraints and references."""
    descriptors, params, selected, _ = _setup(tmp_path, gmx_executable)
    sphere = ExclusionSphere(center=_solute_center, radius=4.0)
    context = make_candidate_context(tmp_path, params=params, keep_scratch=True)
    every = _solvator(descriptors, restrain='generator_all', water_exclusion=sphere,
                      em_nsteps=50)(selected, context)
    assert every.ok, every.reason
    workdir = Path(every.record['directory'])
    blocks = _posres_blocks((workdir / 'topol_restrained.top').read_text())
    assert blocks == {'Other': [(i, 1) for i in range(1, 23)],
                      'SOLR': [(1, 1), (2, 1), (3, 1)],
                      '#include "charmm27.ff/tip3p.itp"': [(1, 2)]}      # the flat bottom
    # restraint references: the generator atoms (cluster waters included) where they are, the
    # bulk water oxygens at the sphere's center (the cluster waters are never pushed out)
    initial, _ = read_gro(workdir / 'full_initial.gro')
    reference, _ = read_gro(workdir / 'restraint_reference.gro')
    np.testing.assert_allclose(reference[:22 + 3 * K], initial[:22 + 3 * K], atol=1e-5)
    center = np.asarray(every.record['build']['water_exclusion']['center_A'])
    np.testing.assert_allclose(reference[22 + 3 * K:22 + 3 * 695:3],
                               np.broadcast_to(center, (695 - K, 3)), atol=2e-3)
    np.testing.assert_allclose(reference[23 + 3 * K:22 + 3 * 695:3],
                               initial[23 + 3 * K:22 + 3 * 695:3], atol=1e-5)   # bulk H as is
    # 'none' without an exclusion: the run's topology itself, no -r, no restraint define
    none = _solvator(descriptors, restrain='none', em_nsteps=50)(selected, make_candidate_context(
        tmp_path / 'none', params=params, keep_scratch=True))
    assert none.ok, none.reason
    workdir = Path(none.record['directory'])
    assert not (workdir / 'topol_restrained.top').exists()
    assert not (workdir / 'restraint_reference.gro').exists()
    assert 'define' not in _read_mdp(workdir / 'em_restrained.mdp')
    assert 'water_copy' not in none.record['build']


def test_candidate_rejections(tmp_path, monkeypatch, gmx_executable):
    """The solvator's physical checks reject candidates with their reasons (prepare_failed)."""
    descriptors, params, selected, universe = _setup(tmp_path, gmx_executable)

    def run(name, **kwargs):
        return _solvator(descriptors, em_nsteps=50, **kwargs)(selected, make_candidate_context(
            tmp_path / name, params=params))

    moved = run('displacement', max_displacement=1e-4)
    assert not moved.ok and moved.reason.startswith('prepare_failed: a generator heavy atom moved')
    solvate_moved = run('solvate', moved_atom_tolerance=1e-6)            # .gro rounding
    assert solvate_moved.reason.startswith('prepare_failed: gmx solvate moved a generator atom')
    trimmed = run('trim', trim_min_distance=100.0)
    assert trimmed.reason.startswith('prepare_failed: cannot delete')
    placed = _placed(selected, universe.dimensions)
    face = min(placed.min(), (universe.dimensions[:3] - placed).min())
    margin = run('margin', box_margin=face + 0.01)
    assert margin.reason.startswith('prepare_failed: does not fit'), margin.reason
    # genion output whose generator atoms moved (checked separately from the solvate output)
    real = solvate._gro_records

    def shifted(path):
        resnames, names, x, dims = real(path)
        if Path(path).name == 'genion.gro':
            x = x.copy()
            x[0] += 0.1
        return resnames, names, x, dims

    monkeypatch.setattr(solvate, '_gro_records', shifted)
    genion_moved = run('genion')
    assert genion_moved.reason.startswith(
        'prepare_failed: gmx genion moved a generator atom by 0.1')
