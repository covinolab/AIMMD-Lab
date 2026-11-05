"""Masses of the shooting velocities: guessed from atom names, or read from the topology.

AIMMD draws every shooting point's velocities from ``Params.masses``, which
MDAnalysis guesses from the atom *names* of ``params.topology``. For CHARMM
residues the guess is wrong for carbons whose names start with ``CL`` (read as
chlorine, 35.45 instead of 12.011 amu): in NANMA the terminal methyl carbons
``CL`` and ``CLP`` got velocities at about 102 K instead of 300 K in every shot.

``masses_from_topology=True`` (opt-in; the default keeps the guess) reads the
masses of the GROMACS topology grompp uses (the ``-p`` of ``gmx_grompp``, else
``topol.top`` next to the params file). Whenever both are available and differ,
a warning says so.
"""

from pathlib import Path

import numpy as np
import pytest

import aimmd


def _gro_text():
    atoms = [('CL', (1.0, 1.0, 1.0)), ('HL1', (1.1, 1.0, 1.0)), ('OL', (1.0, 1.12, 1.0))]
    lines = ['test system', f'{len(atoms):5d}']
    for i, (name, (x, y, z)) in enumerate(atoms, 1):
        lines.append(f'{1:>5d}{"ALAD":<5s}{name:>5s}{i:>5d}{x:8.3f}{y:8.3f}{z:8.3f}')
    lines.append(f'{3.0:10.5f}{3.0:10.5f}{3.0:10.5f}')
    return '\n'.join(lines) + '\n'


GRO = _gro_text()

TOP = """; a self-contained CHARMM-like topology (no force-field include)
[ defaults ]
1 2 yes 1.0 1.0

[ atomtypes ]
CT3   6  12.011  0.0  A  0.3  0.3
HA    1   1.008  0.0  A  0.2  0.1
O     8  15.999  0.0  A  0.3  0.8

[ moleculetype ]
Other 3

[ atoms ]
  1  CT3  1  ALAD  CL   1  -0.27  12.011
  2  HA   1  ALAD  HL1  2   0.09   1.008
  3  O    1  ALAD  OL   3  -0.51  15.999

[ system ]
test

[ molecules ]
Other 1
"""

TOPOLOGY_MASSES = np.array([12.011, 1.008, 15.999])
GUESSED_MASSES = np.array([35.45, 1.008, 15.999])


def _params_file(tmp_path, *, top='topol.top', top_text=TOP, grompp=None, extra=''):
    tmp_path = Path(tmp_path)
    (tmp_path / 'conf.gro').write_text(GRO)
    if top is not None and top_text is not None:
        (tmp_path / top).write_text(top_text)
    grompp_line = '' if grompp is None else f'gmx_grompp = {grompp!r}\n'
    source = f"""
import numpy as np
import aimmd

def states_function(traj):
    return np.full(len(traj), 'R')

engine = 'gromacs'
topology = 'conf.gro'
network = aimmd.network.utils.placeholder
{grompp_line}{extra}
"""
    fname = tmp_path / 'params.py'
    fname.write_text(source)
    return str(fname)


def test_default_keeps_the_guessed_masses(tmp_path):
    params = aimmd.Params.load(_params_file(tmp_path), save=False)
    assert params.masses_from_topology is False
    np.testing.assert_allclose(params.masses, GUESSED_MASSES, atol=1e-3)


def test_masses_from_topology_reads_the_top_file(tmp_path):
    params = aimmd.Params.load(_params_file(tmp_path, extra='masses_from_topology = True'),
                               save=False)
    np.testing.assert_allclose(params.masses, TOPOLOGY_MASSES, atol=1e-6)


def test_topology_file_is_the_p_of_gmx_grompp(tmp_path):
    fname = _params_file(tmp_path, top='system.top', grompp='gmx grompp -maxwarn 1 -p system.top',
                         extra='masses_from_topology = True')
    (Path(tmp_path) / 'topol.top').write_text(TOP.replace('12.011', '99.000'))  # not read
    params = aimmd.Params.load(fname, save=False)
    np.testing.assert_allclose(params.masses, TOPOLOGY_MASSES, atol=1e-6)


def test_masses_from_topology_is_saved(tmp_path):
    params = aimmd.Params.load(_params_file(tmp_path, extra='masses_from_topology = True'),
                               save=False)
    assert '\nmasses_from_topology = True\n' in str(params)
    default = aimmd.Params.load(_params_file(tmp_path), save=False)
    assert '\nmasses_from_topology = False\n' in str(default)


def test_disagreement_is_warned_about_by_default(tmp_path, capsys):
    params = aimmd.Params.load(_params_file(tmp_path), save=False)
    capsys.readouterr()
    params.masses
    out = capsys.readouterr().out
    assert 'masses' in out and 'CL' in out and 'masses_from_topology = True' in out
    params.masses                                           # once per params object
    assert 'masses_from_topology' not in capsys.readouterr().out


def test_disagreement_is_warned_about_when_opted_in(tmp_path, capsys):
    params = aimmd.Params.load(_params_file(tmp_path, extra='masses_from_topology = True'),
                               save=False)
    capsys.readouterr()
    params.masses
    out = capsys.readouterr().out
    assert 'CL' in out and '35.45' in out and '12.011' in out


def test_no_warning_when_the_masses_agree(tmp_path, capsys):
    top = TOP.replace('12.011', '35.450')                   # "agrees" with the guess
    params = aimmd.Params.load(_params_file(tmp_path, top_text=top), save=False)
    capsys.readouterr()
    np.testing.assert_allclose(params.masses, GUESSED_MASSES, atol=1e-3)
    assert 'masses' not in capsys.readouterr().out


def test_unreadable_topology_is_ignored_by_default(tmp_path, capsys):
    params = aimmd.Params.load(_params_file(tmp_path, top_text='not a topology ['), save=False)
    capsys.readouterr()
    np.testing.assert_allclose(params.masses, GUESSED_MASSES, atol=1e-3)
    assert 'Warning' not in capsys.readouterr().out


def test_masses_from_topology_needs_a_readable_topology(tmp_path):
    with pytest.raises(TypeError, match='masses_from_topology'):
        aimmd.Params.load(_params_file(tmp_path, top=None, extra='masses_from_topology = True'),
                          save=False)


def test_masses_from_topology_atom_count_must_match(tmp_path):
    top = TOP.replace('Other 1', 'Other 2')                 # 6 atoms against 3
    with pytest.raises(TypeError, match='6 atoms'):
        aimmd.Params.load(_params_file(tmp_path, top_text=top, extra='masses_from_topology = True'),
                          save=False)


def test_masses_from_topology_rejects_non_bool(tmp_path):
    with pytest.raises(TypeError):
        aimmd.Params.load(_params_file(tmp_path, extra="masses_from_topology = 'yes'"), save=False)


def test_shooting_velocities_use_the_topology_masses(tmp_path, monkeypatch):
    params = aimmd.Params.load(_params_file(tmp_path, extra='masses_from_topology = True'),
                               save=False)
    seen = []

    def fake_randomize(masses, temperature):
        seen.append(np.asarray(masses, dtype=float))
        return np.zeros((len(masses), 3))

    monkeypatch.setattr('aimmd.params._methods.randomize_velocities', fake_randomize)
    params.__dict__['engine'] = 'toy'                       # write the frame, no grompp
    frame = params.universe.trajectory.ts
    params.initialize_simulation(frame, str(Path(tmp_path) / 'back'))
    np.testing.assert_allclose(seen[0], TOPOLOGY_MASSES, atol=1e-6)


def test_include_directory_found_from_the_grompp_executable(tmp_path, monkeypatch):
    from aimmd.params.utils import gromacs_include_dir
    prefix = Path(tmp_path) / 'gromacs'
    (prefix / 'bin').mkdir(parents=True)
    (prefix / 'share' / 'gromacs' / 'top').mkdir(parents=True)
    gmx = prefix / 'bin' / 'gmx'
    gmx.write_text('#!/bin/sh\n')
    gmx.chmod(0o755)
    monkeypatch.delenv('GMXLIB', raising=False)
    monkeypatch.delenv('GMXDATA', raising=False)
    assert gromacs_include_dir(f'{gmx} grompp -maxwarn 1') == str(prefix / 'share' / 'gromacs' / 'top')
    monkeypatch.setenv('GMXDATA', str(prefix / 'share' / 'gromacs'))
    assert gromacs_include_dir('nonexistent-gmx grompp') == str(prefix / 'share' / 'gromacs' / 'top')
