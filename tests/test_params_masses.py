"""Masses of the shooting velocities: guessed from atom names, or read from the topology.

AIMMD draws every shooting point's velocities from ``Params.masses``, which
MDAnalysis guesses from the atom *names* of ``params.topology``. For CHARMM
residues the guess is wrong for carbons whose names start with ``CL`` (read as
chlorine, 35.45 instead of 12.011 amu): in NANMA the terminal methyl carbons
``CL`` and ``CLP`` got velocities at about 102 K instead of 300 K in every shot.

``masses_from_topology=True`` (opt-in; the default keeps the guess) reads the
masses of the GROMACS topology grompp uses (the ``-p`` of ``gmx_grompp``, else
``topol.top`` next to the params file), under ``engine='gromacs'`` only: other
engines keep the guess. Whenever both are available and differ, a warning says
so.
"""

import os
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


def test_last_p_wins(tmp_path):
    """grompp reads the last of repeated -p options."""
    fname = _params_file(tmp_path, top='second.top',
                         grompp='gmx grompp -p first.top -maxwarn 1 -p second.top',
                         extra='masses_from_topology = True')
    (Path(tmp_path) / 'first.top').write_text(TOP.replace('12.011', '99.000'))  # not read
    params = aimmd.Params.load(fname, save=False)
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


def test_rounding_is_no_disagreement(tmp_path, capsys):
    """Force fields round masses differently (1.00794 vs MDAnalysis' 1.008): no warning."""
    top = TOP.replace('12.011', '35.450').replace('1.008', '1.00794')
    params = aimmd.Params.load(_params_file(tmp_path, top_text=top), save=False)
    capsys.readouterr()
    params.masses
    assert 'masses' not in capsys.readouterr().out
    top = TOP.replace('12.011', '35.450').replace('   1.008', '   1.030')  # 2 %: a disagreement
    params = aimmd.Params.load(_params_file(tmp_path, top_text=top), save=False)
    capsys.readouterr()
    params.masses
    assert 'HL1: 1.008 vs 1.03' in capsys.readouterr().out


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


def test_massless_atom_refused(tmp_path, capsys):
    """A virtual site (mass 0, as TIP4P's M) would divide the velocity draw by zero (spec K.2)."""
    top = TOP.replace('-0.51  15.999', '-0.51  0.0')
    with pytest.raises(TypeError, match='masses_from_topology'):
        aimmd.Params.load(_params_file(tmp_path, top_text=top, extra='masses_from_topology = True'),
                          save=False)
    params = aimmd.Params.load(_params_file(tmp_path, top_text=top), save=False)
    capsys.readouterr()
    np.testing.assert_allclose(params.masses, GUESSED_MASSES, atol=1e-3)   # the default: the guess
    assert 'Warning' not in capsys.readouterr().out


FF_ITP = """[ defaults ]
1 2 yes 1.0 1.0

[ atomtypes ]
CT3   6  12.011  0.0  A  0.3  0.3
HA    1   1.008  0.0  A  0.2  0.1
O     8  15.999  0.0  A  0.3  0.8
"""

TOP_WITH_FORCE_FIELD = """#include "fake.ff/forcefield.itp"

[ moleculetype ]
Other 3

[ atoms ]
  1  CT3  1  ALAD  CL   1  -0.27
  2  HA   1  ALAD  HL1  2   0.09
  3  O    1  ALAD  OL   3  -0.51

[ system ]
test

[ molecules ]
Other 1
"""


def test_force_field_include_found(tmp_path, monkeypatch):
    """The NANMA case: the masses come from the atom types of a force field in GROMACS' share folder."""
    prefix = Path(tmp_path) / 'gromacs'
    (prefix / 'bin').mkdir(parents=True)
    force_field = prefix / 'share' / 'gromacs' / 'top' / 'fake.ff'
    force_field.mkdir(parents=True)
    (force_field / 'forcefield.itp').write_text(FF_ITP)
    gmx = prefix / 'bin' / 'gmx'
    gmx.write_text('#!/bin/sh\n')
    gmx.chmod(0o755)
    monkeypatch.delenv('GMXLIB', raising=False)
    monkeypatch.delenv('GMXDATA', raising=False)
    fname = _params_file(tmp_path, top_text=TOP_WITH_FORCE_FIELD, grompp=f'{gmx} grompp',
                         extra='masses_from_topology = True')
    params = aimmd.Params.load(fname, save=False)
    np.testing.assert_allclose(params.masses, TOPOLOGY_MASSES, atol=1e-6)


def test_recheck_on_assignment(tmp_path):
    """Assigning gmx_grompp or topology reads the topology masses again."""
    params = aimmd.Params.load(_params_file(tmp_path, extra='masses_from_topology = True'),
                               save=False)
    with pytest.raises(TypeError, match='missing.top'):
        params.gmx_grompp = 'gmx grompp -p missing.top'
    header, _, *atoms, box = GRO.splitlines()
    (Path(tmp_path) / 'six.gro').write_text('\n'.join([header, f'{6:5d}', *atoms, *atoms, box]) + '\n')
    with pytest.raises(TypeError, match='3 atoms'):                   # the topology's 3 against 6
        params.topology = str(Path(tmp_path) / 'six.gro')


def test_cache_per_file_version(tmp_path):
    """A rewritten topology is read again (the cache key is the file's mtime and size)."""
    params = aimmd.Params.load(_params_file(tmp_path, extra='masses_from_topology = True'),
                               save=False)
    np.testing.assert_allclose(params.masses, TOPOLOGY_MASSES, atol=1e-6)
    top = Path(tmp_path) / 'topol.top'
    top.write_text(TOP.replace('15.999', '16.0000'))                    # another size as well
    os.utime(top, ns=(top.stat().st_atime_ns, top.stat().st_mtime_ns + 10**9))
    np.testing.assert_allclose(params.masses, [12.011, 1.008, 16.0], atol=1e-6)


def test_other_engines_keep_the_guessed_masses(tmp_path, capsys):
    """masses_from_topology applies to GROMACS only: a toy-engine run loads and keeps the guess."""
    fname = _params_file(tmp_path, top=None, extra="masses_from_topology = True")
    Path(fname).write_text(Path(fname).read_text().replace("engine = 'gromacs'", "engine = 'toy'"))
    params = aimmd.Params.load(fname, save=False)                      # no topol.top: no error
    capsys.readouterr()
    np.testing.assert_allclose(params.masses, GUESSED_MASSES, atol=1e-3)
    assert 'Warning' not in capsys.readouterr().out
    (Path(tmp_path) / 'topol.top').write_text(TOP)                      # present, but not read
    np.testing.assert_allclose(params.masses, GUESSED_MASSES, atol=1e-3)


def test_multi_system_per_system_topologies(tmp_path, monkeypatch):
    """Each system reads the topology of its own gmx_grompp."""
    tmp_path = Path(tmp_path)
    monkeypatch.chdir(tmp_path)          # a re-check rebuilds the universes from the cwd
    (tmp_path / 's1.gro').write_text(GRO)
    (tmp_path / 's2.gro').write_text(GRO)
    (tmp_path / 's1.top').write_text(TOP)
    (tmp_path / 's2.top').write_text(TOP.replace('15.999', '17.000'))
    fname = tmp_path / 'params.py'
    fname.write_text("""
import numpy as np
import aimmd

def states_function(traj, system_id=None):
    return np.full(len(traj), 'R')

engine = 'gromacs'
multi_system = True
system_ids = ['s1', 's2']
topology = ['s1.gro', 's2.gro']
gmx_grompp = ['gmx grompp -p s1.top', 'gmx grompp -p s2.top']
masses_from_topology = True
network = aimmd.network.utils.placeholder
""")
    params = aimmd.Params.load(str(fname), save=False)
    np.testing.assert_allclose(params.masses_of('s1'), TOPOLOGY_MASSES, atol=1e-6)
    np.testing.assert_allclose(params.masses_of('s2'), [12.011, 1.008, 17.0], atol=1e-6)
    (tmp_path / 's2.top').unlink()
    params.masses_from_topology = False
    with pytest.raises(TypeError, match='s2.top'):                   # each system is checked
        params.masses_from_topology = True


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
    commands = []
    monkeypatch.setattr('aimmd.params._methods.execute_command',   # no grompp run
                        lambda command, **kwargs: commands.append(command))
    frame = params.universe.trajectory.ts
    params.initialize_simulation(frame, str(Path(tmp_path) / 'back'))
    np.testing.assert_allclose(seen[0], TOPOLOGY_MASSES, atol=1e-6)
    assert len(commands) == 1 and 'grompp' in commands[0]


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
