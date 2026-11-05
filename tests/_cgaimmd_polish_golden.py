"""The recorded sample of the polished validity check.

The validity hook of the archived chignolin production params
(``tests/data/legacy_cgaimmd/archived_chignolin_validity_hook.py``: raw
check, polish through ``aimmd.flow.validate._polish``, check of the polished
sample) was run once with the code of the flow branch (flowdev ``bb98943``)
on the sample below, and what it returned is stored in
``tests/data/cgaimmd_polished_validity_flowdev_bb98943.npz``.
``tests/test_cgaimmd_polished_validity.py`` checks that
:class:`aimmd.cgaimmd.preparation.validity.PolishedValidity`, and the same
hook run through the old names on this checkout, return the same, bit for
bit.

The sample is NANMA (``tests/cgaimmd_nanma``) with the checker of the
production hook, a ``TopologyValidity`` with clash thresholds from the free
MD of both basins, and ten generated structures, stand-ins made from free-MD
frames (:data:`SAMPLES`): valid ones, one with two hydrogens exchanged (the
check relabels it), one with a hydrogen bonded to two heavy atoms (that the
polish repairs), one with a non-finite atom, one whose polish fails, a mirror
image, one with a steric clash (that the polish repairs), one that the
polish damages, and one far from the origin. The minimizer is a stand-in as
well (:class:`RecordedMinimizer`): it adds a stored displacement to each
structure, or fails, so that no GROMACS is needed and the result does not
depend on the GROMACS build.

Recording again is needed only when the sample changes. From the root of
this checkout, with ``<export>`` an empty folder::

    git archive bb98943 aimmd | tar -x -C <export>
    PYTHONPATH=<export> python tests/_cgaimmd_polish_golden.py record \\
        tests/data/cgaimmd_polished_validity_flowdev_bb98943.npz

(``aimmd`` is then imported from ``<export>``). This module imports only the
standard library and numpy at module level.
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
NANMA = HERE / 'cgaimmd_nanma'
HOOK = HERE / 'data' / 'legacy_cgaimmd' / 'archived_chignolin_validity_hook.py'
GOLDEN = HERE / 'data' / 'cgaimmd_polished_validity_flowdev_bb98943.npz'
"""The recorded output of the archived hook on the flow branch's code."""

N_ATOMS = 22
SEED = 1234
"""``context.seed`` of the call: polished structure ``k`` gets the seed ``SEED + k``."""

WORKERS = 2

SAMPLES = ('valid', 'hydrogens_exchanged', 'hydrogen_bonded_twice', 'not_finite',
           'polish_fails', 'mirror_image', 'clash', 'polish_damages', 'far_from_origin',
           'valid_again')
"""The generated structures, in order."""

# atoms of NANMA (0-based, topol.top order)
CL, HL1, NL, HL, CA, HA, CB, HB1, CRP = 0, 1, 6, 7, 8, 9, 10, 11, 14
PSI_SIDE = list(range(15, 22))             # OR NR HR CR HR1 HR2 HR3, beyond the CA-CRP bond


def _checker(package):
    """The ``TopologyValidity`` of NANMA, from `package` (``'aimmd.flow'`` or ``'aimmd.cgaimmd.preparation'``)."""
    if package == 'aimmd.flow':
        from aimmd.flow.descriptors import FlowAtoms as Atoms
        from aimmd.flow.descriptors import FlowDescriptors as Descriptors
        from aimmd.flow.validity import TopologyValidity
    else:
        from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms as Atoms
        from aimmd.cgaimmd.preparation.descriptors import GeneratorDescriptors as Descriptors
        from aimmd.cgaimmd.preparation.validity import TopologyValidity
    descriptors = Descriptors(str(NANMA / 'conf.gro'), Atoms(N_ATOMS))
    return TopologyValidity(str(NANMA / 'topol.top'), descriptors,
                            clash_reference=[str(NANMA / 'free_md_A.xtc'),
                                             str(NANMA / 'free_md_B.xtc')],
                            reflect_mirror_images=False), descriptors


def _frames(descriptors):
    """The whole free-MD frames of basin A, ``(10, 22, 3)``."""
    import MDAnalysis as mda
    universe = mda.Universe(str(NANMA / 'conf.gro'), str(NANMA / 'free_md_A.xtc'))
    return np.stack([descriptors.select(ts.positions, ts.dimensions)
                     for ts in universe.trajectory])


def _rotate(x, axis_from, axis_to, atoms, degrees):
    """`x` with `atoms` rotated about the bond axis_from -> axis_to (Rodrigues)."""
    y = x.copy()
    k = (x[axis_to] - x[axis_from]) / np.linalg.norm(x[axis_to] - x[axis_from])
    t = np.radians(degrees)
    v = x[atoms] - x[axis_to]
    v = (v * np.cos(t) + np.cross(k, v) * np.sin(t)
         + k * (v @ k)[:, None] * (1 - np.cos(t)))
    y[atoms] = v + x[axis_to]
    return y


def build_sample(package):
    """The generated structures and the recorded minimizer's displacements.

    Returns
    -------
    dict
        ``positions`` ``(10, 22, 3)``; ``displacements`` ``(9, 22, 3)`` and
        ``fails`` ``(9,)`` of the nine finite structures, in polish order;
        ``clash_degrees``, the psi turn of the clash sample.
    """
    validity, descriptors = _checker(package)
    frames = _frames(descriptors)
    rng = np.random.default_rng(20261105)
    samples, repairs = [], {}
    samples.append(frames[0])
    exchanged = frames[1].copy()
    exchanged[[HL1, HA]] = frames[1][[HA, HL1]]
    samples.append(exchanged)
    twice = frames[2].copy()
    twice[HL] = (frames[2][NL] + frames[2][CA]) / 2                    # bonded to NL and CA
    samples.append(twice)
    repairs[2] = frames[2] - twice
    broken = frames[3].copy()
    broken[CB] = np.nan
    samples.append(broken)
    samples.append(frames[4])
    samples.append(frames[5] * np.array([-1.0, 1.0, 1.0]))
    context = SimpleNamespace(log=print)
    clash, clash_degrees = None, None
    for degrees in range(10, 360, 10):                                 # the first clashing psi turn
        turned = _rotate(frames[6], CA, CRP, PSI_SIDE, degrees)
        if str(validity(turned[None], context).reason[0]) == 'clash':
            clash, clash_degrees = turned, degrees
            break
    if clash is None:
        raise RuntimeError('no psi turn of frame 6 gives a clash')
    samples.append(clash)
    repairs[6] = frames[6] - clash
    samples.append(frames[7])
    samples.append(frames[8] + np.array([100.0, -50.0, 30.0]))
    samples.append(frames[9])
    positions = np.stack(samples)
    assert len(positions) == len(SAMPLES)
    finite = [i for i in range(len(positions)) if np.isfinite(positions[i]).all()]
    displacements = np.zeros((len(finite), N_ATOMS, 3))
    fails = np.zeros(len(finite), dtype=bool)
    for k, i in enumerate(finite):
        displacements[k] = repairs.get(i, 0.01 * rng.standard_normal((N_ATOMS, 3)))
        if SAMPLES[i] == 'polish_fails':
            fails[k] = True
        if SAMPLES[i] == 'polish_damages':                             # HB1 bonded to CB and CA
            displacements[k, HB1] = (positions[i][CB] + positions[i][CA]) / 2 - positions[i][HB1]
    return {'positions': positions, 'displacements': displacements, 'fails': fails,
            'clash_degrees': clash_degrees}


class RecordedMinimizer:
    """A stand-in minimizer: structure ``k`` (context seed ``SEED + k``) moves by ``displacements[k]``.

    It needs no GROMACS and no aimmd: it returns a ``SimpleNamespace`` with
    the members of a ``PreparedFrame`` that the polish reads. A structure
    whose ``fails[k]`` is set fails with the reason
    ``'minimize_failed: recorded failure of structure k'``.
    """

    def __init__(self, displacements, fails, heavy):
        self.displacements = np.asarray(displacements, dtype=float)
        self.fails = np.asarray(fails, dtype=bool)
        self.heavy = np.asarray(heavy, dtype=bool)
        self.settings = {'function': 'tests._cgaimmd_polish_golden.RecordedMinimizer'}

    def __call__(self, frame, context):
        k = int(context.seed) - SEED
        x = np.asarray(frame.positions, dtype=float)
        if self.fails[k]:
            return SimpleNamespace(positions=x, dimensions=frame.dimensions, ok=False,
                                   reason=f'minimize_failed: recorded failure of structure {k}',
                                   record={'settings': self.settings, 'error': 'recorded'})
        moved = self.displacements[k]
        rms = float(np.sqrt((moved[self.heavy] ** 2).sum(-1).mean()))
        return SimpleNamespace(positions=x + moved, dimensions=frame.dimensions, ok=True,
                               reason='', record={'settings': self.settings, 'steps': 7,
                                                  'heavy_rms_displacement_A': rms})


def heavy_atoms():
    """Heavy-atom mask of NANMA from its topology masses (mass above 1.5)."""
    masses = []
    in_atoms = False
    for line in (NANMA / 'topol.top').read_text().splitlines():
        body = line.split(';')[0].strip()
        if body.startswith('['):
            in_atoms = body.strip('[] ').lower() == 'atoms'
            continue
        if in_atoms and body:
            masses.append(float(body.split()[7]))
    return np.asarray(masses) > 1.5


def system_folder(directory):
    """`directory` holding ``conf_A_box.gro`` (NANMA's structure), the hook's ``_SYSTEM``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(NANMA / 'conf.gro', directory / 'conf_A_box.gro')
    return str(directory)


def load_archived_hook(validity, minimizer, system, workers=WORKERS):
    """The module of the archived hook, with the objects it uses set."""
    spec = importlib.util.spec_from_file_location('archived_chignolin_validity_hook', HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.N_PROTEIN = N_ATOMS
    module._AtomCount.n_atoms = N_ATOMS
    module._validity = validity
    module._polisher = minimizer
    module._SYSTEM = system
    module._POLISH_WORKERS = workers
    return module


def context(scratch):
    """The hook context: what the archived hook and the checker read."""
    return SimpleNamespace(scratch=str(scratch), seed=SEED, log=print, params=None,
                           must_stop=lambda: False, settings={})


def as_arrays(result):
    """``(positions, ok, reason, info_json)`` of a ``ValidityResult``."""
    return (np.asarray(result.positions, dtype=np.float64), np.asarray(result.ok, dtype=bool),
            np.asarray([str(r) for r in result.reason]), json.dumps(result.info))


def _record(output):
    import aimmd
    if Path(aimmd.__file__).resolve().parents[1] == HERE.parent:
        raise SystemExit(f'aimmd is imported from this checkout ({aimmd.__file__}); put the '
                         'flow branch export first on PYTHONPATH')
    sample = build_sample('aimmd.flow')
    validity, _ = _checker('aimmd.flow')
    minimizer = RecordedMinimizer(sample['displacements'], sample['fails'], heavy_atoms())
    with tempfile.TemporaryDirectory() as tmp:
        hook = load_archived_hook(validity, minimizer, system_folder(Path(tmp) / 'system'))
        result = hook.flow_validity_function(sample['positions'], context(Path(tmp) / 'scratch'))
    positions, ok, reason, info = as_arrays(result)
    np.savez(output, sample=sample['positions'], displacements=sample['displacements'],
             fails=sample['fails'], clash_degrees=sample['clash_degrees'], positions=positions,
             ok=ok, reason=reason, info=info, names=np.asarray(SAMPLES),
             source='aimmd of flowdev bb98943 (git archive bb98943 aimmd)')
    for name, good, why in zip(SAMPLES, ok, reason):
        print(f'{name:24s} {bool(good)!s:6s} {why}')
    print(f'recorded with {aimmd.__file__} -> {output}')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != 'record':
        raise SystemExit(__doc__)
    _record(sys.argv[2])
