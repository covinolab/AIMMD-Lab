"""Validity checks of generated samples: bond graph, chirality, clashes.

NANMA (``tests/cgaimmd_nanma``) is the test molecule; its unbiased free-MD
frames are the clash reference and the stand-in "generated" samples.
"""

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('networkx')
mda = pytest.importorskip('MDAnalysis')

from aimmd.cgaimmd.flow_matching import embedding                   # noqa: E402
from aimmd.cgaimmd.preparation import symmetry, topology          # noqa: E402
from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors    # noqa: E402
from aimmd.cgaimmd.types import ValidityResult                      # noqa: E402
from aimmd.cgaimmd.preparation.validity import (Canonicalizer, TopologyValidity,  # noqa: E402
                                 covalent_bond_edges, detect_stereocenters)
from tests._cgaimmd_helpers import make_candidate_context               # noqa: E402

NANMA = Path(__file__).resolve().parent / 'cgaimmd_nanma'
TOP = str(NANMA / 'topol.top')
REFERENCE = [str(NANMA / 'free_md_A.xtc'), str(NANMA / 'free_md_B.xtc')]
CA, HA, CB, NL, CRP = 8, 9, 10, 6, 14


@pytest.fixture(scope='module')
def nanma():
    descriptors = GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(22))
    universe = mda.Universe(str(NANMA / 'conf.gro'), REFERENCE)
    frames = np.array([descriptors.select(ts.positions, ts.dimensions)
                       for ts in universe.trajectory])
    mt = topology.model_topology(TOP, n_solute=22)
    report = symmetry.analyse(mt)
    tokens = embedding.atom_tokens(mt.records, 'residue_atomname', 'orbits', orbits=report.orbits)
    return {'descriptors': descriptors, 'frames': frames, 'mt': mt, 'report': report,
            'tokens': tokens}


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _scramble(x, elements, rng, classes=None):
    """Randomly relabel atoms within classes (elements by default) and rotate: (sample, labels)."""
    classes = elements if classes is None else classes
    labels = np.arange(len(x))
    for value in np.unique(classes):
        members = np.flatnonzero(classes == value)
        labels[members] = rng.permutation(members)
    return (x[labels] - x.mean(0)) @ _rotation(rng).T + 40.0, labels


def _edges(x, elements):
    return {tuple(e) for e in covalent_bond_edges(x, elements).tolist()}


def test_covalent_bond_edges(nanma):
    """The 1.30 x covalent-radii rule reproduces the topology's 21 bonds on MD frames."""
    bonds = {tuple(b) for b in nanma['mt'].bonds.tolist()}
    for frame in nanma['frames']:
        edges = covalent_bond_edges(frame, nanma['mt'].elements)
        assert edges.dtype.kind == 'i' and edges.shape == (21, 2)
        assert {tuple(e) for e in edges.tolist()} == bonds
        assert (edges[:, 0] < edges[:, 1]).all()
    none = covalent_bond_edges(nanma['frames'][0], nanma['mt'].elements, cutoff_scale=0.5)
    assert none.shape == (0, 2)


def test_scrambled_labels_restored(nanma):
    """Relabelled and rotated frames are matched back onto the topology order."""
    rng = np.random.default_rng(0)
    elements = nanma['mt'].elements
    bonds = {tuple(b) for b in nanma['mt'].bonds.tolist()}
    element_only = Canonicalizer(elements, nanma['mt'].bonds)
    typed = Canonicalizer(elements, nanma['mt'].bonds, tokens=nanma['tokens'])
    for frame in nanma['frames']:
        sample, _ = _scramble(frame, elements, rng)
        perm, reason = element_only(sample)
        assert reason == '' and sorted(perm) == list(range(22))
        assert _edges(sample[perm], elements) == bonds
        # a sample of a typed network carries its tokens: only relabellings within a token
        # class are accepted, so the frame comes back up to methyl hydrogen swaps
        typed_sample, labels = _scramble(frame, elements, rng, classes=nanma['tokens'])
        perm, reason = typed(typed_sample)
        assert reason == ''
        restored = typed_sample[perm]
        np.testing.assert_array_equal(nanma['tokens'][labels[perm]], nanma['tokens'])
        assert _edges(restored, elements) == bonds
    # index-first: a sample in topology order keeps its labels
    perm, reason = element_only(nanma['frames'][0])
    assert reason == '' and perm.tolist() == list(range(22))
    # tokens that are not tied along orbits tell the methyl hydrogens apart: a swap between two
    # methyls is a relabelling of identical elements but not of identical tokens
    untied = embedding.atom_tokens(nanma['mt'].records, 'residue_atomname', 'none')
    swapped = nanma['frames'][0].copy()
    swapped[[1, 11]] = nanma['frames'][0][[11, 1]]                 # HL1 <-> HB1
    assert element_only(swapped)[1] == ''
    assert Canonicalizer(elements, nanma['mt'].bonds, tokens=untied)(swapped) == (
        None, 'not_isomorphic')
    assert typed(swapped)[1] == 'not_isomorphic'                    # orbit ties: methyl-local
    validity = TopologyValidity(TOP, nanma['descriptors'], clash_reference=REFERENCE)
    samples = np.stack([nanma['frames'][1], _scramble(nanma['frames'][1], elements, rng)[0]])
    result = validity(samples)
    assert isinstance(result, ValidityResult)
    assert result.ok.tolist() == [True, True] and result.reason.tolist() == ['', '']
    assert result.positions.shape == (2, 22, 3) and result.positions.dtype == np.float64
    assert result.info[0]['permutation_is_identity'] is True
    assert result.info[1]['permutation_is_identity'] is False
    assert result.info[1]['n_stereocenters'] == 1
    assert result.info[1]['min_nonbonded_A'] > 2.0
    assert _edges(result.positions[1], elements) == bonds
    assert result.info[0]['settings'] is validity.settings
    assert validity.settings['clash_reference'] == REFERENCE


def test_bond_count_rejected(nanma):
    """A broken bond changes the edge count: 'bond_count'; non-finite: 'non_finite'."""
    sample = nanma['frames'][2].copy()
    sample[1] = sample[0] + 3.0 * (sample[1] - sample[0]) / np.linalg.norm(sample[1] - sample[0])
    canonicalizer = Canonicalizer(nanma['mt'].elements, nanma['mt'].bonds)
    assert canonicalizer(sample) == (None, 'bond_count')
    bad = nanma['frames'][2].copy()
    bad[5, 1] = np.nan
    assert canonicalizer(bad) == (None, 'non_finite')
    validity = TopologyValidity(TOP, nanma['descriptors'], clash_reference=REFERENCE)
    result = validity(np.stack([sample, bad, nanma['frames'][2]]))
    assert result.reason.tolist() == ['bond_count', 'non_finite', '']
    assert result.ok.tolist() == [False, False, True]
    np.testing.assert_array_equal(result.positions[0], sample)       # unchanged when unmatched


def test_not_isomorphic(nanma):
    """Same number of bonds, different graph: 'not_isomorphic'."""
    sample = nanma['frames'][3].copy()
    carbon, oxygen = sample[4], sample[5]                           # CLP=OL
    direction = (oxygen - carbon) / np.linalg.norm(oxygen - carbon)
    sample[1] = oxygen + 0.97 * direction                           # HL1 moves from CL to OL
    edges = covalent_bond_edges(sample, nanma['mt'].elements)
    assert len(edges) == 21 and (1, 5) in {tuple(e) for e in edges.tolist()}
    canonicalizer = Canonicalizer(nanma['mt'].elements, nanma['mt'].bonds)
    assert canonicalizer(sample) == (None, 'not_isomorphic')


def test_stereocenter_detection(nanma):
    """NANMA has one stereocenter, CA: four neighbours in four distinct orbits."""
    centers = detect_stereocenters(nanma['mt'], nanma['report'], nanma['frames'][0])
    assert len(centers) == 1
    quad, sign = centers[0]
    assert quad == (NL, CA, HA, CB) and sign in (-1, 1)
    # L-alanine: every unbiased frame has the same sign
    for frame in nanma['frames']:
        assert detect_stereocenters(nanma['mt'], nanma['report'], frame)[0][1] == sign
    # the orbit order: neighbours sorted by the first atom of their orbits
    mirrored = detect_stereocenters(nanma['mt'], nanma['report'], -nanma['frames'][0])
    assert mirrored[0] == (quad, -sign)


def test_mirror_rejected_not_reflected(nanma):
    """A mirror image passes the bond graph but is rejected for chirality, never reflected."""
    validity = TopologyValidity(TOP, nanma['descriptors'], node_labels=nanma['tokens'],
                                clash_reference=REFERENCE)
    mirror = nanma['frames'][4] * np.array([-1.0, 1.0, 1.0])
    result = validity(mirror[None])
    assert result.ok.tolist() == [False] and result.reason.tolist() == ['chirality']
    quad, sign = detect_stereocenters(nanma['mt'], nanma['report'], nanma['frames'][4])[0]
    kept = detect_stereocenters(nanma['mt'], nanma['report'], result.positions[0])[0][1]
    assert kept == -sign                                            # still the mirror image
    explicit = TopologyValidity(TOP, nanma['descriptors'], stereocenters=[quad],
                                clash_reference=REFERENCE)
    assert explicit(mirror[None]).reason.tolist() == ['chirality']
    unchecked = TopologyValidity(TOP, nanma['descriptors'], stereocenters=(),
                                 clash_reference=REFERENCE)
    assert unchecked(mirror[None]).ok.tolist() == [True]


def test_mirror_reflected_when_asked(nanma):
    """reflect_mirror_images=True: a sample whose stereocenters are ALL inverted is replaced by its mirror image.

    For a generator that is equivariant under reflections (E(3), no parity breaking) with a parity-symmetric prior,
    x and its mirror image are equally likely outputs of the same request, so reflecting the wrong-handed ones back
    samples the correctly handed density exactly. The sample counts as valid and is flagged ``reflected``.
    """
    validity = TopologyValidity(TOP, nanma['descriptors'], node_labels=nanma['tokens'],
                                clash_reference=REFERENCE, reflect_mirror_images=True)
    frame = nanma['frames'][4]
    mirror = frame * np.array([-1.0, 1.0, 1.0])
    result = validity(np.stack([frame, mirror]))
    assert result.ok.tolist() == [True, True]
    assert [r['reflected'] for r in result.info] == [False, True]
    np.testing.assert_allclose(result.positions[1], frame, atol=1e-12)   # reflected back onto the frame
    assert validity.settings['reflect_mirror_images'] is True
    default = TopologyValidity(TOP, nanma['descriptors'], node_labels=nanma['tokens'],
                               clash_reference=REFERENCE)
    assert default.settings['reflect_mirror_images'] is False
    assert [r['reflected'] for r in default(mirror[None]).info] == [False]


def test_mirror_reflection_keeps_partial_inversions_invalid(nanma):
    """Only a full mirror image is reflected: with a second (explicit) stereocenter whose sign the inversion of CA
    alone does not change, inverting CA alone stays a 'chirality' rejection."""
    centers = detect_stereocenters(nanma['mt'], nanma['report'], nanma['frames'][0])
    quad = centers[0][0]
    frame = nanma['frames'][4].copy()
    # invert CA alone (umbrella flip): reflect CA and HA through the plane of CA's other neighbours N, CB, C'; every
    # bond length is kept, the methyl group (CB's quad) keeps its handedness
    p0, p1, p2 = frame[NL], frame[CB], frame[CRP]
    normal = np.cross(p1 - p0, p2 - p0)
    normal /= np.linalg.norm(normal)
    partial = frame.copy()
    for atom in (CA, HA):
        partial[atom] = frame[atom] - 2.0 * np.dot(frame[atom] - p0, normal) * normal
    second = (CA, CB, 11, 12)          # CB's improper (CA, CB, HB1, HB2): unchanged by moving HA
    validity = TopologyValidity(TOP, nanma['descriptors'], stereocenters=[quad, second],
                                clash_reference=None, reflect_mirror_images=True)
    result = validity(np.stack([frame, partial]))
    assert result.reason.tolist()[1] == 'chirality'
    assert [r['reflected'] for r in result.info] == [False, False]
    # control: with CA as the only stereocenter the same sample IS a full mirror image there, and is reflected
    only_ca = TopologyValidity(TOP, nanma['descriptors'], stereocenters=[quad], clash_reference=None,
                               reflect_mirror_images=True)
    assert [r['reflected'] for r in only_ca(partial[None]).info] == [True]


def test_clash(nanma):
    """Non-bonded heavy atoms closer than the reference minimum minus the margin: 'clash'."""
    validity = TopologyValidity(TOP, nanma['descriptors'], clash_reference=REFERENCE)
    thresholds = validity.settings['clash_thresholds_A']
    assert set(thresholds) >= {'CO', 'CC', 'NO'} and all(v > 1.5 for v in thresholds.values())
    ok = validity(nanma['frames'])
    assert ok.ok.all()                                              # the reference itself passes
    # the pairs are the heavy-atom pairs four or more bonds apart (1-4 pairs are excluded); the
    # thresholds are their minima per element pair over the reference frames, minus the margin
    elements = nanma['mt'].elements
    distances = topology.topological_distances(22, nanma['mt'].bonds, max_distance=4)
    heavy = elements != 1
    i, j = np.triu_indices(22, 1)
    d = np.linalg.norm(nanma['frames'][:, i] - nanma['frames'][:, j], axis=-1)
    symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
    classes = np.array([''.join(sorted(symbols[elements[a]] + symbols[elements[b]]))
                        for a, b in zip(i, j)])
    nonbonded = (distances[i, j] >= 4) & heavy[i] & heavy[j]
    one_four = (distances[i, j] == 3) & heavy[i] & heavy[j]
    assert set(thresholds) == set(classes[nonbonded])
    closer = []
    for c in thresholds:
        members = nonbonded & (classes == c)
        assert thresholds[c] == pytest.approx(d[:, members].min() - 0.10, abs=1e-4)
        near = one_four & (classes == c)
        if near.any() and d[:, near].min() < d[:, members].min():
            closer.append(c)
    assert closer                                     # the 1-4 exclusion changes a threshold
    assert ok.info[0]['min_nonbonded_A'] == pytest.approx(d[0, nonbonded].min())
    frame = nanma['frames'][0]
    compressed = frame.mean(0) + 0.85 * (frame - frame.mean(0))
    result = validity(compressed[None])
    assert result.reason.tolist() == ['clash']
    assert result.info[0]['min_nonbonded_A'] < ok.info[0]['min_nonbonded_A']
    strict = TopologyValidity(TOP, nanma['descriptors'], clash_reference=REFERENCE,
                              clash_margin=-1.0)
    assert not strict(nanma['frames'][:1]).ok.any()


def test_clash_fallback_logged(nanma, tmp_path):
    """Without reference frames: 0.75 x Bondi radii, logged once."""
    context = make_candidate_context(tmp_path)
    validity = TopologyValidity(TOP, nanma['descriptors'])
    assert validity.settings['clash_reference'] is None
    assert validity.settings['clash_thresholds_A']['CO'] == pytest.approx(0.75 * (1.70 + 1.52))
    validity(nanma['frames'][:2], context)
    validity(nanma['frames'][2:4], context)
    messages = [m for m in context.log.messages if 'Bondi' in m]
    assert len(messages) == 1 and 'fallback' in messages[0]


def test_vf2_timeout(nanma, monkeypatch):
    """A VF2 search longer than vf2_timeout gives 'vf2_timeout' and restores the alarm."""
    import signal

    from networkx.algorithms import isomorphism

    def slow(self):
        time.sleep(5.0)
        return True

    def sentinel(signum, frame):                                     # the caller's own handler
        pass

    monkeypatch.setattr(isomorphism.GraphMatcher, 'is_isomorphic', slow)
    old = signal.signal(signal.SIGALRM, sentinel)
    # a timer that runs already (pytest-timeout's) is stopped here and restored at the end
    outer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    started = time.monotonic()
    try:
        timer = signal.getitimer(signal.ITIMER_REAL)
        assert timer == (0.0, 0.0)
        canonicalizer = Canonicalizer(nanma['mt'].elements, nanma['mt'].bonds, vf2_timeout=0.2)
        sample, _ = _scramble(nanma['frames'][5], nanma['mt'].elements, np.random.default_rng(1))
        start = time.monotonic()
        assert canonicalizer(sample) == (None, 'vf2_timeout')
        assert time.monotonic() - start < 3.0
        assert signal.getsignal(signal.SIGALRM) is sentinel
        assert signal.getitimer(signal.ITIMER_REAL) == timer
        # a timer the caller had running is re-armed with the time it had left
        signal.setitimer(signal.ITIMER_REAL, 60.0)
        assert canonicalizer(sample) == (None, 'vf2_timeout')
        remaining = signal.getitimer(signal.ITIMER_REAL)[0]
        assert 55.0 < remaining < 60.0, remaining
        assert signal.getsignal(signal.SIGALRM) is sentinel
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)
        if outer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL,
                             max(outer[0] - (time.monotonic() - started), 1e-3), outer[1])


def test_fast_paths_skip_vf2(nanma, monkeypatch):
    """Index-first and the colour-refinement prefilter decide without a VF2 search."""
    from networkx.algorithms import isomorphism
    calls = []
    real = isomorphism.GraphMatcher.is_isomorphic

    def counted(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(isomorphism.GraphMatcher, 'is_isomorphic', counted)
    elements, bonds, frame = nanma['mt'].elements, nanma['mt'].bonds, nanma['frames'][0]
    assert Canonicalizer(elements, bonds)(frame)[1] == '' and calls == []           # index first
    assert Canonicalizer(elements, bonds, index_first=False)(frame)[1] == '' and len(calls) == 1
    untied = embedding.atom_tokens(nanma['mt'].records, 'residue_atomname', 'none')
    swapped = frame.copy()
    swapped[[1, 11]] = frame[[11, 1]]                               # HL1 <-> HB1
    calls.clear()
    assert Canonicalizer(elements, bonds, tokens=untied)(swapped) == (None, 'not_isomorphic')
    assert calls == []                                              # refused by the histograms


def test_validity_errors(nanma, tmp_path):
    """Shape errors and unsupported generator atoms raise (bugs, not rejections)."""
    validity = TopologyValidity(TOP, nanma['descriptors'], clash_reference=REFERENCE)
    with pytest.raises(ValueError, match='22'):
        validity(np.zeros((2, 21, 3)))
    assert validity(nanma['frames'][0]).positions.shape == (1, 22, 3)    # one sample
    with pytest.raises(ValueError, match='tokens'):
        Canonicalizer(nanma['mt'].elements, nanma['mt'].bonds, tokens=[1, 2, 3])
    # a structure whose solute atoms are in another order than the topology's
    lines = (NANMA / 'conf.gro').read_text().splitlines()
    lines[2], lines[3] = lines[2].replace('   CL', '  HL1'), lines[3].replace('  HL1', '   CL')
    (tmp_path / 'swapped.gro').write_text('\n'.join(lines) + '\n')
    swapped = GeneratorDescriptors(str(tmp_path / 'swapped.gro'), GeneratorAtoms(22))
    with pytest.raises(ValueError, match='disagree'):
        TopologyValidity(TOP, swapped, stereocenters=())
    # explicit stereocenters: four distinct generator atoms
    for quad in [(NL, CA, HA, HA), (NL, CA, HA, 22)]:
        with pytest.raises(ValueError, match='four distinct'):
            TopologyValidity(TOP, nanma['descriptors'], stereocenters=[quad])


PLANAR_TOP = """\
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
  CT   6  12.011  0.0  A  0.35  0.27
  HC   1   1.008  0.0  A  0.25  0.12
  MW   0   0.000  0.0  V  0.0   0.0

[ moleculetype ]
MET 3
[ atoms ]
 1  CT  1  MET  C1   1  -0.30  12.011
 2  HC  1  MET  H1   1   0.10   1.008
 3  HC  1  MET  H2   1   0.10   1.008
 4  HC  1  MET  H3   1   0.10   1.008
[ bonds ]
 1 2 1
 1 3 1
 1 4 1

[ system ]
planar
[ molecules ]
MET 1
"""


def _write_gro(path, names, resname, positions):
    lines = ['test', f'{len(names):5d}']
    for n, (name, p) in enumerate(zip(names, np.asarray(positions) / 10.0), start=1):
        lines.append(f'{1:5d}{resname:<5s}{name:>5s}{n:5d}{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}')
    lines.append('   3.00000   3.00000   3.00000')
    Path(path).write_text('\n'.join(lines) + '\n')


def test_planar_explicit_stereocenter(tmp_path):
    """An explicit stereocenter whose reference improper is planar (sign 0) is refused."""
    (tmp_path / 'topol.top').write_text(PLANAR_TOP)
    positions = [[15.0, 15.0, 15.0], [16.09, 15.0, 15.0], [14.45, 15.94, 15.0],
                 [15.8, 15.8, 15.0]]                    # in the plane z = 15, H3 next to H1
    _write_gro(tmp_path / 'conf.gro', ['C1', 'H1', 'H2', 'H3'], 'MET', positions)
    descriptors = GeneratorDescriptors(str(tmp_path / 'conf.gro'), GeneratorAtoms(4))
    with pytest.raises(ValueError, match='planar'):
        TopologyValidity(str(tmp_path / 'topol.top'), descriptors, stereocenters=[(1, 0, 2, 3)])
    assert TopologyValidity(str(tmp_path / 'topol.top'), descriptors).stereocenters == []


@pytest.mark.filterwarnings('ignore:Unknown masses')
def test_virtual_sites_not_supported(tmp_path):
    """A virtual site (mass 0) among the generator atoms raises NotImplementedError."""
    (tmp_path / 'topol.top').write_text(PLANAR_TOP.replace(
        ' 4  HC  1  MET  H3   1   0.10   1.008', ' 4  MW  1  MET  M1   1   0.10   0.000'))
    assert topology.read_atomic_numbers(tmp_path / 'topol.top').tolist() == [6, 1, 1, 0]
    positions = [[15.0, 15.0, 15.0], [16.09, 15.0, 15.0], [14.45, 15.94, 15.0],
                 [15.0, 15.0, 15.3]]
    _write_gro(tmp_path / 'conf.gro', ['C1', 'H1', 'H2', 'M1'], 'MET', positions)
    descriptors = GeneratorDescriptors(str(tmp_path / 'conf.gro'), GeneratorAtoms(4))
    with pytest.raises(NotImplementedError, match='virtual'):
        TopologyValidity(str(tmp_path / 'topol.top'), descriptors)


def test_symmetry_keywords(nanma, monkeypatch):
    """stereocenters='auto' passes the symmetry keywords to symmetry.analyse and records them."""
    seen = []
    real = symmetry.analyse

    def analyse(mt, **kwargs):
        seen.append(kwargs)
        return real(mt, **kwargs)

    monkeypatch.setattr(symmetry, 'analyse', analyse)
    default = TopologyValidity(TOP, nanma['descriptors'])
    twins = TopologyValidity(TOP, nanma['descriptors'], symmetry_twins_only_min_atoms=10,
                             symmetry_max_factor_order=24)
    assert seen == [{'max_factor_order': 720, 'twins_only_min_atoms': None},
                    {'max_factor_order': 24, 'twins_only_min_atoms': 10}]
    assert twins.stereocenters == default.stereocenters and len(twins.stereocenters) == 1
    assert twins.settings['symmetry_twins_only_min_atoms'] == 10
    assert default.settings['symmetry_max_factor_order'] == 720
    TopologyValidity(TOP, nanma['descriptors'], stereocenters=())
    assert len(seen) == 2                                           # no analysis without 'auto'
