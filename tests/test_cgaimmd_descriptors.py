"""Generator atoms and their descriptors.

Synthetic solvated frames are written as ``.gro`` files into ``tmp_path``;
the vacuum case uses the NANMA structure of ``tests/cgaimmd_nanma``.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('MDAnalysis')

from aimmd.cgaimmd.preparation.descriptors import (ClusterSelection, GeneratorAtoms, GeneratorDescriptors,   # noqa: E402
                                    RankingRule, coordinates_descriptors, make_whole)

NANMA = Path(__file__).resolve().parent / 'cgaimmd_nanma'
L = 30.0                                     # Å, the synthetic box
BOX = np.array([L, L, L, 90.0, 90.0, 90.0])


def _write_gro(path, names, resnames, resids, positions_A, box_A):
    lines = ['synthetic', f'{len(names):5d}']
    for n, (name, resname, resid, p) in enumerate(zip(names, resnames, resids,
                                                      np.asarray(positions_A) / 10.0), start=1):
        lines.append(f'{resid % 100000:5d}{resname:<5s}{name:>5s}{n % 100000:5d}'
                     f'{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}')
    lines.append(''.join(f'{v / 10.0:10.5f}' for v in box_A[:3]))
    Path(path).write_text('\n'.join(lines) + '\n')


def _solvated(tmp_path, n_waters=40, seed=0, solute=None):
    """A 3-atom solute (C1-C2-H3) near the box center plus waters; written to tmp_path/sys.gro."""
    rng = np.random.default_rng(seed)
    if solute is None:
        solute = np.array([[15.0, 15.0, 15.0], [16.5, 15.0, 15.0], [17.0, 15.9, 15.0]])
    names, resnames, resids = ['C1', 'C2', 'H3'], ['LIG'] * 3, [1, 1, 1]
    positions = [p for p in solute]
    oxygens = rng.uniform(0.0, L, size=(n_waters, 3))
    for k, o in enumerate(oxygens):
        positions += [o, o + [0.96, 0.0, 0.0], o + [-0.24, 0.93, 0.0]]
        names += ['OW', 'HW1', 'HW2']
        resnames += ['SOL'] * 3
        resids += [2 + k] * 3
    positions = np.asarray(positions) % L                    # wrapped into the box, as MD writes
    path = tmp_path / 'sys.gro'
    _write_gro(path, names, resnames, resids, positions, BOX)
    import MDAnalysis as mda
    stored = mda.Universe(str(path)).atoms.positions.astype(np.float64)
    return path, stored


def _mi(v, box=L):
    return v - box * np.round(v / box)


def test_vacuum_make_whole():
    """A molecule split across the box is made whole; atom 0 stays; = make_whole for NANMA."""
    import MDAnalysis as mda
    universe = mda.Universe(str(NANMA / 'conf.gro'))
    whole = universe.atoms.positions.astype(np.float64)
    box = universe.dimensions.astype(np.float64)
    shifted = whole + (box[:3] - whole[0])[None, :] - [0.5, 20.0, 0.3]   # atom 0 near a face
    split = shifted % box[:3]
    assert np.ptp(split, axis=0).max() > 40.0                            # really split
    descriptors = GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(n_solute=22))
    assert descriptors.n_atoms == 22 and descriptors.generator_atoms.n_waters == 0
    out = descriptors.select(split, box)
    assert out.dtype == np.float64 and out.shape == (22, 3)
    np.testing.assert_allclose(out[0], split[0])
    np.testing.assert_allclose(out - out[0], whole - whole[0], atol=1e-4)
    np.testing.assert_allclose(out, make_whole(split, box), atol=1e-9)
    # without a box (or whole=False) the frame is taken as stored
    np.testing.assert_array_equal(descriptors.select(split, None), split)
    raw = GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(22), whole=False)
    np.testing.assert_array_equal(raw.select(split, box), split)
    # make_whole itself: batches, no box, reference atom
    batch = make_whole(np.stack([split, split]), box)
    np.testing.assert_allclose(batch[1], out)
    assert make_whole(split, None) is split
    other = make_whole(split, box[:3], reference_atom=5)
    np.testing.assert_allclose(other[5], split[5])


def test_extended_molecule_walk(tmp_path):
    """A chain longer than half the box stays whole (minimum image against atom 0 would not)."""
    chain = np.stack([np.arange(20) * 1.4 + 1.0, np.full(20, 15.0), np.full(20, 15.0)], 1)
    names = [f'C{i + 1}' for i in range(20)]
    _write_gro(tmp_path / 'chain.gro', names, ['CHN'] * 20, [1] * 20, chain, BOX)
    descriptors = GeneratorDescriptors(str(tmp_path / 'chain.gro'), GeneratorAtoms(20))
    frame = (chain + [20.0, 0.0, 0.0]) % L                              # wraps the middle
    out = descriptors.select(frame, BOX)
    np.testing.assert_allclose(np.diff(out[:, 0]), 1.4, atol=1e-3)
    naive = make_whole(frame, BOX)
    assert not np.allclose(np.diff(naive[:, 0]), 1.4, atol=1e-3)


def test_cluster_ranking_rules(tmp_path):
    """Each rule takes the n not-yet-chosen waters whose O is nearest its reference."""
    path, frame = _solvated(tmp_path)
    ranking = ({'reference': 'name C1', 'mode': 'min_distance', 'n': 2},
               {'reference': 'resname LIG', 'mode': 'com', 'n': 1},
               {'reference': None, 'mode': 'min_distance', 'n': -1})
    cluster = ClusterSelection(n_waters=6, ranking=ranking)
    descriptors = GeneratorDescriptors(str(path), GeneratorAtoms(3, cluster))
    assert descriptors.n_atoms == 3 + 6 * 3
    out = descriptors.select(frame, BOX)
    oxygens = frame[3::3]
    masses = np.array([12.011, 12.011, 1.008])
    com = (frame[:3] * masses[:, None]).sum(0) / masses.sum()
    d1 = np.linalg.norm(_mi(oxygens - frame[0]), axis=-1)
    d2 = np.linalg.norm(_mi(oxygens - com), axis=-1)
    d3 = np.linalg.norm(_mi(oxygens[:, None] - frame[None, :2]), axis=-1).min(1)   # heavy atoms
    expected = list(np.argsort(d1, kind='stable')[:2])
    d2[expected] = np.inf
    expected.append(int(np.argmin(d2)))
    d3[expected] = np.inf
    expected += list(np.argsort(d3, kind='stable')[:3])
    chosen = out[3::3]
    for slot, water in enumerate(expected):
        np.testing.assert_allclose(_mi(chosen[slot] - oxygens[water]), 0.0, atol=1e-9)
    # the default ranking: one min_distance rule on the solute heavy atoms
    default = GeneratorDescriptors(str(path), GeneratorAtoms(3, ClusterSelection(n_waters=4)))
    d = np.linalg.norm(_mi(oxygens[:, None] - frame[None, :2]), axis=-1).min(1)
    for slot, water in enumerate(np.argsort(d, kind='stable')[:4]):
        np.testing.assert_allclose(_mi(default.select(frame, BOX)[3 + 3 * slot] - oxygens[water]),
                                   0.0, atol=1e-9)
    # deterministic
    np.testing.assert_array_equal(descriptors.select(frame, BOX), out)


def test_cluster_whole_next_to_its_references(tmp_path):
    """Solute and waters are whole; each water's oxygen is next to its ranking reference.

    The default rule ranks by the nearest solute heavy atom, so in the cluster
    the distance from every oxygen to the nearest heavy atom is the
    minimum-image distance of the frame, even for a solute split across the box.
    """
    solute = np.array([[0.4, 15.0, 15.0], [-1.1, 15.0, 15.0], [-1.6, 15.9, 15.0]])  # split
    path, frame = _solvated(tmp_path, solute=solute)
    descriptors = GeneratorDescriptors(str(path), GeneratorAtoms(3, ClusterSelection(n_waters=8)))
    out = descriptors.select(frame, BOX)
    np.testing.assert_allclose(np.linalg.norm(out[1] - out[0]), 1.5, atol=0.02)   # .gro precision
    np.testing.assert_allclose(np.linalg.norm(out[2] - out[1]), np.hypot(0.5, 0.9), atol=0.02)
    oxygens = frame[3::3]
    ranked = np.linalg.norm(_mi(oxygens[:, None] - frame[None, :2]), axis=-1).min(1)
    for k in range(8):
        water = out[3 + 3 * k: 6 + 3 * k]
        np.testing.assert_allclose(np.linalg.norm(water[1] - water[0]), 0.96, atol=0.02)
        np.testing.assert_allclose(np.linalg.norm(water[2] - water[0]), np.hypot(0.24, 0.93),
                                   atol=0.02)
        in_cluster = np.linalg.norm(water[0] - out[:2], axis=-1).min()     # no periodic images
        assert in_cluster == pytest.approx(np.sort(ranked)[k], abs=1e-9)
    # one solute molecule: center_selection moves nothing
    centered = GeneratorDescriptors(str(path), GeneratorAtoms(3, ClusterSelection(
        n_waters=8, center_selection='name C1')))
    np.testing.assert_allclose(centered.select(frame, BOX), out, atol=1e-12)


def test_center_selection_places_molecules_not_waters(tmp_path):
    """A water stays next to its ranking reference, wherever center_selection is.

    The chain C1..C20 spans 26.6 A of the 30 A box; the water's oxygen is 2.4 A
    from C20 and 2.8 A from C1 (through the box face). The default rule ranks
    it by C20, so it is placed next to C20 (x = 29), also with
    ``center_selection='name C1'``: the port in the development branch moved
    it next to C1 (x = -1), where its distance to the reference it was ranked
    by is not the ranking distance.
    """
    chain = np.stack([np.arange(20) * 1.4 + 1.0, np.full(20, 15.0), np.full(20, 15.0)], 1)
    oxygen = np.array([29.0, 15.0, 17.0])
    positions = np.concatenate([chain, [oxygen, oxygen + [0.96, 0.0, 0.0],
                                        oxygen + [-0.24, 0.93, 0.0]]])
    names = [f'C{i + 1}' for i in range(20)] + ['OW', 'HW1', 'HW2']
    _write_gro(tmp_path / 'chain_w.gro', names, ['CHN'] * 20 + ['SOL'] * 3, [1] * 20 + [2] * 3,
               positions % L, BOX)
    structure = str(tmp_path / 'chain_w.gro')
    near_c1 = GeneratorDescriptors(structure, GeneratorAtoms(20, ClusterSelection(
        n_waters=1, center_selection='name C1')))
    default = GeneratorDescriptors(structure, GeneratorAtoms(20, ClusterSelection(n_waters=1)))
    frame = positions % L
    assert near_c1.select(frame, BOX)[20, 0] == pytest.approx(29.0, abs=1e-3)
    assert default.select(frame, BOX)[20, 0] == pytest.approx(29.0, abs=1e-3)
    # a selection that extends over more than half the box (C1 and C20, 26.6 A apart)
    ends = GeneratorDescriptors(structure, GeneratorAtoms(20, ClusterSelection(
        n_waters=1, center_selection='name C1 or name C20')))
    assert ends.select(frame, BOX)[20, 0] == pytest.approx(29.0, abs=1e-3)
    shifted = ends.select((frame + [10.0, 0.0, 0.0]) % L, BOX)          # C1 at 11, O at 9
    assert shifted[20, 0] - shifted[0, 0] == pytest.approx(28.0, abs=1e-3)
    # a rule ranked by C1 places the same water next to C1
    by_c1 = GeneratorDescriptors(structure, GeneratorAtoms(20, ClusterSelection(
        n_waters=1, ranking=({'reference': 'name C1', 'n': -1},))))
    assert by_c1.select(frame, BOX)[20, 0] == pytest.approx(-1.0, abs=1e-3)


def _unbound_host_guest(tmp_path, shift=(0.0, 0.0, 0.0)):
    """A host (C1, BR1, BR2) and a guest (C3-C4) 14.9 A apart, waters, one straddling the box edge.

    The heavy host keeps the solute's center of mass 16.1 A from the oxygen
    of water 0 (x = 29.8), which is 2.75 A from the guest atom C4 and whose
    hydrogen HW1 is written across the x = 30 face (x = 0.76). Water 1 sits
    next to the host, the others are far from both. Returns the structure,
    the frame (shifted by `shift` and wrapped into the box, as MD writes it)
    and the masses.
    """
    solute = np.array([[11.0, 15.0, 15.0], [12.9, 15.0, 15.0], [11.0, 16.9, 15.0],   # host
                       [26.0, 15.0, 15.0], [27.5, 15.0, 15.0]])                      # guest
    names = ['C1', 'BR1', 'BR2', 'C3', 'C4']
    resnames, resids = ['HST'] * 3 + ['GST'] * 2, [1, 1, 1, 2, 2]
    masses = [12.011, 79.904, 79.904, 12.011, 12.011]
    oxygens = [[29.8, 15.0, 16.5], [9.0, 15.0, 17.0], [20.0, 4.0, 4.0], [5.0, 25.0, 25.0],
               [20.0, 26.0, 6.0]]
    positions = list(solute)
    for k, o in enumerate(np.asarray(oxygens)):
        positions += [o, o + [0.96, 0.0, 0.0], o + [-0.24, 0.93, 0.0]]
        names += ['OW', 'HW1', 'HW2']
        resnames += ['SOL'] * 3
        resids += [3 + k] * 3
        masses += [15.999, 1.008, 1.008]
    frame = (np.asarray(positions) + np.asarray(shift)) % L
    _write_gro(tmp_path / 'unbound.gro', names, resnames, resids, frame, BOX)
    import MDAnalysis as mda
    stored = mda.Universe(str(tmp_path / 'unbound.gro')).atoms.positions.astype(np.float64)
    return tmp_path / 'unbound.gro', stored, np.asarray(masses)


@pytest.mark.parametrize('shift', [(0.0, 0.0, 0.0), (7.3, -12.1, 3.0)])
def test_water_straddling_the_box_edge_in_an_unbound_frame(tmp_path, shift):
    """A water ranked by a guest far from its host comes out whole and next to the guest.

    The bug of the development branch: every water went to the image nearest
    the solute's center of mass, which in this unbound frame is 16.1 A from
    the guest's water, more than half the box, so the water landed one box
    away from the guest. Each water is now next to the reference of the rule
    that ranked it, whole across the box face.
    """
    path, frame, masses = _unbound_host_guest(tmp_path, shift)
    ranking = ({'reference': 'resname GST', 'n': 1}, {'reference': 'resname HST', 'n': -1})
    descriptors = GeneratorDescriptors(str(path), GeneratorAtoms(5, ClusterSelection(
        n_waters=2, ranking=ranking)), masses=masses)
    assert len(descriptors._walk) == 2                                  # host and guest
    out = descriptors.select(frame, BOX)
    guest, host = out[3:5], out[0:3]
    guest_water, host_water = out[5:8], out[8:11]
    for water in (guest_water, host_water):                            # whole
        assert np.linalg.norm(water[1] - water[0]) == pytest.approx(0.96, abs=0.02)
        assert np.linalg.norm(water[2] - water[0]) == pytest.approx(np.hypot(0.24, 0.93),
                                                                    abs=0.02)
    # the guest's water next to the guest, at the frame's minimum-image distance
    o_frame, guest_frame = frame[5], frame[3:5]
    ranked = np.linalg.norm(_mi(o_frame - guest_frame), axis=-1).min()
    assert ranked == pytest.approx(np.hypot(2.3, 1.5), abs=0.01)
    assert np.linalg.norm(guest_water[0] - guest, axis=-1).min() == pytest.approx(ranked,
                                                                                  abs=1e-9)
    # the host's water next to the host
    ranked_host = np.linalg.norm(_mi(frame[8] - frame[:3]), axis=-1).min()
    assert np.linalg.norm(host_water[0] - host, axis=-1).min() == pytest.approx(ranked_host,
                                                                                abs=1e-9)
    # the guest stays 14.9 A from the host; the guest's water is more than half a box from
    # the solute's center of mass (where the development branch put it one box away)
    center = (out[:5] * masses[:5, None]).sum(0) / masses[:5].sum()
    assert np.linalg.norm(guest_water[0] - center) > L / 2
    # the radii of the rules on the cluster alone are the ranking distances of the frame,
    # which the solvator's selection consistency relies on
    assert descriptors.cluster_radii(out) == pytest.approx([ranked, ranked_host], abs=1e-9)
    # without a shift: the cluster sits where the frame has it, HW1 unwrapped to x = 30.76
    if not any(shift):
        np.testing.assert_allclose(guest_water[:, 0], [29.8, 30.76, 29.56], atol=2e-3)


def test_com_rule_is_mass_weighted(tmp_path):
    """A 'com' rule ranks by the center of mass of its reference atoms, not by their mean."""
    solute = np.array([[15.0, 15.0, 15.0], [16.5, 15.0, 15.0], [17.0, 15.9, 15.0]])
    masses = np.array([12.011, 1.008])                              # C1 and H3
    com = (solute[[0, 2]] * masses[:, None]).sum(0) / masses.sum()
    mean = solute[[0, 2]].mean(0)
    first, second = com + [0.0, 0.0, 4.0], mean + [0.0, 0.0, 4.05]
    assert np.linalg.norm(first - com) < np.linalg.norm(second - com)
    assert np.linalg.norm(second - mean) < np.linalg.norm(first - mean)
    positions, names = list(solute), ['C1', 'C2', 'H3']
    for o in (second, first):                                       # the COM's water second
        positions += [o, o + [0.96, 0.0, 0.0], o + [-0.24, 0.93, 0.0]]
        names += ['OW', 'HW1', 'HW2']
    _write_gro(tmp_path / 'com.gro', names, ['LIG'] * 3 + ['SOL'] * 6, [1] * 3 + [2] * 3 + [3] * 3,
               np.asarray(positions), BOX)
    rule = {'reference': 'name C1 or name H3', 'mode': 'com', 'n': -1}
    descriptors = GeneratorDescriptors(str(tmp_path / 'com.gro'), GeneratorAtoms(3, ClusterSelection(
        n_waters=1, ranking=(rule,))))
    chosen = descriptors.select(np.asarray(positions), BOX)[3]
    np.testing.assert_allclose(chosen, first, atol=0.01)            # .gro precision


def test_rules_hashable():
    """Ranking rules are frozen mappings, so ClusterSelection and GeneratorAtoms are hashable."""
    rules = ({'reference': 'name C1', 'mode': 'com', 'n': 2}, {'reference': None, 'n': -1})
    cluster = ClusterSelection(n_waters=4, ranking=rules)
    assert all(isinstance(rule, RankingRule) for rule in cluster.ranking)
    assert cluster.ranking[0]['mode'] == 'com' and cluster.ranking[1]['mode'] == 'min_distance'
    assert dict(cluster.ranking[0]) == rules[0] and cluster.ranking[1].get('n') == -1
    again = ClusterSelection(n_waters=4, ranking=cluster.ranking)
    assert again == cluster and hash(again) == hash(cluster)
    atoms = GeneratorAtoms(3, cluster)
    assert {atoms: 1}[GeneratorAtoms(3, ClusterSelection(n_waters=4, ranking=rules))] == 1
    assert hash(GeneratorAtoms(22)) == hash(GeneratorAtoms(22))
    with pytest.raises(KeyError):
        cluster.ranking[0]['radius']
    with pytest.raises(ValueError, match='unknown'):
        ClusterSelection(n_waters=2, ranking=({'reference': None, 'n': -1, 'radius': 3.0},))


def test_cluster_radii(tmp_path):
    """Per rule, the largest reference distance of its waters; no other water is closer."""
    path, frame = _solvated(tmp_path, n_waters=60, seed=3)
    ranking = ({'reference': 'name H3', 'mode': 'min_distance', 'n': 3},
               {'reference': None, 'mode': 'com', 'n': -1})
    descriptors = GeneratorDescriptors(str(path), GeneratorAtoms(3, ClusterSelection(n_waters=7,
                                                                            ranking=ranking)))
    selected = descriptors.select(frame, BOX)
    radii = descriptors.cluster_radii(selected)
    assert len(radii) == 2 and all(isinstance(r, float) for r in radii)
    oxygens = selected[3::3]
    assert radii[0] == pytest.approx(np.linalg.norm(oxygens[:3] - selected[2], axis=-1).max())
    com = selected[:2].mean(0)                     # 'reference': None is the solute heavy atoms
    assert radii[1] == pytest.approx(np.linalg.norm(oxygens[3:] - com, axis=-1).max())
    # the consistency property the solvator relies on: in the source frame no unchosen
    # water is closer to a rule's reference than that rule's radius
    all_o = frame[3::3]
    chosen = {tuple(np.round(o % L, 6)) for o in oxygens}
    unchosen = np.array([o for o in all_o if tuple(np.round(o % L, 6)) not in chosen])
    assert len(unchosen) == 60 - 7
    assert np.linalg.norm(_mi(unchosen - frame[2]), axis=-1).min() >= radii[0]
    assert np.linalg.norm(_mi(unchosen - com), axis=-1).min() >= radii[1] - 1e-9
    # a rule that chose nothing (the cluster was full) has radius 0.0
    full = GeneratorDescriptors(str(path), GeneratorAtoms(3, ClusterSelection(n_waters=2, ranking=ranking)))
    assert full.cluster_radii(full.select(frame, BOX))[1] == 0.0
    with pytest.raises(ValueError):
        GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(22)).cluster_radii(np.zeros((22, 3)))


def _shuffle_waters(selected, n_solute, apw, rng):
    """`selected` with its water blocks in a random order (a generated cluster has any order)."""
    blocks = selected[n_solute:].reshape(-1, apw, 3)
    order = rng.permutation(len(blocks))
    assert (order != np.arange(len(blocks))).any()
    return np.concatenate([selected[:n_solute], blocks[order].reshape(-1, 3)])


def test_cluster_radii_any_water_order(tmp_path):
    """The radii are those of the rules applied again to the cluster, whatever its water order."""
    path, frame = _solvated(tmp_path, n_waters=60, seed=3)
    ranking = ({'reference': 'name H3', 'mode': 'min_distance', 'n': 2},
               {'reference': 'name C1', 'mode': 'com', 'n': 2},
               {'reference': None, 'mode': 'min_distance', 'n': -1})
    cluster = ClusterSelection(n_waters=7, ranking=ranking)
    descriptors = GeneratorDescriptors(str(path), GeneratorAtoms(3, cluster))
    selected = descriptors.select(frame, BOX)
    radii = descriptors.cluster_radii(selected)
    oxygens = selected[3::3]
    assert radii[0] == pytest.approx(np.linalg.norm(oxygens[:2] - selected[2], axis=-1).max())
    assert radii[1] == pytest.approx(np.linalg.norm(oxygens[2:4] - selected[0], axis=-1).max())
    rng = np.random.default_rng(1)
    for _ in range(5):
        assert descriptors.cluster_radii(_shuffle_waters(selected, 3, 3, rng)) == pytest.approx(radii)
    # the measured case of the review: NANMA in water, two rules on the two carbonyl oxygens
    water = NANMA.parent / 'cgaimmd_nanma_water'
    import MDAnalysis as mda
    universe = mda.Universe(str(water / 'conf.gro'))
    nanma = GeneratorDescriptors(str(water / 'conf.gro'), GeneratorAtoms(22, ClusterSelection(
        n_waters=8, ranking=({'reference': 'name OL', 'n': 3}, {'reference': 'name OR', 'n': -1}))))
    cluster = nanma.select(universe.atoms.positions, universe.dimensions)
    expected = nanma.cluster_radii(cluster)
    for _ in range(5):
        assert nanma.cluster_radii(_shuffle_waters(cluster, 22, 3, rng)) == pytest.approx(expected)


def test_ties_keep_lower_index(tmp_path):
    """Two waters at the same distance from the reference: the lower water index is chosen."""
    solute = np.array([[15.0, 15.0, 15.0], [16.5, 15.0, 15.0], [17.0, 15.9, 15.0]])
    positions, names = list(solute), ['C1', 'C2', 'H3']
    for z in (18.0, 12.0):                                      # mirror images in z = 15
        o = np.array([15.75, 15.0, z])
        positions += [o, o + [0.96, 0.0, 0.0], o + [-0.24, 0.0, 0.93]]
        names += ['OW', 'HW1', 'HW2']
    _write_gro(tmp_path / 'tie.gro', names, ['LIG'] * 3 + ['SOL'] * 6, [1] * 3 + [2] * 3 + [3] * 3,
               np.asarray(positions), BOX)
    for mode in ('min_distance', 'com'):
        descriptors = GeneratorDescriptors(str(tmp_path / 'tie.gro'), GeneratorAtoms(3, ClusterSelection(
            n_waters=1, ranking=({'reference': 'name C1 or name C2', 'mode': mode, 'n': -1},))))
        frame = np.asarray(positions, dtype=float)
        swapped = frame.copy()
        swapped[[3, 4, 5, 6, 7, 8]] = frame[[6, 7, 8, 3, 4, 5]]
        for x in (frame, swapped):
            d = np.linalg.norm(x[[3, 6]] - x[:2].mean(0), axis=-1)
            assert d[0] == d[1]                                      # a real tie
            np.testing.assert_allclose(descriptors.select(x, BOX)[3:], x[3:6])   # water 0


def test_masses_keyword(tmp_path):
    """masses= replaces the structure's guessed masses (NANMA's CL carbons are not chlorine)."""
    from aimmd.cgaimmd.preparation.topology import read_masses
    guessed = GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(22))
    assert guessed._masses[0] == pytest.approx(35.45, abs=0.01)             # CL read as Cl
    exact = GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(22),
                            masses=read_masses(NANMA / 'topol.top', expected=22))
    assert exact._masses[0] == pytest.approx(12.011)
    np.testing.assert_array_equal(exact._heavy, guessed._heavy)
    for bad in (np.ones(21), np.full(22, np.nan), -np.ones(22)):
        with pytest.raises(ValueError, match='masses'):
            GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(22), masses=bad)


def _write_pdb(path, names, resnames, resids, positions, box, conect=()):
    lines = [f'CRYST1{box[0]:9.3f}{box[1]:9.3f}{box[2]:9.3f}  90.00  90.00  90.00 P 1           1']
    for n, (name, resname, resid, p) in enumerate(zip(names, resnames, resids, positions), 1):
        lines.append(f'ATOM  {n:5d} {name:<4s} {resname:>3s} A{resid:4d}    '
                     f'{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           {name[0]}')
    lines += [f'CONECT{a:5d}{b:5d}' for a, b in conect]
    Path(path).write_text('\n'.join(lines + ['END']) + '\n')


def test_partial_file_bonds(tmp_path):
    """A file's bonds count only when they cover every atom the covalent rule bonds."""
    ligand = np.array([[5.0, 15.0, 15.0], [6.5, 15.0, 15.0], [7.0, 15.9, 15.0]])
    chain = np.stack([np.arange(20) * 1.4 + 1.0, np.full(20, 20.0), np.full(20, 15.0)], 1)
    positions = np.concatenate([ligand, chain])
    names = ['C1', 'C2', 'H3'] + [f'C{k}' for k in range(4, 24)]
    resnames, resids = ['LIG'] * 3 + ['CHN'] * 20, [1] * 3 + [2] * 20
    # CONECT for the ligand only: the chain would fall apart into 20 molecules
    _write_pdb(tmp_path / 'partial.pdb', names, resnames, resids, positions, BOX,
               conect=[(1, 2), (2, 3)])
    partial = GeneratorDescriptors(str(tmp_path / 'partial.pdb'), GeneratorAtoms(23))
    assert len(partial._walk) == 2                                    # the covalent rule
    out = partial.select((positions + [15.0, 0.0, 0.0]) % L, BOX)
    np.testing.assert_allclose(np.diff(out[3:, 0]), 1.4, atol=1e-3)   # the chain is whole
    # complete CONECT records are used as they are, even a bond the covalent rule would miss
    bonds = [(1, 2), (2, 3), (3, 4)] + [(k, k + 1) for k in range(4, 23)]
    _write_pdb(tmp_path / 'full.pdb', names, resnames, resids, positions, BOX, conect=bonds)
    assert len(GeneratorDescriptors(str(tmp_path / 'full.pdb'), GeneratorAtoms(23))._walk) == 1


def test_width(tmp_path):
    """__call__: (n_frames, 3 M) float32; empty input; validation of the settings."""
    import MDAnalysis as mda
    path, frame = _solvated(tmp_path)
    descriptors = GeneratorDescriptors(str(path), GeneratorAtoms(3, ClusterSelection(n_waters=5)))
    universe = mda.Universe(str(path))
    out = descriptors(universe.trajectory)
    assert out.shape == (1, 3 * 18) and out.dtype == np.float32
    np.testing.assert_allclose(out[0], descriptors.select(frame, BOX).reshape(-1), atol=1e-5)
    assert descriptors(universe.trajectory.ts).shape == (1, 54)            # a single Timestep
    empty = descriptors([])
    assert empty.shape == (0, 54) and empty.dtype == np.float32
    with pytest.raises(ValueError, match='atoms'):
        descriptors.select(frame[:-3], BOX)
    with pytest.raises(ValueError, match='orthorhombic'):
        descriptors.select(frame, [30.0, 30.0, 30.0, 90.0, 90.0, 60.0])
    with pytest.raises(ValueError):
        ClusterSelection(n_waters=0)
    with pytest.raises(ValueError, match='-1'):
        ClusterSelection(n_waters=2, ranking=({'reference': None, 'mode': 'com', 'n': 2},))
    with pytest.raises(ValueError, match='mode'):
        ClusterSelection(n_waters=2, ranking=({'reference': None, 'mode': 'max', 'n': -1},))
    with pytest.raises(ValueError, match='solute'):
        GeneratorDescriptors(str(path), GeneratorAtoms(3, ClusterSelection(n_waters=2, ranking=(
            {'reference': 'resname SOL', 'mode': 'com', 'n': -1},))))
    with pytest.raises(ValueError, match='solute'):
        GeneratorDescriptors(str(path), GeneratorAtoms(3, ClusterSelection(n_waters=2,
                                                                 center_selection='name OW')))
    with pytest.raises(ValueError, match='waters'):
        GeneratorDescriptors(str(path), GeneratorAtoms(3, ClusterSelection(n_waters=41)))
    with pytest.raises(ValueError):
        GeneratorDescriptors(str(path), GeneratorAtoms(3, ClusterSelection(n_waters=2, atoms_per_water=4)))
    with pytest.raises(ValueError):
        GeneratorAtoms(n_solute=0)
    assert GeneratorAtoms(3, ClusterSelection(n_waters=2)).n_waters == 2


def test_trajectory_frame_by_frame():
    """__call__ on a reader equals select frame by frame (NANMA free MD, split frames)."""
    import MDAnalysis as mda
    universe = mda.Universe(str(NANMA / 'conf.gro'), str(NANMA / 'free_md_A.xtc'),
                            str(NANMA / 'free_md_B.xtc'))
    descriptors = GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(22))
    out = descriptors(universe.trajectory[2:15])
    expected = [descriptors.select(ts.positions, ts.dimensions).reshape(-1)
                for ts in universe.trajectory[2:15]]
    assert out.shape == (13, 66) and out.dtype == np.float32
    np.testing.assert_allclose(out, np.asarray(expected, dtype=np.float32), atol=1e-6)


def test_coordinates_descriptors():
    """Every atom's coordinates as stored, flattened x0, y0, z0, x1, ..."""
    import MDAnalysis as mda
    universe = mda.Universe(str(NANMA / 'conf.gro'), str(NANMA / 'free_md_A.xtc'))
    out = coordinates_descriptors(universe.trajectory)
    assert out.shape == (10, 66) and out.dtype == np.float32
    universe.trajectory[3]
    np.testing.assert_array_equal(out[3], universe.atoms.positions.reshape(-1))
    assert coordinates_descriptors([]).shape == (0, 0)                  # no atom count to give
    empty = coordinates_descriptors(universe.trajectory[3:3])            # an empty slice: 3 N
    assert empty.shape == (0, 66) and empty.dtype == np.float32
    assert coordinates_descriptors(universe.trajectory[[]]).shape == (0, 66)


def test_two_solute_molecules(tmp_path):
    """Molecules after the first are translated next to the center (a host and its guest)."""
    solute = np.array([[2.0, 15.0, 15.0], [3.5, 15.0, 15.0],        # host, near the -x face
                       [29.0, 15.0, 15.0], [29.0, 16.5, 15.0]])     # guest, across the face
    names, resnames = ['C1', 'C2', 'C3', 'C4'], ['HST', 'HST', 'GST', 'GST']
    _write_gro(tmp_path / 'two.gro', names, resnames, [1, 1, 2, 2], solute, BOX)
    out = GeneratorDescriptors(str(tmp_path / 'two.gro'), GeneratorAtoms(4)).select(solute, BOX)
    np.testing.assert_allclose(out[:2], solute[:2])                  # the first stays
    np.testing.assert_allclose(out[2:], solute[2:] - [L, 0.0, 0.0])  # the second comes over


def test_three_solute_molecules_second_pass(tmp_path):
    """The second pass places every molecule next to the center, not next to the first molecule.

    A (C-C at x 1-2.5) is first, B (a heavy Br-Br at x 10-12.28) sits between A and C (C-C at x
    17-18.5, 16 A from A). Next to A alone, C would go to x -13; next to the solute's center of
    mass it stays where it is, and so do A and B.
    """
    solute = np.array([[1.0, 15.0, 15.0], [2.5, 15.0, 15.0],
                       [10.0, 15.0, 15.0], [12.28, 15.0, 15.0],
                       [17.0, 15.0, 15.0], [18.5, 15.0, 15.0]])
    names = ['C1', 'C2', 'BR1', 'BR2', 'C3', 'C4']
    _write_gro(tmp_path / 'three.gro', names, ['AAA', 'AAA', 'BBB', 'BBB', 'CCC', 'CCC'],
               [1, 1, 2, 2, 3, 3], solute, BOX)
    descriptors = GeneratorDescriptors(str(tmp_path / 'three.gro'), GeneratorAtoms(6))
    assert len(descriptors._walk) == 3                               # three molecules
    np.testing.assert_allclose(descriptors.select(solute, BOX), solute, atol=1e-9)
    # the same system shifted by a lattice vector comes out shifted back as a whole
    shifted = (solute + [20.0, 0.0, 0.0]) % L
    out = descriptors.select(shifted, BOX)
    np.testing.assert_allclose(out - out[0], solute - solute[0], atol=1e-9)


@pytest.mark.gromacs
def test_tpr_structure(tmp_path, gmx_executable):
    """A .tpr: its own bonds (the waters' filtered out), exact masses, the .gro's selection."""
    import subprocess

    import MDAnalysis as mda

    from aimmd.cgaimmd.preparation.topology import read_masses
    gmx = gmx_executable
    water = NANMA.parent / 'cgaimmd_nanma_water'
    for system, cluster, trajectory in ((NANMA, None, ['free_md_A.xtc', 'free_md_B.xtc']),
                                        (water, ClusterSelection(n_waters=6), [])):
        tpr = tmp_path / f'{system.name}.tpr'
        proc = subprocess.run([gmx, 'grompp', '-f', str(system / 'em.mdp'), '-c',
                               str(system / 'conf.gro'), '-p', str(system / 'topol.top'), '-o',
                               str(tpr), '-po', str(tmp_path / 'mdout.mdp'), '-maxwarn', '2'],
                              cwd=tmp_path, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-2000:]
        generator_atoms = GeneratorAtoms(22, cluster)
        from_tpr = GeneratorDescriptors(str(tpr), generator_atoms)
        from_gro = GeneratorDescriptors(str(system / 'conf.gro'), generator_atoms)
        assert len(from_tpr._walk) == 1                          # NANMA, by the file's bonds
        np.testing.assert_allclose(from_tpr._masses, read_masses(system / 'topol.top'), rtol=1e-6)
        assert from_tpr._masses[0] == pytest.approx(12.011, abs=1e-3)       # CL is a carbon
        assert from_gro._masses[0] == pytest.approx(35.45, abs=0.01)        # guessed: chlorine
        bonds = np.asarray(mda.Universe(str(tpr)).bonds.indices)
        if cluster is not None:
            assert (bonds >= 22).any()                           # the waters' bonds are filtered
        universe = mda.Universe(str(system / 'conf.gro'),
                                *[str(system / name) for name in trajectory])
        for ts in universe.trajectory:
            np.testing.assert_allclose(from_tpr.select(ts.positions, ts.dimensions),
                                       from_gro.select(ts.positions, ts.dimensions), atol=1e-9)


def test_old_names_of_the_generator_atoms(recwarn):
    """Old module and class names and ``.flow_atoms`` are read for one release, one warning each.

    The archived production params build their descriptors as
    ``FlowDescriptors(structure, FlowAtoms(n_solute=...), masses=read_masses(top))``.
    """
    from aimmd.cgaimmd import _legacy
    _legacy._reset()
    from aimmd.flow.descriptors import FlowAtoms, FlowDescriptors
    from aimmd.flow.topology import read_masses
    assert FlowAtoms is GeneratorAtoms and FlowDescriptors is GeneratorDescriptors
    descriptors = FlowDescriptors(str(NANMA / 'conf.gro'), FlowAtoms(n_solute=22),
                                  masses=read_masses(NANMA / 'topol.top'))
    assert descriptors.flow_atoms is descriptors.generator_atoms
    assert descriptors.flow_atoms == GeneratorAtoms(22)
    messages = [str(w.message) for w in recwarn if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 4, messages     # FlowAtoms, FlowDescriptors, read_masses, .flow_atoms
    assert any('GeneratorDescriptors.flow_atoms is deprecated; use '
               'GeneratorDescriptors.generator_atoms' in m for m in messages)
