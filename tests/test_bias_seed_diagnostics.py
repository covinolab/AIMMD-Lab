"""
Unit tests for the always-on bias validity diagnostics in
`aimmd.pathensemble.bias_utils`.

Two failure modes are covered, both of which produced a wrong published rate
before the checks existed.

Bias zero point
---------------
`gamma = <exp(bias)>` is absolute: adding a constant c to every recorded bias
value multiplies every gamma, and therefore the rate, by exp(c). A frozen PLUMED
OPES fill floored at -BARRIER is mapped to a non-negative bias by adding BARRIER
inside `params.bias_function`; add the wrong number and every rate is off by a
constant factor with nothing else to show for it.

test_zero_point_passes_when_r_is_bias_free
test_zero_point_catches_a_constant_offset      — reports the offset and the factor
test_zero_point_flags_negative_bias            — V >= 0 for a fill that raises energy
test_zero_point_survives_an_empty_ensemble     — early return carries every key

More than one zero point in one ensemble
----------------------------------------
The offset above is one question; whether the ensemble has *one* zero point at
all is a different one, with a different remedy. `register_path` writes a
shooting path's `<traj>.bias.npy` once and never revisits it, and
`_cache_bias_files` rewrites a cache only while it is SHORTER than its
trajectory — so editing `params.bias_function` mid-campaign leaves every
already-complete cache on the old zero point and the ensemble genuinely holds
two. That is calixarene_G2_opes_v5: 67 caches written 2026-09-10 with a bias
floor of -3.0068 kT, 106 caches written 2026-09-13/14 with a floor of 0.0000 kT.

A mixture has no single offset — one subset of the gammas is wrong by a
constant factor while the rest are right — so `offset`/`factor` are nan and the
remedy is a list of caches to rebuild. It is decided by a census of the reactive
bias floor per cache FILE, not per path: one `<traj>.bias.npy` is written in one
call by one `bias_function`, while a `Path` routinely spans several `.partNNNN`
caches of different vintage, so a per-path statistic is wrong either way.

Two things the census must not do. It must not pick its reference level by
population — the correct zero point is 0 by definition, so the level AT zero is
the reference and a frame-count majority names the already-correct caches for
deletion for as long as the stale ones outnumber them (most of
calixarene_G2_opes_v5). And it must not report a scalar it did not establish:
`len(levels) <= 1` is also what an ensemble the census could not read looks
like, so `offset`/`factor` are gated on `census_conclusive` — one level with
every reactive cache file placed on it — and not on `uniform_ok`.

test_mixed_zero_points_are_reported_as_a_mixture
test_mixed_zero_points_never_claim_a_zero_offset   — no "+0.000 kT, factor 1.000"
test_mixed_zero_points_name_the_stale_caches       — the remedy is an `rm`, not a number
test_mixed_zero_points_with_the_stale_block_in_the_majority
                                                   — the level AT zero is the reference
test_mixed_zero_points_with_no_level_at_zero       — then every placed cache is named
test_mixed_zero_points_shifted_up_are_caught       — neither old test fires here
test_a_path_spanning_a_stale_and_a_fresh_cache_is_caught
test_census_degrades_when_per_frame_filenames_are_unavailable
test_clean_ensemble_passes_and_says_so             — no false alarm
test_clean_ensemble_tolerates_a_reactive_boundary_tail
test_pure_offset_still_reports_offset_and_factor   — one level, in the wrong place
test_negative_bias_alone_reports_its_own_failure
test_a_mixture_below_the_voting_bar_publishes_no_offset
                                                   — one level is not one zero point
test_the_healthy_line_accounts_for_every_cache_file
test_a_huge_offset_does_not_crash_the_report       — exp(-median) underflows to 0.0
test_result_key_set_is_stable                      — every branch, same keys
test_factor_is_finite_exactly_when_a_constant_correction_is_valid

The corrected-rate line in bias_reweighted_rates
------------------------------------------------
`bias_reweighted_rates` prints a "zero-point corrected" rate under
`not median_ok and isfinite(factor)`. Both halves are load-bearing: `isfinite`
rejects a mixture, `median_ok` rejects a negative recorded bias under a correct
zero point (where `factor` is a perfectly finite 1.0). Gating on `ok` instead —
which also carries those two failures — printed a "corrected" rate identical to
the uncorrected one under an `offset +0.000 kT` label.

test_reweighted_rates_still_correct_a_pure_offset  — the working case, preserved
test_reweighted_rates_print_no_correction_for_a_mixture
test_reweighted_rates_print_no_correction_for_an_unreadable_census
test_reweighted_rates_print_no_correction_for_negative_bias_alone
test_reweighted_rates_survive_an_ensemble_with_no_bias

(The estimator arithmetic of `bias_reweighted_rates` — that 1/k is a frame sum,
and that a caller-supplied L which is not the counted window is reported — is
covered in tests/test_bias_utils.py, not here.)

Non-equilibrium free-basin seeds
--------------------------------
`k = N / sum(w*L*gamma)` is a mean-first-passage estimator; it measures the
escape rate only if each passage starts from the equilibrium distribution inside
the state. Re-seeding at the state boundary breaks that when in-state relaxation
is not fast compared with escape. The statistic is the realised acceleration
Gamma_i/Gamma_eq, not a "did it reach the deep well" test on max(bias): the fill
is not monotonic in depth (for calixarene-G2 the recorded bias peaks at 6.8 kT
around d = 0.42 nm and falls back to ~0 below d = 0.27 nm, which the frozen bias
never filled), so a depth criterion mis-ranks trajectories.

test_seed_report_is_quiet_for_an_equilibrated_basin
test_seed_report_flags_low_boost_passages      — the 4-of-5 G2-v2 signature
test_seed_report_flags_skewed_first_passages   — median/mean far below ln2
test_seed_report_excludes_open_trajectories    — censored, not counted
test_seed_report_ignores_shooting_paths        — only free trajectories
test_seed_report_is_silent_without_a_fill      — unbiased run: no boost ratio
test_seed_report_without_free_trajectories_is_a_single_line
test_ks_exponential_is_small_for_an_exponential_sample
test_ks_exponential_is_large_for_a_spike_plus_tail
"""

import contextlib
import warnings as _warnings

import numpy as np
import pytest

from aimmd.pathensemble.bias_utils import (SEED_BOOST_FRACTION,
                                           _ks_exponential,
                                           bias_reweighted_rates,
                                           check_bias_zero_point,
                                           report_nonequilibrium_seeds)

KT = 8.314462618e-3 * 300.0          # kJ/mol at 300 K, as in the run's params

#: 7.5 kJ/mol — the BARRIER the first calixarene_G2_opes_v5 job was short by.
GAP = 7.5 / KT                       # == 3.00681 kT
#: exp(GAP): the factor every gamma on a stale cache is wrong by.
GAP_FACTOR = float(np.exp(GAP))      # == 20.2227

#: The fill plateau inside the states, for a BARRIER of 15 kJ/mol.
PLATEAU = 15.0 / KT


@contextlib.contextmanager
def no_warnings():
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        yield


def _flat(text):
    """The report as one line of prose: no wrap, no block markers.

    A content assertion must not depend on where `textwrap` breaks a line,
    because the `***` that opens each continuation line would otherwise land in
    the middle of the phrase being asserted ("are too *** fast by somewhere").
    """
    words = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('***'):
            line = line[3:].lstrip()
        if line.startswith('WARNING:'):
            line = line[len('WARNING:'):].lstrip()
        words.extend(line.split())
    return ' '.join(words)


def _rate_on(out, label):
    """The number printed on the (single) line containing *label*."""
    lines = [line for line in out.splitlines() if label in line]
    assert len(lines) == 1, f'{label!r} appears {len(lines)} times'
    return float(lines[0].split(':')[-1].replace('[1/dt]', '').strip())


class Block:
    """One `Path.split` block: its files, its state triplet, its arrays.

    The zero-point census attributes each frame to the cache file it was read
    from through `Path.filenames`, which the real class defines as
    `np.repeat(self._fnames, self.lengths).astype(str)` — per-frame and aligned
    with what `_get` returns. This double mirrors that, so one block can also
    stand for a path that spans several cache files (`Block.spanning`).
    """

    def __init__(self, fname, type_, states=None, bias=None):
        self._fnames = [fname]
        self.type = type_
        self._states = None if states is None else np.asarray(list(states),
                                                              dtype='<U1')
        self._bias = None if bias is None else np.asarray(bias, dtype=float)
        self._lengths = [len(self)]

    @classmethod
    def spanning(cls, caches, type_='ARBR'):
        """One path over several `<traj>.bias.npy` caches.

        A free trajectory is stored as `.partNNNN.xtc` segments and one `Path`
        routinely spans several of them, each with its own cache file written
        at its own time — which is exactly why the census is per file and not
        per path. *caches* is a list of `(filename, states, bias)`.
        """
        block = cls(caches[0][0], type_,
                    states=''.join(states for _, states, _ in caches),
                    bias=[value for _, _, bias in caches for value in bias])
        block._fnames = [name for name, _, _ in caches]
        block._lengths = [len(states) for _, states, _ in caches]
        return block

    @property
    def lengths(self):
        """Per-file segment lengths, as `Path.lengths`."""
        return np.array(self._lengths, dtype=int)

    @property
    def filenames(self):
        """Per-frame source file, as `Path.filenames`."""
        return np.repeat(self._fnames, self.lengths).astype(str)

    def __len__(self):
        return 0 if self._states is None else len(self._states)

    def _get(self, attribute, raise_if_missing=False):
        if attribute in ('states', 'true_states'):
            if self._states is None:
                raise AttributeError(attribute)
            return self._states.copy()
        if attribute == 'bias':
            if self._bias is None:
                if raise_if_missing:
                    raise TypeError('no bias')
                return None
            return self._bias.copy()
        raise AttributeError(attribute)


class Ensemble(list):
    """A list of blocks is enough for both diagnostics."""


# ════════════════════════════════════════════════════════════════════════════
# Bias zero point
# ════════════════════════════════════════════════════════════════════════════

def _ensemble_with_offset(offset):
    """Two blocks: an in-A dwell at the fill plateau, and an R excursion."""
    return Ensemble([
        Block('run1/freeA/traj000001.part0001.xtc', 'AAAA',
              states='A' * 6, bias=[PLATEAU + offset] * 6),
        Block('run1/freeA/traj000001.part0002.xtc', 'ARBR',
              states='ARRRB', bias=[PLATEAU + offset] + [offset] * 4),
    ])


def _ensemble_with_a_negative_frame():
    """An in-A dwell holding one impossible frame, on a correct zero point.

    The R frames sit at 0, so the zero point itself is where it belongs: this
    is neither a constant offset nor a mixture, and no scalar repairs it. The
    excursion carries ZERO_POINT_MIN_FILE_FRAMES reactive frames so that its
    cache can vote and the census is conclusive — that is the premise of
    test_reweighted_rates_print_no_correction_for_negative_bias_alone, which
    needs a FINITE factor to show that `median_ok` is the half that suppresses.
    """
    return Ensemble([
        Block('run1/freeA/traj000001.part0001.xtc', 'AAAA',
              states='AAAA', bias=[0.0, 0.0, -2.0, 0.0]),
        Block('run1/freeA/traj000001.part0002.xtc', 'ARBR',
              states='ARRRB', bias=[0.0, 0.0, 0.0, 0.0, 0.0]),
    ])


def _cache(name, offset=0.0, n_reactive=5):
    """One `<traj>.bias.npy`'s worth of frames, at zero point *offset*.

    `A` at the fill plateau, `n_reactive` bias-free `R` frames, `B` at the
    plateau — the shape a shooting path through the barrier actually has.
    """
    return (name,
            'A' + 'R' * n_reactive + 'B',
            [PLATEAU + offset] + [offset] * n_reactive + [PLATEAU + offset])


def _cached(name, offset=0.0, n_reactive=5):
    """One path holding exactly one cache file, at zero point *offset*."""
    return Block.spanning([_cache(name, offset, n_reactive)])


def _mixed_ensemble(n_stale=2, n_fresh=5, gap=-GAP):
    """`n_stale` caches at zero point *gap*, `n_fresh` at zero point 0.

    calixarene_G2_opes_v5 in miniature: `params.bias_function` shipped a
    constant 7.5 kJ/mol too small, the job was killed, params were corrected
    and the job restarted — but every cache that was already complete kept the
    old zero point, so the ensemble now genuinely holds two of them.
    """
    return Ensemble(
        [_cached(f'run1/chainR0/path{i:06d}.xtc', gap)
         for i in range(n_stale)]
        + [_cached(f'run1/chainR0/path{i:06d}.xtc', 0.0)
           for i in range(n_stale, n_stale + n_fresh)])


def _clean_ensemble(n=6):
    return Ensemble([_cached(f'run1/chainR0/path{i:06d}.xtc', 0.0)
                     for i in range(n)])


def _pure_offset_ensemble(n=5, offset=-GAP):
    """Every cache on one and the same wrong zero point."""
    return Ensemble([_cached(f'run1/chainR0/path{i:06d}.xtc', offset)
                     for i in range(n)])


def test_zero_point_passes_when_r_is_bias_free(capsys):
    result = check_bias_zero_point(_ensemble_with_offset(0.0), 'ARB')
    assert result['ok']
    assert result['median'] == pytest.approx(0.0)
    assert result['factor'] == pytest.approx(1.0)
    assert 'Bias zero point' in capsys.readouterr().out


def test_zero_point_catches_a_constant_offset():
    """The G2-v3 case: plumed BARRIER 18, params shift 15 -> -3 kJ/mol.

    The recorded bias in R is then -3/kT = -1.2027 kT and every gamma is
    exp(-1.2027) = 0.3004 of its correct value, so the reweighted rate is
    exp(+1.2027) = 3.329x too fast. The check must name that factor.
    """
    offset = -3.0 / KT
    with pytest.warns(UserWarning, match='zero point'):
        result = check_bias_zero_point(_ensemble_with_offset(offset), 'ARB')
    assert not result['ok']
    assert result['median'] == pytest.approx(-1.2027, abs=1e-3)
    assert result['factor'] == pytest.approx(3.329, rel=1e-3)
    assert 'BARRIER' in result['report']


def test_zero_point_flags_negative_bias():
    """A fill that raises the energy gives V >= 0; negative cannot be physical."""
    with pytest.warns(UserWarning):
        result = check_bias_zero_point(_ensemble_with_a_negative_frame(), 'ARB')
    assert not result['ok']
    assert result['minimum'] == pytest.approx(-2.0)


def test_zero_point_survives_an_empty_ensemble(capsys):
    """The early return must carry every key the caller indexes.

    `bias_reweighted_rates` reads `zero_point['median_ok']` and
    `zero_point['factor']` unconditionally, so a short early-return dict would
    turn an ensemble with no reactive frames into a KeyError.
    """
    result = check_bias_zero_point(Ensemble(), 'ARB')
    assert result['ok'] and result['n_frames'] == 0
    assert set(result) == EXPECTED_KEYS
    assert result['offset'] == 0.0 and result['factor'] == 1.0
    assert result['fix_command'] is None
    assert 'no reactive frames' in capsys.readouterr().out


# ── more than one zero point in one ensemble ────────────────────────────────

def test_mixed_zero_points_are_reported_as_a_mixture():
    """Two zero points, the fresh caches in the majority."""
    with pytest.warns(UserWarning, match='not unique'):
        result = check_bias_zero_point(
            _mixed_ensemble(n_stale=2, n_fresh=5), 'ARB')

    assert not result['ok']
    assert result['median_ok'], 'the median itself is fine — that is the trap'
    assert not result['uniform_ok'], 'the mixture must be what fails'
    assert result['n_levels'] == 2
    values = sorted(level['value'] for level in result['levels'])
    assert values[0] == pytest.approx(-GAP, abs=1e-3)
    assert values[1] == pytest.approx(0.0, abs=1e-9)
    assert result['mismatched_fraction'] == pytest.approx(2 / 7, rel=1e-9)
    assert result['gap'] == pytest.approx(-GAP, abs=1e-3)
    assert result['gap_factor'] == pytest.approx(GAP_FACTOR, rel=1e-6)


def test_mixed_zero_points_never_claim_a_zero_offset():
    """The reported symptom: a "correction" of +0.000 kT by a factor 1.000.

    `ok` used to be `abs(median) <= tolerance and not (minimum < -tolerance)`
    while `offset`/`factor` were derived from the median alone, so a mixture
    whose fresh caches outnumber the stale ones warned with every number in
    the message meaningless. `offset` and `factor` describe a shift of the
    WHOLE ensemble, so for a mixture they must not exist at all — nan, not zero.
    """
    with pytest.warns(UserWarning) as caught:
        result = check_bias_zero_point(_mixed_ensemble(), 'ARB')

    assert np.isnan(result['offset']), 'a mixture has no single offset'
    assert np.isnan(result['factor']), 'a mixture has no single factor'

    report = _flat(result['report'])
    assert 'off by +0.000' not in report
    assert 'Subtract +0.000' not in report
    assert 'factor 1.000' not in report
    # ... and the numbers it does print are about the defect that was found.
    assert f'{GAP:.3f} kT apart' in report
    assert f'{GAP_FACTOR:.3f}' in report

    messages = ' '.join(str(w.message) for w in caught)
    assert '+0.000 kT' not in messages
    assert 'factor 1.000' not in messages


def test_mixed_zero_points_name_the_stale_caches():
    """The remedy is a list of files to rebuild, not a number to subtract."""
    with pytest.warns(UserWarning):
        result = check_bias_zero_point(
            _mixed_ensemble(n_stale=2, n_fresh=5), 'ARB')

    assert result['mismatched_files'] == ['run1/chainR0/path000000.xtc',
                                          'run1/chainR0/path000001.xtc']
    assert result['fix_command'].startswith('rm -f ')
    assert result['fix_command_is_complete']
    for name in result['mismatched_files']:
        assert f'{name}.bias.npy' in result['fix_command']
    # a cache that is already right must not be named for deletion
    assert 'path000002.xtc.bias.npy' not in result['fix_command']
    # the short, complete list is spelled out in the log too
    assert 'rm -f' in result['report']


def test_mixed_zero_points_with_the_stale_block_in_the_majority():
    """The other side of the crossover, where the median lands on the stale level.

    The old code then reported a genuine-looking `offset -3.007 kT` and scaled
    the whole ensemble by it — wrong for the caches that were already right.
    It is still a mixture, and a scalar must still be refused.

    The reference level must be the one AT zero, not the one with the most
    frames: the correct zero point is 0 by definition. Picking by population
    inverts the entire remedy here — it names the two caches that are already
    right, prints `rm -f` on them under `fix_command_is_complete`, and says the
    rates are too slow when they are too fast. On calixarene_G2_opes_v5 the
    stale caches were the frame majority until 2026-09-13, i.e. for most of the
    campaign, so this is the regime the check actually ran in.
    """
    with pytest.warns(UserWarning, match='not unique'):
        result = check_bias_zero_point(
            _mixed_ensemble(n_stale=5, n_fresh=2), 'ARB')

    assert result['median'] == pytest.approx(-GAP, abs=1e-3)
    assert not result['median_ok'], 'the median is now on the stale level'
    assert not result['uniform_ok']
    assert np.isnan(result['offset']) and np.isnan(result['factor'])
    assert result['mismatched_files'] == [f'run1/chainR0/path{i:06d}.xtc'
                                          for i in range(5)], \
        'the five STALE caches, not the two that are already right'
    for i in (5, 6):
        assert f'path{i:06d}.xtc.bias.npy' not in result['fix_command']
    report = _flat(result['report'])
    assert 'too fast' in report, 'a floor below zero makes gamma too small'
    assert '(at zero)' in report


def test_mixed_zero_points_with_no_level_at_zero():
    """Two wrong vintages and nothing at zero: every placed cache is named.

    With no level within `tolerance` of 0 there is no level to trust, so the
    reference is not "the closest" — it is nothing, and the remedy is every
    cache the census placed, not the smaller half of two wrong answers.
    """
    pe = Ensemble(
        [_cached(f'run1/chainR0/path{i:06d}.xtc', -GAP) for i in range(3)]
        + [_cached(f'run1/chainR0/path{i:06d}.xtc', -2 * GAP)
           for i in range(3, 6)])
    with pytest.warns(UserWarning, match='not unique'):
        result = check_bias_zero_point(pe, 'ARB')

    assert result['n_levels'] == 2
    assert not result['uniform_ok'] and not result['median_ok']
    assert np.isnan(result['offset'])
    assert result['mismatched_files'] == [f'run1/chainR0/path{i:06d}.xtc'
                                          for i in range(6)]
    assert 'NO level here is at zero' in _flat(result['report'])


def test_mixed_zero_points_shifted_up_are_caught():
    """The mirror case, which neither of the two older tests can see.

    With the stale block ABOVE the majority no frame is negative, so the
    `minimum < -tolerance` guard is silent, and the median is still 0, so the
    median test is silent too. The per-cache-file census is the only thing that
    catches it — and it must, because those files' gamma are still a factor
    exp(GAP) wrong.
    """
    with pytest.warns(UserWarning, match='not unique'):
        result = check_bias_zero_point(
            _mixed_ensemble(n_stale=2, n_fresh=5, gap=+GAP), 'ARB')

    assert result['median_ok'], 'the median test cannot see an upward shift'
    assert result['positive_ok'], 'nothing is negative here'
    assert not result['uniform_ok'] and not result['ok']
    assert np.isnan(result['factor'])
    assert 'too slow' in _flat(result['report'])


def test_a_path_spanning_a_stale_and_a_fresh_cache_is_caught():
    """One free trajectory, two `.partNNNN` caches of different vintage.

    `_cache_bias_files` rewrites a part only while it is still growing, so the
    finished part keeps the old zero point while the part that was still being
    written gets the new one — inside a single `Path`. A per-PATH statistic
    classifies such a path wholly in or wholly out by frame majority, and is
    wrong either way; the per-cache-file census places each part separately.
    """
    stem = 'run1/freeA/traj000001'
    pe = Ensemble([
        Block.spanning([_cache(f'{stem}.part0001.xtc', -GAP, n_reactive=4),
                        _cache(f'{stem}.part0002.xtc', 0.0, n_reactive=12)]),
        _cached('run1/chainR0/path000001.xtc', 0.0, n_reactive=12),
    ])
    with pytest.warns(UserWarning, match='not unique'):
        result = check_bias_zero_point(pe, 'ARB')

    assert not result['uniform_ok']
    assert result['n_files'] == 3, 'three cache files, not two paths'
    assert result['mismatched_files'] == [f'{stem}.part0001.xtc']
    assert np.isnan(result['offset'])


def test_census_degrades_when_per_frame_filenames_are_unavailable(monkeypatch):
    """Without `Path.filenames` the census falls back to one unit per path.

    A weaker detector — a path spanning two vintages is then classified whole —
    but it must still run, and the file-level remedy must be withheld rather
    than printed naming paths as if they were caches.
    """
    monkeypatch.delattr(Block, 'filenames')
    with pytest.warns(UserWarning, match='not unique'):
        result = check_bias_zero_point(_mixed_ensemble(), 'ARB')

    assert result['census_degraded']
    assert not result['uniform_ok'], 'still caught, one pseudo-file per path'
    assert not result['fix_command_is_complete']
    assert 'rm -f' not in result['report']
    assert 'census ran per PATH' in _flat(result['report'])


def test_clean_ensemble_passes_and_says_so(capsys):
    with no_warnings():
        result = check_bias_zero_point(_clean_ensemble(), 'ARB')

    assert result['ok']
    assert result['median_ok'] and result['uniform_ok'] and result['positive_ok']
    assert result['n_levels'] == 1
    assert result['offset'] == pytest.approx(0.0)
    assert result['factor'] == pytest.approx(1.0)
    assert result['mismatched_files'] == []
    assert result['fix_command'] is None
    out = capsys.readouterr().out
    assert 'WARNING' not in out
    assert 'one zero point at +0.000 kT' in out


def test_clean_ensemble_tolerates_a_reactive_boundary_tail():
    """The genuine near-boundary OPES tail must not be read as a second level.

    `params.states_function` calls a frame R while the biasing engine's own CV
    still puts it inside the filled region, so a few R frames sit several kT
    above the floor — 41 of 28171 R frames (0.15 %) in calixarene_G2_opes_v5,
    at 4.4-5.7 kT. That is `check_reactive_bias`' business, not a zero point.
    """
    pe = _clean_ensemble(n=8)
    # give one path a two-frame tail at 5 kT, as the A*/PLUMED-d mismatch does
    pe[0]._bias[1:3] = 5.0

    with no_warnings():
        result = check_bias_zero_point(pe, 'ARB')

    assert result['ok'], 'a tail is not a second zero point'
    assert result['n_levels'] == 1
    assert result['mismatched_files'] == []


def test_pure_offset_still_reports_offset_and_factor():
    """Every cache on one wrong zero point: still a correctable constant.

    This is the one failure a scalar repairs, seen through the census: one
    level, in the wrong place. It must keep reporting the number.
    """
    with pytest.warns(UserWarning, match=r'off by -3\.007 kT'):
        result = check_bias_zero_point(_pure_offset_ensemble(), 'ARB')

    assert result['uniform_ok'], 'one level, in the wrong place'
    assert not result['median_ok']
    assert result['median'] == pytest.approx(-GAP, abs=1e-3)
    assert result['offset'] == pytest.approx(-GAP, abs=1e-3)
    assert result['factor'] == pytest.approx(GAP_FACTOR, rel=1e-6)
    assert 'BARRIER' in result['report']


def test_negative_bias_alone_reports_its_own_failure():
    """A negative recorded bias under a correct, single zero point.

    `offset`/`factor` stay 0.0/1.0 here — they are voided only for a mixture —
    so `isfinite(factor)` on its own does NOT suppress a bogus "corrected"
    line in this case; the `median_ok` half of the call-site gate does. This
    pins the half the mixture tests do not exercise.
    """
    with pytest.warns(UserWarning, match='negative'):
        result = check_bias_zero_point(_ensemble_with_a_negative_frame(), 'ARB')

    assert result['median_ok'] and result['uniform_ok']
    assert not result['positive_ok'] and not result['ok']
    assert result['n_negative_frames'] == 1
    assert result['negative_files'] == ['run1/freeA/traj000001.part0001.xtc']
    report = _flat(result['report'])
    assert 'Subtract +0.000' not in report, \
        'there is nothing to subtract: the zero point is where it belongs'
    assert 'off by +0.000' not in report


def test_a_mixture_below_the_voting_bar_publishes_no_offset():
    """One level is evidence of one zero point only if every cache is on it.

    A cache votes with at least ZERO_POINT_MIN_FILE_FRAMES reactive frames, so
    a mixture carried entirely by shorter caches produces NO level at all. That
    is the absence of a finding, not a finding of uniformity: gating `offset` on
    `len(levels) <= 1` published a finite `-3.007 kT` / `20.223` for a genuine
    two-zero-point ensemble and printed a "corrected" rate from it — the exact
    line this check exists to delete. `census_conclusive` is the gate instead.
    """
    pe = Ensemble(
        [_cached(f'clean{i}.xtc', 0.0, n_reactive=2) for i in range(3)]
        + [_cached(f'stale{i}.xtc', -GAP, n_reactive=2) for i in range(8)])
    with pytest.warns(UserWarning):
        result = check_bias_zero_point(pe, 'ARB')

    assert result['n_levels'] == 0, 'nothing cleared the voting bar'
    assert result['uniform_ok'], 'no level found is not two levels found'
    assert not result['census_conclusive']
    assert np.isnan(result['offset']) and np.isnan(result['factor'])

    report = _flat(result['report'])
    assert 'finds no second zero point' not in report, \
        'the census found nothing; it may not be quoted as clearing the run'
    assert 'could not establish' in report
    assert 'No zero-point corrected rate is printed below' in report
    # and the plural prose about those files reads as English
    assert 'cache files have reactive frames' in report


def test_the_healthy_line_accounts_for_every_cache_file():
    """The pass line is a partition, so it must name all three categories.

    Caches shifted UP whose reactive frames spread over several kT never vote
    (floor_mass below ZERO_POINT_MIN_FLOOR_MASS) and match no level, so they are
    `unplaced`. The line used to read "one zero point across 20 of 24 cache
    files (0 hold no bias-free frame)" — a per-file account that does not add
    up, printed as the reassuring summary of an ensemble holding two zero
    points. It must disclose them, and no scalar may be published.
    """
    def _band(name, base):
        return (name, 'A' + 'R' * 5 + 'B',
                [PLATEAU + base] + [base + v for v in (0.0, .9, 1.8, 2.7, 3.6)]
                + [PLATEAU + base])

    pe = Ensemble([_cached(f'clean{i}.xtc', 0.0) for i in range(20)]
                  + [Block.spanning([_band(f'up{i}.xtc', GAP)])
                     for i in range(4)])
    with no_warnings():
        result = check_bias_zero_point(pe, 'ARB')

    assert result['n_levels'] == 1 and len(result['unplaced_files']) == 4
    assert not result['census_conclusive']
    assert np.isnan(result['factor']), 'the census did not read four caches'

    line = [ln for ln in result['report'].splitlines()
            if 'one zero point at' in ln]
    assert len(line) == 1
    assert 'across 20 of 24 cache files' in line[0]
    assert '4 on no recognised zero point' in line[0], \
        '20 + 0 != 24: the line may not claim a census it did not perform'


def test_a_huge_offset_does_not_crash_the_report():
    """A units error in `bias_function` underflows exp(-median) to 0.0.

    The inflation factor printed in the report used to be
    `max(factor, 1.0 / factor)`, which raises ZeroDivisionError for a median
    above ~746 kT — a 1000x kJ/mol-vs-kT mistake reaches that easily, and a
    diagnostic that dies on a wrong bias is the one case it exists for.
    """
    with pytest.warns(UserWarning, match='zero point'):
        result = check_bias_zero_point(
            _pure_offset_ensemble(offset=800.0), 'ARB')

    assert result['median'] == pytest.approx(800.0)
    assert result['factor'] == 0.0
    assert 'inf' in result['report']


# ── the return-dict contract ────────────────────────────────────────────────

EXPECTED_KEYS = {
    'median', 'minimum', 'n_frames', 'offset', 'factor', 'ok',
    'median_ok', 'uniform_ok', 'positive_ok', 'census_conclusive',
    'gap', 'gap_factor',
    'levels', 'n_levels', 'n_files',
    'mismatched_files', 'mismatched_fraction',
    'negative_files', 'n_negative_frames',
    'unplaced_files', 'no_reactive_files', 'dropped_levels',
    'census_degraded', 'fix_command', 'fix_command_is_complete', 'report',
}


def test_result_key_set_is_stable():
    """Every branch returns the same keys, so no caller needs `.get()`."""
    cases = [_clean_ensemble(), _mixed_ensemble(), Ensemble(),
             _ensemble_with_offset(-GAP), _ensemble_with_offset(0.0),
             _ensemble_with_a_negative_frame(),
             Ensemble([_cached(f'thin{i}.xtc', 0.0, n_reactive=1)
                       for i in range(3)])]
    for pe in cases:
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore')
            result = check_bias_zero_point(pe, 'ARB')
        assert set(result) == EXPECTED_KEYS
        assert result['ok'] == (result['median_ok'] and result['uniform_ok']
                                and result['positive_ok'])


@pytest.mark.parametrize('name,builder,finite', [
    ('clean', _clean_ensemble, True),
    ('pure offset', _pure_offset_ensemble, True),
    ('mixture', _mixed_ensemble, False),
    ('mixture, stale majority', lambda: _mixed_ensemble(5, 2), False),
    ('mixture, shifted up', lambda: _mixed_ensemble(2, 5, +GAP), False),
    ('empty', Ensemble, True),
    ('mixture below the voting bar',
     lambda: Ensemble([_cached(f'clean{i}.xtc', 0.0, 2) for i in range(3)]
                      + [_cached(f'stale{i}.xtc', -GAP, 2) for i in range(8)]),
     False),
])
def test_factor_is_finite_exactly_when_a_constant_correction_is_valid(
        name, builder, finite):
    """The contract a caller relies on.

    `isfinite(factor)` if and only if `census_conclusive`: the per-file census
    placed every cache file with reactive frames on one and the same level. NOT
    `uniform_ok` — the last case is a genuine mixture in which no cache clears
    the voting bar, so `uniform_ok` is True while nothing was established.

    Necessary but not sufficient for "dividing every rate by factor is a
    correction worth printing": a negative bias under a correct median leaves
    factor at 1.0 (see test_negative_bias_alone_reports_its_own_failure), and it
    is the `not median_ok` half of the call-site gate that suppresses it there.
    """
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        result = check_bias_zero_point(builder(), 'ARB')
    assert bool(np.isfinite(result['factor'])) is finite, name
    assert bool(np.isfinite(result['offset'])) is finite, name
    assert bool(result['census_conclusive']) is finite, name


# ════════════════════════════════════════════════════════════════════════════
# The corrected-rate line in bias_reweighted_rates
# ════════════════════════════════════════════════════════════════════════════

def _rates_on(pe, capsys):
    """Run the full reweighting step on *pe* and return (k12, k21, stdout)."""
    weights = np.ones(len(pe))
    lengths = np.array([len(path) for path in pe], dtype=float)
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        k12, k21, _, _ = bias_reweighted_rates(
            pe, weights, weights, lengths=lengths, states='ARB',
            seed_diagnostics=False)
    return k12, k21, capsys.readouterr().out


def test_reweighted_rates_still_correct_a_pure_offset(capsys):
    """The working case must be preserved exactly: the number is still printed."""
    k12, _, out = _rates_on(_pure_offset_ensemble(), capsys)

    assert f'zero-point corrected (offset {-GAP:+.3f} kT)' in out
    corrected = _rate_on(out, 'k12 zero-point corrected')
    assert corrected == pytest.approx(k12 / GAP_FACTOR, rel=1e-3)
    assert _rate_on(out, 'k12 bias-reweighted') == pytest.approx(k12, rel=1e-3)


def test_reweighted_rates_print_no_correction_for_a_mixture(capsys):
    """The line the log used to carry for a mixture must be gone.

        k12 zero-point corrected (offset +0.000 kT): <the uncorrected value>

    With `factor` nan and the gate on `median_ok`, no corrected line is printed.
    """
    k12, k21, out = _rates_on(_mixed_ensemble(n_stale=2, n_fresh=5), capsys)

    assert 'k12 bias-reweighted' in out, 'the rate itself is still printed'
    assert 'zero-point corrected (offset +0.000' not in out
    # the phrase also occurs inside the warning prose ("no zero-point corrected
    # rate is printed below"), so assert on the printed *line*, which is what
    # the log reader sees as a number
    assert 'k12 zero-point corrected' not in out
    assert 'k21 zero-point corrected' not in out
    assert np.isfinite(k12) and np.isfinite(k21)


def test_reweighted_rates_print_no_correction_for_an_unreadable_census(capsys):
    """The mixture below the voting bar must not reach the log as a rate.

    With `factor` derived from `len(levels) <= 1` this printed
    `k12 zero-point corrected (offset -3.007 kT)` for an ensemble holding two
    zero points — scaling the three already-correct caches by 20.223 as well.
    """
    pe = Ensemble(
        [_cached(f'clean{i}.xtc', 0.0, n_reactive=2) for i in range(3)]
        + [_cached(f'stale{i}.xtc', -GAP, n_reactive=2) for i in range(8)])
    k12, k21, out = _rates_on(pe, capsys)

    assert 'k12 bias-reweighted' in out
    # the phrase also occurs in the warning prose, so assert on the printed
    # line, which is what a log reader sees as a number
    assert 'k12 zero-point corrected' not in out
    assert 'k21 zero-point corrected' not in out
    assert np.isfinite(k12) and np.isfinite(k21)


def test_reweighted_rates_print_no_correction_for_negative_bias_alone(capsys):
    """The `median_ok` gate, not `isfinite(factor)`, is what saves this case."""
    pe = _ensemble_with_a_negative_frame()
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        assert np.isfinite(check_bias_zero_point(pe, 'ARB')['factor']), \
            'the premise of this test'
    capsys.readouterr()

    _, _, out = _rates_on(pe, capsys)
    assert 'k12 bias-reweighted' in out
    assert 'k12 zero-point corrected' not in out
    assert 'k21 zero-point corrected' not in out


def test_reweighted_rates_survive_an_ensemble_with_no_bias(capsys):
    """No bias cache anywhere -> the empty early return -> no KeyError."""
    pe = Ensemble([Block('run1/chainR0/path000001.xtc', 'ARBR',
                         states='ARRB', bias=None)])
    k12, k21, out = _rates_on(pe, capsys)
    assert 'no reactive frames' in out
    assert 'zero-point corrected' not in out
    assert np.isfinite(k12) and np.isfinite(k21)


# ════════════════════════════════════════════════════════════════════════════
# Non-equilibrium free-basin seeds
# ════════════════════════════════════════════════════════════════════════════

def _free_trajectory(index, n_in_state, boost, state='A', done=True):
    """One free trajectory as blocks plus their (length, gamma) rows.

    An in-state dwell of `n_in_state` frames whose realised boost is `boost`,
    plus (when the passage completed) the escaping block that reaches the other
    end state and contributes no in-state frames.
    """
    stem = f'run1/free{state}/traj{index:06d}'
    blocks = [Block(f'{stem}.part0001.xtc', f'{state}{state}{state}{state}')]
    rows = [(float(n_in_state), float(boost))]
    if done:
        other = 'B' if state == 'A' else 'A'
        blocks.append(Block(f'{stem}.part0002.xtc',
                            f'{state}R{other}{state}'))
        rows.append((1.0, 1.0))
    return blocks, rows


def _assemble(trajectories):
    pe, lengths, gammas = Ensemble(), [], []
    for blocks, rows in trajectories:
        pe.extend(blocks)
        for length, gamma in rows:
            lengths.append(length)
            gammas.append(gamma)
    return pe, np.array(lengths, float), np.array(gammas, float)


def _exponential_quantiles(n, mean=1.0):
    """Deterministic sample whose empirical CDF is exponential by construction."""
    i = np.arange(1, n + 1)
    return -mean * np.log(1.0 - (i - 0.5) / n)


def test_seed_report_is_quiet_for_an_equilibrated_basin(capsys):
    """The G4 signature: every passage gets the equilibrium boost, times memoryless.

    Realised boost ratios measured on the three G4 replicates span 0.69 - 1.14
    over 68 passages; the durations are set by the dwell lengths at fixed boost.
    """
    boost = 70.0
    lengths = _exponential_quantiles(40, mean=1000.0)
    pe, L, g = _assemble(
        [_free_trajectory(i, n, boost) for i, n in enumerate(lengths, start=1)])
    with no_warnings():
        result = report_nonequilibrium_seeds(pe, L, g)
    out = capsys.readouterr().out
    assert 'WARNING' not in out
    assert result['A']['n_low_boost'] == 0
    assert result['A']['boost_equilibrium'] == pytest.approx(boost)
    assert result['A']['median_over_mean'] == pytest.approx(0.693, abs=0.05)
    assert result['A']['ks_distance'] < result['A']['ks_critical']
    assert result['A']['ok']


def test_seed_report_flags_low_boost_passages(capsys):
    """The G2-v2 signature: 4 of 5 passages realised ~0.18 of the equilibrium boost.

    Real values: realised boost 53.0 / 0.0 / 313.2 / 53.7 / 54.9 against a pooled
    equilibrium boost of 299.5, i.e. ratios 0.18 / 0.00 / 1.05 / 0.18 / 0.18, and
    physical clocks 1.14 / 0.00 / 225.9 / 0.08 / 1.07 us.
    """
    trajectories = [
        _free_trajectory(1, 2036, 53.03),
        _free_trajectory(2, 1, 1.0),
        _free_trajectory(3, 72001, 313.17),
        _free_trajectory(4, 138, 53.73),
        _free_trajectory(5, 1837, 54.94),
    ]
    pe, L, g = _assemble(trajectories)
    with pytest.warns(UserWarning, match='Non-equilibrium free-basin seeds'):
        result = report_nonequilibrium_seeds(pe, L, g)
    r = result['A']
    assert r['n_passages'] == 5
    assert r['boost_equilibrium'] == pytest.approx(299.5, rel=0.02)
    assert r['n_low_boost'] == 4
    assert r['frac_low_boost'] == pytest.approx(0.8)
    assert np.all(np.sort(r['boost_ratios'])[:4] < SEED_BOOST_FRACTION)
    assert not r['ok']
    out = capsys.readouterr().out
    assert 'free_restart_source' in out
    assert 'upper bound' in out


def test_seed_report_flags_skewed_first_passages():
    """Even at full boost, a median/mean far below ln2 is not a rate."""
    trajectories = [_free_trajectory(i, 1, 70.0) for i in range(1, 20)]
    trajectories.append(_free_trajectory(20, 1_000_000, 70.0))
    pe, L, g = _assemble(trajectories)
    with pytest.warns(UserWarning):
        result = report_nonequilibrium_seeds(pe, L, g)
    assert result['A']['n_low_boost'] == 0
    assert result['A']['median_over_mean'] < 0.05
    assert not result['A']['ok']


def test_seed_report_excludes_open_trajectories():
    """A still-running trajectory has not completed a passage."""
    trajectories = [_free_trajectory(i, n, 70.0) for i, n
                    in enumerate(_exponential_quantiles(10, 1000.0), start=1)]
    trajectories.append(_free_trajectory(99, 10_000_000, 70.0, done=False))
    pe, L, g = _assemble(trajectories)
    result = report_nonequilibrium_seeds(pe, L, g)
    assert result['A']['n_passages'] == 10
    assert result['A']['n_censored'] == 1
    assert result['A']['times'].max() < 1e8


def test_seed_report_ignores_shooting_paths(capsys):
    """Shooting-chain paths are not first passages and must not be grouped."""
    pe = Ensemble([Block('run1/chainR0/path000001.xtc', 'ARBR')])
    result = report_nonequilibrium_seeds(pe, np.array([10.0]), np.array([2.0]))
    assert result == {}
    assert 'no free first passages' in capsys.readouterr().out


def test_seed_report_is_silent_without_a_fill():
    """An unbiased run has gamma = 1 everywhere; the boost ratio says nothing."""
    trajectories = [_free_trajectory(i, n, 1.0) for i, n
                    in enumerate(_exponential_quantiles(20, 500.0), start=1)]
    pe, L, g = _assemble(trajectories)
    with no_warnings():
        result = report_nonequilibrium_seeds(pe, L, g)
    assert np.isnan(result['A']['frac_low_boost'])
    assert result['A']['ok']


def test_seed_report_without_free_trajectories_is_a_single_line(capsys):
    result = report_nonequilibrium_seeds(Ensemble(), np.array([]),
                                        np.array([]))
    assert result == {}
    assert capsys.readouterr().out.strip().count('\n') == 0


# ════════════════════════════════════════════════════════════════════════════
# The memorylessness statistic itself
# ════════════════════════════════════════════════════════════════════════════

def test_ks_exponential_is_small_for_an_exponential_sample():
    d, crit = _ks_exponential(_exponential_quantiles(50, mean=3.0))
    assert d < crit
    assert crit == pytest.approx(1.094 / np.sqrt(50))


def test_ks_exponential_is_large_for_a_spike_plus_tail():
    times = [1e-4] * 19 + [1000.0]
    d, crit = _ks_exponential(times)
    assert d > crit
