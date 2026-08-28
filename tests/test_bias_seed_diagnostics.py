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
test_zero_point_survives_an_empty_ensemble

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
                                           check_bias_zero_point,
                                           report_nonequilibrium_seeds)

KT = 8.314462618e-3 * 300.0          # kJ/mol at 300 K, as in the run's params


@contextlib.contextmanager
def no_warnings():
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        yield


class Block:
    """One `Path.split` block: its file, its state triplet, its arrays."""

    def __init__(self, fname, type_, states=None, bias=None):
        self._fnames = [fname]
        self.type = type_
        self._states = None if states is None else np.asarray(list(states),
                                                             dtype='<U1')
        self._bias = None if bias is None else np.asarray(bias, dtype=float)

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
    plateau = 15.0 / KT                      # BARRIER = 15 kJ/mol
    return Ensemble([
        Block('run1/freeA/traj000001.part0001.xtc', 'AAAA',
              states='A' * 6, bias=[plateau + offset] * 6),
        Block('run1/freeA/traj000001.part0002.xtc', 'ARBR',
              states='ARRRB', bias=[plateau + offset] + [offset] * 4),
    ])


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
    pe = Ensemble([
        Block('run1/freeA/traj000001.part0001.xtc', 'AAAA',
              states='AAAA', bias=[0.0, 0.0, -2.0, 0.0]),
        Block('run1/freeA/traj000001.part0002.xtc', 'ARBR',
              states='ARRB', bias=[0.0, 0.0, 0.0, 0.0]),
    ])
    with pytest.warns(UserWarning):
        result = check_bias_zero_point(pe, 'ARB')
    assert not result['ok']
    assert result['minimum'] == pytest.approx(-2.0)


def test_zero_point_survives_an_empty_ensemble(capsys):
    result = check_bias_zero_point(Ensemble(), 'ARB')
    assert result['ok'] and result['n_frames'] == 0
    assert 'no reactive frames' in capsys.readouterr().out


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
    assert 'restart_free_simulations_from' in out
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
