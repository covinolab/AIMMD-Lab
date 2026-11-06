"""Shooting validation of generated configurations (``aimmd.cgaimmd.analysis.shooting_validation``).

Ported from the shooting analysis of the chignolin generator-adherence campaign
(its ``test_analyse_shooting_chignolin.py``), on the package's API and with
generic state letters:

1. the estimator: on 20,000 synthetic frames per request whose two halves are
   independent Bernoulli(p) draws (the assumption of the variance estimate),
   the committor mean and ``(2 p (1 - p) - T) / 2`` recover E[p] and Var(p);
2. the logistic dial fit recovers a known line; the comparison of two
   generators resolves a known difference of slopes;
3. a fake sweep folder written as AIMMD writes registered two-way shots
   (backward half reversed, times running down to 0 at the shooting frame, then
   the forward half), with states caches, sweep-frame tags and ``worker.log``
   launch records, and a fake manifest of :func:`write_validation_frames`. Edge
   cases: a frame shot twice (the second shot in a lower-numbered folder, so
   only the log-based rule keeps the original), a backward half that used the
   whole maximum length (forward half not simulated), a forward half
   uncommitted at the maximum length, a frame in the reactive region without a
   shot, an untagged path (AIMMD's old rule ``i mod N``), failed preparations
   and frames prepared in an end state. The per-request numbers of the first
   launched shot per frame, and of all shots, must equal the ones computed here
   from the truth, and nothing is written into the sweep folder;
4. attribution by coordinates, the brute-force reference of a sweep against
   AIMMD's own ``PathEnsemble.shooting_results``;
5. the run's own shots (:func:`from_shot_table`) and the frames writer on the
   toy network.
"""

import json
import os
import shutil

import numpy as np
import pytest

from aimmd.cgaimmd.analysis import shooting_validation as sv
from aimmd.cgaimmd.analysis._stats import sigmoid

LEVELS = (-3.0, -1.5, 0.0, 1.5, 3.0)
N = 16


def _listing(root):
    out = {}
    for directory, _, files in os.walk(root):
        for name in files:
            fname = os.path.join(directory, name)
            stat = os.stat(fname)
            out[os.path.relpath(fname, root)] = (stat.st_size, stat.st_mtime_ns)
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_estimator_recovers_committor_mean_and_variance():
    rng = np.random.default_rng(20261030)
    for a, b in ((0.5, 0.5), (2.0, 5.0), (8.0, 2.0)):
        p = rng.beta(a, b, 20000)
        rows = [sv.frame_outcomes([tuple('B' if u < q else 'A' for u in rng.random(2))])
                for q in p]
        s = sv.level_statistics(rows, n_boot=0)
        var_true = a * b / ((a + b) ** 2 * (a + b + 1))
        assert abs(s['committor_mean'] - p.mean()) < 0.01
        assert abs(s['committor_variance'] - var_true) < 0.01
        assert s['n_frames_with_outcome'] == 20000
        assert s['committor_sd_if_positive'] == pytest.approx(np.sqrt(s['committor_variance']))
        assert s['expected_transition_share'] == pytest.approx(
            2 * s['committor_mean_fully_committed'] * (1 - s['committor_mean_fully_committed']))


def test_frame_outcomes_with_other_state_letters_and_open_halves():
    out = sv.frame_outcomes([('U', 'F'), ('X', '-'), ('U', 'U')], states='UXF')
    assert out['n_first_halves'] == 3 and out['n_last_halves'] == 1
    assert out['n_uncommitted_halves'] == 1 and out['n_not_simulated_halves'] == 1
    assert out['n_fully_committed_shots'] == 2 and out['n_transition_shots'] == 1
    assert out['fraction_last'] == pytest.approx(1 / 4)
    assert out['transition_share'] == pytest.approx(1 / 2)
    assert out['fraction_last_lower'] == pytest.approx(1 / 6)
    assert out['fraction_last_upper'] == pytest.approx(3 / 6)
    assert out['transition_share_lower'] == pytest.approx(1 / 3)
    assert out['transition_share_upper'] == pytest.approx(2 / 3)
    empty = sv.frame_outcomes([])
    assert empty['n_shots'] == 0 and np.isnan(empty['fraction_last'])


def test_logistic_dial_recovers_known_line():
    rng = np.random.default_rng(1)
    x = np.repeat(np.array(LEVELS), 2000)
    for a_true, b_true in ((0.3, 1.0), (-0.5, 0.4), (0.0, 0.0)):
        y = (rng.random(len(x)) < sigmoid(a_true + b_true * x)).astype(float)
        a_hat, b_hat = sv.logistic_dial(x, y)
        assert abs(b_hat - b_true) < 0.05 and abs(a_hat - a_true) < 0.1
    # fractional outcomes count as fractional halves; one request: no slope
    a_hat, b_hat = sv.logistic_dial([-1, -1, 1, 1], [0.0, 0.5, 0.5, 1.0])
    assert a_hat == pytest.approx(0.0, abs=1e-6) and b_hat > 0
    assert np.isnan(sv.logistic_dial([1.0, 1.0], [0.0, 1.0])[1])
    # perfectly separated halves stay finite thanks to the ridge
    _, b_sep = sv.logistic_dial(x, (x > 0).astype(float))
    assert np.isfinite(b_sep) and b_sep > 5


def _fake_rows(rng, b_true, n=24, open_first=True):
    """Samples with one two-way shot each, halves Bernoulli(sigmoid(b lambda*)), one open half per request."""
    rows = []
    for level in LEVELS:
        for rank in range(n):
            ends = ['B' if u < sigmoid(b_true * level) else 'A' for u in rng.random(2)]
            if open_first and rank == 0:
                ends[1] = 'R'
            row = {'level': level, 'request': level, 'rank': rank, 'outcome_class': 'shoot'}
            row.update(sv.frame_outcomes([tuple(ends)]))
            rows.append(row)
    return rows


def test_dial_slope_and_comparison():
    rng = np.random.default_rng(2)
    base, cand = _fake_rows(rng, 0.3), _fake_rows(rng, 1.2)
    dial = sv.dial_slope(cand, n_boot=300, seed=1)
    x, y = sv.half_outcomes(cand)
    assert dial['slope'] == pytest.approx(sv.logistic_dial(x, y)[1])
    assert dial['n_committed_halves'] == len(x) == 2 * len(cand) - len(LEVELS)
    assert dial['slope_interval'][0] < dial['slope'] < dial['slope_interval'][1]
    assert dial['slope_bounds'][0] <= dial['slope'] <= dial['slope_bounds'][1]
    assert set(dial['slope_with_open_halves']) == set(sv.UNCOMMITTED_RULES[1:])
    assert not dial['large_slope']
    out = sv.compare_dials(base, cand, n_boot=500, seed=3)
    assert out['delta_slope'] == pytest.approx(dial['slope'] - sv.dial_slope(base, n_boot=0)['slope'])
    assert out['delta_slope_interval'][0] > 0
    lo, hi = out['delta_slope_bounds_open_halves']
    assert lo <= out['delta_slope'] <= hi
    assert out['fraction_of_resamples_above_zero'] > 0.95
    paired = sv.compare_dials(base, cand, n_boot=200, paired=True, seed=3)
    assert paired['bootstrap'] == 'paired ranks' and paired['delta_slope_interval'][0] > 0
    # matched ranks: drop one sample of the candidate
    fewer = [r for r in cand if not (r['level'] == 0.0 and r['rank'] == 5)]
    b2, c2, n_matched = sv.matched_ranks(base, fewer)
    assert n_matched == len(base) - 1 and len(b2) == len(c2) == n_matched
    with pytest.raises(ValueError, match='uncommitted'):
        sv.half_outcomes(cand, 'somewhere')


# ---------------------------------------------------------------------------
# a fake sweep folder
# ---------------------------------------------------------------------------

def write_shot(fname, back_end, forw_end, n_back, n_forw, frame, *, dt=5.0, tag=True,
               x_shooting=0.0, states='ARB'):
    """A registered shot as AIMMD writes it: back[n_back-1::-1] + forw[1:n_forw] (n_forw = 1: no forward half)."""
    import MDAnalysis as mda
    times = [(n_back - 1 - i) * dt for i in range(n_back)] + [j * dt for j in range(1, n_forw)]
    labels = ([back_end] + [states[1]] * (n_back - 1) + [states[1]] * max(0, n_forw - 2)
              + ([forw_end] if n_forw > 1 else []))
    assert len(labels) == len(times)
    universe = mda.Universe.empty(1, trajectory=True)
    with mda.Writer(str(fname), n_atoms=1) as writer:
        for k, t in enumerate(times):
            universe.atoms.positions = np.array([[x_shooting + (0.0 if k == n_back - 1 else 7.0),
                                                  0.0, 0.0]])
            universe.dimensions = [41.1, 41.1, 41.1, 90, 90, 90]
            universe.trajectory.ts.time = t
            writer.write(universe.atoms)
    np.save(f'{fname}.states.npy', np.array(labels, dtype='<U1'))
    if tag:
        np.save(f'{fname}.sweep_frame.npy', np.array([frame]))


def build_sweep(work, rng):
    """The fake manifest, the fake sweep folder and the truth per sample ([(back, forw), ...])."""
    samples, frame = [], 0
    special = {(-3.0, 0): 'failed_clash', (-3.0, 1): 'first_state_at_time_zero',
               (-3.0, 2): 'first_state_at_time_zero', (3.0, 0): 'last_state_at_time_zero',
               (1.5, 5): 'failed_solvation'}
    for level in LEVELS:
        for rank in range(N):
            oc = special.get((level, rank), 'shoot')
            rec = {'level': level, 'request': level, 'rank': rank, 'outcome_class': oc,
                   'reason': '' if oc == 'shoot' else f'fake {oc}', 'frame': None}
            if oc == 'shoot':
                rec['frame'] = frame
                frame += 1
            samples.append(rec)
    n_frames = frame
    run = work / 'sweep_run'
    folders = [run / f'sweepR{k}' for k in range(4)]
    for folder in folders:
        folder.mkdir(parents=True)
    counters = [0, 0, 0, 0]
    shots_of = {}
    logs = {k: [] for k in range(4)}
    clock = [0]

    def log_launch(k, number, frame, late=False):
        clock[0] += 1
        t = 3600 * 10 + clock[0] + (5000 if late else 0)       # duplicates are launched last
        stamp = f'Fri Oct 30 {t // 3600 % 24:02d}:{t // 60 % 60:02d}:{t % 60:02d} 2026'
        logs[k].append(f'Selecting shooting point for run1/sweepR{k}/path{number:06d} ({stamp})')
        logs[k].append(f'=== selecting frame run1/initialARB/shooting_frames.xtc, {frame} '
                       f'(global coverage: total {clock[0]}, min 0, max 1)')

    shoot = [r for r in samples if r['outcome_class'] == 'shoot']
    awaiting = shoot[-1]['frame']                              # the last frame never gets a shot
    for r in shoot:
        f = r['frame']
        if f == awaiting:
            continue
        p = sigmoid(r['level'])
        k = f % 4
        ends = tuple('B' if u < p else 'A' for u in rng.random(2))
        n_back, n_forw = int(rng.integers(2, 40)), int(rng.integers(2, 40))
        if f == 3:                                   # backward half reached max_length
            ends, n_back, n_forw = ('R', '-'), 40, 1
        if f == 7:                                   # forward half uncommitted at max_length
            ends = (ends[0], 'R')
        counters[k] += 1
        write_shot(folders[k] / f'path{counters[k]:06d}.xtc', ends[0], ends[1], n_back, n_forw, f)
        log_launch(k, counters[k], f)
        shots_of.setdefault(f, []).append(ends)
        if f == 10:                                  # shot again, in a LOWER-numbered folder
            k2 = (k + 3) % 4
            counters[k2] += 1
            ends2 = ('B', 'B') if ends != ('B', 'B') else ('A', 'A')
            write_shot(folders[k2] / f'path{counters[k2]:06d}.xtc', ends2[0], ends2[1], 5, 5, f)
            log_launch(k2, counters[k2], f, late=True)
            shots_of[f].append(ends2)
    k = 3                                            # one untagged shot: AIMMD's old rule
    i = counters[k]
    counters[k] += 1
    legacy_frame = i % n_frames
    write_shot(folders[k] / f'path{counters[k]:06d}.xtc', 'A', 'B', 6, 6, legacy_frame, tag=False)
    log_launch(k, counters[k], legacy_frame, late=True)
    shots_of.setdefault(legacy_frame, []).append(('A', 'B'))
    for k in range(4):
        (folders[k] / 'worker.log').write_text('\n'.join(logs[k]) + '\n')
    table = {'meta': {'levels': list(LEVELS), 'n_per_level': N, 'states': 'ARB',
                      'n_frames': n_frames}, 'samples': samples}
    truth = []
    for r in samples:
        if r['outcome_class'] == 'first_state_at_time_zero':
            truth.append([('A', 'A')])
        elif r['outcome_class'] == 'last_state_at_time_zero':
            truth.append([('B', 'B')])
        elif r['outcome_class'] == 'shoot':
            truth.append(shots_of.get(r['frame'], []))
        else:
            truth.append(None)
    return run, table, truth, n_frames, {'awaiting': awaiting, 'legacy_frame': legacy_frame}


def expected_level(samples, truth, level, all_shots):
    f, t, f_full = [], [], []
    for r, shots in zip(samples, truth):
        if r['level'] != level or not shots:
            continue
        shots = shots if all_shots else shots[:1]            # the original is the first launched
        ends = [h for s in shots for h in s]
        na, nb = ends.count('A'), ends.count('B')
        f.append(nb / (na + nb) if na + nb else np.nan)
        full = [s for s in shots if s[0] in ('A', 'B') and s[1] in ('A', 'B')]
        t.append(sum(s[0] != s[1] for s in full) / len(full) if full else np.nan)
        f_full.append(sum(h == 'B' for x in full for h in x) / (2 * len(full)) if full else np.nan)
    with np.errstate(all='ignore'):
        p, T, pf = np.nanmean(f), np.nanmean(t), np.nanmean(f_full)
    return {'committor_mean': p, 'transition_share': T, 'committor_mean_fully_committed': pf,
            'committor_variance': 0.5 * (2 * pf * (1 - pf) - T), 'n': len(f)}


def test_sweep_outcomes_and_per_request_numbers(tmp_path):
    rng = np.random.default_rng(20261030)
    run, table, truth, n_frames, info = build_sweep(tmp_path, rng)
    before = _listing(run)
    sweep = sv.sweep_outcomes(run, n_frames)
    assert _listing(run) == before                              # nothing written
    assert sweep['folders'] == ['sweepR0', 'sweepR1', 'sweepR2', 'sweepR3']
    assert sweep['in_flight'] == [] and sweep['coordinates_checked'] is False
    sources = {s['frame_source'] for s in sweep['shots']}
    assert sources == {'tag', 'legacy rule'}
    by_frame = {}
    for s in sweep['shots']:
        by_frame.setdefault(s['frame'], []).append(s)
    shot3 = by_frame[3][0]
    assert (shot3['backward_end'], shot3['forward_end']) == ('R', sv.NOT_SIMULATED)
    assert shot3['forward_frames'] == 0 and shot3['backward_frames'] == 40
    for all_shots in (False, True):
        rows = sv.validation_rows(table, sweep, first_launched=not all_shots)
        summary = sv.level_summary(rows, n_boot=200, seed=0)
        for s in summary:
            e = expected_level(table['samples'], truth, s['level'], all_shots)
            assert s['n_frames_with_outcome'] == e['n']
            for key in ('committor_mean', 'transition_share', 'committor_mean_fully_committed',
                        'committor_variance'):
                got, want = s.get(key), e[key]
                assert (not np.isfinite(want) and not np.isfinite(got)) or \
                    abs(got - want) < 1e-12, (s['level'], all_shots, key, got, want)
    rows = sv.validation_rows(table, sweep)
    row10 = next(r for r in rows if r['frame'] == 10)
    assert row10['shot_rule'] == 'first launched (worker.log)' and row10['n_registered_shots'] == 2
    # the original (sweepR2) was launched first; the duplicate sits in the lower folder sweepR1
    assert [s['folder'] for s in by_frame[10]] == ['sweepR1', 'sweepR2']
    assert row10['shots'] == [s['path'] for s in by_frame[10] if s['folder'] == 'sweepR2']
    summary = {s['level']: s for s in sv.level_summary(rows, n_boot=0)}
    assert summary[-3.0]['n_failed_preparation'] == 1 and summary[-3.0]['n_first_state_at_time_zero'] == 2
    assert summary[3.0]['n_last_state_at_time_zero'] == 1 and summary[1.5]['n_failed_preparation'] == 1
    assert summary[1.5]['failed_reasons'] == ['fake failed_solvation']
    assert sum(s['n_without_shot'] for s in summary.values()) == 1
    assert sum(s['halves']['not_simulated'] for s in summary.values()) == 1
    assert sum(s['halves']['uncommitted'] for s in summary.values()) == 2
    for s in summary.values():
        assert s['requested_committor'] == pytest.approx(sigmoid(s['level']))
    # the dial from the halves of the first launched shots equals a fit of the truth
    x, y = [], []
    for r, shots in zip(table['samples'], truth):
        if shots:
            for half in shots[0]:
                if half in ('A', 'B'):
                    x.append(r['level'])
                    y.append(float(half == 'B'))
    dial = sv.dial_slope(rows, n_boot=100)
    assert dial['slope'] == pytest.approx(sv.logistic_dial(x, y)[1], abs=1e-9)
    json.dumps(sv.level_summary(rows, n_boot=10), default=float)
    # a folder with a shot in flight is reported
    (run / 'sweepR2' / 'back.xtc').write_bytes(b'')
    assert sv.sweep_outcomes(run, n_frames)['in_flight'] == ['sweepR2']
    # a shot of a frame the table does not list
    stray = dict(table, samples=[s for s in table['samples'] if s['frame'] != 0])
    with pytest.raises(ValueError, match='does not describe this sweep'):
        sv.validation_rows(stray, sweep)


def test_sweep_attribution_by_coordinates_and_the_brute_force_reference(tmp_path):
    """Frames told apart by their coordinates; the reference's counts equal AIMMD's shooting_results."""
    import MDAnalysis as mda
    import aimmd
    from aimmd.cgaimmd.analysis import validate
    run = tmp_path / 'sweep_run'
    initial = run / 'initialARB'
    initial.mkdir(parents=True)
    xs = [0.0, 3.0, 6.0]
    universe = mda.Universe.empty(1, trajectory=True)
    with mda.Writer(str(initial / 'shooting_frames.xtc'), n_atoms=1) as writer:
        for t, x in enumerate(xs):
            universe.atoms.positions = np.array([[x, 0.0, 0.0]])
            universe.dimensions = [41.1, 41.1, 41.1, 90, 90, 90]
            universe.trajectory.ts.time = float(t)
            writer.write(universe.atoms)
    np.save(initial / 'shooting_frames.xtc.states.npy', np.array(['R'] * 3, dtype='<U1'))
    folder = run / 'sweepR0'
    folder.mkdir()
    plan = [(0, 'A', 'B', True), (1, 'B', 'B', True), (2, 'A', 'A', True), (1, 'A', 'B', False),
            (0, 'A', 'A', True)]
    for n, (frame, back, forw, tag) in enumerate(plan, start=1):
        write_shot(folder / f'path{n:06d}.xtc', back, forw, 4, 4, frame, tag=tag,
                   x_shooting=xs[frame])
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(states='ARB', trajectory_extension='.xtc', round_shooting_index=True)
    before = _listing(run)
    sweep = sv.sweep_outcomes(run, 3, params=params, frames=initial / 'shooting_frames.xtc')
    assert _listing(run) == before
    assert [s['frame'] for s in sweep['shots']] == [0, 1, 2, 1, 0]
    assert sweep['shots'][3]['frame_source'] == 'coordinates'
    assert all(s['deviation'] < 0.02 for s in sweep['shots'])
    reference = validate.sweep_reference(params, run)
    assert _listing(run) == before
    assert reference.n_first.tolist() == [3.0, 1.0, 2.0]
    assert reference.n_last.tolist() == [1.0, 3.0, 0.0]
    assert reference.positions.shape == (3, 1, 3)
    assert np.allclose(reference.positions[:, 0, 0], xs, atol=0.01)
    assert reference.dimensions.shape == (3, 6) and reference.notes['n_shots'] == 5
    first_only = validate.sweep_reference(params, run, first_launched=True)
    assert (first_only.n_first + first_only.n_last).sum() <= 10
    # AIMMD's own aggregation of a sweep (on a copy: it writes trajectory index files)
    copy = tmp_path / 'copy'
    shutil.copytree(run, copy)
    ensemble = params.shot_paths(str(copy), 'sweep', 'R', 0)
    counts = ensemble.shooting_results('ARB', sweep_size=3)
    # the untagged path is the 4th of its folder: AIMMD attributes it to frame 3 mod 3 = 0
    counts_by_rule = np.array(counts)
    assert counts_by_rule[:, 0].sum() + counts_by_rule[:, 1].sum() == 10
    tagged = reference.n_first.copy(), reference.n_last.copy()
    tagged[0][1] -= 1                                           # move the untagged shot (A, B)
    tagged[1][1] -= 1                                           # from frame 1 to frame 0
    tagged[0][0] += 1
    tagged[1][0] += 1
    assert counts_by_rule[:, 0].tolist() == tagged[0].tolist()
    assert counts_by_rule[:, 1].tolist() == tagged[1].tolist()
    # a tag that contradicts the coordinates stops the analysis
    np.save(folder / 'path000005.xtc.sweep_frame.npy', np.array([2]))
    with pytest.raises(ValueError, match='tagged with frame 2'):
        sv.sweep_outcomes(run, 3, params=params, frames=initial / 'shooting_frames.xtc')


# ---------------------------------------------------------------------------
# the run's own shots
# ---------------------------------------------------------------------------

def test_from_shot_table():
    rows = [
        {'shot_index': 0, 'chain': 'chainR0', 'path': 'path000001.xtc', 'lambda_target': -0.4,
         'first_state': 'A', 'last_state': 'B', 'acceptance_enforced': False},
        {'shot_index': 1, 'chain': 'chainR0', 'path': 'path000002.xtc', 'lambda_target': 0.3,
         'first_state': 'A', 'last_state': 'A', 'acceptance_enforced': True},
        {'shot_index': 2, 'chain': 'chainR1', 'path': 'path000001.xtc', 'lambda_target': None,
         'first_state': 'B', 'last_state': 'B', 'acceptance_enforced': None},       # RFPS shot
        {'shot_index': 3, 'chain': 'chainR1', 'path': 'path000002.xtc', 'lambda_target': 2.6,
         'first_state': 'R', 'last_state': 'B', 'acceptance_enforced': True},
    ]
    out = sv.from_shot_table(rows)
    assert [r['shot_index'] for r in out] == [0, 1, 3]
    assert [r['level'] for r in out] == [0.0, 0.0, 3.0]
    assert [r['rank'] for r in out] == [0, 1, 0]
    assert [r['request'] for r in out] == [-0.4, 0.3, 2.6]
    assert out[0]['n_transition_shots'] == 1 and out[1]['n_first_halves'] == 2
    assert out[2]['n_uncommitted_halves'] == 1 and out[2]['n_last_halves'] == 1
    edges = sv.from_shot_table(rows, level_edges=[-1.0, 1.0, 5.0])
    assert [r['level'] for r in edges] == [0.0, 0.0, 3.0]
    summary = sv.level_summary(out, n_boot=0)
    assert [s['level'] for s in summary] == [0.0, 3.0]
    assert summary[0]['committor_mean'] == pytest.approx(0.25)
    # the dial uses the exact requests, not the bins
    x, _ = sv.half_outcomes(out)
    assert sorted(set(x.tolist())) == [-0.4, 0.3, 2.6]


# ---------------------------------------------------------------------------
# writing the frames to validate (toy network, no GROMACS)
# ---------------------------------------------------------------------------

class _RejectHighY:
    """A validity function stand-in: rejects samples with y of atom 0 above 0.4 ('clash')."""

    def __call__(self, positions, context):
        from aimmd.cgaimmd.types import ValidityResult
        x = np.asarray(positions, dtype=float)
        ok = x[:, 0, 1] <= 0.4
        reason = np.array(['' if o else 'clash: too close' for o in ok], dtype=object)
        return ValidityResult(positions=x, ok=ok, reason=reason, info=None)


def test_write_validation_frames(tmp_path):
    import torch
    from aimmd.cgaimmd.analysis._trajectory import read_frames
    from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
    from tests._cgaimmd_helpers import cgaimmd_params_namespace
    torch.manual_seed(0)
    network = TinyFlowNetwork(n_atoms=2, prior_center=0.0, prior_width=0.5, lambda_clip=10.0)
    params = cgaimmd_params_namespace(network=network, cgaimmd_parameters={'placement': 'none'},
                                      cgaimmd_validity_function=_RejectHighY())
    run = tmp_path / 'run'
    run.mkdir()
    with pytest.raises(ValueError, match='inside the run folder'):
        sv.write_validation_frames(params, run / 'validation', [0.0], 4, run_dir=run)
    out = tmp_path / 'validation'
    table = sv.write_validation_frames(params, out, [-1.0, 0.0, 1.0], 12, seed=7, run_dir=run)
    # (no preparation folder: the toy params have no preparation or minimization hook)
    assert sorted(os.listdir(out)) == ['frames_table.json', 'raw_samples.npz',
                                       'shooting_frames.xtc']
    assert not os.listdir(run)                                  # nothing in the run folder
    on_disk = sv.load_validation_table(out / 'frames_table.json')
    assert on_disk == table
    meta = table['meta']
    assert meta['generator']['method'] == 'ode' and meta['committor_window_applied'] is False
    assert meta['level_seeds'] == [7 + 1000 * (round(2 * v) + 100) for v in (-1.0, 0.0, 1.0)]
    samples = table['samples']
    assert len(samples) == 36
    assert [(s['level'], s['rank']) for s in samples[:3]] == [(-1.0, 0), (-1.0, 1), (-1.0, 2)]
    classes = {s['outcome_class'] for s in samples}
    assert classes <= {'shoot', 'first_state_at_time_zero', 'last_state_at_time_zero',
                       'failed_clash'}
    assert {'shoot', 'failed_clash'} <= classes
    raw = np.load(out / 'raw_samples.npz')['positions']
    for s in samples:
        x = raw[[-1.0, 0.0, 1.0].index(s['level']), s['rank']]
        if s['outcome_class'] == 'failed_clash':
            assert x[0, 1] > 0.4 and s['reason'].startswith('clash')
        elif x[0, 1] <= 0.4:
            x0 = round(float(x[0, 0]), 2)                     # the final check rounds as xtc does
            state = 'A' if x0 <= -0.5 else 'B' if x0 >= 0.5 else 'R'
            assert s['prepared_state'] == state
            assert {'A': 'first_state_at_time_zero', 'B': 'last_state_at_time_zero',
                    'R': 'shoot'}[state] == s['outcome_class']
    shoot = [s for s in samples if s['outcome_class'] == 'shoot']
    assert [s['frame'] for s in shoot] == list(range(len(shoot))) == list(range(meta['n_frames']))
    positions, _, _ = read_frames(out / 'shooting_frames.xtc')
    assert len(positions) == len(shoot)
    for s, x in zip(shoot, positions):
        assert np.allclose(x, raw[[-1.0, 0.0, 1.0].index(s['level']), s['rank']], atol=0.006)
        assert -0.5 < x[0, 0] < 0.5                               # in the reactive region
    # the same seed draws the same samples; a request keeps its seed without the others
    again = tmp_path / 'again'
    sv.write_validation_frames(params, again, [1.0], 12, seed=7)
    assert np.array_equal(np.load(again / 'raw_samples.npz')['positions'][0], raw[2])


# ---------------------------------------------------------------------------
# end to end on the toy engine (--runslow): a run's own shots, frames to validate,
# a real AIMMD sweep of them, and its analysis
# ---------------------------------------------------------------------------

from tests.test_toy_1d_cgaimmd import (make_campaign, published_network,  # noqa: E402
                                       restore_signals, short_sys_path)    # noqa: F401


SWEEP_PARAMS = '''
"""A committor sweep of the toy validation frames: the toy dynamics and states, plain shots."""
import numpy as np

engine = 'toy'
states = 'ARB'
chain_type = 'rfps'
initial_paths = ['shooting_frames.xtc']
round_shooting_index = True


def toy_mdrun(ts):
    for _ in range(100):
        ts.positions = (ts.positions + .02 * np.random.normal()) % 10


def states_function(trajectory):
    letters = 'ARBSCTDUE'
    out = []
    for frame in trajectory:
        x = frame.positions[0, 0]
        out.append('A' if x < 1 or x > 9 else letters[min(int(x), 8)])
    return np.array(out, dtype='<U1')
'''
"""The sweep's params: the toy campaign's dynamics and states (``tests/toy_1d_cgaimmd``)."""


@pytest.mark.slow
def test_validation_frames_shot_by_an_aimmd_sweep(tmp_path, monkeypatch, restore_signals,
                                                 short_sys_path):
    """The toy C-GAIMMD campaign: its shots as a validation (shot table with a CV), fresh frames
    written for a sweep, the sweep run by AIMMD's shooting worker, and its outcomes read back."""
    import aimmd
    import torch
    from aimmd.cgaimmd import sidecar
    from aimmd.cgaimmd.analysis import validate
    folder = make_campaign(tmp_path / 'campaign')
    monkeypatch.chdir(folder)
    np.random.seed(0)
    torch.manual_seed(0)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    run = str(folder / 'run')
    published_network(run, 0.0, round=1, n_trained_shots=0)
    aimmd.Worker(params, run, nsteps=3, walltime=600).shoot(1, 0)

    # the run's own shots, with the x of the shooting point
    def x_of(reader):
        return [ts.positions[0, 0] for ts in reader]

    rows = sidecar.shot_table(run, params=params, cv={'x': x_of})
    assert len(rows) == 3 and all(1.1 <= row['x'] <= 1.9 for row in rows)   # the validity window
    own = sv.from_shot_table(rows, states=params.sorted_states)
    assert len(own) == 3 and sum(r['n_shots'] for r in own) == 3
    summary = sv.level_summary(own, n_boot=20)
    assert sum(s['n_shot'] for s in summary) == 3

    # fresh frames: the run's generator, validity and minimization hooks, the state check
    out = tmp_path / 'validation'
    table = sv.write_validation_frames(params, out, [-1.0, 1.0], 3, seed=3, run_dir=run)
    shoot = [s for s in table['samples'] if s['outcome_class'] == 'shoot']
    assert shoot and table['meta']['n_frames'] == len(shoot)
    assert {s['outcome_class'] for s in table['samples']} <= {
        'shoot', 'failed_toy_window', 'first_state_at_time_zero', 'last_state_at_time_zero'}

    # a sweep of these frames with AIMMD's own shooting worker (one shot per frame)
    sweep_folder = tmp_path / 'sweep'
    sweep_folder.mkdir()
    shutil.copy(out / 'shooting_frames.xtc', sweep_folder / 'shooting_frames.xtc')
    (sweep_folder / 'params.py').write_text(SWEEP_PARAMS)
    monkeypatch.chdir(sweep_folder)
    sweep_params = aimmd.Params.load(str(sweep_folder / 'params.py'), save=False)
    sweep_run = str(sweep_folder / 'run')
    aimmd.Worker(sweep_params, sweep_run, nsteps=10 * len(shoot), walltime=600).shoot(
        1, 0, sweep=True, sweep_target=len(shoot))
    sweep = sv.sweep_outcomes(sweep_run, len(shoot), params=sweep_params,
                              frames=out / 'shooting_frames.xtc')
    assert len(sweep['shots']) == len(shoot) and sweep['in_flight'] == []
    assert sorted(s['frame'] for s in sweep['shots']) == list(range(len(shoot)))
    assert all(s['frame_source'] == 'tag, coordinates agree' for s in sweep['shots'])
    ends = {s['backward_end'] for s in sweep['shots']} | {s['forward_end'] for s in sweep['shots']}
    assert ends <= {'A', 'B', 'R', sv.NOT_SIMULATED}
    validation = sv.validation_rows(table, sweep)
    assert sum(1 for r in validation if r.get('n_registered_shots')) == len(shoot)
    per_level = sv.level_summary(validation, n_boot=20)
    assert sum(s['n_shot'] for s in per_level) == len(shoot)
    assert sum(s['n_without_shot'] for s in per_level) == 0
    # the brute-force reference of the same sweep counts the same halves
    reference = validate.sweep_reference(sweep_params, sweep_run,
                                         frames=out / 'shooting_frames.xtc')
    halves = sum(int(r['n_first_halves']) + int(r['n_last_halves']) for r in validation
                 if r['outcome_class'] == 'shoot')
    assert int(reference.n_first.sum() + reference.n_last.sum()) == halves
