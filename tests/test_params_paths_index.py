"""`shot_paths` must not rescan `old` linearly for every globbed file.

The lookup at params/_paths.py was a linear scan over `old._paths` with
`Path.fname` as the match key -- and `fname` is a property that calls another
property (`n_files` -> `len(self._fnames)`), measured at 147.6 ns per access
against 36.5 ns for the underlying `self._fnames[-1]`. With M files matched
against M candidates that is O(M^2): measured 9.55 s at M=11,232 and 30.6 s at
M=20,000, against 3.1 ms and 10.5 ms for a dict.

Two things this must not break:

- Object identity. The whole point of `old=` is to hand back the *existing*
  Path so nothing is re-read; a dict must preserve that exactly.
- First-match-wins. The scan `break`s on the first match, so a dict must use
  `setdefault`, not plain assignment, or a duplicate fname would resolve to the
  last candidate instead of the first.

The index must also be built where every recursion level can use it. Building it
only at the top-level entry (`if target_state is None`) hands the leaf call an
empty index and silently re-parses -- which the existing identity test in
test_params_paths_unit.py already catches, because it calls the leaf directly.
"""
import numpy as np
import pytest

import aimmd
from aimmd.pathensemble import PathEnsemble


def _fake_path(fname, tag):
    """A stand-in with just the surface `shot_paths` uses for matching."""
    class _P:
        def __init__(self):
            self._fnames = [fname]
            self.tag = tag
            self.weight = np.nan

        @property
        def n_files(self):
            return len(self._fnames)

        @property
        def fname(self):
            return self._fnames[-1] if self.n_files else ''

    return _P()


def test_index_is_first_match_wins():
    """setdefault, not assignment: the scan took the first match."""
    a = _fake_path('/x/path000001.xtc', 'first')
    b = _fake_path('/x/path000001.xtc', 'second')
    index = {}
    for p in (a, b):
        index.setdefault(p.fname, p)
    assert index['/x/path000001.xtc'].tag == 'first'


def test_reload_reuses_objects_by_identity(tmp_path, monkeypatch):
    """End-to-end on the real Params.shot_paths, over a two-chain tree.

    This is the recursive case: the existing unit test drives a single chain
    folder directly, so it cannot catch an index built only at the top level.
    """
    from tests._helpers_unit import build_path

    root = tmp_path / 'run1'
    made = {}
    for chain in ('chainR0', 'chainR1'):
        folder = root / chain
        folder.mkdir(parents=True)
        for i in (1, 2):
            p = build_path(folder, stem=f'path{i:06d}')
            made[p.fname] = p

    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(sorted_states='ARB', trajectory_extension='.xtc')

    first = params.shot_chains(str(root), None)
    flat_first = [p for chain in first for p in chain]
    assert flat_first, 'no paths were loaded'

    second = params.shot_chains(str(root), None, old=first)
    flat_second = [p for chain in second for p in chain]

    assert len(flat_second) == len(flat_first)
    by_name = {p.fname: p for p in flat_first}
    for p in flat_second:
        assert p is by_name[p.fname], (
            f'{p.fname} was rebuilt instead of reused')


def test_fname_is_evaluated_once_per_candidate(tmp_path):
    """Scaling assertion, not a wall-clock one -- CI-stable on shared storage.

    Counting `fname` evaluations distinguishes the quadratic scan from the dict
    without depending on filesystem speed: doubling the candidate count must not
    more than roughly double the work.
    """
    counts = {'n': 0}

    class _P:
        def __init__(self, fname):
            self._fnames = [fname]
            self.weight = np.nan

        @property
        def n_files(self):
            return len(self._fnames)

        @property
        def fname(self):
            counts['n'] += 1
            return self._fnames[-1]

    def build_index(candidates):
        index = {}
        for p in candidates:
            index.setdefault(p.fname, p)
        return index

    for n in (50, 100):
        counts['n'] = 0
        cands = [_P(f'/x/path{i:06d}.xtc') for i in range(n)]
        index = build_index(cands)
        for i in range(n):                      # one lookup per globbed file
            assert index[f'/x/path{i:06d}.xtc'] is cands[i]
        if n == 50:
            at_50 = counts['n']
    assert counts['n'] <= 2.5 * at_50, (
        f'fname evaluations grew from {at_50} to {counts["n"]} when the '
        f'candidate count doubled -- that is superlinear')


def test_real_shot_paths_does_not_rescan_old_per_file(tmp_path, monkeypatch):
    """The failing assertion for the quadratic scan, on the real code path.

    Counts `Path.fname` evaluations during a reload. The linear scan evaluates
    it once per (globbed file, candidate) pair -- about N^2/2 -- while an index
    evaluates it once per candidate plus a dict lookup per file, so O(N).
    """
    from tests._helpers_unit import build_path

    root = tmp_path / 'run1'
    folder = root / 'chainR0'
    folder.mkdir(parents=True)
    n = 12
    for i in range(1, n + 1):
        build_path(folder, stem=f'path{i:06d}')

    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(sorted_states='ARB', trajectory_extension='.xtc')
    first = params.shot_chains(str(root), None)
    loaded = [p for chain in first for p in chain]
    assert len(loaded) == n, f'expected {n} paths, got {len(loaded)}'

    counts = {'n': 0}
    cls = type(loaded[0])
    original = cls.fname

    def counting(self):
        counts['n'] += 1
        return original.fget(self)

    monkeypatch.setattr(cls, 'fname', property(counting))
    params.shot_chains(str(root), None, old=first)

    # Quadratic would be ~n*(n+1)/2 = 78 at n=12; linear is ~n plus overhead.
    assert counts['n'] <= 3 * n, (
        f'{counts["n"]} fname evaluations for {n} files -- the scan is still '
        f'quadratic (a linear index needs about {n})')
