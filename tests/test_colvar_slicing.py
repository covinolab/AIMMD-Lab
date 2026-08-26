"""
Unit tests for cumulative-COLVAR slicing in aimmd.params._methods.

PLUMED writes one accumulating ``COLVAR`` per worker directory even when GROMACS
uses ``-noappend``; AIMMD slices it into one ``{deffnm}.partNNNN_COLVAR`` per
trajectory part so that ``params.bias_function`` can find its rows. These tests
characterise that slicing, then pin the row-range calculation that both the
slicer and the training worker's out-of-cache fallback share.

Tests
-----
test_slices_each_part_in_order
test_honours_an_existing_slice
test_skips_the_part0000_seed
test_stops_when_the_colvar_is_short
test_no_colvar_is_a_noop
test_ranges_flag_already_sliced_parts
test_partial_covers_most_frames_of_a_lagging_trailing_part
test_partial_still_omits_a_part_with_no_rows_at_all
test_partial_leaves_a_middle_shortfall_alone
test_shortfall_is_reported_even_behind_an_unopenable_part
"""

import os
import types

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _write_colvar(path, n_rows):
    """Cumulative COLVAR whose column 2 encodes the absolute row index."""
    lines = ['#! FIELDS time d opes.bias']
    for i in range(n_rows):
        lines.append(f'{i * 10.0:.6f} 0.500000 {float(i):.6f}')
    path.write_text('\n'.join(lines) + '\n')


def _make_dir(tmp_path, n_rows, part_frames, base='traj000001', ext='.xtc'):
    """Worker directory with a cumulative COLVAR and placeholder part files."""
    d = tmp_path / 'freeA'
    d.mkdir(exist_ok=True)
    for part in part_frames:
        (d / f'{base}.part{part:04d}{ext}').write_text('')
    if n_rows is not None:
        _write_colvar(d / 'COLVAR', n_rows)
    return str(d), base


def _patch_frame_counts(monkeypatch, counts, ext='.xtc'):
    """Point _methods.MDA_CACHE at a stub reporting *counts* frames per part."""
    import re
    from aimmd.params import _methods

    def get(fname):
        m = re.search(r'\.part(\d{4})', fname)
        if not m:
            return None
        n = counts.get(int(m.group(1)))
        return None if n is None else [None] * n

    monkeypatch.setattr(_methods, 'MDA_CACHE', types.SimpleNamespace(get=get))


def _rows_in(d):
    """Row count of the cumulative COLVAR in directory *d* (0 if absent)."""
    f = os.path.join(d, 'COLVAR')
    if not os.path.exists(f):
        return 0
    with open(f) as fh:
        return sum(1 for line in fh if not line.startswith('#'))


def _rows_of(path):
    """Column-2 values of a written slice — i.e. which cumulative rows it took."""
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 2].astype(int).tolist()


# ════════════════════════════════════════════════════════════════════════════
# Characterisation — _split_cumulative_colvar
# ════════════════════════════════════════════════════════════════════════════

def test_slices_each_part_in_order(tmp_path, monkeypatch):
    """Parts take consecutive, non-overlapping blocks of the cumulative rows."""
    from aimmd.params._methods import _split_cumulative_colvar

    d, base = _make_dir(tmp_path, 30, {1: 10, 2: 10, 3: 10})
    _patch_frame_counts(monkeypatch, {1: 10, 2: 10, 3: 10})
    _split_cumulative_colvar(d, base, '.xtc')

    assert _rows_of(os.path.join(d, f'{base}.part0001_COLVAR')) == list(range(0, 10))
    assert _rows_of(os.path.join(d, f'{base}.part0002_COLVAR')) == list(range(10, 20))
    assert _rows_of(os.path.join(d, f'{base}.part0003_COLVAR')) == list(range(20, 30))


def test_honours_an_existing_slice(tmp_path, monkeypatch):
    """An existing slice is left alone and still advances the row cursor.

    This is why publishing a slice for a part that is still growing is unsafe:
    its row count is taken as final for every part after it.
    """
    from aimmd.params._methods import _split_cumulative_colvar

    d, base = _make_dir(tmp_path, 30, {1: 10, 2: 10})
    existing = os.path.join(d, f'{base}.part0001_COLVAR')
    _write_colvar(tmp_path / 'tmp', 10)
    os.replace(str(tmp_path / 'tmp'), existing)
    before = open(existing).read()

    _patch_frame_counts(monkeypatch, {1: 10, 2: 10})
    _split_cumulative_colvar(d, base, '.xtc')

    assert open(existing).read() == before, 'existing slice must not be rewritten'
    assert _rows_of(os.path.join(d, f'{base}.part0002_COLVAR')) == list(range(10, 20))


def test_skips_the_part0000_seed(tmp_path, monkeypatch):
    """part0000 is written by python, so PLUMED produced no rows for it."""
    from aimmd.params._methods import _split_cumulative_colvar

    d, base = _make_dir(tmp_path, 20, {0: 1, 1: 10, 2: 10})
    _patch_frame_counts(monkeypatch, {0: 1, 1: 10, 2: 10})
    _split_cumulative_colvar(d, base, '.xtc')

    assert not os.path.exists(os.path.join(d, f'{base}.part0000_COLVAR'))
    assert _rows_of(os.path.join(d, f'{base}.part0001_COLVAR')) == list(range(0, 10))
    assert _rows_of(os.path.join(d, f'{base}.part0002_COLVAR')) == list(range(10, 20))


def test_stops_when_the_colvar_is_short(tmp_path, monkeypatch):
    """Too few rows for a part: that part and every later one are left unwritten."""
    from aimmd.params._methods import _split_cumulative_colvar

    d, base = _make_dir(tmp_path, 15, {1: 10, 2: 10, 3: 10})
    _patch_frame_counts(monkeypatch, {1: 10, 2: 10, 3: 10})
    _split_cumulative_colvar(d, base, '.xtc')

    assert os.path.exists(os.path.join(d, f'{base}.part0001_COLVAR'))
    assert not os.path.exists(os.path.join(d, f'{base}.part0002_COLVAR')), \
        'a partial slice must never be written'
    assert not os.path.exists(os.path.join(d, f'{base}.part0003_COLVAR'))


def test_no_colvar_is_a_noop(tmp_path, monkeypatch):
    """A directory with no cumulative COLVAR must not raise."""
    from aimmd.params._methods import _split_cumulative_colvar

    d, base = _make_dir(tmp_path, None, {1: 10})
    _patch_frame_counts(monkeypatch, {1: 10})
    _split_cumulative_colvar(d, base, '.xtc')  # must not raise
    assert not os.path.exists(os.path.join(d, f'{base}.part0001_COLVAR'))


# ════════════════════════════════════════════════════════════════════════════
# The shared row-range calculation
# ════════════════════════════════════════════════════════════════════════════

def test_ranges_flag_already_sliced_parts(tmp_path, monkeypatch):
    """Callers must be able to tell a final part from a live one."""
    from aimmd.params._methods import _part_row_ranges

    d, base = _make_dir(tmp_path, 20, {1: 10, 2: 10})
    _write_colvar(tmp_path / 'tmp', 10)
    os.replace(str(tmp_path / 'tmp'),
               os.path.join(d, f'{base}.part0001_COLVAR'))
    _patch_frame_counts(monkeypatch, {1: 10, 2: 10})

    ranges = _part_row_ranges(d, base, '.xtc', _rows_in(d))
    assert ranges[1] == (0, 10, True)
    assert ranges[2] == (10, 20, False)


# ════════════════════════════════════════════════════════════════════════════
# Partial ranges — the reader takes what is there, the writer does not
# ════════════════════════════════════════════════════════════════════════════

def test_partial_covers_most_frames_of_a_lagging_trailing_part(tmp_path, monkeypatch):
    """A flush lag should cost a few frames, not the whole segment.

    The writer must stay all-or-nothing (a published slice is taken as final),
    but a read-only caller publishes nothing, so the rows that are on disk are
    usable and correctly aligned from the part's first frame.
    """
    from aimmd.params._methods import _part_row_ranges

    # part0002 has 20 frames but only 17 of its rows have been flushed
    d, base = _make_dir(tmp_path, 27, {1: 10, 2: 20})
    _write_colvar(tmp_path / 'tmp', 10)
    os.replace(str(tmp_path / 'tmp'),
               os.path.join(d, f'{base}.part0001_COLVAR'))
    _patch_frame_counts(monkeypatch, {1: 10, 2: 20})

    strict = _part_row_ranges(d, base, '.xtc', _rows_in(d))
    partial = _part_row_ranges(d, base, '.xtc', _rows_in(d), allow_partial=True)

    assert 2 not in strict, 'the writer must not take a partial part'
    assert partial[2] == (10, 27, False), 'the reader should take the 17 available rows'


def test_partial_still_omits_a_part_with_no_rows_at_all(tmp_path, monkeypatch):
    """A part that starts beyond the last row has nothing to offer."""
    from aimmd.params._methods import _part_row_ranges

    d, base = _make_dir(tmp_path, 10, {1: 10, 2: 10})
    _write_colvar(tmp_path / 'tmp', 10)
    os.replace(str(tmp_path / 'tmp'),
               os.path.join(d, f'{base}.part0001_COLVAR'))
    _patch_frame_counts(monkeypatch, {1: 10, 2: 10})

    partial = _part_row_ranges(d, base, '.xtc', _rows_in(d), allow_partial=True)
    assert 1 in partial
    assert 2 not in partial


def test_partial_leaves_a_middle_shortfall_alone(tmp_path, monkeypatch):
    """A shortfall that lands on a middle part stops the walk, as for the writer."""
    from aimmd.params._methods import _part_row_ranges

    # only 15 rows: part0002 (10 frames) is short AND is not the last part
    d, base = _make_dir(tmp_path, 15, {1: 10, 2: 10, 3: 10})
    _patch_frame_counts(monkeypatch, {1: 10, 2: 10, 3: 10})

    partial = _part_row_ranges(d, base, '.xtc', _rows_in(d), allow_partial=True)
    assert 1 in partial
    assert 2 not in partial, 'a short middle part would shift every later offset'
    assert 3 not in partial


def test_shortfall_is_reported_even_behind_an_unopenable_part(tmp_path, monkeypatch,
                                                             capsys):
    """The shortfall diagnostic is the only detector of an under-writing stride.

    _part_row_ranges omits unopenable/zero-frame parts as well as the part the
    row budget cannot reach, so a report loop that stops at the first omission
    is silenced for the rest of that trajectory's life by one unreadable part.
    """
    from aimmd.params._methods import _split_cumulative_colvar

    # part0001 unopenable, part0002 sliceable, part0003 short of rows
    d, base = _make_dir(tmp_path, 15, {1: 10, 2: 10, 3: 10})
    _patch_frame_counts(monkeypatch, {1: 0, 2: 10, 3: 10})
    _split_cumulative_colvar(d, base, '.xtc')

    out = capsys.readouterr().out
    assert 'cumulative COLVAR' in out, \
        f'the shortfall must still be reported behind an unopenable part:\n{out}'
    assert 'part0003' in out, out
