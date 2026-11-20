"""
MDAReaderCache does not reopen a trajectory file that has not changed.

While ``mdrun`` runs, ``WorkerSimulate._simulate`` polls its trajectory about
ten times a second, and every poll asks ``Path.extend`` for a reader
``trajectory_update_batch_size`` frames longer than the path, which the cached
reader almost never is. Before the fix every such request reopened the file,
and every reopen read all its frame headers again (twice while its last frame
was still being written): ~10 full rescans a second, on mdrun's cores, for a
new frame every second or so. Now ``MDAReaderCache.get`` returns the cached
reader, however short, while the file is unchanged since that reader was
opened (same device, inode, size, mtime and ctime): a reopen of an unchanged
file gives the same safe frames.

The fix must remove reopens and change nothing else. These tests check that

- an unchanged file is not reopened, however short its reader;
- appended frames (complete or partially written), truncation, replacement,
  touching and deletion give exactly what a reopen gave before;
- a reader that leaves the cache takes its signature with it, and anything
  unusual (no signature, a reader that cannot be measured, a signature that
  cannot be taken) falls back to the base class ``get``;
- no file that ``_open`` refuses by its name (the ``.npy`` caches a glob of
  a folder hands to the cache) is ever opened, and a reader from a direct
  ``load()`` (the toy engine's, read in another thread) is never handed out
  for a request it is too short for, as before;
- the worker's own polling -- ``WorkerSimulate._simulate`` with its real stop
  condition, ``Path.check_stop``, ``Path.extend`` and its pipeline -- ingests
  the same frames at every poll, takes the same stop decisions, prints the
  same lines and writes the same caches with the cache as it was before the
  fix and with this one, while a fake mdrun writes, cuts, touches and
  replaces the trajectory between polls (partial frames included).

``BeforeTheFix`` is the reader cache as it was: the base class with the same
``_open`` and ``_close``.
"""

import os
import sys
import time
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.lib.formats.libmdaxdr import XTCFile

import aimmd
import aimmd.cache.mda as mda_cache
from aimmd import NPY_CACHE
from aimmd.cache import forget
from aimmd.cache.base import AbstractCache
from aimmd.cache.mda import MDAReaderCache, file_signature
from aimmd.path import Path
from aimmd.worker._simulate import WorkerSimulate

N_ATOMS = 12        # >= 10: XTC frames are compressed, of varying size, and their offsets are scanned
DT = 10.0
BOX = np.array([60, 60, 60, 90, 90, 90], dtype=np.float32)


# ---------------------------------------------------------------------------
# the cache before the fix, and opens counted
# ---------------------------------------------------------------------------

class BeforeTheFix(AbstractCache):
    """MDAReaderCache before the fix: the base class with the same _open and _close."""
    max_size = MDAReaderCache.max_size
    _open = MDAReaderCache._open
    _close = MDAReaderCache._close


class Counting:
    """Counts the calls of `_open` (successful or not)."""

    def __init__(self):
        super().__init__()
        self.opens = 0

    def _open(self, fname, ntries=10):
        self.opens += 1
        return super()._open(fname, ntries)


class Old(Counting, BeforeTheFix):
    pass


class New(Counting, MDAReaderCache):
    pass


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def label(x):
    return 'A' if x < 20 else ('B' if x > 40 else 'R')


def write_frames(fname, xs, first=0, seed=0):
    """Frames `first`, `first + 1`, ... (t = index * DT) with x of atom 0 = xs."""
    universe = mda.Universe.empty(N_ATOMS, trajectory=True)
    rng = np.random.default_rng(seed)
    with mda.Writer(str(fname), N_ATOMS) as writer:
        for k, x in enumerate(xs):
            positions = (30.0 + rng.normal(size=(N_ATOMS, 3))).astype(np.float32)
            positions[0] = (x, 30.0, 30.0)
            universe.trajectory.ts.positions = positions
            universe.trajectory.ts.dimensions = BOX
            universe.trajectory.ts.time = (first + k) * DT
            writer.write(universe.atoms)
    return str(fname)


def frame_bytes(fname):
    """Byte offset of every frame of `fname`, then its size."""
    with XTCFile(str(fname)) as file:
        return [int(offset) for offset in file.offsets] + [os.path.getsize(fname)]


def describe(reader):
    """What a reader gives: its length, and time and coordinates of every frame.

    Frames are read by index, not by iteration: iterating ends with a rewind,
    which reopens the file by name and fails once the file is deleted. Each
    frame's values are taken at once: a reader returns the same Timestep
    object for every frame, updated in place.
    """
    if reader is None:
        return None
    times, positions = [], []
    for i in range(len(reader)):
        ts = reader[i]
        times.append(ts.time)
        positions.append(ts.positions.round(3).tolist())
    return len(reader), times, positions


def grow(fname, data, size):
    """Make `fname` the first `size` bytes of `data`: append (as mdrun does) or cut."""
    current = os.path.getsize(fname) if os.path.exists(fname) else 0
    if size >= current:
        with open(fname, 'ab') as fh:
            fh.write(data[current:size])
    else:
        os.truncate(fname, size)


def touch_later(fname, seconds=1):
    """Move the modification time on, leaving the contents alone."""
    st = os.stat(fname)
    os.utime(fname, ns=(st.st_atime_ns, st.st_mtime_ns + int(seconds * 1e9)))


def replace_by_copy(fname, data=None):
    """Replace `fname` by a new file (written beside it, then renamed)."""
    if data is None:
        with open(fname, 'rb') as fh:
            data = fh.read()
    with open(f'{fname}.tmp', 'wb') as fh:
        fh.write(data)
    os.replace(f'{fname}.tmp', fname)


@pytest.fixture
def source(tmp_path):
    """A 12-frame trajectory, its bytes and frame offsets."""
    fname = write_frames(tmp_path / 'source.xtc', 30 + 5 * np.sin(np.arange(12) / 3))
    with open(fname, 'rb') as fh:
        data = fh.read()
    return SimpleNamespace(fname=fname, data=data, offsets=frame_bytes(fname))


def copies(tmp_path, source, size):
    """The first `size` bytes of the source, once for the old and once for the new cache."""
    files = {}
    for name in ('old', 'new'):
        os.makedirs(tmp_path / name, exist_ok=True)
        files[name] = str(tmp_path / name / 'traj.xtc')
        grow(files[name], source.data, size)
    return files


# ---------------------------------------------------------------------------
# 1. an unchanged file is not reopened
# ---------------------------------------------------------------------------

def test_an_unchanged_file_is_not_reopened(tmp_path, source):
    files = copies(tmp_path, source, source.offsets[5])
    new, old = New(), Old()
    reader = new.get(files['new'])
    assert new.opens == 1 and len(reader) == 5
    for min_length in (0, 5, 6, 1005, np.inf):
        for extend in (False, True):
            assert new.get(files['new'], min_length, extend) is reader
    assert new.opens == 1
    # before the fix, every request for a longer reader reopened the file,
    # and the reopened reader held the same frames
    first = old.get(files['old'])
    again = old.get(files['old'], 1005)
    assert old.opens == 2 and again is not first
    assert describe(again) == describe(first) == describe(reader)


def test_a_long_enough_reader_is_returned_without_looking_at_the_file(
        tmp_path, source, monkeypatch):
    """The base class path, untouched: no signature is taken."""
    files = copies(tmp_path, source, source.offsets[5])
    new = New()
    reader = new.get(files['new'])
    taken = []

    def signature(fname):
        taken.append(fname)
        return file_signature(fname)

    monkeypatch.setattr(mda_cache, 'file_signature', signature)
    for min_length in (0, 1, 5):
        assert new.get(files['new'], min_length) is reader
    assert taken == [] and new.opens == 1
    assert new.get(files['new'], 6) is reader
    assert taken == [files['new']] and new.opens == 1


def test_the_signature_is_taken_before_the_file_is_opened(
        tmp_path, source, monkeypatch):
    """A file that grows while it is being opened is reopened at the next request."""
    files = copies(tmp_path, source, source.offsets[4])
    new = New()
    real_open = MDAReaderCache._open

    def open_while_mdrun_appends(self, fname, ntries=10):
        reader = real_open(self, fname, ntries)
        grow(fname, source.data, source.offsets[6])
        return reader

    monkeypatch.setattr(MDAReaderCache, '_open', open_while_mdrun_appends)
    reader = new.get(files['new'])
    monkeypatch.setattr(MDAReaderCache, '_open', real_open)
    assert len(reader) == 4
    again = new.get(files['new'], 1000)
    assert again is not reader and len(again) == 6 and new.opens == 2


# ---------------------------------------------------------------------------
# 2. appended frames, complete or partially written
# ---------------------------------------------------------------------------

def test_appended_frames_are_read_as_before(tmp_path, source):
    offsets = source.offsets
    sizes = [offsets[3], offsets[3],                    # unchanged
             offsets[5],                                # two complete frames
             offsets[5] + 6,                            # a partial header
             (offsets[5] + offsets[6]) // 2,            # a partial frame
             (offsets[5] + offsets[6]) // 2,            # unchanged
             offsets[6],                                # the frame completed
             offsets[9] + 3, offsets[9] + 3,            # three frames, and a partial one
             offsets[12]]                               # the rest
    files = copies(tmp_path, source, sizes[0])
    new, old = New(), Old()
    versions, last_size = 0, None
    for size in sizes:
        for name in ('old', 'new'):
            grow(files[name], source.data, size)
        versions += size != last_size
        last_size = size
        got_old = old.get(files['old'], 1000)
        got_new = new.get(files['new'], 1000)
        assert describe(got_new) == describe(got_old)
        # a partially written last frame is left out, until it is complete
        assert len(got_new) == max(k for k, o in enumerate(offsets) if o <= size)
        assert new.opens == versions
    assert old.opens == len(sizes)


# ---------------------------------------------------------------------------
# 3. truncated, replaced, touched, deleted
# ---------------------------------------------------------------------------

def reversed_frames(source, nframes):
    """The first `nframes` frames of the source in reverse order: other contents, same size."""
    offsets = source.offsets
    return b''.join(source.data[offsets[k]:offsets[k + 1]] for k in reversed(range(nframes)))


def rewrite_in_place(fname, data):
    """Overwrite the bytes of `fname` in place (same inode, same size).

    Only the file's times can tell this version from the last one, so the
    write is repeated until the file system clock has moved past the time of
    the last change (a few milliseconds on Linux; up to seconds elsewhere).
    """
    before = os.stat(fname)
    for _ in range(60):
        time.sleep(0.05)
        with open(fname, 'r+b') as fh:
            fh.write(data)
        after = os.stat(fname)
        if (after.st_mtime_ns, after.st_ctime_ns) != (before.st_mtime_ns, before.st_ctime_ns):
            assert after.st_size == before.st_size and after.st_ino == before.st_ino
            return
    raise RuntimeError(f'the clock of the file system of {fname} did not move in 3 s')


CHANGES = {
    # a checkpoint rollback cuts the file to the frames of the checkpoint
    'truncate': lambda f, s: os.truncate(f, s.offsets[5]),
    # a cut to the length the file already has (mdrun -cpi after a rollback)
    'truncate_to_its_size': lambda f, s: os.truncate(f, os.path.getsize(f)),
    'append': lambda f, s: grow(f, s.data, s.offsets[10]),
    'touch': lambda f, s: touch_later(f),
    # written beside it and renamed over it, as AIMMD replaces files
    'replace': lambda f, s: replace_by_copy(f),
    # other contents of the same size, with the old modification time
    'replace_same_size_and_mtime': lambda f, s: (
        replace_by_copy(f, reversed_frames(s, 8)),
        os.utime(f, ns=(os.stat(f).st_atime_ns, s.mtime_ns))),
    # other contents of the same size in the same file: only its times change
    'rewrite_in_place_same_size': lambda f, s: rewrite_in_place(f, reversed_frames(s, 8)),
    'delete': lambda f, s: os.remove(f),
    'delete_and_recreate_reversed': lambda f, s: (
        os.remove(f), open(f, 'wb').write(reversed_frames(s, 8))),
}


@pytest.mark.parametrize('change', sorted(CHANGES))
def test_a_changed_file_gives_what_a_reopen_gave(tmp_path, source, change):
    files = copies(tmp_path, source, source.offsets[8])
    results = {}
    for name, cache in (('old', Old()), ('new', New())):
        fname = files[name]
        first = cache.get(fname)
        assert len(first) == 8
        source.mtime_ns = os.stat(fname).st_mtime_ns
        CHANGES[change](fname, source)
        got = cache.get(fname, 1000)
        results[name] = (describe(got), got is first, cache.opens)
    if change == 'truncate_to_its_size':
        # the same contents: the same frames, whether or not the file system
        # moved the file's times (and the cache reopened it)
        assert results['new'][0] == results['old'][0]
        return
    assert results['new'][:2] == results['old'][:2]
    # the new cache, too, reopened the file (and failed to, when it is gone)
    assert results['new'][2] == results['old'][2] == 2
    if change == 'delete':
        # the old reader is kept, as before; it has no signature any more
        assert results['new'][1] is True
    if change.startswith(('replace', 'delete_and', 'rewrite')):
        # what is in the file now, read afresh
        assert describe(mda.coordinates.XTC.XTCReader(files['new'], refresh_offsets=True)[:]) \
            == results['new'][0]


def test_a_deleted_file_loses_its_signature_until_it_is_back(tmp_path, source):
    files = copies(tmp_path, source, source.offsets[6])
    new, old = New(), Old()
    readers = {'new': new.get(files['new']), 'old': old.get(files['old'])}
    for name in ('old', 'new'):
        os.remove(files[name])
    for _ in range(2):      # the old reader, each request trying to reopen
        assert new.get(files['new'], 1000) is readers['new']
        assert old.get(files['old'], 1000) is readers['old']
    assert files['new'] not in new._signatures
    assert new.opens == old.opens == 3
    for name in ('old', 'new'):
        grow(files[name], source.data, source.offsets[9])
    got = {'new': new.get(files['new'], 1000), 'old': old.get(files['old'], 1000)}
    assert describe(got['new']) == describe(got['old']) and len(got['new']) == 9
    assert new.opens == old.opens == 4
    assert new.get(files['new'], 1000) is got['new'] and new.opens == 4


# ---------------------------------------------------------------------------
# signatures leave with their readers; otherwise: the base class
# ---------------------------------------------------------------------------

def _evict_oldest(cache, fname, other, third):
    cache.max_size = 2 * cache._size(cache._cache[fname]) + 1
    cache.get(third)


def _failed_load(cache, fname, other, third):
    """The file changed and cannot be opened: the old reader stays, without signature."""
    def busy(*args, **kwargs):
        raise OSError('busy')

    touch_later(fname)
    cache._open = busy
    reader = cache._cache[fname]
    assert cache.get(fname, 1000) is reader
    del cache._open


LEAVES = {
    'pop': lambda c, f, o, t: c.pop(f),
    'pop_oldest': lambda c, f, o, t: c.pop(),
    'remove': lambda c, f, o, t: c.remove(f),
    'remove_oldest': lambda c, f, o, t: c.remove(),
    'clear': lambda c, f, o, t: c.clear(),
    'put': lambda c, f, o, t: c.put(f, c._cache[f]),
    'evict': _evict_oldest,
    'failed_load': _failed_load,
}


@pytest.mark.parametrize('leave', sorted(LEAVES) + ['forget'])
def test_a_reader_takes_its_signature_along(tmp_path, source, leave, monkeypatch):
    names = []
    for name in ('a', 'b', 'c'):
        names.append(str(tmp_path / f'{name}.xtc'))
        grow(names[-1], source.data, source.offsets[5])
    fname, other, third = names
    cache = New()
    cache.get(fname)
    cache.get(other)
    assert set(cache._signatures) == {fname, other}
    if leave == 'forget':
        monkeypatch.setattr(aimmd._config, 'MDA_CACHE', cache)
        forget(fname)
    else:
        LEAVES[leave](cache, fname, other, third)
    assert fname not in cache._signatures
    assert set(cache._signatures) <= set(cache._cache)
    for key, (signature, reader) in cache._signatures.items():
        assert cache._cache[key] is reader
    # without its signature a too-short reader is reopened, as before the fix;
    # the reopen stores a new one
    opens = cache.opens
    reader = cache.get(fname, 1000)
    assert cache.opens == opens + 1 and cache._signatures[fname][1] is reader
    assert cache.get(fname, 1000) is reader and cache.opens == opens + 1


def test_a_reader_from_load_is_not_handed_out_for_a_longer_request(tmp_path, source):
    """The toy engine load()s the last part and reads it in its writer thread
    while the stop condition polls in another thread: as before the fix, the
    polls get readers of their own, never that one."""
    files = copies(tmp_path, source, source.offsets[6])
    seen = {}
    for name, cache in (('old', Old()), ('new', New())):
        engine = cache.load(files[name])
        polled = [cache.get(files[name], 1000) for _ in range(3)]
        assert all(reader is not engine for reader in polled)
        assert all(describe(reader) == describe(engine) for reader in polled)
        seen[name] = (polled[2] is polled[1] is polled[0], cache.opens)
    # before the fix every poll reopened; now the polls share their own reader
    assert seen == {'old': (False, 4), 'new': (True, 2)}
    assert New().load(files['new']) is not None


def test_without_a_signature_to_compare_get_is_the_base_get(tmp_path, source, monkeypatch):
    files = copies(tmp_path, source, source.offsets[5])
    fname = files['new']
    cache = New()
    reader = cache.get(fname)

    # the signature cannot be taken: every request for a longer reader reopens
    def no_signature(fname):
        raise PermissionError(fname)

    monkeypatch.setattr(mda_cache, 'file_signature', no_signature)
    cache.get(fname, 1000)
    cache.get(fname, 1000)
    assert cache.opens == 3 and fname not in cache._signatures
    monkeypatch.undo()
    reader = cache.get(fname, 1000)
    assert cache.opens == 4 and cache.get(fname, 1000) is reader and cache.opens == 4

    # a reader whose length cannot be taken (its file handle closed) is reopened,
    # for any min_length, as the base class does
    reader.trajectory.close()
    with pytest.raises(OSError):
        len(reader)
    for min_length in (0, 1000):
        reader = cache.get(fname, min_length)
        assert len(reader) == 5
        reader.trajectory.close()
    assert cache.opens == 6

    # a reader that is not the one the signature was taken for (the cache
    # dictionary changed behind the cache's back) is treated as unsigned
    cache.get(fname, 1000)
    cache._cache[fname] = MDAReaderCache._open(cache, fname)
    opens = cache.opens
    cache.get(fname, 1000)
    assert cache.opens == opens + 1


@pytest.mark.parametrize('key', [None, 3, 'missing.xtc', 'traj.txt'])
def test_a_key_that_cannot_be_opened_gives_none_as_before(tmp_path, key):
    if isinstance(key, str):
        key = str(tmp_path / key)
        if key.endswith('.txt'):
            open(key, 'w').write('not a trajectory')
    old, new = Old(), New()
    for cache in (old, new):
        assert cache.get(key) is None and cache.get(key, 1000) is None
        assert cache.load(key) is None and len(cache) == 0
    assert new._signatures == {} and new.opens == old.opens == 3


def test_a_cache_file_handed_to_the_reader_cache_is_never_opened(tmp_path, monkeypatch):
    """A glob of a folder hands its .npy caches to the reader cache as well
    (pathensemble.utils.get_paths -> Path -> extend -> get); _open refuses
    them by their name alone, and no signature may open them either."""
    xtc = write_frames(tmp_path / 'traj.xtc', np.linspace(25, 35, 4))
    npy = f'{xtc}.descriptors.npy'
    np.save(npy, np.zeros((4, 1)))
    opened = []
    real_open = os.open

    def recording(file, *args, **kwargs):
        opened.append(os.fsdecode(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(os, 'open', recording)
    for cache in (Old(), New()):
        assert cache.get(npy) is None and cache.get(npy, 1000) is None
        assert cache.load(npy) is None
        assert len(Path(npy)) == 0
    assert opened == []


def test_the_signed_extensions_are_those_open_accepts(tmp_path, monkeypatch):
    monkeypatch.setattr(mda_cache, 'time', SimpleNamespace(sleep=lambda seconds: None))
    for ext in mda_cache.TRAJECTORY_EXTENSIONS + ('.npy', '.XTC', '.xtc.npy', '.tpr', ''):
        fname = str(tmp_path / f'traj{ext}')
        with open(fname, 'w') as fh:
            fh.write('not a trajectory')
        try:
            MDAReaderCache._open(None, fname, ntries=1)
        except TypeError:           # refused by its name
            accepted = False
        except Exception:           # tried to read it (and failed)
            accepted = True
        assert accepted == (ext in mda_cache.TRAJECTORY_EXTENSIONS), ext


# ---------------------------------------------------------------------------
# 4. the worker's polling, before and after the fix
# ---------------------------------------------------------------------------

def states_function(trajectory):
    return np.array([label(float(ts.positions[0, 0])) for ts in trajectory], dtype='<U1')


def cv_function(trajectory):
    return np.array([float(ts.positions[0, 0]) for ts in trajectory])


PIPELINE = [(cv_function, 'cv'), (states_function, 'states'), (None, 'values')]


class SimWorker(WorkerSimulate):
    def __init__(self, params):
        self.params = params
        self.must_stop = False
        self.termination_timeout = 1.0


def part_file(deffnm, part, mode):
    return f'{deffnm}.xtc' if mode == 'shoot' else f'{deffnm}.part{part + 1:04d}.xtc'


def schedule(parts, seed, truncate=True):
    """What a fake mdrun does to the parts' files, one step between two polls.

    Each part is written from its own source (bytes, frame offsets): it
    appears with 3 complete frames (a reader needs 2) and grows by complete
    and partially written frames; between polls nothing may change, the file
    may be touched, replaced by a copy, or (`truncate`) cut back by a frame or
    two, never below 3 frames.
    """
    rng = np.random.default_rng(seed)
    steps = []
    for part, (data, offsets) in enumerate(parts):
        n = len(offsets) - 1
        k = 3
        steps.append(('size', part, offsets[k]))
        while k < n:
            r = rng.random()
            if r < 0.22:
                steps.append(('poll', part))
            elif r < 0.45:
                k += 1
                steps.append(('size', part, offsets[k]))
            elif r < 0.55:
                k = min(n, k + int(rng.integers(2, 6)))
                steps.append(('size', part, offsets[k]))
            elif r < 0.80:      # frame k written in part
                steps.append(('size', part, offsets[k] + int(rng.integers(1, offsets[k + 1] - offsets[k]))))
            elif r < 0.88:
                steps.append(('touch', part))
            elif r < 0.93:
                steps.append(('replace', part))
            elif truncate and k > 4:
                k -= int(rng.integers(1, 3))
                steps.append(('size', part, offsets[k]))
            else:
                steps.append(('poll', part))
    return steps


def use_cache(monkeypatch, cache):
    """Make `cache` the MDA_CACHE of every AIMMD module that has one."""
    for name, module in list(sys.modules.items()):
        if (name == 'aimmd' or name.startswith('aimmd.')) and hasattr(module, 'MDA_CACHE'):
            monkeypatch.setattr(module, 'MDA_CACHE', cache)


def replay(folder, parts, steps, mode, t, batch_size, cache, monkeypatch, capsys,
           pipeline=PIPELINE, states='ARB'):
    """Poll a fake mdrun applying `steps` through the worker; return what was seen.

    The worker is AIMMD's own: ``WorkerSimulate._simulate`` with its stop
    condition, ``Path.check_stop`` and ``Path.extend`` with the pipeline, all
    reading through `cache`; ``_simulate`` is called again whenever it returns
    before the path is complete and steps are left, as the shooting and free
    workers do. Its clock is a counter (0.25 s per call), so that the
    stop condition's 10-second rule fires at the same polls whatever the
    speed of the cache.
    """
    os.makedirs(folder, exist_ok=True)
    deffnm = os.path.join(folder, 'back')
    open(f'{deffnm}.tpr', 'w').close()
    use_cache(monkeypatch, cache)
    NPY_CACHE.clear()
    clock = iter(np.arange(1e6) * 0.25)
    monkeypatch.setattr('aimmd.worker._simulate.time',
                        SimpleNamespace(time=lambda: float(next(clock))))
    log = []

    def plain(value):
        if isinstance(value, (tuple, list)):
            return tuple(plain(v) for v in value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    def shape(path):
        return (len(path), tuple(os.path.basename(f) for f in path._fnames),
                tuple(path._first), tuple(path._last))

    extend, check_stop = Path.extend, Path.check_stop

    def logged_extend(self, *args, **kwargs):
        result = extend(self, *args, **kwargs)
        log.append(('extend', plain(result), shape(self)))
        return result

    def logged_check_stop(self, *args, **kwargs):
        result = check_stop(self, *args, **kwargs)
        log.append(('check_stop', plain(result)))
        return result

    monkeypatch.setattr(Path, 'extend', logged_extend)
    monkeypatch.setattr(Path, 'check_stop', logged_check_stop)

    todo = list(steps)

    def caches():
        out = {}
        for fname in sorted(os.listdir(folder)):
            if fname.endswith('.npy'):
                out[fname] = np.load(os.path.join(folder, fname)).tolist()
        return out

    def mdrun(deffnm, noappend=False, stop_condition=None, termination_timeout=None):
        assert noappend == (mode == 'free')
        while todo:
            step = todo.pop(0)
            kind, part = step[:2]
            fname = part_file(deffnm, part, mode)
            data = parts[part][0]
            if kind == 'size':
                grow(fname, data, step[2])
            elif kind == 'touch':
                touch_later(fname)
            elif kind == 'replace':
                replace_by_copy(fname)
            stop = stop_condition()
            log.append(('poll', step, stop, caches()))
            if stop:
                break
        return 0

    params = SimpleNamespace(
        engine='gromacs', trajectory_extension='.xtc', max_length=np.inf,
        trajectory_update_batch_size=batch_size, pipeline=pipeline, states=states,
        run_simulation=mdrun)
    worker = SimWorker(params)
    path = Path()
    for _ in range(len(steps) + 2):
        result = worker._simulate(deffnm, path, t, mode)
        log.append(('simulate', plain(result), shape(path)))
        if result[0] is not None or not todo:
            break
    printed = [line.replace(folder, '<run>') for line in capsys.readouterr().out.splitlines()
               if 'simulating' not in line]      # '+++ starting simulating <deffnm> <now>'
    files = {f: open(os.path.join(folder, f), 'rb').read() for f in sorted(os.listdir(folder))
             if f.endswith('.npy') or f.endswith('.xtc')}
    monkeypatch.undo()
    return SimpleNamespace(log=log, printed=printed, files=files, opens=cache.opens,
                           polls=sum(entry[0] == 'poll' for entry in log), result=result)


def compare(tmp_path, parts, steps, mode, t, batch_size, monkeypatch, capsys, **kwargs):
    """Replay `steps` with the cache before the fix and with this one; they must agree."""
    old = replay(str(tmp_path / 'old'), parts, steps, mode, t, batch_size, Old(),
                 monkeypatch, capsys, **kwargs)
    new = replay(str(tmp_path / 'new'), parts, steps, mode, t, batch_size, New(),
                 monkeypatch, capsys, **kwargs)
    for k, (a, b) in enumerate(zip(old.log, new.log)):
        assert a == b, f'entry {k} differs:\n  before the fix {a}\n  after the fix  {b}'
    assert len(old.log) == len(new.log)
    assert old.printed == new.printed
    assert old.files == new.files
    assert new.opens < old.opens
    return old, new


def shooting_half(tmp_path, n=60):
    """A half that stays in R (x in 25..35), then climbs into B (x > 40) at its end."""
    i = np.arange(n)
    xs = np.where(i < n - 10, 30 + 5 * np.sin(i / 7), 30 + 2.0 * (i - (n - 11)))
    fname = write_frames(tmp_path / 'half.xtc', xs, seed=1)
    return [(open(fname, 'rb').read(), frame_bytes(fname))]


def free_run(tmp_path):
    """Three -noappend parts of a run in A with an excursion into R, ending in B;
    each part starts at the time of a frame its predecessor already holds."""
    i = np.arange(57)
    xs = np.where(i < 52, 15 + 2 * np.sin(i / 5), 45.0)
    xs[25:29] = 25.0
    parts = []
    for p, (a, b) in enumerate(((0, 20), (18, 38), (37, 57))):
        fname = write_frames(tmp_path / f'part{p}.xtc', xs[a:b], first=a, seed=p)
        parts.append((open(fname, 'rb').read(), frame_bytes(fname)))
    return parts


@pytest.mark.parametrize('batch_size', [1000, 4])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_the_shooting_worker_polls_the_same_with_and_without_reopens(
        tmp_path, monkeypatch, capsys, batch_size, seed):
    parts = shooting_half(tmp_path)
    steps = schedule(parts, seed)
    old, new = compare(tmp_path, parts, steps, 'shoot', 'R', batch_size, monkeypatch, capsys)
    assert old.result[2] == 'B' and old.result[0] is not None     # the half completed in B
    assert new.polls > 30


@pytest.mark.parametrize('batch_size', [1000, 4])
@pytest.mark.parametrize('seed', [0, 1])
def test_the_free_worker_polls_the_same_with_and_without_reopens(
        tmp_path, monkeypatch, capsys, batch_size, seed):
    parts = free_run(tmp_path)
    steps = schedule(parts, seed, truncate=False)
    old, new = compare(tmp_path, parts, steps, 'free', 'A', batch_size, monkeypatch, capsys)
    assert old.result[2] == 'B'                                     # the run reached B
    assert len({e[1][1] for e in new.log if e[0] == 'poll'}) == 3     # in its third part
