"""
aimmd.cgaimmd.restart
=====================

Continuing a run folder as ``chain_type='cgaimmd'``, and post-training a
C-GAIMMD network on an existing run.

A trainer alone already works on any run folder
(``aimmd.Worker(params, run_dir).train(nrounds=K)`` or a trainer-only
``aimmd.Launcher(params, run_dir).run(0, 0, 0, nrounds=K)``). Two things stand
in the way of doing that with a C-GAIMMD network on a finished RFPS or TPS
run:

- training in place overwrites the source run's ``network<s>.h5``, bins and
  densities: :func:`derive_run` mirrors the run into a new folder instead,
  linking the trajectories and copying only the caches that stay valid;
- the source's descriptor caches hold the source network's descriptors
  (distances, heavy-atom graphs), not the flat coordinates of the generator
  atoms.

There are two ways around the second, both checked by
:func:`check_cgaimmd_restart`:

- **a descriptor series of its own** (recommended): the C-GAIMMD params set
  ``descriptors_series = DESCRIPTORS.series``, the fingerprinted series of
  their :class:`~aimmd.cgaimmd.preparation.descriptors.GeneratorDescriptors`
  (``'descriptors-cg'`` and 10 hex characters). The source's
  ``*.descriptors.npy`` caches stay untouched and readable; the new series is
  refilled by featurizing, once, before any worker starts
  (``GeneratorDescriptors(..., refill=True)``, :mod:`aimmd.core.series`), or
  the job stops and says how to fill it. Its name changes with every setting
  that changes the rows, the water placement of a cluster included, so caches
  of other settings are never mixed in;
- **the default series** ``'descriptors'`` (what the archived runs use, so
  that they resume without a refill): :func:`check_descriptor_caches` detects
  caches of other settings on a random sample, :func:`migrate_descriptors`
  sets them aside so that the trainer recomputes them, and the run folder
  records the settings its caches were written with
  (:data:`DESCRIPTOR_RECORD`, :func:`check_descriptor_record`): a later start
  with other settings is refused, and so is a cluster run whose caches
  predate the record (they may hold the old water placement).

Post-training recipe (the source run keeps its ``chain_type``, so that
``shot_paths`` keeps its path weights; the protocol values function is the
default of a C-GAIMMD network under every chain type):

>>> import aimmd
>>> from aimmd.cgaimmd.restart import derive_run
>>> params = aimmd.Params.load('params_cgaimmd.py')                # doctest: +SKIP
>>> report = derive_run('rfps_run', 'cgaimmd_post', params=params)  # doctest: +SKIP
>>> aimmd.Worker(params, 'cgaimmd_post').train(nrounds=10)         # doctest: +SKIP

Restart recipe (RFPS -> C-GAIMMD on the same folder, relaunched with the
Launcher, which checks the folder before any worker starts), with the
default series:

>>> import aimmd
>>> from aimmd.cgaimmd.restart import migrate_descriptors
>>> params = aimmd.Params.load('params_cgaimmd.py')                # doctest: +SKIP
>>> report = migrate_descriptors(params, 'rfps_run')               # doctest: +SKIP
>>> aimmd.Launcher(params, 'rfps_run').run(4, 1, 1)                # doctest: +SKIP

With a series of its own, no migration is needed: the Launcher refills the
series (``refill=True``) or names the command that fills it.

Standard library and numpy at import.
"""

import glob
import json
import os
import shutil

import numpy as np

from .._config import print
from .values import MIGRATE_HINT


TPS_RESTART_AS_CGAIMMD = False
"""Whether a run with TPS history may continue as ``chain_type='cgaimmd'``.

False, so :func:`check_cgaimmd_restart` raises ``NotImplementedError`` for a
chain folder holding ``tps_weights.npy``: such a chain would mix TPS
multiplicities on its old paths with RFPS-style weights on the new generated
shots, and whether AIMMD's reweighting of that mixture is right is not
established. Post-training on a TPS folder (:func:`derive_run`, which keeps
the source's ``chain_type``) is unaffected. Setting it to True only removes
the refusal; it does not answer the question.
"""

DEFAULT_SERIES = 'descriptors'
"""The descriptor series of params that do not name one (``params.descriptors_series``)."""

TRAJECTORY_EXTENSIONS = ('.xtc', '.trr')
"""The trajectory formats looked for; a ``params.trajectory_extension`` of another kind is added when params are given."""

PLUMED_SUFFIX = '_COLVAR'
"""Suffix of the PLUMED output of a trajectory, ``<stem>_COLVAR`` next to ``<stem><ext>`` (``bias_source='file'``)."""

CUMULATIVE_COLVAR = 'COLVAR'
"""The cumulative PLUMED output of a free folder, which the trainer reads for a part without its own slice."""

MIRRORED_FOLDERS = ('chain', 'free', 'initial')
"""Prefixes of the run subfolders :func:`derive_run` mirrors."""

LINK_MODES = ('symlink', 'hardlink', 'copy')
"""How :func:`derive_run` puts the trajectory files into the new run folder."""

DESCRIPTOR_RECORD = 'cgaimmd_descriptors.json'
"""The file in a run folder that records the settings of its ``'descriptors'`` caches.

Written by :func:`check_descriptor_record` (the trainer and the Launcher,
through :func:`check_cgaimmd_restart`) for a run whose descriptors function
is a :class:`~aimmd.cgaimmd.preparation.descriptors.GeneratorDescriptors`
and whose params use the default series. A fingerprinted series carries its
settings in its name and needs no record.
"""

DESCRIPTOR_RECORD_SCHEMA = 'aimmd.cgaimmd.descriptors/1'
"""Schema tag of :data:`DESCRIPTOR_RECORD`."""


def _series(params):
    """``params.descriptors_series``, or the default series for params without one."""
    return getattr(params, 'descriptors_series', None) or DEFAULT_SERIES


def _extensions(params=None):
    """:data:`TRAJECTORY_EXTENSIONS`, plus ``params.trajectory_extension`` when it is another one."""
    extension = getattr(params, 'trajectory_extension', None)
    if extension and extension not in TRAJECTORY_EXTENSIONS:
        return TRAJECTORY_EXTENSIONS + (extension,)
    return TRAJECTORY_EXTENSIONS


def _split_cache(name, extensions=TRAJECTORY_EXTENSIONS):
    """``('path000001.xtc', 'states')`` for ``'path000001.xtc.states.npy'``, else None."""
    if not name.endswith('.npy'):
        return None
    stem = name[:-len('.npy')]
    for ext in extensions:
        index = stem.rfind(ext + '.')
        if index > 0:
            return stem[:index + len(ext)], stem[index + len(ext) + 1:]
    return None


def _is_trajectory(name, extensions=TRAJECTORY_EXTENSIONS):
    """Whether `name` is a trajectory file (it ends with one of `extensions`)."""
    return name.endswith(tuple(extensions))


def _is_segment_leftover(name):
    """``*back*``/``*forw*`` in a chain folder: AIMMD's clean-up glob of the shot in flight.

    ``Worker._shoot`` removes ``{folder}/*back*`` and ``{folder}/*forw*``
    before every new shot, so in a chain folder these names are the segments,
    caches, GROMACS files and PLUMED outputs of a shot that never got
    registered. Other folders never hold segments, so their files keep such
    names (an initial path may be called ``seeds_backbone.xtc``).
    """
    return 'back' in name or 'forw' in name


def _is_plumed_output(name, folder, trajectories, extensions=TRAJECTORY_EXTENSIONS):
    """Whether `name` is the PLUMED output of a mirrored trajectory (or a free folder's cumulative one)."""
    if name == CUMULATIVE_COLVAR:
        return folder.startswith('free')
    if not name.endswith(PLUMED_SUFFIX):
        return False
    stem = name[:-len(PLUMED_SUFFIX)]
    return any(stem + ext in trajectories for ext in extensions)


def _mirrored_folders(run_dir, include_initial=True):
    """The ``chain*/``, ``free*/`` (and ``initial*/``) folders of `run_dir`, sorted."""
    prefixes = MIRRORED_FOLDERS if include_initial else MIRRORED_FOLDERS[:2]
    return [entry for entry in sorted(os.listdir(run_dir))
            if entry.startswith(prefixes) and os.path.isdir(os.path.join(run_dir, entry))]


def _put(source, target, link):
    """Put `source` at `target` as an absolute symlink, a hard link or a copy (`link`)."""
    if link == 'symlink':
        os.symlink(os.path.abspath(source), target)
    elif link == 'hardlink':
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def _inside(path, folder):
    """Whether `path` is `folder` or lies inside it, symlinks resolved."""
    path = os.path.realpath(path)
    folder = os.path.realpath(folder)
    return path == folder or path.startswith(folder + os.sep)


def _mirror_action(name, folder, trajectories, *, extensions, copy_caches, descriptors,
                   series):
    """What :func:`derive_run` does with file `name` of `folder`: ``'link'``, ``'copy'`` or None (skip)."""
    if name.startswith(('.', 'temp.')) or (folder.startswith('chain')
                                           and _is_segment_leftover(name)):
        return None
    cache = _split_cache(name, extensions)
    if cache is not None:
        attribute = cache[1]
        if attribute in copy_caches:
            return 'copy'
        if attribute == series and descriptors == 'link':
            return 'link'
        return None
    if name in trajectories or name.endswith('.edr'):
        return 'link'
    if _is_plumed_output(name, folder, trajectories, extensions):
        return 'link'
    if name == 'tps_weights.npy':
        return 'copy'
    return None


def derive_run(source, target, *, link='symlink', copy_caches=('states', 'bias'),
               descriptors='recompute', include_initial=True, params=None):
    """Mirror a run folder into a new one for post-training.

    Parameters
    ----------
    source : str
        The finished run folder (RFPS or TPS); nothing is ever written into it.
    target : str
        The new run folder; it must not exist or be empty, and must not lie inside `source`.
    link : {'symlink', 'hardlink', 'copy'}, optional
        How trajectory files, their ``.edr`` and PLUMED outputs are put into `target`; symlinks are absolute.
    copy_caches : tuple of str, optional
        The per-trajectory caches (``<traj>.<name>.npy``) copied as files; the default copies states and bias.
    descriptors : {'recompute', 'link'}, optional
        ``'recompute'``: no descriptor cache (the trainer computes them);
        ``'link'``: the caches of the params' descriptor series are linked and checked.
    include_initial : bool, optional
        Whether ``initial*/`` is mirrored as well.
    params : aimmd.Params, optional
        The C-GAIMMD params; required for ``descriptors='link'`` (their
        ``descriptors_series`` and ``descriptors_function``); their
        ``trajectory_extension`` is looked for too.

    Returns
    -------
    dict
        The report: ``source``, ``target``, ``link``, ``descriptors``, ``series``,
        ``folders``, ``linked``, ``copied``, ``linked_bytes``, ``copied_bytes``,
        ``skipped``, ``bias_caches_left_out``, ``descriptor_check``.

    Raises
    ------
    ValueError
        For a missing `source`, a source without trajectories, a non-empty
        `target`, a `target` inside `source`, a bad `link` or `descriptors`,
        or linked descriptor caches that fail the check.
    TypeError
        For ``descriptors='link'`` without `params`.

    Notes
    -----
    The ``chain*/``, ``free*/`` and (with `include_initial`) ``initial*/``
    folders of `source` are recreated in `target`. In them, trajectory files
    (``.xtc``, ``.trr`` and ``params.trajectory_extension``), ``.edr`` files
    and the PLUMED outputs that a ``bias_source='file'`` run rebuilds its bias
    caches from (``<stem>_COLVAR`` next to ``<stem><ext>``, and the
    cumulative ``COLVAR`` of a free folder) are linked with `link`; the
    caches named in `copy_caches` and ``tps_weights.npy`` (the TPS path
    weights, which ``params.shot_paths`` reads under ``chain_type='tps'``)
    are copied. Everything else is skipped and listed in the report's
    ``skipped``: in chain folders every ``*back*``/``*forw*`` file (the
    segments of a shot in flight, which AIMMD itself removes before the
    next shot; other folders keep such names), ``pool.log``, logs,
    checkpoints and run inputs (``.cpt``, ``.tpr``), C-GAIMMD sidecars and
    in-flight records, hidden and temporary files (locks, offsets) and every
    other cache. The top level of `source` (networks and their metadata,
    bins, densities, logs, fit states, :data:`DESCRIPTOR_RECORD`) is not
    mirrored at all. A source without any trajectory is refused before
    anything is written.

    ``copy_caches=('states', 'bias')``, the default, copies the two caches
    that do not depend on the network (a missing one is simply not there)
    and leaves the values to the C-GAIMMD network's first value pass. Bias
    caches that `copy_caches` leaves out are counted in
    ``bias_caches_left_out`` and warned about: the post-training run would
    reweight without the recorded bias wherever the PLUMED outputs cannot
    rebuild it. ``('states', 'bias', 'values')`` saves the value pass, but
    the first round then reweights with the *source* network's committor.
    The copied caches assume the source run's definitions: the states the
    post-training params' ``states`` and ``states_function`` give, and the
    bias of their ``record_bias``, ``bias_function`` and ``bias_source``.
    The trainer computes only *missing* rows, so copied states of other
    basin definitions would be reweighted silently: pass
    ``copy_caches=('bias',)`` when the state definitions change, and
    ``copy_caches=()`` when the bias changes too.
    With ``descriptors='link'`` the caches of the params' descriptor series
    are linked and :func:`check_descriptor_caches` must pass on `target`;
    on a failure the linked descriptor caches are removed again and the
    ``ValueError`` propagates (the rest of the mirror stays, and the trainer
    recomputes the missing descriptors).

    With ``link='hardlink'`` the linked files are shared with `source`, and
    with ``'symlink'`` they point into it: a write to them in `target`
    changes `source` too. A post-training run only reads them (the trainer
    writes caches next to the links).
    """
    if link not in LINK_MODES:
        raise ValueError(f'link must be one of {LINK_MODES}, got {link!r}')
    if descriptors not in ('recompute', 'link'):
        raise ValueError(f"descriptors must be 'recompute' or 'link', got {descriptors!r}")
    if descriptors == 'link' and params is None:
        raise TypeError("descriptors='link' needs the C-GAIMMD params, to check the linked "
                        "descriptor caches (params=...)")
    if not os.path.isdir(source):
        raise ValueError(f'source run folder {source!r} does not exist')
    if _inside(target, source):
        raise ValueError(f'target {target!r} lies inside the source run folder {source!r}; '
                         f'derive_run never writes into the source')
    if os.path.exists(target) and (not os.path.isdir(target) or os.listdir(target)):
        raise ValueError(f'target {target!r} exists and is not empty')
    copy_caches = tuple(copy_caches)
    extensions = _extensions(params)
    series = _series(params)

    # plan first, so that a source without trajectories is refused before
    # anything is written
    folders = _mirrored_folders(source, include_initial)
    plan = []                                   # (folder, name, action)
    skipped = []
    n_trajectories = 0
    bias_left_out = 0
    for folder in folders:
        source_folder = os.path.join(source, folder)
        names = sorted(os.listdir(source_folder))
        files = [name for name in names if os.path.isfile(os.path.join(source_folder, name))]
        # the trajectories this folder mirrors (a chain's segments are not)
        trajectories = {name for name in files
                        if _is_trajectory(name, extensions)
                        and not name.startswith(('.', 'temp.'))
                        and not (folder.startswith('chain') and _is_segment_leftover(name))}
        n_trajectories += len(trajectories)
        for name in names:
            if name not in files:
                skipped.append(f'{folder}/{name}')
                continue
            action = _mirror_action(name, folder, trajectories, extensions=extensions,
                                    copy_caches=copy_caches, descriptors=descriptors,
                                    series=series)
            if action is not None:
                plan.append((folder, name, action))
                continue
            skipped.append(f'{folder}/{name}')
            cache = _split_cache(name, extensions)
            if cache is not None and cache[1] == 'bias' and cache[0] in trajectories:
                bias_left_out += 1
    if not n_trajectories:
        hint = ('' if params is not None else
                '; pass params=... when the run writes another trajectory_extension')
        raise ValueError(f'no trajectory ({", ".join(extensions)}) in the chain*/, free*/ '
                         f'or initial*/ folders of {source!r}: nothing to post-train '
                         f'on{hint}')

    os.makedirs(target, exist_ok=True)
    report = {'source': os.path.abspath(source), 'target': os.path.abspath(target),
              'link': link, 'descriptors': descriptors, 'series': series, 'folders': folders,
              'linked': 0, 'copied': 0, 'linked_bytes': 0, 'copied_bytes': 0,
              'skipped': skipped, 'bias_caches_left_out': bias_left_out,
              'descriptor_check': None}
    for folder in folders:
        os.makedirs(os.path.join(target, folder), exist_ok=True)
    linked_descriptors = []
    for folder, name, action in plan:
        source_file = os.path.join(source, folder, name)
        target_file = os.path.join(target, folder, name)
        size = os.path.getsize(source_file)
        if action == 'copy':
            shutil.copy2(source_file, target_file)
            report['copied'] += 1
            report['copied_bytes'] += size
        else:
            _put(source_file, target_file, link)
            report['linked'] += 1
            report['linked_bytes'] += size
            if _split_cache(name, extensions) is not None:
                linked_descriptors.append(target_file)
    if bias_left_out:
        print(f'WARNING: derive_run left out {bias_left_out} bias cache(s) '
              f'(<traj>.bias.npy) of {source!r} (copy_caches={copy_caches!r}); the '
              f'post-training run reweights without the recorded bias wherever the '
              f'trainer cannot rebuild it from the linked PLUMED outputs')
    if descriptors == 'link':
        try:
            report['descriptor_check'] = check_descriptor_caches(params, target)
        except ValueError:
            for fname in linked_descriptors:
                os.remove(fname)
            raise
    return report


def _descriptor_caches(run_dir, extensions=TRAJECTORY_EXTENSIONS, series=DEFAULT_SERIES,
                       include_initial=True):
    """Every ``<traj>.<series>.npy`` (of an existing trajectory) under the mirrored folders of `run_dir`."""
    caches = []
    if not os.path.isdir(run_dir):
        return caches
    for folder in _mirrored_folders(run_dir, include_initial):
        pattern = os.path.join(glob.escape(os.path.join(run_dir, folder)), f'*.{series}.npy')
        for fname in sorted(glob.glob(pattern)):
            cache = _split_cache(os.path.basename(fname), extensions)
            if cache is None or cache[1] != series:
                continue
            if os.path.isfile(os.path.join(run_dir, folder, cache[0])):
                caches.append(fname)
    return caches


CACHE_CHECK_ATOL = 0.01
"""Default tolerance (Å) of :func:`check_descriptor_caches`: one quantum of AIMMD's XTC files.

A registered shooting path is written by AIMMD at the XTC writer's default
precision (0.001 nm = 0.01 Å), while its cached rows were computed from the
segments ``back``/``forw``, which mdrun may write more precisely
(``compressed-x-precision``; 10000 in the NANMA test system). A cached row
can then differ from the one recomputed from the path file by half a quantum,
0.005 Å. Descriptors of another network or another settings differ by much
more.
"""


def check_descriptor_caches(params, run_dir, *, n_files=8, n_frames=2, atol=CACHE_CHECK_ATOL,
                            seed=0):
    """Check the descriptor caches of a run against ``params.descriptors_function``.

    Parameters
    ----------
    params : aimmd.Params
        Whose ``descriptors_function`` the caches of ``params.descriptors_series`` must match.
    run_dir : str
        The run folder (its ``chain*/``, ``free*/`` and ``initial*/`` caches are sampled).
    n_files : int, optional
        Cache files sampled (all of them if there are fewer).
    n_frames : int, optional
        Frames sampled per file, among the rows the cache holds.
    atol : float, optional
        Largest absolute difference accepted between cached and recomputed descriptors (Å);
        default :data:`CACHE_CHECK_ATOL`, one XTC quantum (the path files AIMMD writes are
        rounded to it, the segments their rows came from may not be).
    seed : int, optional
        Seed of the sample.

    Returns
    -------
    dict
        ``series``, ``n_files``, ``n_frames`` (checked), ``width``, ``max_abs_diff``
        and ``files`` (the sampled caches).

    Raises
    ------
    ValueError
        With the ``migrate_descriptors`` hint, when a cache has another width or other values.

    Notes
    -----
    Only the caches of the params' series are read (``<traj>.<series>.npy``),
    so the caches an RFPS history left under another series never fail the
    check. The sampled frames are recomputed through ``aimmd.Path(trajectory)
    [i:i + 1].compute(params.descriptors_function)``, the reader type the
    trainer's descriptor pass hands to the function. Rows that are not
    finite or all zero (not computed yet: the compute ledger's convention),
    frames past the end of a cache (a file being written) and rows past the
    end of the trajectory (a rewritten file) are not compared; a cache
    without any computed row is checked for its width only (on frame 0). A
    run without caches passes (``n_files = 0``).
    """
    from ..path import Path
    series = _series(params)
    caches = _descriptor_caches(run_dir, _extensions(params), series)
    rng = np.random.default_rng(seed)
    if len(caches) > n_files:
        caches = sorted(rng.choice(caches, size=n_files, replace=False).tolist())
    report = {'series': series, 'n_files': len(caches), 'n_frames': 0, 'width': None,
              'max_abs_diff': 0.0, 'files': caches}
    for cache in caches:
        rows = np.load(cache, mmap_mode='r')
        trajectory = cache[:-len(f'.{series}.npy')]
        path = Path(trajectory)
        n_available = min(len(rows), len(path))
        if not n_available:
            continue
        rows_2d = np.asarray(rows[:n_available]).reshape(n_available, -1)
        computed = np.isfinite(rows_2d).all(axis=1) & np.any(rows_2d != 0, axis=1)
        valid = np.flatnonzero(computed)
        if len(valid):
            chosen = np.sort(rng.choice(valid, size=min(n_frames, len(valid)), replace=False))
        else:
            chosen = np.array([0])                  # no computed row: the width only
        for i in chosen:
            recomputed = np.asarray(path[int(i):int(i) + 1].compute(params.descriptors_function))
            recomputed = recomputed.reshape(1, -1)
            cached = rows_2d[int(i):int(i) + 1]
            report['width'] = recomputed.shape[1]
            if cached.shape[1] != recomputed.shape[1]:
                raise ValueError(
                    f'the descriptor cache {cache!r} has width {cached.shape[1]}, while '
                    f'params.descriptors_function returns width {recomputed.shape[1]}; '
                    f'{MIGRATE_HINT}')
            if not len(valid):
                continue
            diff = float(np.max(np.abs(cached.astype(np.float64)
                                       - recomputed.astype(np.float64))))
            if not diff <= atol:
                raise ValueError(
                    f'the descriptor cache {cache!r} differs from params.descriptors_function '
                    f'at frame {int(i)} (largest difference {diff:.3g} > {atol:g}); '
                    f'{MIGRATE_HINT}')
            report['max_abs_diff'] = max(report['max_abs_diff'], diff)
            report['n_frames'] += 1
    return report


def migrate_descriptors(params, run_dir, *, backup_suffix='pre_cgaimmd', dry_run=False):
    """Set the descriptor caches of a run aside, so that the trainer recomputes them.

    Parameters
    ----------
    params : aimmd.Params
        The new (C-GAIMMD) params; the trainer recomputes the caches of their
        ``descriptors_series`` with their ``descriptors_function``.
    run_dir : str
        The run folder (``chain*/``, ``free*/`` and ``initial*/``).
    backup_suffix : str, optional
        ``<traj>.<series>.npy`` becomes ``<traj>.<series>.<backup_suffix>.npy``.
    dry_run : bool, optional
        Only report what would be renamed.

    Returns
    -------
    dict
        ``renamed`` (count), ``files`` (the caches), ``series``, ``backup_suffix``,
        ``dry_run`` and ``record`` (the :data:`DESCRIPTOR_RECORD` set aside, or None).

    Raises
    ------
    ValueError
        If a backup name exists already (nothing is renamed then).

    Notes
    -----
    With a descriptor series of its own (``descriptors_series =
    DESCRIPTORS.series``) no migration is needed: the old caches are of
    another series and stay as they are. Renaming a linked cache (a run made
    by :func:`derive_run`) renames the link, never the file it points to. Run
    it before the relaunch, while no worker writes into the run folder. Of
    `params` only ``descriptors_series`` and ``trajectory_extension`` are
    read; they name the params the caches will be recomputed with. The run's
    :data:`DESCRIPTOR_RECORD` (if any) is set aside with the same suffix,
    because it describes the caches that were set aside.
    """
    if not backup_suffix or os.sep in backup_suffix:
        raise ValueError(f'backup_suffix must be a plain name, got {backup_suffix!r}')
    series = _series(params)
    caches = _descriptor_caches(run_dir, _extensions(params), series)
    backups = [f'{cache[:-len(".npy")]}.{backup_suffix}.npy' for cache in caches]
    record = os.path.join(run_dir, DESCRIPTOR_RECORD)
    record_backup = f'{record[:-len(".json")]}.{backup_suffix}.json'
    has_record = series == DEFAULT_SERIES and os.path.isfile(record)
    existing = [backup for backup in backups if os.path.lexists(backup)]
    if has_record and os.path.lexists(record_backup):
        existing.append(record_backup)
    if existing:
        raise ValueError(f'{len(existing)} backup file(s) exist already, e.g. {existing[0]!r}; '
                         f'pass another backup_suffix')
    if not dry_run:
        for cache, backup in zip(caches, backups):
            os.replace(cache, backup)
        if has_record:
            os.replace(record, record_backup)
    return {'renamed': len(caches), 'files': caches, 'series': series,
            'backup_suffix': backup_suffix, 'dry_run': bool(dry_run),
            'record': record if has_record else None}


def _descriptor_record_text(spec):
    """One line naming what a descriptor spec describes, for the messages."""
    atoms = spec.get('generator_atoms') or {}
    cluster = atoms.get('cluster')
    text = f"{atoms.get('n_solute')} solute atoms"
    if cluster:
        text += (f", a cluster of {cluster.get('n_waters')} waters (water placement "
                 f"{cluster.get('water_placement')})")
    return text


def _warn_unrecorded_descriptors(function, run_dir):
    """WARNING (once per run folder and process) when a GeneratorDescriptors may be used unrecorded.

    :func:`check_descriptor_record` can keep a record only of the one
    :class:`~aimmd.cgaimmd.preparation.descriptors.GeneratorDescriptors` it
    finds behind the descriptors function. When it finds none, or two, but
    the descriptors function's module builds one (or the function calls
    several), the run may well write its caches with such an object; the
    record that would catch caches of other settings is then not kept.
    """
    from ._once import once
    from .preparation.descriptors import generator_descriptors_near
    near = generator_descriptors_near(function)
    if not near or not once(('unrecorded descriptor settings', os.path.abspath(str(run_dir)))):
        return
    print(f"WARNING: the settings of the descriptor caches of {str(run_dir)!r} are not recorded "
          f"({DESCRIPTOR_RECORD}): params.descriptors_function is not, and does not call, "
          f"exactly one GeneratorDescriptors that the record can identify ({len(near)} in "
          f"reach or in its module). A resumed run is then not checked against caches written "
          f"with other settings (e.g. an older water placement). Set descriptors_function to "
          f"the GeneratorDescriptors object, or to a wrapper that calls it by its global name, "
          f"or give the params a descriptor series of their own "
          f"(descriptors_series = DESCRIPTORS.series).")


def check_descriptor_record(params, run_dir, *, write=True):
    """Check (and record) the settings a run's ``'descriptors'`` caches were written with.

    Parameters
    ----------
    params : aimmd.Params
        The params of the start; nothing is done unless their series is the
        default ``'descriptors'`` and their descriptors function is (or wraps)
        a :class:`~aimmd.cgaimmd.preparation.descriptors.GeneratorDescriptors`
        (:func:`~aimmd.cgaimmd.preparation.descriptors.find_generator_descriptors`).
        When none is found but one is in reach (two called, or one built in
        the function's module), a WARNING says that the record is not kept.
    run_dir : str
        The run folder.
    write : bool, optional
        Write :data:`DESCRIPTOR_RECORD` when the check passes and there is none.

    Returns
    -------
    dict or None
        The record (``schema``, ``series``, ``fingerprint``, ``spec``), None
        when nothing was checked.

    Raises
    ------
    ValueError
        If the record names other settings than the descriptors compute now,
        or if a run whose descriptors have a cluster holds ``'descriptors'``
        caches in its ``chain*/`` or ``free*/`` folders but no record: such
        caches may come from a version that placed the cluster waters next to
        the solute's centre (water placement 1), which a resumed run would
        silently mix with the current placement. The message names the ways
        out: a series of its own, or :func:`migrate_descriptors`.

    Notes
    -----
    A fingerprinted series carries its settings in its name (a change starts
    a new series, which :mod:`aimmd.core.series` refills or refuses), so it
    needs no record. For the default series the record does the same job:
    it holds :meth:`GeneratorDescriptors.spec` and its fingerprint. A run
    whose caches predate the record continues without one only when its
    descriptors have no cluster (the solute's coordinates did not change
    between versions); the record is then written.

    Every C-GAIMMD worker task calls it before it starts (``Worker.run``),
    and so do the Launcher and the trainer (through
    :func:`check_cgaimmd_restart`), so the first process of a new run writes
    the record before any cache exists; the run folder is created if needed.
    """
    from .preparation.descriptors import find_generator_descriptors
    if _series(params) != DEFAULT_SERIES:
        return None
    function = getattr(params, 'descriptors_function', None)
    descriptors = find_generator_descriptors(function)
    if descriptors is None:
        _warn_unrecorded_descriptors(function, run_dir)
        return None
    fname = os.path.join(run_dir, DESCRIPTOR_RECORD)
    record = {'schema': DESCRIPTOR_RECORD_SCHEMA, 'series': DEFAULT_SERIES,
              'fingerprint': descriptors.fingerprint, 'spec': descriptors.spec()}
    hint = (f"give the C-GAIMMD params a descriptor series of their own "
            f"(descriptors_series = DESCRIPTORS.series, with GeneratorDescriptors(..., "
            f"refill=True) to refill it), or set the old caches aside with "
            f"aimmd.cgaimmd.restart.migrate_descriptors(params, {str(run_dir)!r})")
    if os.path.isfile(fname):
        with open(fname) as file:
            old = json.load(file)
        if old.get('fingerprint') != record['fingerprint']:
            raise ValueError(
                f"the descriptor caches of {run_dir!r} ('*.descriptors.npy') were written with "
                f"other descriptor settings ({_descriptor_record_text(old.get('spec') or {})}; "
                f"{fname}) than params.descriptors_function computes now "
                f"({_descriptor_record_text(record['spec'])}); AIMMD fills only missing rows, "
                f"so a resumed run would mix both: {hint}")
        return old
    if descriptors.generator_atoms.cluster is not None and _descriptor_caches(
            run_dir, _extensions(params), DEFAULT_SERIES, include_initial=False):
        raise ValueError(
            f"{run_dir!r} holds descriptor caches ('*.descriptors.npy') but no record of the "
            f"settings they were written with ({DESCRIPTOR_RECORD}); with a cluster of waters "
            f"they may come from a version that placed the waters next to the solute's centre "
            f"(water placement 1; now {record['spec']['generator_atoms']['cluster']['water_placement']}: "
            f"next to the reference of their ranking rule), and a resumed run would mix both: "
            f"{hint}")
    if write:
        from .io import atomic_write_json
        os.makedirs(str(run_dir), exist_ok=True)
        atomic_write_json(fname, record)
    return record


def check_cgaimmd_restart(params, run_dir, *, descriptors=False):
    """Refuse a run folder that cannot continue as ``chain_type='cgaimmd'``.

    Parameters
    ----------
    params : aimmd.Params
        The params of the relaunch; nothing is checked unless ``chain_type == 'cgaimmd'``.
    run_dir : str
        The run folder (``worker._directory`` inside a worker).
    descriptors : bool, optional
        Also check the descriptor caches of the params' series
        (:func:`check_descriptor_caches`) and, for the default series, the
        settings they were recorded with (:func:`check_descriptor_record`,
        which writes the record when there is none). False (the default) is
        the shooting worker's call: its source checks the descriptor width
        itself; the trainer and the Launcher pass True.

    Raises
    ------
    NotImplementedError
        If a ``chain*/`` folder holds ``tps_weights.npy`` (TPS history) and
        :data:`TPS_RESTART_AS_CGAIMMD` is False.
    ValueError
        From :func:`check_descriptor_caches` (stale caches) or
        :func:`check_descriptor_record` (caches of other settings), with the
        ways out.
    """
    if getattr(params, 'chain_type', None) != 'cgaimmd':
        return
    if (not TPS_RESTART_AS_CGAIMMD
            and glob.glob(os.path.join(glob.escape(str(run_dir)), 'chain*', 'tps_weights.npy'))):
        raise NotImplementedError(
            "continuing a TPS run as chain_type='cgaimmd' is not supported: the chains would "
            "mix TPS multiplicities with RFPS-style weights, and whether AIMMD's reweighting "
            "of that mixture is right is not established (aimmd.cgaimmd.restart."
            "TPS_RESTART_AS_CGAIMMD)")
    if descriptors:
        check_descriptor_caches(params, str(run_dir))
        check_descriptor_record(params, str(run_dir))
