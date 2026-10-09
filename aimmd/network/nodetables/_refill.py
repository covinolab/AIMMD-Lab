"""Refill a node-table series that trajectories of a run lack.

Every `NodeTableFeaturizer` (and `MultiSystemNodeTableFeaturizer`) that a
params file builds registers a `NodeTableRefiller` for its series with
`aimmd.core.series.register_series`, together with its ``refill`` flag (the
copies that ``with_n_max`` makes here and in the tools do not). When
trajectories of a run have no file of the series (the node-table settings
changed since they were featurized, or the campaign comes from the
graph-cache input) and the flag is set, one process of the run calls the
refiller before any MD or training (`aimmd.core.series.ensure_series_coverage`)
while the other processes wait. It takes the cheapest correct route per
trajectory:

1. **repack**: next to the trajectory is a node-table series whose name is
   that of this featurizer with another ``n_max`` (the capacity its row width
   gives, ``(width - 2) / 4`` per system, checked with
   `NodeTableFeaturizer.with_n_max`). Its rows are rewritten for the current
   ``n_max`` (`_tool.repack`), exactly and without reading the trajectory;
   zero rows stay zero for the ledger. A trajectory that cannot be repacked
   (e.g. a row with more nodes than a smaller ``n_max``) is featurized.
2. **extract**: the run holds the coordinate series ``*.descriptors.npy`` of
   the graph-cache input (``'sqlite'``, a campaign that was not migrated) and
   that input's graph cache, `GRAPH_CACHE`, is in the job's working
   directory (where the params file is). Rows come from the cached graphs
   (`_tool.prefill` with ``db``), checked against a direct featurization on
   `_tool.DEFAULT_DB_VERIFY` frames per trajectory; a trajectory that fails
   the check is featurized.
3. **recompute**: every frame is featurized from the trajectory
   (`_tool.prefill`).

Only trajectories without a file of the series are written (files of the
series that exist are never rewritten), and the files of other series stay
where they are. With more than one process, the processes the tools spawn
rebuild the featurizer from the params file that defines it (a module-level
name of a module in the params folder) and check its series; GPUs are hidden
from them.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from collections import namedtuple

#: The graph cache of the graph-cache input, in the job's working directory.
GRAPH_CACHE = 'graphs_cache.sqlite'

#: Command that fills a node-table series offline.
PREFILL_COMMAND = 'python -m aimmd.network.nodetables prefill'

# a route for some trajectories: kind is 'repack', 'extract' or 'recompute';
# source is (series, from_n_max) for repack and the graph cache for extract
_Step = namedtuple('_Step', 'kind trajectories source text')

# per-file lines of the tools left out of the refill log (one per file that
# went well), and the end-of-command verdict
_QUIET = ('written', 'complete', 'exists', 'OK')
# the tools' advice for the command line, which does not apply here
_ADVICE = ('The params file needs',)


class NodeTableRefiller:
    """The refill callable of a node-table series (see the module notes).

    Parameters
    ----------
    featurizer : NodeTableFeaturizer or MultiSystemNodeTableFeaturizer
        The featurizer whose series it refills.
    """

    label = 'node-table'
    legacy_note = ("this campaign has not been migrated from the graph-cache "
                   "('sqlite') input")

    def __init__(self, featurizer):
        self.featurizer = featurizer

    def __repr__(self):
        return f'{type(self).__name__}({self.featurizer!r})'

    def claims(self, series):
        """Whether `series` is a node-table series name (``'descriptors-gn'``
        and 10 hex digits): one this featurizer would compute with other
        settings."""
        from ._featurizer import SERIES_PREFIX
        return _is_node_table_series(str(series), SERIES_PREFIX)

    @property
    def parts(self):
        """tuple: the series of the systems of a multi-system featurizer
        (registered by their own featurizers; not the run's series)."""
        featurizers = getattr(self.featurizer, 'featurizers', None) or {}
        return tuple(featurizer.series for featurizer in featurizers.values())

    def prefill_command(self, params_file, run):
        """The command that fills the series of `run` offline."""
        source = _source(self.featurizer, params_file)
        params = source[0] if source else (params_file or 'PARAMS')
        return (f"{PREFILL_COMMAND} --params {params} --run {run} (or the "
                f"campaign's prefill_nodetables.sh)")

    # ------------------------------------------------------------------
    # routes

    def plan(self, coverage, workdir=None):
        """The routes for the trajectories of ``coverage.missing``.

        Parameters
        ----------
        coverage : aimmd.core.series.SeriesCoverage
            The run's coverage of this featurizer's series.
        workdir : str, optional
            The job's working directory, where the graph cache would be.

        Returns
        -------
        list of _Step
            Repacks first, then extract, then recompute; every missing
            trajectory is in exactly one step.
        """
        remaining = list(coverage.missing)
        steps = []
        for series, from_n_max, holders in self._repack_sources(coverage):
            take = [item for item in remaining if item in holders]
            if not take:
                continue
            steps.append(_Step('repack', take, (series, from_n_max),
                               self._repack_text(series, from_n_max)))
            taken = set(take)
            remaining = [item for item in remaining if item not in taken]
        if remaining and coverage.legacy:
            database = _graph_cache(workdir)
            if database is not None:
                from ._tool import DEFAULT_DB_VERIFY
                steps.append(_Step(
                    'extract', remaining, database,
                    f'extracting the rows from the graph cache {database} '
                    f'(checked against a direct featurization on '
                    f'{DEFAULT_DB_VERIFY} frames per trajectory; frames it '
                    f'lacks are featurized)'))
                remaining = []
        if remaining:
            steps.append(_Step('recompute', remaining, None,
                               'featurizing every frame of the trajectories'))
        return steps

    def route(self, coverage, workdir=None):
        """How `__call__` would refill, as text (see `plan`)."""
        return _describe(self.plan(coverage, workdir))

    def _repack_sources(self, coverage):
        """``(series, from_n_max, trajectories)`` of the other node-table
        series next to missing trajectories that differ from this
        featurizer's only in ``n_max``, most trajectories first."""
        from ._featurizer import SERIES_PREFIX
        featurizer = self.featurizer
        multi = hasattr(featurizer, 'system_ids')
        holders = {}
        for item in coverage.missing:
            for name in coverage.siblings.get(item[0], ()):
                if name != featurizer.series and _is_node_table_series(
                        name, SERIES_PREFIX):
                    holders.setdefault(name, set()).add(item)
        sources = []
        for name in sorted(holders, key=lambda name: (-len(holders[name]),
                                                      name)):
            capacities = {}
            for (series, system_id), fname in coverage.examples.items():
                if series != name:
                    continue
                width = _width(fname)
                if width is None or width < 6 or (width - 2) % 4:
                    capacities = None
                    break
                capacities[system_id] = (width - 2) // 4
            if not capacities:
                continue
            try:
                if multi:
                    from_n_max = {system_id: n_max for system_id, n_max
                                  in capacities.items()
                                  if system_id in featurizer.system_ids}
                else:
                    from_n_max = capacities.get(None)
                if not from_n_max:
                    continue
                source = featurizer.with_n_max(from_n_max)
            except ValueError:
                continue
            if source.series == name:
                sources.append((name, from_n_max, holders[name]))
        return sources

    def _n_max(self):
        """The current capacity: an int, or ``{system_id: n_max}``."""
        featurizer = self.featurizer
        if hasattr(featurizer, 'system_ids'):
            return {system_id: featurizer[system_id].n_max
                    for system_id in featurizer.system_ids}
        return featurizer.n_max

    def _repack_text(self, series, from_n_max):
        current = self._n_max()
        if isinstance(from_n_max, dict):
            sizes = ', '.join(f'{system_id} {old} -> {current[system_id]}'
                              for system_id, old in sorted(
                                  from_n_max.items()))
        else:
            sizes = f'{from_n_max} -> {current}'
        return (f'repacking the rows of {series} (n_max {sizes}; no '
                f'trajectory is read)')

    # ------------------------------------------------------------------
    # refill

    def __call__(self, coverage, params_file=None, workdir=None, jobs=1,
                 log=print, progress=None, stop=None):
        """Write the series of every trajectory of ``coverage.missing``.

        Parameters
        ----------
        coverage : aimmd.core.series.SeriesCoverage
            The run's coverage of this featurizer's series.
        params_file : str, optional
            The params file of the job; the module of its folder that defines
            the featurizer is what the spawned processes import.
        workdir : str, optional
            The job's working directory (where the graph cache would be).
        jobs : int, default=1
            Processes.
        log : callable, default=print
            Called with each line of the report.
        progress : callable, optional
            Called with the trajectories and frames done so far.
        stop : callable, optional
            A stop request, polled between the routes and by the tools
            between files and chunks of frames.

        Returns
        -------
        dict
            ``route`` (text) and per route the trajectories it wrote
            (``repacked``, ``extracted``, ``recomputed``).

        Raises
        ------
        aimmd.core.series.RefillStopped
            When `stop` asked to stop; the files written so far stay.
        """
        from . import _tool
        steps = self.plan(coverage, workdir)
        jobs = max(1, int(jobs))
        source = _source(self.featurizer, params_file)
        if jobs > 1 and source is None:
            log('the featurizer is not a module-level name of a module in '
                'the params folder, so other processes cannot rebuild it: '
                'refilling in this process only')
            jobs = 1
        params = _tool.ParamsFeaturizer(
            *(source or (params_file, type(self.featurizer).__name__)),
            self.featurizer, None)
        run = coverage.run
        tracker = _Tracker(progress)
        write = _quiet(log)
        counts = dict(repacked=0, extracted=0, recomputed=0)
        fallback = []
        with _hidden_gpus(jobs > 1):
            for step in steps:
                _tool._stop_requested(stop)
                if step.kind == 'repack':
                    failed = self._repack(_tool, params, run, step, jobs,
                                          write, tracker, stop)
                    counts['repacked'] += len(step.trajectories) - len(failed)
                elif step.kind == 'extract':
                    failed = self._extract(_tool, params, run, step, jobs,
                                           write, tracker, stop)
                    counts['extracted'] += (len(step.trajectories)
                                            - len(failed))
                else:
                    continue
                fallback += failed
            todo = [item for step in steps if step.kind == 'recompute'
                    for item in step.trajectories] + fallback
            if todo:
                _tool._stop_requested(stop)
                self._recompute(_tool, params, run, todo, jobs, write,
                                tracker, stop)
                counts['recomputed'] += len(todo)
        route = _describe(steps)
        if fallback:
            route += (f' ({len(fallback)} of them featurized instead, see '
                      f'above)')
        return dict(route=route, **counts)

    def _repack(self, _tool, params, run, step, jobs, log, tracker, stop):
        series, from_n_max = step.source
        report = _tool.repack(params, [run], self._n_max(),
                              from_n_max=from_n_max, jobs=jobs, log=log,
                              trajectories=step.trajectories,
                              progress=tracker.repacked, stop=stop)
        failed = [(entry['trajectory'], entry['system_id'])
                  for entry in report['files'] if entry['status'] == 'failed']
        if failed:
            log(f'{len(failed)} trajectories could not be repacked from '
                f'{series}; featurizing them instead')
        return failed

    def _extract(self, _tool, params, run, step, jobs, log, tracker, stop):
        try:
            _tool._databases([step.source], self.featurizer)
        except _tool.UsageError as error:
            log(f'the graph cache cannot be used ({error}); featurizing the '
                f'{len(step.trajectories)} trajectories instead')
            return list(step.trajectories)
        report = _tool.prefill(params, [run], db=[step.source], jobs=jobs,
                               verify=_tool.DEFAULT_DB_VERIFY,
                               only_missing=True, log=log,
                               trajectories=step.trajectories,
                               progress=tracker.prefilled, stop=stop)
        failed = [(entry['trajectory'], entry['system_id'])
                  for entry in report['files']
                  if entry['status'] not in ('written', 'complete')]
        if failed:
            log(f'{len(failed)} trajectories failed the check against a '
                f'direct featurization (or failed); featurizing them '
                f'instead')
        return failed

    def _recompute(self, _tool, params, run, todo, jobs, log, tracker, stop):
        _tool.prefill(params, [run], jobs=jobs, verify=0, only_missing=True,
                      log=log, trajectories=todo, progress=tracker.prefilled,
                      stop=stop)


# ----------------------------------------------------------------------
# helpers

def _describe(steps):
    """The routes of a plan, as text."""
    if len(steps) == 1:
        return steps[0].text
    return '; then '.join(f'{step.text} for {len(step.trajectories)} '
                          f'trajector'
                          f'{"y" if len(step.trajectories) == 1 else "ies"}'
                          for step in steps)


def _is_node_table_series(name, prefix):
    digits = name[len(prefix):]
    return (name.startswith(prefix) and len(digits) == 10
            and all(character in '0123456789abcdef' for character in digits))


def _width(fname):
    """Row width of a node-table series file; None if it is not one."""
    from ._tool import _series_info
    try:
        with open(fname, 'rb') as file:
            return _series_info(file, fname).width
    except (OSError, ValueError):
        return None


def _graph_cache(workdir):
    """The graph cache in the working directory, if it can be read here."""
    database = os.path.abspath(os.path.join(workdir or '.', GRAPH_CACHE))
    if not os.path.isfile(database):
        return None
    if importlib.util.find_spec('torch_geometric') is None:
        return None
    return database


def _source(featurizer, params_file):
    """``(file, name)``: the module-level name that holds `featurizer` in a
    module of the params folder (the params file that built it, which a
    spawned process can import again); None if there is none."""
    if not params_file:
        return None
    folder = os.path.dirname(os.path.abspath(str(params_file)))
    for key, module in list(sys.modules.items()):
        if key in ('__main__', '__mp_main__'):
            continue
        fname = getattr(module, '__file__', None)
        if not isinstance(fname, str) or not fname.endswith('.py'):
            continue
        if os.path.dirname(os.path.abspath(fname)) != folder:
            continue
        try:
            names = list(vars(module).items())
        except TypeError:
            continue
        for name, value in names:
            if value is featurizer and not name.startswith('__'):
                return os.path.abspath(fname), name
    return None


def _quiet(log):
    """`log` without the per-file lines of files that went well and the
    command-line advice of the tools."""
    def write(line):
        words = line.split(maxsplit=1)
        if not words or words[0] in _QUIET or line.startswith(_ADVICE):
            return
        log(line)
    return write


@contextlib.contextmanager
def _hidden_gpus(active=True):
    """Hide the GPUs from the processes a parallel refill spawns: they import
    the params file, which may move its network to a GPU."""
    if not active:
        yield
        return
    old = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = old


class _Tracker:
    """Trajectories and frames done, over the routes, for progress lines."""

    def __init__(self, progress):
        self.progress = progress
        self.files = {}

    def prefilled(self, entry):
        done = entry['status'] in ('written', 'complete', 'mismatch',
                                   'failed')
        self._update(entry['trajectory'], done, entry['computed'])

    def repacked(self, entry):
        self._update(entry['trajectory'], entry['status'] != 'pending',
                     entry['rows'])

    def _update(self, trajectory, done, frames):
        self.files[trajectory] = (done, frames)
        if self.progress is not None:
            self.progress(sum(done for done, _ in self.files.values()),
                          sum(frames for _, frames in self.files.values()))
