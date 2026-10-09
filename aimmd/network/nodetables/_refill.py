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
from them. What they print goes to a file (`_tool.CHILD_OUTPUT`) whose lines
the refill passes on to its log. When they cannot start (e.g. a params file
that needs a GPU when it is imported) or die, the refill logs their error
and goes on in its own process.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import importlib.util
import os
import sys
import tempfile
from collections import namedtuple

#: The graph cache of the graph-cache input, in the job's working directory.
GRAPH_CACHE = 'graphs_cache.sqlite'

#: Command that fills a node-table series offline.
PREFILL_COMMAND = 'python -m aimmd.network.nodetables prefill'

#: Command that rewrites a node-table series for another n_max.
REPACK_COMMAND = 'python -m aimmd.network.nodetables repack'

# a route for some trajectories: kind is 'repack', 'extract' or 'recompute';
# source is (series, from_n_max) for repack and the graph cache for extract
_Step = namedtuple('_Step', 'kind trajectories source text')

# lines of the spawned processes passed on to the log per tool call, at most
_RELAYED_LINES = 200

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

    def prefill_command(self, params_file, run, coverage=None,
                        workdir=None):
        """The command that fills the series of `run` offline, by the route
        the refill would take for the trajectories of `coverage` (see
        `plan`): a repack for an ``n_max`` change (then ``prefill
        --only-missing`` for the rows that did not fit), else ``prefill
        --only-missing``, from the graph cache when it would be extracted."""
        source = _source(self.featurizer, params_file)
        params = source[0] if source else (params_file or 'PARAMS')
        where = f'--params {params} --run {run}'
        prefill = f'{PREFILL_COMMAND} {where} --only-missing'
        steps = self.plan(coverage, workdir) if coverage is not None else []
        repacks = [self._repack_command(where, step.source[1])
                   for step in steps if step.kind == 'repack']
        if repacks and all(repacks):
            others = any(step.kind != 'repack' for step in steps)
            return (f'{"; ".join(repacks)} (no trajectory is read), then '
                    f'{prefill} (the rows that did not fit'
                    + (', and the trajectories without a series of another '
                       'n_max' if others else '') + ')')
        for step in steps:
            if step.kind == 'extract':
                prefill += f' --db {step.source}'
        return f"{prefill} (or the campaign's prefill_nodetables.sh)"

    def _repack_command(self, where, from_n_max):
        """The repack command from `from_n_max` to the current ``n_max``;
        None if the command line cannot give it (systems whose new
        ``n_max`` differ)."""
        current = self._n_max()
        if isinstance(current, dict):
            sizes = set(current.values())
            if len(sizes) != 1:
                return None
            n_max = sizes.pop()
            old = ' '.join(f'--from-n-max {system_id}={size}'
                           for system_id, size in sorted(from_n_max.items()))
        else:
            n_max, old = current, f'--from-n-max {from_n_max}'
        return f'{REPACK_COMMAND} {where} --n-max {n_max} {old}'

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
        with _hidden_gpus(jobs > 1), _Execution(jobs, log, stop) as execute:
            for step in steps:
                _tool._stop_requested(stop)
                if step.kind == 'repack':
                    failed = self._repack(_tool, params, run, step, execute,
                                          write, tracker)
                    counts['repacked'] += len(step.trajectories) - len(failed)
                elif step.kind == 'extract':
                    failed = self._extract(_tool, params, run, step, execute,
                                           write, tracker)
                    counts['extracted'] += (len(step.trajectories)
                                            - len(failed))
                else:
                    continue
                fallback += failed
            todo = [item for step in steps if step.kind == 'recompute'
                    for item in step.trajectories] + fallback
            if todo:
                _tool._stop_requested(stop)
                self._recompute(_tool, params, run, todo, execute, write,
                                tracker)
                counts['recomputed'] += len(todo)
        route = _describe(steps)
        if fallback:
            route += (f' ({len(fallback)} of them featurized instead, see '
                      f'above)')
        return dict(route=route, **counts)

    def _repack(self, _tool, params, run, step, execute, log, tracker):
        series, from_n_max = step.source
        report = execute(_tool.repack, params, [run], self._n_max(),
                         from_n_max=from_n_max, log=log,
                         trajectories=step.trajectories,
                         progress=tracker.repacked)
        failed = [(entry['trajectory'], entry['system_id'])
                  for entry in report['files'] if entry['status'] == 'failed']
        if failed:
            log(f'{len(failed)} trajectories could not be repacked from '
                f'{series}; featurizing them instead')
        return failed

    def _extract(self, _tool, params, run, step, execute, log, tracker):
        try:
            _tool._databases([step.source], self.featurizer)
        except _tool.UsageError as error:
            log(f'the graph cache cannot be used ({error}); featurizing the '
                f'{len(step.trajectories)} trajectories instead')
            return list(step.trajectories)
        report = execute(_tool.prefill, params, [run], db=[step.source],
                         verify=_tool.DEFAULT_DB_VERIFY, only_missing=True,
                         log=log, trajectories=step.trajectories,
                         progress=tracker.prefilled)
        failed = [(entry['trajectory'], entry['system_id'])
                  for entry in report['files']
                  if entry['status'] not in ('written', 'complete')]
        if failed:
            log(f'{len(failed)} trajectories failed the check against a '
                f'direct featurization (or failed); featurizing them '
                f'instead')
        return failed

    def _recompute(self, _tool, params, run, todo, execute, log, tracker):
        execute(_tool.prefill, params, [run], verify=0, only_missing=True,
                log=log, trajectories=todo, progress=tracker.prefilled)


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


class _Execution:
    """Runs the tools with `jobs` processes and the stop request.

    With more than one process, what the spawned processes print goes to a
    temporary file (`_tool.CHILD_OUTPUT`), whose new lines are passed on to
    `log` after every tool call (each distinct line once per call). When the
    processes could not start or died (a broken pool, e.g. a params file
    that fails to import without a GPU), the call runs again in this
    process, and so does every later one. Use it as a context manager.
    """

    def __init__(self, jobs, log, stop):
        self.jobs = jobs
        self.log = log
        self.stop = stop
        self.output = None
        self.offset = 0
        self.environment = None

    def __enter__(self):
        if self.jobs > 1:
            handle, self.output = tempfile.mkstemp(prefix='aimmd-refill-',
                                                   suffix='.out')
            os.close(handle)
            self.environment = os.environ.get(_tool_child_output())
            os.environ[_tool_child_output()] = self.output
        return self

    def __exit__(self, *exc_info):
        if self.output is None:
            return
        try:
            self.relay(final=True)
        finally:
            if self.environment is None:
                os.environ.pop(_tool_child_output(), None)
            else:
                os.environ[_tool_child_output()] = self.environment
            with contextlib.suppress(OSError):
                os.remove(self.output)
            self.output = None

    def __call__(self, tool, *args, **kwargs):
        """``tool(*args, jobs=..., stop=..., **kwargs)``; its report."""
        from . import _tool
        if self.jobs == 1:
            return tool(*args, jobs=1, stop=self.stop, **kwargs)
        try:
            report = tool(*args, jobs=self.jobs, stop=self.stop, **kwargs)
            broken = bool(report.get('broken_pool'))
        except concurrent.futures.BrokenExecutor as error:
            report, broken = None, True
            self.log(f'{_tool._error_text(error)}')
        finally:
            self.relay()
        if not broken:
            return report
        _tool._stop_requested(self.stop)     # stopping is no breakdown
        self.log(f'the {self.jobs} processes of the refill could not start '
                 f'or died (see the lines above): refilling in this process '
                 f'instead')
        self.jobs = 1
        return tool(*args, jobs=1, stop=self.stop, **kwargs)

    def relay(self, final=False):
        """Pass the new lines of the spawned processes on to the log."""
        if self.output is None:
            return
        try:
            with open(self.output, 'rb') as file:
                file.seek(self.offset)
                data = file.read()
        except OSError:
            return
        end = len(data) if final else data.rfind(b'\n') + 1
        self.offset += end
        lines, seen = [], set()
        for line in data[:end].decode(errors='replace').splitlines():
            if line.strip() and line not in seen:
                seen.add(line)
                lines.append(line)
        for line in lines[:_RELAYED_LINES]:
            self.log(line)
        if len(lines) > _RELAYED_LINES:
            self.log(f'... and {len(lines) - _RELAYED_LINES} more lines of '
                     f'the spawned processes')


def _tool_child_output():
    from ._tool import CHILD_OUTPUT
    return CHILD_OUTPUT


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
