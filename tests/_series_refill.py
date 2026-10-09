"""Helpers of the series-refill tests.

`make_full_campaign` writes a toy campaign that `aimmd.Params` loads: a
params file in node-table mode (toy engine, the toy protein-ligand system of
`tests._nodetables_toy`, ``descriptors_series = FEATURIZER.series``), its
initial path and a run folder with the trajectory layout of
`tests._nodetables_run`. Run as a script, this module is one process of a
run for the barrier tests: it loads the params as a worker does and passes
the worker's series check, logging to a file.
"""
import os
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

from tests._nodetables_run import LAYOUT, write_run
from tests._nodetables_toy import (ATOM_TYPES, CUTOFF, ENVIRONMENT_SELECTION,
                                   SYSTEM_SELECTION, toy_frames,
                                   write_toy_gro, write_toy_xtc)

FULL_PARAMS = '''
import numpy as np
import torch
import MDAnalysis as mda
from aimmd.network.nodetables import NodeTableFeaturizer

engine = 'toy'
initial_paths = ['initial.xtc']
topology = 'toy.gro'

FEATURIZER = NodeTableFeaturizer(
    mda.Universe('toy.gro', to_guess=['types', 'bonds']),
    {system!r}, {environment!r}, {atom_types!r}, cutoff={cutoff!r},
    n_max={n_max!r}, refill={refill!r})
descriptors_series = FEATURIZER.series


def descriptors_function(trajectory):
    return FEATURIZER.descriptors_function(trajectory)


def states_function(trajectory):
    y = np.array([ts.positions[:6, 1].mean() for ts in trajectory])
    labels = np.full(len(y), 'R', dtype='<U1')
    labels[y < 7.0] = 'A'
    labels[y > 13.0] = 'B'
    return labels


def toy_mdrun(ts):
    ts.positions[:] = ts.positions + 0.1


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.lin(torch.as_tensor(x, dtype=torch.float32)[:, :1])


network = Network()
'''

Campaign = namedtuple('Campaign', 'folder params run')

# the ligand moves along y: A (y < 7) -> R -> B (y > 13)
INITIAL_FRAMES = toy_frames(12, seed=4, ligand_y=np.linspace(5.0, 15.0, 12))


def write_full_params(folder, n_max=64, refill=False,
                      environment=ENVIRONMENT_SELECTION):
    """Write the params file ``params.py`` into `folder`."""
    params = Path(folder) / 'params.py'
    params.write_text(FULL_PARAMS.format(
        system=SYSTEM_SELECTION, environment=environment,
        atom_types=ATOM_TYPES, cutoff=CUTOFF, n_max=n_max, refill=refill))
    return str(params)


def make_full_campaign(folder, n_max=64, refill=False,
                       environment=ENVIRONMENT_SELECTION):
    """toy.gro (as `make_campaign` writes it), initial.xtc, params.py and
    the run folder run1/ (with a stale ``*.descriptors.npy`` next to every
    trajectory)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_toy_gro(folder / 'toy.gro', toy_frames(1, seed=10)[0])
    write_toy_xtc(folder / 'initial.xtc', INITIAL_FRAMES)
    params = write_full_params(folder, n_max, refill, environment)
    write_run(folder / 'run1')
    return Campaign(str(folder), params, str(folder / 'run1'))


def run_trajectories(run, layout=LAYOUT):
    """The trajectories of the toy run that the series check covers (the
    initial path left out), absolute paths."""
    return [os.path.abspath(Path(run) / name) for name in layout
            if not name.startswith('initial')]


def remove_legacy_series(run):
    """Delete the stale ``*.descriptors.npy`` of the toy run."""
    for fname in Path(run).rglob('*.descriptors.npy'):
        fname.unlink()


# ----------------------------------------------------------------------
# a process of a run (barrier tests)

class _Held:
    """The registered refill callable, counted in a file, held for a while
    (so that the other processes find the lock taken) and, in mode
    'fail', failing instead."""

    def __init__(self, refiller, counter, mode):
        self.refiller = refiller
        self.counter = counter
        self.mode = mode

    def __getattr__(self, name):
        return getattr(self.refiller, name)

    def _count(self, event):
        with open(self.counter, 'a') as file:
            file.write(f'{event} {os.getpid()} {time.time()!r}\n')

    def __call__(self, *args, **kwargs):
        self._count('start')
        time.sleep(2.0)
        if self.mode == 'fail':
            raise RuntimeError('the refill failed on purpose')
        result = self.refiller(*args, **kwargs)
        self._count('end')
        return result


def child(params_file, directory, log_file, counter, ready, n_ready, mode):
    """Load the params as a worker does, wait for the other processes, then
    pass the worker's series check. Exit status 0 when it may start, 3 when
    the check raised."""
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    import aimmd
    from aimmd.core import series
    series.WAIT_SECONDS = 0.2
    series.WAIT_REPORT_SECONDS = 1.0
    series.PROGRESS_SECONDS = 0.
    os.chdir(os.path.dirname(params_file))
    params = aimmd.Params(params_file, initial_paths=None, save=False)
    policy = series.series_policy(params.descriptors_series)
    series.register_series(policy.series, policy.refill,
                           _Held(policy.refiller, counter, mode))
    Path(ready, str(os.getpid())).touch()
    deadline = time.time() + 300
    while len(os.listdir(ready)) < int(n_ready):
        if time.time() > deadline:
            return 4
        time.sleep(0.02)
    with open(log_file, 'a') as file:
        def log(line):
            file.write(f'{line}\n')
            file.flush()
        try:
            series.ensure_series_coverage(
                params, directory, role=f'test process {os.getpid()}',
                log=log)
        except series.SeriesCoverageError:
            log(f'RAISED {time.time()!r}')
            return 3
        log(f'STARTED {time.time()!r}')
    return 0


if __name__ == '__main__':
    sys.exit(child(*sys.argv[1:]))
