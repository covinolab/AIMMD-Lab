"""Command line of ``python -m aimmd.network.nodetables``.

::

    python -m aimmd.network.nodetables prefill --params PARAMS --run RUN
        [--run RUN ...] [--db [SYSTEM_ID=]DB ...] [-j N] [--verify K]
        [--only-missing] [--chunk-frames N] [--seed S] [--report FILE]
    python -m aimmd.network.nodetables repack --params PARAMS --run RUN
        [--run RUN ...] --n-max N [--from-n-max M] [--overwrite] [-j N]
        [--report FILE]
    python -m aimmd.network.nodetables verify --params PARAMS --run RUN
        [--run RUN ...] [--sample K] [-j N] [--seed S] [--report FILE]

Exit status: 0 when everything is complete and verified, 1 when a file
failed, a verified row mismatched, rows are left empty (``prefill``) or
series are incomplete (``verify``), 2 on bad input (params file, runs,
options). See `aimmd.network.nodetables._tool` for what each command does.
"""

import argparse
import json
import os
import sys

from . import _tool

DESCRIPTION = '''\
Node-table series of AIMMD runs (graph-network inputs stored per trajectory
as {trajectory}.{descriptors_series}.npy, see aimmd.network.nodetables).

  prefill  write the series of every trajectory of the runs (those with a
           states series), extracting rows from an old graph cache (--db)
           or featurizing the frames
  repack   rewrite the series for another row capacity (n_max); reads no
           trajectory
  verify   report missing, short and zero rows, and check sampled rows

The featurizer comes from the params file, which must be in node-table mode
(it defines a NodeTableFeaturizer); it is imported as a module in its folder,
without aimmd.Params. Run the commands while no job runs on the runs. The
tools never open *.descriptors.npy and never write a graph cache. GPUs are
not used.
'''


def _positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'must be at least 1, got {value}')
    return value


def _non_negative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f'must be at least 0, got {value}')
    return value


def _parser():
    parser = argparse.ArgumentParser(
        prog='python -m aimmd.network.nodetables', description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    def common(command, help, description):
        sub = commands.add_parser(command, help=help, description=description)
        sub.add_argument('--params', required=True,
                         help='params file in node-table mode')
        sub.add_argument('--run', required=True, action='append',
                         metavar='RUN', help='run folder (repeatable)')
        sub.add_argument('--featurizer', metavar='NAME',
                         help='name of the featurizer in the params file '
                              '(default: FEATURIZER, or the only one)')
        sub.add_argument('-j', '--jobs', type=_positive, default=1,
                         metavar='N', help='processes (default: 1)')
        sub.add_argument('--report', metavar='FILE',
                         help='write the report (per file, with timings) as '
                              'JSON to FILE')
        return sub

    prefill = common(
        'prefill', 'write the series of every trajectory of the runs',
        'Write {trajectory}.{series}.npy for every trajectory with a states '
        'series. With --db, rows come from the graphs an old graph cache '
        'holds for the frames (opened read-only); frames it has no usable '
        'graph for are featurized. Files are replaced only when complete '
        '(and verified).')
    prefill.add_argument(
        '--db', action='append', metavar='[SYSTEM_ID=]DB',
        help='graph cache (graphs_cache.sqlite) of the coordinate-descriptor '
             'mode to extract rows from; SYSTEM_ID=DB for one system of a '
             'multi-system run (repeatable)')
    prefill.add_argument(
        '--verify', type=_non_negative, metavar='K',
        help='featurize K random frames per trajectory directly and compare '
             'them bit for bit with the rows (default: '
             f'{_tool.DEFAULT_DB_VERIFY} with --db, else 0)')
    prefill.add_argument(
        '--only-missing', action='store_true',
        help='compute only missing and zero rows and keep the others '
             '(default: compute every row)')
    prefill.add_argument(
        '--chunk-frames', type=_positive, default=_tool.DEFAULT_CHUNK_FRAMES,
        metavar='N', help=f'frames per task (default: '
                          f'{_tool.DEFAULT_CHUNK_FRAMES})')
    prefill.add_argument('--seed', type=int, default=0,
                         help='seed of the verified frames (default: 0)')

    repack = common(
        'repack', 'rewrite the series for another n_max',
        'Rewrite every series file of the featurizer into the layout of '
        'capacity --n-max (a new series name, printed at the end). No '
        'trajectory is read; rows of frames that did not fit stay zero for '
        'prefill --only-missing to fill.')
    repack.add_argument('--n-max', type=_positive, required=True,
                        metavar='N', help='row capacity of the new series')
    repack.add_argument('--from-n-max', type=_positive, metavar='M',
                        help='row capacity of the existing series, if the '
                             'params file already holds the new one')
    repack.add_argument('--overwrite', action='store_true',
                        help='replace existing files of the new series')

    check = common(
        'verify', 'report missing, short and zero rows',
        'Report, per trajectory with a states series, whether its series '
        'file is complete: missing file, missing (short) and zero rows, rows '
        'of another layout; with --sample, compare sampled rows bit for bit '
        'with a direct featurization.')
    check.add_argument('--sample', type=_non_negative, default=0,
                       metavar='K', help='rows per trajectory to featurize '
                                         'directly and compare (default: 0)')
    check.add_argument('--seed', type=int, default=0,
                       help='seed of the sampled frames (default: 0)')
    return parser


def _write_report(fname, report):
    """Write the JSON report (via a temporary file)."""
    temporary = f'{fname}.tmp'
    with open(temporary, 'w') as file:
        json.dump(report, file, indent=1, default=_json_value)
        file.write('\n')
    os.replace(temporary, fname)


def _json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def main(argv=None):
    """Run a command; returns the exit status.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (default: ``sys.argv[1:]``).

    Returns
    -------
    int
        0 if everything is complete and verified, 1 if not, 2 on bad input.
    """
    args = _parser().parse_args(argv)
    report_file = os.path.abspath(args.report) if args.report else None
    try:
        params = _tool.load_featurizer(args.params, args.featurizer)
        if args.command == 'prefill':
            report = _tool.prefill(
                params, args.run, db=args.db, jobs=args.jobs,
                verify=args.verify, only_missing=args.only_missing,
                chunk_frames=args.chunk_frames, seed=args.seed)
        elif args.command == 'repack':
            report = _tool.repack(
                params, args.run, args.n_max, from_n_max=args.from_n_max,
                jobs=args.jobs, overwrite=args.overwrite)
        else:
            report = _tool.verify(params, args.run, sample=args.sample,
                                  jobs=args.jobs, seed=args.seed)
    except _tool.UsageError as error:
        print(f'error: {error}', file=sys.stderr, flush=True)
        return 2
    if report_file:
        _write_report(report_file, report)
    return 0 if report['ok'] else 1
