"""
aimmd.engines.gromacs
=====================

What a GROMACS checkpoint says about the run that resumes from it.

AIMMD resumes every interrupted ``mdrun`` from ``<deffnm>.cpt`` (``-cpi``).
Before it does, the worker rolls the trajectory and its per-frame caches back
to that checkpoint (see :mod:`aimmd.worker.rollback`). For that it needs what
the checkpoint records:

- the simulation part, and the step and time the checkpoint was written at;
- for every output file, the number of bytes the checkpoint accounts for (the
  length GROMACS truncates the file to when it appends) and the MD5 checksum of
  the bytes before that offset (which GROMACS checks before it appends).

They are read with ``gmx dump -cp``, the only interface GROMACS offers to its
checkpoint format. The dump also lists every coordinate and velocity, so it is
parsed as a stream, line by line: the header comes first and the list of output
files last.

Functions
---------
read_checkpoint(fname, gmx)
    Run ``gmx dump -cp`` on a checkpoint and parse what it prints.
parse_checkpoint_dump(lines)
    Parse the output of ``gmx dump -cp``.
gmx_command(params)
    The command that runs a GROMACS tool the way AIMMD runs ``grompp``.
checksum_matches(fname, output)
    Whether a file still holds the bytes a checkpoint recorded for it.
format_ps(time)
    A time in ps for log messages.
"""

# external
import os
import shlex
import signal
import hashlib
import tempfile
import threading
import subprocess
from dataclasses import dataclass


class CheckpointError(RuntimeError):
    """A checkpoint could not be read, or its dump could not be parsed."""


@dataclass(frozen=True)
class CheckpointOutput:
    """
    One output file as a GROMACS checkpoint records it.

    Parameters
    ----------
    name : str
        File name as GROMACS wrote it, relative to the working directory of
        ``mdrun`` (e.g. ``'forw.xtc'`` or ``'traj000001.part0002.xtc'``).
    offset : int
        Bytes of the file the checkpoint accounts for. When ``mdrun`` resumes
        by appending, it truncates the file to this length first.
    checksum_size : int
        Number of bytes before `offset` covered by `checksum` (1 MiB, or the
        whole file when it is shorter); -1 if GROMACS recorded no checksum.
    checksum : str
        Hexadecimal MD5 digest of those bytes ('' if none was recorded).
    """
    name: str
    offset: int
    checksum_size: int = -1
    checksum: str = ''


@dataclass(frozen=True)
class Checkpoint:
    """
    The parts of a GROMACS checkpoint that AIMMD uses.

    Parameters
    ----------
    fname : str
        The checkpoint file.
    part : int
        Simulation part the checkpoint was written by. ``mdrun -noappend``
        resuming from it writes part ``part + 1``.
    step : int
        MD step the checkpoint was written at.
    time : float
        Simulation time (ps) the checkpoint was written at.
    outputs : tuple of CheckpointOutput
        The output files ``mdrun`` had open (log, trajectories, energies).
    """
    fname: str
    part: int
    step: int
    time: float
    outputs: tuple = ()

    def output(self, fname):
        """
        The record of output file `fname`, matched by base name.

        Parameters
        ----------
        fname : str
            An output file, e.g. ``'run1/chainR0/back.xtc'``.

        Returns
        -------
        CheckpointOutput or None
            None if the checkpoint lists no file of that name.
        """
        name = os.path.basename(str(fname))
        for output in self.outputs:
            if os.path.basename(output.name) == name:
                return output
        return None

    def describe(self):
        """Short description, e.g. ``'part 1, step 72829100, t = 145658.2 ps'``."""
        return f'part {self.part}, step {self.step}, t = {format_ps(self.time)} ps'


def format_ps(time):
    """A time in ps, with up to three decimals: ``145658.2``, ``0.002``."""
    return f'{time:.3f}'.rstrip('0').rstrip('.')


# keys of `gmx dump -cp` lines -> fields; the first occurrence (the header) wins
_HEADER = {'simulation part #': 'part', 'step': 'step', 't': 'time'}
_OUTPUT_KEYS = ('file_offset_high', 'file_offset_low',
                'file_checksum_size', 'file_checksum')


def _scan(lines):
    """
    Collect the raw checkpoint fields from the lines of a ``gmx dump -cp``.

    Nothing is validated and nothing is kept but the few fields AIMMD uses, so
    the coordinate and velocity listings of a large system pass through
    without being stored.

    Returns
    -------
    header : dict
        ``{'part': str, 'step': str, 'time': str}``, as far as found.
    n_outputs : str or None
        The announced number of output files.
    outputs : list of dict
        One dict per ``output filename`` line, with the keys that followed it.
    """
    header = {}
    n_outputs = None
    outputs = []
    current = None
    for line in lines:
        # array elements and matrix rows are indented: skip them fast
        if not line or line[0] in ' \t':
            continue
        key, separator, value = line.partition(' = ')
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if key in _HEADER:
            header.setdefault(_HEADER[key], value)
        elif key == 'number of output files':
            n_outputs = value
            outputs = []
            current = None
        elif key == 'output filename':
            current = {'name': value}
            outputs.append(current)
        elif current is not None and key in _OUTPUT_KEYS:
            current[key] = value
    return header, n_outputs, outputs


def _build(fname, header, n_outputs, outputs):
    """Validate what :func:`_scan` collected and build the Checkpoint."""
    missing = [key for key in ('part', 'step', 'time') if key not in header]
    if missing:
        raise CheckpointError(
            f'the dump of {fname!r} has no {" / ".join(missing)} line')
    try:
        part = int(header['part'])
        step = int(header['step'])
        time = float(header['time'])
    except ValueError as exception:
        raise CheckpointError(
            f'unexpected header in the dump of {fname!r}: {exception}') from None
    if n_outputs is None:
        raise CheckpointError(
            f'the dump of {fname!r} ends before its list of output files')

    records = []
    for entry in outputs:
        name = entry['name']
        try:
            high = int(entry['file_offset_high'])
            # GROMACS prints the low 32 bits as a *signed* int
            low = int(entry['file_offset_low']) & 0xFFFFFFFF
        except (KeyError, ValueError):
            raise CheckpointError(
                f'the dump of {fname!r} has no complete offset for output file '
                f'{name!r}') from None
        checksum = ''.join(c for c in entry.get('file_checksum', '').lower()
                           if c in '0123456789abcdef')
        try:
            checksum_size = int(entry.get('file_checksum_size', -1))
        except ValueError:
            checksum_size = -1
        if len(checksum) != 32:  # not an MD5 digest: treat as unrecorded
            checksum, checksum_size = '', -1
        records.append(CheckpointOutput(
            name, (high << 32) | low, checksum_size, checksum))

    try:
        expected = int(n_outputs)
    except ValueError:
        expected = -1
    if expected != len(records):
        raise CheckpointError(
            f'the dump of {fname!r} announces {n_outputs} output files but '
            f'lists {len(records)}')
    return Checkpoint(str(fname), part, step, time, tuple(records))


def parse_checkpoint_dump(lines, fname='<dump>'):
    """
    Parse the output of ``gmx dump -cp``.

    Parameters
    ----------
    lines : iterable of str
        The lines ``gmx dump -cp`` printed (a file object, a list, a stream).
        They are read once, in order, and not stored.
    fname : str, optional
        Checkpoint name used in error messages and stored in the result.

    Returns
    -------
    Checkpoint

    Raises
    ------
    CheckpointError
        If the header (part, step, time) or the list of output files is
        missing or incomplete, e.g. because the dump was cut short.

    Notes
    -----
    The offsets are printed as two 32-bit halves, the low one as a signed
    integer; they are combined as ``(high << 32) | (low & 0xFFFFFFFF)``, as
    GROMACS does when it reads them back.
    """
    return _build(fname, *_scan(lines))


def _gmx_error(text, max_lines=4):
    """The 'Fatal error' paragraph of a GROMACS error, else its last lines."""
    lines = [line.strip() for line in text.splitlines()]
    if 'Fatal error:' in lines:
        found = []
        for line in lines[lines.index('Fatal error:') + 1:]:
            if line.startswith(('For more information', '-----')):
                break
            if line:
                found.append(line)
        if found:
            return ' '.join(found)
    return ' | '.join([line for line in lines if line][-max_lines:]) \
        or 'no error message'


def read_checkpoint(fname, gmx='gmx', timeout=600.):
    """
    Read a GROMACS checkpoint with ``gmx dump -cp``.

    Parameters
    ----------
    fname : str
        Checkpoint file (``.cpt``).
    gmx : str, optional
        Command that runs GROMACS tools, e.g. ``'gmx'``, ``'/opt/bin/gmx_mpi'``
        or ``'srun -n 1 gmx_mpi'`` (see :func:`gmx_command`). It is run through
        the shell as ``f'{gmx} dump -cp {fname}'``.
    timeout : float, optional
        Seconds after which the dump is killed. The dump of a system of 70,000
        atoms takes about 0.3 s.

    Returns
    -------
    Checkpoint

    Raises
    ------
    CheckpointError
        If the file does not exist, the command cannot be started, fails, times
        out, or prints something that is not a complete checkpoint dump. The
        message quotes the GROMACS error.
    """
    fname = str(fname)
    if not os.path.isfile(fname):
        raise CheckpointError(f'{fname!r} does not exist')
    command = f'{gmx} dump -cp {shlex.quote(fname)}'

    with tempfile.TemporaryFile(mode='w+', errors='replace') as stderr:
        try:
            process = subprocess.Popen(
                command, shell=True, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=stderr, text=True,
                errors='replace', start_new_session=True)
        except OSError as exception:
            raise CheckpointError(
                f'could not run {command!r}: {exception}') from None

        def kill():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass

        timed_out = threading.Event()
        timer = threading.Timer(timeout, lambda: (timed_out.set(), kill()))
        timer.daemon = True
        timer.start()
        try:
            header, n_outputs, outputs = _scan(process.stdout)
        finally:
            timer.cancel()
            process.stdout.close()
            try:
                process.wait(timeout=60.)
            except subprocess.TimeoutExpired:
                kill()
                process.wait()
        returncode = process.returncode
        stderr.seek(0)
        message = _gmx_error(stderr.read())

    if timed_out.is_set():
        raise CheckpointError(
            f'{command!r} did not finish within {timeout:g} s')
    if returncode:
        raise CheckpointError(
            f'{command!r} failed with exit code {returncode}: {message}')
    try:
        return _build(fname, header, n_outputs, outputs)
    except CheckpointError as exception:
        raise CheckpointError(f'{exception} ({command!r}: {message})') from None


def gmx_command(params=None):
    """
    The command that runs a GROMACS tool the way AIMMD runs ``grompp``.

    Parameters
    ----------
    params : aimmd.Params or None
        Its ``gmx_grompp`` (e.g. ``'/opt/gromacs/bin/gmx grompp -maxwarn 1'``)
        gives the command: everything before the ``grompp`` word. A tool then
        runs with the same executable, and the same launcher prefix if any, as
        the ``grompp`` calls that built the run's ``.tpr`` files.

    Returns
    -------
    str
        That prefix, else the executable AIMMD resolved from ``PATH`` at import
        (``aimmd._config.GROMACS``), else ``'gmx'``.
    """
    grompp = getattr(params, 'gmx_grompp', None)
    if isinstance(grompp, (list, tuple)):  # multi-system: one per system
        grompp = grompp[0] if grompp else None
    if isinstance(grompp, str):
        words = grompp.split()
        if 'grompp' in words:
            prefix = ' '.join(words[:words.index('grompp')])
            if prefix and prefix != 'None':  # 'None': GROMACS was not found
                return prefix
    from .. import _config
    return getattr(_config, 'GROMACS', None) or 'gmx'


def checksum_matches(fname, output):
    """
    Whether `fname` still holds the bytes a checkpoint recorded for it.

    GROMACS resumes by appending only if the file is at least `output.offset`
    bytes long and the MD5 digest of the `output.checksum_size` bytes before
    that offset equals the recorded one; otherwise ``mdrun`` stops with "The
    file has been replaced or its contents have been modified".

    Parameters
    ----------
    fname : str
        The output file on disk.
    output : CheckpointOutput
        Its record in the checkpoint.

    Returns
    -------
    bool or None
        Whether GROMACS will accept the file; None if the checkpoint recorded
        no checksum (GROMACS then appends without checking).
    """
    if output.checksum_size < 0 or not output.checksum:
        return None
    start = output.offset - output.checksum_size
    if start < 0:
        return False
    with open(fname, 'rb') as file:
        file.seek(start)
        data = file.read(output.checksum_size)
    if len(data) != output.checksum_size:
        return False
    return hashlib.md5(data).hexdigest() == output.checksum
