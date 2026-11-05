"""
aimmd.cgaimmd.preparation.gromacs
=================================

GROMACS plumbing shared by :mod:`aimmd.cgaimmd.preparation.minimize` and
:mod:`aimmd.cgaimmd.preparation.solvate`.

All GROMACS work of a candidate happens in a directory of its own under
``context.scratch`` and uses the executables of the run's params
(``gmx_grompp``, ``gmx_mdrun``; :func:`gmx_executable` for ``solvate`` and
``genion``): every call gets the full executable path and nothing relies on
a login shell, because spawned AIMMD workers need not have sourced
``GMXRC``.

GROMACS refuses an option that is given twice, so the tools remove from the
configured commands the options they set themselves before they add theirs
(:func:`split_grompp`, :func:`sanitize_mdrun`). They remove them only from
the GROMACS part of a command, so a launcher prefix (``srun -c 8``,
``mpirun -np 4``) keeps its options. A configured index file (grompp
``-n``) is kept, with an absolute path, for every grompp of the full
system. Coordinates handed to GROMACS are written at 6 decimals (nm), so a
candidate is not rounded to 0.01 Å before grompp; GROMACS reads ``.gro``
files at the precision they are written in.

Standard library and numpy at import; ``aimmd.execute`` and MDAnalysis
inside the functions that need them.
"""

import os
import re
import shlex
import shutil
import time
from pathlib import Path

import numpy as np

from .topology import RESTRAINT_DEFINE

__all__ = ['gmx_executable', 'split_grompp', 'sanitize_mdrun', 'topology_file', 'mdp_text',
           'run_gmx', 'write_gro', 'read_gro', 'read_trr_last_frame', 'parse_em_log',
           'RESTRAINT_MODES', 'RESTRAINT_DEFINE', 'restraint_mode']

RESTRAINT_MODES = ('generator_heavy', 'generator_all', 'none')
"""What the minimizer and the solvator restrain: the heavy generator atoms (mass above 1.5),
all generator atoms, or nothing."""

#: mdrun options sanitize_mdrun drops (with their value) and flags it drops
_MDRUN_DROP_WITH_VALUE = ('-plumed', '-cpi', '-cpt', '-maxh', '-deffnm')
_MDRUN_DROP_FLAGS = ('-noappend',)
#: options the tools set on grompp and mdrun themselves (dropped before they add theirs); an
#: index file (-n) is kept, and so is -maxwarn unless a tool sets it
_GROMPP_OWN_WITH_VALUE = ('-f', '-c', '-r', '-p', '-o', '-po', '-t')
_GROMPP_OWN_FLAGS = ('-nobackup', '-backup')
#: -nsteps on the command line would override the nsteps of the tools' mdp files
_MDRUN_OWN_WITH_VALUE = ('-s', '-o', '-c', '-e', '-g', '-x', '-cpo', '-deffnm', '-nsteps')
_MDRUN_OWN_FLAGS = ('-nobackup', '-backup')
#: GPU offloads GROMACS does not support for energy minimization (only -nb gpu is)
_MDRUN_NOT_FOR_EM = ('-pme', '-pmefft', '-bonded', '-update')
#: shell words that end one command of a command line
_SEPARATORS = ('&&', '||', ';', '|')
_TAIL_LINES = 25


# ---------------------------------------------------------------------------
# restraints
# ---------------------------------------------------------------------------

def restraint_mode(restrain):
    """The restraint mode of a ``restrain=`` argument.

    Parameters
    ----------
    restrain : str
        One of :data:`RESTRAINT_MODES`, or an old name (``'flow_heavy'``,
        ``'flow_all'``), read for one release with a ``DeprecationWarning``.

    Returns
    -------
    str
        The mode, under its new name.

    Raises
    ------
    ValueError
        For any other value.
    """
    from .. import _legacy
    mode = _legacy.translate(_legacy.RESTRAINT_MODES, restrain, 'restrain')
    if not isinstance(mode, str) or mode not in RESTRAINT_MODES:
        raise ValueError(f'restrain must be one of {RESTRAINT_MODES}, got {restrain!r}')
    return mode


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def _tokens(command):
    """Shell words of `command` with their quoting kept (``&&``, ``;`` and quotes survive a rejoin)."""
    return shlex.split(command, posix=False)


def _unquote(token):
    words = shlex.split(token)
    return words[0] if words else token


def _is_option_word(token):
    """Whether a shell word is a command-line option (``-`` and a letter; ``-1`` is a value)."""
    return re.match(r'-[A-Za-z]', _unquote(token)) is not None


def _is_gromacs_word(token):
    """Whether a shell word names GROMACS: ``gmx*``, or a name containing ``mdrun``/``grompp``."""
    name = os.path.basename(_unquote(token))
    return name.startswith('gmx') or 'mdrun' in name or 'grompp' in name


def _gromacs_options(tokens):
    """Mask of the words that are options of GROMACS (not of a launcher prefix or the shell).

    In each shell segment (split at ``&&``, ``||``, ``;``, ``|``) the words
    after the first GROMACS word (:func:`_is_gromacs_word`); a segment without
    one has none.
    """
    mask, seen = [], False
    for token in tokens:
        if token in _SEPARATORS:
            seen = False
            mask.append(False)
            continue
        mask.append(seen)
        seen = seen or _is_gromacs_word(token)
    return mask


def _strip_options(command, with_value=(), flags=()):
    """`command` without the given GROMACS options (and their values) and flags.

    Only words after the GROMACS word of their shell segment are removed
    (:func:`_gromacs_options`): ``srun -n 1 -c 8 gmx_mpi mdrun -ntomp 8``
    keeps ``-c 8``. An option's value is the next word unless that word is
    an option itself (GROMACS file options may come without a file, ``-cpi``
    alone; ``-nsteps -1`` has a value).
    """
    tokens = _tokens(command)
    options = _gromacs_options(tokens)
    out, skip = [], False
    for token, is_option in zip(tokens, options):
        if skip:
            skip = False
            if is_option and not _is_option_word(token):
                continue
        if is_option and token in with_value:
            skip = True
            continue
        if is_option and token in flags:
            continue
        out.append(token)
    return ' '.join(out)


def _option_value(command, option):
    """The value of a GROMACS option of `command` (by the rule of :func:`_strip_options`)."""
    tokens = _tokens(command)
    options = _gromacs_options(tokens)
    for k, token in enumerate(tokens[:-1]):
        if (options[k] and token == option and options[k + 1]
                and not _is_option_word(tokens[k + 1])):
            return _unquote(tokens[k + 1])
    return None


def _with_absolute_path(command, option, base):
    """`command` with the file of a GROMACS option made absolute (relative to `base`), quoted."""
    tokens = _tokens(command)
    options = _gromacs_options(tokens)
    out = list(tokens)
    for k, token in enumerate(tokens[:-1]):
        if (options[k] and token == option and options[k + 1]
                and not _is_option_word(tokens[k + 1])):
            value = os.path.expanduser(_unquote(tokens[k + 1]))
            out[k + 1] = shlex.quote(os.path.abspath(os.path.join(str(base), value)))
    return ' '.join(out)


def gmx_executable(params):
    """The GROMACS executable of a run, for tools other than grompp and mdrun.

    Parameters
    ----------
    params : aimmd.Params
        The run's params.

    Returns
    -------
    str
        The first token of ``params.gmx_grompp`` when its basename starts with ``gmx``, else ``aimmd._config.GROMACS``.

    Raises
    ------
    RuntimeError
        If neither names a GROMACS executable.

    Notes
    -----
    A first token that is not a path is resolved with ``shutil.which``; a
    name it does not find, and a wrapper script in ``gmx_grompp`` (``bash
    grompp.sh``), fall back to the executable AIMMD found at import. A bare
    name is never returned: spawned workers need not have GROMACS on their
    ``PATH``.
    """
    tokens = _tokens(str(getattr(params, 'gmx_grompp', '') or ''))
    first = _unquote(tokens[0]) if tokens else ''
    if first and os.path.basename(first).startswith('gmx'):
        if os.sep in first:
            return os.path.abspath(first)
        found = shutil.which(first)
        if found:
            return found
    from ... import _config
    found = getattr(_config, 'GROMACS', None)
    if found:
        return found
    raise RuntimeError(f'no GROMACS executable: the first token of gmx_grompp '
                       f'{getattr(params, "gmx_grompp", None)!r} is not one (or is not on PATH) '
                       'and aimmd._config.GROMACS is not set')


def split_grompp(gmx_grompp):
    """Remove the topology option from a grompp command.

    Parameters
    ----------
    gmx_grompp : str
        ``params.gmx_grompp``, e.g. ``'/path/gmx grompp -maxwarn 5 -p topol.top'``.

    Returns
    -------
    tuple
        ``(command without '-p X', X)``; X is None when the command has no ``-p``.

    Notes
    -----
    Only a ``-p`` after the GROMACS word counts (``srun -p queue gmx_mpi
    grompp -p topol.top`` keeps the launcher's partition).
    """
    topology = _option_value(gmx_grompp, '-p')
    return _strip_options(gmx_grompp, with_value=('-p',)), topology


def sanitize_mdrun(gmx_mdrun):
    """An mdrun command fit for the preparation runs of C-GAIMMD candidates.

    Parameters
    ----------
    gmx_mdrun : str
        ``params.gmx_mdrun``.

    Returns
    -------
    str
        The command without ``-plumed``, ``-cpi``, ``-cpt``, ``-maxh``, ``-deffnm`` (with their values) and ``-noappend``.

    Notes
    -----
    A PLUMED bias would bias the minimization, checkpoints are not wanted and
    the tools set ``-deffnm`` themselves. Options are removed only after the
    GROMACS word of the command (``gmx*``, or a name containing ``mdrun``),
    so a launcher prefix keeps its own; a command without such a word is
    returned unchanged. Options inside a wrapper script are invisible here:
    with a wrapper (``bash mdrun_gpu.sh``), give the tools an explicit
    ``mdrun`` command.
    """
    return _strip_options(gmx_mdrun, with_value=_MDRUN_DROP_WITH_VALUE, flags=_MDRUN_DROP_FLAGS)


def _grompp_command(grompp, *, maxwarn=None, index=True):
    """A grompp command without the options the tools set.

    ``-maxwarn`` goes too when the tool sets it, ``-n`` without `index`.
    """
    with_value = (_GROMPP_OWN_WITH_VALUE + (('-maxwarn',) if maxwarn is not None else ())
                  + (() if index else ('-n',)))
    return _strip_options(grompp, with_value=with_value, flags=_GROMPP_OWN_FLAGS)


def _mdrun_command(mdrun, *, minimization=False):
    """An mdrun command without the options the tools set (and EM-unsupported offloads)."""
    with_value = _MDRUN_OWN_WITH_VALUE + (_MDRUN_NOT_FOR_EM if minimization else ())
    return _strip_options(mdrun, with_value=with_value, flags=_MDRUN_OWN_FLAGS)


def _params_parent(params):
    parent = getattr(params, 'parent', None)
    return os.path.abspath(str(parent)) if parent is not None else os.getcwd()


def _params_grompp(params):
    """``split_grompp(params.gmx_grompp)[0]``, its index file absolute (relative to ``params.parent``)."""
    return _with_absolute_path(split_grompp(str(params.gmx_grompp))[0], '-n',
                               _params_parent(params))


def topology_file(params):
    """The GROMACS topology that grompp uses in a run.

    Parameters
    ----------
    params : aimmd.Params
        The run's params.

    Returns
    -------
    str
        The absolute ``-p`` file of ``params.gmx_grompp`` (relative to ``params.parent``), else ``<params.parent>/topol.top``.
    """
    _, topology = split_grompp(str(params.gmx_grompp))
    parent = _params_parent(params)
    if topology is None:
        return os.path.join(parent, 'topol.top')
    return topology if os.path.isabs(topology) else os.path.abspath(os.path.join(parent, topology))


def _default_em_mdp():
    """AIMMD's default EM template, ``aimmd._config.EM_MDP`` (``aimmd/engines/em.mdp``)."""
    from ... import _config
    return getattr(_config, 'EM_MDP', None) or str(
        Path(__file__).resolve().parents[2] / 'engines' / 'em.mdp')


def _gromacs_seed(seed):
    """A 32-bit GROMACS seed that is never 0 (``genion`` draws a random seed for 0)."""
    return int(seed) % (2 ** 31 - 1) + 1


def _work_dir(scratch, kind):
    """A new directory ``{scratch}/{kind}_{counter:04d}`` (the next free counter)."""
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    counter = 0
    while True:
        path = scratch / f'{kind}_{counter:04d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            counter += 1


# ---------------------------------------------------------------------------
# mdp files
# ---------------------------------------------------------------------------

def _mdp_key(key):
    return key.strip().lower().replace('-', '_')


def _mdp_value(value):
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (list, tuple)):
        return ' '.join(_mdp_value(v) for v in value)
    if isinstance(value, float):
        return f'{value:.10g}'
    return str(value)


def _read_mdp(template):
    """``{normalized key: value}`` of an mdp file (the last entry wins)."""
    out = {}
    for line in Path(template).read_text().splitlines():
        body = line.split(';', 1)[0]
        if '=' in body:
            key, value = body.split('=', 1)
            out[_mdp_key(key)] = value.strip()
    return out


def mdp_text(template, overrides, header=''):
    """An mdp derived from a template.

    Parameters
    ----------
    template : str or os.PathLike
        The template mdp file.
    overrides : dict
        Keys to set (``'-'``/``'_'`` and case do not matter); ``True``/``False`` become ``yes``/``no``, sequences are joined.
    header : str, optional
        Comment lines put first.

    Returns
    -------
    str
        The template's lines with the overridden keys replaced and the missing ones appended, each marked ``'; set here'``.

    Notes
    -----
    ``define`` and ``include`` overrides are appended to an existing value of
    the template (``define = -DPOSRES`` with an override ``-DAIMMD_CGAIMMD_POSRES``
    gives ``define = -DPOSRES -DAIMMD_CGAIMMD_POSRES``); the other keys replace
    it.
    """
    overrides = {_mdp_key(k): (k, _mdp_value(v)) for k, v in overrides.items()}
    used = set()
    out = [f'; {line}' for line in header.strip().splitlines()] if header.strip() else []
    out.append(f"; derived from {template}; entries marked 'set here' differ from it")
    for line in Path(template).read_text().splitlines():
        body = line.split(';', 1)[0]
        if '=' in body:
            key = _mdp_key(body.split('=', 1)[0])
            if key in overrides and key not in used:
                name, value = overrides[key]
                if key in ('define', 'include'):
                    existing = body.split('=', 1)[1].strip()
                    value = f'{existing} {value}'.strip()
                out.append(f'{name:<24s} = {value:<16s} ; set here (template: {line.strip()})')
                used.add(key)
                continue
            if key in overrides and key in used:
                out.append(f'; {line.strip()}   ; superseded (set here above)')
                continue
        out.append(line)
    for key, (name, value) in overrides.items():
        if key not in used:
            out.append(f'{name:<24s} = {value:<16s} ; set here (not in the template)')
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------------------------
# running GROMACS
# ---------------------------------------------------------------------------

def _command_name(command):
    """The GROMACS word and subcommand of a command, e.g. ``'gmx mdrun'``; else its first word."""
    tokens = _tokens(command) if command.strip() else []
    for k, token in enumerate(tokens):
        if _is_gromacs_word(token):
            name = os.path.basename(_unquote(token))
            following = tokens[k + 1] if k + 1 < len(tokens) else ''
            if following and following not in _SEPARATORS and not _is_option_word(following):
                name = f'{name} {_unquote(following)}'
            return name
    return os.path.basename(_unquote(tokens[0])) if tokens else command


def run_gmx(command, cwd, log, *, timeout=3600., must_stop=None, stdin=None):
    """Run one GROMACS command in a directory.

    Parameters
    ----------
    command : str
        The command (full executable path first).
    cwd : str or os.PathLike
        Working directory.
    log : str or os.PathLike
        Log file the output is appended to.
    timeout : float, optional
        Seconds before the command is terminated.
    must_stop : callable, optional
        ``must_stop() -> bool``, polled while the command runs; True terminates it.
    stdin : str, optional
        Text piped into the command (``genion``'s group choice).

    Returns
    -------
    str
        What the command wrote to `log` (after a successful command possibly without its last lines, see Notes).

    Raises
    ------
    RuntimeError
        On a non-zero exit (with the last 25 log lines), a stop by `must_stop`, or the timeout.

    Notes
    -----
    Runs ``cd <cwd> && [printf '%s' <stdin> |] <command>`` through
    ``aimmd.execute.utils.execute_command``, which streams the output into
    the log and terminates the process group on a stop. ``execute_command``
    returns 0 for a stopped or timed-out command, so both raise here: a
    stopped minimization is a failed candidate, never a success. The
    messages name the command by its GROMACS word and subcommand (``gmx
    mdrun stopped by must_stop``).

    ``execute_command`` stops reading the output when the process exits, so
    after a successful command the returned text and the log can miss the
    last lines (a pre-existing issue of ``execute_command``); the tools read
    their results from GROMACS' own files, never from this text.
    """
    from ...execute.utils import execute_command

    cwd = os.path.abspath(str(cwd))
    log = os.path.abspath(str(log))
    pipe = f"printf '%s' {shlex.quote(stdin)} | " if stdin is not None else ''
    full = f'cd {shlex.quote(cwd)} && {pipe}{command}'
    stop = must_stop or (lambda: False)
    state = {'stopped': False}

    def stop_condition():
        if stop():
            state['stopped'] = True
            return True
        return False

    offset = os.path.getsize(log) if os.path.exists(log) else 0
    failure = None
    start = time.monotonic()
    with open(log, 'a') as handle:
        try:
            execute_command(full, stop_condition=stop_condition, walltime=float(timeout),
                            log_file=handle, raise_if_failure=True)
        except RuntimeError as exc:
            failure = exc
    elapsed = time.monotonic() - start
    with open(log) as handle:
        handle.seek(offset)
        text = handle.read()
    tail = '\n'.join(text.strip().splitlines()[-_TAIL_LINES:])
    executable = _command_name(command)
    if failure is not None:
        raise RuntimeError(f'{executable} failed in {cwd} ({failure}); last log lines:\n{tail}')
    if state['stopped']:
        raise RuntimeError(f'{executable} stopped by must_stop in {cwd}')
    if elapsed >= float(timeout):
        raise RuntimeError(f'{executable} timed out after {float(timeout):g} s in {cwd}')
    return text


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def _box_line(dimensions):
    if dimensions is None:
        return f'{0.0:10.5f}{0.0:10.5f}{0.0:10.5f}'
    d = np.asarray(dimensions, dtype=float).reshape(-1)
    if d.size < 6 or np.allclose(d[3:6], 90.0):
        return ''.join(f'{v / 10.0:10.5f}' for v in d[:3])
    from MDAnalysis.lib.mdamath import triclinic_vectors
    v = triclinic_vectors(d) / 10.0
    values = [v[0, 0], v[1, 1], v[2, 2], v[0, 1], v[0, 2], v[1, 0], v[1, 2], v[2, 0], v[2, 1]]
    return ''.join(f'{x:10.5f}' for x in values)


def write_gro(path, atoms, positions_A, dimensions, *, decimals=3):
    """Write a ``.gro`` file.

    Parameters
    ----------
    path : str or os.PathLike
        Output file.
    atoms : MDAnalysis AtomGroup or object with ``names``, ``resnames`` and ``resids``
        Atom and residue names and residue numbers.
    positions_A : numpy.ndarray
        ``(N, 3)`` Å.
    dimensions : array_like or None
        ``(6,)`` box (Å, degrees); None writes a zero box.
    decimals : int, optional
        Decimals of the coordinates in nm (GROMACS reads any precision).
    """
    names, resnames, resids = (list(atoms.names), list(atoms.resnames), list(atoms.resids))
    x = np.asarray(positions_A, dtype=float) / 10.0
    if len(x) != len(names):
        raise ValueError(f'{len(x)} positions for {len(names)} atoms')
    width = int(decimals) + 5
    lines = ['written by aimmd.cgaimmd.preparation.gromacs.write_gro', f'{len(names):5d}']
    for n, (name, resname, resid, p) in enumerate(zip(names, resnames, resids, x), start=1):
        lines.append(f'{int(resid) % 100000:5d}{str(resname)[:5]:<5s}{str(name)[:5]:>5s}'
                     f'{n % 100000:5d}' + ''.join(f'{v:{width}.{int(decimals)}f}' for v in p))
    lines.append(_box_line(dimensions))
    Path(path).write_text('\n'.join(lines) + '\n')


def _gro_records(path):
    """(resnames, names, positions Å, dimensions) of a .gro of any precision."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[1].split()[0])
    body = lines[2:2 + n]
    if len(body) != n:
        raise ValueError(f'{path}: {len(body)} atom lines, header says {n}')
    width = 8
    if n:
        dots = [i for i, ch in enumerate(body[0]) if ch == '.' and i >= 20]
        if len(dots) >= 2:
            width = dots[1] - dots[0]
    resnames = [line[5:10].strip() for line in body]
    names = [line[10:15].strip() for line in body]
    x = np.array([[float(line[20 + k * width: 20 + (k + 1) * width]) for k in range(3)]
                  for line in body]).reshape(-1, 3) * 10.0
    box = [float(v) for v in lines[2 + n].split()]
    if len(box) >= 9 and any(abs(v) > 0 for v in box[3:9]):
        from MDAnalysis.lib.mdamath import triclinic_box
        v = np.array([[box[0], box[3], box[4]], [box[5], box[1], box[6]],
                      [box[7], box[8], box[2]]]) * 10.0
        dimensions = np.asarray(triclinic_box(v[0], v[1], v[2]), dtype=float)
    else:
        dimensions = np.array([box[0] * 10.0, box[1] * 10.0, box[2] * 10.0, 90.0, 90.0, 90.0])
    return resnames, names, x, dimensions


def read_gro(path):
    """Positions and box of a ``.gro`` file of any precision.

    Parameters
    ----------
    path : str or os.PathLike
        The file.

    Returns
    -------
    tuple
        ``(positions, dimensions)``: ``(N, 3)`` float64 Å and ``(6,)`` Å/degrees.
    """
    _, _, x, dimensions = _gro_records(path)
    return x, dimensions


def read_trr_last_frame(path):
    """The last frame with coordinates of a ``.trr`` file.

    Parameters
    ----------
    path : str or os.PathLike
        The trajectory (GROMACS writes the final energy-minimized frame to it).

    Returns
    -------
    tuple
        ``(positions, dimensions)``: ``(N, 3)`` float64 Å and ``(6,)`` Å/degrees.

    Raises
    ------
    OSError
        If the file is missing or holds no frame with coordinates.
    """
    from MDAnalysis.lib.formats.libmdaxdr import TRRFile
    from MDAnalysis.lib.mdamath import triclinic_box

    last = None
    if not os.path.exists(str(path)):
        raise FileNotFoundError(f'{path} does not exist')
    with TRRFile(str(path)) as handle:
        for frame in handle:
            if frame.hasx:
                last = (np.array(frame.x, dtype=float) * 10.0, np.array(frame.box, dtype=float))
    if last is None:
        raise OSError(f'{path} holds no frame with coordinates')
    x, box = last
    box = box * 10.0
    dimensions = np.asarray(triclinic_box(box[0], box[1], box[2]), dtype=float)
    return x, dimensions


_CONVERGED = re.compile(r'converged to Fmax\s*<\s*\S+\s+in\s+(\d+)\s+steps')
_MACHINE = re.compile(r'converged to machine precision in\s+(\d+)\s+steps')
_NOT_CONVERGED = re.compile(r'did not converge to Fmax\s*<\s*\S+\s+in\s+(\d+)\s+steps')
_EPOT = re.compile(r'^Potential Energy\s*=\s*(\S+)', re.M)
_FMAX = re.compile(r'^Maximum force\s*=\s*(\S+)', re.M)


def _first_potential(text):
    """The Potential of the first ``Energies`` block of an mdrun log."""
    lines = text.splitlines()
    for k, line in enumerate(lines):
        if line.strip().startswith('Energies ('):
            for header, values in zip(lines[k + 1::2], lines[k + 2::2]):
                if not header.strip():
                    return None
                names = [header[i:i + 15].strip() for i in range(0, len(header), 15)]
                if 'Potential' in names:
                    column = names.index('Potential')
                    try:
                        return float(values[15 * column: 15 * (column + 1)])
                    except ValueError:
                        return None
            return None
    return None


def parse_em_log(path):
    """The outcome of an energy minimization from its mdrun log.

    Parameters
    ----------
    path : str or os.PathLike
        The ``.log`` of the EM mdrun.

    Returns
    -------
    dict
        ``{'epot', 'fmax', 'steps', 'converged', 'epot_start'}`` (kJ/mol, kJ/mol/nm; None where the log has no value).

    Notes
    -----
    ``converged`` is True only for "converged to Fmax < ..."; convergence to
    machine precision without reaching Fmax and "did not converge" are False.
    ``epot_start`` is the potential energy of the first energies block (step
    0).
    """
    text = Path(path).read_text(errors='replace')
    out = {'epot': None, 'fmax': None, 'steps': None, 'converged': None, 'epot_start': None}
    for pattern, converged in ((_CONVERGED, True), (_MACHINE, False), (_NOT_CONVERGED, False)):
        match = pattern.search(text)
        if match:
            out['steps'], out['converged'] = int(match.group(1)), converged
            break
    for key, pattern in (('epot', _EPOT), ('fmax', _FMAX)):
        match = pattern.search(text)
        if match:
            try:
                out[key] = float(match.group(1))
            except ValueError:
                pass
    out['epot_start'] = _first_potential(text)
    return out
