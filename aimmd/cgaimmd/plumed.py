"""
aimmd.cgaimmd.plumed
====================

The PLUMED bias of a C-GAIMMD run, as the shots see it: which input file
``gmx_mdrun`` passes with ``-plumed``, which files that input reads, and where
PLUMED writes the bias that AIMMD reads back.

A C-GAIMMD run under a static bias (a frozen OPES state, a wall) shoots with
the bias, exactly like an RFPS run: ``params.gmx_mdrun`` carries ``-plumed
<file>``, each shooting half writes its PLUMED output ``COLVAR`` in the chain
folder, AIMMD moves it next to the half (``<deffnm>_COLVAR``) and
``params.bias_function`` turns it into the per-frame bias (``bias_source =
'file'``). Candidate preparation removes ``-plumed`` from its mdrun commands
(:func:`aimmd.cgaimmd.preparation.gromacs.sanitize_mdrun`), so generated
structures are never prepared under the bias.

``mdrun`` runs inside each chain folder (``cd <chain folder> && gmx mdrun
...``), so a relative ``-plumed`` file, and every relative file the PLUMED
input reads, is looked for there. :func:`plumed_setup` and
:mod:`aimmd.cgaimmd.preflight` therefore ask for absolute paths. The sidecars
record the input file and the files it reads with their sha256
(:func:`aimmd.cgaimmd.sidecar.bias_settings`).

Standard library only.
"""

import hashlib
import os
import shlex


READ_KEYWORDS = ('STATE_RFILE', 'GRID_RFILE', 'REFERENCE', 'TEMPLATE')
"""Keywords whose value is a file PLUMED reads, in any action."""

READ_FILE_ACTIONS = ('INCLUDE', 'READ', 'PDB2CONSTANT')
"""Actions whose ``FILE=`` is a file PLUMED reads (every other ``FILE=`` is an output)."""

COLVAR = 'COLVAR'
"""The PLUMED output AIMMD reads the bias of a shooting half from (``PRINT ... FILE=COLVAR``)."""


def _sha256(fname):
    """The sha256 of a file, None if it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(fname, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def plumed_file_of(gmx_mdrun):
    """The ``-plumed`` file of an mdrun command, as written (None without one).

    Parameters
    ----------
    gmx_mdrun : str or None
        ``params.gmx_mdrun``.

    Returns
    -------
    str or None
        The word after ``-plumed`` (after the GROMACS word of the command, so a
        launcher prefix's options are not read), unquoted.
    """
    if not isinstance(gmx_mdrun, str) or '-plumed' not in gmx_mdrun:
        return None
    from .preparation.gromacs import _option_value
    return _option_value(gmx_mdrun, '-plumed')


def _logical_lines(text):
    """The PLUMED input as logical lines: comments removed, ``...`` continuation blocks joined."""
    lines, block = [], None
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        if block is not None:
            if line.startswith('...'):
                lines.append(' '.join(block))
                block = None
            else:
                block.append(line)
            continue
        if line.endswith('...'):
            block = [line[:-3].strip()]
            continue
        lines.append(line)
    if block is not None:
        lines.append(' '.join(block))
    return lines


def parse_actions(text):
    """The actions of a PLUMED input: ``(action, {KEYWORD: value}, flags)`` per logical line.

    Parameters
    ----------
    text : str
        The PLUMED input.

    Returns
    -------
    list of tuple
        The action name (upper case; a ``label:`` prefix is dropped), its
        ``KEYWORD=value`` pairs (keywords upper case, values as written) and
        its other words (flags such as ``NOPBC``).
    """
    actions = []
    for line in _logical_lines(text):
        try:
            words = shlex.split(line, posix=True)
        except ValueError:
            words = line.split()
        name, keywords, flags = None, {}, []
        for word in words:
            if name is None and word.endswith(':') and '=' not in word:
                continue                                     # label:
            if '=' in word:
                key, value = word.split('=', 1)
                keywords[key.upper()] = value
            elif name is None:
                name = word.upper()
            else:
                flags.append(word)
        if name is not None:
            actions.append((name, keywords, flags))
    return actions


def plumed_input_files(plumed_file, *, _seen=None):
    """The files a PLUMED input reads, followed into ``INCLUDE`` files.

    Parameters
    ----------
    plumed_file : str
        The PLUMED input (absolute, or relative to the current directory).

    Returns
    -------
    list of dict
        One per file read: ``keyword`` (e.g. ``'STATE_RFILE'``, ``'INCLUDE
        FILE'``), ``action``, ``written`` (the value as written), ``path``
        (absolute: relative values are taken relative to the folder mdrun
        runs in, which is unknown here, so they are resolved against the
        input's own folder and flagged), ``relative`` (bool), ``exists`` and
        ``sha256`` (None if it cannot be read).

    Notes
    -----
    Values with a PLUMED replica pattern or a ``$`` variable are listed as
    written and not checked (``exists`` None).
    """
    seen = set() if _seen is None else _seen
    plumed_file = os.path.abspath(plumed_file)
    if plumed_file in seen:
        return []
    seen.add(plumed_file)
    try:
        with open(plumed_file) as handle:
            text = handle.read()
    except OSError:
        return []
    folder = os.path.dirname(plumed_file)
    files = []
    for action, keywords, _flags in parse_actions(text):
        for key, value in keywords.items():
            reads = key in READ_KEYWORDS or (key == 'FILE' and action in READ_FILE_ACTIONS)
            if not reads:
                continue
            label = key if key != 'FILE' else f'{action} FILE'
            entry = {'keyword': label, 'action': action, 'written': value,
                     'relative': not os.path.isabs(os.path.expanduser(value)),
                     'path': None, 'exists': None, 'sha256': None}
            if '$' not in value and '@' not in value:
                path = os.path.abspath(os.path.join(folder, os.path.expanduser(value)))
                entry['path'] = path
                entry['exists'] = os.path.isfile(path)
                entry['sha256'] = _sha256(path) if entry['exists'] else None
            files.append(entry)
            if action == 'INCLUDE' and key == 'FILE' and entry['exists']:
                files.extend(plumed_input_files(entry['path'], _seen=seen))
    return files


def print_files(plumed_file):
    """The ``FILE=`` values of the ``PRINT`` actions of a PLUMED input (``'COLVAR'`` when a PRINT names none)."""
    try:
        with open(plumed_file) as handle:
            text = handle.read()
    except OSError:
        return []
    return [keywords.get('FILE', COLVAR) for action, keywords, _flags in parse_actions(text)
            if action == 'PRINT']


def plumed_setup(params):
    """The PLUMED bias of a run's shots: the input file and the files it reads.

    Parameters
    ----------
    params : aimmd.Params
        The run's params (``engine``, ``gmx_mdrun``, ``record_bias``, ``bias_source``).

    Returns
    -------
    dict or None
        None when the GROMACS ``gmx_mdrun`` has no ``-plumed``. Otherwise
        ``plumed_file`` (as written), ``path`` (absolute; a relative file is
        resolved against ``params.parent`` and flagged), ``relative``,
        ``exists``, ``sha256``, ``inputs`` (:func:`plumed_input_files`) and
        ``print_files`` (:func:`print_files`).
    """
    if getattr(params, 'engine', None) != 'gromacs':
        return None
    written = plumed_file_of(getattr(params, 'gmx_mdrun', None))
    if written is None:
        return None
    relative = not os.path.isabs(os.path.expanduser(written))
    base = getattr(params, 'parent', None) or os.getcwd()
    path = os.path.abspath(os.path.join(str(base), os.path.expanduser(written)))
    exists = os.path.isfile(path)
    return {'plumed_file': written, 'path': path, 'relative': relative, 'exists': exists,
            'sha256': _sha256(path) if exists else None,
            'inputs': plumed_input_files(path) if exists else [],
            'print_files': print_files(path) if exists else []}


def mdrun_executable(gmx_mdrun):
    """The words of an mdrun command up to and including its mdrun (``'gmx mdrun'``, ``'srun -n 1 gmx_mpi mdrun'``).

    Returns None when the command has no GROMACS word.
    """
    from .preparation.gromacs import _is_gromacs_word, _tokens, _unquote
    tokens = _tokens(str(gmx_mdrun or ''))
    for k, token in enumerate(tokens):
        if _is_gromacs_word(token):
            name = os.path.basename(_unquote(token))
            if 'mdrun' in name:
                return ' '.join(tokens[:k + 1])
            if k + 1 < len(tokens) and _unquote(tokens[k + 1]) == 'mdrun':
                return ' '.join(tokens[:k + 2])
            return None
    return None

