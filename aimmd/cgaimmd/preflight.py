"""
aimmd.cgaimmd.preflight
=======================

Check a C-GAIMMD params file before compute time is spent::

    python -m aimmd.cgaimmd.preflight params.py [--run RUN] [--candidates N]
                                                [--no-generate] [--device DEVICE]
                                                [--scratch DIR]

It runs what a job would run first, in this process and in a scratch folder,
and prints one plain line per check (``OK``, ``WARNING``, ``FAIL`` or
``SKIPPED``) and a verdict. The exit status is 0 when no check failed and 1
otherwise (2 for a usage error), so a job script can run it as its first step
and stop there.

The checks, in order:

1. **params**: the file loads (``Params.load(..., save=False)``: nothing is
   written next to it), and its ``chain_type`` is ``'cgaimmd'``.
2. **files**: the files the params name exist: the structure
   (``topology``) and the ``.mdp`` of a GROMACS run, the topology and index
   file of ``gmx_grompp`` (``-p``, ``-n``), and ``initial_network``, which
   must also load into the network (strictly, tensors only).
3. **generator**: the generator of ``cgaimmd_generator`` (or the network's
   default) is built and checked against the network, strictly, also for
   params read through old names; the network's atom count is that of the
   :class:`~aimmd.cgaimmd.preparation.descriptors.GeneratorAtoms`
   (:func:`aimmd.cgaimmd.source.build_checked_generator`,
   :func:`~aimmd.cgaimmd.source.check_generator_atoms`).
4. **numerics**: the descriptor width and the network's numbers on the first
   frames of the first initial path, with one tiny generation
   (:func:`aimmd.cgaimmd.source.check_start_up_numerics`).
5. **configuration**: the lambda window (with the trainer's bins of
   ``--run``, if given) and, without ``cgaimmd_prepare_function``, the
   placement in the box.
6. **candidate**: ``--candidates`` structures (default 1) generated at a
   target committor of 0, through the validity function, and the first valid
   one through :func:`aimmd.cgaimmd.source.prepare_candidate` (preparation,
   minimization, final check). An exception is a failure; a candidate that
   is rejected is a warning (an untrained generator may produce one), with
   the reason.
7. **gromacs** (GROMACS engine): the executables of ``gmx_grompp`` and
   ``gmx_mdrun`` exist and answer ``-version``.
8. **bias** (a ``-plumed`` in ``gmx_mdrun``, or ``record_bias``): the
   PLUMED input exists and is named by an absolute path, every file it reads
   (e.g. a frozen OPES state, ``STATE_RFILE``) exists and is absolute
   (``mdrun`` runs inside each chain folder, where relative paths do not
   lead), a file-mode bias is printed to ``COLVAR``, the mdrun executable
   knows ``-plumed``, ``record_bias`` and ``-plumed`` go together, and the
   **zero point**: the bias of frames in the reactive region (those of the
   initial paths and the prepared candidate) must be zero, as AIMMD's
   reweighting assumes (``|median| <=``
   :data:`aimmd.pathensemble.bias_utils.BIAS_ZERO_POINT_TOLERANCE` kT). A
   file-mode bias is evaluated as the shots evaluate it: the frame is made a
   ``.tpr`` with ``params.initialize_simulation`` and run for zero steps with
   the run's mdrun and ``-plumed`` (step 0 of a shooting half is its
   shooting point); PLUMED's ``COLVAR`` is then read with
   ``params.bias_function``. A reader-mode bias is called on the frames
   directly. Without a frame in the reactive region the zero point is not
   checked (a warning).

Relative files are resolved as in a worker: the checks run in the params
file's folder. Generation and the network run on ``--device`` (default: the
``generation_device`` setting, ``'auto'`` meaning a GPU if one is visible).
Nothing is written outside the scratch folder (a temporary folder unless
``--scratch`` is given; a temporary one is removed afterwards).
"""

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass

import numpy as np


OK, WARNING, FAIL, SKIPPED = 'OK', 'WARNING', 'FAIL', 'SKIPPED'
"""The outcomes of a check."""

TARGET_LAMBDA = 0.0
"""The target logit committor of the preflight candidates (the middle of the reactive region)."""

BIAS_FRAMES = 4
"""Frames in the reactive region of the initial paths whose bias is checked, at most."""

COMMAND_TIMEOUT = 120.0
"""Seconds a GROMACS command of the preflight may take."""


@dataclass
class Check:
    """One line of the preflight report."""

    name: str
    """What was checked (``'params'``, ``'files'``, ``'generator'``, ...)."""

    status: str
    """:data:`OK`, :data:`WARNING`, :data:`FAIL` or :data:`SKIPPED`."""

    detail: str = ''
    """What was found, in plain words."""


class PreflightReport:
    """The checks of one preflight run, in order.

    Attributes
    ----------
    params_file : str
        The params file checked.
    checks : list of Check
        One per check, in the order they ran.
    """

    def __init__(self, params_file):
        self.params_file = str(params_file)
        self.checks = []

    def add(self, name, status, detail=''):
        """Append a check and return it."""
        check = Check(name, status, str(detail))
        self.checks.append(check)
        return check

    def __getitem__(self, name):
        """The checks named `name` (a list: some checks run more than once)."""
        return [check for check in self.checks if check.name == name]

    @property
    def ok(self):
        """bool: no check failed."""
        return not any(check.status == FAIL for check in self.checks)

    def text(self):
        """The report as plain lines, with the verdict last."""
        lines = [f'C-GAIMMD preflight of {self.params_file}']
        for check in self.checks:
            lines.append(f'{check.status:<8} {check.name}: {check.detail}')
        failed = sum(check.status == FAIL for check in self.checks)
        warned = sum(check.status == WARNING for check in self.checks)
        if failed:
            lines.append(f'preflight FAILED: {failed} of {len(self.checks)} checks failed'
                         + (f', {warned} warnings' if warned else ''))
        else:
            lines.append(f'preflight OK: {len(self.checks)} checks'
                         + (f', {warned} warnings' if warned else ''))
        return '\n'.join(lines)


def _error(exc):
    """``'TypeName: message'`` of an exception, on one line where it is short."""
    text = str(exc).strip()
    return f'{type(exc).__name__}: {text}' if text else type(exc).__name__


@contextlib.contextmanager
def _in_folder(folder):
    """Run in `folder` (relative files resolve as in a worker), then go back."""
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(cwd)


# ════════════════════════════════════════════════════════════════════════════
# the checks
# ════════════════════════════════════════════════════════════════════════════

def _check_files(report, params):
    """The files the params name exist; initial_network loads into the network."""
    missing = []
    named = []
    if getattr(params, 'engine', None) == 'gromacs':
        from .preparation.gromacs import _option_value, split_grompp
        named += [('topology', getattr(params, 'topology', None)),
                  ('gmx_mdp', getattr(params, 'gmx_mdp', None))]
        grompp = str(getattr(params, 'gmx_grompp', '') or '')
        named += [('gmx_grompp -p', split_grompp(grompp)[1]),
                  ('gmx_grompp -n', _option_value(grompp, '-n'))]
    for field, fname in named:
        if fname is None:
            continue
        if not os.path.isfile(os.path.expanduser(str(fname))):
            missing.append(f'{field} {fname!r}')
    for path in getattr(params, 'initial_paths', None) or []:
        for fname in getattr(path, 'fnames', None) or []:
            if not os.path.isfile(str(fname)):
                missing.append(f'initial path {fname!r}')
    n_initial = len(getattr(params, 'initial_paths', None) or [])
    if missing:
        report.add('files', FAIL, 'missing: ' + ', '.join(missing))
    elif named:
        present = ', '.join(f'{field} {fname}' for field, fname in named if fname is not None)
        report.add('files', OK, f'{present} and the {n_initial} initial paths exist')
    else:
        report.add('files', OK, f'the {n_initial} initial paths exist')
    fname = params._initial_network_fname()
    if fname is None:
        report.add('initial_network', WARNING if params.chain_type == 'cgaimmd' else SKIPPED,
                   'not set: the run starts from the network the params file builds, or '
                   'from an existing network<states>.h5')
        return
    from ..worker.utils import load_network_weights
    try:
        load_network_weights(params.network, fname)
    except Exception as exc:                                  # noqa: BLE001
        report.add('initial_network', FAIL, f'{fname} does not load into params.network: '
                                            f'{_error(exc)}')
    else:
        report.add('initial_network', OK, f'{fname} loads into {type(params.network).__name__}')


def _check_generator(report, params, device):
    """Build and check the generator; ``(generator, n_atoms)`` or ``(None, None)``."""
    from .protocol import n_atoms_of
    from .source import build_checked_generator, check_generator_atoms
    try:
        params.network.to(device)
        generator = build_checked_generator(params)
        n_atoms = int(n_atoms_of(params.network))
        check_generator_atoms(params, n_atoms)
    except Exception as exc:                                  # noqa: BLE001
        report.add('generator', FAIL, _error(exc))
        return None, None
    report.add('generator', OK, f'{generator.method!r} {dict(generator.settings())} for '
                                f'{type(params.network).__name__} with {n_atoms} atoms on '
                                f'{device}')
    return generator, n_atoms


def _check_numerics(report, params, generator, n_atoms):
    from .source import check_start_up_numerics
    try:
        params.network.eval()
        numerics = check_start_up_numerics(params, generator, n_atoms,
                                           list(params.initial_paths or []))
    except Exception as exc:                                  # noqa: BLE001
        report.add('numerics', FAIL, _error(exc))
        return False
    report.add('numerics', OK, f'descriptors {3 * n_atoms} wide, finite values and one tiny '
                               f'generation ({_short(numerics)})')
    return True


def _short(record, limit=160):
    text = str(record)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _run_bins(run_dir, params):
    if run_dir is None:
        return None
    fname = os.path.join(str(run_dir), f'bins{params.sorted_states}.npy')
    try:
        return np.load(fname)
    except (OSError, ValueError):
        return None


def _check_configuration(report, params, n_atoms, run_dir):
    from .lambda_law import resolve_window
    from .protocol import lambda_clip_of
    from .source import place_in_box
    settings = params.cgaimmd_settings
    try:
        L = lambda_clip_of(params.network, settings)
        (lo, hi), _record = resolve_window(
            settings, L, _run_bins(run_dir, params),
            reweight_parameters=getattr(params, 'reweight_parameters', None))
        if getattr(params, 'cgaimmd_prepare_function', None) is None:
            place_in_box(np.zeros((n_atoms, 3)), params, settings['placement'])
    except Exception as exc:                                  # noqa: BLE001
        report.add('configuration', FAIL, _error(exc))
        return None
    report.add('configuration', OK, f'lambda window ({lo:.3g}, {hi:.3g}), acceptance '
                                    f'{settings["acceptance"]!r} after '
                                    f'{settings["window_after_trained_shots"]} trained shots')
    return (lo, hi)


def _check_candidate(report, params, generator, n_atoms, n_candidates, scratch, device):
    """Generate, check validity and prepare a candidate; its prepared frame, or None."""
    import torch
    from .source import candidate_context, prepare_candidate
    from .types import ValidityResult
    folder = os.path.join(scratch, 'candidate')
    os.makedirs(folder, exist_ok=True)
    seed = 0
    context = candidate_context(params, folder, path_name='preflight', seed=seed)
    try:
        started = time.perf_counter()
        torch_generator = torch.Generator(device=device)
        torch_generator.manual_seed(seed)
        lam = torch.full((int(n_candidates),), TARGET_LAMBDA, dtype=torch.float32,
                         device=device)
        with torch.no_grad():
            samples = generator.generate(params.network, lam, generator=torch_generator)
        samples = torch.as_tensor(samples).detach().to('cpu', torch.float64).numpy()
        if samples.shape != (int(n_candidates), n_atoms, 3):
            raise ValueError(f'the generator returned shape {samples.shape}, expected '
                             f'({int(n_candidates)}, {n_atoms}, 3)')
        seconds = time.perf_counter() - started
        validity = getattr(params, 'cgaimmd_validity_function', None)
        ok = np.isfinite(samples).all(axis=(1, 2))
        reasons = np.array(['' if keep else 'not_finite' for keep in ok], dtype=object)
        if validity is not None:
            result = validity(samples.copy(), context)
            if not isinstance(result, ValidityResult):
                raise TypeError(f'cgaimmd_validity_function must return a ValidityResult, '
                                f'got {type(result).__name__}')
            samples = np.asarray(result.positions, dtype=np.float64)
            ok = ok & np.asarray(result.ok, dtype=bool).reshape(-1)
            reasons = np.asarray(result.reason, dtype=object).reshape(-1)
    except Exception as exc:                                  # noqa: BLE001
        report.add('candidate', FAIL, f'generation or validity: {_error(exc)}')
        return None
    if not ok.any():
        shown = '; '.join(sorted({str(reason) for reason in reasons if reason}))[:300]
        report.add('candidate', WARNING, f'none of {len(ok)} generated candidates passed the '
                                         f'validity function ({shown}); preparation not tried')
        return None
    k = int(np.flatnonzero(ok)[0])
    try:
        prepared = prepare_candidate(samples[k], TARGET_LAMBDA, context)
    except Exception as exc:                                  # noqa: BLE001
        report.add('candidate', FAIL, f'preparation of a valid candidate: {_error(exc)}')
        return None
    value = None if prepared.check is None else prepared.check.get('lambda')
    where = (f'{int(ok.sum())} of {len(ok)} valid, generated in {seconds:.1f} s; '
             f'prepared in {sum(prepared.seconds.values()):.1f} s')
    if prepared.ok:
        report.add('candidate', OK, f'{where}; final check passed (state '
                                    f'{prepared.check["state"]}, lambda {value:.3g} for a '
                                    f'target of {TARGET_LAMBDA:g})')
        return prepared.frame
    report.add('candidate', WARNING, f'{where}; rejected at preparation ({prepared.reason}; '
                                     f'lambda {value})')
    if prepared.check is not None and prepared.check.get('state') == params.states[1]:
        return prepared.frame                                 # still a frame in R
    return None


def _executable_words(command):
    """The GROMACS part of a command: from its GROMACS word on, e.g. ``'gmx mdrun'``."""
    from .preparation.gromacs import _is_gromacs_word, _tokens, _unquote
    tokens = _tokens(str(command or ''))
    for k, token in enumerate(tokens):
        if _is_gromacs_word(token):
            return [_unquote(word) for word in tokens[k:k + 1]]
    return []


def _gromacs_version(word):
    """``(found, text)`` of ``<word> -version`` (the first lines naming the version)."""
    path = word if os.sep in word else shutil.which(word)
    if path is None or not os.path.isfile(path):
        return False, f'{word!r} not found (PATH of this process)'
    try:
        done = subprocess.run([path, '-version'], capture_output=True, text=True,
                              timeout=COMMAND_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f'{path} -version: {_error(exc)}'
    text = done.stdout + done.stderr
    version = [line.strip() for line in text.splitlines() if 'GROMACS version' in line]
    if done.returncode or not version:
        return False, f'{path} -version exited with {done.returncode}'
    return True, f'{path}: {version[0]}'


def _check_gromacs(report, params):
    problems, found = [], []
    for field in ('gmx_grompp', 'gmx_mdrun'):
        words = _executable_words(getattr(params, field, None))
        if not words:
            problems.append(f'{field} names no GROMACS executable (gmx*, *mdrun*): '
                            f'{getattr(params, field, None)!r}')
            continue
        ok, text = _gromacs_version(words[0])
        (found if ok else problems).append(f'{field}: {text}')
    if problems:
        report.add('gromacs', FAIL, '; '.join(problems + found))
    else:
        report.add('gromacs', OK, '; '.join(found))


def _initial_frames_in_r(params, limit=BIAS_FRAMES):
    """Up to `limit` one-frame paths of the initial paths in the reactive region."""
    r = params.states[1]
    frames = []
    for path in getattr(params, 'initial_paths', None) or []:
        try:
            states = np.asarray(path.compute(params.states_function, '', 'reader',
                                             return_result=True)).astype(str)
        except Exception:                                     # noqa: BLE001
            continue
        for i in np.flatnonzero(states == r):
            frames.append(path[int(i):int(i) + 1])
            if len(frames) >= limit:
                return frames
    return frames


def _plumed_executable(params):
    """``(ok, detail)``: the mdrun of ``gmx_mdrun`` knows ``-plumed``."""
    from .plumed import mdrun_executable
    command = mdrun_executable(getattr(params, 'gmx_mdrun', None))
    if command is None:
        return False, 'gmx_mdrun names no mdrun'
    words = _executable_words(command)
    from .preparation.gromacs import _tokens, _unquote
    tokens = [_unquote(token) for token in _tokens(command)]
    start = tokens.index(words[0]) if words and words[0] in tokens else 0
    argv = tokens[start:] + ['-h']
    path = argv[0] if os.sep in argv[0] else shutil.which(argv[0])
    if path is None:
        return False, f'{argv[0]!r} not found'
    try:
        done = subprocess.run([path] + argv[1:], capture_output=True, text=True,
                              timeout=COMMAND_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f'{" ".join(argv)}: {_error(exc)}'
    if '-plumed' in done.stdout + done.stderr:
        return True, f'{" ".join(argv[:-1])} knows -plumed'
    return False, (f'{" ".join(argv[:-1])} has no -plumed option: this GROMACS is not '
                   f'patched with PLUMED (load the PLUMED build, e.g. the module with '
                   f'"plumed" in its name)')


def _frame_reader(frame):
    """A one-frame reader of a prepared frame or a one-frame path, as ``Path.compute`` passes it."""
    from .source import _reader
    if hasattr(frame, 'positions') and not hasattr(frame, 'fnames'):
        dims = None if frame.dimensions is None else np.asarray(frame.dimensions, np.float32)
        return _reader(np.asarray(frame.positions, dtype=np.float32), dims)
    return None


def _bias_by_reader(params, frames):
    values = []
    for frame in frames:
        reader = _frame_reader(frame)
        if reader is not None:
            result = params.bias_function(reader)
        else:
            result = frame.compute(params.bias_function, '', 'reader', return_result=True)
        values.extend(np.asarray(result, dtype=float).reshape(-1).tolist())
    return values


def _timestep(frame):
    """A Timestep-like of a prepared frame (positions, dimensions) or a one-frame path."""
    if hasattr(frame, 'fnames'):
        return frame
    from types import SimpleNamespace
    return SimpleNamespace(positions=np.asarray(frame.positions, dtype=np.float32),
                           dimensions=(None if frame.dimensions is None
                                       else np.asarray(frame.dimensions, dtype=np.float32)))


def _bias_by_step_zero(params, frames, plumed_path, scratch):
    """The bias of each frame as a shot sees it: step 0 of an mdrun with ``-plumed``, then ``bias_function``.

    Each frame becomes a ``.tpr`` through ``params.initialize_simulation``
    (the shots' own grompp call) and runs ``-nsteps 0`` with the run's mdrun
    and ``-plumed``, so PLUMED evaluates the bias at the frame exactly as at
    the first frame of a shooting half (``mdrun -rerun`` is not used: the
    PLUMED patch of GROMACS 2024 does not initialize PLUMED in a rerun).
    PLUMED's ``COLVAR`` is moved to ``<deffnm>_COLVAR`` and read with
    ``params.bias_function(<deffnm>.xtc)``, as AIMMD reads a half.
    """
    from ..execute.utils import execute_command
    from .plumed import COLVAR
    from .preparation.gromacs import _mdrun_command, sanitize_mdrun
    # the run's mdrun with its performance options, without -plumed, checkpoints
    # and the options set here
    mdrun = _mdrun_command(sanitize_mdrun(str(params.gmx_mdrun)))
    folder = os.path.join(scratch, 'bias')
    os.makedirs(folder, exist_ok=True)
    values = []
    for k, frame in enumerate(frames):
        name = f'frame{k}'
        deffnm = os.path.join(folder, name)
        params.initialize_simulation(_timestep(frame), deffnm, timeout=COMMAND_TIMEOUT,
                                     verbose=False)
        colvar = os.path.join(folder, COLVAR)
        if os.path.exists(colvar):
            os.remove(colvar)
        log = f'{deffnm}.mdrun.out'
        command = (f'cd {folder} && {mdrun} -s {name}.tpr -deffnm {name} -nsteps 0 '
                   f'-plumed {plumed_path} -nobackup')
        with open(log, 'w') as handle:
            execute_command(command, walltime=COMMAND_TIMEOUT, log_file=handle,
                            raise_if_failure=True)
        if not os.path.exists(colvar):
            raise RuntimeError(f'mdrun -plumed wrote no {COLVAR} in {folder} (see {log})')
        os.replace(colvar, f'{deffnm}_COLVAR')
        trajectory = f'{deffnm}{params.trajectory_extension}'
        result = np.asarray(params.bias_function(trajectory), dtype=float).reshape(-1)
        if not len(result):
            raise RuntimeError(f'bias_function({trajectory!r}) returned no value')
        values.append(float(result[0]))
    return values


def _check_bias(report, params, frames, scratch):
    """The PLUMED input, the files it reads, the executable, and the zero point."""
    from .plumed import COLVAR, plumed_setup
    record_bias = bool(getattr(params, 'record_bias', False))
    source = getattr(params, 'bias_source', 'reader')
    setup = plumed_setup(params)
    if setup is None and not record_bias:
        report.add('bias', SKIPPED, 'no bias: record_bias is False and gmx_mdrun has no -plumed')
        return
    file_mode = record_bias and source == 'file'
    if setup is None:
        if file_mode:
            report.add('plumed', FAIL, "record_bias with bias_source='file' reads the PLUMED "
                                       "output of every shot, but gmx_mdrun has no -plumed")
            return
    else:
        problems = []
        if not setup['exists']:
            problems.append(f"the -plumed file {setup['plumed_file']!r} does not exist "
                            f"(looked for {setup['path']})")
        if setup['relative']:
            problems.append(f"the -plumed file {setup['plumed_file']!r} is relative: mdrun runs "
                            f"in each chain folder, give an absolute path")
        for entry in setup['inputs']:
            if entry['relative']:
                problems.append(f"{entry['keyword']}={entry['written']} is relative: PLUMED runs "
                                f"in each chain folder, give an absolute path")
            elif entry['exists'] is False:
                problems.append(f"{entry['keyword']}={entry['written']} does not exist")
        if file_mode and setup['exists'] and COLVAR not in setup['print_files']:
            problems.append(f"no PRINT ... FILE={COLVAR}: AIMMD reads the bias of every shot "
                            f"from the {COLVAR} PLUMED writes (found PRINT files "
                            f"{setup['print_files']})")
        if problems:
            report.add('plumed', FAIL, '; '.join(problems))
        else:
            inputs = ', '.join(f"{entry['keyword']}={entry['path']}" for entry in setup['inputs'])
            report.add('plumed', OK, f"{setup['path']} (sha256 {setup['sha256'][:12]})"
                                     + (f', reads {inputs}' if inputs else ''))
        ok, detail = _plumed_executable(params)
        report.add('plumed executable', OK if ok else FAIL, detail)
        if not record_bias:
            report.add('bias', WARNING, 'gmx_mdrun biases the shots with -plumed, but '
                                        'record_bias is False: the trainer reweights as if '
                                        'the shots were unbiased')
            return
        if problems or not ok:
            report.add('bias zero point', SKIPPED, 'not checked: the PLUMED setup failed')
            return
    if getattr(params, 'bias_function', None) is None:
        report.add('bias zero point', FAIL, 'record_bias is True but bias_function is None')
        return
    if not frames:
        report.add('bias zero point', WARNING,
                   f'not checked: no frame in the reactive region {params.states[1]!r} (the '
                   f'initial paths have none and no candidate was prepared)')
        return
    try:
        if file_mode:
            values = _bias_by_step_zero(params, frames, setup['path'], scratch)
        else:
            values = _bias_by_reader(params, frames)
    except Exception as exc:                                  # noqa: BLE001
        report.add('bias zero point', FAIL, f'the bias of frames in the reactive region could '
                                            f'not be evaluated: {_error(exc)}')
        return
    from ..pathensemble.bias_utils import BIAS_ZERO_POINT_TOLERANCE
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    shown = ', '.join(f'{value:.3g}' for value in values)
    if not np.isfinite(values).all() or abs(median) > BIAS_ZERO_POINT_TOLERANCE:
        report.add('bias zero point', FAIL,
                   f'the bias in the reactive region is not zero: {shown} kT on {len(values)} '
                   f'frame(s) (median {median:.3g}, tolerance {BIAS_ZERO_POINT_TOLERANCE} kT); '
                   f'AIMMD reweights with exp(bias) and needs the bias to be 0 where it is '
                   f'not applied: shift it in bias_function (e.g. by the OPES BARRIER)')
        return
    threshold = getattr(params, 'bias_reactive_threshold', None)
    if isinstance(threshold, (int, float)) and np.max(np.abs(values)) > threshold:
        report.add('bias zero point', WARNING, f'median {median:.3g} kT, but a frame has '
                                               f'|bias| above bias_reactive_threshold '
                                               f'{threshold}: {shown}')
        return
    report.add('bias zero point', OK, f'{shown} kT on {len(values)} frame(s) in the reactive '
                                      f'region ({"step 0 of mdrun -plumed" if file_mode else "bias_function"})')


# ════════════════════════════════════════════════════════════════════════════
# entry points
# ════════════════════════════════════════════════════════════════════════════

def run_preflight(params_file, *, run_dir=None, candidates=1, generate=True, device=None,
                  scratch=None):
    """Run the preflight checks of a params file.

    Parameters
    ----------
    params_file : str
        The params file.
    run_dir : str, optional
        The run folder, for the trainer's bins of a ``'bins'`` window.
    candidates : int, optional
        Structures generated for the candidate check.
    generate : bool, optional
        False skips the candidate check (no generation beyond the numerics' tiny one).
    device : str, optional
        The device of the network and the generation; default the
        ``generation_device`` setting (``'auto'``: a GPU if one is visible).
    scratch : str, optional
        Folder for the files of the checks (kept); default a temporary folder (removed).

    Returns
    -------
    PreflightReport
    """
    import aimmd
    from .devices import resolve_device
    report = PreflightReport(params_file)
    try:
        params = aimmd.Params.load(str(params_file), save=False)
    except Exception as exc:                                  # noqa: BLE001
        report.add('params', FAIL, f'{params_file} does not load: {_error(exc)}')
        return report
    cgaimmd = getattr(params, 'chain_type', None) == 'cgaimmd'
    report.add('params', OK if cgaimmd else FAIL,
               f"loaded (chain_type {params.chain_type!r}, engine {params.engine!r})"
               + ('' if cgaimmd else "; not a C-GAIMMD params file: only its files, GROMACS "
                                     "and the bias are checked"))
    own_scratch = scratch is None
    scratch = tempfile.mkdtemp(prefix='cgaimmd_preflight_') if own_scratch else str(scratch)
    os.makedirs(scratch, exist_ok=True)
    scratch = os.path.abspath(scratch)
    try:
        with _in_folder(str(params.parent)):
            _check_files(report, params)
            frames = _initial_frames_in_r(params)
            if cgaimmd:
                spec = device or params.cgaimmd_settings['generation_device']
                try:
                    torch_device = resolve_device(spec)
                except Exception as exc:                      # noqa: BLE001
                    report.add('device', FAIL, _error(exc))
                    torch_device = None
                generator = n_atoms = None
                if torch_device is not None:
                    generator, n_atoms = _check_generator(report, params, torch_device)
                if generator is not None and _check_numerics(report, params, generator,
                                                             n_atoms):
                    window = _check_configuration(report, params, n_atoms, run_dir)
                    if not generate:
                        report.add('candidate', SKIPPED, 'not asked for (--no-generate)')
                    elif window is not None:
                        frame = _check_candidate(report, params, generator, n_atoms,
                                                 candidates, scratch, torch_device)
                        if frame is not None:
                            frames = frames + [frame]
            if getattr(params, 'engine', None) == 'gromacs':
                _check_gromacs(report, params)
            _check_bias(report, params, frames, scratch)
    except Exception as exc:                                  # noqa: BLE001
        report.add('preflight', FAIL, f'unexpected error: {_error(exc)}\n'
                                      + traceback.format_exc())
    finally:
        if own_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    return report


def main(argv=None):
    """``python -m aimmd.cgaimmd.preflight params.py ...``: print the report; 0 if no check failed."""
    parser = argparse.ArgumentParser(
        prog='python -m aimmd.cgaimmd.preflight',
        description='Check a C-GAIMMD params file before a job: the params, the files, the '
                    'generator and the network, one prepared candidate, GROMACS and the PLUMED '
                    'bias. Exit status 0 when no check failed, 1 otherwise.')
    parser.add_argument('params', help='the params file')
    parser.add_argument('--run', default=None, help='the run folder (for its bins)')
    parser.add_argument('--candidates', type=int, default=1,
                        help='structures generated for the candidate check (default 1)')
    parser.add_argument('--no-generate', action='store_true',
                        help='skip the candidate check')
    parser.add_argument('--device', default=None,
                        help="device of the network and the generation (default: the "
                             "generation_device setting)")
    parser.add_argument('--scratch', default=None,
                        help='folder for the files of the checks (kept); default a temporary '
                             'folder')
    args = parser.parse_args(argv)
    if args.candidates < 1:
        parser.error('--candidates must be at least 1')
    report = run_preflight(args.params, run_dir=args.run, candidates=args.candidates,
                           generate=not args.no_generate, device=args.device,
                           scratch=args.scratch)
    print(report.text(), flush=True)
    return 0 if report.ok else 1


if __name__ == '__main__':
    sys.exit(main())
