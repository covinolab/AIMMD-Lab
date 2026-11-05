"""
aimmd.cgaimmd.devices
=====================

Device roles of C-GAIMMD code.

C-GAIMMD code runs a network in three roles, each with its own setting in
``params.cgaimmd_parameters``:

``'generation'`` (``generation_device``)
    candidate generation in shooting workers;
``'values'`` (``values_device``)
    committor values in shooting and free workers (final checks, value
    passes); in the tasks that fit (``'train'``, ``'kinetics_convergence'``)
    the training device is used instead;
``'training'`` (``training_device``)
    the generator fit and the value passes of the tasks that fit.

Every setting is ``'auto'`` (the default), ``'cpu'``, ``'cuda'`` or
``'cuda:N'``. Devices are resolved *at use*, inside a worker task, after
AIMMD has bound the worker's resources (``CUDA_VISIBLE_DEVICES``), so
``'auto'`` means "the GPU this worker was bound to, if any". They are never
cached: one process may run several tasks in sequence.

Outside a worker task (``aimmd._config.WORKER_TASK is None``: loading a params
file, interactive use, most unit tests) :func:`device_for` returns None and
no ``torch.cuda`` function is called; callers then keep the network where it
is.
"""

import os

from .. import _config
from ..params.utils import CGAIMMD_PARAMETER_DEFAULTS, is_device_spec

ROLES = ('generation', 'values', 'training')
"""The device roles of C-GAIMMD code."""

# worker tasks that fit and evaluate in one process: their 'values' role uses training_device
_TRAINING_TASKS = ('train', 'kinetics_convergence')


def current_task():
    """The task of the running worker.

    Returns
    -------
    str or None
        ``'shoot'``, ``'free'``, ``'train'`` or ``'kinetics_convergence'`` inside ``Worker.run``, else None.

    Notes
    -----
    This is ``aimmd._config.WORKER_TASK``, read at call time
    (``aimmd.WORKER_TASK``, the copy made at ``import aimmd``, is not
    updated).
    """
    return getattr(_config, 'WORKER_TASK', None)


def resolve_device(spec):
    """Turn a device setting into a ``torch.device``.

    Parameters
    ----------
    spec : str
        ``'auto'``, ``'cpu'``, ``'cuda'`` or ``'cuda:N'`` (exactly; no normalization).

    Returns
    -------
    torch.device
        ``'auto'``: ``cuda:0`` if CUDA is available, else ``cpu``; ``'cuda'``: ``cuda:0``.

    Raises
    ------
    ValueError
        For any other value.
    RuntimeError
        For a CUDA setting without a visible CUDA device, or an index beyond the device count.
    """
    if not is_device_spec(spec):
        raise ValueError(f"device setting must be 'auto', 'cpu', 'cuda' or "
                         f"'cuda:N', got {spec!r}")
    import torch
    if spec == 'cpu':
        return torch.device('cpu')
    if spec == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda', 0)
        return torch.device('cpu')
    index = int(spec.split(':')[1]) if ':' in spec else 0
    if not torch.cuda.is_available():
        raise RuntimeError(f'device {spec!r} requested but no CUDA device is '
                           f'visible (CUDA_VISIBLE_DEVICES='
                           f'{os.environ.get("CUDA_VISIBLE_DEVICES")!r})')
    count = torch.cuda.device_count()
    if index >= count:
        raise RuntimeError(f'device cuda:{index} requested but only {count} '
                           f'CUDA device(s) are visible')
    return torch.device('cuda', index)


def _settings_of(params):
    settings = getattr(params, 'cgaimmd_settings', None)
    if settings is not None:
        return settings
    # a plain namespace (tests): the defaults, updated with its cgaimmd_parameters
    return {**CGAIMMD_PARAMETER_DEFAULTS,
            **(getattr(params, 'cgaimmd_parameters', None) or {})}


def device_for(role, params):
    """The device of `role` in the current worker task.

    Parameters
    ----------
    role : {'generation', 'values', 'training'}
        See :data:`ROLES`.
    params : aimmd.Params or namespace
        Read through ``params.cgaimmd_settings`` (see the notes for namespaces).

    Returns
    -------
    torch.device or None
        None outside a worker task; else :func:`resolve_device` of the role's setting.

    Raises
    ------
    ValueError
        For an unknown role.

    Notes
    -----
    Outside ``Worker.run`` (:func:`current_task` is None) no ``torch.cuda``
    function is called and the caller keeps the network's current device.
    The role ``'values'`` in the tasks that fit (``'train'`` and
    ``'kinetics_convergence'``) resolves ``training_device``: their value
    passes run where they train, so the network is not moved between devices
    every round. An object without ``cgaimmd_settings`` (a plain namespace in
    tests) gets :data:`~aimmd.params.utils.CGAIMMD_PARAMETER_DEFAULTS`
    updated with its ``cgaimmd_parameters`` dict, if any.
    """
    if role not in ROLES:
        raise ValueError(f'unknown device role {role!r}; expected one of {ROLES}')
    task = current_task()
    if task is None:
        return None
    if role == 'training' or (role == 'values' and task in _TRAINING_TASKS):
        key = 'training_device'
    else:
        key = f'{role}_device'
    settings = _settings_of(params)
    return resolve_device(settings.get(key, CGAIMMD_PARAMETER_DEFAULTS[key]))
