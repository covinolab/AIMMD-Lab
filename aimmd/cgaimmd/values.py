"""
aimmd.cgaimmd.values
====================

The protocol values function: the default ``values_function`` of C-GAIMMD
networks.

AIMMD evaluates committor values through ``params.values_function``: in the
trainer's value passes, for bins, densities and reweighting, and for the
shooting-point values of the shooting workers. When a params file gives no
``values_function``, AIMMD builds one from the network
(:func:`aimmd.params.utils.create_default_values_function`), which fixes the
network's device when it is built, at params load, on the CPU. A C-GAIMMD
network does not stay there: its fit trains it on ``training_device`` and the
shooting worker generates on ``generation_device``. The protocol values
function therefore resolves the device at every call:

- inside a worker task (``aimmd._config.WORKER_TASK`` set by ``Worker.run``)
  the network is moved to ``device_for('values', params)``: ``values_device``
  in shooting and free workers, ``training_device`` in the trainer, whose
  value passes run where it trains;
- outside a worker task (loading params, interactive use) it stays where it
  is, and no ``torch.cuda`` function is called;
- inputs always go to the network's device at call time.

``Params`` picks it as the default values function under
``chain_type='cgaimmd'`` and, whatever the chain type, when
``aimmd.cgaimmd.protocol.is_cgaimmd_network(params.network)``, so
post-training a C-GAIMMD network on an RFPS or TPS run needs no
``values_function`` either. A user-defined ``values_function`` for a C-GAIMMD
network must likewise put its inputs on the network's device at call time.

>>> import numpy as np
>>> from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
>>> from aimmd.cgaimmd.values import protocol_values_function
>>> values_function = protocol_values_function(TinyFlowNetwork())
>>> values_function(np.zeros((2, 3), dtype=np.float32))
array([0., 0.])
"""

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import torch

from ..core.utils import accepts_system_id
from .devices import device_for
from .protocol import _batch_dict_on, _values_input, n_atoms_of, network_device


MIGRATE_HINT = ('the descriptors are not the flat Angstrom coordinates of the network\'s '
                'atoms: a descriptor cache written for another network (e.g. before '
                'continuing an RFPS run as chain_type=\'cgaimmd\'); give the C-GAIMMD params '
                'a descriptors_series of their own, or rename the old caches with '
                'aimmd.cgaimmd.restart.migrate_descriptors(params, run_dir), so that they are '
                'recomputed with params.descriptors_function')
"""Hint appended to the descriptor-width errors of C-GAIMMD code."""


def descriptor_width_error(width, n_atoms, where='descriptors'):
    """The ``ValueError`` of a descriptor width other than ``3 M``.

    Parameters
    ----------
    width : int
        The width found.
    n_atoms : int
        ``M``, the network's number of generated atoms.
    where : str, optional
        What had the wrong width, for the message.

    Returns
    -------
    ValueError
        With the :data:`MIGRATE_HINT`.
    """
    return ValueError(f'{where} have width {width}, but the C-GAIMMD network has {n_atoms} '
                      f'atoms and needs 3 x {n_atoms} = {3 * n_atoms}; {MIGRATE_HINT}')


def _is_data_list(value):
    """Whether `value` is a non-empty list of PyG ``Data`` (graph-mode input)."""
    return (isinstance(value, list) and len(value) > 0
            and callable(getattr(value[0], 'to_dict', None))
            and hasattr(value[0], 'num_nodes'))


class ProtocolValuesFunction:
    """The values function of a C-GAIMMD network: descriptors -> committor logits.

    Parameters
    ----------
    network : torch.nn.Module
        A C-GAIMMD network (``params.network``); its weights are read at every call.
    descriptor_transform : callable, optional
        ``params.descriptor_transform``; None is dense mode (the descriptors are the input).
    batch_size : int, optional
        Frames per network call.
    params : aimmd.Params, optional
        Whose ``cgaimmd_settings`` give the device inside a worker task; None: the defaults.

    Notes
    -----
    A call with ``(n, F)`` descriptors returns the ``(n,)`` float64 logits
    ``forward(input)[:, 0]``, where ``input`` is built as AIMMD's values
    function builds it: the flattened float32 descriptors (dense mode); the
    transform's batch dict (graph mode, e.g. the node-table featurizer's
    ``batch_dict``), its tensors moved to the network's device;
    ``Batch.from_data_list(data).to(device).to_dict()`` when
    `descriptor_transform` returns a list of PyG ``Data`` (graph mode); or the
    transform's flattened output otherwise. The network is put in eval mode
    and evaluated under ``torch.inference_mode``, in chunks of `batch_size`
    frames.

    At every call: ``d = device_for('values', params)``; when ``d`` is not
    None (inside a worker task) the network is moved there, else it stays
    where it is; the inputs go to ``network_device(network)``. A width ``F``
    other than ``3 * n_atoms_of(network)`` raises ``ValueError`` with a hint
    (a descriptor cache written for another network); ``M`` is measured once,
    at the first call. An input that is already a list of PyG ``Data`` is
    batched as is, and an input that is already a batch dict is evaluated as
    one batch (no width check for either). Empty input returns
    ``np.zeros(0)`` without calling the network.

    The object holds `params` (for its settings), so ``Params`` rebuilds its
    default values function for a copy of itself (see
    ``aimmd.params.Params.copy``).
    """

    def __init__(self, network, descriptor_transform=None, *, batch_size=4096, params=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError(f'batch_size must be an int >= 1, got {batch_size!r}')
        self.network = network
        """The C-GAIMMD network whose committor is evaluated."""
        self.descriptor_transform = descriptor_transform
        """``params.descriptor_transform`` at construction, None in dense mode."""
        self.batch_size = int(batch_size)
        """Frames per network call."""
        self.params = params
        """The params whose ``cgaimmd_settings`` give the device, or None for the defaults."""
        self._n_atoms = None
        self._transform_takes_system_id = (descriptor_transform is not None
                                           and accepts_system_id(descriptor_transform))

    def __repr__(self):
        return (f'{type(self).__name__}({type(self.network).__name__}, '
                f'descriptor_transform={self.descriptor_transform!r}, '
                f'batch_size={self.batch_size})')

    def _n_atoms_of(self):
        """``n_atoms_of(self.network)``, computed once (it does not change for a network)."""
        if self._n_atoms is None:
            self._n_atoms = n_atoms_of(self.network)
        return self._n_atoms

    def _transform(self, system_id):
        """The descriptor transform for `system_id` (bound to it when the transform takes one)."""
        if self._transform_takes_system_id and system_id is not None:
            return lambda chunk: self.descriptor_transform(chunk, system_id=system_id)
        return self.descriptor_transform

    def _logits(self, output, n):
        """``output[:, 0]`` as float64 numpy, after checking the ``(n, 1)`` shape."""
        network = self.network
        if output.ndim != 2 or output.shape != (n, 1):
            raise ValueError(f'{type(network).__name__}.forward must return (B, 1) '
                             f'committor logits, got shape {tuple(output.shape)} for '
                             f'B = {n}')
        return output[:, 0].detach().to('cpu', torch.float64).numpy()

    def __call__(self, input_data, system_id=None):
        """``(n, F)`` descriptors -> ``(n,)`` float64 logits (see the class notes)."""
        network = self.network
        batch = input_data if isinstance(input_data, Mapping) else None
        data = input_data if batch is None and _is_data_list(input_data) else None
        rows = None
        if batch is None and data is None:
            rows = np.asarray(input_data)
            n = len(rows) if rows.ndim else 0
            if n == 0:
                return np.zeros(0)
            rows = rows.reshape(n, -1)
            n_atoms = self._n_atoms_of()
            if rows.shape[1] != 3 * n_atoms:
                raise descriptor_width_error(rows.shape[1], n_atoms)
        device = device_for('values', self.params)
        if device is not None:
            network.to(device)
        network.eval()
        device = network_device(network)
        with torch.inference_mode():
            if batch is not None:                           # already a batch dict: one call
                output = network(_batch_dict_on(batch, device))
                ptr = batch.get('ptr')
                n = len(ptr) - 1 if ptr is not None else len(output)
                return self._logits(output, n)
            n = len(data) if data is not None else len(rows)
            transform = None if rows is None else self._transform(system_id)
            out = []
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                if rows is None:
                    from torch_geometric.data import Batch
                    inp = Batch.from_data_list(data[start:stop]).to(device).to_dict()
                else:
                    inp = _values_input(rows[start:stop], device, transform)
                out.append(self._logits(network(inp), stop - start))
        return np.concatenate(out)


def protocol_values_function(network, descriptor_transform=None, batch_size=4096, params=None):
    """The default values function of a C-GAIMMD network.

    Parameters
    ----------
    network : torch.nn.Module
        A C-GAIMMD network.
    descriptor_transform : callable, optional
        ``params.descriptor_transform`` (None: dense mode).
    batch_size : int, optional
        Frames per network call.
    params : aimmd.Params, optional
        For the device settings (``cgaimmd_settings``).

    Returns
    -------
    ProtocolValuesFunction
        Callable as ``values_function(descriptors, system_id=None)``.
    """
    return ProtocolValuesFunction(network, descriptor_transform, batch_size=batch_size,
                                  params=params)
