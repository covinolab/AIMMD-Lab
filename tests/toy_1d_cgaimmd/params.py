"""Params of the toy 1-D C-GAIMMD chain test (tests/test_toy_1d_cgaimmd.py).

The toy engine and state definition of ``tests/toy_1d`` (one atom, ``x`` in
``[0, 10)``; A ``x < 1`` or ``x > 9``, R ``1 <= x < 2``, B ``2 <= x < 3``)
with ``chain_type = 'cgaimmd'`` and the minimal flow-matching network
``aimmd.cgaimmd.flow_matching.toy.TinyFlowNetwork``: a one-atom MLP without
``committor``, ``to_flow``/``from_flow`` or any center-of-mass handling, so
that the run goes through every fallback of the network protocol. The
generator is picked by name: the flow-matching ODE (``'ode'``), with the
settings given here.

The test copies this file into a temporary folder and writes the files it
names there: ``initial_A.xtc`` and ``initial_B.xtc`` (end-state frames only,
one file per basin: a C-GAIMMD run needs no initial transition) and
``initial_network.h5`` (the state dict of a ``Network()``).

With the zero-initialized committor output of the untrained network every
value is 0, so ``|lambda - lambda*| <= 3`` holds on the whole window
``(-3, 3)`` and acceptance cannot stall; the rule switches from the
bootstrap rule to the target rule after 5 trained shots.

Two cheap hooks put candidate filtering and minimization into the chain:
``cgaimmd_validity_function`` rejects every sample whose ``x`` lies outside
``[1.1, 1.9]`` (reason code ``toy_window``, per-sample info ``{'x': x}``),
and ``cgaimmd_minimize_function`` moves ``y`` and ``z`` of the atom onto its
``x`` (the toy's own ``x == y == z``), which changes neither the state nor the
committor (both read ``x`` only), and records the frame before and after.
The test checks that the shot starts from the frame this hook returned.
"""

import numpy as np

from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.preparation.descriptors import coordinates_descriptors
from aimmd.cgaimmd.types import PreparedFrame, ValidityResult

engine = 'toy'
chain_type = 'cgaimmd'
states = 'ARB'
initial_paths = ['initial_A.xtc', 'initial_B.xtc']
initial_network = 'initial_network.h5'
round_shooting_index = True
gen_temperature = 300.0
masses_from_topology = True    # every C-GAIMMD params file sets it; a no-op on the toy engine
cgaimmd_generator = {'method': 'ode', 'sampler': 'midpoint', 'n_steps': 8, 'chunk': 16}
cgaimmd_parameters = {
    'placement': 'none', 'lambda_clip': 10.0, 'window_after_trained_shots': 5,
    'lambda_window': (-3.0, 3.0), 'lambda_tolerance': 3.0, 'n_targets': 8,
    'max_refills': 3, 'wait_seconds': 1.0, 'generation_seed': 0,
    'generation_device': 'cpu', 'values_device': 'cpu', 'training_device': 'cpu'}


def toy_mdrun(ts):
    """The dynamics of tests/toy_1d: 100 random steps of 0.02 per frame, x == y == z."""
    for _ in range(100):
        ts.positions = (ts.positions + .02 * np.random.normal()) % 10


def states_function(trajectory):
    """The states of tests/toy_1d from the x of the one atom."""
    result = []
    for frame in trajectory:
        x = frame.positions[0, 0]
        if x < 1 or x > 9:
            result.append('A')
        elif x < 2:
            result.append('R')
        elif x < 3:
            result.append('B')
        elif x < 4:
            result.append('S')
        elif x < 5:
            result.append('C')
        elif x < 6:
            result.append('T')
        elif x < 7:
            result.append('D')
        elif x < 8:
            result.append('U')
        else:
            result.append('E')
    return np.array(result, dtype='<U1')


def descriptors_function(trajectory):
    """The flat Angstrom coordinates of the one generator atom."""
    return coordinates_descriptors(trajectory)


class Network(TinyFlowNetwork):
    """Local subclass: Params.save writes 'from <this module> import network'."""


network = Network()

TOY_WINDOW = (1.1, 1.9)                   # x range the validity hook accepts (R is [1, 2))


def cgaimmd_validity_function(positions, context):
    """Reject the samples whose x lies outside TOY_WINDOW; nothing to relabel for one atom."""
    x = positions[:, 0, 0]
    ok = (x >= TOY_WINDOW[0]) & (x <= TOY_WINDOW[1])
    reason = np.array(['' if keep else f'toy_window: x = {xi:.3f} outside {TOY_WINDOW}'
                       for keep, xi in zip(ok, x)], dtype=object)
    return ValidityResult(positions=positions, ok=ok, reason=reason,
                          info=[{'x': float(xi)} for xi in x])


def cgaimmd_minimize_function(frame, context):
    """Move y and z onto x (the toy's x == y == z); the record holds the frame before and after."""
    positions = np.array(frame.positions, dtype=np.float64)
    before = positions.copy()
    positions[:, 1] = positions[:, 0]
    positions[:, 2] = positions[:, 0]
    record = {'before': before.tolist(), 'after': positions.tolist(),
              'max_displacement_A': float(np.abs(positions - before).max())}
    return PreparedFrame(positions=positions, dimensions=frame.dimensions, record=record)
