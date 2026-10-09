#%%
# AIMMD params template: a graph-network committor (e.g. PaiNN) for ligand unbinding.
#
# Copy this file next to the structure file, the initial path and the network module
# (painn.py), and fill in the settings below. See the documentation page "Graph Networks
# and Node Tables" (docs/source/graph_networks.rst).
#
# GRAPH INPUT. The network sees one graph per frame: the ligand heavy atoms plus the heavy
# atoms within 8 A of them, wrapped around the ligand, with edges between nodes closer than
# CUTOFF. GRAPH_INPUT chooses how AIMMD stores what it needs to build these graphs:
#
#   'nodetables'  (recommended) the descriptor series holds the nodes of every frame: up
#                 to N_MAX node positions and atom types, 12.3 kB per frame for N_MAX=768.
#                 Edges and one-hot features are rebuilt per batch, bitwise equal to the
#                 cached graphs of the 'sqlite' block. No database, no hashing.
#   'sqlite'      the descriptor series holds all-atom coordinates and every graph is
#                 cached in a global sqlite database (graphs_cache.sqlite). Kept for
#                 campaigns that started with it and for rolling back.
#
# Switching a running campaign from 'sqlite' to 'nodetables' needs the node tables of its
# frames: a prefill (python -m aimmd.network.nodetables prefill), or refill=True below for
# the next job to fill them; see the documentation.
import os

import numpy as np
import torch
import MDAnalysis as mda

_here = os.path.dirname(os.path.abspath(__file__))

# --- engine and files --------------------------------------------------------------------
engine = 'gromacs'
topology = 'run.gro'
gmx_grompp = 'gmx grompp -maxwarn 3'
gmx_mdrun = 'gmx mdrun'
gmx_mdp = os.path.join(_here, 'run.mdp')
trajectory_extension = '.xtc'
initial_paths = ['initial.xtc']
states = 'ARB'
name = 'AIMMD_gnn'

# --- graph settings ----------------------------------------------------------------------
CUTOFF = 4.0                                    # edge cutoff, A
# Fixed atom-type table: the one-hot node features (the network input width) do not depend
# on the ligand, so one network can serve a series of ligands.
ATOM_TYPES = ['H', 'C', 'N', 'O', 'F', 'NA', 'P', 'S', 'CL', 'BR', 'I']
ATOMIC_NUMBERS = [1, 6, 7, 8, 9, 11, 15, 16, 17, 35, 53]
SYSTEM_SELECTION = '(resname LIG) and not type H'
ENVIRONMENT_SELECTION = 'not type H and around 8.0 (resname LIG)'
# Graph nodes a node-table row can hold. A frame with more nodes gets an empty row and an
# ERROR line, and training and the workers' value passes (selection, TPS acceptance) stop on
# it (a larger N_MAX is a new series: python -m aimmd.network.nodetables repack --n-max N
# widens the rows, or refill=True below lets the next job do it); frames above 80 % of
# N_MAX are reported as a warning.
N_MAX = 768

# Topology (with bonds, for unwrapping the ligand) and reference box of the graphs
tmp_universe = mda.Universe(topology, to_guess=['bonds', 'masses', 'types'])

# --- states ------------------------------------------------------------------------------
# Bound (A) and unbound (B) from the distance between the ligand center and the pocket
# center (minimum image). Replace with the state definition of your system.
POCKET_SELECTION = 'name CA and around 6.0 (resname LIG)'
BOUND_DISTANCE = 3.0                            # A
UNBOUND_DISTANCE = 20.0                         # A
_LIGAND = tmp_universe.select_atoms(SYSTEM_SELECTION).indices
_POCKET = tmp_universe.select_atoms(POCKET_SELECTION).indices


def states_function(trajectory) -> np.ndarray:
    distance = []
    for ts in trajectory:
        vector = ts.positions[_LIGAND].mean(0) - ts.positions[_POCKET].mean(0)
        box = ts.dimensions[:3]
        vector -= box * np.round(vector / box)
        distance.append(np.linalg.norm(vector))
    distance = np.array(distance)
    labels = np.full(len(distance), 'R', dtype='<U1')
    labels[distance < BOUND_DISTANCE] = 'A'
    labels[distance > UNBOUND_DISTANCE] = 'B'
    return labels


# --- graph input -------------------------------------------------------------------------
GRAPH_INPUT = 'nodetables'                      # 'nodetables' | 'sqlite'

if GRAPH_INPUT == 'nodetables':
    from aimmd.network.nodetables import NodeTableFeaturizer

    FEATURIZER = NodeTableFeaturizer(tmp_universe, SYSTEM_SELECTION, ENVIRONMENT_SELECTION,
                                     ATOM_TYPES, cutoff=CUTOFF, n_max=N_MAX, refill=False)
    # The series name is computed from everything a row holds (selections, atom types,
    # N_MAX, topology and box), so a changed setting starts a new series that the frames
    # of existing runs lack. The launcher and every worker find such frames in the run
    # folder before the job starts. refill=False stops the job with an error naming the
    # remedies (restore the settings, or prefill the new series with python -m
    # aimmd.network.nodetables prefill); refill=True lets one process of the run refill
    # the series first (repack, extract from the graph cache, or featurize) while the
    # others wait.
    descriptors_series = FEATURIZER.series

    # Module-level wrappers: aimmd.Params stores functions, not bound methods.
    def descriptors_function(trajectory):
        return FEATURIZER.descriptors_function(trajectory)

    def descriptor_transform(rows, verbose=False):
        # a ready batch (dict) for fit; a multi-system run returns FEATURIZER.graphs(...)
        return FEATURIZER.batch_dict(rows)

    def _network_input(rows, device):
        return FEATURIZER.batch_dict(rows, device=device)

elif GRAPH_INPUT == 'sqlite':
    from torch_geometric.data import Batch
    from aimmd.network.graph_utils import (
        atom_coordinate_descriptors_function, init_db, process_descriptors_pyg)

    DB_PATH = 'graphs_cache.sqlite'
    conn = init_db(db_path=DB_PATH)

    def descriptor_transform(desc, verbose=False):
        return process_descriptors_pyg(
            desc, mdanalysis_universe=tmp_universe,
            system_selection=SYSTEM_SELECTION, environment_selection=ENVIRONMENT_SELECTION,
            cutoff=CUTOFF, conn=conn, verbose=verbose, atom_types=ATOM_TYPES)['data_list']

    def descriptors_function(trajectory) -> np.ndarray:
        desc = atom_coordinate_descriptors_function(trajectory)
        descriptor_transform(desc, verbose=False)       # caches the graphs
        return desc

    def _network_input(rows, device):
        return Batch.from_data_list(descriptor_transform(rows)).to(device=device).to_dict()

else:
    raise ValueError(f"GRAPH_INPUT must be 'nodetables' or 'sqlite', got {GRAPH_INPUT!r}")

# --- network -----------------------------------------------------------------------------
from painn import PaiNNModel


class PaiNNWrapper(PaiNNModel):
    def __init__(self):
        super().__init__(n_out=1, cutoff=CUTOFF, atomic_numbers=ATOMIC_NUMBERS,
                         n_bases=10, n_polynomials=0, n_layers=3, n_hidden_channels=32,
                         drop_rate=0.1, aggr='add', w_out_after_sum=True, basis_type='gaussian')


network = PaiNNWrapper().to('cuda' if torch.cuda.is_available() else 'cpu')


def values_function(descriptors, use_network=None, batchsize: int = 32, verbose=False):
    if not len(descriptors):
        return np.zeros(0)
    global network
    if use_network is not None:
        network = use_network
    device = next(network.parameters()).device
    values = []
    with torch.no_grad():
        for i in range(0, len(descriptors), batchsize):
            output = network(_network_input(descriptors[i:i + batchsize], device))
            output = output.squeeze().cpu().numpy()
            values.append(output[None] if output.ndim == 0 else output)
    return np.concatenate(values)


from aimmd.network import fit as _fit


def fit(params, pathensemble, verbose=False, worker=None):
    # graphs=True batches the rows through descriptor_transform; in_memory=False (graph
    # inputs are built per batch, not kept for the whole ensemble); worker lets the trainer
    # stop a fit early (e.g. near the job's time limit)
    return _fit(params, pathensemble, nbins=0, lr=1e-3, epochs=3000,
                state_bins='AB', augment='yes', loss_bayesian_factor=20,
                loss_regularization_weight=1e-8, loss_regularization_exponent=2,
                in_memory=False, graphs=True, batch_size=32, verbose=verbose,
                worker=worker)


# --- sampling ----------------------------------------------------------------------------
chain_type = 'rfps'
selection_pool_size = 10
nbins = 10
cutoff_min = 0.5
cutoff_max = 20.0
gen_temperature = 300.0
max_length = 50000
network_batch_size = 4096
trajectory_update_batch_size = 1000
network_save_interval = 10
