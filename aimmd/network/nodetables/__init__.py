"""
aimmd.network.nodetables
========================

Node tables: the inputs of a graph network, stored as a descriptor series.

For graph-network committor models (e.g. PaiNN), `NodeTableFeaturizer` turns
each frame into one fixed-width row holding the positions and atom types of
the frame's graph nodes. The rows are AIMMD descriptors like any other: they
are cached per trajectory in ``{trajectory}.{descriptors_series}.npy`` under a
fingerprinted series name (``'descriptors-gn...'``), and the network input,
bitwise equal to the graphs of `aimmd.network.graph_utils.get_graphs_pyg`, is
rebuilt from them per batch. No graph cache or database is involved.

Public API
----------
NodeTableFeaturizer
    Frames to node-table rows (``descriptors_function``), rows to graphs or a
    ready batch dict (``graphs``, ``batch_dict``), and the series name.
NodeTableOverflowError
    A graph has more nodes than the rows can hold.
NODE_TABLE_LAYOUT, DEFAULT_N_MAX, SERIES_PREFIX
    Layout version, default row capacity and series-name prefix.

Writing rows needs only numpy and MDAnalysis; building graphs from them needs
the optional ``graphs`` extra (torch_geometric, torch_cluster).
"""

from ._featurizer import (NodeTableFeaturizer, NodeTableOverflowError,
                          NODE_TABLE_LAYOUT, DEFAULT_N_MAX, SERIES_PREFIX)
