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

The params file sets ``descriptors_series = FEATURIZER.series``: the name is
computed from the settings, so a changed setting starts a new series. Before
a job does any work, the launcher and every worker check the run folder for
trajectories without it (`aimmd.core.series`): the job stops with an error
naming the remedies, or, for a featurizer built with ``refill=True``, one
process of the run refills the series first (by repacking, extracting from
the graph cache, or featurizing) while the others wait.

Public API
----------
NodeTableFeaturizer
    Frames to node-table rows (``descriptors_function``), rows to graphs or a
    ready batch dict (``graphs``, ``batch_dict``), the series name, and what a
    job does when trajectories lack the series (``refill``).
MultiSystemNodeTableFeaturizer
    One featurizer per system of a multi-system run, dispatched on
    ``system_id``, with one series name (and ``refill``) for the campaign.
NodeTableOverflowError
    A graph has more nodes than the rows can hold.
NODE_TABLE_LAYOUT, DEFAULT_N_MAX, SERIES_PREFIX
    Layout version, default row capacity and series-name prefix.

Command line
------------
``python -m aimmd.network.nodetables {prefill,repack,verify}`` writes the
node-table series of existing runs (from an old graph cache or from the
trajectories), rewrites them for another ``n_max`` and reports missing rows;
see ``python -m aimmd.network.nodetables --help``.

Writing rows needs only numpy and MDAnalysis; building graphs from them needs
the optional ``graphs`` extra (torch_geometric, torch_cluster).
"""

from ._featurizer import (NodeTableFeaturizer, MultiSystemNodeTableFeaturizer,
                          NodeTableOverflowError, NODE_TABLE_LAYOUT,
                          DEFAULT_N_MAX, SERIES_PREFIX)
