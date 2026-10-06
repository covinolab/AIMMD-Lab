"""Helpers for the graph-key tests (``descriptor_cache = 'graphkeys'``).

None of this needs torch_geometric: the toy caches stand in for the sqlite
graph cache of ``aimmd.network.graph_utils`` and follow its contract.

- Coordinate rows are keyed, looked up, built if missing and stored; the
  graphs returned go to the open overlays (``collect_graphs``), as
  ``process_descriptors_pyg`` does.
- Key rows are only looked up; a zero row or an unknown key raises
  ``GraphCacheMiss``.

A toy graph is ``('graph', sum of the frame's coordinates)``, so values
computed from key rows can be compared with values computed from coordinate
rows.
"""

import numpy as np

from aimmd.core.graphkey import graph_keys, is_key_batch, keys_to_hex
from aimmd.network.graph_lookup import (GraphCacheMiss, collect_graphs,
                                        overlay_get)


def descriptors_function(trajectory):
    """Flattened float32 coordinates, as atom_coordinate_descriptors_function."""
    rows = [ts.positions.ravel().copy() for ts in trajectory]
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def toy_graph(row):
    return ('graph', float(np.asarray(row, dtype=np.float64).sum()))


class ToyCache:
    """A dict-backed graph cache with a key-aware transform."""

    def __init__(self, fail_store=False):
        self.store = {}
        self.built = []
        self.calls = []
        self.fail_store = fail_store

    def transform(self, x):
        if is_key_batch(x):
            self.calls.append(('keys', len(x)))
            out, missing = [], []
            for h, row in zip(keys_to_hex(x), x):
                graph = None
                if row.any():
                    graph = overlay_get(h)
                    if graph is None:
                        graph = self.store.get(h)
                if graph is None:
                    missing.append(row)
                out.append(graph)
            if missing:
                raise GraphCacheMiss(np.unique(np.stack(missing), axis=0))
            return out
        x = np.asarray(x)
        self.calls.append(('rows', len(x)))
        hexes = keys_to_hex(graph_keys(x)) if len(x) else []
        graphs = []
        for h, row in zip(hexes, x):
            graph = self.store.get(h)
            if graph is None:
                graph = toy_graph(row)
                self.built.append(h)
                if not self.fail_store:
                    self.store[h] = graph
            graphs.append(graph)
        collect_graphs(hexes, graphs)
        return graphs

    def values(self, x):
        if not len(x):
            return np.zeros(0)
        return np.array([g[1] for g in self.transform(x)])

    def cache(self, rows):
        rows = np.asarray(rows)
        for h, row in zip(keys_to_hex(graph_keys(rows)), rows):
            self.store[h] = toy_graph(row)
