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

import builtins
import io
import pickle
import sqlite3
from contextlib import contextmanager

import numpy as np

from aimmd.core.graphkey import graph_keys, is_key_batch, keys_to_hex
from aimmd.network.graph_lookup import (GraphCacheMiss, collect_graphs,
                                        lookup_graphs, overlay_get)


def descriptors_function(trajectory):
    """Flattened float32 coordinates, as atom_coordinate_descriptors_function."""
    rows = [ts.positions.ravel().copy() for ts in trajectory]
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def toy_graph(row):
    return ('graph', float(np.asarray(row, dtype=np.float64).sum()))


class ToyCache:
    """A dict-backed graph cache with a key-aware transform.

    ``graph(row)`` builds the graph of a frame (default :func:`toy_graph`).
    """

    def __init__(self, fail_store=False, graph=toy_graph):
        self.store = {}
        self.built = []
        self.calls = []
        self.fail_store = fail_store
        self.graph = graph

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
                graph = self.graph(row)
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
            self.store[h] = self.graph(row)


class SqliteToyCache:
    """A toy graph cache in a real sqlite file (the graph_utils schema).

    Key rows go through ``graph_lookup.lookup_graphs`` with the connection,
    so ``GraphKeysFunction`` finds the cache by its connection and asks it
    for presence, exactly as with the production cache.
    """

    def __init__(self, db_path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute('CREATE TABLE IF NOT EXISTS graphs_cache '
                          '(key TEXT PRIMARY KEY, data BLOB)')
        self.conn.commit()
        self.built = []

    def transform(self, x):
        if is_key_batch(x):
            return lookup_graphs(x, self.conn, decode=pickle.loads)
        x = np.asarray(x)
        hexes = keys_to_hex(graph_keys(x)) if len(x) else []
        graphs = []
        for h, row in zip(hexes, x):
            blob = self.conn.execute(
                'SELECT data FROM graphs_cache WHERE key = ?', (h,)).fetchone()
            if blob is None:
                graph = toy_graph(row)
                self.built.append(h)
                self.conn.execute('INSERT OR REPLACE INTO graphs_cache '
                                  'VALUES (?, ?)', (h, pickle.dumps(graph)))
            else:
                graph = pickle.loads(blob[0])
            graphs.append(graph)
        self.conn.commit()
        collect_graphs(hexes, graphs)
        return graphs

    def values(self, x):
        if not len(x):
            return np.zeros(0)
        return np.array([g[1] for g in self.transform(x)])

    def add(self, rows):
        """Store graphs of ``rows`` (as an earlier campaign would have)."""
        self.transform(np.asarray(rows, dtype=np.float32))

    def count(self):
        return self.conn.execute('SELECT COUNT(*) FROM graphs_cache').fetchone()[0]


class DescriptorFileOpened(AssertionError):
    """A ``*.descriptors.npy`` file was opened although it must not be."""


@contextmanager
def forbid_descriptor_files(monkeypatch, opened=None):
    """Fail on any open (read or write) of a ``*.descriptors.npy`` file.

    Covers ``builtins.open`` and ``io.open`` (np.load, np.save, update_npy
    and pathlib use them) and ``np.load``. ``opened`` collects the
    offending names, for tests that check the count instead.
    """
    original_open = builtins.open
    original_io_open = io.open
    original_load = np.load
    record = opened if opened is not None else []

    def check(file):
        name = str(getattr(file, 'name', file))
        if name.endswith('.descriptors.npy'):
            record.append(name)
            raise DescriptorFileOpened(f'opened {name!r}')

    def guarded_open(file, *args, **kwargs):
        check(file)
        return original_open(file, *args, **kwargs)

    def guarded_io_open(file, *args, **kwargs):
        check(file)
        return original_io_open(file, *args, **kwargs)

    def guarded_load(file, *args, **kwargs):
        check(file)
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', guarded_open)
    monkeypatch.setattr(io, 'open', guarded_io_open)
    monkeypatch.setattr(np, 'load', guarded_load)
    try:
        yield record
    finally:
        monkeypatch.setattr(builtins, 'open', original_open)
        monkeypatch.setattr(io, 'open', original_io_open)
        monkeypatch.setattr(np, 'load', original_load)
