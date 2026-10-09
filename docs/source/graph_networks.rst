Graph Networks and Node Tables
==============================

AIMMD supports graph-neural-network committor models (for example PaiNN). For
every frame the network sees one graph: the *system* atoms (e.g. the ligand
heavy atoms) and the *environment* atoms (e.g. the heavy atoms within 8 Å of
the ligand), after the ligand is made whole, centered in the box and every
atom is wrapped into it; edges join nodes closer than a cutoff. The graphs are
the same as those of :func:`aimmd.network.graph_utils.get_graphs_pyg`.

A params file chooses how AIMMD stores what it needs to build these graphs,
with a ``GRAPH_INPUT`` switch in the params file itself:

``'nodetables'`` (recommended)
   The descriptor series holds the *node table* of every frame: the positions
   and atom types of its graph nodes, in one fixed-width row. Edges, one-hot
   node features and zero shifts are rebuilt per batch, bit for bit equal to
   the graphs of the graph cache. The rows live in AIMMD's ordinary
   per-trajectory series, so they are written, registered, exported and
   deleted together with their trajectory. No database, no locks shared
   between workers, no hashing.

``'sqlite'`` (graph cache)
   The descriptor series holds all-atom coordinates, and every graph is cached
   in a global sqlite database (``graphs_cache.sqlite``) under a hash of the
   frame's coordinates. This is the mode campaigns started with; it is kept
   for them and for rolling back.

The node tables are implemented in :mod:`aimmd.network.nodetables` (see
:doc:`api/network`).

Node Tables
-----------

``NodeTableFeaturizer.descriptors_function`` turns each frame into one
float32 row of ``2 + 4 * n_max`` columns:

=========================  ===================================================
columns                    content
=========================  ===================================================
``0``                      number of graph nodes ``n``
``1``                      layout version (1)
``2 : 2+3*n_max``          node positions in Å after the periodic-boundary
                           handling: system atoms, then environment atoms,
                           each in atom-index order
``2+3*n_max : 2+4*n_max``  atom-type index of each node into ``atom_types``
=========================  ===================================================

Columns after the ``n`` nodes are zero. With the default ``n_max = 768`` a row
has 12,296 bytes, against ``12 * n_atoms`` bytes of coordinates (683,640 bytes
for a 57k-atom protein-ligand system). A computed row never is all zero; an
all-zero row is a frame that has not been featurized (or does not fit, see
below), and the compute ledger fills it.

``graphs(rows)`` and ``batch_dict(rows, device)`` rebuild the network input:
``node_attrs`` one-hot over ``atom_types``, ``edge_index`` from
``radius_graph(positions, r=cutoff, loop=False, max_num_neighbors=128)`` and
zero ``shifts``. The edges are always built **on the CPU** and only then moved
to the device, so the result is bitwise reproducible and equal to the cached
graphs; ``batch_dict`` equals ``Batch.from_data_list(graphs).to_dict()``.
Both raise on an all-zero row, so a missing row can never become a wrong
value.

**Series name.** Rows are stored as ``{trajectory}.{descriptors_series}.npy``
under a fingerprinted name, ``'descriptors-gn'`` plus 10 hex characters of a
SHA-256 over what a row holds: the layout version, ``n_max``, both selections,
``atom_types``, the atom count and a hash of the atom names, types, residue
names and the reference box. Any change to these gives a new series, so rows
are never read under settings they were not written with. ``cutoff`` and
``max_num_neighbors`` are deliberately *not* part of the name: the edges are
rebuilt from the rows every time, so changing them changes every graph without
touching the stored rows, and the network must be retrained. The params file
sets ``descriptors_series = FEATURIZER.series``: the name is computed, never
typed, and a changed setting is caught from the run folder before a job
starts (see `When the Series Changes`_).

The Params File
---------------

The template :download:`examples/graph_network/params.py
<../../examples/graph_network/params.py>` holds a complete params file for a
ligand-unbinding run. Its graph-input block:

.. literalinclude:: ../../examples/graph_network/params.py
   :language: python
   :start-after: # --- graph input
   :end-before: # --- network

The network's ``values_function`` and ``fit`` are the same in both modes:
``values_function`` passes blocks of rows through ``_network_input`` to the
network, and ``fit`` is called with ``graphs=True, in_memory=False``.

**Module-level wrappers.** :class:`aimmd.Params` stores functions, not
objects: a bound method such as ``FEATURIZER.descriptors_function`` would lose
its featurizer, and ``Params`` refuses it with an error. So does a
``functools.partial`` of one, which ``params1.py`` could not import. Define
module-level functions that call the featurizer, as above.

**The series name and refill.** ``descriptors_series = FEATURIZER.series``
follows the settings. A changed selection, ``n_max`` or topology, or atom
types that MDAnalysis guesses differently on another host, start a new series
that the trajectories of an existing run do not have. The launcher and every
worker find that out before a job does any work, and the featurizer's
``refill`` flag (``NodeTableFeaturizer(..., refill=False)``, the default)
decides what happens: an error that names the remedies, or a refill of the
series by the job itself (see `When the Series Changes`_). ``refill`` is not
part of the series name. ``FEATURIZER.check_series(name)`` still checks a
pinned name, but pinning is no longer recommended.

**n_max and frames that do not fit.** ``n_max = 768`` leaves about twice the
headroom of what protein-ligand runs need (267 to 359 nodes over the HSP90
compound-9 campaign). A frame with more than ``n_max`` nodes (or with an
environment atom whose type is not in ``atom_types``) never stops the MD: its
row stays zero and an ``ERROR: node tables`` line is printed. Every pass that
needs the frame's value or uses it for training then stops on the zero row
with an error naming ``python -m aimmd.network.nodetables repack --n-max N``:
the trainer, and also a shooting worker whose pool selection or TPS
acceptance reaches the frame (with TPS and a pool of one, the worker that
produced the path, right after registering it). A restart stops at the same
place again. This is deliberate: a value made up for the frame, or a path
accepted or rejected without it, would bias the sampling silently. Widen the
rows: set the new ``n_max`` (a new series) and either repack the rows before
the next job (``repack --n-max N --from-n-max M``, then ``prefill
--only-missing``), or construct the featurizer with ``refill=True`` and
restart: the job then repacks the rows itself, and the ledger fills the empty
ones. Frames above 80 % of ``n_max`` are reported with a warning; act on it
before the first frame overflows.

**fit.** With ``graphs=True``, ``fit`` passes each batch of rows through
``descriptor_transform``. A single-system node-table transform returns
``FEATURIZER.batch_dict(rows)``, which ``fit`` uses as the batch directly.
``in_memory=True`` together with ``graphs=True`` is ignored with a warning:
graph inputs are built per batch.

**Multi-system runs.** Build one featurizer per system and combine them with
:class:`aimmd.network.nodetables.MultiSystemNodeTableFeaturizer`; the wrappers
take the ``system_id`` keyword. All systems share one ``atom_types`` table and
one series name. A multi-system ``fit`` transforms each system separately, so
its ``descriptor_transform`` returns graphs, not a batch dict:

.. code-block:: python

    FEATURIZERS = MultiSystemNodeTableFeaturizer({
        'lig1': NodeTableFeaturizer(universe1, ..., ATOM_TYPES, cutoff=CUTOFF),
        'lig2': NodeTableFeaturizer(universe2, ..., ATOM_TYPES, cutoff=CUTOFF)},
        refill=False)
    descriptors_series = FEATURIZERS.series

    def descriptors_function(trajectory, system_id):
        return FEATURIZERS.descriptors_function(trajectory, system_id)

    def descriptor_transform(rows, system_id, verbose=False):
        return FEATURIZERS.graphs(rows, system_id)

The Command-Line Tool
---------------------

``python -m aimmd.network.nodetables`` writes, rewrites and checks the node
tables of existing runs. It imports the params file (in node-table mode) as a
module in its folder, without :class:`aimmd.Params`, and takes its featurizer
(``FEATURIZER``, the only featurizer it defines, or ``--featurizer NAME``). A
params file that sets ``GRAPH_INPUT`` to another string is refused before it
runs, since in ``'sqlite'`` mode it would open its graph cache; a params file
without ``GRAPH_INPUT`` is run, so do not pass an old one. It
covers every trajectory of the given run folders that has a states series:
exported initial paths, chain paths, the halves of shots in flight and
free-simulation parts. Like AIMMD, it covers the frames that can be read: a
last frame cut short (by a job killed while it wrote) gets no row and is
reported. It runs on CPUs only.

.. code-block:: bash

    # write the node tables of every trajectory, from the old graph cache
    python -m aimmd.network.nodetables prefill --params params.py \
        --run run1 --db graphs_cache.sqlite -j 64 --verify 200 \
        --report prefill.json

    # report missing, short and zero rows; check 32 rows per file
    python -m aimmd.network.nodetables verify --params params.py \
        --run run1 --sample 32

    # rewrite the node tables for a larger n_max (no trajectory is read)
    python -m aimmd.network.nodetables repack --params params.py \
        --run run1 --n-max 1024

``prefill``
   Writes ``{trajectory}.{series}.npy`` for every trajectory. With ``--db``
   (the graph cache of a ``'sqlite'`` campaign) each frame's row comes from
   its cached graph: the frame is decoded, its cache key computed and the
   graph looked up; frames without a usable graph are featurized. Without
   ``--db`` every frame is featurized. ``--only-missing`` keeps the rows that
   are there and computes only missing, short and zero rows; running it twice
   changes nothing. ``--verify K`` featurizes ``K`` random frames per file
   directly and compares them bit for bit; a mismatching file is not written.
   With ``--db`` it defaults to 4: the cache keys hash the coordinates only,
   so graphs cached with other selections would otherwise go unnoticed.
``repack``
   Rewrites every series file of the featurizer for ``--n-max N`` (a new
   series name, printed at the end). Rows of frames that did not fit stay
   zero: set ``n_max = N`` in the params file and run ``prefill
   --only-missing`` (or let the trainer's ledger fill them). If the params
   file already holds the new ``n_max``, give the old one with
   ``--from-n-max M``; in a multi-system
   run ``--from-n-max SYSTEM_ID=M`` (repeatable) gives one system's, and a
   plain ``M`` that of the systems not named.
``verify``
   Reports, per trajectory, a missing series file, missing (short), zero and
   extra rows and rows of another layout; ``--sample K`` compares ``K`` filled
   rows bit for bit with a direct featurization.

Every command takes ``-j N`` (processes; each imports the params file once)
and ``--report FILE`` (a JSON report with per-file counts and timings). The
exit status is 0 when everything is complete and verified, 1 when a file
failed, a row mismatched, rows are left empty or a series is incomplete, and 2
on bad input.

Files are written to a temporary file next to the series file, which replaces
it only once it is complete and verified, under the series file's lock; a
crash leaves the series file as it was. The tool never opens
``*.descriptors.npy`` and opens graph caches read-only and immutable, so it
never writes them (not even ``-wal`` or ``-shm`` files). Run it while no job
runs on the runs; a write-ahead log that is not empty is reported, because the
graphs in it are not seen (checkpoint the cache first, or accept that those
frames are featurized).

When the Series Changes
-----------------------

Every featurizer registers its series name and ``refill`` flag when it is
built (:mod:`aimmd.core.series`), in every process that executes the params
file. Before a job does any work, the series of the run is checked twice:
by the launcher when it creates or runs a job (``create_job``, ``run``), so
that you hear about it on the login node before you submit, and by every
worker (shoot, free and train) before any MD or training, as a backstop for a
job resubmitted without a new job script. The check lists the trajectories
of the run (those with a states series, in every system folder of a
multi-system run), leaves out the initial paths in ``initial*/`` (exported and
featurized again at every launch) and looks for trajectories that have *no*
file ``{trajectory}.{series}.npy``; a file with zero or missing rows is the
ledger's normal lazy filling and counts as there. A new run passes.

``refill=False`` (the default)
   The launcher writes no job and every worker stops before its task, so the
   job ends. The error says how many of how many trajectories (and frames)
   of which run lack the series, what is next to them (another
   ``descriptors-gn`` series: the node-table settings changed since these
   frames were featurized; ``*.descriptors.npy``: the campaign has not been
   migrated from the graph-cache input) and the remedies: restore the
   settings, prefill the series (``python -m aimmd.network.nodetables
   prefill --params ... --run ...``, or the campaign's
   ``prefill_nodetables.sh``), or construct the featurizer with
   ``refill=True``.
``refill=True``
   The launcher prints a notice of how many trajectories and frames the job
   will refill first, and by which route, and writes the job. When the job
   starts, the first process of the run to take the lock file
   ``.series-refill.lock`` in the run folder checks again and refills, with
   as many processes as it has CPUs; every other process of the run waits
   for the lock, then checks again and starts. If the series is still
   incomplete after the refilling process let go of the lock (it failed),
   the waiting processes stop with the error instead of refilling again.

The refill writes exactly the trajectories that lack the series, by the
cheapest correct route per trajectory; files of the series that exist are
never rewritten, and the files of the old series stay where they are:

1. **repack**, when next to the trajectory is a node-table series that
   differs only in ``n_max`` (its row width gives the old ``n_max``, and
   ``FEATURIZER.with_n_max(old).series`` must be its name): exact, and no
   trajectory is read; zero rows stay zero for the ledger;
2. **extract**, when the run holds the ``*.descriptors.npy`` of the
   graph-cache input and ``graphs_cache.sqlite`` is in the job's working
   directory (the campaign folder, where ``DB_PATH`` of the ``'sqlite'``
   block points): rows from the cached graphs, checked against a direct
   featurization on 4 frames per trajectory; a trajectory that fails the
   check is featurized instead;
3. **featurize** every frame from the trajectory, otherwise.

Every line it writes starts with ``SERIES REFILL:``, in the log of the
refilling process and of each waiting one::

    SERIES REFILL: 14 of 36 trajectories (412,733 frames) of run '/.../run1'
      have no 'descriptors-gn...' rows; this process (host, pid 4711, train
      run1) refills them now with 16 processes by featurizing every frame of
      the trajectories; the other AIMMD processes of this run wait until it
      is done.
    SERIES REFILL: progress: 9 of 14 trajectories, 280,114 of 412,733 frames,
      7 min, about 3 min left
    SERIES REFILL: done: 'descriptors-gn...' of 14 trajectories (412,733
      frames) of run '/.../run1' refilled by featurizing every frame of the
      trajectories in 655.0 s; the files of the old series were left in
      place.

    SERIES REFILL: waiting for host, pid 4711, train run1 to refill
      'descriptors-gn...' (14 of 36 trajectories ... lack it) before starting.
    SERIES REFILL: still waiting for host, pid 4711, train run1 to refill
      'descriptors-gn...' (5 min so far).
    SERIES REFILL: 'descriptors-gn...' is complete after 11 min of waiting
      (refilled by host, pid 4711, train run1); starting.

A refill inside a job uses the CPUs of one worker task; for a large campaign
a prefill on a dedicated node (``-j 64``, the graph cache copied to
``/dev/shm``) is faster.

Switching a Running Campaign
----------------------------

A campaign that started in ``'sqlite'`` mode switches to node tables between
two jobs. **The node tables of the existing frames come first**: without
them, every resumed worker would featurize its whole in-flight half again and
the trainer's ledger every frame of the ensemble, at the featurization cost
per frame. A job therefore refuses to start on such a run (see `When the
Series Changes`_), unless the featurizer has ``refill=True`` and the job
fills them first. Filling them is cheap only while the old graph cache
exists: on a workstation, the 57k-atom HSP90 system took about 5 ms per frame
on the extract route (2.5 ms to decode the frame, 2.2 ms to hash it, look it
up and convert the graph) against about 25 ms per frame to featurize it.

1. **Deploy** this AIMMD version where the campaign runs. The default mode is
   unchanged. Run the test suite there, and the opt-in golden test
   (``AIMMD_NODETABLES_GOLDEN_DATA=... pytest
   tests/test_nodetables_golden.py --rungraph``) on the architecture that
   will featurize. Saved params files now hold the ``descriptors_series``
   field, so the first load with this version saves its params under the
   next free number (e.g. ``params2.py`` next to an older ``params1.py``);
   an older file without the field loads with the default series.
2. **Stop**: no job may run on the runs (check the queue).
3. **Edit the params file**: add the graph-input block of the template with
   ``GRAPH_INPUT = 'nodetables'`` (``descriptors_series =
   FEATURIZER.series``).
4. **Prefill** with the old graph cache, on one CPU node per campaign (or
   construct the featurizer with ``refill=True`` and let the first job
   extract the rows in one of its processes, more slowly)::

       python -m aimmd.network.nodetables prefill --params params.py \
           --run run1 --db graphs_cache.sqlite -j 64 --verify 200 \
           --report prefill.json

   Every frame is one random read from the cache: for a cache of several GB
   on a parallel filesystem, copy it to node-local storage (e.g.
   ``/dev/shm``) first and pass the copy to ``--db``. ``--report`` records
   the time per frame of each step.

5. **Check**: ``verify --sample 32`` exits with status 0.
6. **Regenerate** the worker params file (``params1.py``, or the next free
   number) and the job script on the cluster: :meth:`aimmd.Params.load`
   writes this host's paths into them.
7. **Resubmit and validate** over two or three rounds: no graph-cache staging
   lines, the trainer reports about no missing descriptor frames, values of
   frames that did not change agree before the first fit, and there are no
   ``n_max`` warnings or empty-row errors. The network inputs are bitwise
   equal in both modes, but the network's output moves in the last bits with
   the batches it is evaluated in (which frames, ``network_batch_size``),
   the device and the number of threads, in either mode: compare values with
   a tolerance (about ``1e-5``), or bitwise only after recomputing both sides
   in the same setup.
8. **Clean up**: delete ``*.descriptors.npy`` after the validation, and the
   graph cache (``graphs_cache.sqlite*``) once you no longer want a cheap
   rollback. Until then, keep the cache: it is the only cheap source for a
   prefill and for rolling back.

Shots in flight when you switch have their ``back``/``forw`` halves
prefilled, so they continue and register node tables only.

Rolling Back
------------

Set ``GRAPH_INPUT = 'sqlite'``: the descriptor series is ``'descriptors'``
again and the graph cache is used. Frames written in node-table mode have no
coordinate rows and no cached graphs: the ledger computes their coordinate
rows from the trajectories, and their graphs are built on first use. While the
``*.descriptors.npy`` files and the graph cache of the old frames exist, this
is cheap; after you have deleted them, every frame is featurized again. The
node-table files are ignored in ``'sqlite'`` mode and can stay or go.

Analysis
--------

In a node-table run, ``path.descriptors`` (the ``'descriptors'`` series) does
not exist: it raises an error instead of returning zeros. Use

- ``path.coordinates`` (or ``path.positions``) for coordinates, read from the
  trajectory: the same numbers the ``'descriptors'`` series held;
- ``getattr(path, params.descriptors_series)`` for the node tables, and
  ``params.values_function(rows)`` for committor values;
- ``FEATURIZER.graphs(rows)`` for the network's graphs (the ligand centered
  and made whole, with its environment), e.g. for attributions, and
  ``FEATURIZER.batch_dict(rows)`` for a batch.

Graph Cache (``'sqlite'`` Mode)
-------------------------------

Everything that maintains the graph cache serves the ``'sqlite'`` block
only: :func:`~aimmd.network.graph_utils.init_db`,
:func:`~aimmd.network.graph_utils.process_descriptors_pyg`,
:func:`~aimmd.network.graph_utils.atom_coordinate_descriptors_function`, the
``/dev/shm`` replica and memo of :mod:`aimmd.network.shm_cache`, and the
environment variables that tune them (``AIMMD_SHM_*``, ``AIMMD_STAGE_*``,
``AIMMD_STORE_*``, ``AIMMD_SQLITE_BUSY_SECONDS``, ``AIMMD_GRAPH_MEMO_BYTES``,
``AIMMD_PENDING_WRITE_BYTES``, ``AIMMD_TRAINER_WRITES_CACHE``). A node-table
run opens no database, so the trainer stages nothing and none of these
settings has an effect.
