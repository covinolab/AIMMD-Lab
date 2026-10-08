Advanced Usage
==============

This page covers AIMMD's advanced modes. Each is optional and disabled by
default, so a basic single-system run (see :doc:`workflow`) is unaffected.

Kinetics-Convergence Analysis
-----------------------------

After a completed AIMMD run you may want to verify that the estimated rate
constants are stable with respect to the amount of training data used.
:meth:`aimmd.Worker.kinetics_convergence` provides a built-in convergence
analysis directly on an existing run directory.

The method iterates over a list of *fractions* (defaulting to
``[0.2, 0.4, 0.6, 0.8, 1.0]``). For each fraction ``f`` it:

1. Sub-samples the path ensemble: the first ``round(N * f)`` paths are taken from
   each shooting chain, and the first ``round(N * f)`` frames from each
   free-simulation trajectory. Sub-sampling is done **per source** (chain / free
   trajectory) so all sources contribute the same fraction.
2. Retrains the committor network from scratch on the sub-sampled data.
3. Reweights the sub-sampled ensemble and estimates ``k12`` and ``k21``.
4. Saves the trained network to a per-fraction checkpoint file.

The results are returned as a structured NumPy array with fields ``fraction``,
``k12``, and ``k21``, and are saved to a ``.npy`` file in the worker directory.

.. code-block:: python

    import aimmd
    import numpy as np
    import matplotlib.pyplot as plt

    params  = aimmd.Params.load('params.py')
    worker  = aimmd.Worker(params, 'run_directory')

    # Run with default 20 %-increment fractions.
    # Per-fraction networks are saved as run_directory/networkARB.kcv020.h5 etc.
    results = worker.kinetics_convergence()

    # Plot k12 convergence.
    plt.semilogy(results['fraction'], results['k12'], 'o-')
    plt.xlabel('Fraction of training data')
    plt.ylabel('k12 [1/dt]')
    plt.title('Rate convergence')
    plt.show()

    # Load saved results later.
    results = np.load('run_directory/kinetics_convergence.npy')

By default the network trained on each fraction is saved next to the run
directory's normal network file, following the pattern
``{directory}/network{states}.kcv{fraction_pct:03d}.h5`` (e.g. with
``states='ARB'`` and ``fraction=0.40`` the file is
``run_directory/networkARB.kcv040.h5``). The pattern can be customised via the
``network_save_pattern`` keyword (placeholders ``{directory}``, ``{states}``,
``{fraction}``, ``{fraction_pct}``); pass ``network_save_pattern=None`` to skip
saving networks. Training hyperparameters can be overridden for a faster
exploratory run:

.. code-block:: python

    results = worker.kinetics_convergence(
        fractions=[0.25, 0.5, 0.75, 1.0],
        epochs=200,
        save_file='run_directory/kcv_quick.npy',
    )

.. note::

   ``kinetics_convergence`` saves and restores the **original** trained network
   after the analysis finishes, so the worker is left in the same state as
   before the call. Per-fraction checkpoint files remain on disk and can be
   loaded via ``torch.load`` or :meth:`aimmd.Params.update_network`.

Multi-System (Multi-Ligand) Runs
--------------------------------

A single params file can drive **several chemical systems at once** (for example
two ligands binding the same host) and train **one shared committor model** that
takes a graph/descriptor from *either* system and returns its committor. This is
fully backward compatible: it is enabled only when ``multi_system=True``;
otherwise everything behaves exactly like the single-system workflow.

**Enabling multi-system mode.** Set ``multi_system=True`` and provide one entry
per system for the fields that are otherwise single-valued:

.. code-block:: python

    multi_system = True
    multi_system_share_network = True            # one shared network (see below)
    system_ids   = ['G2', 'G4']                  # per-system labels
    topology     = ['G2.gro', 'G4.gro']          # one topology per system
    initial_paths = [['G2_tp.trr'], ['G4_tp.trr']]   # one group per system
    atom_types   = ['H', 'C', 'N', 'O', 'F', 'NA', 'P', 'S', 'CL', 'BR', 'I']

The ``system_ids`` name the per-system subfolders ``<run>/<system_id>/`` and
index the per-system entries of the list-valued fields. If left empty they
default to ``['0', '1', ...]``.

**The ``system_id`` keyword.** In multi-system mode the user data functions
receive an extra ``system_id`` keyword so a single function can encode per-ligand
differences (e.g. different state cutoffs or atom selections):

.. code-block:: python

    def states_function(trajectory, system_id=None):
        cutoff = 4.5 if system_id == 'G2' else 4.1   # per-ligand state boundary
        ...

``system_id`` is passed **only if a function declares it** (detected via
:func:`aimmd.core.utils.accepts_system_id`), so existing single-system functions
keep working unchanged. The same applies to ``descriptors_function``,
``values_function`` and ``descriptor_transform``.

**Shared graph encoding.** For one network to consume graphs from several
systems, set ``atom_types`` to a fixed, ordered atom-type table. Every system is
then encoded into the same one-hot node columns (unused columns stay zero) and
the network's input width equals ``len(atom_types)``. With ``atom_types=None``
the legacy per-universe encoding (``sorted(set(types))``) is used.

**Directory layout.** A multi-system run nests one level: each system gets its
own subfolder, reusing the ordinary per-directory worker machinery::

    run1/
      G2/  initialARB/ chainR0/ freeA/ freeB/ binsARB.npy densitiesARB.npy
      G4/  initialARB/ chainR0/ freeA/ freeB/ binsARB.npy densitiesARB.npy
      networkARB.h5            # the ONE shared network (share-network mode)

**Shared vs separate networks** (``multi_system_share_network``):

* **True** — one shared network is trained by a single trainer that hands the
  params ``fit`` function a **list** of per-system PathEnsembles. The default
  AIMMD ``fit`` pools them in a *balanced* way (each system carries ``1/N`` of the
  selection weight in every bin, including the in-state anchor bins), so neither
  ligand dominates regardless of how much data each has. The shared network is
  written once at the run root (``run1/networkARB.h5``) and read by every
  system's shooting workers. Rates/kinetics are still computed **per system, in
  sequence**.
* **False** — each system trains its own network
  (``run1/<system_id>/networkARB.h5``) with its own trainer. The flag
  ``trainers_share_gpu`` (default ``True``) controls whether those trainers share
  one GPU or are spread across GPUs.

**Worker counts.** The per-run worker counts ``n`` / ``n1`` / ``n2`` apply **per
system** in multi-system mode (e.g. ``launcher.run(n=2, n1=1, n2=1)`` gives every
system 2 shooting + 1 freeA + 1 freeB worker). Launching is otherwise identical:

.. code-block:: python

    import aimmd
    params   = aimmd.Params.load('params.py')   # multi_system=True
    launcher = aimmd.Launcher(params, 'run1')
    launcher.run(n=2, n1=1, n2=1, nframes=25000)
    # or generate a SLURM script (per-system srun lines):
    launcher.create_job('job.sh', n=2, n1=1, n2=1, walltime=86400)

See the multi-system tutorial (:doc:`tutorials/2_multi_system`) for a complete
runnable example on the toy engine.

.. note::

   The first release of multi-system support targets ``chain_type='rfps'`` with
   the committor balancing described above. LSR/MAR regularization and
   ``rescale_committor`` are not yet combined with ``multi_system`` and raise a
   clear ``NotImplementedError``; single-system runs retain full support for all
   of them.

Biased (OPES / PLUMED) Dynamics
-------------------------------

For runs that apply an **in-state bias** during dynamics (e.g. a frozen
OPES_METAD that flattens a bound well), AIMMD records the per-frame bias and
recovers unbiased kinetics with the Tiwary-Parrinello correction:

.. math::

    k = \frac{1}{\sum_i w_i \, L_i \, \gamma_i},
    \qquad \gamma_i = \langle e^{\beta V_i} \rangle,

where :math:`w_i` and :math:`L_i` are the reweight and length of path :math:`i`
and :math:`\gamma_i` is its mean bias factor. Enable it with ``record_bias=True``
and a ``bias_function`` (see :ref:`bias-recording` for the parameter details).

In a **multi-system** run the bias enters GROMACS through the per-system
``gmx_mdrun`` string, which is already list-valued, so each system gets its own
PLUMED input:

.. code-block:: python

    record_bias = True
    bias_source = 'file'                     # read each frame's bias from COLVAR
    gmx_mdrun = ['gmx mdrun -plumed /abs/G2/plumed.dat',
                 'gmx mdrun -plumed /abs/G4/plumed.dat']
    bias_reactive_threshold = [0.5, 0.3]     # per-system (scalar also allowed)

The trainer builds the per-frame bias cache **per system**, runs the
reactive-region bias check against each system's
``bias_reactive_threshold_of(system_id)``, and prints **per-system**
Tiwary-Parrinello bias-reweighted rates next to the raw ones. Kinetics
convergence fills the ``k12_rw`` / ``k21_rw`` columns per system.

.. important::

   Each system's PLUMED ``PRINT STRIDE`` (COLVAR output stride) must equal that
   system's ``nstxout-compressed`` so that COLVAR row *i* lines up with
   trajectory frame *i*; a mismatch silently misaligns the cached bias.

Where Free Simulations Start
----------------------------

Free simulations sample the in-state equilibrium and the exit flux. There are two
distinct moments at which one has to be given a starting configuration, and they
are controlled independently:

``free_seeding_position``
   Where the **first** free trajectory of a state starts, chosen from the initial
   path.

``free_restart_source``
   Where **every later** one restarts from, after the previous trajectory has
   committed to a state.

Both default to the historical behaviour, and both accept a per-state mapping such
as ``{'A': 'deepest'}``.

Why the first seed is not where you might expect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`aimmd.Params` replaces each entry of ``initial_paths`` by its *transition
block* -- the first :meth:`aimmd.Path.split` block whose type is a transition.
Because ``split()`` overlaps neighbouring blocks by two frames, that block begins
at the **last in-state frame before the reactive region**. Everything deeper is
dropped.

So by default the first free simulation starts *just inside* the state boundary,
however deep the initial path reached. That is harmless when the boundary sits
close to the bound minimum and badly wrong when it does not -- an accelerated run
whose boundary was pushed outwards to clear a frozen bias can find its free
simulations starting on the shoulder of the barrier, leaving almost immediately,
and reporting a rate biased fast.

``free_seeding_position`` addresses that. It is a fraction over the target state's
own frames, ordered from the far side of the state towards the reactive region:
``0.0`` (``'deepest'``) is as deep as the initial path reaches, ``1.0``
(``'boundary'``, the default) is the frame adjacent to the reactive region, and
``0.5`` (``'middle'``) is halfway. The same number means the same thing for both
end states, which for a state at the end of the path is the mirror image of the
file order.

.. code-block:: python

   # start the first free-A run as deep in the basin as the initial path goes,
   # and draw every later restart from the accumulated in-A ensemble
   free_seeding_position = {'A': 'deepest'}
   free_restart_source   = {'A': 'equilibrium'}

Brief recrossings do not truncate the candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An unbinding path often pops across the state boundary for a frame and comes
straight back. Such a blip is **skipped**, not treated as the end of the state:
the frames behind it remain candidates. Frames outside the state are excluded
from selection rather than merely traversed, so an intermediate position can
never land on a reactive frame and start the run outside the basin.

This is not hypothetical. One calixarene guest's initial path reads

.. code-block:: text

   BBBBBBBRRRRRRRRRRAAAAAAAAA R AAAAAAAAAAAAAAAAAAAAAAA
          boundary=17 ^       ^26                    ^49

Frame 26 sits at 6.35 A against a 5.5 A state boundary -- one frame -- then drops
back to 5.37 A. Stopping there would leave ``'deepest'`` pointing at frame 25
(4.97 A) and hide the 23 deeper frames behind it, the deepest being frame 49 at
3.64 A. That is 1.33 A of depth given up to a single-frame excursion, more than
the offset the setting exists to remove.

.. note::

   Nothing is exported at build time and nothing is cached on disk for this. The
   deeper frames are read from the file ``initial_paths`` names, resolved exactly
   as the field itself resolves it, so a hand-edited ``params.py`` takes effect on
   an already-built run directory without rebuilding it. The file therefore has to
   still be where ``params.py`` says; if it is not, a non-default position fails
   with a message naming it. The default never reads it at all.

.. important::

   A path with no transition -- what a brute-force shooting setup uses, in practice
   every frame in the reactive state -- has neither a state boundary nor an
   in-state side, so a position cannot be placed in it. Setting anything other
   than ``'boundary'`` for such a run raises ``ValueError``.

Bounding the Value Pass (``subsample_caps``)
--------------------------------------------

Each training round the trainer recomputes the committor on **every** reactive
frame of the ensemble (with the freshly trained network) before binning and
reweighting. That value pass grows without bound as sampling accumulates — and
with several ligands feeding one shared trainer it can stop fitting inside a
job's walltime. Setting ``subsample_caps`` (see :doc:`parameters`) makes the
trainer run the value pass / bins / reweighting / rate estimate on a **fresh
random subsample** of the ensemble each round (capped per path category), while
``fit`` still trains on the full ensemble:

.. code-block:: python

    subsample_caps = {'shot': 100, 'free': 500, 'in_state': 5000}
    # multi-system: a single dict (all systems) or a list of dicts, one per system

Selection is uniform within each category, so the reweighting remains a
consistent rate estimate (generous caps keep the variance low); in-state-only
paths carry zero reweight so dropping them never biases the rate. With
``nbins == 0`` no adaptive bins are generated, but the (capped) value pass and
per-round rate estimate still run. ``None`` (default) disables subsampling
entirely.

Sweep Mode (Committor Validation)
---------------------------------

The launcher and worker support a validation-oriented sweep mode through
``reactive_region_mode='sweep'`` (or calling :meth:`aimmd.Worker.shoot` with
``sweep=True``). In this mode the workers repeatedly shoot unbiased trajectories
from a fixed set of frames, giving a brute-force estimate of the committor for
validating the learned model.

Sweep workers coordinate purely through the shared filesystem — there is no
trainer and no central coordinator. Each worker writes only into its own
``sweep{t}{k}`` folder and tags every committed shot with the validation frame it
was launched from (a ``...sweep_frame.npy`` sidecar). Before each shot a worker
reads *all* workers' committed shots (plus their in-flight markers) and shoots
whichever frame is currently **least covered across all workers**. This
round-robin-by-coverage keeps the per-frame shot counts flat no matter how
unevenly the workers progress.

Termination is governed by a **global** target rather than a per-worker step cap:
``create_job`` / ``run`` interpret the configured ``nsteps`` as each worker's
share of the total, so ``n`` sweep workers run until the *combined* committed
shot count reaches ``n * nsteps``. A finished campaign can be extended simply by
raising the target and resubmitting — only the deficit is shot, and the extra
shots land on the least-covered frames.

Aggregation via :meth:`aimmd.PathEnsemble.shooting_results` attributes each shot
to its tagged frame (falling back to positional ``i % sweep_size`` for untagged
legacy shots), and :meth:`aimmd.PathEnsemble.report_shooting_results` compares
the empirical committor against the model prediction.

.. _graph-keys:

Graph Keys for Graph-Network Runs (``descriptor_cache``)
--------------------------------------------------------

A graph network never reads coordinates directly: ``descriptor_transform``
turns each frame's coordinate row into a graph, which
:mod:`aimmd.network.graph_utils` builds once and keeps in a SQLite graph cache
under the SHA-256 key of the row. By default AIMMD still caches every row in
``<traj>.descriptors.npy`` -- about 680 kB per frame for a solvated protein,
hundreds of GB per campaign -- and reads and hashes those rows again whenever
it needs the graphs.

With

.. code-block:: python

    descriptor_cache = 'graphkeys'

AIMMD keeps, next to every trajectory, ``<traj>.graphkeys.npy`` instead: the
32-byte graph-cache key of each frame, an ``(n_frames, 32)`` ``uint8`` array
in which an all-zero row means "not computed yet" (32 B per frame, 21 MB for
650k frames). Shooting-point selection, TPS acceptance, the trainer's value
passes and ``fit`` receive key rows and look the graphs up by key; no
``<traj>.descriptors.npy`` is read or written. Coordinates are decoded from the
trajectory only to key new frames and to rebuild a graph that is missing. The
keys are exactly the keys the graph cache already uses (the computation is
pinned in :mod:`aimmd.core.graphkey`), so switching an existing campaign
rebuilds no graph, and switching back is a one-line change.

Setting it up
~~~~~~~~~~~~~

The switch takes effect together with a ``descriptors_function``. It needs a
``descriptor_transform`` and a ``values_function`` that accept key rows as well
as coordinate rows, as functions built on
:func:`aimmd.network.graph_utils.process_descriptors_pyg` do -- the usual GNN
params need no other change:

.. code-block:: python

    from aimmd.network.graph_utils import (
        atom_coordinate_descriptors_function, init_db, process_descriptors_pyg)

    conn = init_db('graphs_cache.sqlite')

    def descriptor_transform(descriptors):         # coordinate or key rows
        return process_descriptors_pyg(
            descriptors, mdanalysis_universe=universe,
            system_selection=SYSTEM, environment_selection=ENVIRONMENT,
            cutoff=CUTOFF, conn=conn, atom_types=ATOM_TYPES)['data_list']

    def descriptors_function(trajectory):          # frames -> coordinate rows
        return atom_coordinate_descriptors_function(trajectory)

    descriptor_cache = 'graphkeys'

``descriptors_function`` still defines the rows whose hash is the key. In
graph-key runs it no longer needs to build the graphs as a side effect (the
``descriptor_transform(desc)`` call many GNN params make inside it): AIMMD
builds and stores the graphs of new frames itself, once. Keeping the call is
harmless.

``fit`` reads training batches as key rows; ``in_memory=True`` is ignored with
a notice, and ``graphs=False`` raises.

At every ``Params.load`` the frames of the initial paths are keyed (and their
graphs built if missing), and two checks run:

- if the graph cache already held graphs but none of the initial frames'
  graphs, the keys cannot be the ones the cache was filled with (a different
  ``descriptors_function``, atom selection or trajectory decoder), and loading
  stops with an explanation;
- ``values_function`` must return one value for one key row, and must raise
  ``GraphCacheMiss`` for an all-zero key rather than read the key bytes as
  numbers.

What happens where
~~~~~~~~~~~~~~~~~~

- **Ingestion** (shooting and free workers): new frames are decoded, keyed,
  and only the graphs the cache lacks are built and stored; the 32-byte rows
  are written.
- **Registration** of a shot copies the key rows of its halves, like the
  states.
- **Trainer**: at the start of a round, after the key ledger, a verify pass
  checks every frame of the ensemble against the cache (one index scan) and
  rebuilds missing graphs in bulk. It logs ``graph keys: N frame(s) checked``
  with the frames without a key, without a graph, and repaired.
- **Misses**: a frame whose key row is zero, or whose graph is missing, is
  decoded from its trajectory, its graph rebuilt and its key row rewritten;
  the call is then retried once with the rebuilt graphs, and if the cache still
  misses, the frames are evaluated from coordinates as without graph keys. A
  cache miss never stops a worker or the trainer.

Environment variables:

``AIMMD_GRAPHKEYS_VERIFY``
   ``0`` skips the trainer's round-start verify pass (default: on).
``AIMMD_GRAPHKEYS_SKIP_SELFTEST``
   ``1`` skips the ``Params.load`` check against the graph cache (for a cache
   that was filled for other trajectories) and the all-zero-key check of
   ``values_function``.
``AIMMD_STAGE_DEADLINE``
   Seconds the trainer may spend copying the graph cache to its ``/dev/shm``
   replica (default ``max(300 s, cache size / 50 MB/s)``); see
   :mod:`aimmd.network.shm_cache`.

Working with a graph-key run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trajectories of a graph-key run have no descriptor file, and ``path.descriptors``
raises an error that says so. Use the coordinates, or look the graphs up by
key:

.. code-block:: python

    params = aimmd.Params.load('params.py')
    path = params.pathensemble('run1')[0]

    coordinates = path.coordinates      # (n_frames, 3 * n_atoms), from the trajectory
    keys = path.graphkeys               # (n_frames, 32) uint8
    graphs = params.descriptor_transform(keys)  # from the graph cache
    values = params.values_function(keys)

``path.compute(params.descriptors_function)`` recomputes the coordinate rows
exactly as AIMMD keyed them.

Switching a running campaign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All steps that touch run folders run **between two jobs**, while no worker or
trainer runs.

1. **Deploy AIMMD first**, with the params unchanged (``'npy'``), and run the
   tests on the cluster. An older AIMMD silently ignores the
   ``descriptor_cache`` line (``Params`` keeps only the fields it knows) and
   stays in ``'npy'`` mode: safe, but nothing is gained.
2. **Backfill the keys** on a CPU node of the cluster that runs the campaign:

   .. code-block:: bash

       python -m aimmd.network.graph_keys_cli backfill --run run1 \
           --db graphs_cache.sqlite -j 64 --verify-npy 5 --report backfill.json

   It writes ``<traj>.graphkeys.npy`` for every trajectory with a
   ``<traj>.states.npy`` (initial paths, chain paths, the halves of shots in
   flight, free parts), decoding each frame from the trajectory; offsets are
   built once per trajectory, chunks of frames are keyed in parallel, and each
   file is written atomically. It only reads the graph cache, and reports how
   many keys have no graph there (expect about 0). ``--verify-npy K`` compares
   ``K`` keys per trajectory with the keys of its old descriptor rows, which
   checks that the trajectories decode on this machine exactly as they did when
   the rows were written. Exit status 0 means all is well. Decoding and keying
   take about 4 ms of CPU per frame for a 57k-atom system (about 45 CPU-min
   for 650k frames), so reading the trajectories once usually bounds the wall
   time.

   The backfill is optional: without it, the trainer's first round keys the
   whole ensemble itself (about 6 ms per frame, serially) and resumed workers
   key their halves in flight. It assumes the all-atom rows of
   ``atom_coordinate_descriptors_function``; a run with another
   ``descriptors_function`` relies on that lazy keying instead (the backfill
   would report almost every key as missing). A multi-system run keeps one
   graph cache per system: backfill each system folder with its own cache,
   ``--run run1/<system_id> --db <that system's cache>``.
3. **Switch**: add ``descriptor_cache = 'graphkeys'`` to the params file, and
   regenerate the derived params file and job script where the jobs run
   (``Params.load`` writes local paths into them).
4. **Resubmit and validate** over a few rounds: the round-start log should
   report 0 frames without a key and next to no missing graphs, the first value
   pass should reproduce the values of unchanged frames, and repairs should stay
   rare.
5. **Delete the descriptor files** once validated:
   ``find run1 -name '*.descriptors.npy' -delete``. Keep them until then:
   switching back after deleting them means recomputing about 680 kB per frame.
6. **Collect garbage** in the graph cache now and then, between jobs. It only
   grows: graphs of frames no path holds any more stay. First make every key
   file complete, then look, then delete:

   .. code-block:: bash

       python -m aimmd.network.graph_keys_cli backfill --run run1 \
           --db graphs_cache.sqlite -j 64 --only-missing
       python -m aimmd.network.graph_keys_cli gc --run run1 --db graphs_cache.sqlite
       python -m aimmd.network.graph_keys_cli gc --run run1 --db graphs_cache.sqlite \
           --apply --vacuum

   ``gc`` deletes every graph that no key file under the given runs references,
   so give it **every** run that uses the cache. It refuses while a trajectory
   lacks a complete key file, for a ``--run`` that is not a run folder (no
   ``initial*/`` folder; a multi-system run gives each system's folder), for
   a key file it cannot read, and when more than 1 % of the referenced keys
   have no graph (the cache of another run). ``--apply`` also refuses while
   another process has the cache open, and then holds it exclusively until
   it is done. It cannot tell that another run uses the cache: that is up
   to you. ``python -m aimmd.network.graph_keys_cli verify``
   reports key files and missing graphs without writing anything.

**Switching back** is the line ``descriptor_cache = 'npy'`` (or removing it).
Frames keyed in the meantime have no descriptor file; workers compute it
before their value passes and the trainer at its round start, from the
trajectories, bitwise as before -- the same rows, so the same keys, and no
graph is rebuilt. That costs a decode and about 680 kB written per frame, so
expect slow first rounds after a long graph-key period.
