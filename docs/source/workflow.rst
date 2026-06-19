Workflow
========

End-to-End AIMMD Run
--------------------

An AIMMD run is organized around a parameter file and a working directory.
Conceptually, the workflow looks like this:

1. Define a ``params.py`` file.
2. Seed the run with one or more initial transition paths.
3. Launch workers that perform ``shoot``, ``free``, and ``train`` tasks.
4. Let the training worker update the committor model, bins, and densities.
5. Reload the resulting path ensemble for reporting, projection, and
   reweighting (reweighting is also done automatically at the end of each trainer iteration).


Step 1: Define the System and Analysis Pipeline
-----------------------------------------------

The user-facing entry point is :class:`aimmd.Params`. A parameter file typically
defines:

- ``states_function`` to classify frames into state labels,
- ``descriptors_function`` and optional ``descriptor_transform`` for more complex featurization pipelines (eg. graph-based),
- a neural network object,
- a ``fit`` function or the default AIMMD network trainer,
- engine configuration such as GROMACS commands or a toy-engine callback,
- and initial paths used to seed the run.

The retinal test parameters in ``tests/retinal/params.py`` are a good example
of the expected shape of a full configuration file; see also the example notebook for configuration on toy systems.

Step 2: Seed the Run with Initial Paths
---------------------------------------

AIMMD expects one or more initial transition paths. They are used to:

- initialize the path ensemble,
- seed shooting workers,
- and provide starting data for early training rounds.

Internally these are normalized into :class:`aimmd.Path` or :class:`aimmd.PathEnsemble` objects by
the parameter helpers. In practice: they may come from unbiased MD, a previous enhanced sampling run, simulations at higher temperature,
or any other method to produce paths.

Step 3: Launch Workers
----------------------

Workers are the execution units in AIMMD. Each worker runs a single
task type:

``shoot``
   Select a shooting point, run backward and forward dynamics, assemble the new
   path, and add it to the shooting chain.

``free``
   Run unbiased simulations around a chosen state to collect internal segments
   and short excursions.

``train``
   Load the current ensemble, fit the committor network, recompute values,
   update bins, estimate densities, and persist the resulting artifacts.

The repository supports running workers one by one through :class:`aimmd.Worker` or
as a coordinated group through :class:`aimmd.Launcher`.

Step 4: Adaptive Sampling Loop
------------------------------

The scientific logic of AIMMD is encoded in the interaction between the three
worker types:

- shooting workers produce new reactive-region data,
- free workers maintain state-local data and provide additional from-state excursion trajectory segments,
- training workers translate the current ensemble into updated model guidance.

When ``chain_type='rfps'``, this loop implements the rejection-free path
sampling logic described in the latest reference paper. When
``chain_type='tps'``, AIMMD instead follows a more traditional TPS-style
accept/reject chain.

Step 5: Analyze the Resulting Path Ensemble
-------------------------------------------

Once data has been written to disk, :class:`aimmd.Params` can reconstruct the corresponding
path ensembles from the run directory. Those ensembles can then be used to:

- report path statistics,
- compute or refresh derived frame-wise quantities,
- project observables into bins,
- estimate crossing statistics,
- and reweight the sampled paths to recover equilibrium-like observables.

Kinetics Convergence Analysis
-----------------------------

After a completed AIMMD run you may want to verify that the estimated rate
constants are stable with respect to the amount of training data used.
:meth:`aimmd.Worker.kinetics_convergence` provides a built-in convergence
analysis directly on an existing run directory.

The method iterates over a list of *fractions* (defaulting to
``[0.2, 0.4, 0.6, 0.8, 1.0]``). For each fraction ``f`` it:

1. Sub-samples the path ensemble: the first ``round(N * f)`` paths are taken
   from each shooting chain, and the first ``round(N * f)`` frames from each
   free-simulation trajectory.  Sub-sampling is done **per source** (chain /
   free trajectory) so that all sources contribute the same fraction.
2. Retrains the committor network from scratch on the sub-sampled data.
3. Reweights the sub-sampled ensemble and estimates ``k12`` and ``k21``.
4. Saves the trained network to a per-fraction checkpoint file.

The results are returned as a structured NumPy array with fields
``fraction``, ``k12``, and ``k21``, and are saved to a ``.npy`` file in the
worker directory.

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

**Network checkpoint naming**

By default the network trained on each fraction is saved next to the run
directory's normal network file, following the pattern::

    {directory}/network{states}.kcv{fraction_pct:03d}.h5

For example, with ``states='ARB'`` and ``fraction=0.40`` the file is
``run_directory/networkARB.kcv040.h5``.  The pattern can be customised via
the ``network_save_pattern`` keyword:

.. code-block:: python

    results = worker.kinetics_convergence(
        network_save_pattern='checkpoints/net_{states}_f{fraction:.2f}.h5',
    )

The following placeholders are available in the pattern string:

* ``{directory}`` — the worker directory.
* ``{states}`` — the sorted state label string (e.g. ``'ARB'``).
* ``{fraction}`` — the fraction as a float (e.g. ``0.4``).
* ``{fraction_pct}`` — the fraction as an integer percentage (e.g. ``40``).

Pass ``network_save_pattern=None`` to skip saving networks entirely.

You can also override training hyperparameters for a faster exploratory run::

    results = worker.kinetics_convergence(
        fractions=[0.25, 0.5, 0.75, 1.0],
        epochs=200,
        save_file='run_directory/kcv_quick.npy',
    )

.. note::

   ``kinetics_convergence`` saves and restores the **original** trained
   network after the analysis finishes, so the worker is left in the same
   state it was in before the call.  Per-fraction checkpoint files remain on
   disk and can be loaded via ``torch.load`` or
   :meth:`aimmd.Params.update_network`.  Temporary ``*.kcv.npy`` cache files
   written during the analysis are removed automatically.

Multi-System (Multi-Ligand) Runs
--------------------------------

A single params file can drive **several chemical systems at once** (for example
two ligands binding the same host) and train **one shared committor model** that
takes a graph/descriptor from *either* system and returns its committor. This is
fully backward compatible: it is enabled only when ``multi_system=True``;
otherwise everything behaves exactly as the single-system workflow above.

**Enabling multi-system mode.** Set ``multi_system=True`` and provide one entry
per system for the fields that are otherwise single-valued:

.. code-block:: python

    multi_system = True
    multi_system_share_network = True            # one shared network (see below)
    system_ids   = ['G2', 'G4']                  # per-system labels
    topology     = ['G2.gro', 'G4.gro']          # one topology per system
    initial_paths = [['G2_tp.trr'], ['G4_tp.trr']]   # one group per system
    atom_types   = ['H', 'C', 'N', 'O', 'F', 'NA', 'P', 'S', 'CL', 'BR', 'I']

The ``system_ids`` name the per-system subfolders ``<run>/<system_id>/`` and index
the per-system entries of the list-valued fields. If ``system_ids`` is left
empty it defaults to ``['0', '1', ...]``.

**The ``system_id`` keyword.** In multi-system mode the user data functions
receive an extra ``system_id`` keyword so a single function can encode
per-ligand differences (e.g. different state cutoffs or atom selections):

.. code-block:: python

    def states_function(trajectory, system_id=None):
        cutoff = 4.5 if system_id == 'G2' else 4.1   # per-ligand state boundary
        ...

``system_id`` is passed **only if a function declares it** (detected via
:func:`aimmd.core.utils.accepts_system_id`), so existing single-system functions
keep working unchanged. The same applies to ``descriptors_function``,
``values_function`` and ``descriptor_transform``.

**Shared graph encoding.** For a single network to consume graphs from several
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
* **False** — each system trains its own network (``run1/<system_id>/networkARB.h5``)
  with its own trainer. The params flag ``trainers_share_gpu`` (default ``True``)
  controls whether those trainers share one GPU or are spread across GPUs.

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

**Kinetics convergence** works with a shared model: the per-fraction retrain uses
the list of per-system ensembles and the result array gains a ``system`` field
(one row per ``(fraction, system)``).

Biased (OPES) multi-system runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-system runs support **in-state biases** (e.g. a frozen OPES_METAD that
flattens each ligand's bound well) exactly as single-system runs do — set
``record_bias = True`` and a ``bias_function`` (see :ref:`bias-recording`). The
bias enters GROMACS through the per-system ``gmx_mdrun`` string, which is already
list-valued in multi-system mode, so each system gets its own PLUMED input:

.. code-block:: python

    record_bias = True
    bias_source = 'file'                     # read each frame's bias from COLVAR
    gmx_mdrun = ['gmx mdrun -plumed /abs/G2/plumed.dat',
                 'gmx mdrun -plumed /abs/G4/plumed.dat']
    bias_reactive_threshold = [0.5, 0.3]     # per-system (scalar also allowed)

The trainer builds the per-frame bias cache **per system** (forwarding
``system_id`` to ``bias_function`` when its signature accepts it), runs the
reactive-region bias check against each system's
``bias_reactive_threshold_of(system_id)``, and prints **per-system**
Tiwary-Parrinello bias-reweighted rates next to the raw ones. Kinetics
convergence fills the ``k12_rw`` / ``k21_rw`` columns per system.

.. important::

   Each system's PLUMED ``PRINT STRIDE`` (COLVAR output) must equal that system's
   ``nstxout-compressed`` so COLVAR row *i* corresponds to trajectory frame *i*;
   otherwise the cached per-frame bias is misaligned with the trajectory.

.. note::

   The first release of multi-system support targets ``chain_type='rfps'`` with
   the committor balancing described above. LSR/MAR regularization and
   ``rescale_committor`` are not yet combined with ``multi_system`` and raise a
   clear ``NotImplementedError``; single-system runs retain full support for all
   of them.

Sweep Mode
----------

The launcher and worker also support a validation-oriented sweep mode through
``reactive_region_mode='sweep'`` or calling :meth:`aimmd.Worker.shoot` with
``sweep=True``. In this
mode the code cycles deterministically through a fixed frame set and repeatedly
shoots from those frames, which gives a brute-force estimate of the committor
for validating the learned model. See the example notebook for details of how to use this in practice.

Important Output Files
----------------------

A typical run directory contains:

- shot chains and pool logs for reactive-region sampling,
- ``chain*`` trajectory folders for the path sampling trajectories,
- ``free*`` trajectory folders for unbiased simulations,
- ``network*.h5`` snapshots for the learned model,
- ``bins*.npy`` and ``densities*.npy`` for adaptive sampling state,
- and the cached ``states``, ``descriptors``, and ``values`` arrays used during
  training and analysis, in the appropriate subfolders.
