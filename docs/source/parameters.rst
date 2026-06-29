Parameters
==========

Params as the Run Contract
--------------------------

:class:`aimmd.Params` is the central contract between the scientific definition of a
problem and the code that executes it. Nearly every other high-level component
depends on it:

- :class:`aimmd.Launcher` reads it to build execution plans and directory structures.
- :class:`aimmd.Worker` reads it to run simulations and training rounds.
- :class:`aimmd.PathEnsemble` loaders use it to reconstruct runs from disk.

The implementation is mixin-based, but users should treat ``Params`` as one
coherent object.

Required and Commonly Overridden Inputs
---------------------------------------

In practice, the most important inputs are:

``states_function``
   Required. Maps frames to state labels such as ``A``, ``R``, and ``B``.

``initial_paths``
   Initial transition paths that seed the run.

``network``
   Torch model used to predict the committor-like value.

``descriptors_function`` and ``descriptor_transform``
   Optional feature pipeline before the network is evaluated.

``values_function``
   Optional, if evaluating the network forward pass requires any special care. If not given, will default to network(descriptors).

``fit``
   Training hook. If omitted, AIMMD uses the default trainer in
   :func:`aimmd.network.fit.fit`.

``engine`` and engine-specific settings
   Choose between gromacs and the toy engine and provide the relevant command
   strings or callbacks.

State and Region Conventions
----------------------------

The ``states`` string defines the label ordering used throughout the code.
The default is ``'ARB'``:

- the first label is the reactant-like metastable state,
- the middle label is the reactive region,
- the last label is the product-like metastable state.

This convention is used everywhere: shooting logic, free simulations, path
typing, training labels, and reweighting.

Sampling Controls
-----------------

Several parameters directly control how AIMMD explores path space:

``chain_type``
   ``'rfps'`` for rejection-free path sampling or ``'tps'`` for TPS-style
   acceptance.

``selection_pool_size``
   Number of candidate paths used when selecting the next shooting point.

``at_least_one_transition_in_pool``
   Optional heuristic that may improve sampling in early rounds by ensuring that at least one path in the selection pool is reactive.

``always_select_inside_the_bins``
   Do not select shooting points from paths entirely outside the selection
   bins range.

``nbins``, ``cutoff_min``, ``cutoff_max``, ``marginal_bins``
   Define how value space is discretized for adaptive sampling and density
   estimation.

``density_adjustment`` and ``shared_density_adjustment``
   Optional heuristics that may improve convergence by reweighting the density
   of shooting points in each bin.

When launching AIMMD, the ``nchains_per_worker`` parameter controls how many
independent shooting chains are run in parallel on each worker. This can be used
to improve sampling efficiency and reduce correlation between chains when
``selection_pool_size=1``.

Multi-System (Multi-Ligand) Parameters
---------------------------------------

These fields turn one params file into a multi-system run that trains a single
shared committor model across several chemical systems. They all default to the
single-system behavior, so existing params files are unaffected.

``multi_system``
   Bool, default ``False``. Enables multi-system mode. When on, the per-system
   fields below become lists (one entry per system), each system runs in its own
   subfolder ``<run>/<system_id>/``, and the user data functions receive a
   ``system_id`` keyword (passed only if their signature accepts it).

``multi_system_share_network``
   Bool, default ``False``. If ``True``, one shared network is trained by a
   single trainer that hands ``fit`` a *list* of per-system PathEnsembles
   (pooled balanced, ``1/N`` per system per bin); the shared network is stored at
   the run root. If ``False``, each system trains its own network with its own
   trainer.

``system_ids``
   List of per-system labels (e.g. ``['G2', 'G4']``); they name the per-system
   subfolders and index the list-valued fields. Defaults to ``['0', '1', ...]``.

``atom_types``
   Fixed, ordered list of MDAnalysis atom-type strings defining a *shared* one-hot
   graph node encoding (e.g.
   ``['H','C','N','O','F','NA','P','S','CL','BR','I']``). This is what lets one
   graph network consume graphs from multiple systems. ``None`` (default) keeps
   the legacy per-universe encoding.

``trainers_share_gpu``
   Bool, default ``True``. When training separate networks (share OFF), controls
   whether the per-system trainers bind the same GPU or distinct GPUs.

In multi-system mode ``topology`` and ``initial_paths`` accept lists (one
topology file per system; one *group* of initial paths per system), and the
per-run worker counts ``n``/``n1``/``n2`` apply per system. See
:doc:`workflow` for the full multi-system workflow and the example notebook
``examples/notebooks/2_multi_system.ipynb``.

.. _bias-recording:

Bias Recording (OPES / PLUMED)
------------------------------

For runs that apply an **in-state bias** during dynamics (e.g. a frozen
OPES_METAD that flattens a bound well), AIMMD records the per-frame bias and
recovers unbiased kinetics with the Tiwary-Parrinello correction. All of these
default to *off*, so unbiased runs are unaffected.

``record_bias``
   Bool, default ``False``. When ``True``, the per-frame bias is cached as
   ``<traj>.bias.npy`` and the trainer prints bias-reweighted rate estimates
   ``k = 1 / Σ(wᵢ·Lᵢ·γᵢ)`` with ``γᵢ = ⟨exp(bias)⟩`` per path.

``bias_function`` / ``bias_source``
   The callable that returns the per-frame bias in ``kT``. With
   ``bias_source='reader'`` it is called ``bias_function(reader)`` (toy /
   position-based); with ``bias_source='file'`` it is called
   ``bias_function(fname)`` and reads the associated PLUMED COLVAR file. In a
   multi-system run a ``system_id`` keyword is forwarded when the signature
   accepts it; the bias itself usually enters GROMACS through the per-system
   ``gmx_mdrun`` string (``gmx mdrun -plumed <system>/plumed.dat``).

``bias_reactive_threshold``
   Float, default ``0.5``. Maximum acceptable mean ``|bias|`` (in ``kT``) inside
   the reactive region R (the Tiwary-Parrinello assumption). In multi-system mode
   this may be a single float (applied to every system) or a **list**, one per
   ``system_ids`` entry.

.. important::

   When biasing with PLUMED, each system's ``PRINT STRIDE`` (COLVAR output stride)
   must equal that system's ``nstxout-compressed`` so that COLVAR row *i* lines up
   with trajectory frame *i*; a mismatch silently misaligns the cached bias.

Value-Pass Subsampling
----------------------

Each training round the trainer recomputes the committor on every reactive frame
of the (growing) path ensemble before binning and reweighting. With several
ligands feeding one trainer this value pass can outgrow the job walltime. The
optional ``subsample_caps`` bounds it by running the value pass / bin generation
/ reweighting / rate estimate on a fresh **random subsample** of the ensemble
each round, while ``fit`` (network training) still sees the full ensemble.

``subsample_caps``
   ``None`` (default) means no subsampling — behaviour is unchanged. Otherwise a
   dict with any of:

   - ``'shot'`` — max PATHS kept *per shot-excursion direction-type*. The four
     direction-types ``sAA``, ``sAB``, ``sBA``, ``sBB`` are capped
     **independently**, so ``'shot': 100`` keeps up to ``4 * 100 = 400`` shot
     paths per system.
   - ``'free'`` — max PATHS kept *per free-excursion direction-type* (``fAA``,
     ``fAB``, ``fBA``, ``fBB`` each), so ``'free': 500`` keeps up to ``2000`` free
     paths per system.
   - ``'in_state'`` — max FRAMES kept per state (the in-A and in-B paths are kept
     until this many frames accumulate, per state).

   A missing key leaves that category uncapped. In a multi-system run this may be
   a single dict (broadcast to all systems) or a list of dicts/``None`` (one per
   ``system_ids`` entry). Pick caps generously (e.g. ``shot=100, free=500``):
   selection is uniform within each category so the reweighting stays a
   consistent rate estimate, and in-state-only paths carry zero reweight so
   dropping them never biases the rate.

Engine Integration
------------------

For ``engine='gromacs'``, the parameter object stores the command templates used
to create and run simulations, including:

- ``gmx_grompp``,
- ``gmx_mdrun``,
- ``gmx_eneconv``,
- ``gmx_mdp``,
- and topology and trajectory-format settings.

For ``engine='toy'``, the main inputs are:

- ``toy_mdrun`` for advancing the timestep,
- ``toy_slowdown`` to throttle the loop for testing and development.

Persistence and Reproducibility
-------------------------------

One unusual but important design choice is that parameters are saved back to
Python files rather than to a plain data format. This allows AIMMD to preserve
callables such as:

- state classifiers,
- descriptor functions,
- network classes,
- and custom fit hooks.

That design is why ``Params`` includes helper code for tracking Python source
and why the tests pay special attention to relative imports and reloading.

See Also
--------

For the field-by-field API, see :doc:`api/params` and :doc:`api/lowlevel/params`.
