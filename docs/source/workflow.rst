Workflow
========

This page describes the core end-to-end run loop. Advanced modes
(multi-system runs, biased dynamics, subsampling, kinetics-convergence, and
sweep-mode validation) are covered in :doc:`advanced`.

End-to-End AIMMD Run
--------------------

An AIMMD run is organized around a parameter file and a working directory.
Conceptually, the workflow looks like this:

1. Define a ``params.py`` file.
2. Seed the run with one or more initial transition paths.
3. Launch workers that perform ``shoot``, ``free``, and ``train`` tasks.
4. Let the training worker update the committor model, bins, and densities.
5. Reload the resulting path ensemble for reporting, projection, and
   reweighting (for ``rfps`` runs, reweighting is also done automatically at the
   end of each trainer iteration).

Step 1: Define the System and Analysis Pipeline
-----------------------------------------------

The user-facing entry point is :class:`aimmd.Params`. A parameter file typically
defines:

- ``states_function`` to classify frames into state labels,
- ``descriptors_function`` and an optional ``descriptor_transform`` for more
  complex featurization pipelines (e.g. graph-based),
- a neural network object,
- a ``fit`` function or the default AIMMD network trainer,
- engine configuration such as GROMACS commands or a toy-engine callback,
- and initial paths used to seed the run.

The retinal test parameters in ``tests/retinal/params.py`` are a good example of
the expected shape of a full configuration file; see also the
:doc:`tutorials/index` for configuration on toy systems, and :doc:`parameters`
for every field.

Step 2: Seed the Run with Initial Paths
---------------------------------------

AIMMD expects one or more initial transition paths. They are used to:

- initialize the path ensemble,
- seed shooting workers,
- and provide starting data for early training rounds.

Internally these are normalized into :class:`aimmd.Path` / :class:`aimmd.PathEnsemble`
objects by the parameter helpers. In practice they may come from unbiased MD, a
previous enhanced-sampling run, higher-temperature simulations, or any other
method that produces transitions.

Step 3: Launch Workers
----------------------

Workers are the execution units in AIMMD. Each worker runs a single task type:

``shoot``
   Select a shooting point, run backward and forward dynamics, assemble the new
   path, and add it to the shooting chain.

``free``
   Run unbiased simulations around a chosen state to collect internal segments
   and short excursions.

``train``
   Load the current ensemble, fit the committor network, recompute values,
   update bins, estimate densities, and persist the resulting artifacts.

Workers can be run one at a time through :class:`aimmd.Worker` or as a
coordinated group through :class:`aimmd.Launcher`.

Step 4: Adaptive Sampling Loop
------------------------------

The scientific logic of AIMMD is encoded in the interaction between the three
worker types:

- shooting workers produce new reactive-region data,
- free workers maintain state-local data and provide additional from-state
  excursion segments,
- training workers translate the current ensemble into updated model guidance.

When ``chain_type='rfps'``, this loop implements the rejection-free path
sampling logic described in the latest reference paper. When
``chain_type='tps'``, AIMMD instead follows a more traditional TPS-style
accept/reject chain.

The same worker can handle one or more shooting chains sequentially, and
different workers also run simultaneously. Constantly updating the path ensemble
with many diverse shooting paths and free simulations regularizes the training
set and improves training. You tune this through the number of workers,
``nchains_per_worker``, and ``selection_pool_size``. In this way the ``train``
worker provides frequent model updates using the most recent available data.

Step 5: Analyze the Resulting Path Ensemble
-------------------------------------------

Once data has been written to disk, :class:`aimmd.Params` can reconstruct the
path ensemble from the run directory. That ensemble can then be used to:

- report path statistics,
- compute or refresh derived frame-wise quantities,
- project observables into bins,
- estimate crossing statistics,
- and reweight the sampled paths to recover equilibrium-like observables.

See :doc:`advanced` for validating the learned committor (sweep mode) and for
checking that estimated rates have converged.

Important Output Files
----------------------

A typical run directory contains:

- shot chains and pool logs for reactive-region sampling,
- ``chain*`` trajectory folders for the path-sampling trajectories,
- ``free*`` trajectory folders for unbiased simulations,
- ``network*.h5`` snapshots for the learned model,
- ``bins*.npy`` and ``densities*.npy`` for adaptive sampling state,
- and the cached ``states``, ``descriptors``, and ``values`` arrays used during
  training and analysis, in the appropriate subfolders.
