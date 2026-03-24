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

The same worker can handle more shooting chains sequentially, and different workers also run simultaneusly. Having many chains and free simulations is important for regularizing the training set and improving the training performances: in this way, the **`train`** worker provides frequent updates to the committor model, using always the most recent available training data.

Step 5: Analyze the Resulting Path Ensemble
-------------------------------------------

Once data has been written to disk, :class:`aimmd.Params` can reconstruct the corresponding
path ensembles from the run directory. Those ensembles can then be used to:

- report path statistics,
- compute or refresh derived frame-wise quantities,
- project observables into bins,
- estimate crossing statistics,
- and reweight the sampled paths to recover equilibrium-like observables.

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

- shot chains for reactive-region sampling,
- ``chain*`` trajectory folders for the path sampling trajectories,
- ``free*`` trajectory folders for unbiased simulations,
- ``network*.h5`` snapshots for the learned model,
- ``bins*.npy`` and ``densities*.npy`` for adaptive sampling state,
- and the cached ``states``, ``descriptors``, and ``values`` arrays used during
  training and analysis, in the appropriate subfolders.
