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
