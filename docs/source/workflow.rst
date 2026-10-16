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
- and the cached ``states``, descriptors (series ``descriptors_series``,
  ``'descriptors'`` by default), and ``values`` arrays used during training and
  analysis, in the appropriate subfolders.

Restarts and Checkpoints
------------------------

Workers can be stopped at any time (walltime, ``scancel``, a node failure) and
started again with the same command: each worker picks up where the files on
disk left off. GROMACS segments resume from their checkpoint ``<deffnm>.cpt``,
by appending to the trajectory of a shooting half (``back.xtc``, ``forw.xtc``)
or by writing the next ``.partNNNN`` file of a free trajectory.

**AIMMD always rolls a run back to the checkpoint it resumes from.** The
per-frame caches next to a trajectory (``<trajectory>.states.npy``, descriptor
series, ``values``, ``bias``, ...) hold one row per frame, and AIMMD reuses a
row for as long as the frame index exists. When the checkpoint is older than the
last frame on disk, GROMACS simulates the frames after it again, under the same
indices, so before ``mdrun`` resumes the worker makes the files agree with the
checkpoint, however far back that is:

- shooting halves: the trajectory is cut to the length the checkpoint recorded
  (as ``mdrun`` is about to do itself), and every per-frame cache of it (and the
  half's PLUMED ``<deffnm>_COLVAR``) is cut to the frames that are left;
- free trajectories: every part after the checkpoint's part is moved, with all
  its files (caches, PLUMED ``_COLVAR`` slice, log, energies), into a new folder
  ``free<state>/rolled_back_<UTC time>/``, and its rows are moved out of the
  cumulative PLUMED ``COLVAR``. Nothing is deleted, and AIMMD no longer sees
  these parts.

Each rollback is written to the worker log, in an entry that starts with
``!!! CHECKPOINT ROLLBACK`` and says which checkpoint was used (part, step,
time), how many frames and how much simulated time were dropped, and which
caches were cut, for example (one line in the log)::

    !!! CHECKPOINT ROLLBACK run1/chainR0/back.xtc: mdrun resumes from back.cpt
    (part 1, step 72829100, t = 145658.2 ps); dropped the 70506 frames written
    after it (t = 145660 to 850710 ps; 705.052 ns of simulated time past the
    checkpoint), back.xtc cut from 85072 to 14566 frames; caches cut to 14566
    rows: descriptors (85072), states (85072)

A restart from a checkpoint at the end of the trajectory, the normal case,
logs one short ``... nothing to roll back`` line. A job killed before
``mdrun`` could write its last checkpoint loses the few frames since then. A
trajectory that is already complete is not resumed, and not rolled back. If the
checkpoint cannot be read (``gmx dump -cp`` fails), the log says so loudly and
nothing is rolled back. The checkpoint is read with ``gmx dump -cp``, run with
the executable of ``gmx_grompp``. See :mod:`aimmd.worker.rollback`.

.. warning::

   Never replace checkpoint files by older copies. The checkpoint decides how
   far a run goes back: if ``.cpt`` files are restored from an older copy of a
   run folder -- for example by syncing a downloaded copy back to the cluster --
   every segment resumes from that older state. AIMMD then rolls the
   trajectories back accordingly and simulates the lost time again, all of it,
   without asking. When you copy a run folder back, leave out ``*.cpt`` (and the
   GROMACS ``*.log`` and PLUMED ``COLVAR`` files), or copy the trajectories and
   their caches from the same moment as the checkpoints.
