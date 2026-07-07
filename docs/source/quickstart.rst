Quickstart
==========

This page walks through a minimal AIMMD run end to end. For a complete,
runnable example see the :doc:`tutorials/index` (the 1-D toy notebook runs with
no GROMACS or GPU); for the full meaning of every setting see
:doc:`parameters`.

Install
-------

.. code-block:: bash

   pip install aimmd-lab

AIMMD also needs a working GROMACS (``gmx`` or ``gmx_mpi`` on ``PATH``) for real
molecular systems; the toy engine used in the tutorials needs nothing extra. See
:doc:`installation` for details, conda environments, and optional
graph-neural-network dependencies.

The two files that define a run
-------------------------------

An AIMMD run is described by a single **parameter file** plus a short **launch
script**.

``params.py`` collects everything scientific about the run — how to label
states, how to featurize frames, the committor network, the engine, and the
sampling controls:

.. code-block:: python

   # params.py  (assigned module-level names are collected into Params)
   import aimmd

   states = 'ARB'          # reactant A, reactive region R, product B

   def states_function(trajectory):
       """Return a per-frame state label ('A', 'R', or 'B')."""
       ...

   def descriptors_function(trajectory):
       """Return an (n_frames, n_features) array fed to the network."""
       ...

   network = aimmd.network.placeholder(...)   # any torch.nn.Module

   engine = 'gromacs'      # or 'toy' for the tutorials
   topology = 'conf.gro'
   gmx_mdrun = 'gmx mdrun'
   gmx_grompp = 'gmx grompp -f run.mdp -p topol.top'

   initial_paths = ['transition0.trr']   # one or more seed transition paths

Only ``states_function`` is strictly required; everything else has a sensible
default (for example ``fit`` defaults to :func:`aimmd.network.fit.fit`, and
``values_function`` defaults to evaluating the network on the descriptors). The
full field-by-field reference is in :doc:`parameters`.

Load, launch, analyze
---------------------

.. code-block:: python

   import aimmd

   # 1. Load the parameter file.
   params = aimmd.Params.load('params.py')

   # 2. Attach it to a working directory.
   launcher = aimmd.Launcher(params, 'run1')

   # 3. Launch workers locally: 5 shooting workers, 1 free-A, 1 free-B,
   #    stopping once 25 000 frames have been collected.
   launcher.run(5, 1, 1, nframes=25000)

The shooting workers select committor-guided shooting points and generate new
paths; the free workers add unbiased state-local sampling; and a training worker
continuously refits the committor model, recomputes values, and refreshes the
adaptive bins. They communicate only through the run directory, so the run
survives interruptions and can be resumed.

When enough data has accumulated, reload the ensemble and analyze it:

.. code-block:: python

   ensemble = params.pathensemble('run1')

   ensemble.report()             # summary of path counts and types
   ensemble.reweight()           # equilibrium reweighting -> free energies / rates

For a rejection-free (``rfps``) run, reweighting and bin/density adaptation also
happen automatically at the end of every training round, so intermediate
estimates are available while the run is still going.

On a cluster, replace :meth:`~aimmd.Launcher.run` with
:meth:`~aimmd.Launcher.create_job` to emit a SLURM script that launches the same
workers with ``srun`` (remember to put ``gmx`` on ``PATH`` inside the job
script).

Next steps
----------

- :doc:`tutorials/index` — complete, runnable notebooks.
- :doc:`workflow` — the run loop in more depth.
- :doc:`parameters` — every configuration field.
- :doc:`advanced` — multi-system runs, biased dynamics, subsampling, and validation.
