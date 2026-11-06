Running production campaigns
============================

This page covers what a C-GAIMMD campaign needs besides its params file: the
install on a workstation and on clusters, where to put run folders, the
preflight, job scripts, monitoring, restarts, post-training and backups. What
C-GAIMMD does and how to configure it is in :doc:`cgaimmd`; the messages of the
checks below and their fixes are in :doc:`cgaimmd_troubleshooting`.

The cluster examples name two machines the method has run on, an NVIDIA GH200
cluster (JUPITER at the Jülich Supercomputing Centre) and an AMD MI210 cluster
(Goethe-HLR in Frankfurt). Module names, partitions and accounts differ between
sites and change over time: treat the lines below as patterns, and check them
against your site's documentation (``module avail``).

.. contents:: On this page
   :local:
   :depth: 1

A campaign at a glance
----------------------

A campaign folder typically holds:

.. code-block:: text

    campaign/
      params.py              the params file (network, fit, hooks, cgaimmd_parameters)
      states.py              the state definition, imported by params.py
      run.gro topol.top      the MD system (and its .itp files)
      run.mdp em.mdp         the production and minimization settings
      initial_A.xtc ...      unbiased end-state frames that seed the free simulations
      pretrained/            the initial network (networkARB.h5 and its record)
      system/                for BioEmu2: the system folder (build-system)
      job.sh                 the job script, written on the machine that runs it
      run/                   the run folder (better on local disk, see below)

and goes through these steps:

1. install AIMMD-Lab with the extras the generator needs;
2. prepare the system: topology, states, unbiased MD of both end states, for
   BioEmu2 the system folder (tutorials 4 to 6 show each step);
3. pretrain the initial network on the unbiased end-state frames;
4. write the params file and run the preflight;
5. launch: locally with the Launcher, or as a cluster job;
6. monitor the logs, the network metadata and the sidecars (the record
   ``path<n>.xtc.cgaimmd.json`` that every C-GAIMMD shot leaves next to its
   path);
7. restart after the time limit, post-train if needed, back up.

Installing
----------

Which extras a campaign needs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - extra
     - what it adds
   * - ``cgaimmd``
     - C-GAIMMD with the PaiNN flow-matching generator and the GROMACS candidate
       preparation (networkx). Python 3.10 or later, any torch AIMMD supports.
   * - ``graphs``
     - PyTorch Geometric for graph descriptors and the ``'pyg'`` edge backend
       of PaiNN. ``torch-cluster`` has no wheels on PyPI: install the wheel
       built for your torch and CUDA from https://data.pyg.org/whl/ first.
   * - ``bioemu2``
     - C-GAIMMD with BioEmu2 (bioemu2 0.1.0rc0, Python 3.12 or later, torch
       2.11 to 2.13).
   * - ``bioemu2-cuda``
     - The same with the cuEquivariance kernels the production BioEmu2 model
       needs (Linux, CUDA 13, NVIDIA GPUs).

Install with the constraints file ``devtools/constraints-bioemu2.txt``, which
pins the versions the tests ran with for the packages that matter to the
BioEmu2 adaptor (biotite, RDKit, hydra, omegaconf, openmm, numpy). It pins
neither torch nor triton: the install route chooses torch, and every CUDA build
of torch brings its own triton.

GROMACS is not installed by pip. Use your site's module or your own build
(the tests use GROMACS 2024.6, and campaigns have run with 2025.4 and
2026.1; PLUMED for biased runs), and set the
executables in the params file (``gmx_grompp``, ``gmx_mdrun``): workers
started by ``srun`` or the Launcher do not run your login shell, so a
``gmx`` that only your shell knows is not found.

A workstation with an NVIDIA GPU (x86_64, CUDA 13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    conda create -n cgaimmd python=3.13 -y
    conda activate cgaimmd
    # torch first, from the PyTorch index of your CUDA (PyPI has no CUDA build of torch)
    pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu130
    # only for graph descriptors or the 'pyg' edge backend
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
    git clone https://github.com/covinolab/AIMMD-Lab.git
    pip install -e "./AIMMD-Lab[cgaimmd,bioemu2-cuda,graphs]" -c AIMMD-Lab/devtools/constraints-bioemu2.txt

torch 2.11 is the version that BioEmu2 accepts and for which PyG publishes
``torch-cluster`` wheels; without graph descriptors, any torch from 2.11 to 2.13
works. For PaiNN only, ``[cgaimmd]`` is enough.

BioEmu2 downloads its weights from Hugging Face on first use. Fetch them once,
on a machine with internet access, into a cache with enough space (several
GB), and use the same ``HF_HOME`` in every job:

.. code-block:: bash

    export HF_HOME=/path/with/space/huggingface
    python -c "from bioemu2.checkpoints import get_checkpoint_files as g; print(g('bioemu2-preview'))"

Then check the install on the machine and in the environment of the campaign
with the tests that need the real weights or a GPU (the *weights lane*); it
fails when the weights are not in the cache or no GPU is visible:

.. code-block:: bash

    HF_HUB_OFFLINE=1 AIMMD-Lab/devtools/run_weights_tests.sh

A CPU environment for development and tests (conda-forge only, with GROMACS,
PyG and BioEmu2 on a CPU build of torch 2.13) is described by
``devtools/conda-envs/cgaimmd_env.yaml``:

.. code-block:: bash

    conda env create -f devtools/conda-envs/cgaimmd_env.yaml
    conda activate cgaimmd
    pip install -e ".[cgaimmd,bioemu2,graphs,tests]" -c devtools/constraints-bioemu2.txt

An NVIDIA GH200 cluster (example: JUPITER)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GH200 nodes are aarch64 (ARM) machines with four GPUs each. What differs from
a workstation:

- **Python comes from the site's modules**, and the virtual environment is
  built on top of it. Load the modules before you activate the environment,
  in every shell and every job: the environment's ``python`` links against
  the module's Python library and does not start without it.
- **torch** has aarch64 CUDA wheels (``torch==2.11.*`` from the ``cu130``
  index), and ``bioemu2[cuda]`` installs as on a workstation. PyG publishes no
  aarch64 wheels: build PaiNN networks with ``edge_backend='torch'`` and leave
  out the ``graphs`` extra.
- **Caches go to project space.** pip, Hugging Face, BioEmu2's own caches and
  triton write gigabytes; a small home quota fills up. Point ``HF_HOME``,
  ``PIP_CACHE_DIR`` and ``TRITON_CACHE_DIR`` into project space when you build
  the environment and in every job.
- **Compute nodes have no internet.** Fetch the BioEmu2 weights on a login
  node and set ``HF_HUB_OFFLINE=1`` in the job.
- **Tell GROMACS its thread count.** If a job step's CPU set is the whole
  node, GROMACS starts as many OpenMP threads as the node has cores and runs
  several times slower. Pass ``-ntomp`` (and ``-ntmpi 1``) in ``gmx_mdrun``,
  and let AIMMD bind each worker's CPUs (``create_job(...,
  skip_binding=False)``).

The module lines of the jobs that ran there looked like this (GROMACS with
CUDA; add the PLUMED module for biased runs):

.. code-block:: bash

    module load Stages/2026 nvidia-compilers/25.9-CUDA-13 OpenMPI/5.0.8 GCC/14.3.0
    module load GROMACS/2025.4-CUDA-13            # PLUMED/2.9.4 for biased runs
    source /path/to/project/cgaimmd_venv/bin/activate
    export HF_HOME=/path/to/project/cache/huggingface HF_HUB_OFFLINE=1

``import aimmd`` together with BioEmu2 takes one to two minutes on a compute
node; that is normal.

An AMD MI210 cluster (example: Goethe-HLR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On AMD GPUs (ROCm), PaiNN C-GAIMMD and plain AIMMD run as usual: install a ROCm
build of torch and ``[cgaimmd]``. **BioEmu2 C-GAIMMD is not supported there as
shipped**: the production BioEmu2 model uses cuEquivariance kernels, which
need CUDA. Use an NVIDIA machine for BioEmu2 campaigns.

GROMACS comes from the site's ROCm modules, for example:

.. code-block:: bash

    module use --append /path/to/site/modulefiles
    module load gromacs/v2026.1/rocm-7.13.0-nightly_plumed_libtorch_cp2k

Run one replica per GPU: several GROMACS runs on one MI210 slow each other
down far more than the sum of their parts.

Run folders on local disk
-------------------------

A running AIMMD job writes many small files (trajectory segments, per-frame
caches, locks, offsets, sidecars) and locks them for every update. On a network
file system (NFS), these locks can stall workers for minutes. Put the run
folder on a local disk of the machine that runs the job, or on the site's
parallel scratch file system, and keep the campaign folder (params, system,
pretrained network) wherever you like; the params file names the run folder
when you launch.

Two cautions for local disks:

- a ``/tmp`` or local scratch disk may be wiped at reboot, so back up the run
  folder regularly (`Backups`_);
- a run folder holds hundreds of thousands of files, and some local disks run
  out of inodes before they run out of bytes (``df -i``).

The preflight
-------------

``python -m aimmd.cgaimmd.preflight params.py`` checks a params file before
compute time is spent. It runs what a job runs first, in a scratch folder, and
prints one line per check:

1. the params load and use ``chain_type='cgaimmd'``;
2. the files they name exist (structure, mdp, topology, index file,
   ``initial_network``), and ``initial_network`` loads into the network;
3. the fit's keywords are valid (the fit checks them without training);
4. the generator can be built and can use the network, and the network's atom
   count is that of the descriptors;
5. the descriptors and the network give finite numbers on the first frames of
   the initial paths, and one tiny generation works;
6. the lambda window and the placement in the box are consistent;
7. one candidate is generated at a target of 0 and goes through the validity
   function and the whole candidate preparation, including the final check;
8. the GROMACS executables answer;
9. with a PLUMED bias: the input and the files it reads exist and are absolute
   paths, the bias is printed, and it is zero in the reactive region.

The exit status is 0 when no check failed and 1 otherwise. A rejected candidate
is a warning, because an untrained generator may produce one. Useful options:

``--run RUN``
   Use the trainer's bins of an existing run folder for the window.
``--candidates N``
   Generate N candidates instead of one.
``--no-generate``
   Skip generation and candidate preparation (a quick check of files and
   settings).
``--device DEVICE``
   Run the network there (default: ``generation_device``).
``--scratch DIR``
   Keep the scratch folder there instead of a temporary one.

Run the preflight on the machine and in the environment of the job: paths,
GROMACS executables and modules differ between machines, and the preflight
only finds what is wrong *there*. The simplest way is to let the job run it
(``create_job(..., preflight=True)``, below).

Job scripts
-----------

``Launcher.create_job`` writes a SLURM job script that starts every worker of
one or more runs with ``srun``: the shooting chains, the free simulations of
both states and the trainer. A small script next to the params file makes the
job reproducible:

.. code-block:: python

    """create_job.py: write job.sh for the C-GAIMMD campaign of this folder (run it on the cluster)."""
    import aimmd

    params = aimmd.Params.load('params.py')
    launcher = aimmd.Launcher(params, '/local/scratch/campaign/run')
    launcher.create_job(
        'job.sh', 5, 1, 1,                          # 5 shooting chains, free A, free B (+ the trainer)
        walltime=12 * 3600, ntasks_per_node=4, cpus_per_task=12, gpus_per_task=1,
        skip_binding=False,                         # each worker gets its own CPUs
        preamble=['module load Stages/2026 nvidia-compilers/25.9-CUDA-13 OpenMPI/5.0.8 GCC/14.3.0',
                  'module load GROMACS/2025.4-CUDA-13',
                  'source /path/to/project/cgaimmd_venv/bin/activate',
                  'export HF_HOME=/path/to/project/cache/huggingface HF_HUB_OFFLINE=1'],
        preflight=True)

The account, partition and other ``#SBATCH`` lines come from the params'
``slurm_header`` (for example ``'#SBATCH --account=<project>\n#SBATCH
--partition=<gpu partition>'``). The C-GAIMMD-related keywords:

``preamble``
   Shell lines written right after the SLURM header: modules, the environment,
   cache locations, ``HF_HUB_OFFLINE``.
``preflight``
   ``True`` runs ``python -m aimmd.cgaimmd.preflight <params> --run <run folder>``
   for every C-GAIMMD run before any worker starts; the job stops if a check
   fails. A string adds options, for example ``'--candidates 2'``.
``guard_run_folders`` (on by default)
   The job writes its SLURM job id into ``<run folder>/aimmd_job.txt`` and does
   not start while another job that claimed the folder is still running. Two
   jobs in one run folder would write the same chains, caches and network
   files.

Generate the job script on the machine that runs it, after loading the job's
modules and environment: AIMMD writes the Python interpreter, the worker
script and the GROMACS executables it finds into the job script and into the
copy of the params file it saves in the run folder (``params1.py``), and a
script written on another machine points at the wrong files. Do not copy
``job.sh`` or ``params*1.py`` from one machine to another.

When any worker exits, the job ends all others (``wait -n; scancel``), so a
crash in one chain does not leave the trainer running alone. Jobs normally end
at their time limit; resubmit the same script to continue (`Restarts`_).

Bound free simulations with ``walltime``. AIMMD counts a free worker's frames
(``nframes``) only when a trajectory is complete, and a trajectory that stays
in its state runs until the walltime. A free worker of a C-GAIMMD run
keeps adding in-state frames for the fits, so give the job a walltime that
fits the campaign.

Monitoring
----------

The files to watch, all in the run folder:

``chainR<k>/worker.log``
   One line per pool refill (stage 1: the candidates generated, how many passed
   the validity check, and why the others did not) and one per shooting point
   (each is one line in the log, wrapped here)::

       C-GAIMMD pool refill 3 for run/chainR0/path000412: 32 candidates for 16 targets
           (network 5c1e9a02, uniform); 21 passed stage 1 (bond_count: 7, clash: 4) in 95.3 s
       C-GAIMMD shooting point for run/chainR0/path000412: lambda*=-0.812 lambda=-0.455
           (target), 3 candidates tried (lambda_window: 2), network 5c1e9a02, 41.7 s

   A ``WARNING: no acceptable C-GAIMMD shooting point after ... pool refills``
   means that no candidate passed for ``max_refills`` refills; the line counts
   the reasons (:doc:`cgaimmd_troubleshooting`).
``train<states>.log``
   The trainer's rounds: the fit's phases and losses, the value pass, the bins
   and the rate estimates (``k12 estimate``, and with a bias ``k12
   bias-reweighted``). :func:`aimmd.analysis.utils.extract_rate_estimates_from_log_file`
   reads the estimates into arrays.
``network<states>.json``
   The metadata of the network the shooting workers use: the round, the number
   of trained shooting points, the fit's keywords and settings, where the round
   started from and the seed.
``flow_fit<states>/rounds.jsonl``
   For the flow-matching fit, one record per round (data, losses, relabels,
   time).
``chainR<k>/path<n>.xtc.cgaimmd.json``
   The sidecars. :func:`~aimmd.cgaimmd.sidecar.shot_table` and
   :func:`~aimmd.cgaimmd.sidecar.summarise_shots` turn them into a table and
   a summary without writing anything into the run folder:

.. code-block:: python

    from aimmd.cgaimmd.sidecar import shot_table, summarise_shots

    rows = shot_table('run', states='ARB')
    summary = summarise_shots(rows, frame_interval_ps=20.0)
    print(summary['n_shots'], summary['reactive_share'], summary['candidates_rejected'])

Early in a campaign, check that the trainer's rounds come in at a steady pace,
that the shooting chains accept candidates (the share of rejected candidates
and the reasons), that free simulations add in-state frames, and that GROMACS
runs at the speed you expect (``grep Performance`` in the segments' logs).

Restarts
--------

A stopped run continues with the same command: resubmit the job script, or
call ``Launcher.run`` again. Every worker picks up from the files on disk:

- GROMACS segments resume from their checkpoints, and AIMMD rolls every
  trajectory and its caches back to the checkpoint it resumes from
  (:doc:`workflow`, "Restarts and Checkpoints"). Never restore ``.cpt`` files
  from an older copy of the run folder.
- A shot that was killed leaves its in-flight record
  (``chainR<k>/.cgaimmd_shooting_point.json``, written before the shot's
  segments start: the shooting point and how it was made). When the worker restarts, the
  record is attached to the path it names, or to the registered path whose
  shooting frame matches its coordinates; otherwise it is kept aside, and that
  shot's sidecar says ``source: 'unknown'``.
- The trainer continues its round count from ``network<states>.json``. Every
  round starts from ``initial_network`` as before.
- A trainer stopped during or right after a fit keeps that round's network as
  ``network<states>.final.h5`` with its metadata.

Before any worker starts, the trainer and the Launcher check the run folder:
a folder with a TPS history cannot continue as C-GAIMMD, and descriptor caches
written with other descriptor settings are refused, because AIMMD fills only
missing rows and a resumed run would mix both. The message names the ways out.

Settings you may change between submissions: the generator's settings, the
target law and its window, the acceptance settings, devices, the number of
workers and the fit's keywords (the next round uses them). Settings you may not
change under the same run folder: the network's architecture (the network file
would no longer load; the trainer would replace it by ``initial_network`` and
keep the old one as ``network<states>.replaced<k>.h5``), the generator atoms
and their descriptors, and the states.

Post-training and continuing other runs
---------------------------------------

The trainer works on any run folder. To train a C-GAIMMD network on an existing
RFPS or TPS run without changing it, mirror the run into a new folder and run
only a trainer there:

.. code-block:: python

    import aimmd
    from aimmd.cgaimmd.restart import derive_run

    params = aimmd.Params.load('params_cgaimmd.py')
    report = derive_run('rfps_run', 'cgaimmd_post', params=params)   # links trajectories, copies valid caches
    aimmd.Worker(params, 'cgaimmd_post').train(nrounds=10)

To continue an RFPS run as a C-GAIMMD run in its own folder, either give the
C-GAIMMD params a descriptor series of their own (``descriptors_series =
DESCRIPTORS.series`` with ``GeneratorDescriptors(..., refill=True)``: the new
series is computed once before the workers start, and the old caches stay
untouched), or set the old caches aside with
:func:`~aimmd.cgaimmd.restart.migrate_descriptors` and launch as usual. The
kinetics-convergence analysis of :doc:`advanced` works for C-GAIMMD runs too;
every data fraction is trained from ``initial_network``.

Backups
-------

What a run cannot do without, and what can be recomputed:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - files
     - why
   * - the campaign folder: ``params.py`` and every file it imports or names,
       ``pretrained/``, the BioEmu2 system folder
     - the run cannot be loaded or reproduced without them
   * - ``params*1.py`` in the run folder
     - the exact params of each submission
   * - ``network<states>.h5`` and ``.json``, ``network<states>.step*.h5``,
       ``flow_fit<states>/``
     - the trained networks and the fit's bookkeeping
   * - ``chain*/``: ``path*.xtc``, their ``*.cgaimmd.json`` sidecars and
       ``*.states.npy``
     - the shots and how they were made
   * - ``free*/``: trajectories and their state caches
     - the in-state frames and the exit statistics
   * - ``*.descriptors*.npy``, ``*.values.npy``
     - recomputable from the trajectories, but costly for large systems
   * - worker and trainer logs
     - the history of the run

Copy with ``rsync`` without ``--delete``, and never copy a backup back over a
running or resumable run folder: older ``.cpt``, GROMACS log or PLUMED
``COLVAR`` files make the run roll back and simulate the lost time again. A run
folder on a disk that is wiped at reboot needs a backup schedule (for example a
nightly copy of the network files, sidecars and trajectories).

A backup is a different folder: if you later analyse or continue it there, the
seeds derived from the run folder's path change unless ``generation_seed`` is
set, and the BioEmu2 fit's seed with them.

Before a campaign
-----------------

- The install passes the weights lane (BioEmu2) on the machine of the job.
- The end-state frames are unbiased MD, and the initial network was pretrained
  on them.
- ``generation_seed`` is set explicitly (one value per replicate).
- The acceptance rule is the default ``'transition_region'`` (no committor
  filter; the preflight's configuration line says so). Only a run that is
  meant to filter by the committor sets ``'acceptance': 'target'``, and then
  ``window_after_trained_shots`` and ``lambda_tolerance`` explicitly.
- The GROMACS executables are given in the params file, with the thread count.
- The preflight passes on the machine of the job, or the job runs it.
- The run folder is on a local or scratch disk, with a backup schedule.
- For a biased run: the PLUMED input and its state files are absolute paths,
  and the zero point check passes.
