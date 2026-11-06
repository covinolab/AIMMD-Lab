Troubleshooting C-GAIMMD
========================

The messages C-GAIMMD prints when something is wrong, what they mean and how
to fix them. Most of them appear in three places: when the params file loads
(in your script, the Launcher's output or a worker log), in the preflight
(``python -m aimmd.cgaimmd.preflight params.py``) and at the start of a
shooting worker or the trainer. The checks are the same in all three, so a
params file that passes the preflight on the machine of the job starts there.

Messages are quoted as they appear, with the variable parts in angle brackets.
Run the preflight first: it shows every problem of a params file at once,
one line per check, and it needs no GPU time for most of them
(``--no-generate``).

.. contents:: On this page
   :local:
   :depth: 1

Loading the params file
-----------------------

``... has no default_generator(), so params.cgaimmd_generator must name the generation method ...``
   The network does not say how it generates, and the params set no
   generator. Set ``cgaimmd_generator``, for example ``cgaimmd_generator =
   'ode'`` for a flow-matching network. ``JointPaiNN``, ``TinyFlowNetwork`` and
   ``BioEmu2Network`` have a default and need nothing.

``unknown generation method '<name>'; known methods are [...] (add one with aimmd.cgaimmd.generation.register_generator)``
   A typo, or a method that is registered in a module the params file does not
   import. ``'bioemu2_edm'`` is always listed but needs the ``bioemu2`` extra.
   A method of your own must be registered in the params file or in a module it
   imports, because every worker loads the params file afresh.

``generation method '<name>' with settings {...}: <Class>.__init__() got an unexpected keyword argument '<key>'. Did you mean '<other>'?``
   A misspelled setting in ``cgaimmd_generator``. The settings of ``'ode'`` are
   ``sampler``, ``n_steps``, ``chunk`` and ``guidance_scale``; those of
   ``'bioemu2_edm'`` are ``n_steps``, ``cfg_scale``, ``cfg_schedule`` and
   ``chunk``.

``chain_type='cgaimmd': the generator cannot use <Network>:`` followed by ``- generator 'ode': needs velocity(x_t, t, lam), which the network lacks``
   The generator's own check found members missing from the network. ``'ode'``
   needs a flow-matching network (``velocity`` and ``sample_prior``);
   ``'bioemu2_edm'`` needs a ``BioEmu2Network``. Pick the generator that fits
   the network, or leave ``cgaimmd_generator`` unset to use the network's
   default.

``<Network> does not implement the C-GAIMMD network protocol: ...``
   A required member of the committor network is missing or has the wrong
   signature; the message lists every problem. A C-GAIMMD network needs
   AIMMD's network members (``forward``, ``state_dict``, ``load_state_dict``,
   ``parameters``, ``to``, ``eval``, ``train``) and an integer ``n_atoms``
   (:func:`~aimmd.cgaimmd.protocol.check_network_protocol`).

``define the network's class in the params file, e.g. 'class Network(TinyFlowNetwork): pass': Params.save writes 'from <module> import network'``
   The network is an instance of a class imported from a package. AIMMD saves
   a copy of the params file that imports the network from the module that
   defines its class, so that module must hold the network. Define a subclass
   in the params file (``class Network(JointPaiNN): pass``) and build the
   network from it.

``unknown cgaimmd_parameters key '<key>'; allowed keys are [...]``
   A misspelled key of ``cgaimmd_parameters``; the message lists every allowed
   key. The generator's settings belong in ``cgaimmd_generator``, not here.

``chain_type='cgaimmd': the descriptors_function returns descriptors of width <w> but the network generates <M> atoms and needs 3 x <M> = <3M> ...``
   Under C-GAIMMD the descriptors are the flat Ångström coordinates of the
   generator atoms, three numbers per atom, in the order the network generates
   them. Use :class:`~aimmd.cgaimmd.preparation.descriptors.GeneratorDescriptors`
   with the same atoms the network was built for (``GeneratorAtoms(n_solute=...)``,
   and the same cluster of waters, if any).

``the network generates <M> atoms but the descriptors function (GeneratorDescriptors) selects <K> generator atoms (GeneratorAtoms); ...``
   The network and the descriptors describe different atoms. For BioEmu2 the
   network's atoms come from its system folder (``--n-atoms`` of
   ``build-system``): rebuild the folder or fix ``GeneratorAtoms``.

``initial_network '<file>' does not exist (looked for '<path>'; a relative path is relative to the folder of the params file)``
   The pretrained network is not where the params say. Relative paths are
   resolved against the params file's folder, not the current folder.

``WARNING: window_after_trained_shots is at its default 1000; this is highly system dependent ..., set it explicitly``
   Not an error. The final check enforces the target rule only after the
   network was trained on this many shooting points. Set the value for your
   system in ``cgaimmd_parameters``.

``WARNING: chain_type='cgaimmd' with round_shooting_index=False: ...``
   Set ``round_shooting_index = True``; otherwise shots reloaded later may take
   the frame before the shooting point as their shooting frame.

``C-GAIMMD: initial path '<file>' has no 'ARB' transition; it seeds free simulations only``
   Not an error. C-GAIMMD needs no initial transition; per-state files of
   end-state frames only seed the free simulations.

``Warning: <old name> is deprecated; use <new name> (the old names of C-GAIMMD are read for one release)``
   The params, sidecars or in-flight records use a name of earlier versions
   (``chain_type='flow'``, ``flow_parameters``, ``flow_*_function``,
   ``aimmd.flow``, ``*.xtc.flow.json``, ``restrain='flow_heavy'``). It is read
   and works for one release; the warning names the new spelling to use.

The preflight
-------------

A preflight line starts with ``OK``, ``WARNING``, ``FAIL`` or ``SKIPPED``.
The verdict is ``preflight OK`` (exit status 0) or ``preflight FAILED`` (exit
status 1). A failure in loading the params stops it after the first line.

``FAIL     fit: the flow-matching fit: relabel_every=0 with steps_per_round=<n>: every round starts again from the same network, ...``
   The generator's labels would stay those of the start network for the whole
   round, so the generator would never learn from the shooting points. Set
   ``relabel_every`` between 1 and ``steps_per_round - 1`` (the default is
   1,000 of 4,000 steps).

``FAIL     fit: the flow-matching fit: TypeError: flow_matching_fit() got unexpected keyword argument(s) '<key>'; ...``
   A keyword the fit does not know, in the params file's ``fit``. The keywords
   and their defaults are
   :data:`~aimmd.cgaimmd.flow_matching.fit.FLOW_FIT_DEFAULTS` (for example the
   learning rate is ``lr``) and the arguments of
   :func:`~aimmd.cgaimmd.bioemu2.fit.sequential_round` for BioEmu2.

``SKIPPED  fit: ...``
   The params file's ``fit`` does not call one of C-GAIMMD's fits, so its
   keywords cannot be checked without training. Nothing is wrong.

``WARNING  candidate: none of <n> generated candidates passed the validity function (<reason>); preparation not tried``
   Expected for an untrained generator, and harmless if the reason makes
   sense. With a pretrained generator, look at the reason: a systematic
   ``bond_count`` or ``clash`` suggests that the validity check needs the
   polish (:class:`~aimmd.cgaimmd.preparation.validity.PolishedValidity`) or
   unbiased clash reference frames; ``chirality`` with an E(3) network needs
   ``reflect_mirror_images=True``. Generate more candidates with
   ``--candidates 8`` to see the distribution of reasons.

``WARNING  candidate: ... rejected at preparation (<reason>; lambda <value>)``
   A valid candidate failed its preparation or the final check. With the reason
   ``state_<X>``, the prepared candidate is not in the reactive region. An untrained
   generator may produce such candidates; a pretrained one should not, so check
   the state definition and the preparation (a minimization that moves the
   candidate too far, a re-solvation that changes it).

``FAIL     gromacs: gmx_grompp names no GROMACS executable ...`` or ``... -version exited with <code>``
   The GROMACS executable of the params file is not found or does not start in
   this environment. Load the GROMACS module (or give the full path) in the
   shell or the job that runs the preflight, and give the full path in the
   params file; spawned workers do not run your login shell.

``FAIL     plumed: the -plumed file '<file>' is relative: mdrun runs in each chain folder, give an absolute path``
   ``mdrun`` runs inside each chain folder, where a relative path does not lead
   to the file. The same holds for files the PLUMED input reads (``<keyword>=<file>
   is relative: ...``), for example the state file of a frozen OPES bias.

``FAIL     plumed: no PRINT ... FILE=COLVAR: ...``
   With ``bias_source='file'``, AIMMD reads the bias of every shot from the
   ``COLVAR`` file PLUMED writes. Add a ``PRINT`` line with ``FILE=COLVAR`` and
   a ``STRIDE`` equal to the trajectory's output stride (``nstxout-compressed``).

``FAIL     bias zero point: the bias in the reactive region is not zero: ...``
   AIMMD's reweighting assumes that the bias is zero where it does not act,
   in the reactive region. Shift the bias in ``bias_function`` (for a frozen
   OPES bias, by its offset, for example the barrier) so that R frames read 0.

``WARNING  bias: gmx_mdrun biases the shots with -plumed, but record_bias is False: ...``
   The shots run under a bias that the trainer does not know about. Set
   ``record_bias = True`` and a ``bias_function``, or remove ``-plumed``.

Starting a run or a job
-----------------------

``job <id> is still running in the run folder <folder>: this job (<id>) does not start``
   Another job uses the run folder (the guard of ``create_job``). Wait for it,
   cancel it, or use another run folder. A job that ended normally does not
   block the folder; its claim in ``<folder>/aimmd_job.txt`` is taken over.

``the descriptor caches of <run> ('*.descriptors.npy') were written with other descriptor settings (...) than params.descriptors_function computes now (...); ...``
   The run folder records the descriptor settings its caches were written with,
   and the params compute something else (other generator atoms, another
   cluster, another whole-making). AIMMD fills only missing rows, so a resumed
   run would mix both. Either give the params a descriptor series of their own
   (``descriptors_series = DESCRIPTORS.series``, with
   ``GeneratorDescriptors(..., refill=True)`` to compute it once before the
   workers start), or set the old caches aside with
   ``aimmd.cgaimmd.restart.migrate_descriptors(params, '<run>')``.

``the descriptor cache <file> differs from params.descriptors_function at frame <i> (largest difference <d> > <tol>); ...`` or ``... has width <w>, while params.descriptors_function returns width <w2>; ...``
   The same, found by recomputing a sample of cached rows. A difference of up to
   0.01 Å (one step of the trajectory's stored precision) is accepted.

``continuing a TPS run as chain_type='cgaimmd' is not supported: ...``
   The chains hold TPS weights, which cannot be mixed with C-GAIMMD's
   RFPS-style weights. Post-train on a mirror of the run instead
   (:func:`~aimmd.cgaimmd.restart.derive_run`).

``SERIES CHECK: ...`` (``SeriesCoverageError``)
   Trajectories of the run have no cache of the params' descriptor series, and
   the series may not be refilled. The message says what is missing and how to
   fill it; with ``GeneratorDescriptors(..., refill=True)`` the job fills it
   itself.

``ImportError: C-GAIMMD's BioEmu2 adaptor needs bioemu2 0.1.0rc0 (Python >= 3.12): pip install 'aimmd-lab[bioemu2]' (CUDA: 'aimmd-lab[bioemu2-cuda]') (...)``
   The BioEmu2 network or generator was used in an environment without the
   ``bioemu2`` extra, or with a Python older than 3.12. A BioEmu2 run can still
   be loaded and analysed there; only BioEmu2 passes need the extra.

``<system folder> was built with bioemu2 <version>, bioemu2 <other> is installed: ...``
   A system folder records the bioemu2 version it was built with, because its
   atom map and trunk embeddings depend on it. Install the version the folder
   was built with (``bioemu2==0.1.0rc0``), or rebuild the folder with
   ``python -m aimmd.cgaimmd.bioemu2 build-system``.

``bioemu2 <version> builds ligand chain '<chain>' differently from <folder>: ...``
   Newer bioemu2 versions order the atoms of a SMILES ligand differently, which
   would silently scramble the ligand's atom map. Install
   ``bioemu2==0.1.0rc0`` or rebuild the system folder with the installed
   version.

During a run
------------

``WARNING: no acceptable C-GAIMMD shooting point after <n> pool refills (<k> candidates: <reasons>); waiting <s> s``
   No candidate passed in ``max_refills`` refills; the worker waits and tries
   again, and it keeps doing so until a candidate passes or the job ends. The
   reasons say where candidates fail:

   ``non_finite``, ``generation_oom``
      The generator returned non-finite coordinates, or ran out of GPU memory
      even after halving its batch ``generation_oom_halvings`` times (lower
      the generator's ``chunk``).
   ``bond_count``, ``not_isomorphic``, ``vf2_timeout``, ``chirality``, ``clash``, ``polish``
      The validity check rejected the raw sample: wrong number of bonds, a
      bond graph that is not the topology's, a graph match that took too long,
      a wrong-handed stereocenter, a contact closer than in the reference MD, or
      a failed polish.
   ``lambda_nonfinite``, ``raw_lambda``
      The committor of the raw sample was not finite, or (with
      ``raw_lambda_tolerance``) too far from the target.
   ``prepare_failed``, ``minimize_failed``
      A GROMACS step of the preparation failed; the sidecars of later shots and
      the scratch folders (``keep_scratch=True``) hold the GROMACS messages.
   ``state_<X>``
      The prepared frame is in state X, not in the reactive region.
   ``lambda_window``
      The prepared frame's committor is outside the tolerance around its target
      (the ``'target'`` rule) or outside the window (the ``'window'`` rule).
   ``pool_reset``
      Candidates discarded because a new network arrived; not a failure.

   Persistent ``lambda_window`` rejections mean that the generator does not
   reach its targets well enough for the tolerance: wait for more training
   rounds, or widen ``lambda_tolerance``. Persistent ``state_<X>`` means the
   generator produces configurations outside R at the requested targets,
   typically targets near the edges of the window: narrow the window.

``Warning: flow-matching fit: relabel_every=<n> with steps_per_round=<m>: the flow-matching labels of the reactive frames are the committor of the start network ...`` (printed once by the trainer)
   ``relabel_every`` is 0 or not below ``steps_per_round``, so the generator is
   trained on the start network's labels for the whole round. Set it as above.

A trainer that refuses its run folder at its second start
   See the descriptor-cache messages under `Starting a run or a job`_: a run
   folder that was started with other descriptor settings, or whose caches were
   modified, is refused on every start until the caches are fixed.

A free worker that runs for its whole walltime
   AIMMD counts a free worker's frames when a trajectory is complete, so a
   trajectory that stays in its state runs until the walltime, whatever
   ``nframes`` says. Bound free workers with the walltime.

Shots that start far from their target committor
   Before the network has been trained on ``window_after_trained_shots``
   shooting points, any frame in R is accepted (the sidecar says
   ``bootstrap_any_R``). This is intended: an untrained committor cannot judge
   the targets.
