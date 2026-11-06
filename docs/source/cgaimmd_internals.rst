C-GAIMMD internals
==================

This page describes how C-GAIMMD is built: the contracts between its parts,
the order in which things happen, what is written to disk and what a restart
does with it. It is meant for readers who extend C-GAIMMD (a new generation
method, a new network, a new fit), debug a run, or review the code. Users who
want to run C-GAIMMD start with :doc:`cgaimmd` and :doc:`cgaimmd_production`.

Throughout, *the run's states* are ``params.states`` (for example ``'ARB'``),
``sorted_states`` puts the end states first and last (``A``, ``R``, ``B``),
and every committor value is a logit toward ``sorted_states[2]``. Coordinates
are in Ångström unless a name says "flow units".

.. contents:: On this page
   :local:
   :depth: 2

Package layout
--------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - module
     - responsibility
   * - :mod:`aimmd.cgaimmd.protocol`
     - The network protocol (the committor only) and the helpers that apply
       the fallbacks of its optional members.
   * - :mod:`aimmd.cgaimmd.generation`
     - The generator interface, the registry of generation methods,
       ``make_generator`` and ``check_generator``.
   * - :mod:`aimmd.cgaimmd.types`
     - The dataclasses that cross the params-file hooks: ``CandidateContext``,
       ``ValidityResult``, ``PreparedFrame``, ``ShootingPoint``, ``NetworkInfo``.
   * - :mod:`aimmd.cgaimmd.values`
     - The protocol values function, the default ``values_function`` of a
       C-GAIMMD network.
   * - :mod:`aimmd.cgaimmd.devices`
     - The three device roles and their resolution inside worker tasks.
   * - :mod:`aimmd.cgaimmd.io`
     - Atomic writes and file hashing.
   * - :mod:`aimmd.cgaimmd.dataset`
     - The training data of one round, shared by every fit.
   * - :mod:`aimmd.cgaimmd.fitting`
     - What the online fits share: run folder, round counter, seed, and the
       settings-only mode of the preflight.
   * - :mod:`aimmd.cgaimmd.lambda_law`
     - The target-committor laws and the lambda window.
   * - :mod:`aimmd.cgaimmd.pool`
     - The candidate pool, the network watcher and the seed derivation.
   * - :mod:`aimmd.cgaimmd.source`
     - The shooting-point source of a shooting worker: start-up checks, the
       two stages, the final check, ``prepare_candidate``.
   * - :mod:`aimmd.cgaimmd.sidecar`
     - In-flight records, sidecars, their reconciliation, provenance and the
       shot table.
   * - :mod:`aimmd.cgaimmd.restart`
     - Checks and tools for restarts, migration and post-training.
   * - :mod:`aimmd.cgaimmd.preflight`
     - ``python -m aimmd.cgaimmd.preflight``.
   * - :mod:`aimmd.cgaimmd.plumed`
     - The PLUMED bias of the shots, as AIMMD sees it.
   * - :mod:`aimmd.cgaimmd.preparation`
     - GROMACS tools of candidate preparation.
   * - :mod:`aimmd.cgaimmd.flow_matching`
     - The PaiNN network, the ODE generator ``'ode'``, the flow-matching fit
       and the end-state pretraining.
   * - :mod:`aimmd.cgaimmd.bioemu2`
     - The BioEmu2 adaptor and its generator ``'bioemu2_edm'``.

Import rules:

- ``import aimmd`` imports no C-GAIMMD module. RFPS and TPS runs import none
  either: the shooting worker, the trainer, the free worker and the Launcher
  import :mod:`aimmd.cgaimmd` modules inside the branches that need them.
- ``aimmd.cgaimmd`` and its subpackages resolve their public names on first
  use. Heavy optional dependencies (PyG, networkx, MDAnalysis in the
  preparation tools, bioemu2) are imported inside the functions that use them.
- :mod:`aimmd.cgaimmd.bioemu2._api` is the only module of AIMMD-Lab that
  imports ``bioemu2`` (and RDKit). Without the ``bioemu2`` extra it raises one
  ``ImportError`` with the install command; the network, its head and
  adapters, the fit and the system folder still import, so a BioEmu2 run can
  be loaded and analysed anywhere.
- The old names of earlier versions live in one module,
  ``aimmd.cgaimmd._legacy`` (`Old names`_).

Conventions
-----------

Shapes and units
   Generated configurations are ``(B, M, 3)`` in Ångström, with ``M`` the
   network's ``n_atoms``. Descriptor rows are their flattening, ``(B, 3 M)``,
   in the order ``x0, y0, z0, x1, ...``. Target committors are ``(B,)``
   tensors on the network's device.
Records
   Every record that crosses an interface (``record``, ``info``,
   ``metadata``, sidecars, settings) is a JSON-serializable dict. Numbers that
   cannot be represented (NaN) are written as ``null``.
Writes
   Every file C-GAIMMD writes into a run folder goes through
   :func:`~aimmd.cgaimmd.io.atomic_write_bytes`: a temporary file next to the
   target, flushed and synced, then renamed over it, so a reader sees the old
   or the new file and never a partial one. Readers match files to each other
   by content hash (sha256), never by locks.
Devices
   A network is used in three roles (`Devices`_). Devices are resolved at use,
   inside a worker task, and never cached.
Seeds
   Seeds are derived from settings and names, never drawn from a global stream
   (`Seeds`_).

The network protocol
--------------------

A C-GAIMMD network is the committor model of the run. There is no base class:
any object with the members below is one, and
:func:`~aimmd.cgaimmd.protocol.check_network_protocol` lists every problem at
once.

Required
~~~~~~~~

``forward(inp) -> Tensor (B, 1)``
   The logit committor toward ``sorted_states[2]`` of the input the values
   function builds: the dense descriptor rows ``(B, 3 M)``, or in graph mode
   the batch dict of ``params.descriptor_transform``. Differentiable in the
   positions (the flow-matching fit's gradient penalty needs it).
``state_dict()``, ``load_state_dict(sd)``, ``parameters()``
   Tensors only, loadable with ``torch.load(..., weights_only=True)``.
``to(device)``, ``eval()``, ``train(mode)``
   As ``torch.nn.Module``. A ``torch.nn.Module`` must override ``forward``.
``n_atoms``
   An int ``M >= 1``. (Networks of earlier versions may leave it out when they
   have ``sample_prior``; ``M`` is then read from a prior sample. This fallback
   goes away with the old names.)

Optional, with the fallback C-GAIMMD code applies (through the helper named)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``committor(x) -> Tensor (B,)``
   A fast path for coordinates that are not on disk; it must equal
   ``forward``. Fallback: ``forward`` of the values-function input built from
   the flat coordinates (:func:`~aimmd.cgaimmd.protocol.coordinate_committor`).
``lambda_clip``
   The conditioning range :math:`L > 0`; targets beyond :math:`\pm L` are
   clipped. Fallback: ``cgaimmd_parameters['lambda_clip']``, which is then
   required (:func:`~aimmd.cgaimmd.protocol.lambda_clip_of`).
``reset_parameters()``
   Re-initializes the weights. A fit that trains from reset every round uses
   it when there is no ``initial_network``.
``default_generator()``
   The generation method used when ``params.cgaimmd_generator`` is None.
   Without it, ``cgaimmd_generator`` must be set.
``provenance()``
   A dict of what the network's results depend on beyond its weights (for
   BioEmu2: the checkpoint, its Hugging Face snapshot, the parameter checksum
   of the frozen model). The shooting worker calls it once at start-up and
   writes it into every sidecar.

Generation members (``velocity``, ``sample_prior``, ``generate``) are not part
of the protocol. Each generation method checks what it needs.

Numerics at start-up
~~~~~~~~~~~~~~~~~~~~

:func:`~aimmd.cgaimmd.protocol.check_network_numerics` runs on the first
frames of the first initial path: the descriptors have width ``3 M``, the
network's values are finite, ``committor`` (when present) agrees with
``forward``, and one tiny generation through the run's generator returns a
finite ``(B, M, 3)`` tensor on the network's device. The shooting worker and
the preflight both run it (:func:`~aimmd.cgaimmd.source.check_start_up_numerics`).

The protocol values function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without a ``values_function`` in the params file, AIMMD builds one from the
network when the params load, on the CPU. A C-GAIMMD network trains on
``training_device`` and generates on ``generation_device``, so
:class:`~aimmd.cgaimmd.values.ProtocolValuesFunction` resolves the device at
every call instead: inside a worker task it moves the network to the values
device of that task (``values_device`` in shooting and free workers,
``training_device`` in the trainer), outside a worker task it leaves the
network where it is, and it always moves the inputs to the network's device.
``Params`` picks it under ``chain_type='cgaimmd'``, and under any chain type
when the network is a C-GAIMMD network, so post-training on an RFPS run needs
no values function either.

The generation interface
------------------------

A generator turns target committors into candidate configurations. The
interface (:class:`~aimmd.cgaimmd.generation.Generator`) is structural; nothing
is inherited:

``method``
   The registry name (a string), written to every sidecar.
``check(network) -> list of str``
   What the method needs from ``network`` and does not find; ``[]`` when the
   network can be used. Cheap, and never raises for a network that lacks
   something.
``generate(network, lam, *, generator=None) -> Tensor``
   ``lam`` is ``(B,)`` on the network's device; the result is ``(B, M, 3)``
   Ångström on the same device, without autograd history, one candidate per
   entry of ``lam``. ``generator`` is the ``torch.Generator`` of the random
   stream; every random number is drawn from it. How the batch is chunked is
   the generator's own setting.
``settings() -> dict``
   JSON-serializable; ``make_generator({'method': method, **settings()})``
   builds an equal generator.

The registry
~~~~~~~~~~~~

:func:`~aimmd.cgaimmd.generation.register_generator` maps a method name to a
factory (usually the class); the factory is called with the settings as
keywords and raises ``TypeError`` or ``ValueError`` for settings it does not
accept. The registry is per process. Built-in methods are known by name from
the start and their module is imported on first use (``'ode'``:
:mod:`aimmd.cgaimmd.flow_matching.sampler`; ``'bioemu2_edm'``:
:mod:`aimmd.cgaimmd.bioemu2.sampling`, which needs the ``bioemu2`` extra).

Re-registration rules, because ``Params.load`` executes the params file (and
the modules next to it) again on every load:

- registering the same factory again does nothing;
- a new definition of the registered factory (same qualified name, defined in
  the same file, whatever folder the file was loaded from) replaces it
  silently;
- any other factory under a registered name raises ``ValueError`` unless
  ``replace=True``.

``make_generator(spec, network)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The value of ``params.cgaimmd_generator`` is resolved in this order:

1. a generator object is used as it is (a class where an instance belongs is
   a ``TypeError`` that says so);
2. a method name builds the method with its defaults;
3. a dict with ``'method'`` builds that method with the other keys as
   settings;
4. a dict without ``'method'`` updates the settings of the network's default
   method;
5. None takes ``network.default_generator()``; a network without one is a
   ``TypeError`` that names the field.

:func:`~aimmd.cgaimmd.generation.canonical_generator_spec` validates a value
when it is assigned to ``Params`` without building a generator that would
import an optional extra; ``Params`` stores names and dicts as given, and a
generator object of a registered method as its settings dict, so that
``Params.save`` writes a value the workers can rebuild.

:func:`~aimmd.cgaimmd.generation.check_generator` returns every problem of
using a generator with a network: a missing or mistyped member, settings that
are not a JSON-serializable dict, and what the generator's own ``check``
reports. An exception raised by ``check`` or ``settings`` is reported, not
raised.

The start-up check
~~~~~~~~~~~~~~~~~~

Wherever a generator is about to be used (loading the params, the shooting
worker's start-up, the preflight), the same sequence runs:
:func:`~aimmd.cgaimmd.protocol.check_network_protocol`, then
``make_generator`` and ``check_generator``
(:func:`~aimmd.cgaimmd.source.build_checked_generator`), then the atom count
of the network against the generator atoms of the descriptors
(:func:`~aimmd.cgaimmd.source.check_generator_atoms`), then the numerics with
one tiny generation. Params read through old names only warn at the second
step, so that runs of earlier versions load for analysis; their shooting
workers check strictly.

The method ``'ode'``
~~~~~~~~~~~~~~~~~~~~

:class:`~aimmd.cgaimmd.flow_matching.sampler.ODEGenerator` integrates the
network's flow-matching ODE from the prior (flow time 0) to configurations
(flow time 1). Its ``check`` requires
``velocity(x_t, t, lam, *, uncond_mask=None) -> (B, M, 3)`` and
``sample_prior(lam, n, *, generator=None) -> (n, M, 3)``, both in flow units,
and ``uncond_mask`` only when ``guidance_scale != 1``. Optional
``to_flow``/``from_flow`` convert between Ångström and flow units (identity
without them). The integrator is fixed-step (``'euler'``, ``'midpoint'``,
``'rk4'``) with ``n_steps`` steps; the batch is split into calls of ``chunk``
samples. With guidance, every step evaluates the field twice, with and
without the label, and uses :math:`w \, v_{\text{cond}} + (1 - w) \,
v_{\text{uncond}}`. With ``guidance_scale=1`` the result is bit-identical to
no guidance. The sampler never centres or projects: a network whose flow
lives in a subspace (PaiNN: free of centre-of-mass motion) returns velocities
in that subspace itself.

The method ``'bioemu2_edm'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~aimmd.cgaimmd.bioemu2.sampling.BioEmu2Generator` samples with
BioEmu2's own EDM sampler (Karras schedule, stochastic churn, a random rigid
motion per step, ``n_steps`` steps) on a model that wraps the frozen BioEmu2,
the network's adapters, the requested logits and the guidance schedule as a
BioEmu2 ``InferenceModel``, one system copy per requested logit. Each chunk of
``chunk`` samples starts from a seed drawn from the caller's
``torch.Generator`` plus the chunk's offset, on a forked random stream, so a
generation is reproducible and chunks never share noise. The BioEmu2 samples
(heavy atoms, centred) are completed to the generator atoms from the system's
template (hydrogens, terminal oxygens, ions); a sample with non-finite
coordinates comes back as NaN, which every validity check rejects. Guidance
(``cfg_scale``, ``cfg_schedule``) uses the adapters' learned null or plain
BioEmu2 as the unconditioned model, per window of noise levels. Its ``check``
requires a ``BioEmu2Network``.

How the source calls a generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shooting-point source knows nothing about how a generator works. It
slices a refill into batches of the generator's ``chunk`` setting (the whole
refill for a generator without one), checks ``must_stop`` between batches,
and after a CUDA out-of-memory error halves the batch, at most
``generation_oom_halvings`` times; a refill that still fails marks all its
candidates ``generation_oom``. The batch size it used is written to every
sidecar next to the method and settings.

Params fields and rules
-----------------------

The fields of ``chain_type='cgaimmd'`` are ``cgaimmd_parameters``,
``cgaimmd_generator``, the four hooks (``cgaimmd_validity_function``,
``cgaimmd_prepare_function``, ``cgaimmd_minimize_function``,
``cgaimmd_lambda_sampler``) and ``initial_network``, which works under every
chain type. They are validated when they are assigned and when the params
load.

``cgaimmd_parameters`` stores only the keys the user gave;
``Params.cgaimmd_settings`` returns a new dict
``{**CGAIMMD_PARAMETER_DEFAULTS, **cgaimmd_parameters}`` (see
:data:`aimmd.params.utils.CGAIMMD_PARAMETER_DEFAULTS`). Unknown keys and values
of the wrong type are errors at assignment.

The rules checked under ``chain_type='cgaimmd'``:

1. the network implements the protocol;
2. the generator can be built and can use the network (a warning only for
   params read through old names);
3. a single system (``multi_system`` is not supported);
4. a ``descriptors_function`` exists;
5. no ``rescale_committor`` (it rescales the network in place, behind the fit);
6. ``gen_temperature > 0`` (generated shooting points carry no velocities);
7. a conditioning range exists (the network's ``lambda_clip`` or the setting);
8. no ``placement='none'`` together with a prepare function;
9. the network's class is defined in the params file (also under other chain
   types, for C-GAIMMD networks);
10. warnings: no ``initial_network``, ``round_shooting_index=False``,
    ``window_after_trained_shots`` at its default;
11. the descriptors of the initial paths have width ``3 M`` in the run's
    descriptor series;
12. the initial paths hold frames of both end states (C-GAIMMD needs no
    transition: a path without one seeds free simulations only).

``initial_network`` must exist (a relative path is relative to the params
file and is written that way on save), and it is not supported with
``multi_system``.

The shooting worker seam
------------------------

``Worker.shoot`` runs C-GAIMMD only for reactive-region chains of a
``'cgaimmd'`` run. Internal shooting chains of the end states and sweeps are
unchanged. For a C-GAIMMD chain the worker:

1. calls :func:`~aimmd.cgaimmd.restart.check_cgaimmd_restart` (without the
   descriptor check, which the source does itself) and imports the source and
   sidecar modules;
2. loads its chains without transition extraction and without an RFPS
   selection pool (``pool.log`` is never written);
3. builds the source (:func:`~aimmd.cgaimmd.source.make_cgaimmd_source`) and
   blocks in ``wait_for_network`` until the trainer's network file loads and
   the start-up checks pass. C-GAIMMD workers never wait for bins;
4. reconciles each chain folder's in-flight record (`Reconciliation after a
   restart`_), only now, so that a recovered sidecar's ``lambda_registered``
   is computed with the run's network;
5. per shot: calls ``source.next_shooting_point(chain=..., folder=...,
   path_name=...)`` (None means the worker must stop), writes the in-flight
   record and the shooting-point coordinates, initializes and runs the two
   segments as for RFPS, registers the path, writes its sidecar, clears the
   in-flight record and calls ``source.note_registered()``.

When AIMMD's read-back check fails (frame 0 of the new segments is not in R),
an RFPS worker stops unless ``retry_with_state_definition_glitches`` is set.
For C-GAIMMD chains, where candidates sit near the edges of R on purpose,
:data:`~aimmd.cgaimmd.source.READBACK_FAILURE_IS_REJECTION` makes this a
candidate rejection (``state_readback``): the source's ``note_glitch``
counts it, the segments are cleared, and the next candidate is popped.

Free workers of a C-GAIMMD run seed from the initial files of their own state
and skip files without a frame of that state, so per-state files of end-state
frames are valid initial paths.

The shooting-point source
-------------------------

:class:`~aimmd.cgaimmd.source.GenerativeShootingPointSource` returns one
shooting point per call. Nothing is loaded, built or moved to a device when it
is constructed.

Start-up
~~~~~~~~

``wait_for_network`` polls ``network<states>.h5`` through the
:class:`~aimmd.cgaimmd.pool.NetworkWatcher` until it loads into
``params.network``, then runs the start-up checks once: the settings
(``params.cgaimmd_settings``, read once), the generator (built and checked
strictly), the atom count, the numerics on the initial paths, the window and
the placement, the base seed and the provenance every sidecar will carry
(aimmd version and commit, software versions, the network's provenance, the
bias settings). A failed check raises and stops the worker; a worker that
must stop while waiting returns False.

The network watcher
~~~~~~~~~~~~~~~~~~~

Before every pop the watcher compares the network file's ``(mtime, size,
inode)`` with the last one it read; only on a change does it read the bytes
and hash them. A new sha256 is loaded strictly into ``params.network``; a file
that does not load is skipped until its stat changes (one warning per
version), and the weights already loaded stay. The trainer writes
``network<states>.json`` before the ``.h5``; the watcher takes its ``round``
and ``n_trained_shots`` only when the json's ``sha256`` matches the loaded
file (``metadata_matched``), and otherwise keeps the last matched values, so
a normal hand-off never drops the acceptance rule back to the bootstrap rule.
If no json matches a minute after a load, it warns once.

The pool
~~~~~~~~

:class:`~aimmd.cgaimmd.pool.CandidatePool` holds the stage-1 survivors of one
network. Pops are uniformly random over the pool (not target by target), with
the refill's own pop stream, so consecutive shots rarely share a target. The
pool resets when the watcher loads a new network; its candidates count as
``pool_reset``. Exception: a refill whose network was replaced while it was
being generated (before its first pop) serves
:data:`~aimmd.cgaimmd.source.STALE_REFILL_POPS` (one) pop first. That pop's
final check uses the new network, and its sidecar says
``pool.stale_refill_pop`` and names the refill's network
(``pool.refill_network_sha256``).

Stage 1: a refill
~~~~~~~~~~~~~~~~~

When the pool is empty:

1. the refill's seed and its four streams are derived (`Seeds`_);
2. the trainer's bins and densities are loaded if present;
3. ``n_targets`` targets are drawn from the law (`Target laws`_), each
   repeated ``candidates_per_target`` times;
4. the generator produces the candidates on the generation device;
5. candidates with non-finite coordinates are rejected (``non_finite``);
6. ``cgaimmd_validity_function`` checks the rest, on the validity stream. It
   returns relabelled positions, a per-sample ``ok``, a reason and an optional
   per-sample ``info``. The reason code is the text before the first colon
   (``'clash: O7-N12 1.21 A'`` counts as ``clash``);
7. the committor of the survivors is computed (``lambda_raw``); non-finite
   values are rejected (``lambda_nonfinite``). With ``raw_lambda_tolerance``
   set and the acceptance rule enforced, candidates whose raw value is further
   than that from their target are rejected (``raw_lambda``);
8. the survivors enter the pool; the refill's bookkeeping (size, targets,
   passed, rejections, seconds, batch size, device, network sha256) is kept
   for the sidecars of its candidates, and one log line summarizes it.

Stage 2: one candidate
~~~~~~~~~~~~~~~~~~~~~~

Each pop goes through :func:`~aimmd.cgaimmd.source.prepare_candidate`, which
is public so that analysis code prepares candidates exactly as a shooting
worker does:

1. ``cgaimmd_prepare_function(positions, context)`` if set, else
   :func:`~aimmd.cgaimmd.source.place_in_box` (``'box_center'``: the centroid
   at the centre of the run structure's box; ``'none'``: as generated). A
   prepared frame with ``ok=False`` is rejected with its reason code
   (``prepare_failed`` when the hook gives none);
2. ``cgaimmd_minimize_function(frame, context)`` if set (reason
   ``minimize_failed`` when the hook gives none);
3. :func:`~aimmd.cgaimmd.source.final_check`.

A stop request during a hook ends the pop without a rejection. Exceptions from
hooks, the generator, the network or AIMMD propagate and stop the worker:
hooks report bad candidates through ``ok=False``, not by raising.

The final check
~~~~~~~~~~~~~~~

:func:`~aimmd.cgaimmd.source.final_check` evaluates the prepared frame as
grompp will receive it:

1. the positions, as float32, are rounded to ``final_check_round`` decimals
   (``'auto'``: 2 for ``.xtc``, the precision at which frame 0 of ``back.xtc``
   is stored) and wrapped in a one-frame reader of the type ``Path.compute``
   hands to user functions;
2. ``params.states_function`` must return ``states[1]``; otherwise the reason
   is ``state_<X>``;
3. :math:`\lambda` = ``values_function(descriptors_function(reader))``; a
   non-finite value is ``lambda_nonfinite``;
4. with fewer than ``window_after_trained_shots`` trained shooting points (the
   watcher's matched ``n_trained_shots``, 0 without metadata) any R frame
   passes (``bootstrap_any_R``). From then on ``'target'`` requires
   :math:`\lvert\lambda - \lambda^*\rvert \le` ``lambda_tolerance`` and
   ``'window'`` requires :math:`\lambda` inside the resolved window; a failure
   is ``lambda_window``.

The record holds ``ok``, ``reason``, ``state``, ``lambda``, ``rule``,
``enforced``, ``tolerance``, ``window``, ``window_after_trained_shots``,
``n_trained_shots``, ``decimals`` and ``seconds``; the rule, tolerance and
window are filled for every outcome.

Waiting and failure policy
~~~~~~~~~~~~~~~~~~~~~~~~~~

The source never raises for want of a candidate. After ``max_refills`` refills
without a shooting point it logs a warning with the counts of every rejection
reason since the last wait, waits ``wait_seconds`` (polling ``must_stop``
every second), and starts counting again. It returns None as soon as the
worker must stop.

Devices
~~~~~~~

:func:`~aimmd.cgaimmd.devices.device_for` maps a role to a device inside a
worker task (``aimmd._config.WORKER_TASK`` set by ``Worker.run``):
``'generation'`` → ``generation_device``; ``'values'`` → ``values_device`` in
shooting and free workers and ``training_device`` in the tasks that fit
(``'train'``, ``'kinetics_convergence'``); ``'training'`` →
``training_device``. ``'auto'`` is the GPU the worker was bound to, if any.
Outside a worker task it returns None and no ``torch.cuda`` function is called.
The source moves the network to the generation device for a refill and to the
values device for the final check.

Seeds
~~~~~

:func:`~aimmd.cgaimmd.pool.base_seed` is ``generation_seed``, or a 63-bit hash
of the absolute run folder path. :func:`~aimmd.cgaimmd.pool.refill_seed` is a
hash of the base seed, the chain folder's name, the path name and the number
of refills already made for that path, so a restarted worker repeats the
refills of the path it was preparing. :func:`~aimmd.cgaimmd.pool.refill_streams`
spawns four independent streams from it: the law (targets and
``cgaimmd_lambda_sampler``), the pops, the preparation (one seed per
candidate) and the validity check. Drawing more numbers from one stream never
moves another. The seeds of every shot are in its sidecar (``seed``).

Target laws
-----------

:func:`~aimmd.cgaimmd.lambda_law.draw_targets` draws the targets of a refill.

The window
   :func:`~aimmd.cgaimmd.lambda_law.resolve_window`: a ``(lo, hi)`` setting is
   clipped to :math:`[-L, L]`. ``'bins'`` is the trainer's shooting-value
   cutoffs ``(sp_cutoff_min, sp_cutoff_max)``, computed with the trainer's own
   helper from its bins (user values in ``reweight_parameters`` win) and
   clipped to :math:`[-L, L]`; without bins it is :math:`[-L, L]`. The trainer
   reweights shots whose shooting value lies outside those cutoffs as free
   excursions, so ``'bins'`` keeps the targets inside them.
``'uniform'``
   Uniform on the window.
``'bins'``
   AIMMD's RFPS shooting-point law without a selection pool: an adaptive bin of
   the trainer restricted to the window, with weight inversely proportional to
   its selection density (with the Lorentzian and density adjustments of
   RFPS selection, and the selection densities shared over all chains of the
   run, as :func:`aimmd.worker.utils.select_shooting_point` computes them),
   then uniform inside the bin. Without bins it falls back to uniform draws.
``cgaimmd_lambda_sampler``
   ``(n, context) -> (n,)``: replaces the law. Values must be finite and inside
   the window; ``context.rng`` is the law stream.

The law's name, inputs and window go into every sidecar (``lambda_law``).
Whatever the law, shots are registered and reweighted as RFPS shots.

Candidate preparation
---------------------

The hook contracts
~~~~~~~~~~~~~~~~~~

Hooks are functions defined in the params file (they may call objects built
there). They receive a :class:`~aimmd.cgaimmd.types.CandidateContext` with
the params, the settings, the network and its metadata (``network_info``),
the chain folder and path name, a scratch folder of their own, a seed and a
``numpy.random.Generator`` stream, the device, ``must_stop`` and a log
function.

``cgaimmd_validity_function(positions, context) -> ValidityResult``
   ``positions`` is ``(n, M, 3)``. The result holds ``positions`` (relabelled
   into topology order where the check found an isomorphism), ``ok`` (n,),
   ``reason`` (n,) and optional per-sample ``info``.
``cgaimmd_prepare_function(positions, context) -> PreparedFrame``
   ``positions`` is one ``(M, 3)`` candidate; the result is the full system of
   ``params.topology`` (positions, box), with ``ok``, ``reason``, ``record``
   and ``seconds``.
``cgaimmd_minimize_function(frame, context) -> PreparedFrame``
   A prepared frame in, a minimized frame out.
``cgaimmd_lambda_sampler(n, context) -> array``
   See `Target laws`_.

:func:`~aimmd.cgaimmd.source.candidate_context` builds a context outside a
worker (analysis, the preflight).

The GROMACS tools
~~~~~~~~~~~~~~~~~

The tools of :mod:`aimmd.cgaimmd.preparation` share these rules:

- every GROMACS call of a candidate runs in its own folder under
  ``context.scratch``, removed afterwards unless ``keep_scratch``;
- executables come from the params (``gmx_grompp``, ``gmx_mdrun``) or the
  tool's own arguments, by full path; nothing relies on a login shell;
- options a tool sets itself are removed from the configured commands first
  (GROMACS refuses an option given twice), only from the GROMACS part of a
  command, so a launcher prefix (``srun -c 8``) keeps its options;
- ``-plumed`` is removed from every mdrun command of preparation
  (:func:`~aimmd.cgaimmd.preparation.gromacs.sanitize_mdrun`): candidates are
  never prepared under a bias;
- coordinates handed to GROMACS are written at 6 decimals (nm), so a
  candidate is not rounded to 0.01 Å before grompp;
- position restraints hold the generator atoms (``restrain='generator_heavy'``
  or ``'generator_all'``) through a derived topology with its own define
  (``AIMMD_CGAIMMD_POSRES``) and the candidate itself as the reference; the
  production topology carries no restraints;
- a changed box, non-finite output, a GROMACS failure or a timeout make
  ``ok=False`` with the reason ``prepare_failed: <message>`` or
  ``minimize_failed: <message>``; only a composition that does not add up after
  assembly is a bug and raises.

``GeneratorDescriptors`` makes molecules whole by a walk along their bonds,
places several molecules next to each other by lattice vectors, and, for a
cluster, places every chosen water at the image nearest the reference of the
ranking rule that chose it. The ``GromacsSolvator``'s selection-consistency
step deletes every added water that the descriptors' own ranking would have
chosen instead of a cluster water, so the cluster is still what the
descriptors select from the solvated system.

In-flight records and sidecars
------------------------------

Files
~~~~~

``chain<R><k>/.cgaimmd_shooting_point.npy`` and ``.json``
   The in-flight record of the shot being simulated in that chain folder: the
   shooting-point coordinates (float32, ``(N, 3)``) and the record. Written
   just before the segments are initialized (coordinates first, then the
   record with ``positions_sha256`` and ``created``, both atomically), cleared
   after registration.
``chain<R><k>/path<n>.xtc.cgaimmd.json``
   The sidecar of a registered shot: the record plus what registration adds.

None of these names matches ``*back*``, ``*forw*`` or ``path??????<ext>``,
so AIMMD's per-shot clean-up and path loaders never touch them. The
reweighting reads nothing from them.

The sidecar schema
~~~~~~~~~~~~~~~~~~

Schema ``aimmd.cgaimmd.shot/1``. Every key of
:data:`~aimmd.cgaimmd.sidecar.SIDECAR_KEYS` is present, with ``null`` where it
does not apply.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - key
     - content
   * - ``schema``, ``source``
     - ``'aimmd.cgaimmd.shot/1'``; ``'cgaimmd'`` (``'unknown'`` for a record
       that could not be attached).
   * - ``note``, ``stale_record``
     - A note on unusual registration; a quarantined record that did not match
       the path.
   * - ``path``, ``path_name``, ``chain``
     - The registered file name, the expected path name, the chain folder.
   * - ``created``, ``registered``, ``recovered_after_restart``
     - UTC times of the record and the registration; whether the sidecar was
       written from an in-flight record after a restart.
   * - ``lambda_target``, ``lambda_raw``, ``lambda_final``, ``lambda_registered``
     - The target; the committor of the raw sample (stage 1), of the prepared
       frame (final check), and of the registered shooting frame. The last is
       computed from the frame's descriptor row with the run's network and
       values function and is not cached; ``registration.lambda_registered_missing``
       says why when it is null.
   * - ``state_final``
     - The state of the prepared frame.
   * - ``lambda_law``
     - ``name``, ``window`` and ``window_source``, and the law's inputs.
   * - ``acceptance``
     - ``rule``, ``enforced``, ``tolerance``, ``window``,
       ``window_after_trained_shots``, ``n_trained_shots``.
   * - ``network``
     - ``file``, ``sha256``, ``round``, ``n_trained_shots``,
       ``metadata_matched``, ``provenance``.
   * - ``generator``
     - ``method``, ``settings``, ``batch_size``.
   * - ``pool``
     - ``refill``, ``candidate``, ``refill_size``, ``n_targets``,
       ``candidates_per_target``, ``passed_stage1``, ``pool_left_after_pop``,
       ``reset_count``, ``stale_refill_pop``, ``refill_network_sha256``.
   * - ``stage1_rejected``
     - ``{reason: count}`` of the refill that produced the candidate.
   * - ``candidates``
     - ``tried`` (candidates popped for this shot) and ``rejected`` (``{reason:
       count}`` in stage 2).
   * - ``validity``, ``prepare``, ``minimize``
     - The hook (qualified name) and its record: ``info`` of the validity
       check; ``placement``, ``ok``, ``reason``, ``seconds``, ``record`` of
       preparation and minimization.
   * - ``cost_seconds``
     - ``generate``, ``validity``, ``raw_lambda`` (the refill's), ``prepare``,
       ``minimize``, ``final_check``, ``total``.
   * - ``seed``
     - ``base``, ``refill``, ``refill_path``, ``refill_index``,
       ``candidate_index``, ``prep``.
   * - ``device``
     - ``generation``, ``values``.
   * - ``aimmd``
     - ``version``, ``git_commit``, ``git_dirty``, ``git_reason``,
       ``git_source``, ``source_sha256``.
   * - ``software``
     - Versions of Python, numpy, torch, MDAnalysis, bioemu2 and
       cuequivariance, read from the installed distributions (nothing is
       imported); null for what is not installed.
   * - ``bias``
     - Null without a bias; otherwise ``record_bias``, ``bias_source``,
       ``bias_function``, ``bias_reactive_threshold``, ``plumed_file``,
       ``plumed_file_relative``, ``plumed_sha256``, ``plumed_inputs``
       (files the PLUMED input reads, with sha256), ``preparation_unbiased``
       (True when every preparation hook is known to run without the bias,
       None when a hook is the user's own code) and
       ``preparation_unverified_hooks``.
   * - ``registration``
     - ``n_frames``, ``path_type``, ``shooting_index``, ``complete``,
       ``lambda_registered_missing``.

Provenance of the code
~~~~~~~~~~~~~~~~~~~~~~

``aimmd.git_commit`` comes from the first source that gives one: a git
checkout of the package, the ``.git`` files read directly, pip's
``direct_url.json`` of the install, or the build record ``setup.py`` writes
into a built package (``aimmd/_build_record.json``). ``git_source`` names the
source; without any, ``git_commit`` is null and ``git_reason`` says why.
``source_sha256`` is a digest of the package's source files, which
identifies the code even without a commit.

Attaching a record to a path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A record is attached to the path it names. If AIMMD registered the shot under
another index (a renumbering at a restart), it is attached to a path whose
shooting frame matches its coordinates (RMS below 0.05 Å after removing each
centroid; xtc stores 0.01 Å). Otherwise it is kept under ``stale_record`` and
the sidecar's ``source`` is ``'unknown'``.

Reconciliation after a restart
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~aimmd.cgaimmd.sidecar.reconcile_in_flight` runs for each chain folder
when a shooting worker starts, after the network has loaded:

``'recovered'``
   The record's path is registered but has no sidecar (a crash between
   registration and the sidecar write): the sidecar is written with
   ``recovered_after_restart=True`` and the record cleared.
``'cleared'``
   The path is registered with a sidecar (a crash before the record was
   cleared), or the segments are not both initialized: the record is cleared
   and the worker selects afresh.
``'kept'``
   Both segments are initialized (GROMACS: both ``.tpr``): the shot resumes
   and keeps its record.

A checkpoint rollback of the segments (:doc:`workflow`) leaves the record in
place; it reaches the sidecar when the resumed shot is registered.

Fit contracts
-------------

What AIMMD's trainer expects from ``fit``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The params file's ``fit(params, pathensemble, verbose=False, worker=None)``
trains ``params.network`` in place for one round. It returns either the list
of losses, as for any AIMMD fit, or ``(losses, info)``:

- an empty list means "nothing trained, do not save";
- ``info['status']``: ``'trained'``, ``'no-op'`` (a round that changes nothing
  but counts, for example the first round of a fresh run without data, so that
  the trainer writes the first bins), ``'skipped'`` (no data yet) or
  ``'stopped'`` (``must_stop`` during the round);
- ``info['reason']``, ``info['round']``, ``info['record']``;
- ``info['network_metadata']``: what the trainer stores next to the network
  (round, trained shots, seed, settings, start), or None for a round that did
  not count.

The trainer writes ``network_metadata`` as ``network<states>.json``, with
``total_steps``, ``sha256`` and ``saved`` added, *before* ``network<states>.h5``,
both atomically; step backups (``network<states>.step<n>.h5``) get their json
too. A stop during or right after a fit keeps ``network<states>.h5`` and writes
that round's network as ``network<states>.final.h5`` with its json. Before its
first round the trainer runs
:func:`~aimmd.cgaimmd.restart.check_cgaimmd_restart` with the descriptor check
and writes ``initial_network`` (`Restart rules`_).

What a C-GAIMMD fit promises
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Every round starts from the same weights**: ``initial_network`` when it is
  set, else ``reset_parameters()`` under the run's seed. Optimizer, EMA and
  normalizers are new every round. Nothing of the previous round's weights
  enters a round, neither as start nor as labels.
- The data come from :func:`aimmd.cgaimmd.dataset.extract_dataset` on the
  trainer's path ensemble, in the descriptor series
  ``params.descriptors_series``; the extraction is cached in memory between
  rounds of one trainer process (an explicit cache keyed by run folder,
  states, series and atom count), never the weights.
- The round number continues from ``network<states>.json``
  (:func:`aimmd.cgaimmd.fitting.previous_round`), so it survives trainer
  restarts.
- The seed is :func:`aimmd.cgaimmd.fitting.fit_seed`: the fit's own argument,
  else ``generation_seed``, else a hash of the run folder; round ``r`` derives
  its own seeds from it.
- The metadata name every setting, the keywords passed, the seed and its
  source, the start (file and sha256) and the series.
- Inside :func:`aimmd.cgaimmd.fitting.settings_only` the fit validates its
  keywords and raises :class:`aimmd.cgaimmd.fitting.FitSettings` instead of
  training (the preflight's fit check).
- In the ``kinetics_convergence`` task, a fit that keeps state between calls
  keeps it in ``worker.kinetics_convergence_state_dir``, and every data
  fraction starts from ``initial_network``.

The flow-matching fit
~~~~~~~~~~~~~~~~~~~~~

:func:`~aimmd.cgaimmd.flow_matching.fit.flow_matching_fit`, one call per
round:

1. **Set-up.** The training device, the conditioning range :math:`L`, the
   state folder ``flow_fit<states>/`` with ``state.json``, the round number,
   the seed and the round seed (``seed + 1000 (round - 1)``).
2. **Data.** Missing committor values of reactive frames are computed with the
   run's current network (the only use of its weights), AIMMD's reweighting
   (with the trainer's bins) gives every path its density weight, and
   ``extract_dataset`` returns shooting points and frames. Held-out chains are
   chosen once (``holdout_fraction``, ``holdout_seed``) and kept.
3. **No data.** With fewer than ``min_basin_frames`` in-state frames of either
   state, the first round of a fresh run is a no-op and later rounds return
   ``[]`` after ``idle_sleep_s``.
4. **Start.** The start weights are loaded, and a new
   :class:`~aimmd.cgaimmd.flow_matching.fit.FlowTrainer` is built.
5. **Training.** The flow-matching labels of reactive frames are the
   committor of the round's EMA, taken at the start and again every
   ``relabel_every`` steps. ``steps_per_round`` steps follow; with
   ``holdout_every`` and ``early_stopping_patience`` the round stops after that
   many evaluations without improvement of the held-out chains' shooting
   likelihood, and the network ends as the best EMA. ``max_round_seconds``
   bounds the round's time.
6. **Records.** ``state.json``, a line of ``rounds.jsonl`` and the return
   value. The EMA is left in ``params.network``.

One step minimizes

.. math::

   L = N_q\left(w_{sp} L_{sp} + w_{gp} L_{gp} + w_a L_{anchor}\right) + w_{fm} N_{fm}(L_{FM})

with :math:`L_{sp}` the binomial likelihood of the two-way outcomes of the
shooting points, :math:`L_{gp}` a gradient penalty of the committor on the same
batch, :math:`L_{anchor}` the likelihood of in-state frames with the soft
targets :math:`\sigma(\mp L)`, :math:`L_{FM}` conditional flow matching on
frames of the reweighted ensemble (optimal-transport pairing, label dropout
``p_uncond`` for guidance), and :math:`N_q`, :math:`N_{fm}` divisions by a
running magnitude of each term, so that the weights mean the same for every
system and stage of training. A CUDA out-of-memory error restarts the round
from step 4 (the same start and seed), at most ``round_oom_retries`` times.
``split_backward`` (two backward passes per step, off by default) lowers the
peak memory for small GPUs. Graph mode
(``graphs=True``) feeds the committor terms through
``params.descriptor_transform``, by default cache-free batch dicts
(:class:`~aimmd.cgaimmd.flow_matching.graphs.FlowGraphTransform`); reading or
writing a SQLite graph cache is an explicit option.

End-state pretraining
(:func:`~aimmd.cgaimmd.flow_matching.pretrain.pretrain_end_states`) is the
same trainer without shooting points: anchors and flow matching on basin
frames conditioned at :math:`\mp L`, written as ``network<states>.h5`` with a
``pretrain.json`` report.

The BioEmu2 sequential fit
~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~aimmd.cgaimmd.bioemu2.fit.sequential_fit` runs one
:func:`~aimmd.cgaimmd.bioemu2.fit.sequential_round` per call:

1. **Data**, as for every fit.
2. **Features** of unmodified BioEmu2, computed once and stored in float16:
   every shooting point clean and ``noisy_copies`` times noised, up to
   ``n_anchor_max`` in-state frames clean and once noised, up to ``pool``
   frames clean for the generator.
3. **Head.** A fresh MLP with the frozen feature scaler of the initial
   network, trained on the two-way outcomes plus half-weighted in-state
   anchors, with an EMA, early-stopped on a random ``head_holdout`` share of
   the shooting points.
4. **Generator**, once ``min_shots_for_generator`` shooting points exist:
   fresh adapters (zero output layers, so they start as plain BioEmu2),
   ``adapter_steps`` steps of BioEmu2's denoising loss on pool frames labelled
   by the head of step 3, drawn so that ten label bins are equally likely,
   with label dropout ``p_uncond``, a ligand weight, a divergence guard and
   out-of-memory halving of the micro-batch.
5. **Label probe**, logged only: the trained adapters' loss with the frames'
   own labels, shuffled labels and none, which shows whether the adapters use
   the label.

Options, all off by default: labels averaged over a time window along each
trajectory (``label_time_window_ps``), a floor on the share of sparse label
bins (``bin_min``), and a learned null with label dropout for guidance.
:func:`~aimmd.cgaimmd.bioemu2.pretrain.pretrain_initial_network` writes the
initial network: identity adapters and a head (with its scaler) trained on
unbiased in-state frames.

Training data
-------------

:class:`~aimmd.cgaimmd.dataset.TrainingDataset` holds the shooting points
(descriptor rows, two-way outcomes toward each end state, chain number, file)
and the frames with positive weight (rows, state code: 1 for the first state,
2 for the last, 0 for R; density weight; chain; file; frame index; trajectory).

- A frame's weight is the sum of the AIMMD reweighting weights of the paths
  that contain it, times the stride with which its file is read
  (``free_stride`` outside chain folders, ``shot_stride`` for shots).
- A path is a shooting point if its single file lives in a chain folder, it
  has a shooting index and outcomes.
- The part files of one free trajectory form one trajectory, with frame
  indices that run across parts (the frame a restarted part repeats is counted
  once); every other file is a trajectory of its own.
- ``exclude_frames`` removes frames (and shooting points) by file and index.
- Rows are read from ``{trajectory}.{series}.npy`` of the run's series. Files
  whose frame counts did not change since the previous round are not read
  again.
- :func:`~aimmd.cgaimmd.dataset.time_averaged_labels` averages committor
  probabilities over a time window along each trajectory (an option of the
  fits).

Restart rules
-------------

At every start of the trainer and the Launcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~aimmd.cgaimmd.restart.check_cgaimmd_restart` with
``descriptors=True``:

- a ``chain*/tps_weights.npy`` (TPS history) is refused
  (:data:`~aimmd.cgaimmd.restart.TPS_RESTART_AS_CGAIMMD`): the chains would mix
  TPS multiplicities with RFPS-style weights;
- the descriptor caches of the params' series are checked on a random sample of
  frames: the width must be ``3 M`` and recomputed rows must agree with cached
  rows within :data:`~aimmd.cgaimmd.restart.CACHE_CHECK_ATOL` (0.01 Å, one
  step of the xtc precision; registered paths are written at that precision
  while their rows come from the segments);
- for the default series ``'descriptors'``, the run folder records the
  ``GeneratorDescriptors`` settings its caches were written with
  (``cgaimmd_descriptors.json``, :func:`~aimmd.cgaimmd.restart.check_descriptor_record`).
  Other settings later are refused, and so is a run with a water cluster whose
  caches predate the record. Every C-GAIMMD worker task writes or checks the
  record before it writes a cache.

The Launcher also requires a transition in every initial path when an end
state is shot internally (``state1_mode`` or ``state2_mode == 'shoot'``), and a
network file when a sampling-only launch (``nrounds=0``) starts a C-GAIMMD
shooter. These checks are skipped inside a worker task, where every worker
runs its own.

The initial network
~~~~~~~~~~~~~~~~~~~

:func:`aimmd.worker.utils.write_initial_network` (trainer and Launcher, any
chain type) writes ``initial_network`` as ``network<states>.h5`` when that file
does not exist (``'written'``), keeps an existing file that loads into the
network (``'kept'``), and moves one that does not load aside as
``network<states>.replaced<k>.h5`` (with its json and ``flow_fit<states>/``)
before writing. Without ``initial_network`` it does nothing (``'unset'``).

Descriptor series
~~~~~~~~~~~~~~~~~

Two ways keep caches of different descriptor settings apart:

- **A series of its own.** ``GeneratorDescriptors.series`` is
  ``'descriptors-cg'`` plus ten hex characters of a SHA-256 over
  ``GeneratorDescriptors.spec()``: the row layout, the generator atoms,
  whole-making, the structure's atom count, names, residue names and masses,
  and for a cluster the version of the water placement. With
  ``descriptors_series = DESCRIPTORS.series`` every change of these starts a
  new series. The object registers its series with :mod:`aimmd.core.series`
  when it is built; with ``refill=True``, one process of a job computes the
  missing rows before any MD, while the others wait; with ``refill=False``,
  a job with missing rows stops with a message that names the command that
  fills them.
- **The default series with a record**, as above. An RFPS history under
  ``'descriptors'`` (the source network's descriptors) is set aside with
  :func:`~aimmd.cgaimmd.restart.migrate_descriptors` (renamed with the suffix
  ``pre_cgaimmd``, the record with them), and the trainer recomputes the
  rows.

Post-training
~~~~~~~~~~~~~

:func:`~aimmd.cgaimmd.restart.derive_run` mirrors a run into a new folder:
trajectories are linked (``'symlink'``, ``'hardlink'`` or ``'copy'``), the
caches that stay valid (states, bias) are copied, descriptor caches are
recomputed or linked after a check, and the source keeps its network, bins
and densities. A trainer in the new folder then trains as many rounds as
asked, each from ``initial_network``. The source run's ``chain_type`` is kept,
so its paths keep their weights.

What else continues across a restart
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The fit's round counter (``network<states>.json``) and its held-out chains
  (``flow_fit<states>/state.json``).
- The seeds, as long as the run folder's path or ``generation_seed`` is the
  same.
- In-flight records (`Reconciliation after a restart`_).
- The source's pool does not: a restarted worker starts with an empty pool
  and refills.

Preflight internals
-------------------

:func:`~aimmd.cgaimmd.preflight.run_preflight` loads the params with
``Params.load(..., save=False)`` (nothing is written next to the params file),
changes into the params file's folder so that relative files resolve as in a
worker, and runs the checks in the order of :doc:`cgaimmd_production`. It uses
the shooting worker's own start-up functions
(``build_checked_generator``, ``check_generator_atoms``,
``check_start_up_numerics``) and ``prepare_candidate``, so it checks exactly
what a worker runs. The candidate is generated at a target of 0 with
``candidate_context`` in a scratch folder. A file-mode bias is evaluated the
way the shots evaluate it: the frame becomes a ``.tpr`` through
``params.initialize_simulation`` and runs for zero steps with the run's mdrun
and ``-plumed``; ``bias_function`` reads the resulting ``COLVAR``.

``Launcher.create_job(..., preflight=True)`` writes ``"${PYTHON}" -m
aimmd.cgaimmd.preflight <params file> --run <run folder>`` for every C-GAIMMD
run before the workers; a non-zero exit status ends the job.

The PLUMED bias
---------------

:mod:`aimmd.cgaimmd.plumed` reads the run's PLUMED setup as the shots see it:
the ``-plumed`` file of ``gmx_mdrun`` (resolved against the params folder for
the record, flagged when relative), the files the input reads (``FILE=``,
``STATE_RFILE=`` and similar keywords) with their sha256, and the files it
prints to. Shots and free simulations run with the bias; AIMMD moves each
half's ``COLVAR`` next to it, and ``bias_function`` turns it into the
per-frame bias cache. Preparation removes ``-plumed``. The trainer's
bias-reweighted rates, the zero-point convention and the Tiwary-Parrinello
factors are AIMMD's (:doc:`advanced`). The fits read in-state frames without a
bias weight.

Old names
---------

Runs made with earlier versions of this code wrote these names into their
params files, sidecars and markers. ``aimmd.cgaimmd._legacy`` holds every
old-to-new table; each old name in use emits one ``DeprecationWarning`` per
process that names the new spelling, and only new names are written.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - old
     - new
   * - ``chain_type='flow'`` (any value starting with ``'fl'``)
     - ``chain_type='cgaimmd'``
   * - ``flow_parameters``, ``flow_validity_function``,
       ``flow_prepare_function``, ``flow_minimize_function``,
       ``flow_lambda_sampler``
     - ``cgaimmd_parameters`` and the ``cgaimmd_*`` hooks
   * - ``flow_parameters`` keys ``ode_sampler``, ``ode_steps``,
       ``generate_chunk``
     - ``cgaimmd_generator = {'method': 'ode', 'sampler': ..., 'n_steps': ...,
       'chunk': ...}``
   * - the package ``aimmd.flow`` and ``aimmd.network.flow_fit``,
       ``aimmd.network.joint_painn``
     - :mod:`aimmd.cgaimmd` and its subpackages (compatibility modules
       resolve the old names)
   * - ``restrain='flow_heavy'``, ``'flow_all'``
     - ``'generator_heavy'``, ``'generator_all'``
   * - sidecars ``*.xtc.flow.json``, schema ``aimmd.flow.shot/1``
     - ``*.xtc.cgaimmd.json``, ``aimmd.cgaimmd.shot/1``
   * - in-flight markers ``.flow_shooting_point.{json,npy}``
     - ``.cgaimmd_shooting_point.{json,npy}``
   * - sidecar generator labels ``'network.generate'``, ``'aimmd sampler'``
     - ``'bioemu2_edm'``, ``'ode'``

Networks of earlier versions without ``default_generator()`` get the method
their old worker used (``'bioemu2_edm'`` for a network with its own
``generate``, else ``'ode'``). BioEmu2 networks of earlier runs load into
:class:`~aimmd.cgaimmd.bioemu2.network.BioEmu2Network` through
:func:`~aimmd.cgaimmd.bioemu2.network.load_archived_network`. The fit-state
folder ``flow_fit<states>/`` keeps its name: it belongs to the flow-matching
technique, not to the method.

Tests
-----

The C-GAIMMD tests are ``tests/test_cgaimmd_*.py`` (the method) and
``tests/test_flow_matching_*.py`` (the technique). Markers select the lanes:

``slow``
   Long end-to-end runs (the launched toy campaign, kinetics convergence,
   GROMACS runs, executing tutorial 3); ``--runslow``.
``graph``
   Needs PyTorch Geometric; ``--rungraph``.
``gromacs``
   Starts ``gmx``; skips itself without GROMACS.
``bioemu2``
   Needs the ``bioemu2`` extra; uses a tiny random-weight BioEmu2 built from a
   configuration in ``tests/data`` (no download).
``bioemu2_weights``, ``gpu``
   Need the real BioEmu2 weights or a GPU; run with
   ``devtools/run_weights_tests.sh``.

The in-process toy campaign of ``tests/test_toy_1d_cgaimmd.py`` is the fastest
end-to-end test of the whole loop (generation, preparation, final check, shot,
sidecar) and runs in the default lane.
