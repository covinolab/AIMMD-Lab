Analysis and estimators
=======================

This page says which estimators give valid rates and free energies for a
C-GAIMMD run, what each of them rests on, and what a static bias changes. It
then lists the analyses a study of a C-GAIMMD run needs, from the run
progression to the committor against brute-force shooting, with the functions
that make each of them. Tutorial 7 (:doc:`tutorials/7_cgaimmd_analysis`) runs
every one of them on a small run of a toy system, on a CPU; the API pages
(:doc:`api/analysis` and :doc:`api/cgaimmd_analysis`) define every number in
detail.

.. contents:: On this page
   :local:
   :depth: 2

Terms used on this page
-----------------------

Committor
   :math:`p_B(x)`, the probability that a trajectory started from
   configuration :math:`x` with random velocities reaches state B before state
   A. Its logit :math:`\lambda = \ln\left(p_B / (1 - p_B)\right)` is what
   AIMMD's networks output; :math:`\lambda^*` is the committor a shooting
   worker asks the generator for (the *request*).
Shot
   A two-way shot: two trajectories from one shooting point with fresh
   Maxwell-Boltzmann velocities, one integrated forward, the other backward,
   each until it reaches a state. Its *halves* are these two trajectories.
Free simulation
   Plain MD started inside a state. Its *excursions* are its pieces between
   leaving a state and reaching one.
Reweighting
   AIMMD's estimate of how much each shot and each excursion counts in
   equilibrium (its *path weight*). Every frame of a path carries that path's
   weight; summed over frames, these *equilibrium weights* give populations and
   free energies.
Shooting-point law
   The probability with which a configuration becomes a shooting point. RFPS
   draws shooting points from the frames of earlier paths; C-GAIMMD generates
   them, prepares them and accepts them with a final check.
Brute-force set
   Configurations whose committor was measured by shooting many trajectories
   from each of them, independently of any model.

What a run folder holds, and what the tools do with it
-------------------------------------------------------

A C-GAIMMD run folder holds what every AIMMD run folder holds: the shots of
the shooting chains (``chainR*/path*.xtc``) with their per-frame caches
(states, descriptors, committor values), the free simulations
(``freeA``, ``freeB``), the trainer's network files and its log. Next to every
shot it also holds a sidecar (``path*.xtc.cgaimmd.json``), the record of how its
shooting point was made: the request, the committor of the raw and of the
prepared candidate, the acceptance rule, the network round, the generator and
its settings, the rejected candidates, the preparation and the cost.

The analysis tools only read a run folder. They return objects; whatever is
saved goes where you say, and a target inside a run folder is refused. They
do not leave index or lock files next to the trajectories, so an archived run
on a read-only file system is read as it is. Analyse a finished run, or a copy
of a running one: while a worker updates a cache, a read can mix old and new
rows.

Rates and free energies: which estimators apply
-----------------------------------------------

Every two-way shot of a C-GAIMMD run is an ordinary unbiased shot: once the
shooting point exists, it is registered, scored and reweighted exactly like an
RFPS shot. What differs is how the shooting point came about. That decides
which rate estimators can use the shots.

AIMMD's reweighting
^^^^^^^^^^^^^^^^^^^

**How it works.** The shots and the free-simulation excursions are ordered by
the committor values of their frames. For every committor level, the
probability that an excursion leaving state A climbs to the next level is
estimated from how many of the excursions that reached one level went on to
the next: a product over levels, built like a survival curve. Every path is
then weighted so that the weighted ensemble has these crossing probabilities.
The weighted ensemble stands in for one long equilibrium trajectory: the rate
:math:`k_{AB}` is the inverse of the mean time that trajectory spends with A as
the last state it visited before it reaches B, and :math:`k_{BA}` likewise.
The trainer does this after every round and prints both rates;
:func:`aimmd.analysis.reweighting.reweight_run` repeats it offline by calling
the same code.

**What it rests on.**

1. *The dynamics after the shooting point are those of the system.* Each half
   of a shot is plain MD from its first frame on. This holds for C-GAIMMD:
   generation, the validity check and the preparation act on the shooting
   point only.
2. *Progress along the committor has no memory at the frame interval.* The
   chance that an excursion at one level goes on to the next is taken to
   depend on the level only, not on how fast or from where it got there. This
   holds for diffusive dynamics saved at frame intervals that are long
   compared with the memory of the velocities and short compared with the time
   to move by about one logit, which is the usual case for solvated molecules
   saved every few picoseconds. Dynamics that cross the barrier within a few
   frames (a small molecule in vacuum, little friction) violate it: the rate
   out of the state that is left then comes out too low; frames that are far
   apart bias it the other way. When at least ``free_threshold`` (20 by
   default, in ``params.reweight_parameters``) free-simulation excursions from a
   state reached the other state, the crossing probability out of that state
   comes from the free simulations alone, and the rate out of it is in effect
   their direct count.
3. *Every committor level between the states is crossed* by enough shots or
   excursions.
4. *At each committor level, the shooting points are distributed as
   equilibrium configurations at that level.* How many shooting points sit at
   each level is free: the reweighting corrects for it, which is why RFPS and
   C-GAIMMD may choose their levels. Among configurations with the same
   committor value, however, the shooting points should be distributed as in
   equilibrium. If the committor were exact and condition 2 held exactly, this
   would not matter, since every configuration at one level would then have the
   same future; with a learned committor it matters. For C-GAIMMD it is what the
   generator is trained toward, and what generator adherence, sample quality
   and shooting validation measure (below). Where a generator departs from it,
   the weights, the free energies and the rates carry an error that the
   reweighting itself cannot see.
5. *The numbers depend on the committor values used.* The levels are those of
   a network. Cached values (each frame scored by the network of the round that
   scored it) and recomputed values (every frame scored by one network) give
   different numbers, and another network gives different numbers again.

**What it does not need** is the shooting-point law across levels. The target
law of a C-GAIMMD run decides how many shots land at each level, and the final
acceptance check, which accepts a candidate only if the network puts it near
its request, reshapes that further; neither biases the reweighting, as long as
condition 4 holds within each level.

Estimators that need the shooting-point law
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two estimators do without condition 2:

- *exact path importance sampling* weights every path by the inverse of the
  sum, over its frames, of the ratio between the probability that the frame
  was proposed as a shooting point and its equilibrium probability;
- *path-ensemble MBAR* (the multistate Bennett acceptance ratio applied to
  paths) needs the shooting points to be distributed as in equilibrium within
  each window of a recorded placement coordinate, and the windows recorded.

Both therefore need to know, for every frame of every path, how likely it was
to become a shooting point. A C-GAIMMD run cannot tell:

- the generator's probability density is not evaluated during a run (the
  diffusion sampler with guidance has no tractable density at all);
- the validity check and the preparation (re-solvation, restrained
  minimisation, a short relaxation) change each sample in ways whose effect on
  the density is not known;
- the final check accepts or rejects candidates by the network's committor;
- the network, and with it the generator, changes every round.

These estimators therefore do not apply to C-GAIMMD shots, and the package
provides neither. The sidecars record what is known about each shooting point,
so that the law can be studied later.

Rates counted in the free simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`aimmd.analysis.free_md.free_md_rates` counts the transitions of the
free simulations and the time they spent on each side: :math:`k_{AB}` is the
number of transitions from A to B divided by the time spent with A as the last
state, with an exact 95 % Poisson interval. It needs no committor and no
shots, so it is the model-free check of the reweighted rates. It rests on one
condition: every free simulation must restart from the equilibrium inside its
state (``free_restart_source = 'equilibrium'`` or ``'basin'``).
With ``'crossing'`` (the default), ``'transitions'`` or ``'seed'`` the restarts
sit on the state boundary or at fixed frames, the first-passage times are too
short, and the function warns. With few transitions the interval is wide.

The trainer's printed rates
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The trainer prints its estimates during the run, each with the network of its
round and with the AIMMD version that ran it.
:func:`aimmd.analysis.reweighting.trainer_rate_estimates` reads them, with the
round and the number of shots of each, which gives the online kinetics
convergence. For the numbers you report, use ``reweight_run``:

- ``values='cached'`` uses the committor values stored in the run and
  reproduces the trainer's last estimate exactly when the run folder holds the
  data of that estimate. A run usually goes on after the trainer's last value
  pass; the frames written since have no stored value (a missing value would
  read as a committor of one half), so free simulations are cut at their first
  reactive frame without one and shots with such frames are left out
  (``result.left_out``).
- ``values='recompute'`` scores every reactive frame again, in memory, with one
  network (by default the run's ``network{states}.h5``). This is the choice for
  the final analysis: every frame then has a value from the same network.
  It is also the only choice for a run with ``params.subsample_caps``.

``reweight_run`` warns when more than 1 % of the counted frames have no known
state: frames that a worker wrote but had not yet labelled when the run
stopped. The trainer counts them like any frame, and so does ``reweight_run``
to reproduce it; check ``result.frames_without_state`` before you quote rates.

Under a static bias
^^^^^^^^^^^^^^^^^^^

A run can shoot under a static bias that acts only inside the states, such as
a frozen OPES bias that flattens a deep bound state (``record_bias = True``;
see :doc:`cgaimmd`, section "C-GAIMMD under a static bias"). Then:

- the reactive region must be unbiased. AIMMD's reweighting assumes that the
  bias is zero there, the trainer checks this every round, and the preflight
  checks the zero point before the run;
- the plain rates ``k12`` and ``k21`` are those of the accelerated dynamics.
  The rates of the unbiased system are the *bias-reweighted* ones
  (``k12_bias_reweighted``, ``k21_bias_reweighted``), which count every frame's
  time multiplied by :math:`e^{\text{bias}}` (bias in kT), the time rescaling
  of Tiwary and Parrinello;
- every frame's equilibrium weight is multiplied by :math:`e^{\text{bias}}`, so
  that populations and free energies describe the unbiased system (without it,
  the biased wells look shallower than they are);
- ``free_md_rates`` counts the boosted time :math:`\sum e^{\text{bias}}` of the
  free simulations: every biased frame stands for :math:`e^{\text{bias}}`
  frames of the unbiased dynamics;
- candidate preparation runs without the bias, so the shooting point itself is
  unbiased; but the fits see the biased in-state frames, and the generator
  learns the biased distribution inside the states;
- frames of biased simulations do not become training data, an input scaler or
  a committor reference unless you decide so: the distance-matrix baseline,
  ``free_md_reference``, ``heldout_chain_reference`` and ``sweep_reference``
  refuse a run with a recorded bias unless ``allow_biased_frames=True``.

Which analysis, which functions
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - analysis
     - functions
   * - rates, populations
     - :func:`~aimmd.analysis.reweighting.reweight_run`,
       :meth:`~aimmd.analysis.reweighting.RunReweighting.populations`,
       :func:`~aimmd.analysis.free_md.free_md_rates`
   * - free energy on a pair of collective variables, with the generated
       shooting points on top
     - :func:`~aimmd.analysis.reweighting.frame_table`,
       :func:`~aimmd.analysis.reweighting.free_energy`,
       :func:`~aimmd.cgaimmd.sidecar.shot_table` with ``cv=``
   * - run progression: requests, shooting points and outcomes over the run
     - :func:`~aimmd.cgaimmd.sidecar.shot_table`,
       :func:`~aimmd.cgaimmd.sidecar.summarise_shots`
   * - kinetics and free-energy convergence, with and without a bias
     - ``reweight_run(fractions=...)``,
       :meth:`~aimmd.analysis.reweighting.RunReweighting.convergence`,
       :func:`~aimmd.analysis.reweighting.trainer_rate_estimates`,
       :func:`~aimmd.analysis.reweighting.free_energy_convergence`
   * - committor parity against a brute-force set, next to a baseline model
     - :func:`~aimmd.cgaimmd.analysis.validate.sweep_reference`,
       :class:`~aimmd.cgaimmd.analysis.validate.CommittorReference`,
       :mod:`~aimmd.cgaimmd.analysis.committor_baseline`,
       :func:`~aimmd.cgaimmd.analysis.validate.score_committor`,
       :func:`~aimmd.cgaimmd.analysis.validate.parity_figure`
   * - requested against sampled committor
     - :func:`~aimmd.cgaimmd.analysis.shooting_validation.from_shot_table`,
       :func:`~aimmd.cgaimmd.analysis.shooting_validation.write_validation_frames`,
       :func:`~aimmd.cgaimmd.analysis.shooting_validation.sweep_outcomes`,
       :func:`~aimmd.cgaimmd.analysis.shooting_validation.level_summary`,
       :func:`~aimmd.cgaimmd.analysis.shooting_validation.dial_slope`
   * - generator adherence and sample quality
     - :func:`~aimmd.cgaimmd.analysis.validate.draw_samples`,
       :func:`~aimmd.cgaimmd.analysis.adherence.adherence_metrics`,
       :func:`~aimmd.cgaimmd.analysis.adherence.data_ceiling`,
       :func:`~aimmd.cgaimmd.analysis.adherence.sample_quality`,
       :func:`~aimmd.cgaimmd.analysis.adherence.structural_match`
   * - the whole protocol for one network: validity, committor accuracy,
       adherence, a memorisation check, a report and a model card
     - :func:`~aimmd.cgaimmd.analysis.validate.validate`,
       :func:`~aimmd.cgaimmd.analysis.validate.leaderboard`
   * - every setting of a run, for the tables of a supplement
     - :func:`~aimmd.analysis.run_card.run_card`,
       :func:`~aimmd.analysis.run_card.run_card_rows`

The sections below show each of them in a few lines. ``params`` is the run's
params (``aimmd.Params.load('params.py', save=False)``), ``run`` its run folder,
and ``out`` a folder of your own.

Rates and populations
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from aimmd.analysis.free_md import free_md_rates
   from aimmd.analysis.reweighting import reweight_run

   rates = reweight_run(params, run, values='recompute')
   print(rates.k12, rates.k21)                # per frame interval; divide by it for 1/time
   print(rates.populations())                 # shares of A, R, B and of the state histories
   counted = free_md_rates(params, run)        # the model-free check, with 95 % intervals
   print(counted['AB'], counted['BA'])
   rates.save(f'{out}/rates')                  # <out>/rates.json and rates.npz

The rates are per frame interval of the trajectories. The *populations* are
the shares of the equilibrium weight in each region; the *history* populations
are the shares by the last state visited, and their ratio equals
:math:`k_{BA}/k_{AB}` in a two-state system. A run with several systems gives
one result per system (``system_id=``).

Free energy on a pair of collective variables, with the shooting points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from aimmd.analysis.reweighting import frame_table, free_energy
   from aimmd.cgaimmd.sidecar import shot_table

   def q(reader):                              # any function of the frames of one file
       return [native_contacts(ts.positions) for ts in reader]

   def end_to_end(reader):
       return [distance(ts.positions) for ts in reader]

   cvs = {'Q': q, 'end-to-end': end_to_end}
   table = frame_table(rates, cv=cvs)          # every counted frame, with its weight
   fes = free_energy(table, bins=50)           # F = -ln p in kT, lowest bin at 0
   shots = shot_table(run, params=params, cv=cvs)   # the CVs of every shooting point

``native_contacts`` and ``distance`` stand for your own functions. A callable
gets an MDAnalysis reader over one trajectory file and returns one value per
frame; a callable written for ``frame_table`` works unchanged in
``shot_table``, where it gets the shooting frame. A collective variable that is
a column of a per-frame cache (the descriptors, the committor values, a named
descriptor series) is read much faster with ``cv_source='descriptors'`` (or the
cache's name) and a callable on the cache rows. Under a static bias the weights
include :math:`e^{\text{bias}}`, so ``fes`` is the free energy of the unbiased
system.

For a figure, draw ``fes.free_energy`` on ``fes.centers`` in grey and the
shooting points of ``shots`` as circles coloured by their committor: the value
the run gave them (``lambda_final``) or that of the final network. The rows of
``shot_table`` say which shots were taken before the final check was enforced
(``acceptance_enforced``); those come from a generator that had not yet been
trained.

Run progression
^^^^^^^^^^^^^^^

``shot_table`` returns one row per shot in registration order: the chain and
path, where the two halves ended (``first_state`` for the backward half,
``last_state`` for the forward half), the request ``lambda_target``, the
committor of the prepared frame ``lambda_final``, the acceptance rule, the
network round, the candidates tried and rejected by reason, and the time spent
on each stage. ``summarise_shots`` counts shots, transitions, path types,
acceptance rules and rejection reasons, and gives the cumulative number of
transitions against the shots. Plotting ``lambda_target`` and ``lambda_final``
against the shot index, coloured by outcome, shows how the run moved from the
bootstrap rule (any frame in the reactive region is accepted) to the final
check.

Kinetics and free-energy convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from aimmd.analysis.reweighting import free_energy_convergence, trainer_rate_estimates

   rates = reweight_run(params, run, values='recompute',
                        fractions=(0.2, 0.4, 0.6, 0.8), fraction_by='time')
   for row in rates.convergence():             # one row per part, and one for all data
       print(row['fraction'], row['n_shots'], row['k12'], row['k21'],
             row['k12_bias_reweighted'], row['k21_bias_reweighted'])
   online = trainer_rate_estimates(f'{run}/trainARB.log')   # the trainer's estimates
   parts = free_energy_convergence(rates, cvs, bins=50)      # rms distance to all data

A part holds the shots registered in the first part of the run's time
(``fraction_by='time'``) or the first shots (``'shots'``) and the same share of
every free simulation, all scored by the same network: it shows how far the
estimate still moves with the amount of data. It is not a re-run of the
training, which :meth:`aimmd.Worker.kinetics_convergence` does. With a bias,
the bias-reweighted columns give the convergence of the unbiased rates, and
the plain columns that of the accelerated dynamics. The free-energy distance
is the root-mean-square difference over the bins where both surfaces have
data, weighted by the probability of the full-data surface, after removing the
best constant shift (free energies are defined only up to a constant).

Committor parity against brute-force shooting, with a baseline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A brute-force set is computed once, independently of any model: an AIMMD
sweep shoots every configuration many times. Its params file is a plain RFPS
params file whose initial path holds the configurations, and the sweep runs
with ``Launcher.run(n, reactive_region_mode='sweep', nsteps=...)``, where ``n``
workers times ``nsteps`` is the total number of shots. Every worker loads the
params' network, so the sweep's params define one, as any AIMMD params file
does.

.. code-block:: python

   import torch

   from aimmd.cgaimmd.analysis import committor_baseline, validate

   reference = validate.sweep_reference(sweep_params, 'sweep_run',
                                        frames='configurations.xtc')
   # or a set computed elsewhere:
   # reference = validate.CommittorReference.from_npz('benchmark.npz')
   # reference = validate.CommittorReference.from_trajectory(
   #     'configurations.xtc', *validate.shot_counts('outcomes.npz'))

   # the C-GAIMMD model of the run, on the atoms it generates
   params.network.load_state_dict(torch.load(f'{run}/networkARB.h5', weights_only=True))
   cgaimmd = validate.network_committor(params.network)

   # an ordinary committor model trained on the same run: an MLP on heavy-atom distances
   heavy = [i for i, element in enumerate(elements) if element != 'H']
   scaler = committor_baseline.run_distance_scaler(params, run, heavy)
   mlp = committor_baseline.DistanceMLP(heavy, n_atoms=len(elements),
                                        mean=scaler['mean'], std=scaler['std'])
   committor_baseline.train_on_run(params, run, mlp, epochs=2000, augment='yes')
   torch.save(mlp.state_dict(), f'{out}/distance_mlp.pt')

   scores = {name: validate.score_committor(model, reference) for name, model in
             (('C-GAIMMD', cgaimmd), ('distance MLP', validate.network_committor(mlp)))}
   validate.parity_figure([(name, reference, logits) for name, (_, logits) in scores.items()],
                          f'{out}/parity')

``elements`` stands for the element of each atom the run's descriptors hold.
The reference's coordinates must hold the atoms the scored network reads, in
its order (``atoms=`` of the loaders selects them);
:func:`~aimmd.cgaimmd.analysis.validate.params_committor` instead scores any
network on whole configurations, through the run's descriptors function. The
numbers of ``score_committor``:

- **log-likelihood per endpoint**: the mean log-probability the model gives
  the observed end of each brute-force trajectory. Higher is better. A model
  that predicts the overall share of B everywhere scores the *constant*
  value; the *in-sample ceiling* is what the observed shares themselves score,
  and the estimate for the exact committor lies just below it.
- **calibration slope**: the slope :math:`b` of the best fit
  :math:`q = \sigma(a + b\lambda)` of the observed committor :math:`q` on the
  model's logit. 1 is calibrated; above 1 the model is too cautious (its
  committors are too close to one half), below 1 too confident.
- **Spearman**: the rank correlation of the model's and the observed
  committors: whether the model orders the configurations correctly.
- **R²**, next to its *noise ceiling*: with a finite number of trajectories per
  configuration, even the exact committor does not reach R² = 1; the ceiling
  is the R² it would reach on average on these counts.

Two other references need no extra simulation: the shots of chains the
trainer held out (:func:`~aimmd.cgaimmd.analysis.validate.heldout_chain_reference`;
two endpoints per configuration, so only coarse statistics) and the outcomes
of the run's own free simulations (:func:`aimmd.analysis.free_md.free_md_reference`
and :func:`aimmd.analysis.free_md.free_md_calibration`: every reactive frame of
a free simulation is followed by the state it reaches next, which tests a
network on configurations it never chose).

Requested against sampled committor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The test of a generator that needs no model: shoot from its samples and count
where the trajectories go. Two sources:

- **the run's own shots**, one two-way shot per generated configuration,
  through :func:`~aimmd.cgaimmd.analysis.shooting_validation.from_shot_table`.
  They cost nothing, but they passed the final check, which accepted a
  candidate only if the run's network put it near its request: they measure
  the generator and the check together;
- **fresh samples**, which measure the generator alone:

.. code-block:: python

   from aimmd.cgaimmd.analysis import shooting_validation as sv

   manifest = sv.write_validation_frames(params, f'{out}/frames', [-3, -2, -1, 0, 1, 2, 3], 32,
                                         run_dir=run)
   # shoot every frame of f'{out}/frames/shooting_frames.xtc' once, with a sweep
   sweep = sv.sweep_outcomes('sweep_run', manifest['meta']['n_frames'], params=sweep_params,
                             frames=f'{out}/frames/shooting_frames.xtc')
   rows = sv.validation_rows(manifest, sweep)
   per_request = sv.level_summary(rows)       # measured committor per request, with intervals
   dial = sv.dial_slope(rows)

``write_validation_frames`` draws samples at the requests with the run's
generator, prepares each with the run's own hooks and checks its state (the
committor part of the final check is left out), and writes the frames to shoot
and a manifest. A sample whose prepared frame lies in a state counts as a shot
with both halves there; a refused sample has no outcome. Per request,
``level_summary`` reports the measured committor (the share of halves that
ended in B), the share of shots whose halves went to different states, and the
committor variance among the samples of the request, computed from these two
under the assumption that the two halves of a shot are independent, as they
are for diffusive dynamics. The **dial slope** is the slope :math:`b` of a
logistic fit of every half's outcome on the request,
:math:`P(\text{half ends in B}) = \sigma(a + b\lambda^*)`: 1 for a generator
that delivers the committor it is asked for, 0 for one that ignores the
request. ``compare_dials`` compares two generators, paired by sample rank when
they shared seeds. Plotted as measured against requested committor with the
ideal line, this is the inset of a shooting-point figure.

Generator adherence and sample quality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adherence asks how closely a generator follows its requests when its samples
are *judged* by a committor model. The judge matters:

- the generator's **own committor** is what the final check sees; a generator
  can follow it and still miss the true committor;
- an **independent judge**, a committor model that shares no weights with the
  generator (the distance-matrix baseline above, or a model trained on held-out
  shots), is the fair test;
- every number is read against the **data ceiling** of the same judge
  (:func:`~aimmd.cgaimmd.analysis.adherence.data_ceiling`): training frames
  drawn at their own label, which is what a generator that reproduced its
  training data perfectly would score.

.. code-block:: python

   from aimmd.cgaimmd.analysis import adherence

   samples = validate.draw_samples(params.network, [-4, -2, -1, 0, 1, 2, 4], 64)
   requested = np.repeat(samples.levels, samples.positions.shape[1])
   judged = judge(samples.positions.reshape(-1, *samples.positions.shape[2:]))
   metrics = adherence.adherence_metrics(requested, judged)

``judge`` stands for the judge's committor on the samples (prepared as the run
prepares them when the judge needs whole configurations). The numbers:

- **slope** of the least-squares line :math:`\lambda = a + b\lambda^*`: 1 for a
  perfect dial, 0 for a generator that ignores the request;
- **resolution**: the robust spread of :math:`\lambda` around that line divided
  by the slope, in logits: how far apart two requests must be before their
  samples can be told apart. Smaller is better. It is reported over all
  requests, inside the *band* :math:`|\lambda^*| \le 2` (the transition region)
  and over the *wings*, which get one shared slope so that a jump between them
  does not count as following the request;
- **median slopes** over the *core* :math:`|\lambda^*| \le 1`, the band and the
  wings: a steep band slope can hide a flat core, where the samples only jump
  between the two sides;
- **neighbour AUC**: the probability that a sample of the higher of two
  neighbouring requests is judged above a sample of the lower one (0.5: the
  requests cannot be told apart; 1: they never overlap);
- **large misses**: the share of samples judged more than 3 logits from their
  request.

:func:`~aimmd.cgaimmd.analysis.adherence.sample_quality` asks whether the
samples look like MD at their committor. Structures are compared through
features, for a molecule the distances between its heavy atoms
(:func:`~aimmd.cgaimmd.analysis.adherence.pair_distance_features`; the distance
between two such feature vectors is the distance RMS, which needs no
alignment). The yardstick is a set of independent MD frames binned by their
committor label within :math:`\pm 0.5` of each request
(:func:`~aimmd.cgaimmd.analysis.adherence.label_binned_reference`):

- **variety ratio**: the median distance between two samples of one request,
  divided by the same for the MD frames at that committor. About 1: as varied
  as MD; well below 1: the generator collapses onto a few structures;
- **departure ratio**: the median distance of the samples to their nearest
  training frame, divided by the same for the MD frames. About 1: the samples
  stay where MD goes; well above 1: they leave it;
- **near-copies**: the share of samples closer to a training frame than two
  consecutive MD frames are to each other
  (:func:`~aimmd.cgaimmd.analysis.adherence.consecutive_frame_threshold`),
  compared with a baseline generator's share: a generator that recalls its
  training frames instead of making new ones.

:func:`~aimmd.cgaimmd.analysis.adherence.structural_match` compares the
distribution of one collective variable of the samples with that of the MD
frames at the same committor. When you compare the shooting points of two
methods structurally, compare them at matched committor (by each method's own
model and by shooting outcome), and run the same comparison on paths shot from
MD frames as the control: along a collective variable alone, differences in
committor show up as differences in structure.

The thresholds of these functions are arguments with documented defaults;
none of them returns a verdict. Whether a generator is good enough is your
decision, with the thresholds written next to the numbers.

The whole protocol for one network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~aimmd.cgaimmd.analysis.validate.validate` runs all of it for one network
and writes a JSON report, the sample arrays and a one-page model card: the
validity of raw and polished samples per request, the committor against every
reference given, adherence by the own and by an independent committor, and a
memorisation check (how close the samples come to training frames, against
held-out MD). :func:`~aimmd.cgaimmd.analysis.validate.leaderboard` tabulates
many reports, for example the networks of successive rounds of a run.

The run card
^^^^^^^^^^^^

.. code-block:: python

   import json

   from aimmd.analysis.run_card import run_card, run_card_rows

   card = run_card(params, run)
   json.dump(card, open(f'{out}/run_card.json', 'w'), indent=1)
   rows = run_card_rows(card, sections=['cgaimmd', 'fit'], latex=True, skip_none=True)

``run_card`` collects every setting of a run: every params field and the ones
that differ from the defaults, the engine, the network (class, size,
architecture, the digest of its file), the fit and the keywords of its last
round, the C-GAIMMD settings with their defaults filled in, the generator, the
hooks, the bias, and the software versions. ``run_card_rows`` flattens it into
(section, setting, value) rows, LaTeX-escaped on request, for the parameter
tables of a supplement.
