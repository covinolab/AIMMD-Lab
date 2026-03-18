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

``nbins``, ``cutoff_min``, ``cutoff_max``, ``marginal_bins``
   Define how value space is discretized for adaptive sampling and density
   estimation.

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
