"""
aimmd.params
============

Parameter handling for AIMMD runs and analyses.

This package defines :class:`aimmd.params.Params`, a dataclass-like composite
object that:

- stores all configuration needed for AIMMD sampling and analysis,
- supports loading/saving parameters from/to a Python file (``params.py``),
- supports embedding callable objects (state/descriptor/value functions),
- validates types and internal consistency on assignment,
- provides convenience accessors (properties) and path-ensemble loaders.


Implementation overview
-----------------------
The public :class:`~aimmd.params.Params` class is built via multiple mixins:

- ParamsFields      : dataclass field definitions + descriptions
- ParamsMagic       : magic methods (__repr__, __eq__, __setattr__, __str__)
- ParamsHelpers     : initialization and validation helpers
- ParamsProperties  : derived properties (masses, pipeline, parent, ...)
- ParamsMethods     : engine-dependent simulation operations
- ParamsPaths       : loading trajectories/chains into PathEnsemble
- ParamsIO          : load/save from/to a Python file


Notes
-----
- The `Params` class is intended to be the single entry point for users.
- Many fields are designed to be serialized into a Python file to preserve
  complex callables (neural network classes, functions, transforms, etc.).
"""

from ._io import ParamsIO
from ._magic import ParamsMagic
from ._paths import ParamsPaths
from ._fields import ParamsFields
from ._helpers import ParamsHelpers
from ._methods import ParamsMethods
from ._properties import ParamsProperties


class Params(
    ParamsFields,
    ParamsMagic,
    ParamsHelpers,
    ParamsProperties,
    ParamsMethods,
    ParamsPaths,
    ParamsIO):
    """
    Central configuration object for AIMMD runs.

    A :class:`Params` instance is the single source of truth for a run’s
    configuration: paths, engine settings, analysis pipeline, neural-network
    committor model, sampling controls, and scheduler metadata.

    In the AIMMD architecture, Params is consumed by both:

    - :class:`~aimmd.launcher.Launcher`, which uses it to build execution plans
      and write run directory layouts, and
    - :class:`~aimmd.worker.Worker`, which uses it to execute tasks such as
      ``shoot``, ``free``, and ``train``.


    Typical responsibilities
    ------------------------

    - define end states and state processing conventions,
    - define the analysis pipeline (states/descriptors/values),
    - provide engine integration hooks (initialize_simulation/run_simulation),
    - hold the committor model (e.g., torch network) and training routine,
    - persist and reload run state (paths, bins/densities, network snapshots),
    - carry scheduler hints (e.g., ``slurm_header``).

    Usage
    -----
    >>> import aimmd
    >>>
    >>> # minimal initialization without a parameters file
    >>> params = aimmd.Params(states_function=states_function,
    ...                       initial_paths=initial_paths)
    >>>
    >>> # initialization from a parameters file "params.py"
    >>> params = aimmd.Params("params.py")
    >>> params = aimmd.Params.load("params.py")  # equivalent
    >>>
    >>> # override some parameters from file
    >>> params = aimmd.Params("params.py", initial_paths=initial_paths, nbins=5)

    Notes
    -----
    The concrete API (attributes and helper methods) is defined across the
    Params mixins in :mod:`aimmd.params`. See the documented Params submodules
    for attribute-level semantics.
    """

    # Alias mixin implementations into the final public class.
    __init__ = ParamsHelpers._init
    __repr__ = ParamsMagic.__repr__
    __eq__ = ParamsMagic.__eq__


__all__ = ['Params']
