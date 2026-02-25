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
    Collect all parameters for an AIMMD run and analysis.

    This object is the central configuration container in AIMMD. It supports:
    - declarative parameter fields (dataclass fields),
    - strict validation when assigning or updating fields,
    - loading from a Python parameter script,
    - saving back to a Python script with embedded source for callables.

    Important required fields
    -------------------------
    - `states_function` must be provided by the user (no default).
    - `initial_paths` are required for running AIMMD (to initialize the ensemble).

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
    The actual constructor is provided by `ParamsHelpers._init` and is assigned
    below to preserve a clean API while keeping initialization logic in the
    helper mixin.
    """

    # Alias mixin implementations into the final public class.
    __init__ = ParamsHelpers._init
    __repr__ = ParamsMagic.__repr__
    __eq__ = ParamsMagic.__eq__


__all__ = ['Params']
