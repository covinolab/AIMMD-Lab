"""
...
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
    """Collects all parameters file for an AIMMD run and analysis.
    
    All fields have a detailed descriptions, most of them come with a default.
    However, "states_function", must be specified during initialization.
    "initial_paths" are also required if you use "params" for an AIMMD run.
    
    Usage
    -----
    >>> import aimmd
    >>>
    >>> # minimal initalization without parameters file
    >>> params = aimmd.Params(states_function=states_function,
                              initial_paths=initial_paths)
    >>> # saves params.py if not existing, otherwise new file
    >>>
    >>> # initialization with parameters file "params.py"...
    >>> # (this allows for more complex functions definitions)
    >>> params = aimmd.Params("params.py")
    >>> params = aimmd.Params.load("params.py")
    >>> # the two methods above are equivalent
    >>>
    >>> # initialization where some "params.py" parameters are oversubscribed
    >>> params = aimmd.Params("params.py", initial_paths=initial_paths)
    """
    
    __init__ = ParamsHelpers._init
    __repr__ = ParamsMagic.__repr__
    __eq__ = ParamsMagic.__eq__

__all__ = ['Params']
