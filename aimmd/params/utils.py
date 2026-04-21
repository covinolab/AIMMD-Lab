"""
aimmd.params.utils
=================

Utilities supporting `aimmd.params.Params`.

This module provides helpers used during parameter validation, saving, and
serialization of callable objects.

Provided utilities
------------------

update_source(instance, name)
    Attach a `__source__` string to a function/class instance so that
    `Params.__str__` / `Params.save` can reproduce or import it.

create_default_values_function(network, descriptor_transform=None)
    Build a default `values_function` that:
    - accepts numpy-like input (coordinates/descriptors),
    - optionally applies `descriptor_transform`,
    - evaluates `network` in inference mode,
    - returns a 1D numpy array of outputs.

Notes
-----
These helpers are intentionally lightweight and do not depend on higher-level
AIMMD components.
"""

# external
import numpy as np
import torch
from re import split
from types import FunctionType as Function
from inspect import getsource


def update_source(instance, name):
    """
    Attach a reproducible source/import snippet to a callable-like object.

    This function ensures `instance` has a `__source__` attribute, which is
    used by `Params.__str__` and `Params.save()` to serialize functions and
    classes into a Python parameters file.

    Parameters
    ----------
    instance : object
        A function or class instance intended to be stored in `Params`.
        If `instance` is a function, this captures its source code (if defined
        in `__main__`) or generates an import line (if defined in a module).
        If `instance` is not a function, this generates an import line for its
        class (assumes the class is defined in an importable module).
    name : str
        The attribute name under which this object is stored in `Params`.
        This name is used when generating import aliases or lambda assignments.

    Returns
    -------
    None

    Notes
    -----
    - If `instance` already has `__source__`, nothing is done.

    - For local module handling, module paths are reduced to their basename
      by splitting on "/". This matches the loader’s “local module name”
      remapping strategy implemented in `ParamsIO.load`.
    - Lambda functions are handled by rewriting their source into an assignment:
      `name = lambda ...`.
    """

    cls = instance.__class__

    # In order to import functions and classes correctly, `Params.__str__`
    # needs a stable representation for each callable/class.

    # Not a function: cannot be retrieved from main.
    # The object's *class* must be importable and defined in the same file too.
    if not isinstance(instance, Function):
        # When "local" import, omit path
        module = split(r'[\\/]', cls.__module__)[-1]
        instance.__source__ = f'from {module} import {name}\n'

    # Function defined not in "__main__": represent by import statement.
    elif instance.__module__ != '__main__':
        if hasattr(instance, '__source__'):
            return
        module = split(r'[\\/]', instance.__module__)[-1]
        if instance.__name__ in (name, '<lambda>'):
            instance.__source__ = f'from {module} import {name}\n'
        else:
            instance.__source__ = (f'from {module} import '
                                   f'{instance.__name__} as {name}\n')
    
    # Function defined in "__main__": check name is correct.
    elif instance.__name__ not in (name, '<lambda>'):
        raise TypeError(f'please rename function '
                        f'{instance.__name__!r} to {name!r}')
    
    # Function defined in "__main__": embed its full source code.
    else:
        instance.__source__ = getsource(instance).lstrip()  
        
        # Process lambda functions: represent as an assignment to preserve name.
        if (not instance.__source__.startswith('def ')
            and 'lambda' in instance.__source__):
            instance.__source__ = (f'{name} = lambda' +
                instance.__source__.split('lambda', 1)[1])


def create_default_values_function(network, descriptor_transform=None):
    """
    Create the default `values_function` used by AIMMD params given a neural
    network object. The `network` object will always remain the same, but you
    can update its weights after training the desired reaction coordinate.

    The returned function:

    - accepts numpy-like input (coordinates or descriptors),
    - optionally applies `descriptor_transform`,
    - converts to a torch tensor on the network's device/dtype,
    - flattens per-frame inputs to 2D (batch, features),
    - evaluates the network in inference mode,
    - returns a 1D numpy array.

    Parameters
    ----------
    network : torch.nn.Module
        Neural network whose output defines the “values” (e.g., logit committor).
        The function uses the device and dtype of `network.parameters()`.
    descriptor_transform : callable, optional
        Transform applied to input data *before* conversion to torch.
        If None, identity is used.

    Returns
    -------
    callable
        A function `values_function(input_data)` returning a 1D numpy array.

    Notes
    -----
    - Uses `torch.inference_mode()` for performance and to disable autograd.
    - Calls `network.eval()` to disable dropout and freeze batchnorm behavior.
    - Uses `torch.flatten(start_dim=1)` to ensure a (N, F) shaped input.
    """
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    if descriptor_transform is None:
        descriptor_transform = lambda x: x
    
    def values_function(input_data):
        """
        Compute values from numpy-like input using the provided network.

        Parameters
        ----------
        input_data : array-like
            Input frames. Typically shape (N, ...) where N is the number of frames.
            The function will flatten each frame into a 1D feature vector.

        Returns
        -------
        numpy.ndarray
            1D array of length N containing network outputs.

        Notes
        -----
        The network output is converted to CPU and flattened with `.ravel()`.
        """
        network.eval()  # disables dropout, batchnorm training behavior
        with torch.inference_mode():  # faster than no_grad()
            input_data = descriptor_transform(np.asarray(input_data))
            input_data = torch.tensor(
                input_data, dtype=dtype, device=device)
            input_data = torch.flatten(input_data, start_dim=1)
            output = network(input_data)
        return output.cpu().numpy().ravel()

    return values_function
