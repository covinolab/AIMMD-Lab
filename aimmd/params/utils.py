"""
...
"""

# external
import numpy as np
import torch
from dill.source import getsource

# aimmd imports
from types import FunctionType as Function

# utils
def update_source(instance, name):
    """in order to import functions and classes correctly"""
    
    if hasattr(instance, '__source__'):
        return
    
    cls = instance.__class__
    
    # not a function: cannot be retrieved from main
    # class must be defined in same file, too
    if not isinstance(instance, Function):
        # when "local" import, omit path
        module = cls.__module__.split("/")[-1]
        instance.__source__ = f'from {module} import {name}\n'
    
    # function defined not in "main"
    elif instance.__module__ != '__main__':
        # when "local" import, omit path
        module = instance.__module__.split("/")[-1]
        if instance.__name__ == name:
            instance.__source__ = f'from {module} import {name}\n'
        else:
            instance.__source__ = (f'from {module} import '
                                   f'{instance.__name__} as {name}\n')
    
    # function defined in "main"
    else:
        instance.__source__ = getsource(instance).lstrip()
        
        # process lambda functions
        if (not instance.__source__.startswith('def ')
            and 'lambda' in instance.__source__):
            instance.__source__ = f'{name} = lambda' + (
              'lambda'.join(instance.__source__.split('lambda')[1:]))


def create_default_values_function(network, descriptor_transform=None):
    """
    Default values function from network
    """
    
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    descriptor_transform = descriptor_transform or (lambda x:x)
    
    def values_function(input_data):
        """
        From *numpy* descriptors to *numpy* output.
        Evaluation mode.
        """
        network.eval()  # disables dropout, batchnorm training behavior
        with torch.inference_mode():  # faster than no_grad()
            input_data = descriptor_transform(np.asarray(input_data))
            input_data = torch.as_tensor(
                input_data, dtype=dtype, device=device)
            input_data = torch.flatten(input_data, start_dim=1)
            output = network(input_data)
        return output.cpu().numpy().ravel()
    
    return values_function
