"""
aimmd.params._helpers
====================

Helper mixin for :class:`aimmd.params.Params`.

This module defines :class:`~aimmd.params._helpers.ParamsHelpers`, which provides
the *actual constructor* and the *type/consistency enforcement* logic for Params.

Key responsibilities
--------------------

- `_init`: constructor logic (supports loading from a params file, applying
  overrides, and populating defaults).
- `_setattr`: validated assignment for dataclass fields (type checking,
  callable handling, special-case parsing).
- `_process_and_check`: post-assignment validation that depends on multiple
  fields and/or existing data (e.g., consistency of initial paths, states,
  binning constraints, values function definition).

Notes
-----
- This is a mixin class; it is not intended to be instantiated directly.
- Most validation errors are surfaced as `TypeError` with a short explanation.
- The callable serialization mechanism relies on `update_source` to attach
  a `__source__` attribute to functions/classes for reproducible saving.
"""

# external
import os
import numpy as np
import torch
import inspect
from abc import ABC
from types import MethodType as Method
from typing import List, Callable
from numbers import Integral
from pathlib import Path as PosixPath
from MDAnalysis import Universe
from collections.abc import Iterable

# aimmd imports
from .utils import update_source, create_default_values_function
from ..path import Path
from ..pathensemble import PathEnsemble
from ..network.rescalable import Rescalable as RescalableNetwork


# params' helpers
class ParamsHelpers(ABC):
    def _init(self, *args, **kwargs):
        """
        Initialize a Params instance.

        This initializer supports two usage patterns:

        1) Load from file (preferred for reproducibility):

           - If `args[0]` is a valid path to an existing file, it is treated as
             a params file and loaded via `Params.load(...)`.

        2) No file provided:

           - If no valid file is provided, defaults are loaded from `'params.py'`
             (if present), otherwise initialization proceeds using defaults and
             provided keyword arguments.

        Parameters
        ----------
        *args
            If `args[0]` is a path-like string and exists, it is used as the
            params filename and removed from `args` before passing on.
            Remaining positional args are interpreted by `Params.load` as
            positional overrides in dataclass field order.
        **kwargs
            Field overrides applied with higher priority than file values.

            Special behavior:

            - If `filename` is in `kwargs`, it is treated as a params file
              (handled upstream in `ParamsIO.load`).

        Returns
        -------
        aimmd.params.Params
            The initialized object (assigned in-place).

        Notes
        -----
        This function populates `self.__dict__` directly from a loaded instance’s
        `__dict__` so the dataclass fields appear initialized in the correct
        order (and so that post-init checks can run coherently).
        """
        
        # determine "filename": the file where to take the parameters from
        if not (args or kwargs):
            filename = 'params.py'
        elif args and os.path.exists(args[0]):
            filename = args[0]
            args = args[1:]
        else:
            filename = kwargs.pop('filename', None)
        
        # create separate instance
        # NOTE: importing here avoids circular imports
        from . import Params
        instance = Params.load(filename, *args, **kwargs)
        
        # update state dict with that of instance
        self.__dict__.update(instance.__dict__)

    def _setattr(self, name, value, process_and_check=True):
        """
        Assign a parameter field with validation.

        Parameters
        ----------
        name : str
            Name of the field or a hidden attribute (leading underscore).
        value : object
            Value to assign.
        process_and_check : bool, optional
            If True, run `_process_and_check([name])` after assignment and
            revert on failure. This is disabled internally during bulk loading
            so that mutually-dependent fields can be set first and validated
            together.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the field does not exist, if the type is invalid, or if
            post-validation fails.

        Notes
        -----
        Many fields require special handling:

        - callable fields: disallow lambdas, unwrap bound methods, attach
          `__source__`.
        - `states`: normalized to uppercase letters and validated.
        - `topology`: attempts to build an MDAnalysis Universe to cache masses.
        - `initial_paths`: immediately coerced into a PathEnsemble.
        """
        # assign fields
        if name in self.__dataclass_fields__:
            expected_type = self.__dataclass_fields__[name].type

            # functions
            if expected_type is Callable:
                # These callables may be None (meaning “use defaults”).
                if value is None and name in (
                    'descriptors_function', 'descriptor_transform',
                    'values_function', 'toy_mdrun', 'bias_function'):
                    pass

                elif not callable(value):
                    raise TypeError(f"'values_function' must be callable, "
                                    f'got {type(value).__name__}: {value}')

                # Lambdas are rejected (currently not supported by serialization).
                elif getattr(value, '__name__', None) == "<lambda>":
                    raise TypeError(
                        "lambda functions don't work in params (for now)")

                # Bound methods are converted to underlying function objects.
                elif isinstance(value, Method):
                    value = value.__func__
                    update_source(value, name)

                else:
                    update_source(value, name)

                # Special validation for the `fit` callable signature.
                fit_args = 'params', 'pathensemble', 'verbose', 'worker'
                if name == 'fit':
                    args = tuple(inspect.signature(value).parameters)
                    if args != fit_args:
                        raise TypeError(f"'fit' must have args {fit_args} "
                                        f"in order, got {args} instead")

            # states
            elif name == 'states':
                value = str(value).upper().replace(' ','')
                if not (value.isalpha() and len(set(value)) == 3):
                    raise TypeError(
                        f'{name!r} must be three distinct upper alpha chars')

            # state selector strings (set of letters or 'all')
            elif name in ('extra_bins', 'terminal_bin_extension'):
                value = str(value).replace(' ','')
                if value.lower() != 'all':
                    value = value.upper()
                if value and not value.isalpha():
                    raise TypeError(
                        f'{name!r} must be distinct upper alpha chars')
            
            # topology (update universe)
            elif name == 'topology':
                # Cache an MDAnalysis Universe if possible (used by masses()).
                try:
                    universe = Universe(value)
                except:
                    universe = None
                self.__dict__['_universe'] = universe
            
            # initial paths (converted in real-time)
            elif name == 'initial_paths':
                # `_reload_initial_paths` controls whether we must reload from disk
                # to recompute states on demand (avoids stale precomputed arrays).
                if process_and_check:  # otw bug in reloading paths
                    self.__dict__['_reload_initial_paths'] = \
                        not isinstance(value, (Path, PathEnsemble))
                initial_paths = PathEnsemble(value)
                if value and not initial_paths:
                    raise TypeError(
                        f'could not get any initial paths from {value!r}')
                value = initial_paths

            # engine
            elif name == 'engine':
                value = value.lower()
                if value not in ['gromacs', 'toy']:
                    raise TypeError(f'{name} must be either "gromacs" or '
                                    f'"toy", got {value}')

            # network
            elif name == 'network':
                # check that methods required at AIMMD runtime are present
                for attribute in ('forward', 'state_dict',
                                  'load_state_dict', 'parameters'):
                    if (not hasattr(value, attribute) or
                        not callable(getattr(value, attribute))):
                        raise TypeError(
                            f'{name} must have method {attribute!r}')
                update_source(value, name)

            # chain type
            elif name == 'chain_type':
                value = value.lower()
                if value.startswith('tp'):
                    values = 'tps'
                elif value.startswith('rf'):
                    values = 'rfps'
                else:
                    raise TypeError(f'{name!r} must be either "tps" or '
                                    f'"rfps", got {value!r}')

            # list of strings
            elif expected_type is List[str]:
                if type(value) is not list:
                    raise TypeError(f'{name} must be list of strings, '
                                    f'got {type(value).__name__}')
                elif np.any([type(element) is not str for element in value]):
                    raise TypeError(f'{name!r} must be list of strings, '
                                    f'at least one of its elements is not')
            
            # free overriding bins
            elif name == 'free_overriding_bins':
                if isinstance(value, Iterable):
                    value = [int(v) for v in np.array(value, dtype=int)]
                elif isisntance(value, Integral):
                    value = [value]
                elif value is None:
                    pass
                else:
                    raise TypeError(f'{name!r} must index bins, got {value!r}')
            
            # path
            elif name == 'path':
                if not PosixPath(value).exists():
                    raise TypeError(f'Source path {value!r} does not exist.')

            # all the rest
            elif not isinstance(value, expected_type):
                raise TypeError(f'{name} must be {expected_type}, '
                                f'got {type(value).__name__}')

        elif not name.startswith('_'):
            raise TypeError(f'{name!r} is not a valid Params dataclass field '
                            f'or a hidden parameter')

        # assign
        backup = self.__dict__.get(name, 'absent')
        self.__dict__[name] = value

        # special checks (after assignment)
        if process_and_check:
            try:
                self._process_and_check([name])
            except Exception as exception:  # go back in case of error
                if not isinstance(backup, str) or backup != 'absent':
                    self.__dict__[name] = backup
                raise TypeError(f'can\'t update {name!r} with {value!r}')

    def _process_and_check(self, fields=[]):
        """
        Post-assignment validation and derived-field updates.

        Parameters
        ----------
        fields : list of str, optional
            Names of fields that were assigned since the last check. This is
            used to minimize work (e.g., only recompute default values function
            if the network or transform changed).

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If consistency requirements fail, for example:
            - rescaling requested but network does not implement Rescalable,
            - invalid nbins/extra_bins combination,
            - initial paths not transitions or incompatible with topology.

        Notes
        -----
        This method may:
        - overwrite `values_function` if it is defaulted and the network changes,
        - reload initial paths from disk to recompute states if requested,
        - compute and attach descriptors/values to initial paths.
        """

        # check network
        if self.rescale_committor:
            if not isinstance(self.network, RescalableNetwork):
                raise TypeError(f"params' network {self.network} "
                                "is not aimmd.network.Rescalable")
        
        # check nbins
        if 'nbins' in fields or 'terminal_bin_extension' in fields:
            if self.nbins > 0:
                pass
            elif self.nbins == 0 and (
              self.terminal_bin_extension == 'all' or
              (self.states[+0] in self.terminal_bin_extension or
               self.states[-1] in self.terminal_bin_extension)):
                pass
            else:
                states = f'{self.states[+0]}{self.states[-1]}'
                raise TypeError(f'`nbins` must be > 0, or >= 0 when '
                                f'`extra_bins in ({states!r}, \'all\')`')
        
        # check free_overriding_bins
        if 'free_overriding_bins' in fields:
            try:
                np.arange(self.nbins)[self.free_overriding_bins]
            except Exception as exception:
                raise TypeError(
                  f"can't determine free overriding bins when "
                  f"free_overriding_bins = {self.free_overriding_bins!r}, "
                  f"nbins = {self.nbins}: {exception}")
        
        # redefine values function
        default_values_function = self._default_values_function
        if (self.values_function is None or
           ('network' in fields or 'descriptor_transform' in fields) and
            default_values_function):
            default_values_function = True
            self.__dict__['values_function'] = create_default_values_function(
                self.network, self.descriptor_transform)
            check_values = True

        # no need: there is no default values function
        elif 'values_function' in fields:
            default_values_function = False

        # reassign default values function
        self.__dict__['_default_values_function'] = default_values_function

        # initial paths-related checks
        initial_paths = getattr(self, 'initial_paths', None)
        if not initial_paths:
            return
        check_paths = 'initial_paths' in fields
        check_states = ('states' in fields or
                        'states_function' in fields or
                        check_paths)
        check_descrs = (self.descriptors_function is not None and
                       ('descriptors_function' in fields or check_paths))
        if self.descriptors_function is None:
            check_values = 'values_function' in fields or check_paths
        else:
            check_values = ('values_function' in fields or
                            'descriptors_function' in fields or
                            check_paths)

        # go through paths
        for i, path in enumerate(initial_paths):
            if check_paths:
                # Ensure topology (if available) matches path atom count.
                try:
                    n_atoms = path.reader.n_atoms
                except:
                    n_atoms = path.reader.trajectory.n_atoms
                universe = self._universe
                if universe and n_atoms != len(universe.atoms):
                    raise TypeError(
                        f'{i}-th initial path {path.fname!r} has '
                        f'{n_atoms} atoms, while the topology file '
                        f'{self.topology!r} has {len(universe.atoms)} atoms')

            if check_states:
                # reload path if required
                if self._reload_initial_paths:
                    path = Path(path.fname)
                # recompute states (only if not manually overwritten)
                if 'states' not in path.__dict__:
                    path.states = path.compute(self.states_function)
                # throw a warning if no transition is found
                transition_found = False 
                for split_path in (split_paths := path.split()):
                    if split_path.type[:3] in (self.states, self.states[::-1]):
                        path = split_path
                        transition_found = True
                        break
                if not transition_found:
                    types = ", ".join([t[:3] for t in split_paths.types()])
                    print(f'Warning: the initial trajectory {path.fname!r} '
                        f'has no {self.states!r} transitions '
                        f'(path types: {types}), taking it as it is.\n'
                         'Perhaps you are using it for brute-force shooting?')
            
            if check_descrs:
                # Cache descriptors on the Path object.
                path.descriptors = path.compute(self.descriptors_function)

            if check_values:
                # Ensure values_function returns exactly one value per frame.
                if self.descriptors_function is not None:
                    source = path.descriptors[:1]
                else:
                    source = path.coordinates[:1]
                assert self.values_function(source).shape == (1,)

            # reassign path
            self.initial_paths[i] = path
