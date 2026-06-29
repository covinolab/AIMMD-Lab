"""
aimmd.params._properties
=======================

Derived properties for :class:`aimmd.params.Params`.

This mixin defines read-only convenience properties computed from primary
dataclass fields, plus a lightweight placeholder factory useful for tests,
docs, and dry runs.

Properties
----------
placeholder (classproperty)
    Construct a minimal Params-like object without running normal validation.
parent
    Working directory associated with `params.path` (file parent or directory).
sorted_states
    Deterministic state ordering used for file naming (e.g. network/bins files).
universe
    Cached MDAnalysis Universe created when setting `topology` (may be None).
masses
    Per-atom masses inferred from the cached Universe (or None if unavailable).
compute_states_args
    Argument tuple used by Path/PathEnsemble `.compute()` to compute states.
compute_descriptors_args
    Argument tuple used by Path/PathEnsemble `.compute()` to compute descriptors
    (or None if descriptors are disabled).
compute_values_args
    Argument tuple used by Path/PathEnsemble `.compute()` to compute values,
    specifying whether the source is coordinates or descriptors.
pipeline
    Ordered tuple of compute-argument tuples describing the preferred compute
    pipeline for a Path/PathEnsemble under the current configuration.

Notes
-----
- These are *read-only* properties: `ParamsMagic.__setattr__` forbids setting
  them.
- `placeholder` is a `classproperty` so it can be used as `Params.placeholder`.
"""

# external
import numpy as np
import torch
from abc import ABC
from dataclasses import MISSING

# aimmd imports
from .utils import create_default_values_function
from ..core.utils import guess_masses
from ..network.fit import default as default_fit
from ..core.decorators import classproperty


# params' properties
class ParamsProperties(ABC):

    @classproperty
    def placeholder(cls):
        """
        Construct a minimal placeholder `Params` instance.

        This factory bypasses normal initialization and validation by:
        - creating an instance via `object.__new__`,
        - populating all dataclass fields that have `default_factory`,
        - seeding internal caches expected elsewhere in AIMMD.

        Returns
        -------
        aimmd.params.Params
            A Params-like object suitable for non-production contexts
            (documentation, dry runs, unit tests).

        Notes
        -----
        - `states_function` is set to a trivial function returning all 'R'
          labels (reactive region).
        - `values_function` is set to the default network-evaluation wrapper
          created by `create_default_values_function(...)`.
        - `topology` is set to the empty string, so `universe`/`masses`
          will typically be unavailable.
        """
        from . import Params
        self = object.__new__(Params)

        # Initialize all dataclass fields with a default_factory (e.g. lists/dicts).
        # Fields with `default` but no `default_factory` are left untouched here,
        # because they can be retrieved from the class defaults when needed.
        for name in cls.__dataclass_fields__:
            field = cls.__dataclass_fields__[name]
            if (not hasattr(self, name) and
                field.default_factory is not MISSING):
                    self.__dict__[name] = field.default_factory()

        # Internal caches/flags expected by other Params mixins.
        self.__dict__['_universe'] = None
        self.__dict__['_default_values_function'] = True
        
        # Minimal functional configuration.
        self.__dict__['states_function'] = lambda x: np.full(len(x), 'R')
        self.__dict__['values_function'] = create_default_values_function(
            self.network, None)
        self.__dict__['fit'] = default_fit
        self.__dict__['topology'] = ''
        
        return self

    @property
    def parent(self):
        """
        Parent folder associated with this Params instance.

        Returns
        -------
        pathlib.Path
            If `self.path` is a file, returns `self.path.parent`.
            If `self.path` is a directory, returns `self.path`.

        Notes
        -----
        AIMMD uses `parent` as the base directory for engine operations and for
        saving/loading run artifacts.
        """
        if self.path.is_file():
            return self.path.parent
        return self.path

    @property
    def sorted_states(self):
        """
        Deterministic ordering of endpoint state labels for file naming.

        Returns
        -------
        str
            Either `self.states` or `self.states[::-1]`, chosen so that the
            first and last characters are ordered alphabetically.

        Notes
        -----
        This provides stable artifact names such as:
        - `network{sorted_states}.h5`
        - `bins{sorted_states}.npy`
        independent of whether you conceptually describe a transition as A→B
        or B→A.
        """
        if self.states[0] > self.states[2]:
            return self.states[::-1]
        return self.states

    @property
    def universe(self):
        """
        Cached MDAnalysis Universe derived from `topology`.

        Returns
        -------
        MDAnalysis.Universe or None
            Universe built when setting `topology`, or None if `topology`
            could not be loaded.
        """
        return self._universe

    @property
    def masses(self):
        """
        Per-atom masses inferred from the cached Universe.

        Returns
        -------
        numpy.ndarray or None
            1D array of length `n_atoms` containing masses if `self._universe`
            is available, otherwise None.

        Notes
        -----
        Masses are guessed via `guess_masses(self._universe.atoms)`.
        The exact units depend on the conventions used by `guess_masses` and
        the topology format.
        """
        if self._universe is None:
            return
        return guess_masses(self._universe.atoms)

    def universe_of(self, system_id=None):
        """
        MDAnalysis Universe for a given system.

        Multi-system runs cache one Universe per system in ``_universes``
        (keyed by ``system_id``); single-system runs use the single
        ``_universe``. With ``system_id=None`` this returns the single-system
        universe (backward compatible).
        """
        if system_id is None:
            return self._universe
        universes = self.__dict__.get('_universes', None)
        if not universes:
            return self._universe
        return universes.get(system_id, None)

    def masses_of(self, system_id=None):
        """
        Per-atom masses for a given system (see ``masses``/``universe_of``).
        """
        universe = self.universe_of(system_id)
        if universe is None:
            return
        return guess_masses(universe.atoms)

    def _per_system_value(self, value, system_id):
        """Resolve a scalar-or-per-system-list field for ``system_id``.

        If ``value`` is a list/tuple it is indexed by the position of
        ``system_id`` in ``system_ids``; otherwise ``value`` is returned as-is
        (a scalar is broadcast to every system). ``system_id=None`` (single
        system) always returns ``value`` unchanged.
        """
        if system_id is None or not isinstance(value, (list, tuple)):
            return value
        system_ids = list(self.system_ids or [])
        try:
            idx = system_ids.index(system_id)
        except ValueError:
            return value
        return value[idx]

    def bias_reactive_threshold_of(self, system_id=None):
        """Reactive-bias threshold for a given system (see
        ``bias_reactive_threshold``); a scalar is broadcast to all systems."""
        return self._per_system_value(self.bias_reactive_threshold, system_id)

    def subsample_caps_of(self, system_id=None):
        """Value-pass subsampling caps for a given system (see
        ``subsample_caps``); a single dict is broadcast to all systems."""
        return self._per_system_value(self.subsample_caps, system_id)

    @property
    def compute_states_args(self):
        """
        Arguments for computing states on a Path/PathEnsemble.

        Returns
        -------
        tuple
            `(states_function, 'states')`

        Examples
        --------
        >>> path.compute(*params.compute_states_args)
        """
        return self.states_function, 'states'

    @property
    def compute_descriptors_args(self):
        """
        Arguments for computing descriptors on a Path/PathEnsemble.

        Returns
        -------
        tuple or None
            If `descriptors_function` is configured, returns:
            `(descriptors_function, 'descriptors')`.
            Otherwise returns None.

        Examples
        --------
        >>> if params.compute_descriptors_args is not None:
        ...     path.compute(*params.compute_descriptors_args)
        """
        if not self.descriptors_function:
            return
        return self.descriptors_function, 'descriptors'

    @property
    def compute_values_args(self):
        """
        Arguments for computing values on a Path/PathEnsemble.

        Returns
        -------
        tuple
            ``(values_function, 'values', source)`` where ``source`` is
            ``'coordinates'`` if descriptors are disabled and
            ``'descriptors'`` if descriptors are enabled.

        Examples
        --------
        >>> path.compute(*params.compute_values_args)

        Notes
        -----
        This convention matches `Path.compute(function, name, source=...)`
        semantics used throughout AIMMD.
        """
        if not self.descriptors_function:
            return self.values_function, 'values', 'coordinates'
        return self.values_function, 'values', 'descriptors'

    @property
    def pipeline(self):
        """
        Preferred compute pipeline under the current configuration.

        Returns
        -------
        tuple
            Ordered tuple of compute-argument tuples, suitable for driving
            repeated `Path.compute(...)` and `PathEnsemble.compute(...)` calls.
            If descriptors are disabled, the tuple is
            ``(compute_states_args, compute_values_args)``. If descriptors are
            enabled, it is
            ``(compute_descriptors_args, compute_states_args, compute_values_args)``.

        Examples
        --------
        >>> for args in params.pipeline:
        ...     path.compute(*args)

        Notes
        -----
        Descriptors are computed first (when enabled) so that values can be
        computed from cached descriptors rather than raw coordinates.
        """
        if not self.descriptors_function:
            return self.compute_states_args, self.compute_values_args
        return (self.compute_descriptors_args,  # better first
                self.compute_states_args, self.compute_values_args)
