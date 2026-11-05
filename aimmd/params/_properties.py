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
    Per-atom masses of the system (guessed from the cached Universe, or read
    from the GROMACS topology with ``masses_from_topology``; None if
    unavailable).
compute_states_args
    Argument tuple used by Path/PathEnsemble `.compute()` to compute states.
compute_descriptors_args
    Argument tuple used by Path/PathEnsemble `.compute()` to compute descriptors
    into the `descriptors_series` (or None if descriptors are disabled).
compute_values_args
    Argument tuple used by Path/PathEnsemble `.compute()` to compute values,
    specifying whether the source is coordinates or the descriptors series.
pipeline
    Ordered tuple of compute-argument tuples describing the preferred compute
    pipeline for a Path/PathEnsemble under the current configuration.
cgaimmd_settings
    ``cgaimmd_parameters`` with the defaults of the missing keys filled in.

Notes
-----
- These are *read-only* properties: `ParamsMagic.__setattr__` forbids setting
  them.
- `placeholder` is a `classproperty` so it can be used as `Params.placeholder`.
"""

# external
import os
import numpy as np
import torch
from abc import ABC
from dataclasses import MISSING

# aimmd imports
from .utils import (create_default_values_function, CGAIMMD_PARAMETER_DEFAULTS,
                    gromacs_topology_file, gromacs_include_dir,
                    read_topology_masses, masses_disagreement)
from .._config import print
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
        Per-atom masses of the system, used to draw velocities.

        Returns
        -------
        numpy.ndarray or None
            1D array of length `n_atoms` if `self._universe` is available,
            otherwise None.

        Notes
        -----
        With ``masses_from_topology`` False (the default) the masses are
        guessed via `guess_masses(self._universe.atoms)` (MDAnalysis' guess
        from the atom names); with True and ``engine = 'gromacs'`` they are
        read from the GROMACS topology grompp uses (the ``-p`` of
        ``gmx_grompp``, else ``topol.top`` in ``params.parent``). Other
        engines have no such topology and keep the guess whatever
        ``masses_from_topology`` says. The topology is read once per file
        version and cached; when both sets of masses are available and
        differ, a warning is printed once per file version.
        """
        if self._universe is None:
            return
        return self._masses_of_universe(self._universe, self.gmx_grompp)

    def _masses_of_universe(self, universe, gmx_grompp):
        """Guessed or topology masses of `universe` (see ``masses``)."""
        guessed = guess_masses(universe.atoms)
        if getattr(self, 'engine', 'gromacs') != 'gromacs':
            return guessed                      # no topology grompp reads
        if not isinstance(gmx_grompp, str):     # multi-system, not bound to a system
            return guessed
        strict = bool(getattr(self, 'masses_from_topology', False))
        topology = self._topology_masses(len(universe.atoms), gmx_grompp, strict=strict)
        if topology is None:
            return guessed
        key = self._topology_masses_key(gmx_grompp)
        warned = self.__dict__.setdefault('_topology_masses_warned', set())
        if key not in warned:
            warned.add(key)
            try:
                names = [str(name) for name in universe.atoms.names]
            except Exception:                   # noqa: BLE001  (no names)
                names = [str(i) for i in range(len(universe.atoms))]
            differ = masses_disagreement(names, guessed, topology)
            if differ:
                top = gromacs_topology_file(gmx_grompp, self.parent)
                if strict:
                    used = ('velocities are drawn from the topology masses '
                            '(masses_from_topology = True)')
                else:
                    used = ('velocities are drawn from the guessed masses; set '
                            'masses_from_topology = True to use the topology masses')
                print(f'Warning: the masses MDAnalysis guesses from the atom names '
                      f'of {self.topology!r} differ from those of the GROMACS '
                      f'topology {top!r} for {differ} (guessed vs topology, amu); '
                      f'{used}')
        return topology if strict else guessed

    def _topology_masses_key(self, gmx_grompp):
        """``(absolute topology file, (mtime_ns, size) or None)``: the cache key of one file version."""
        top = os.path.abspath(gromacs_topology_file(gmx_grompp, self.parent))
        try:
            stat = os.stat(top)
            version = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            version = None
        return top, version

    def _topology_masses(self, n_atoms, gmx_grompp, strict=False):
        """Masses of the GROMACS topology of `gmx_grompp`, cached per file version.

        Returns None when the file is missing, unreadable or has another atom
        count than `n_atoms`; with `strict` those cases raise TypeError.
        """
        top, version = key = self._topology_masses_key(gmx_grompp)
        cache = self.__dict__.setdefault('_topology_masses_cache', {})
        if key not in cache:
            if version is None:
                cache[key] = ValueError(f'{top!r} does not exist')
            else:
                try:
                    cache[key] = read_topology_masses(
                        top, include_dir=gromacs_include_dir(gmx_grompp))
                except ValueError as exception:
                    cache[key] = exception
        masses = cache[key]
        if isinstance(masses, Exception):
            if strict:
                raise TypeError(f'masses_from_topology = True needs the GROMACS topology '
                                f'grompp uses (the -p of gmx_grompp, else topol.top next '
                                f'to the params file): {masses}')
            return None
        if len(masses) != n_atoms:
            if strict:
                raise TypeError(f'masses_from_topology = True: the GROMACS topology {top!r} '
                                f'has {len(masses)} atoms, the structure {self.topology!r} '
                                f'{n_atoms}')
            return None
        return masses

    def _initial_network_fname(self):
        """The file of ``initial_network``, joined to ``params.parent`` when relative.

        Returns
        -------
        str or None
            None when ``initial_network`` is None.
        """
        value = getattr(self, 'initial_network', None)
        if value is None:
            return None
        if os.path.isabs(value):
            return value
        return os.path.join(os.fspath(self.parent), value)

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
        return self._masses_of_universe(
            universe, self._per_system_value(self.gmx_grompp, system_id))

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

    def subsample_caps_active(self, system_id=None):
        """True if ``subsample_caps`` caps at least one category for this system
        (any system when ``system_id`` is None and the caps are per system).
        A dict whose caps are all None caps nothing."""
        caps = self.subsample_caps_of(system_id)
        entries = caps if isinstance(caps, (list, tuple)) else [caps]
        return any(entry and any(cap is not None for cap in entry.values())
                   for entry in entries)

    @property
    def cgaimmd_settings(self):
        """
        C-GAIMMD settings with the defaults filled in.

        Returns
        -------
        dict
            A new dict ``{**CGAIMMD_PARAMETER_DEFAULTS, **cgaimmd_parameters}``.

        Notes
        -----
        ``cgaimmd_parameters`` stores only the keys the user gave, so saved
        params files carry no defaults; this property is what C-GAIMMD code
        reads (under any ``chain_type``: the generator fits read
        ``training_device`` and ``lambda_clip`` from it too). All defaults
        when ``cgaimmd_parameters`` is None (see
        ``aimmd.params.utils.CGAIMMD_PARAMETER_DEFAULTS``); changing the
        returned dict does not change the params.
        """
        return {**CGAIMMD_PARAMETER_DEFAULTS,
                **(getattr(self, 'cgaimmd_parameters', None) or {})}

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
            `(descriptors_function, descriptors_series)`, i.e.
            `(descriptors_function, 'descriptors')` by default.
            Otherwise returns None.

        Examples
        --------
        >>> if params.compute_descriptors_args is not None:
        ...     path.compute(*params.compute_descriptors_args)
        """
        if not self.descriptors_function:
            return
        return self.descriptors_function, self.descriptors_series

    @property
    def compute_values_args(self):
        """
        Arguments for computing values on a Path/PathEnsemble.

        Returns
        -------
        tuple
            ``(values_function, 'values', source)`` where ``source`` is
            ``'coordinates'`` if descriptors are disabled and
            `descriptors_series` (by default ``'descriptors'``) if
            descriptors are enabled.

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
        return self.values_function, 'values', self.descriptors_series

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
