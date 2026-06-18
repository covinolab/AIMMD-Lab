"""
aimmd.params._magic
==================

Magic methods for :class:`aimmd.params.Params`.

This mixin centralizes:

- user-facing representations (`__repr__`, `__str__`),
- controlled attribute assignment (`__setattr__`),
- mapping-like access (`__getitem__`, `__setitem__`),
- equality semantics (`__eq__`).

Key behavior
------------

- `__setattr__` routes assignments through `ParamsHelpers._setattr`, enforcing
  type checks and triggering post-validation.
- Properties defined in `ParamsProperties` are read-only at runtime; attempts
  to assign them raise `AttributeError`.
- `__str__` generates a reproducible parameters script body, including:

  - callable source/import blocks (`__source__`),
  - special handling of paths relative to a target directory,
  - field descriptions from `metadata['description']`.

Notes
-----
The text produced by `__str__` is used by `ParamsIO.save`.
"""

# external
import os
from abc import ABC
from math import inf

# aimmd imports
from ..path import Path
from ._properties import ParamsProperties


# params' magic methods
class ParamsMagic(ABC):

    def __repr__(self):
        """
        Concise representation.

        Returns
        -------
        str
            Short string including the relative path of the associated params file.
        """
        return f'Params {os.path.relpath(self.path)}'

    def __setattr__(self, name, value):
        """
        Assign a params field with validation.

        Parameters
        ----------
        name : str
            Field name.
        value : object
            New value.

        Raises
        ------
        AttributeError
            If attempting to assign a read-only property defined on ParamsProperties.
        TypeError
            If assignment fails validation.

        Notes
        -----
        This method:

        - forbids assignment to derived properties,
        - delegates field validation to `self._setattr`,
        - restores the old value if validation fails.
        """
        # can't set properties
        if isinstance(getattr(ParamsProperties, name, None), property):
            raise AttributeError(
                f"can't set aimmd.Params property {name!r}")

        # backup old value
        if hasattr(self, name):
            backup = getattr(self, name)
        else:
            backup = None

        try:
            self._setattr(name, value)

        # in case of errors, back to the old value
        except Exception as exception:
            if backup is not None:
                self.__dict__[name] = backup
            raise exception

    def __setitem__(self, name, value):
        """
        Dict-like assignment alias for `__setattr__`.
        """
        return self.__setattr__(name, value)

    def __getitem__(self, name):
        """
        Dict-like access to raw stored values.

        Notes
        -----
        This returns the underlying value in `self.__dict__`, not properties.
        """
        return self.__dict__[name]

    def __eq__(self, params):
        """
        Equality based on working directory and serialized content.

        Parameters
        ----------
        params : aimmd.params.Params
            Another params object.

        Returns
        -------
        bool
            True if both objects:

            - belong to the same parent folder, and
            - serialize to identical `__str__` output.
        """
        # different working directory
        if self.parent != params.parent:
            return False

        # simple string check
        return str(self) == str(params)

    def __str__(self, go_to=None):
        """
        Verbose and reproducible string representation.

        Parameters
        ----------
        go_to : str or pathlib.Path, optional
            Target directory used to convert certain file paths to relative
            paths. This is used by `save()` so generated scripts remain portable.

        Returns
        -------
        str
            Python source text representing the params fields, including:

            - callable source/import stubs,
            - values (with special casing for inf),
            - per-field description docstrings.

        Notes
        -----
        - Fields are emitted in dataclass definition order.
        - Certain engine-specific fields are skipped depending on `self.engine`.
        - If `values_function` is auto-generated from the network
          (`self._default_values_function`), it is serialized as `None`.
        """
        lines = []

        for name in self.__dataclass_fields__:
            if name == 'path':
                continue

            if self.engine == 'toy' and name.startswith('gmx'):
                continue

            if self.engine == 'gromacs' and name.startswith('toy'):
                continue

            # field and value
            field = self.__dataclass_fields__[name]
            if name == 'values_function' and self._default_values_function:
                value = None
            else:
                value = getattr(self, name)

            # function or network (serialized via __source__)
            if hasattr(value, '__source__'):
                lines.append(value.__source__.rstrip('\n'))

            # selected path-like fields are serialized relative to `go_to`.
            # In multi-system mode `topology` is a list (one per system).
            elif name in ('topology', 'gmx_mdp'):
                if isinstance(value, (list, tuple)):
                    rel = [os.path.relpath(v, go_to) for v in value]
                    lines.append(f'{name} = {rel!r}')
                else:
                    lines.append(f'{name} = {os.path.relpath(value, go_to)!r}')

            # initial paths: write as list of filenames (if reload is enabled).
            # In multi-system mode this is a list of groups (one per system),
            # serialized as a list of lists of filenames.
            elif name == 'initial_paths':
                if getattr(self, 'multi_system', False):
                    groups = []
                    if self._reload_initial_paths:
                        for group in value:
                            fnames = []
                            for path in group:
                                if isinstance(path, Path):
                                    path = path.fnames[0]
                                rel = os.path.relpath(path, go_to)
                                fnames.append(f'"{rel}"')
                            groups.append(f'[{", ".join(fnames)}]')
                    lines.append(f'{name} = [{", ".join(groups)}]')
                else:
                    filenames = []
                    if self._reload_initial_paths:
                        initial_paths = value
                        for path in initial_paths:
                            if isinstance(path, Path):
                                path = path.fnames[0]
                            filename = os.path.relpath(path, go_to)
                            filenames.append(f'"{filename}"')
                    lines.append(f'{name} = [{", ".join(filenames)}]')

            # all the rest
            elif value == inf:
                lines.append(f'{name} = float(\'inf\')')
            else:
                lines.append(f'{name} = {repr(value)}')

            # print description (stored as a docstring literal after each value)
            if desc := field.metadata.get("description", ""):
                lines.append(f"\"\"\"{desc}\"\"\"\n")

        return "\n".join(lines)
