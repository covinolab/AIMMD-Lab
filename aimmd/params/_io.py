"""
aimmd.params._io
===============

Load/save mixin for :class:`aimmd.params.Params`.

This module defines :class:`~aimmd.params._io.ParamsIO`, which provides:

- `load`: execute a Python params file in an isolated module namespace, extract
  dataclass fields, apply overrides, validate, and optionally save a canonical
  `params.py`.
- `update`: convenience wrapper for `load(None, ...)` (no filename).
- `save`: write a reproducible `params.py` (including module imports and
  callable sources) into the working directory.


Key idea: "local modules" namespace rewriting
---------------------------------------------
When loading a params file, AIMMD temporarily rewrites module naming so that:

- local `*.py` files in the params directory can be imported consistently,
  whether:

  - the params file is executed standalone in a fresh interpreter, or
  - it is loaded via `Params.load(...)` within an existing AIMMD session.

This requires:

- temporary removal of local modules from `sys.modules`,
- swapping names to avoid conflicts with already-imported modules,
- updating `__module__` attributes of callables/classes so saving (`__str__`)
  generates correct import statements.

Notes
-----
- This implementation modifies global interpreter state (`os.chdir`, `sys.path`,
  `sys.modules`) but attempts to restore it robustly on error.
- The method is intentionally strict about saving into the original parent
  folder to prevent silent drift of working directories.
"""

# external
import os
import sys
from abc import ABC
from types import ModuleType
from pathlib import PosixPath
from dataclasses import MISSING

# aimmd imports
from ..path import Path
from .._config import print
from ..core.utils import unique_path
from ..pathensemble import PathEnsemble
from ..core.decorators import class_or_instancemethod

# params' methods
class ParamsIO(ABC):

    @class_or_instancemethod
    def load(self_or_cls, filename='params.py', *args, **kwargs):
        """
        Load parameters from a Python params file.

        This method can be called either as:
        - `Params.load(filename, ...)` (class call), or
        - `params.load(filename, ...)` (instance call).

        Behavior
        --------
        - If `filename` is a path to a file:

            * switch to its folder,
            * execute it in a fresh `ModuleType` namespace,
            * populate known dataclass fields from module globals,
            * apply explicit overrides from `args`/`kwargs`,
            * validate all fields together with `_process_and_check`,
            * optionally save a canonical params file in the same folder.

        - If `filename` is falsy/None:

            * do not execute a file; treat the current working directory as the
              params folder and only apply overrides.

        Parameters
        ----------
        filename : str or None, optional
            Params file path. If None, no file is executed and only overrides
            are applied to defaults/current values.
        *args
            Positional overrides mapped to dataclass fields in declared order.
            (Only fields present in `Params.__dataclass_fields__` are used.)
        **kwargs
            Keyword overrides for dataclass fields.

            Special keys:

            save : bool, optional
                If False, do not call `self.save()` after successful load.

        Returns
        -------
        aimmd.params.Params
            Params instance (updated in place or freshly created).

        Raises
        ------
        TypeError
            If the file cannot be found, invalid fields are provided, required
            fields are missing, or validation fails.

        Notes
        -----
        This function temporarily mutates:

        - current working directory,
        - `sys.path`,
        - `sys.modules`.

        It attempts to restore them in `finally` / `except` blocks.
        """

        # which folder we need to go to?
        if filename:
            path = PosixPath(filename).resolve()
            if not path.exists():
                raise TypeError(f'{filename!r} not found')
            if not path.is_file():
                raise TypeError(f'{filename!r} must be a file')
            folder = path.parent
        else:
            folder = path = PosixPath().resolve()

        # go to folder
        cwd = PosixPath().resolve()
        os.chdir(folder)
        sys.path.insert(0, f'{folder}')

        # in case of problems: restore modules and params fields
        backup_modules = sys.modules.copy()
        updated_fields = []

        try:
            # do we need to create or select an instance of Params?
            from . import Params
            if isinstance(self_or_cls, Params):
                self = self_or_cls
                backup_dict = self.__dict__.copy()
                if filename and self.parent != folder:
                    raise TypeError(
                        f"new params' filename '{path}' must be "
                        f"in '{self.parent}', the same folder "
                        f"associated to this aimmd.Params object")
            else:
                # Construct an instance without calling __init__.
                self = object.__new__(Params)
                # Internal caches/flags expected elsewhere in the codebase.
                self.__dict__['_universe'] = None
                self.__dict__['_default_values_function'] = False
                self.__dict__['_reload_initial_paths'] = True
                backup_dict = {}

            # fields with already present values or their default
            fields = {name:
                      getattr(self, name)
                          if hasattr(self, name) else
                      self.__dataclass_fields__[name].default
                          if self.__dataclass_fields__[name].default
                          is not MISSING else
                      self.__dataclass_fields__[name].default_factory()
                          if self.__dataclass_fields__[name
                          ].default_factory is not MISSING else
                      MISSING for name in self.__dataclass_fields__
                          if name != 'path'}

            # defaults
            new_states_function = False
            new_initial_paths = False
            self.__dict__['path'] = path

            # update fields with args (in the right order)
            for value, name in zip(args, self.__dataclass_fields__):
                if name in kwargs:
                    raise TypeError(f'multiple assignements of {name} '
                                    f'when calling Params.load; either '
                                    f'remove the positional argument or '
                                    f'the keyword argument')
                kwargs[name] = value

            # update fields with kwargs
            for name, value in kwargs.items():
                if name in fields:
                    fields[name] = value
                    updated_fields.append(name)

            # execute the file and extract fields...
            num_fields_from_filename = 0
            if path.is_file():

                # assign local modules names
                local_module_names = {}
                temporarily_removed_modules = {}

                # find all local py files in the folder except the main script
                for local_path in folder.glob("*.py"):

                    # Each file in the current folder gets a “local module name”
                    # corresponding to its full path (without .py). This avoids
                    # conflicts with external modules sharing the same stem name.
                    original_module_name = local_path.stem
                    local_module_name = str(
                        local_path.with_name(original_module_name))
                    local_module_names[original_module_name] = \
                        local_module_name

                    # swap name of original modules (if any conflict exists)
                    if original_module_name in sys.modules:
                        sys.modules[local_module_name] = \
                            sys.modules.pop(original_module_name)

                    # ALL local modules are temporarily removed to force refresh:
                    # editing local helper modules takes effect at next load.
                    if local_module_name in sys.modules:
                        temporarily_removed_modules[local_module_name] = \
                            sys.modules.pop(local_module_name)

                # add current directory to sys.path
                sys.path.insert(0, '')

                # treat filename as a (local) module
                module_name = path.stem
                module = ModuleType(module_name)
                module.__file__ = str(path)
                sys.modules[module_name] = module

                # execute the file inside the module’s namespace
                source = path.read_text()
                exec(compile(source, path, "exec"), module.__dict__)

                # populate the fields
                for name in module.__dict__:

                    # ...only if not assigned already
                    if name in fields and name not in kwargs:

                        # get new values
                        fields[name] = module.__dict__[name]
                        updated_fields.append(name)
                        num_fields_from_filename += 1

                # after execution: swap original and local modules names
                # avoiding conflicts
                for original_name, local_name in local_module_names.items():
                    original_module = None
                    local_module = None

                    # retrieve modules
                    if original_name in sys.modules:
                        local_module = sys.modules.pop(original_name)
                    if local_name in sys.modules:
                        original_module = sys.modules.pop(local_name)

                    # actual swapping
                    if original_module is not None:
                        sys.modules[original_name] = original_module
                    if local_module is not None:
                        sys.modules[local_name] = local_module

                        # update __module__ of callables/classes originating
                        # from the now-renamed local module
                        for name, obj in local_module.__dict__.items():
                            if ((callable(obj) or isinstance(obj, type)) and
                                hasattr(obj, '__module__') and
                                obj.__module__ == original_name):
                                obj.__module__ = local_name

                # put back temporarily removed modules that were not
                # reimported in the procedure
                for module_name in temporarily_removed_modules:
                    if module_name not in sys.modules:
                        sys.modules[module_name] = \
                            temporarily_removed_modules[module_name]

            # assign fields; raise error for missing fields
            for name, value in fields.items():
                if value is MISSING:
                    raise TypeError(f'{name} not provided for Params')
                self._setattr(name, value, process_and_check=False)

            # post-init operation: validate field interactions as a batch
            self._process_and_check(updated_fields)

            # save?
            if kwargs.get('save', True):
                self.save()

            return self

        except Exception as exception:

            # restore modules
            sys.modules = backup_modules

            # restore attributes
            self.__dict__.update(backup_dict)

            raise exception

        finally:  # back to the original folder & path
            os.chdir(cwd)
            sys.path.pop(0)

    def update(self, *args, **kwargs):
        """
        Update an existing Params object without executing a file.

        Parameters
        ----------
        *args, **kwargs
            Same override semantics as `load`, but `filename` is forced to None.

        Returns
        -------
        aimmd.params.Params
            Updated instance.
        """
        return self.load(None, *args, **kwargs)

    def save(self, path=None, seek_existing_file=True):
        """
        Save params to a Python file and update `params.path`.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination filename. If None, uses `self.path`. If `self.path` is a
            directory, defaults to `<dir>/params.py`.
        seek_existing_file : bool, optional
            If True, attempt to find an existing `.py` file in the same folder
            whose trailing body matches `self.__str__()` output; if found, reuse
            it instead of writing a new file.

        Returns
        -------
        str
            Relative path (from current working directory) to the params file.

        Raises
        ------
        TypeError
            If attempting to save outside the original `Params.parent` folder.

        Notes
        -----
        The saved file contains:

        - imports for modules present in `__main__`,
        - the verbose parameters body produced by `Params.__str__`, including
          callable bodies/imports and per-field descriptions.
        """
        # determine correct path
        if path:
            filename = PosixPath(path).resolve()
        else:
            filename = self.path
        if filename == self.parent:
            filename = PosixPath(f'{filename}/params.py')
        if filename.parent != self.parent:
            raise TypeError(
                f'can only save to original parent folder: {self.parent}')
        filename = unique_path(filename)

        # create text object
        text = []

        # copy main modules
        modules = vars(sys.modules['__main__'])
        text.append(f'# packages')
        for name in modules:
            if type(modules[name]) is not type(sys):
                continue
            text.append(f'import {modules[name].__name__} as {name}')
        text.append('')
        
        # copy params
        body = self.__str__(filename.parent)
        text.append(body)
        text = "\n".join(text)
        
        # was there? then use it
        writing = True
        if seek_existing_file:
            for old_filename in sorted(filename.parent.glob("*.py")):
                try:
                    if old_filename.read_text()[-len(body):] == body:
                        writing = False
                        filename = old_filename
                        break
                except:
                    continue
        fname = os.path.relpath(filename)

        # only when different: write it
        if writing:
            with open(fname, 'w') as file:
                file.write(text)

            # report
            print(f'Written full params with descriptions to {fname!r}')
        else:
            print(f'Assigned parameters file {fname!r}')

        # update path info and report
        self.__dict__['path'] = filename
        return fname
