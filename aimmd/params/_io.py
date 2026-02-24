"""
...
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
from ..pathensemble.utils import get_paths

# params' methods
class ParamsIO(ABC):

    @class_or_instancemethod
    def load(self_or_cls, filename='params.py', *args, **kwargs):
        """
        Load parameters and functions from a Python file and update Params
        instance *in place*. Load additional `kwargs` with higher priority.
        
        filename: str or None
            if None, just run normal init

        save: if in kwargs and False, then do no save/assign
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
        backup_params = {}
        updated_fields = []
        
        try:
            # do we need to create or select an instance of Params?
            from . import Params
            if isinstance(self_or_cls, Params):
                self = self_or_cls
                if filename and self.parent != folder:
                    raise TypeError(
                        f"New params' filename \"{path}\" must be "
                        f"in the same folder associated to this"
                        f"aimmd.Params object: \"{self.parent}\"")
            else:
                self = object.__new__(Params)
                self.__dict__['_universe'] = None
                self.__dict__['_default_values_function'] = False
                self.__dict__['_reload_initial_paths'] = True
            
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
                    
                    # backup old values
                    if hasattr(self, name):
                        backup_params[name] = fields[name]
                    
                    # get new values
                    fields[name] = value
                    updated_fields.append(name)
                
                # special fields
                if name == 'initial_paths':
                    if isinstance(value, Path):
                        fields[name] = [value]
                        self.__dict__['_reload_initial_paths'] = False
                    elif isinstance(value, PathEnsemble):
                        fields[name] = value._paths
                        self.__dict__['_reload_initial_paths'] = False
                    else:
                        fields[name] = get_paths(value)
                        self.__dict__['_reload_initial_paths'] = True
            
            # execute the file and extract fields...
            num_fields_from_filename = 0
            if path.is_file():
                
                # assign local modules names
                local_module_names = {}
                temporarily_removed_modules = {}
                
                # find all local py files in the folder except the main script
                for local_path in folder.glob("*.py"):
                    
                    # each file in the current folder will get a "local"
                    # module name, corresponding to its full path without .py
                    #
                    # however, the params' "filename" may also be executed
                    # as a standalone file in a "fresh" python interpreter,
                    # and still give the same results
                    #
                    # in order to achieve this goal, ONLY during the params
                    # loading phase, the "local" module name is replaced
                    # by a temporary module name with the relative path
                    # instead of the full path
                    #
                    # importantly, there may be already imported modules with
                    # the same name as those temporary module names
                    #
                    # the "original" modules with conflicting names
                    # temporarily swap their name with the local module name,
                    # and get their original one back only at the end of
                    # "filename"'s execution
                    #
                    # special care is put in restoring the original modules
                    # situation in case some error happens

                    original_module_name = local_path.stem
                    local_module_name = str(
                        local_path.with_name(original_module_name))
                    local_module_names[original_module_name] = \
                        local_module_name
                    
                    # swap name of original modules
                    if original_module_name in sys.modules:
                        sys.modules[local_module_name] = \
                            sys.modules.pop(original_module_name)
                    
                    # ALL local modules are temporarily removed
                    # incidentally, this means that one may just change the
                    # local parameters files, and those modules are updated
                    # at the next Params.load call, in contrast with the usual
                    # behavior when importing modules with python
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
                        
                        # backup old values
                        if hasattr(self, name):
                            backup_params[name] = fields[name]
                        
                        # get new values
                        fields[name] = module.__dict__[name]
                        updated_fields.append(name)
                        num_fields_from_filename += 1
                
                # after "filename" execution, we can swap original and local
                # modules name avoiding conflicts
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
                        
                        # let classes and functions inherit the change
                        # through their __module__ attribute
                        # (necessary only for local modules, used in __str__
                        #  and save methods)
                        for name, obj in local_module.__dict__.items():
                            if (callable(obj) or isinstance(obj, type)) and (
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
            
            # post-init operation: since you set them together, check again
            self._process_and_check(updated_fields)
            
            # save?
            if kwargs.get('save', True):
                self.save()
            
            return self
        
        except Exception as exception:
            
            # restore modules
            sys.modules = backup_modules
            
            # restore attributes
            for name, value in backup_params.items():
                self.__dict__[name] = value
            
            raise exception
        
        finally:  # back to the original folder & path
            os.chdir(cwd)
            sys.path.pop(0)
    
    def update(self, *args, **kwargs):
        """Like load but without filename."""
        return self.load(None, *args, **kwargs)
    
    def save(self, path=None, seek_existing_file=True):
        """Save to file and replace params.path.
        MUST be in the same folder as params' working directory.
        path: if None, assign by default
        seek_existing_file: if True, try to replace path with already
        existing file, only when path is not specified
        
        Returns
        -------
        filename: path where saved
        
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
        text.append(f'# packages\n')
        for name in modules:
            if type(modules[name]) is not type(sys):
                continue
            text.append(f'import {modules[name].__name__} as {name}\n')
        
        # copy params
        body = self.__str__(filename.parent)
        text.append(body)
        text = "".join(text)
        
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
