'''
aimmd — AI for molecular mechanism discovery

Disclaimer: this file was generated with the help of ChatGPT.
'''

from __future__ import annotations
import importlib
import shutil
from typing import Dict

__all__ = ['__version__', 'check_dependencies']

__version__ = '0.1.0'

# required Python dependencies (import name → pip name)
_REQUIRED_DEPS: Dict[str, str] = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'MDAnalysis': 'MDAnalysis',
    'mdtraj': 'mdtraj',
    'torch': 'torch',
    'matplotlib': 'matplotlib',
    'tqdm': 'tqdm',
}


# dependency check function
def check_dependencies(
    verbose: bool = True, check_gromacs: bool = True) -> None:
    '''
    Check that all required dependencies are importable.
    Optionally verify that GROMACS ('gmx' or 'gmx_mpi') is available in PATH.
    
    Parameters
    ----------
    verbose : bool, optional
        If True, prints a confirmation message when all dependencies are found.
    check_gromacs : bool, optional
        If True, also check for the GROMACS command-line executable.
    '''
    missing = []
    for mod, pipname in _REQUIRED_DEPS.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append((mod, pipname))
    
    if missing:
        msg = (
            'The following dependencies are missing for AIMMD:\n'
            + '\n'.join(f' - {mod} (pip install {pip})' for mod, pip in missing)
        )
        raise ImportError(msg)
    
    # check for GROMACS executable
    if check_gromacs:
        if shutil.which('gmx') is None and shutil.which('gmx_mpi') is None:
            raise EnvironmentError(
                'GROMACS executable not found in PATH.\n'
                'Please ensure GROMACS is installed and its `bin/` directory '
                'is included in your PATH environment variable.\n\n'
                'For example:\n'
                '    source /usr/local/gromacs/bin/GMXRC'
            )
    
    if verbose:
        print('All AIMMD dependencies are available.' +
              (' (GROMACS found)' if check_gromacs else ''))


# summary/representation
def _summary() -> str:
    return f'AIMMD v{__version__} — AI for molecular mechanism discovery'


def __repr__():
    return _summary()
