"""
...
"""

from . import _config

def initialize():
    if _config._initialized:
        return
    
    # external (check dependencies)
    import os
    import sys
    import dill
    import tqdm
    import numpy
    import scipy
    import torch  # make torch work
    import torch._dynamo
    import shutil
    import filelock
    import warnings
    import functools
    import matplotlib
    import MDAnalysis
    import multiprocessing
    from pathlib import PosixPath

    # aimmd imports
    from .cache import NpyReaderCache, MDAReaderCache
    
    #########################
    # executables and paths #
    #########################
    
    _config.PYTHON = sys.executable
    _config.GROMACS = shutil.which('gmx') or shutil.which('gmx_mpi')
    if _config.GROMACS is None:
        raise EnvironmentError(
            'GROMACS exec not found in PATH. Please install GROMACS and '
            'ensure \'gmx\' or \'gmx_mpi\' is accessible in your environment.'
        )
    _config.WORKER = str(
        PosixPath(__file__).resolve().parent / "worker" / "run.py")
    
    ##########
    # caches #
    ##########
    
    _config.NPY_CACHE = NpyReaderCache()
    _config.MDA_CACHE = MDAReaderCache()
    
    #################
    # print options #
    #################
    
    # quick logging
    _config.print = functools.partial(print, flush=True)
    
    # suppress some warnings in MDAnalysis
    _old_del = MDAnalysis.coordinates.base.ReaderBase.__del__
    
    def _safe_del(self):
        try:
            _old_del(self)
        except (TypeError, AttributeError):
            # Suppress AttributeError
            # caused by missing _xdr attribute during __del__
            pass
    
    MDAnalysis.coordinates.base.ReaderBase.__del__ = _safe_del
    
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="MDAnalysis.coordinates.XDR")
    
    # more compact array
    numpy.set_printoptions(precision=3)

    ###################
    # multiprocessing #
    ###################
    
    # call just once, for compatibility issues
    # when spawning multiple processes
    torch.set_num_interop_threads(1)

    ####################
    # trajectories i/o #
    ####################

    _config.DEFAULT_DIMENSIONS = numpy.array([0., 0., 0., 90., 90., 90.])
    
    #########
    # done! #
    #########
    
    _config._initialized = True
