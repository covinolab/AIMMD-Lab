"""
aimmd.cgaimmd.preparation
=========================

Candidate preparation for C-GAIMMD: the tools that turn a generated
configuration of the generator atoms into a shooting point of the run's full
system, for systems simulated with GROMACS.

The params file of a run wraps them in its hooks:

- ``descriptors_function``: :class:`~.descriptors.GeneratorDescriptors`, the
  flat coordinates of the generator atoms (:class:`~.descriptors.GeneratorAtoms`:
  the solute, and optionally a cluster of waters chosen by
  :class:`~.descriptors.ClusterSelection`);
- ``cgaimmd_validity_function``: :class:`~.validity.TopologyValidity` (bond
  graph, relabelling into topology order, chirality, clashes), or
  :class:`~.validity.PolishedValidity`, which checks each raw sample, polishes it
  with a restrained minimization (:func:`~.minimize.polish`) and checks the
  polished sample;
- ``cgaimmd_prepare_function``: :class:`~.solvate.GromacsSolvator`, which puts a
  solute-only or cluster candidate back into the run's periodic system
  (solvate, ions, restrained EM, optional restrained NVT stages,
  :class:`~.solvate.NVTStage`);
- ``cgaimmd_minimize_function``: :class:`~.minimize.GromacsMinimizer`, a
  restrained energy minimization of the prepared candidate.

:mod:`~.gromacs` holds the GROMACS plumbing they share, :mod:`~.topology` the
GROMACS topology reader (masses, elements, bonds, restrained topologies), and
:mod:`~.symmetry` the permutation symmetries of the generator atoms.

Importing the package imports no submodule: the names above resolve on first
use, and MDAnalysis and networkx are imported inside the functions that need
them.

Examples
--------
>>> from aimmd.cgaimmd.preparation import GeneratorAtoms, GeneratorDescriptors  # doctest: +SKIP
>>> _descriptors = GeneratorDescriptors('run.gro', GeneratorAtoms(n_solute=166))  # doctest: +SKIP
>>> def descriptors_function(trajectory):                                    # doctest: +SKIP
...     return _descriptors(trajectory)
"""

import importlib as _importlib
import pkgutil as _pkgutil

_LAZY_NAMES = {
    'GeneratorAtoms': 'descriptors',
    'GeneratorDescriptors': 'descriptors',
    'ClusterSelection': 'descriptors',
    'TopologyValidity': 'validity',
    'PolishedValidity': 'validity',
    'GromacsMinimizer': 'minimize',
    'polish': 'minimize',
    'GromacsSolvator': 'solvate',
    'NVTStage': 'solvate',
    'ExclusionSphere': 'solvate',
    'read_masses': 'topology',
}

_SUBMODULES = sorted(info.name for info in _pkgutil.iter_modules(__path__)
                     if not info.name.startswith('_'))

__all__ = sorted({*_SUBMODULES, *_LAZY_NAMES})


def __getattr__(name):
    module = _LAZY_NAMES.get(name)
    if module is not None:
        return getattr(_importlib.import_module(f'{__name__}.{module}'), name)
    if name in _SUBMODULES:
        return _importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted({*globals(), *_SUBMODULES, *_LAZY_NAMES})
