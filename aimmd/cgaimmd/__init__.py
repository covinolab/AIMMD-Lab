"""
aimmd.cgaimmd
=============

Committor-conditioned Generative AIMMD (C-GAIMMD): the tooling of
``chain_type='cgaimmd'``.

With ``chain_type='cgaimmd'`` every reactive-region shooting point is
generated at a drawn target committor by a *generator* that is conditioned on
the committor, checked, prepared and then shot and book-kept exactly like an
RFPS shooting point. The network is the committor model of the run (its
``forward`` is used by all of AIMMD's value machinery,
:mod:`aimmd.cgaimmd.protocol`); how candidates are generated is chosen through
one interface (:mod:`aimmd.cgaimmd.generation`), so that the flow-matching ODE
of a PaiNN network (:mod:`aimmd.cgaimmd.flow_matching`), BioEmu2's diffusion
sampler (:mod:`aimmd.cgaimmd.bioemu2`) and later methods are interchangeable.

This package sits at the learning level, next to :mod:`aimmd.network`.
Importing it imports no submodule: the public names below are resolved on
first use, every submodule can be imported explicitly
(``from aimmd.cgaimmd import protocol``), and heavy optional dependencies
(PyG, networkx, bioemu2, ...) are imported inside the functions or
subpackages that need them, so ``import aimmd`` stays light.

Public names, by module:

- :mod:`~aimmd.cgaimmd.generation`: ``Generator``, ``register_generator``,
  ``make_generator``, ``check_generator``, ``available_generators``;
- :mod:`~aimmd.cgaimmd.protocol`: ``check_network_protocol``,
  ``check_network_numerics``, ``is_cgaimmd_network``;
- :mod:`~aimmd.cgaimmd.values`: ``ProtocolValuesFunction``,
  ``protocol_values_function``;
- :mod:`~aimmd.cgaimmd.dataset`: ``TrainingDataset``, ``extract_dataset``;
- :mod:`~aimmd.cgaimmd.types`: ``CandidateContext``, ``ValidityResult``,
  ``PreparedFrame``, ``ShootingPoint``, ``NetworkInfo``;
- :mod:`~aimmd.cgaimmd.source` (the shooting-point source of the shooting
  worker): ``GenerativeShootingPointSource``, ``make_cgaimmd_source``,
  ``prepare_candidate``, ``candidate_context``, ``PreparedCandidate``;
- :mod:`~aimmd.cgaimmd.sidecar` (the per-shot records of a run):
  ``read_shot_sidecar``, ``shot_table``, ``summarise_shots``;
- :mod:`~aimmd.cgaimmd.restart` (continuing a run as C-GAIMMD, post-training
  on an existing run): ``check_cgaimmd_restart``, ``derive_run``,
  ``migrate_descriptors``;
- :mod:`~aimmd.cgaimmd.preflight` (``python -m aimmd.cgaimmd.preflight
  params.py``, the checks before a job): ``run_preflight``.

The target-committor laws are in :mod:`~aimmd.cgaimmd.lambda_law`, the
candidate pool and the network watcher in :mod:`~aimmd.cgaimmd.pool`, the
PLUMED bias of the shots in :mod:`~aimmd.cgaimmd.plumed`, and what the online
fits share in :mod:`~aimmd.cgaimmd.fitting`.

Subpackages:

- :mod:`aimmd.cgaimmd.preparation`: the GROMACS tools that turn a generated
  configuration into a shooting point (descriptors, validity checks, the
  polish, re-solvation, minimization);
- :mod:`aimmd.cgaimmd.flow_matching`: the PaiNN network with its
  flow-matching generator ``'ode'``, its fit and its end-state pretraining;
- :mod:`aimmd.cgaimmd.bioemu2`: the BioEmu2 adaptor with its generator
  ``'bioemu2_edm'`` (needs the ``bioemu2`` extra);
- :mod:`aimmd.cgaimmd.analysis`: the tools that judge a model and its
  generator after a run (committor parity against brute-force sets,
  generator adherence, shooting validation, a distance-matrix committor
  baseline).
"""

import importlib as _importlib
import pkgutil as _pkgutil

_LAZY_NAMES = {
    'Generator': 'generation',
    'register_generator': 'generation',
    'make_generator': 'generation',
    'check_generator': 'generation',
    'available_generators': 'generation',
    'check_network_protocol': 'protocol',
    'check_network_numerics': 'protocol',
    'is_cgaimmd_network': 'protocol',
    'ProtocolValuesFunction': 'values',
    'protocol_values_function': 'values',
    'TrainingDataset': 'dataset',
    'extract_dataset': 'dataset',
    'CandidateContext': 'types',
    'ValidityResult': 'types',
    'PreparedFrame': 'types',
    'ShootingPoint': 'types',
    'NetworkInfo': 'types',
    'GenerativeShootingPointSource': 'source',
    'make_cgaimmd_source': 'source',
    'prepare_candidate': 'source',
    'candidate_context': 'source',
    'PreparedCandidate': 'source',
    'read_shot_sidecar': 'sidecar',
    'shot_table': 'sidecar',
    'summarise_shots': 'sidecar',
    'check_cgaimmd_restart': 'restart',
    'derive_run': 'restart',
    'migrate_descriptors': 'restart',
    'run_preflight': 'preflight',
}

_OPTIONAL_SUBPACKAGES = ('bioemu2',)
"""Subpackages that need a pip extra; left out of ``__all__`` (a star import would fail)."""

_SUBMODULES = sorted(info.name for info in _pkgutil.iter_modules(__path__)
                     if not info.name.startswith('_'))

__all__ = sorted({*(name for name in _SUBMODULES if name not in _OPTIONAL_SUBPACKAGES),
                  *_LAZY_NAMES})


def __getattr__(name):
    module = _LAZY_NAMES.get(name)
    if module is not None:
        return getattr(_importlib.import_module(f'{__name__}.{module}'), name)
    if name in _SUBMODULES:
        return _importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted({*globals(), *_SUBMODULES, *_LAZY_NAMES})
