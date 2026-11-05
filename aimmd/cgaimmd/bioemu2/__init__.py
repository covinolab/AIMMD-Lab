"""
aimmd.cgaimmd.bioemu2
=====================

C-GAIMMD with BioEmu2 (Microsoft's protein ensemble emulator) as the
generator: the adaptor between AIMMD-Lab and the ``bioemu2`` package.

It needs the ``bioemu2`` extra (bioemu2 0.1.0rc0, Python >= 3.12)::

    pip install 'aimmd-lab[bioemu2]'          # CUDA: 'aimmd-lab[bioemu2-cuda]'

Importing this package imports nothing of bioemu2: the public names below are
resolved on first use, and the first use of a name that needs bioemu2 raises
one ``ImportError`` that says how to install it. Everything else of C-GAIMMD
(PaiNN, the toy, candidate preparation, analysis) works without it.
:mod:`._api` is the only module of AIMMD-Lab that imports bioemu2.

Public names, by module:

- :mod:`~aimmd.cgaimmd.bioemu2.system`: ``BioEmu2System`` (the atom map
  between the MD and BioEmu2, the completion of the atoms BioEmu2 does not
  model, stereocenters, the reference pose, the token pools; a saved system
  loads without bioemu2 when not checked), ``MDAtoms``;
- :mod:`~aimmd.cgaimmd.bioemu2.build`: ``build_system``, ``read_md_atoms``
  (also ``python -m aimmd.cgaimmd.bioemu2 build-system``);
- :mod:`~aimmd.cgaimmd.bioemu2.features`: ``BioEmu2Features`` (the frozen
  BioEmu2 features of the committor head);
- :mod:`~aimmd.cgaimmd.bioemu2.head`: ``CommittorHead``, ``FeatureMLP``,
  ``fit_scaler``, ``SIGMA_REF``, ``LAMBDA_CLIP``;
- :mod:`~aimmd.cgaimmd.bioemu2.adapters`: ``TrunkAdapter``,
  ``ConditionedDenoiser``, ``GuidanceSchedule``, ``parse_guidance_schedule``;
- :mod:`~aimmd.cgaimmd.bioemu2.network`: ``BioEmu2Network`` (the committor
  network of a run; default generator ``'bioemu2_edm'``),
  ``load_archived_network``;
- :mod:`~aimmd.cgaimmd.bioemu2.sampling`: ``BioEmu2Generator``, the
  generation method ``'bioemu2_edm'``;
- :mod:`~aimmd.cgaimmd.bioemu2.fit`: ``sequential_fit`` (the params file's
  ``fit``), ``sequential_round``;
- :mod:`~aimmd.cgaimmd.bioemu2.pretrain`: ``pretrain_initial_network``.

The network, its head and adapters, the fit and the pretraining import
without bioemu2 (a run can be loaded and analysed elsewhere); the features,
the sampling and every BioEmu2 pass need it.
"""

import importlib as _importlib

_LAZY_NAMES = {
    'BioEmu2System': 'system',
    'MDAtoms': 'system',
    'build_system': 'build',
    'read_md_atoms': 'build',
    'BioEmu2Features': 'features',
    'SIGMA_REF': 'head',
    'LAMBDA_CLIP': 'head',
    'CommittorHead': 'head',
    'FeatureMLP': 'head',
    'fit_scaler': 'head',
    'TrunkAdapter': 'adapters',
    'ConditionedDenoiser': 'adapters',
    'GuidanceSchedule': 'adapters',
    'parse_guidance_schedule': 'adapters',
    'BioEmu2Network': 'network',
    'load_archived_network': 'network',
    'BioEmu2Generator': 'sampling',
    'sequential_fit': 'fit',
    'sequential_round': 'fit',
    'pretrain_initial_network': 'pretrain',
}

__all__ = sorted(_LAZY_NAMES)


def __getattr__(name):
    module = _LAZY_NAMES.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(_importlib.import_module(f'{__name__}.{module}'), name)


def __dir__():
    return sorted({*globals(), *_LAZY_NAMES})
