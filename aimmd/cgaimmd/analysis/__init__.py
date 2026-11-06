"""
aimmd.cgaimmd.analysis
======================

Tools to judge a C-GAIMMD model and its generator after (or during) a run.

- :mod:`~aimmd.cgaimmd.analysis.validate`: committor accuracy of any AIMMD
  network against a brute-force reference (reference loaders, parity
  metrics, the parity figure), and the per-checkpoint validation protocol of
  a C-GAIMMD model (validity, committor accuracy, generator against
  committor, guard against copies; a JSON report and a model card);
- :mod:`~aimmd.cgaimmd.analysis.adherence`: how closely a generator follows
  its requests (resolution, slopes per region, neighbour AUC), and how its
  samples compare with MD (variety, near-copies, departure, structural
  match, the data ceiling);
- :mod:`~aimmd.cgaimmd.analysis.shooting_validation`: the committor of
  generated configurations measured by shooting (per-request committor
  mean and variance, the dial slope, comparisons), from a sweep or from the
  run's own shots, and the writer of the frames to shoot;
- :mod:`~aimmd.cgaimmd.analysis.committor_baseline`: a distance-matrix
  committor network trained post hoc on a run's own shooting data, the
  baseline to compare a C-GAIMMD committor with.

Rates, free energies and the per-frame weights of a run are generic AIMMD
analysis: :mod:`aimmd.analysis.reweighting` and :mod:`aimmd.analysis.free_md`.
The per-shot table of a C-GAIMMD run is :func:`aimmd.cgaimmd.sidecar.shot_table`.

No tool here writes into a run folder: results are returned, and files are
written only where the caller names them.

Importing the package imports no submodule; the public names below are
resolved on first use.
"""

import importlib as _importlib
import pkgutil as _pkgutil

_LAZY_NAMES = {
    'CommittorReference': 'validate',
    'committor_metrics': 'validate',
    'calibration_fit': 'validate',
    'shot_counts': 'validate',
    'score_committor': 'validate',
    'network_committor': 'validate',
    'params_committor': 'validate',
    'heldout_chain_reference': 'validate',
    'sweep_reference': 'validate',
    'parity_figure': 'validate',
    'adherence_metrics': 'adherence',
    'structural_match': 'adherence',
    'sample_quality': 'adherence',
    'label_binned_reference': 'adherence',
    'data_ceiling': 'adherence',
    'pair_distance_features': 'adherence',
    'level_summary': 'shooting_validation',
    'dial_slope': 'shooting_validation',
    'compare_dials': 'shooting_validation',
    'sweep_outcomes': 'shooting_validation',
    'validation_rows': 'shooting_validation',
    'from_shot_table': 'shooting_validation',
    'write_validation_frames': 'shooting_validation',
    'DistanceMLP': 'committor_baseline',
    'run_distance_scaler': 'committor_baseline',
    'train_on_run': 'committor_baseline',
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
