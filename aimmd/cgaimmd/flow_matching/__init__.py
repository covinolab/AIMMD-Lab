"""
aimmd.cgaimmd.flow_matching
===========================

Flow-matching generators for C-GAIMMD.

A flow-matching network carries, next to its committor ``forward``, a velocity
field ``velocity(x_t, t, lam)`` whose ODE turns a prior sample
(``sample_prior``) into a configuration at the committor level ``lam``. This
subpackage holds what is specific to that technique:

- the fixed-step ODE and its generator ``'ode'``
  (:mod:`~aimmd.cgaimmd.flow_matching.sampler`);
- the PaiNN network of C-GAIMMD, ``JointPaiNN``
  (:mod:`~aimmd.cgaimmd.flow_matching.painn`), its Gaussian prior
  (:mod:`~aimmd.cgaimmd.flow_matching.prior`) and the token schemes of its
  atoms (:mod:`~aimmd.cgaimmd.flow_matching.embedding`);
- the mathematics of equivariant flow matching: centering, optimal-transport
  pairing, the probability path and flow-time schedules
  (:mod:`~aimmd.cgaimmd.flow_matching.flowmath`);
- graphs: radius graphs built on the fly
  (:mod:`~aimmd.cgaimmd.flow_matching.graph_batch`) and graph mode for MD
  frames, the bond and topology edges
  (:mod:`~aimmd.cgaimmd.flow_matching.graphs`);
- the online fit, every round trained again from the same start
  (:mod:`~aimmd.cgaimmd.flow_matching.fit`), and the end-state pretraining
  that is that start (:mod:`~aimmd.cgaimmd.flow_matching.pretrain`);
- the documented minimal network (:mod:`~aimmd.cgaimmd.flow_matching.toy`).

Importing the package imports no submodule: every submodule is imported
explicitly (``from aimmd.cgaimmd.flow_matching import sampler``), and heavy
optional dependencies (PyG, networkx, ...) are imported inside the functions
that need them. ``__all__`` names every public submodule; it is computed from
the package directory, so a new module needs no edit here.
"""

import pkgutil as _pkgutil

__all__ = sorted(info.name for info in _pkgutil.iter_modules(__path__)
                 if not info.name.startswith('_'))
