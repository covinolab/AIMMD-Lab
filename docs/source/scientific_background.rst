Scientific Background
=====================

The code in this repository is best understood as a practical implementation
of the method family developed across the AIMMD papers.

The central object is the **committor** :math:`q(x)`, the probability that a
trajectory started from configuration :math:`x` reaches the product state
:math:`B` before the reactant state :math:`A`:

.. math::

   q(x) = \Pr\!\left[\, \tau_B < \tau_A \,\mid\, x(0) = x \,\right],

where :math:`\tau_A` and :math:`\tau_B` are the first-passage times to each
state. The committor is the ideal reaction coordinate: the dividing surface
:math:`q(x) = 1/2` is the transition state. AIMMD learns a model of :math:`q`
from sampled trajectories and uses it to bias shooting-point selection toward
the transition-state region, where new trajectories are most informative.

AIMMD as a Learning-Guided Path-Sampling Loop
---------------------------------------------

H. Jung, R. Covino, A. Arjun, C. Leitold, C. Dellago, P. G. Bolhuis, and
G. Hummer, *Machine-Guided Path Sampling to Discover Mechanisms of Molecular
Self-Organization*, Nat. Comput. Sci. 2023, 3, 334-345.
DOI: https://doi.org/10.1038/s43588-023-00428-z

This paper introduces the central AIMMD idea: rare-event path sampling and a
learned mechanistic model form a closed loop. Newly generated trajectories are
used to build and validate a committor-like description, and that description
then guides where the next trajectories should start. This forms
an autonomous mechanism-discovery workflow for rare molecular events.

In the repository, that logic appears most directly in:

- :mod:`aimmd.worker._shoot` for guided two-way shooting,
- :mod:`aimmd.worker._train` for iterative model updates,
- :mod:`aimmd.launcher` for coordinating concurrent workers,
- and :mod:`aimmd.network.fit` for learning the committor surrogate.

Reweighting Paths into Free Energies, Rates, and Mechanisms
-----------------------------------------------------------

G. Lazzeri, H. Jung, P. G. Bolhuis, and R. Covino,
*Molecular Free Energies, Rates, and Mechanisms from Data-Efficient Path
Sampling Simulations*, J. Chem. Theory Comput. 2023, 19, 9060-9076.
DOI: https://doi.org/10.1021/acs.jctc.3c00821

This paper supplies the statistical interpretation layer: once AIMMD has
generated a mixture of free trajectories, excursions, and transition paths, the
ensemble can be reweighted to recover equilibrium observables such as free
energy profiles and rate-related quantities.

That is the scientific rationale behind:

- :mod:`aimmd.pathensemble.reweight`,
- :mod:`aimmd.pathensemble._reweight`,
- :func:`aimmd.analysis.utils.compute_bins`,
- and the reporting and projection methods on :class:`aimmd.PathEnsemble`,

although the details of reweighting have been updated in the 2025 paper.

Rejection-Free Path Sampling and reformulated reweighting
---------------------------------------------------------

G. Lazzeri, P. G. Bolhuis, and R. Covino,
*Optimal Rejection-Free Path Sampling*, arXiv, 2025.
Link: https://arxiv.org/html/2503.21037v1

This shifts the focus from sampling a restricted path ensemble to sampling
the distribution of shooting points directly. The key idea is that if shooting
points are selected with a bias proportional to the inverse free energy along a
good reaction coordinate, the reactive region can be sampled much more
uniformly, and the resulting algorithm can be rejection-free.

In AIMMD, that perspective is implemented as:

- the ``rfps`` chain type,
- the adaptive shooting-point selection pool,
- the interplay between bins, densities, and network values,
- a reformulated reweighting scheme (via the drop-method).

How the Repository Encodes the Theory
-------------------------------------

The repository decomposes the method into clear implementation units:

:class:`aimmd.Params`
   Encodes the scientific parameters of a run: states, descriptors, values, network,
   engine details, and sampling controls.

:meth:`aimmd.Worker.shoot`
   Implements the enhanced sampling step by selecting shooting points and
   generating new two-way trajectories.

:meth:`aimmd.Worker.train`
   Updates the learned committor surrogate, recomputes values, and refreshes
   the adaptive discretization of value space.

:meth:`aimmd.PathEnsemble.reweight`
   Converts the biased sampled data back into equilibrium-weighted estimates.

:mod:`aimmd.analysis.utils`
   Supplies support routines for binning, uncertainty estimation, and related
   post-processing steps.
