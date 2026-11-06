"""
aimmd.cgaimmd.flow_matching.prior
=================================

The prior of flow-matching networks: the center-of-mass-free isotropic Gaussian.

A flow-matching network generates a configuration by integrating its velocity
field from a sample of its prior (flow time ``t = 0``) to a configuration
(``t = 1``, :mod:`aimmd.cgaimmd.flow_matching.sampler`). C-GAIMMD uses one
prior for every system: standard normal flow coordinates projected onto the
subspace without a net translation (:func:`~aimmd.cgaimmd.flow_matching.flowmath.gaussian_prior`).
In Ångström a prior sample is ``coord_scale`` times that
(:meth:`aimmd.cgaimmd.flow_matching.painn.JointPaiNN.from_flow`), so the
network's ``coord_scale`` sets how wide the prior is compared with the data.

What the generator inherits from this prior:

- **No committor dependence.** The prior ignores the target committor; the
  condition enters through the velocity field only.
- **Rotations and atom permutations leave it unchanged**, so the
  optimal-transport pairing of the fit
  (:func:`~aimmd.cgaimmd.flow_matching.flowmath.ot_pair`) may rotate and
  relabel a data sample freely.
- **Mirror images are as likely as the originals.** A velocity field that is
  also symmetric under mirroring (``JointPaiNN(parity_breaking=False)``, the
  E(3) network) therefore generates the wrong enantiomer as often as the right
  one. The SE(3) network (``parity_breaking=True``) can tell the two apart;
  with the E(3) network the validity check must reflect mirror images
  (``reflect_mirror_images`` of
  :class:`aimmd.cgaimmd.preparation.validity.TopologyValidity`).
- **Three of the 3 M degrees of freedom are removed** (the center of mass), so
  with ``M`` atoms every coordinate has variance ``(M - 1) / M``.

The prior has no parameters and no buffers: a checkpoint holds nothing of it.
"""

from numbers import Integral

from . import flowmath


PRIORS = ('gaussian',)
"""The priors of C-GAIMMD's flow-matching networks."""


class GaussianPrior:
    """The center-of-mass-free isotropic Gaussian of ``M`` atoms, in flow units.

    Parameters
    ----------
    n_atoms : int
        Number ``M`` of generated atoms (>= 1).

    Notes
    -----
    :meth:`sample` draws only from the random stream it is given (or torch's
    global stream when none is given): ``torch.randn`` of shape ``(n, M, 3)``,
    minus its mean over the atoms.

    Examples
    --------
    >>> import torch
    >>> prior = GaussianPrior(22)
    >>> z = prior.sample(4, generator=torch.Generator().manual_seed(0))
    >>> tuple(z.shape)
    (4, 22, 3)
    """

    kind = 'gaussian'
    """The name of this prior (one of :data:`PRIORS`)."""

    def __init__(self, n_atoms):
        if isinstance(n_atoms, bool) or not isinstance(n_atoms, Integral) or n_atoms < 1:
            raise ValueError(f'n_atoms must be an int >= 1, got {n_atoms!r}')
        self._n_atoms = int(n_atoms)

    @property
    def n_atoms(self):
        """Number ``M`` of atoms."""
        return self._n_atoms

    def __repr__(self):
        return f'{type(self).__name__}(n_atoms={self._n_atoms})'

    def __eq__(self, other):
        return type(other) is type(self) and other.n_atoms == self.n_atoms

    def __hash__(self):
        return hash((type(self).__name__, self._n_atoms))

    def sample(self, n, *, device=None, generator=None):
        """``(n, M, 3)`` float32 prior samples in flow units, center-of-mass free.

        Parameters
        ----------
        n : int
            Number of samples (>= 0).
        device : torch.device or str, optional
            Of the result (the device of `generator` when one is given).
        generator : torch.Generator, optional
            The random stream; None uses torch's global one.

        Returns
        -------
        torch.Tensor
            ``(n, M, 3)`` float32; every sample's mean over the atoms is zero.
        """
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
            raise ValueError(f'the number of prior samples must be an int >= 0, got {n!r}')
        return flowmath.gaussian_prior((int(n), self._n_atoms, 3), device=device,
                                       generator=generator)

    def settings(self):
        """``{'kind': 'gaussian', 'n_atoms': M}``: what a fit or validation record writes."""
        return {'kind': self.kind, 'n_atoms': self._n_atoms}
