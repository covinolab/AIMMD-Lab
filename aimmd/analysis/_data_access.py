"""How the analysis tools read a run folder: without the frames of biased
simulations unless the caller asks for them.

Private helpers of :mod:`aimmd.analysis` and :mod:`aimmd.cgaimmd.analysis`.

Frames of biased simulations
----------------------------
A bias changes which configurations a simulation visits and how it moves
between them. Frames of biased simulations (umbrella sampling, metadynamics,
OPES, steered pulls) are therefore not used as training data, as committor
references or for input scalers unless the caller decides so explicitly
(:func:`refuse_biased_frames`). A run is recognised as biased by
``params.record_bias``; a bias that the run does not record cannot be seen
here.
"""

# ════════════════════════════════════════════════════════════════════════════
# frames of biased simulations
# ════════════════════════════════════════════════════════════════════════════

def refuse_biased_frames(params, use, allow_biased_frames=False):
    """Refuse a run that records a bias as the source of `use`, unless the caller allows it.

    Parameters
    ----------
    params : aimmd.Params
        The run's params.
    use : str
        What the frames would become, for the message (``'training data'``,
        ``'a committor reference'``, ``'an input scaler'``).
    allow_biased_frames : bool, optional
        True: the caller has decided to use the frames anyway.

    Raises
    ------
    ValueError
        If ``params.record_bias`` is set and `allow_biased_frames` is False.
    """
    if allow_biased_frames or not bool(getattr(params, 'record_bias', False)):
        return
    raise ValueError(
        f'the run records a bias (params.record_bias): its simulations ran under the bias, and '
        f'frames of biased simulations are not used as {use} unless that is decided '
        f'explicitly. Pass allow_biased_frames=True to use them anyway (the docstring says when '
        f'that can be justified).')
