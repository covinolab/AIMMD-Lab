"""
aimmd.resources.gpu
==================

GPU visibility utilities.

This module provides helpers to query which GPUs are visible to the current
process.

Two notions are used:

- total GPUs (as reported by torch)
- available GPUs (as restricted by CUDA_VISIBLE_DEVICES)

The functions are designed for lightweight runtime checks in worker code and
avoid introducing extra dependencies beyond torch.

Notes
-----
- :func:`get_num_gpus` uses ``torch.cuda.device_count()``.
- :func:`get_available_gpus` inspects ``CUDA_VISIBLE_DEVICES``:
    - if unset, GPUs 0..N-1 are considered available
    - if set to a comma-separated list, that list is interpreted as the
      visible device ids
- When CUDA_VISIBLE_DEVICES is set to an empty string, no GPUs are available.
"""

# external
import os
import torch
import warnings


# gpu infos
def get_num_gpus():
    """
    Return the number of CUDA devices detected by torch.

    Returns
    -------
    int
        Number of GPUs returned by ``torch.cuda.device_count()``.
        Returns 0 if torch fails to query CUDA devices.

    Notes
    -----
    Torch can emit warnings when probing device availability (driver mismatch,
    missing runtime, etc.). Warnings are suppressed within this call to keep
    worker logs clean.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore *all* warnings in this block
        try:
            return torch.cuda.device_count()
        except Exception as exception:
            print(f'[Exception]: {exception}')
            return 0


def get_available_gpus():
    """
    Return the GPU ids currently available to the process.

    Returns
    -------
    list[int]
        If ``CUDA_VISIBLE_DEVICES`` is unset, returns ``list(range(get_num_gpus()))``.
        If it is set, parses the comma-separated list and returns the integer ids.
        Empty entries are ignored.

    Notes
    -----
    This function reports the ids visible *inside the current process environment*.
    If a scheduler (e.g., Slurm) sets CUDA_VISIBLE_DEVICES, this reflects the
    scheduler allocation.
    """
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is None:
        return list(range(get_num_gpus()))
    else:
        return sorted([int(id) for id in gpus.split(",") if id != ""])
