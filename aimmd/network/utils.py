"""

"""

# external
import numpy as np
import torch
from torch import nn

# aimmd
from .._config import print

# placeholder network
class PlaceholderNetwork(nn.Module):
    """Just identity. Loaded by params by default."""

    def __init__(self):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.empty(0))
    
    def forward(self, x):
        return x[:, :1]
    
    def state_dict(self, *args, **kwargs):
        return {}
    
    def load_state_dict(self, state_dict, strict=True):
        pass

placeholder = PlaceholderNetwork()
placeholder.__source__ = 'from aimmd.network import placeholder as network'


def extract_indices_and_series(paths, key, *names):
    """
    back_out: mask, True if it belongs to backward segment, False otheriwse
    forw_out: mask, True if it belongs to forward segment, False otheriwse
    series: additional series"""
    if key is None:
        key = range(len(paths))
    else:
        key = np.arange(len(paths))[key].flatten()
    indices_out = []
    series_out = [[] for name in names]
    back_out = []
    forw_out = []
    for k in key:
        path = paths[k]
        path_type = path.type
        path_indices = path.internal('indices')
        path_back = np.zeros(len(path_indices), dtype=bool)
        path_forw = np.zeros(len(path_indices), dtype=bool)
        si = path.shooting_index - path.indices[0]
        path_back[:si + 1] = True
        if path_type[2] != path_type[1]:
            # only if the path is actually going somewhere...
            path_forw[si:] = True
        path_series = []
        try:  # avoid i/o problems: check wether you can load everything
            for name in names:
                series = path.get(name,
                                  start=path_indices[0],
                                  stop=path_indices[-1] + 1,
                                  raise_if_missing=True)
                assert len(series) == len(path_indices)
                path_series.append(series)
        except Exception as exception:
            # print(exception)
            continue
        indices_out.extend(path_indices)
        back_out.extend(path_back)
        forw_out.extend(path_forw)
        for i, series in enumerate(path_series):
            series_out[i].extend(series)
    return (np.array(indices_out, dtype=int),
            np.array(back_out, dtype=bool),
            np.array(forw_out, dtype=bool),
            *[np.array(series) for series in series_out], len(key))
