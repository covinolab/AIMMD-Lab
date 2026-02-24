"""
...
"""

# external
import numpy as np
from abc import ABC
from math import inf
from tqdm import tqdm
from collections.abc import Iterable

# aimmd imports
from ..path.chainreader import ChainReader

# project
class PathEnsembleProject(ABC):
    def _project_batch(self, bins, function, source,
                       batch_input, batch_weight):
        if source == 'reader':
            data = ChainReader(*batch_input)
        else:
            data = np.concatenate(batch_input, axis=0)
        weights = np.concatenate(batch_weight)
        data = np.asarray(function(data))
        data = data.reshape((len(data), -1))
        return np.histogramdd(data, bins,
            density=False, weights=weights)[0]
    
    def project(self, bins=[-inf, +inf],
                key=None, weights=None,
                function=lambda x:x, source='values',
                where='internal', values_source='values',
                vmin=None, vmax=None,
                batch_size=4096, verbose=False):
        
        # process bins
        if isinstance(bins, Iterable):
            if not isinstance(bins[0], Iterable):
                bins = [bins]
        result = np.zeros([len(b) - 1 for b in bins])

        # get paths
        paths = np.arange(len(self))[key].flatten()
        
        # nothing
        if not len(paths):
            return result

        # get unique paths and weights
        indices, counts = np.unique(paths, return_counts=True)
        weights = weights or self.weights[indices]
        keepers = self.accepted[indices] & ~np.isnan(weights) & (weights != 0)
        indices = indices[keepers]
        counts = counts[keepers]
        weights = weights[keepers]
        weights *= counts
        
        # process vmin and vmax
        if vmin is None and vmax is None:
            pass
        elif vmin is None and vmax is not None:
            vmin = -inf
        elif vmax is None and vmin is not None:
            vmax = +inf
        
        # compute in batches 
        batch_input = []
        batch_weight = []
        current_size = 0
        lengths = np.array([len(path) for path in self._paths]).astype(int)
        progress = tqdm(total=lengths.sum(), disable=not verbose)
        for i, weight in zip(indices, weights):
            path = self._paths[i]
            states = path.type
            if where in ('forward', 'backward'):
                shooting_k, shooting_i = path._get_local_index(
                    path.shooting('indices'))
            n_files = path.n_files
            for k in range(n_files):
                if (where == 'forward' and shooting_k > k or
                    where == 'backward' and shooting_k < k):
                    if verbose:
                        progress.update(lengths[i])
                    continue
                try:
                    # data not present: hence cannot project
                    # that's because data weren't computed on the whole source
                    input_data = path._extract(k, source)
                except:
                    continue
                input_data_length = len(input_data)
                if k == 0 and where != 'all' and states[0] != states[1]:
                    start = 1
                else:
                    start = 0
                if (k == n_files - 1 and where != 'all' and
                    states[1] != states[2]):
                    stop = input_data_length - 1
                else:
                    stop = input_data_length
                if where == 'forward' and shooting_k == k:
                    start = max(start, shooting_i)
                if where == 'backward' and shooting_k == k:
                    stop = min(stop, shooting_i + 1)
                if start or stop < input_data_length:
                    input_data = input_data[start:stop]
                    if verbose:
                        progress.update(start + input_data_length - stop)
                if vmin is not None:
                    if source == values_source:
                        values = input_data
                    else:
                        values = path._extract(k, values_source)
                remaining = len(input_data)
                current = 0
                while remaining:
                    delta = min(batch_size - current_size, remaining)
                    batch_input.append(input_data[current:current + delta])
                    if vmin is not None:
                        if vmax == inf:
                            batch_weight.append(weight * (values >= vmin))
                        elif vmin == -inf:
                            batch_weight.append(weight * (values < vmax))
                        else:
                            batch_weight.append(
                                weight * (values >= vmin) * (values < vmax))
                    else:
                        batch_weight.append([weight] * delta)
                    current += delta
                    current_size += delta
                    remaining -= delta
                    if current_size >= batch_size:
                        result += self._project_batch(
                            bins, function, source, batch_input, batch_weight)
                        if verbose:
                            progress.update(current_size)
                        current_size = 0
                        batch_input = []
                        batch_weight = []
        
        # last computation and return
        if current_size:
            result += self._project_batch(
                bins, function, source, batch_input, batch_weight)
        progress.update(progress.total - progress.n)
        progress.close()
        return result
