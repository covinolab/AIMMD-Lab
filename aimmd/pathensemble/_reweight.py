"""
...
"""

# external
import numpy as np
from abc import ABC
from math import inf, nan

# aimmd imports
from .reweight import compute_shooting_density
from .reweight import uniformize_factors
from .reweight import compute_crossing_probability
from .reweight import reweight_excursions

class PathEnsembleReweight(ABC):
    
    def reweight(
        self,
        states='ARB',
        free_threshold=50,
        theoretical_threshold=None,
        crossing_probability_cutoff=0.,
        factors_neighbors=10,
        factors_norm=10,
        factors_cutoff=1.,
        sp_cutoff_min=None,
        sp_cutoff_max=None,
        source='values'):

        # process states
        # s: initial state
        # r: reactive region
        # o: final state
        states = str(states).strip().upper()
        if len(states) < 3:
            r = states[0]
            s = o = ''
        else:
            s, r, o = states[:3]
        
        # initialize weights
        weights = np.zeros(len(self))
        
        # get functions
        if s < o:
            extreme_function = lambda x: + np.nanmax(x)
        else:
            extreme_function = lambda x: - np.nanmin(x)
        
        # get types
        types = self.types().view('U1').reshape(len(self), 4)
        
        # absolute indices
        excursion_indices = np.flatnonzero((types[:, 0] != r) &
                                           (types[:, 0] != '.') &
                                           (types[:, 1] == r))
        internal_indices = np.flatnonzero((types[:, 0] == r) &
                                          (types[:, 1] == s))
        
        # reweight excursions
        excursions = self[excursion_indices]
        
        # within excursions indices
        states = types[excursion_indices]
        free = states[:, 3] == s
        shot = states[:, 3] == r
        from_s = states[:, 0] == s
        from_o = states[:, 0] == o
        to_s = states[:, 2] == s
        to_o = states[:, 2] == o
        
        # all (for now)
        factors = np.ones(len(excursion_indices))
        extremes = np.zeros(len(excursion_indices))  # all for now
        shooting_values = excursions.shooting(source)
                
        # shot before sp_cutoff_min -> free
        if sp_cutoff_min is not None:
            mask = shot & (shooting_values < sp_cutoff_min)
            mask = np.flatnonzero(mask)
            f_mask = []
            for i, k in enumerate(mask):
                values = excursions._paths[k].internal(source)
                f_mask.append(1.0 / np.count_nonzero(values < sp_cutoff_min))
                if to_o[i]:
                    extremes[i] = +inf
                else:
                    extremes[i] = extreme_function(values)
            if len(mask):
                factors[mask] = f_mask
                factors[mask] /= np.mean(factors[mask])
                free[mask] = True
                shot[mask] = False
        
        # shot after sp_cutoff_max -> free
        if sp_cutoff_max is not None:
            mask = shot & (shooting_values > sp_cutoff_max)
            mask = np.flatnonzero(mask)
            f_mask = []
            for i, k in enumerate(mask):
                values = excursions._paths[k].internal(source)
                f_mask.append(1.0 / np.count_nonzero(values > sp_cutoff_max))
                if to_o[i]:
                    extremes[i] = +inf
                else:
                    extremes[i] = extreme_function(values)
            if len(mask):
                factors[mask] = f_mask
                factors[mask] /= np.mean(factors[mask])
                free[mask] = True
                shot[mask] = False

        # invert direction of shooting values
        if s > o:  # going in opposite direction
            shooting_values *= -1.0
        
        # free excursions' shooting values and extremes (only from s)
        mask = free & ~to_o
        mask_extremes = []
        for path in excursions.paths[mask]:
            values = path.internal(source)
            mask_extremes.append(extreme_function(values))
        shooting_values[mask] = -inf
        extremes[mask] = mask_extremes
        extremes[free & to_o] = +inf
        
        # free excursions' shooting values and extremes (only from o)
        free_from_o = from_o & to_s & (states[:, 3] == o)
        shooting_values[free_from_o] = +inf
        extremes[free_from_o] = +inf
        
        # we are only left with shot paths (forw/back)
        xP_extremes = []
        xP_shooting_values = []
        for i in np.flatnonzero(shot):
            path = excursions._paths[i]
            shooting_value = shooting_values[i]
            values = getattr(path, source)
                        
            factors[i] = 1.0 / compute_shooting_density(
                values, shooting_value, factors_neighbors)
            if not (to_s[i] or from_s[i]):
                continue
            if to_s[i] or to_o[i]:
                values = values[1:-1]
            else:
                values = values[1:]
            extremes[i] = extreme_function(values)
            shooting_index = path.shooting_index - 1
            # backward part
            if to_s[i]:
                if from_o[i]:
                    xP_extremes.append(+inf)
                else:  # for sure from initial state
                    # no "+1" because already internal values
                    v = values[:shooting_index + 1]
                    xP_extremes.append(extreme_function(v))
                xP_shooting_values.append(shooting_value)
            # forward part
            if from_s[i]:
                if to_o[i]:
                    xP_extremes.append(+inf)
                else:
                    v = values[shooting_index:]
                    xP_extremes.append(extreme_function(v))
                xP_shooting_values.append(shooting_value)
        
        # uniformize factors among shot excursions
        if factors_norm:
            mask = shot * (factors > 0)
            factors[mask] = uniformize_factors(
                factors[mask], shooting_values[mask],
                factors_cutoff, factors_norm)
        
        # final processing
        shot_transitions = shot & to_s & from_o
        factors[shot_transitions] /= 2  # otw counting them double
        mask = factors > 0
        factors[mask] = np.clip(factors[mask], 1e-3, 1000.)
        
        # just factors
        if s == o:
            return (weights, np.arange(len(self)),
                    factors, shooting_values, extremes,
                    extremes * 0, factors, extremes * 0)

        
        # compute crossing probability
        # for each extreme value: the crossing probability
        xP_extremes, xP = compute_crossing_probability(
            xP_shooting_values,
            xP_extremes,
            extremes[free],
            factors[free],
            free_threshold,
            theoretical_threshold,
            crossing_probability_cutoff)
        
        # reweight excursions linked to state s
        excursions_mask = from_s | to_s
        w, order, shooting_values, extremes, xP, f, m = reweight_excursions(
            shooting_values[excursions_mask],
            extremes[excursions_mask],
            factors[excursions_mask],
            xP_extremes, xP)
        weights[excursion_indices[excursions_mask]] = w
        
        # reweight internal segments: by default, all the same weight
        internal_weights = np.ones(len(internal_indices))

        # special treatment: internal segments which were shot
        int_shot = types[internal_indices, 3] != r
        if int_shot.any():  # if shot in init state, special treatment!
            internal_weights[int_shot] = (
                1 / self[internal_indices[int_shot]].n_frames)
            internal_weights[int_shot] /= internal_weights[int_shot].mean()
        
        # normalization
        norm = internal_weights.sum()
        if norm:
            internal_weights /= norm
            internal_weights *= weights.sum() or 1.
        weights[internal_indices] = internal_weights
        
        # return
        return (weights, excursion_indices[excursions_mask][order],
                factors, shooting_values, extremes, xP, f, m)
    
    def factors(self, neighbors=10, norm=10, cutoff=1.,
                sp_cutoff_min=None, sp_cutoff_max=None):
        return self.reweight('',
                             factors_neighbors=neighbors,
                             factors_norm=norm,
                             factors_cutoff=cutoff,
                             sp_cutoff_min=sp_cutoff_min,
                             sp_cutoff_max=sp_cutoff_max)[0]
