"""
...
"""

# external
import numpy as np
import torch
from math import inf
from torch import Tensor
from scipy.special import expit

# rescaling utils
def find_knots_and_values(extremes1, extremes2, xP1, xP2):
    extremes1 = np.asarray(extremes1)
    extremes2 = np.asarray(extremes2)
    xP1 = np.asarray(xP1)
    xP2 = np.asarray(xP2)

    # process crossing probability
    # codomain of action
    if len(xP1):
        N1 = xP1 / (xP1[-1] * expit(extremes1)[-1])
    else:
        N1 = np.ones(1)
        extremes1 = np.zeros(0)
    if len(xP2):
        N2 = xP2 / (xP2[-1] * expit(extremes2)[-1])
    else:
        N2 = np.ones(1)
        extremes2 = np.zeros(0)
    N10 = N1[0]
    N20 = N2[0]
    vmax = N10
    vmin = 4 / N20
    
    # nothing to do here
    if vmin >= vmax:
        return np.array([]), np.array([])
    
    # remove values at states
    mask = ~np.isinf(extremes1)
    N1 = N1[mask]
    extremes1 = extremes1[mask]
    mask = ~np.isinf(extremes2)
    N2 = N2[mask]
    extremes2 = extremes2[mask]

    # domain of action
    kmin = min(+np.min(extremes1, initial=0.), -np.max(extremes2, initial=0.))
    kmax = min(-np.min(extremes2, initial=0.), +np.max(extremes1, initial=0.))

    # nothing to do here
    if kmin >= kmax:
        return np.array([]), np.array([])
    
    # turn into fine grained interpolation in logit committor space
    # x axis
    q = np.linspace(kmin, kmax, 101)
    x = extremes1[::+1]
    y = N1[::+1]
    N1 = np.exp(np.interp(np.log(expit(q)), np.log(expit(x)), np.log(y)))
    x = -extremes2[::-1]
    y = N2[::-1]
    N2 = np.exp(np.interp(np.log(expit(q)), np.log(expit(x)), np.log(y)))
    
    # TS shift and rescaling computation
    ts = 0.  # initialization
    r = 1.
    from_1_wins = N1 >= N2
    from_2_wins = N1 <  N2
    if N10 > 2 and N20 > 2 and\
        np.sum(from_1_wins) and np.sum(from_2_wins):
        ts = (q[from_1_wins][-1] + q[from_2_wins][0]) / 2
        r = 2. / N1[from_1_wins][-1]
    elif N10 > 2 and np.sum(from_1_wins):
        ts = q[np.argmin(np.abs(N1 / 2 - 1.))]
    elif N20 > 2 and np.sum(from_2_wins):
        ts = np.clip(q[np.argmin(np.abs(N2 / 2 - 1.))], -5., 5.)
    print(f'*** transition state shift by {ts:.3f}, '
          f'total xP rescaling by {r:.3f}')
                    
    # theoretical line
    y = np.zeros(len(q))
    y[q <= ts] = 1 / expit(q[q <= ts] - ts)
    y[q >  ts] = 4 * (1 - expit(q[q > ts] - ts))
    
    # actual line
    y0 = np.append(r * N1[q <= ts], 4 / N2[q > ts] / r)
    
    # determine number of knots / values
    vmin /= r
    vmax *= r
    drop = max(1, N10 * N20 * r ** 2)
    n = min(round(np.log(drop)), 100)
    if not n:
        print(f'!!! rescaling is not possible (yet)')
    else:
        print(f'*** generating {n} (non unique) knots')
    
    # fill
    knots = np.zeros(n)
    values = np.zeros(n)
    for i, v in enumerate(
        np.geomspace(vmin, vmax, n + 2)[::-1][1:-1]):
        knots[i] = q[np.argmin(np.abs(v - y0))]
        values[i] = q[np.argmin(np.abs(v - y))]
    
    # remove redundancies in knots
    knots, indices = np.unique(knots, return_index=True)
    values = np.array(values)[indices]
    
    # remove non-growing or even decreasing knots
    if len(values) > 1:
        keepers = np.diff(values) > 0
        values = np.append(values[0], values[1:][keepers])
        knots = np.append(knots[0], knots[1:][keepers])
    
    return knots, values


def rescale(q, knots, values):
    """in place"""
    I = len(knots)      
    if I < 1:
        return q
    if I < 2:
        x0 = knots[0]
        a = 1
        b = values[0]
        q[:] = a * (q - x0) + b
        return q
    if isinstance(q, Tensor):
        indices = torch.bucketize(q, knots)
    else:
        q = np.asarray(q)
        indices = np.digitize(q, knots)
    for i in range(I + 1):
        if 0 < i < I:
            j = i
        elif i == 0:
            j = 1
        elif i == I:
            j = I - 1
        x0 = knots[max(i - 1, 0)]
        a = ((values[j] - values[j - 1]) /
             (knots[j] - knots[j - 1]))
        b = values[max(i - 1, 0)]
        mask = indices == i
        q[mask] = a * (q[mask] - x0) + b
    return q
