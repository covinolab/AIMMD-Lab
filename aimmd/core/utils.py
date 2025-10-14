import os
import sys
import copy
import time
import torch
import numpy as np
import mdtraj as md
import pickle
import psutil
import shutil
import select
import signal
import asyncio
import inspect
import argparse
import warnings
import importlib
import importlib.util
import itertools
import threading
import MDAnalysis as mda
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from time import sleep
from tqdm import tqdm
from datetime import datetime
from textwrap import wrap
from scipy.special import logit, expit
from mdtraj.formats import TRRTrajectoryFile
from .pathensemble import *

###############################################################################
# Base utils ##################################################################
###############################################################################

def has_param(func, param_name):
    signature = inspect.signature(func)
    return param_name in signature.parameters

def update_results(filename, key=[], result=[]):
    try:
        with open(filename, 'rb') as file:
            dictionary = pickle.load(file)
    except:
        dictionary = {}
    
    save = False
    for k, r in zip(key, result):
        dictionary[k] = r
        save = True
    
    if save:
        with open(filename, 'wb') as file:
            pickle.dump(dictionary, file)
    
    return dictionary


# Suppress only UserWarnings in MDAnalysis
import MDAnalysis.coordinates.base as base

_old_del = base.ReaderBase.__del__

def _safe_del(self):
    try:
        _old_del(self)
    except AttributeError:
        # Suppress AttributeError caused by missing _xdr attribute during __del__
        pass

base.ReaderBase.__del__ = _safe_del


# More compact array visualization
np.set_printoptions(precision=3)


def array2string(array, initial_spaces, wrap_size=80, formatter=None):
    return np.array2string(
        array, wrap_size, prefix=' ' * (initial_spaces + 1),
        formatter=formatter)


def convert_seconds(seconds):
    days = int(seconds // 86400)
    seconds %= 86400
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    seconds = max(seconds - .5, 0.)
    return f'{days} days, {hours:02g}:{minutes:02g}:{seconds:02.0f}'


def now():
    return str(datetime.now())[11:19]


def write(text, *paths, wrap_text=False):
    if wrap_text:
        text = "\n".join(wrap(text, 80,
            break_long_words=False, replace_whitespace=False))
    text = text.replace('"',"'")
    for path in paths:
        if path is not None:
            os.system(f'''echo "{text}" >> {path}''')
    if not len(paths):
        print(text)


def remove(path, verbose=True):
    while os.path.exists(path):
        try:
            os.remove(path)
        except:
            continue
        if verbose:
            print(f'--- removed {path}')


def initialize_plot():
    figure, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    plt.subplots_adjust(left=0.18, bottom=0.18, right=0.99, top=0.8)
    return figure, ax


class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        if instance is None:
            descr_get = super().__get__
        else:
            descr_get = self.__func__.__get__
        return descr_get(instance, type_)

class DummyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # no parameters, empty network
    
    def forward(self, x):
        # just returns the input unchanged
        return x
    
    def state_dict(self, *args, **kwargs):
        # override to return empty dict
        return {}
    
    def load_state_dict(self, state_dict, strict=True):
        # accept any input but do nothing
        pass

###############################################################################
# Math ########################################################################
###############################################################################

def solve_committor_by_relaxation(
        X, Y, Fx, Fy, A, B, P0, progress=[5, 4, 2, 1]):
    """
    Compute committor in 2D with relaxation method (Brownian dynamics).

    Parameters
    ----------
    X: x-coordinates on a 2D grid
    Y: y-coordinates on a 2D grid
    Fx: force's x component on a 2D grid
    Fy: force's y component on a 2D grid
    A: points in A on a 2D grid
    B: points in B on a 2D grid
    P0: initial guess for committor on a 2D grid
    progress: iteratively increase the resolution based on the vector's values

    Returns
    -------
    P0: committor estimate
    """
    for split in tqdm(progress, position=0, file=sys.stdout):
        X1 = X[::split, ::split]
        Y1 = Y[::split, ::split]
        P1 = P0[::split, ::split]
        Fx1 = Fx[::split, ::split]
        Fy1 = Fy[::split, ::split]
        A1 = A[::split, ::split]
        B1 = B[::split, ::split]
        dFx = np.diff(X1, axis=1)[1:-1, :-1] * Fx1[1:-1, 1:-1]
        dFy = np.diff(Y1, axis=0)[:-1, 1:-1] * Fy1[1:-1, 1:-1]
        dFx[dFx > +1.] = +1.
        dFx[dFx < -1.] = -1.
        dFy[dFy > +1.] = +1.
        dFy[dFy < -1.] = -1.
        r = np.max(np.abs(
            (((P1[2:, 1:-1] + P1[:-2, 1:-1] - 2 * P1[1:-1, 1:-1]) +
              (P1[1:-1, 2:] + P1[1:-1, :-2] - 2 * P1[1:-1, 1:-1])) +
             (dFx * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
              dFy * (P1[1:-1, 2:] - P1[1:-1, :-2])) / 2)
        ))
        while True:
            r1 = 0 + r
            for i in range(100):
                P1[1:-1, 1:-1] = (2 * (P1[2:, 1:-1] + P1[:-2, 1:-1] +
                                       P1[1:-1, 2:] + P1[1:-1, :-2]) +
                               (dFy * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
                                dFx * (P1[1:-1, 2:] - P1[1:-1, :-2]))) / 8
                P1[:, 0] = P1[:, 1]
                P1[:, -1] = P1[:, -2]
                P1[0, :] = P1[1, :]
                P1[-1, :] = P1[-2, :]
                P1[P1 < 0] = 0
                P1[P1 > 1] = 1
                P1[A1] = 0
                P1[B1] = 1
            r = np.max(np.abs(((
                 (P1[2:, 1:-1] + P1[:-2, 1:-1] - 2 * P1[1:-1, 1:-1]) +
                 (P1[1:-1, 2:] + P1[1:-1, :-2] - 2 * P1[1:-1, 1:-1])) +
                 (dFx * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
                  dFy * (P1[1:-1, 2:] - P1[1:-1, :-2])) / 2)))
            if np.abs(r - r1) < 1e-16:
                break
        if split > 1:
            P0 = np.array([interpolate(x, y, P1, X1, Y1)
                           for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
        else:
            P0 = P1.copy()
    return P0

def rescale(q, knots, values):
    I = len(knots)
    try:
        indices = np.digitize(q, knots)
    except:
        indices = torch.bucketize(q, knots)
        
    if I < 1:
        return q
    if I < 2:
        x0 = knots[0]
        a = 1
        b = values[0]
        q = a * (q - x0) + b
        return q
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
        q[indices == i] = a * (q[indices == i] - x0) + b
    return q

###############################################################################
# Handle workers ##############################################################
###############################################################################

def get_current_simulation(worker_id):
    if not os.path.exists(worker_id):
        return ''
    with open(worker_id, 'r') as f:
        try:
            return f.read().split()[0]
        except:
            return ''


def continue_simulation(worker_id, fname):
    if get_current_simulation(worker_id) == fname:
        return
    else:  # not simulating the right thing: overriding
        remove(worker_id, False)
        sleep(.5)  # wait worker to realize that
    with open(f'{worker_id}.temp', 'w') as file:
        file.write(fname)
    os.rename(f'{worker_id}.temp', worker_id)
    print(f'>>> starting simulating {fname} ({now()})')


def stop_simulation(worker_id, fname=None):
    if fname is not None:
        if get_current_simulation(worker_id) != fname:
            return False # nothing to do here
    remove(worker_id, False)
    sleep(.5)  # wait worker to realize that
    return True

###############################################################################
# Logit committor fit #########################################################
###############################################################################

def fit(network, pathensemble,
        keys=None,
        initial_paths=None,
        process_descriptors=lambda x: x,
        save_memory=False,
        nbins=0,
        state_bins='',
        augment=False,
        thA=0.1,
        thB=0.1,
        lr=1e-3,
        loss_bayesian_factor=100,
        loss_smoothening_weight=0,
        loss_regularization_weight=0,
        epochs=500,
        batch_size=4096,
        stop=50.,
        verbose=False,
        worker=None):
    
    """
    Train a neural network to predict the logit committor from AIMMD
    simulation data.
    
    This function fits the given `network` using path sampling results
    stored in `pathensemble`. It uses a log-binomial loss and replaces
    the network's weights with the fitted ones.
    
    If the input network is already partially trained and `nbins > 0`,
    the function uses it to regularize the training process. Then, calling
    this function multiple times during an AIMMD simulation campaign can
    help improve convergence.
    
    If `agument` is `True`, the training set includes all available data:
    shooting points, two-way shooting simulations, and free trajectories.
    This broadens coverage and improves learning. Otherwise, it just trains
    on the shooting points and their associated results.
    
    Attention! The network is modified in-place.
    
    Parameters
    ----------
    network : Network
        Neural network model to train.
    
    pathensemble : PathEnsemble or PathEnsemblesCollection
        Collection of sampled paths, including shooting and free simulations.
        Used to build the training set.
    
    keys : array-like of int, range, or slice, optional
        Use only the paths at these indices for training. Useful for
        bootstrapping or block averaging.
    
    initial_paths : PathEnsemble or PathEnsemblesCollection, optional
        Initial paths used to start the simulation. Used to add extra data if
        `pathensemble` has too little.
    
    process_descriptors : function, default is identity
        Transform pathensemble's descriptors into the neural network's input.
    
    save_memory : bool, default=False
        If `True`, compute descriptors on the fly for each training batch to
        reduce memory use. Useful when the training set doesn't fit in RAM.
    
    nbins : int, default=0
        Divide the reactive space in `nbins + 2` bins based on the input
        neural network model, and assign selection probabilities to the
        training set points such that each batch has a uniform population
        across the bins. Plus, regularize the shooting results in the bins.
        `nbins = 9` is usually a good value.
    
    state_bins : str, default=''
        Add two additional bins for the states defined in `state_bins`.
        For example, `state_bins = 'AB'` includes both in A and in B data
        in the training set. Assign selection probabilities to the training
        set points such that each batch has a uniform population across all
        the bins.
    
    augment : bool, default=False
        If `True`, include all available data in the training set: shooting
        points, two-way shooting simulations, and free trajectories. This
        broadens coverage and improves learning. Otherwise, it just trains
        on the shooting points and their associated results.
    
    thA : float, optional
        If `augment == True` and `thA > 0`, modify the training set to
        improve the learning accuracy when approaching the boundary of state
        A, by including fractional shooting results. Specifically, `thA` is the
        threshold for defining the boundary of state A. The function looks at
        all shooting values from paths reaching A (including free ones),
        sorts them, and uses the `thA` percentile as a cutoff. Paths with
        lower values are considered as if coming from state A for the
        additional shooting results. If there are enough free simulations,
        the actual boundary of state A will be considered. `thA = 0.1` is
        usually a good value.
    
    thB : float, optional
        Same idea as `thA`, but for state B. If `augment == True` and
        `thB > 0`, the function uses the `(1 - thB)` percentile as the cutoff.
        Paths with higher shooting values are considered as if coming from
        state B for the additional shooting results.

    lr : float, default=1e-3
        Learning rate. Will use an ADAM optimizer.
    
    loss_bayesian_factor : float, default=100.0
        In defining the logit committor square deviation training loss,
        necessary with discrete data.
    
    loss_smoothening_weight : float, default=0.0
        Weight of an additional term to the loss, corresponding to the
        network's gradient with respect to the input regularization factor.

    loss_regularization_weight : float, default=0.0
        Weight of an additional term to the loss, corresponding to the L1
        regularization factor.
        
    epochs : int, default=500
        Number of training epochs, each of which draws a new batch of
        training set data and minimizes the loss for one step. The fit will
        run up until 50% epochs more until the batch loss (removed the
        smoothening and regularization weights) is the lowest ever recorded.
    
    batch_size : int, default=4096
        Size of each training batch. Since the batch is drawn with selection
        selection probabilities, its size will always be `batch_size`, even
        with very small training sets.
    
    verbose : bool, default=False
        If True, be loud and noise. Among other things, show a progress bar
        during training.
    
    worker : aimmd.Worker
        Linked worker, to know when to interrupt.
    
    Returns
    -------
    losses : list of float
        Loss values from each training epoch.
    
    scales : list of float
        Maximum absolute network output (logit committor) at each epoch.
    
    values : ndarray of float
        Logit committor values for all training points.
    
    selection_probabilities : ndarray of float
        Probability of selecting each training point in a batch. If the
        network was already trained, these probabilities are adjusted to
        improve sampling across committor values.
    
    results : ndarray of shape (N, 2)
        Shooting results linked to each training point. Real numbers (not
        just 0 or 1) since the function includes and regularizes all data.
    
    TODO: include free/shot A-R and B-R paths.
    """
    
    if not augment:
        thA = None
        thB = None
    
    # utils' function
    def _concatenate(l, **kwargs):
        if not len(l):
            return np.concatenate([l], **kwargs)
        return np.concatenate(l, **kwargs)

    if nbins or thA is not None or thB is not None:  # needed in these cases
        print(f'Updating the pathensemble values ({now()})')
        pathensemble.update_values(only_reactive=True)  # with previous model
    
    # getting descriptors size
    if initial_paths is not None:
        descriptors_size = len(initial_paths.frame_descriptors[0])
    else:
        descriptors_size = len(pathensemble.frame_descriptors[0])
    
    t0 = time.time()
    losses, scales, selection_probabilities, results = [], [], [], []
    
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    if len(pathensemble):
        keys = np.arange(len(pathensemble))[keys].ravel()
        keys = keys[pathensemble[keys].are_accepted]
    else:
        keys = np.zeros(0, dtype=int)
    
    # extract info (within keys representation)
    initial_states = pathensemble.initial_states[keys]
    internal_states = pathensemble.internal_states[keys]
    final_states = pathensemble.final_states[keys]
    
    # remove "hopeless" paths
    if len(keys):
        keys = keys[internal_states != initial_states]
    initial_states = pathensemble.initial_states[keys]
    internal_states = pathensemble.internal_states[keys]
    final_states = pathensemble.final_states[keys]
    
    # keep extracting information
    internal_lengths = pathensemble.internal_lengths[keys]
    shooting_indices = pathensemble.shooting_indices[keys]
    shooting_states = pathensemble.shooting_states[keys]
    shooting_values = pathensemble.shooting_values[keys]
    if len(keys):
        shooting_values[shooting_states == 'A'] = -np.inf
        shooting_values[shooting_states == 'B'] = +np.inf
    
    if len(keys):
        # get indices of A paths (use initial_paths if not present)
        inA = keys[np.where(internal_states == 'A')[0]]
    else:
        inA = []
    if not len(inA):
        temp = initial_paths[:].unsplit()
        temp = temp.crop(frame_indices=temp.frame_states == 'A')
        temp.are_accepted[:] = True
        pathensemble += temp
        keys = np.append(keys, len(pathensemble) - 1)
        inA = np.array([len(keys) - 1])
    
    if len(keys):
        # get indices of in B paths (use initial_paths if not present)
        inB = keys[np.where(internal_states == 'B')[0]]
    else:
        inB = []
    if not len(inB):
        temp = initial_paths[:].unsplit()
        temp = temp.crop(frame_indices=temp.frame_states == 'B')
        temp.are_accepted[:] = True
        pathensemble += temp
        keys = np.append(keys, len(pathensemble) - 1)
        inB = np.array([len(keys) - 1])
    
    if len(keys):
        # get indices of shot paths
        shot_paths = np.where((internal_states == 'R') *
                              (shooting_indices > 0))[0]
    else:
        shot_paths = np.zeros(0, dtype=int)
    
    # get indices of ARA paths (within keys representation)
    if len(keys):
        AtoA = np.where((initial_states == 'A') *
                        (internal_states == 'R') * 
                        (final_states == 'A'))[0]
    else:
        AtoA = np.zeros(0, dtype=int)
    
    # get indices of BRB paths (within keys representation)
    if len(keys):
        BtoB = np.where((initial_states == 'B') *
                        (internal_states == 'R') * 
                        (final_states == 'B'))[0]
    else:
        BtoB = np.zeros(0, dtype=int)
    
    # determine effective state A boundary
    thA2 = -np.inf
    if thA is not None and len(AtoA):
        thA2 = +np.quantile(+shooting_values[AtoA], thA)
        if np.isnan(thA2):
            thA2 = -np.inf
        # report
        print(f'\n    thA {thA} associated value: {thA2:.3f}')
    
    # determine effective state B boundary
    thB2 = +np.inf
    if thB is not None and len(BtoB):
        thB2 = -np.quantile(-shooting_values[BtoB], thB)
        if np.isnan(thB2):
            thB2 = +np.inf
        # report
        print(f'    thB {thB} associated value: {thB2:.3f}\n')            

    if len(keys):
        # get indices of equilibrium fromA paths
        # (starting at the effective boundary of state A)
        free_A = np.where((initial_states == 'A') *
                          (internal_states == 'R') * 
                          (shooting_values <= thA2))[0]
        
        # get indices of equilibrium fromB paths
        # (starting or ending at the effective boundary of state B)
        free_B = np.where((initial_states == 'B') *
                          (internal_states == 'R') * 
                          (shooting_values >= thB2))[0]
    else:
        free_A = np.zeros(0, dtype=int)
        free_B = np.zeros(0, dtype=int)

    # which AtoA paths are free? (within keys representation)
    # which AtoA paths are shot? (within keys representation)
    if len(AtoA):
        free_AtoA = AtoA[shooting_values[AtoA] <= thA2]
        shot_AtoA = AtoA[shooting_values[AtoA] > thA2]
    else:
        free_AtoA = np.zeros(0, dtype=int)
        shot_AtoA = np.zeros(0, dtype=int)
    
    # which BtoB paths are free? (within keys representation)
    # which BtoB paths are shot? (within keys representation)
    if len(BtoB):
        free_BtoB = BtoB[shooting_values[BtoB] >= thB2]
        shot_BtoB = BtoB[shooting_values[BtoB] < thB2]
    else:
        free_BtoB = np.zeros(0, dtype=int)
        shot_BtoB = np.zeros(0, dtype=int)
    
    # shot AtoB and BtoA paths (now within shot paths representation)
    if len(shot_paths):
        shot_AtoB = ((initial_states[shot_paths] == 'A') *
                     (final_states[shot_paths] == 'B'))
        shot_BtoA = ((initial_states[shot_paths] == 'B') *
                     (final_states[shot_paths] == 'A'))
        
        # assign weights to shot TPs: 1 / density at shooting interface
        shot_paths_densities = pathensemble.densities(keys[shot_paths])
        shot_AtoB_densities = shot_paths_densities[shot_AtoB]
        shot_BtoA_densities = shot_paths_densities[shot_BtoA]
        if len(shot_AtoB_densities):
            shot_AtoB_densities[shot_AtoB_densities == 0.] = np.inf  # stay safe
        if len(shot_BtoA_densities):
            shot_BtoA_densities[shot_BtoA_densities == 0.] = np.inf  # stay safe
        shot_AtoB_weights = 1 / shot_AtoB_densities
        shot_BtoA_weights = 1 / shot_BtoA_densities
        
        # convert to within keys representation
        shot_AtoB = shot_paths[shot_AtoB]
        shot_BtoA = shot_paths[shot_BtoA]
    else:
        shot_AtoB = np.zeros(0, dtype=int)
        shot_BtoA = np.zeros(0, dtype=int)
        shot_AtoB_weights = np.zeros(0)
        shot_BtoA_weights = np.zeros(0)
    
    # join
    shot_TPs = np.append(shot_AtoB, shot_BtoA).astype(int)
    shot_TPs_weights = np.append(shot_AtoB_weights, shot_BtoA_weights)
    
    # equilibrium TPs (within keys representation)
    if len(freeA):
        free_AtoB = free_A[final_states[free_A] == 'B']
    else:
        free_AtoB = np.zeros(0, dtype=int)
    if len(freeB):
        free_BtoA = free_B[final_states[free_B] == 'A']
    else:
        free_BtoA = np.zeros(0, dtype=int)
    free_TPs = np.append(free_AtoB, free_BtoA).astype(int)
    
    # TPs weights (default for equilibrium: 1)
    TPs = np.append(shot_TPs, free_TPs).astype(int)
    if len(TPs):
        WTPs = np.ones(len(TPs))  # path-wise
        WTPs[:len(shot_TPs)] = shot_TPs_weights
        wTPs = np.repeat(WTPs, internal_lengths[TPs])
        # frame-wise
    else:
        wTPs = np.zeros(0)
    
    # determine transition paths' supplemental results
    free_A_values = _concatenate(
        pathensemble.values(keys[free_A], internal=True), axis=0)
    free_B_values = _concatenate(
        pathensemble.values(keys[free_B], internal=True), axis=0)
    total = np.sum(wTPs)
    if total and thA is not None:
        factor_fromA_toB = min(np.sum(expit(+free_A_values)) / total, 1)
        if verbose:
            print(f'Conversion factor from A to B: {factor_fromA_toB:.5e}')
    else:
        factor_fromA_toB = 0.
    
    if total and thB is not None:
        factor_fromB_toA = min(np.sum(expit(-free_B_values)) / total, 1)
        if verbose:
            print(f'Conversion factor from B to A: {factor_fromB_toA:.5e}')
    else:
        factor_fromB_toA = 0.
    
    # collect values, descriptors, and results
    
    ########
    # in A #
    ########
    
    inA_descriptors = _concatenate(
        pathensemble.descriptors(keys[inA], internal=True), axis=0)
    inA_values = np.repeat(-stop, len(inA_descriptors))
    inA_results = np.zeros((len(inA_values), 2))
    inA_results[:, 0] = 1. * ('A' in state_bins)
    
    ########
    # in B #
    ########
    
    inB_descriptors = _concatenate(
        pathensemble.descriptors(keys[inB], internal=True), axis=0)
    inB_values = np.repeat(-stop, len(inB_descriptors))
    inB_results = np.zeros((len(inB_values), 2))
    inB_results[:, 1] = 1. * ('B' in state_bins)
    
    ###############
    # shot A to A #
    ###############
    
    shot_AtoA_values = _concatenate(
        pathensemble.values(keys[shot_AtoA], internal=True), axis=0)
    shot_AtoA_descriptors = _concatenate(
        pathensemble.descriptors(keys[shot_AtoA], internal=True), axis=0
    ).reshape(-1, descriptors_size)
    
    # initialize results
    shot_AtoA_results = np.zeros((len(shot_AtoA_values), 2))
    
    # base results
    boundaries = np.cumsum(np.append([0], internal_lengths[shot_AtoA]))
    for si, begin, end in zip(
        shooting_indices[shot_AtoA], boundaries, boundaries[1:]):
        
        # backward
        if augment:
            segment = range(begin, begin + si + 1)
        else:  # only the shooting point
            segment = [begin + si]
        shot_AtoA_results[segment, 0] += 1.
        
        # forward
        if augment:
            segment = range(begin + si, end)
        shot_AtoA_results[segment, 0] += 1.
    
    # selection probabilities
    shot_AtoA_selection_probabilities = np.ones(len(shot_AtoA_values))
    
    ###############
    # shot B to B #
    ###############
    
    shot_BtoB_values = _concatenate(
        pathensemble.values(keys[shot_BtoB], internal=True), axis=0)
    shot_BtoB_descriptors = _concatenate(
        pathensemble.descriptors(keys[shot_BtoB], internal=True), axis=0
    ).reshape(-1, descriptors_size)
    
    # initialize results
    shot_BtoB_results = np.zeros((len(shot_BtoB_values), 2))
    
    # base results
    boundaries = np.cumsum(np.append([0], internal_lengths[shot_BtoB]))
    for si, begin, end in zip(
        shooting_indices[shot_BtoB], boundaries, boundaries[1:]):
        
        # backward
        if augment:
            segment = range(begin, begin + si + 1)
        else:  # only the shooting point
            segment = [begin + si]
        shot_BtoB_results[segment, 1] += 1.
        
        # forward
        if augment:
            segment = range(begin + si, end)
        shot_BtoB_results[segment, 1] += 1.
    
    ###############
    # free A to A #
    ###############
    
    free_AtoA_values = _concatenate(
        pathensemble.values(keys[free_AtoA], internal=True), axis=0)
    free_AtoA_descriptors = _concatenate(
        pathensemble.descriptors(keys[free_AtoA], internal=True), axis=0
    ).reshape(-1, descriptors_size)
    
    # initialize results
    free_AtoA_results = np.zeros((len(free_AtoA_values), 2))
    
    # results
    free_AtoA_results[:, 0] += 1. * augment
    
    ###############
    # free B to B #
    ###############
    
    free_BtoB_values = _concatenate(
        pathensemble.values(keys[free_BtoB], internal=True), axis=0)
    free_BtoB_descriptors = _concatenate(
        pathensemble.descriptors(keys[free_BtoB], internal=True), axis=0
    ).reshape(-1, descriptors_size)
    
    # initialize results
    free_BtoB_results = np.zeros((len(free_BtoB_values), 2))
    
    # results
    free_BtoB_results[:, 1] += 1. * augment
    
    ###############
    # shot A to B #
    ###############
    
    shot_AtoB_values = _concatenate(
        pathensemble.values(keys[shot_AtoB], internal=True), axis=0)
    shot_AtoB_descriptors = _concatenate(
        pathensemble.descriptors(keys[shot_AtoB], internal=True), axis=0
    ).reshape(-1, descriptors_size)
    
    # initialize results
    shot_AtoB_results = np.zeros((len(shot_AtoB_values), 2))
    
    # base results
    boundaries = np.cumsum(np.append([0], internal_lengths[shot_AtoB]))
    for si, begin, end in zip(
        shooting_indices[shot_AtoB], boundaries, boundaries[1:]):
        
        # backward
        if augment:
            segment = range(begin, begin + si + 1)
        else:  # only the shooting point
            segment = [begin + si]
        shot_AtoB_results[segment, 0] += 1.
        
        # forward
        if augment:
            segment = range(begin + si, end)
        shot_AtoB_results[segment, 1] += 1.
    
    ###############
    # shot B to A #
    ###############
    
    shot_BtoA_values = _concatenate(
        pathensemble.values(keys[shot_BtoA], internal=True), axis=0)
    shot_BtoA_descriptors = _concatenate(
        pathensemble.descriptors(keys[shot_BtoA], internal=True), axis=0
    ).reshape(-1, descriptors_size)
    
    # initialize results
    shot_BtoA_results = np.zeros((len(shot_BtoA_values), 2))
    
    # base results
    boundaries = np.cumsum(np.append([0], internal_lengths[shot_BtoA]))
    for si, begin, end in zip(
        shooting_indices[shot_BtoA], boundaries, boundaries[1:]):
        
        # backward
        if augment:
            segment = range(begin, begin + si + 1)
        else:  # only the shooting point
            segment = [begin + si]
        shot_BtoA_results[segment, 1] += 1.
        
        # forward
        if augment:
            segment = range(begin + si, end)
        shot_BtoA_results[segment, 0] += 1.
    
    ###############
    # free A to B #
    ###############
    
    free_AtoB_values = _concatenate(
        pathensemble.values(keys[free_AtoB], internal=True), axis=0)
    free_AtoB_descriptors = _concatenate(
        pathensemble.descriptors(keys[free_AtoB], internal=True), axis=0
    ).reshape(-1, descriptors_size)
    
    # initialize results
    free_AtoB_results = np.zeros((len(free_AtoB_values), 2))
    
    # results
    free_AtoB_results[:, 1] += 1. * augment
    
    ###############
    # free B to A #
    ###############
    
    free_BtoA_values = _concatenate(
        pathensemble.values(keys[free_BtoA], internal=True), axis=0)
    free_BtoA_descriptors = _concatenate(
        pathensemble.descriptors(keys[free_BtoA], internal=True), axis=0
    ).reshape(-1, descriptors_size)
    
    # initialize results
    free_BtoA_results = np.zeros((len(free_BtoA_values), 2))
    
    # results
    free_BtoA_results[:, 0] += 1. * augment
    
    ##########
    # Merge! #
    ##########
    
    values = np.concatenate([
        inA_values,
        inB_values,
        shot_AtoA_values,
        shot_BtoB_values,
        free_AtoA_values,
        free_BtoB_values,
        shot_AtoB_values,
        shot_BtoA_values,
        free_AtoB_values,
        free_BtoA_values], axis=0)
    descriptors = np.concatenate([
        inA_descriptors,
        inB_descriptors,
        shot_AtoA_descriptors,
        shot_BtoB_descriptors,
        free_AtoA_descriptors,
        free_BtoB_descriptors,
        shot_AtoB_descriptors,
        shot_BtoA_descriptors,
        free_AtoB_descriptors,
        free_BtoA_descriptors], axis=0)
    results = np.concatenate([
        inA_results,
        inB_results,
        shot_AtoA_results,
        shot_BtoB_results,
        free_AtoA_results,
        free_BtoB_results,
        shot_AtoB_results,
        shot_BtoA_results,
        free_AtoB_results,
        free_BtoA_results], axis=0)
    selection_probabilities = np.ones(len(values))
    
    print(f'\nCollected {len(inA_values):9} in A frames,\n'
          f'          {len(inB_values):9} in B frames,\n'
          f'          {len(shot_AtoA_values):9} shot A to A frames,\n'
          f'          {len(shot_BtoB_values):9} shot B to B frames,\n'
          f'          {len(free_AtoA_values):9} free A to A frames,\n'
          f'          {len(free_BtoB_values):9} free B to B frames,\n'
          f'          {len(shot_AtoB_values):9} shot A to B frames,\n'
          f'          {len(shot_BtoA_values):9} shot B to A frames,\n'
          f'          {len(free_AtoB_values):9} free A to B frames, and\n'
          f'          {len(free_BtoA_values):9} free B to A frames\n'
          f'   TOTAL: {len(values):9} frames')
    
    # useful
    n_internal_frames = len(inA_values) + len(inB_values)
    free_AtoA_frames_begin = n_internal_frames + len(
        shot_AtoA_values) + len(shot_BtoB_values)
    free_BtoB_frames_begin = free_AtoA_frames_begin + len(
        free_AtoA_values)
    shot_AtoB_frames_begin = free_BtoB_frames_begin + len(
        free_BtoB_values)
    shot_BtoA_frames_begin = shot_AtoB_frames_begin + len(
        shot_AtoB_values)
    free_AtoB_frames_begin = shot_BtoA_frames_begin + len(
        shot_BtoA_values)
    free_BtoA_frames_begin = free_AtoB_frames_begin + len(
        free_AtoB_values)
    
    # uniformize selection probabilities in bins (if nbins > 0)
    if nbins:
        bins = get_bins(pathensemble, nbins,
            cutoff_max=20.0, initial_paths=initial_paths, states=True)
    else:
        bins = np.array([-np.inf, +np.inf])

    # avoid frustration due to not both A, B being present in internal bins
    # swipe forward and look for results to B
    i = 3
    v = values[results[:, 1] > 0]
    while i < len(bins) - 1:
        if np.sum((bins[i - 1] <= v) * (v < bins[i])) < 3:
            bins = np.delete(bins, i - 1)
        else:
            i += 1

    # swipe backward and look for results to A
    i = len(bins) - 3
    v = values[results[:, 0] > 0]
    while i > 1:
        if np.sum((bins[i] <= v) * (v < bins[i + 1])) < 3:
            bins = np.delete(bins, i)
        i -= 1
    
    print(f'\nUniformizing selection probabilities\n'
          f'in bins: {array2string(bins, 20)}')
    
    # internal A and internal B
    mask = range(0, len(inA_values))
    norm = np.sum(selection_probabilities[mask])
    if 'A' in state_bins and norm:
        selection_probabilities[mask] /= norm
    else:
        selection_probabilities[mask] = 0.
    mask = range(len(inA_values), n_internal_frames)
    norm = np.sum(selection_probabilities[mask])
    if 'B' in state_bins and norm:
        selection_probabilities[mask] /= norm
    else:
        selection_probabilities[mask] = 0.
    
    # all the rest
    indices = np.digitize(values[n_internal_frames:], bins) - 1
    for i in range(len(bins) - 1):
        if verbose:
            print(f'    bin {i}: '
                  f'({expit(bins[i]):.3e}, {expit(bins[i+1]):.3e})')
        
        # bin "center"
        if not np.isinf(bins[i]) and not np.isinf(bins[i + 1]):
            q = (bins[i] + bins[i + 1]) / 2
        else:
            q = 0.
        if i == 0:
            q = bins[+1]
        elif i == len(bins) - 2:
            q = bins[-2]
        
        # get mask
        mask = n_internal_frames + np.where(indices == i)[0]
        
        # get info
        mask_TPs = mask[mask >= shot_AtoB_frames_begin]
        mask_free_AtoA = mask[(mask >= free_AtoA_frames_begin) *
                              (mask  < free_BtoB_frames_begin)]
        mask_free_BtoB = mask[(mask >= free_BtoB_frames_begin) *
                              (mask  < shot_AtoB_frames_begin)]
        
        # are additional results from A possible?
        if factor_fromA_toB and len(mask_free_AtoA) and \
           0 < len(mask_TPs) / len(mask_free_AtoA) < 100:
            # ratio = (np.sum(results[mask_free_AtoA]) /
            #          np.sum(results[mask])) / factor_fromA_toB
            ratio = 1 / max(factor_fromA_toB, expit(+q))
            results[mask_free_AtoA, 0] += ratio
            results[mask_TPs, 1] += (
                wTPs[mask_TPs - shot_AtoB_frames_begin] *
                ratio * factor_fromA_toB)
            if verbose:
                print(f'    ... {len(mask_free_AtoA):<9} '
                      f'free A to A frames get additional '
                      f'{ratio:.3e} result to A')
                print(f'    ... {len(mask_TPs):<9} '
                      f'transition frames get additional '
                      f'f{ratio * factor_fromA_toB:.3e} result to B')
        
        # are additional results from B possible?
        if factor_fromB_toA and len(mask_free_BtoB) and \
           0 < len(mask_TPs) / len(mask_free_BtoB) < 100:
            # ratio = (np.sum(results[mask_free_BtoB]) /
            #          np.sum(results[mask])) / factor_fromB_toA
            ratio = 1 / max(factor_fromA_toB, expit(-q))
            results[mask_free_BtoB, 1] += ratio
            results[mask_TPs, 0] += (
            wTPs[mask_TPs - shot_AtoB_frames_begin] *
            ratio * factor_fromB_toA)
            if verbose:
                print(f'    ... {len(mask_free_BtoB):<9} '
                      f'free B to B frames get additional '
                      f'{ratio:.3e} result to B')
                print(f'    ... {len(mask_TPs):<9} '
                      f'transition frames get additional '
                      f'f{ratio * factor_fromB_toA:.3e} result to A')
        
        # remove empty points
        selection_probabilities[mask[
            np.sum(results[mask], axis=1) == 0]] = 0.
        mask = mask[selection_probabilities[mask] > 0]
        
        # does it make sense to proceed?
        if not np.sum(selection_probabilities[mask]):
            continue
        
        # correct for imbalance
        results[mask] /= np.mean(np.sum(results[mask], axis=1))
        a = np.average(results[mask, 0],
                       weights=selection_probabilities[mask])
        b = np.average(results[mask, 1],
                       weights=selection_probabilities[mask])
        
        # let selection probability absorb imbalance from A and B
        maskA = mask[results[mask, 1] == 0]  # only A
        maskB = mask[results[mask, 0] == 0]  # only B
        if np.sum(selection_probabilities[maskA]):
            a2 = np.average(
                results[maskA, 0],
                weights=selection_probabilities[maskA])
        else:
            a2 = 1.
        if np.sum(selection_probabilities[maskB]):
            b2 = np.average(
                results[maskB, 1],
                weights=selection_probabilities[maskB])
        else:
            b2 = 1.
        selection_probabilities[maskA] *= results[maskA, 0] / a2
        selection_probabilities[maskB] *= results[maskB, 1] / b2
        results[maskA, 0] = a2
        results[maskB, 1] = b2
        
        # since results with both rA > 0, rB > 0 are rare and
        # mostly uniform already, there is no need to rescale them
        
        # final normalization
        selection_probabilities[mask] /= np.sum(
            selection_probabilities[mask])
        
        if not verbose:
            continue
        
        print(f'    ... {len(mask):<9} frames')
        print(f'    ... {a:.3e} average result to A')
        print(f'    ... {b:.3e} average result to B')
    
    # keep only the training set frames
    keepers = selection_probabilities > 0
    if not np.sum(keepers):  # no other choice
        selection_probabilities += 1.
        keepers = np.ones(len(selection_probabilities), dtype=bool)
    selection_probabilities = selection_probabilities[keepers]
    values = values[keepers]
    descriptors = descriptors[keepers]
    results = results[keepers]
    print(f'\nTraining set size {len(selection_probabilities)}')
    
    selection_probabilities /= np.sum(selection_probabilities)
    
    if not save_memory:  # all together now
        descriptors = process_descriptors(descriptors)
    
    """
    Training loop.
    """
    
    # initialization
    # backups for restoration
    state_dict0 = copy.deepcopy(network.state_dict())  # fixed
    state_dict1 = copy.deepcopy(network.state_dict())  # linked to min_loss1
    state_dict2 = copy.deepcopy(network.state_dict())  # linked to min_loss2
    min_loss1 = np.inf
    min_loss2 = np.inf
    min_loss_step1 = 0
    min_loss_step2 = 0
    
    print(f'Reseting the network parameters ({now()})\n')
    network.reset_parameters()
    
    losses = []
    scales = []
    # D = []
    # R = []
    i = 0
    if verbose:
        counter = tqdm(range(epochs), position=0, file=sys.stdout)
    
    # actual loop
    while True:
        
        if worker is not None and worker.interrupt:
            return [], [], [], [], []
        
        for param_group in optimizer.param_groups:
            # slowly increase lr
            param_group['lr'] = lr * min(1, (i + 1) / (epochs / 20))
        
        # sample batch
        indices = np.random.choice(len(selection_probabilities),
                                   batch_size, p=selection_probabilities)
        if save_memory:  # separately to save memory
            d = process_descriptors(descriptors[indices])
        else:
            d = descriptors[indices]
        d = torch.tensor(d, dtype=dtype, device=device)
        d.requires_grad = True
        r = torch.tensor(results[indices], dtype=dtype, device=device)
        
        # define loss function
        def closure():
            
            optimizer.zero_grad()
            q = network(d)
            
            qA = - (torch.log(1 + torch.exp(-q[:, 0])) +
                    loss_bayesian_factor)
            qB = + (torch.log(1 + torch.exp(+q[:, 0])) +
                    loss_bayesian_factor)
            
            toA_contribution = (q[:, 0] - qA) ** 2
            toB_contribution = (q[:, 0] - qB) ** 2
            
            q = q[:, 0].detach()
            
            loss = torch.sum(q ** 2 *
                (r[:, 0] * toA_contribution +
                 r[:, 1] * toB_contribution))
            
            # normalize
            loss /= torch.sum(q ** 2 * (r[:, 0] + r[:, 1]) *
                              loss_bayesian_factor ** 2)
            loss -= 1.0
            
            # Compute the smoothness penalty
            if loss_smoothening_weight:
                q_grad = torch.autograd.grad(
                    outputs=q.sum(), inputs=d, create_graph=True)[0]
                smoothness_loss = (torch.abs(q_grad) ** 2).mean()
                loss += loss_smoothening_weight * smoothness_loss
            
            # Calculate L1 regularization
            if loss_regularization_weight:
                l1_norm = sum(p.abs().sum() for p in network.parameters())
                
                # Combine original loss with L1 regularization term
                loss += loss_regularization_weight * l1_norm
            loss.backward()
            return loss
        
        # update network
        network.train()
        loss = optimizer.step(closure)
        losses.append(float(loss.detach()))
        
        # report scales
        q = network(d).detach()
        scales.append(max(float(torch.max(q)), -float(torch.min(q))))
        Range = float(torch.min(q)), float(torch.max(q))
        
        if verbose:
            counter.update(1)
        
        # handle termination: too high scales
        if scales[-1] >= stop or np.isnan(scales[-1]):
            print(f'!!! stopping early since scale '
                  f'{scales[-1]:.3f} > {stop:.3f}')
            if (i + 1) < 1.25 * epochs:
                print(f'    restoring lowest loss\' ({min_loss1:.3e}) '
                      f'weights, step {min_loss_step1 + 1}')
                network.load_state_dict(state_dict1)
            else:
                print(f'    restoring lowest loss\' ({min_loss2:.3e}) '
                      f'weights, step {min_loss_step2 + 1}')
                network.load_state_dict(state_dict2)
            break
        
        # save model if goood
        if losses[-1] <= min_loss1:
            min_loss1 = losses[-1]
            min_loss_step1 = i
            state_dict1 = copy.deepcopy(network.state_dict())
        
        # new min loss after reaching target epochs
        if (i + 1) >= epochs and losses[-1] <= min_loss2:
            min_loss2 = losses[-1]
            min_loss_step2 = i
            state_dict2 = copy.deepcopy(network.state_dict())
        
        # handle termination: lowest loss
        if (i + 1) >= epochs and losses[-1] <= min_loss1:
            break
        
        # new target after 1.25 * epochs
        if (i + 1) >= 1.25 * epochs and losses[-1] <= min_loss2:
            break
        
        # at most 1.5 * epochs
        if (i + 1) >= 1.5 * epochs:
            print(f'    restoring lowest loss\' ({min_loss2:.3e}) '
                  f'weights, step {min_loss_step2 + 1}')
            network.load_state_dict(state_dict2)
            break
        
        i += 1
        
        # D.append(d)
        # R.append(r)
        
        # report
        if verbose and i % (epochs // 20) == 0:
            print(f'    loss {losses[-1]:.3e}, '
                  f'scale {scales[-1]:.3f}, '
                  f'range ({Range[0]:.3f}, {Range[1]:.3f})')
    
    if verbose:
        counter.close()
    
    # error handling: result not as expected
    if Range[0] > 0 or Range[1] < 0 or scales[-1] < 1:
        print(f'!!! bad range ({Range[0]:.3f}, {Range[1]:.3f}), '
              f'restoring original parameters')
        network.load_state_dict(state_dict0)
    
    # recompute scales and range in case they changed
    q = network(d).detach()
    scales[-1] = max(float(torch.max(q)), -float(torch.min(q)))
    Range = float(torch.min(q)), float(torch.max(q))
    
    # report and return
    print(f'\nTraining took {time.time()-t0:.1f}s')
    print(f'    {i + 1} epochs')
    print(f'    last loss {losses[-1]:.3e}')
    print(f'    last scale {scales[-1]:.3f}')
    print(f'    last range ({Range[0]:.3f}, {Range[1]:.3f})')
    return losses, scales, values, selection_probabilities, results#, D, R


###############################################################################
# AIMMD run utils #############################################################
###############################################################################

def load_network_and_projections(
    network, directory, backup_directory=None, wait=True, worker=None):
    device = next(network.parameters()).device
    
    # advance only if data are present
    while True:
        try:  
            state_dict = torch.load(
                f'{directory}/network.h5', map_location=device)
            bins = np.load(f'{directory}/bins.npy')
            densities = np.load(f'{directory}/densities.npy')
            break
        except:
            if not wait or (worker is not None and worker.interrupt):
                return [], []
            sleep(.1)
    
    # backup
    if backup_directory is not None:  # at time of shooting init
        torch.save(state_dict, f'{backup_directory}/network.h5')
        np.save(f'{backup_directory}/bins.npy', bins)
        np.save(f'{backup_directory}/densities.npy', densities)
    
    # load network params and return
    network.load_state_dict(state_dict)
    return bins, densities


def update_shooting_simulation(
    backward, forward, worker_id,
    params, batch_size=100, verbose=True):
    
    trajectory_extension = params.trajectory_extension
    max_excursion_length = params.max_excursion_length
    grompp = params.grompp
    
    def _stop_condition(segment, base=0):
        n_frames = 0
        states = segment.frame_states
        mask = states != 'R'
        if np.sum(mask):
            n_frames = np.where(mask)[0][0] + 1
        elif segment.nframes + base >= max_excursion_length:
            n_frames = max(1, max_excursion_length - base)
        return states[:n_frames]
    
    # backward part
    # add last simulated frames in batches,
    # stop if reached any state or max length
    states = _stop_condition(backward)
    n_frames_back = len(states)
    if not n_frames_back:
        added_frames = 1  # flag
    else:
        added_frames = 0
    report = False
    while not n_frames_back and added_frames:
        if grompp:
            try:
                os.path.getsize(f'{backward.directory}/back.tpr')
                tpr_present = True
            except:
                tpr_present = False
            try:
                os.path.getsize(f'{backward.directory}/'
                                f'back{trajectory_extension}')
                trj_present = True
            except:
                trj_present = False
            try:
                os.path.getsize(f'{backward.directory}/back.cpt')
                cpt_present = True
            except:
                cpt_present = False
            if not tpr_present or (cpt_present and not trj_present):
                print(f'!!! {backward.directory}/back missing; resetting!')
                backward.reset() # reset to pristine state
                added_frames = 0
                stop_simulation(worker_id, f'{backward.directory}/back')
                remove(f'{backward.directory}/back.cpt')
                remove(f'{backward.directory}/back{trajectory_extension}')
                return [], [], []
        try:
            added_frames = backward.append(
                f'back{trajectory_extension}',
                start=backward.nframes,
                stop=backward.nframes + batch_size)[0]
        except:
            print(f'!!! error updating {backward.directory}/back; resetting!')
            backward.reset() # reset to pristine state
            added_frames = 0
            stop_simulation(worker_id, f'{backward.directory}/back')
            remove(f'{backward.directory}/back.cpt')
            continue_simulation(worker_id, f'{backward.directory}/back')
        
        if added_frames:
            report = True
        
        # stop, report only if it was running
        states = _stop_condition(backward)
        n_frames_back = len(states)
        if n_frames_back:
            if stop_simulation(worker_id, f'{backward.directory}/back'):
                print(f'xxx stopping {backward.directory}/'
                      f'back{trajectory_extension} in state {states[-1]} '
                      f'after {n_frames_back} frames ({now()})')
            break

        # continue simulation
        continue_simulation(worker_id, f'{backward.directory}/back')
    if report:
        print(f'... {backward.directory}/'
              f'back{trajectory_extension} '
              f'hit {backward.nframes} frames ({now()})')
    
    # forward part
    # add last simulated frames in batches,
    # stop if reached any state or max length
    states = _stop_condition(forward, n_frames_back)
    n_frames_forw = len(states)
    if not n_frames_forw:
        added_frames = 1  # flag
    else:
        added_frames = 0
    report = False
    while not n_frames_forw and added_frames:
        if grompp:
            try:
                os.path.getsize(f'{forward.directory}/forw.tpr')
                tpr_present = True
            except:
                tpr_present = False
            try:
                os.path.getsize(f'{forward.directory}/'
                                f'forw{trajectory_extension}')
                trj_present = True
            except:
                trj_present = False
            try:
                os.path.getsize(f'{forward.directory}/forw.cpt')
                cpt_present = True
            except:
                cpt_present = False
            if not tpr_present or (cpt_present and not trj_present):
                print(f'!!! {forward.directory}/forw missing; resetting!')
                forward.reset() # reset to pristine state
                added_frames = 0
                stop_simulation(worker_id, f'{forward.directory}/forw')
                remove(f'{forward.directory}/forw.cpt')
                remove(f'{forward.directory}/forw{trajectory_extension}')
                return [], [], []
        try:
            added_frames = forward.append(f'forw{trajectory_extension}',
                start=forward.nframes,
                stop=forward.nframes + batch_size)[0]
        except:
            print(f'!!! error updating {forward.directory}/forw; resetting!')
            forward.reset() # reset to pristine state
            added_frames = 0
            stop_simulation(worker_id, f'{forward.directory}/forw')
            remove(f'{forward.directory}/forw.cpt')
            continue_simulation(worker_id, f'{forward.directory}/forw')
        
        if added_frames:
            report = True
        
        # stop, report only if it was running
        states = _stop_condition(forward, n_frames_forw)
        n_frames_forw = len(states)
        if n_frames_forw:
            if report:
                print(f'... {forward.directory}/'
                      f'forw{trajectory_extension} '
                      f'hit {forward.nframes} frames ({now()})')
            
            if stop_simulation(worker_id, f'{forward.directory}/forw'):
                print(f'xxx stopping {forward.directory}/'
                      f'forw{trajectory_extension} in state {states[-1]} '
                      f'after {n_frames_forw} frames ({now()})')
            
            # compose and return full path
            frames_back = range(n_frames_back - 1, 0, -1)
            frames_forw = range(n_frames_forw)
            path = backward.frames(frames_back) + forward.frames(frames_forw)
            states = np.append(
                backward.frame_states[frames_back],
                forward.frame_states[frames_forw], axis=0)
            descriptors = np.append(
                backward.frame_descriptors[frames_back],
                forward.frame_descriptors[frames_forw], axis=0)
            return path, states, descriptors
        
        # continue simulation
        elif get_current_simulation(worker_id) != f'{backward.directory}/back':
            continue_simulation(worker_id, f'{forward.directory}/forw')
    if report:
        print(f'... {forward.directory}/'
              f'forw{trajectory_extension} '
              f'hit {forward.nframes} frames ({now()})')
    return [], [], []  # no path: empty

###############################################################################
# Manager utils ###############################################################
###############################################################################

def initialize_simulation(frames, params, *fnames):
    """
    Fnames without extension.
    Part only if frames has length > 1.
    If len(fnames) > 1, one part is forward, the other backword.
    Always randomize velocities if trajectory extension is xtc.
    """
    
    # get params
    topology = params.topology
    gmx_init_mdp = params.gmx_init_mdp
    gmx_run_mdp = params.gmx_run_mdp
    random_velocities = params.randomize_shooting_velocities
    grompp = params.grompp
    mdrun = params.mdrun
    trajectory_extension = params.trajectory_extension
    
    if trajectory_extension == '.xtc':
        random_velocities = True
    
    # process directories and fname
    directories = ['/'.join(fname.split('/')[:-1])
                   for fname in fnames]
    directories = [directory if len(directory) else '.'
                   for directory in directories]
    fnames = [fname.split('/')[-1] for fname in fnames]
    
    # remove outputs to avoid conflict
    for directory, fname in zip(directories, fnames):
        for file in os.listdir(directory):
            if not os.path.isfile(f'{directory}/{file}'):
                continue
            if (file[:len(fname)] == fname and 
                len(file) > (len(fname) + 1) and
                file[len(fname)] in '._'):
                remove(f'{directory}/{file}')
            if (file[1:len(fname) + 1] == f'.{fname}' and 
                len(file) > (len(fname) + 2) and
                 file[len(fname) + 1] == '.'):
                remove(f'{directory}/{file}', False)
    
    # invert velocities in case they are there
    nframes = len(frames)
    invert_velocities = nframes > 1 and frames[0].time > frames[1].time
    
    # temp file template
    _fname = f'{directories[0]}/.{fnames[0]}'
    
    # randomize velocities
    if grompp:
        if random_velocities:
            print('=== randomize velocities')
        else:
            print('=== sampling kinetic energy')
        remove(f'{_fname}.gro', False)
        frames.write(f'{_fname}.gro', frame_indices=[-1])
        command = (f'{grompp} -nobackup -f {gmx_init_mdp} '
                  f'-r {_fname}.gro -c {_fname}.gro -o {_fname}.tpr')
        os.system(command)
        os.system(f'{mdrun} -deffnm {_fname} -nsteps 0 -nobackup')
        
    # just copy the frame
    else:
        remove(f'{_fname}.trr', False)
        frames.write(f'{_fname}.trr', frame_indices=[-1],
                     invert_velocities=invert_velocities)

    # get results in the universe
    universe = mda.Universe(topology, f'{_fname}.trr')
    universe.transfer_to_memory()
    atomgroup = universe.atoms
    frame = universe.trajectory[-1]
    remove(f'{_fname}.trr', False)
    
    # report kinetic energy
    kinetic_factor = np.sum(frame._velocities ** 2)
    print(f'*** kinetic factor: {kinetic_factor:.3e}')
    
    # just copy the frame and rescale energy
    if not grompp or not random_velocities:
        frames.write(f'{_fname}.trr', frame_indices=[-1], 
                     invert_velocities=invert_velocities)
        
        universe = mda.Universe(topology, f'{_fname}.trr')
        universe.transfer_to_memory()
        atomgroup = universe.atoms
        frame = universe.trajectory[-1]
        remove(f'{_fname}.trr', False)

        if grompp:
            kinetic_factor0 = np.sum(frame._velocities ** 2)
            rescaling = min(10., (kinetic_factor / kinetic_factor0) ** .5)
            print(f'*** shooting point kinetic factor: {kinetic_factor0:.3e}')
            print(f'=== rescaling velocities by {rescaling:.3f}')
            frame._velocities *= rescaling
            
    # iterate through files
    invert_velocities = False
    for fname, directory in zip(fnames, directories):
        
        # invert velocities alternatively
        if invert_velocities:
            frame._velocities *= -1
            invert_velocities = False
        else:
            invert_velocities = True
        
        # generate tpr starting from the last frame
        if grompp:
            with mda.Writer(f'{_fname}.trr', atomgroup.n_atoms) as writer:
                writer.write(atomgroup)
            
            command = (f'{grompp} -nobackup -f {gmx_run_mdp} '
                      f'-r {_fname}.gro -c {_fname}.gro -t {_fname}.trr '
                      f'-o {directory}/{fname}.tpr')
            os.system(command)
            
            # run initial frame (does it work without??)
            # os.system(f'{mdrun} -nobackup -deffnm {directory}/{fname} '
            #          f'-cpo {directory}/{fname}.cpt -nsteps 0')
        else:
            # just copy the frame
            if nframes <= 1:
                filename = f'{directory}/{fname}{trajectory_extension}'
            else:
                filename = f'{directory}/{fname}.part0001{trajectory_extension}'
            with mda.Writer(filename, atomgroup.n_atoms) as writer:
                writer.write(atomgroup)
        
        # previous frames as part 0
        if nframes > 1:
            dt = np.abs(frames[1].time - frames[0].time)
            atomgroups = [universe.atoms for universe in frames.universes]
            print(f'=== saving {directory}/'
                  f'{fname}.part0000{trajectory_extension}')
            with mda.Writer(
                f'{directory}/{fname}.part0000{trajectory_extension}',
                atomgroups[0].n_atoms) as writer:
                for i in range(nframes - 1):
                    atomgroup = atomgroups[frames.frame_trajectory_indices[i]]
                    frame = frames[i]
                    if invert_velocities:
                        frame._velocities *= -1
                    frame.time = -dt * (nframes - 1 - i)
                    writer.write(atomgroup)


def load_initial_paths(directory, topology, states_function,
                      descriptors_function, values_function,
                      verbose=True):
    fnames = sorted([fname for fname in os.listdir(directory)
                     if '.xtc' == fname[-4:] or '.trr' == fname[-4:]])
    initial_paths = PathEnsemble()
    for fname in fnames:
        temp = PathEnsemble()
        temp.directory = directory
        temp.topology = os.path.relpath(topology, directory)
        temp.states_function = states_function
        temp.descriptors_function = descriptors_function
        temp.values_function = values_function
        temp.append(fname, verbose=verbose)
        temp.split()
        try:
            temp = temp[np.where(temp.are_transitions)[0][0]]
        except:
            print(f'!!! no transitions in {fname}')
            raise
        initial_paths = initial_paths.merge(temp)
    return initial_paths


def update_shooting_chain(
    chain,  # PathEnsemble object
    chain_id,  # chain worker id or directly str
    directory,  # main directory of the run
    topology,  # relative to the destination
    states_function,  # function that gives the states
    descriptors_function,  # function that gives the descriptors
    values_function,  # function that gives the values
    load_h5=False,  # if load from saved h5 file (or backup)
    add_missing_paths=True):
    """
    Also returns added_nframes
    """
    added_nframes = 0
    
    # process directory
    if type(chain_id) is int:
        chain_id = f'shots{chain_id}'
    if directory is not None and directory != '.' and directory != './':
        directory = os.path.join(directory, chain_id)
    else:
        directory = chain_id
    
    # load
    if load_h5:
        try:
            chain.load(f'{directory}/chain.h5')
        except:
            try:
                chain.load(f'{directory}/chain_backup.h5')
                print(f'!!! shots{directory}/chain.h5 '
                      f'corrupted, reloaded backup')
            except:
                pass
    
    # attributes
    chain.directory = directory
    chain.topology = os.path.relpath(topology, directory)
    chain.states_function = states_function
    chain.descriptors_function = descriptors_function
    chain.values_function = values_function
    
    # add missing paths
    if add_missing_paths:
        for path in sorted([fname for fname in os.listdir(directory)
            if fname[:4] == 'path' and fname[4:10].isdigit()
            and fname[10:14] in ['.xtc', '.trr']]):
            if path not in chain.trajectory_files:
                nframes, _ = chain.add_path(path, selection_bias=1., weight=1.)
                if not nframes:
                    print(f'!!! no frames in {directory}/{path}')
                    raise
                print(f'+++ added {path} with {nframes} frames '
                      f'to chain in {directory}')
                added_nframes += nframes
    
    return added_nframes


def update_selection_pool(
    pool,  # PathEnsemble object
    chain,  # chain of reference, will add the last if pool_index is not None
    selection_pool_size,  # target size
    pool_index=None,  # index of pool to be removed
    initial_paths=PathEnsemble(),  # will there be?
    at_least_one_transition=False,  # in pool
    load_h5=False):
    """
    Will inherit all pathensemble attributes from chain.
    """
    
    # (re)assign attributes
    pool.directory = chain.directory
    pool.topology = chain.topology
    pool.states_function = chain.states_function
    pool.descriptors_function = chain.descriptors_function
    pool.values_function = chain.values_function
    
    def update_initial_path_directory(initial_paths):
        _initial_paths = initial_paths.copy()
        _initial_paths.directory = pool.directory
        _initial_paths.topology = os.path.relpath(f'{initial_paths.directory}/{initial_paths.topology}', pool.directory)
        _initial_paths._PathEnsemble__trajectory_files = [os.path.relpath(
            f'{initial_paths.directory}/{file}', pool.directory)
            for file in _initial_paths.trajectory_files]
        return _initial_paths
    
    if load_h5 and os.path.exists(f'{chain.directory}/pool.h5'):
        # it must not be corrupted or have weird paths
        pool.load(f'{chain.directory}/pool.h5')
    
    # add missing paths
    if not len(pool):
        if len(chain):
            pool = chain[-selection_pool_size:]
        else:
            pool = update_initial_path_directory(initial_paths)
            pool = pool[np.random.permutation(range(len(pool)))]
    
    # update with last element in the chain?
    if pool_index is not None:
        pool = pool.merge(chain[-1])
        if len(pool) > selection_pool_size:
            keepers = np.ones(len(pool), dtype=bool)
            keepers[pool_index] = False
            pool = pool[keepers]
    
    # restrict to size
    pool = pool[-selection_pool_size:]
    
    if at_least_one_transition and not np.sum(
        pool.are_transitions) and len(chain):
        candidates = np.where(chain.are_transitions * chain.are_accepted)[0]
        if len(candidates):  # try adding the latest transition in chain
            transition = chain[candidates[-1]]
        else:
            initial_paths = update_initial_path_directory(initial_paths)
            transition = initial_paths[np.random.choice(len(initial_paths))]
        print(f'+++ (re)added transition {transition.trajectory_files[0]}')
        pool = transition.merge(pool)
    
    # if pool too small: replicate up to selection_pool_size // 2
    if len(pool) < selection_pool_size // 2:
        pool = pool[(list(range(len(pool))) *
            selection_pool_size)[:selection_pool_size // 2]]
    
    for fname in pool.shooting_trajectory_filenames:
        print(f'    {pool.directory}/{fname}')
    
    return pool


def update_equilibrium_trajectory(
    trajectory,  # PathEnsemble object with already set attributes
    load_h5=False,  # if load from saved h5 file (or backup)
    save_h5=False,  # if update is present
    add_missing_frames=True,
    verbose=True):  # in trajectory file
    """
    Also returns added_nframes
    """
    
    # attributes
    directory = trajectory.directory
    topology = trajectory.topology
    fname = trajectory.trajectory_files[0][:10]
    trajectory_extension = trajectory.trajectory_files[0][-4:]
    
    # initial status
    nframes = trajectory.nframes
    trajectory.unsplit()
    
    # load
    if load_h5:
        try:
            trajectory.load(f'{directory}/{fname}.h5')
        except:
            try:
                trajectory.load(f'{directory}/{fname}_backup.h5')
                print(f'!!! {directory}/{fname}.h5 corrupted, reloaded backup')
            except:
                pass
        nframes = trajectory.nframes
        if verbose and nframes:
            print(f'=== loaded {directory}/{fname} with {nframes} frames')
    
    # retrieve current part and remove temporary files
    part = int(trajectory.trajectory_files[-1][15:19])
    for file in os.listdir(directory):
        if file[:11] == f'.{fname}':
            remove(f'{directory}/{file}', False)
    
    # add latest frames and re-split
    report = False
    if add_missing_frames:
        skip_tolerance = 0
        while True:
            nf = trajectory.append(
                f'{fname}.part{part:04g}{trajectory_extension}')[0]
            if nf:
                report = True
            else:  # do not break at first
                skip_tolerance += 1
            if skip_tolerance == 5:
                break  # at most a jump of 5 parts is tolerated
            part += 1
    
    trajectory.split()

    # report
    if report:
        last_states = ''
        last_state = None
        for state in trajectory.frame_states[-10:]:
            if last_state is not None and last_state != state:
                last_states += '|'
            last_states += state
            last_state = state
        n_file = np.sum(trajectory.frame_trajectory_indices == 
                        trajectory.frame_trajectory_indices[-1])
        print(f'... {trajectory.directory}/'
              f'{trajectory.trajectory_files[-1]} '
              f'hit {trajectory.nframes} frames ({n_file} in file), '
              f'last states {last_states}, '
              f'last path lengths '
              f'{" ".join([str(x) for x in trajectory.lengths[-3:]])} '
              f'({now()})')
    
    # save only if added more frames
    if save_h5 and trajectory.nframes > nframes:
        trajectory.save(f'{directory}/{fname}.h5', directory='.')
    
    return trajectory.nframes - nframes


def update_equilibrium_simulations(
    eq_current, eq_completed,
    directory, nA, nB,
    initial_paths,
    params,
    eA=0, eB=0,
    ext_current=[],
    available_transitions=PathEnsemble(),  
    save_h5=False,
    simulate=False,
    verbose=False):
    
    # retrieve params
    topology = params.topology
    states_function = params.states_function
    descriptors_function = params.descriptors_function
    values_function = params.values_function
    trajectory_extension = params.trajectory_extension
    max_excursion_length = params.max_excursion_length
    extra_extend_frames = params.extra_extend_frames
    
    if not len(eq_completed):
        if not nA:
            eqA, nf = update_pathensemble(directory, topology, states_function,
                descriptors_function, values_function, trajectory_extension,
                add_missing_frames=True, shooting_chains=[],
                equilibriumA=['equilibriumA'], equilibriumB=[], 
                verbose=verbose)
            if save_h5:
                for trajectory, added_nframes in zip(eqA.pathensembles, nf):
                    if not added_nframes:
                        continue
                    trajectory.save(f'{trajectory.directory}/'
                                    f'{trajectory.trajectory_files[0][:10]}.h5',
                                    directory='.')
            eq_completed.extend(list(eqA.pathensembles))
        
        if not nB:
            eqB, nf = update_pathensemble(directory, topology, states_function,
                descriptors_function, values_function, trajectory_extension,
                add_missing_frames=True, shooting_chains=[],
                equilibriumA=[], equilibriumB=['equilibriumB'],
                verbose=verbose)
            if save_h5:
                for trajectory, added_nframes in zip(eqB.pathensembles, nf):
                    if not added_nframes:
                        continue
                    trajectory.save(f'{trajectory.directory}/'
                                    f'{trajectory.trajectory_files[0][:10]}.h5',
                                    directory='.')
            eq_completed.extend(list(eqB.pathensembles))

    t0 = np.array([-np.inf, -np.inf])  # start of the latest ext. trajectory
    for j, folder in enumerate(['extendA', 'extendB']):
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if not (filename[0] != '.' and
                    'part0000' in filename and
                    filename[-4:] == trajectory_extension):
                continue
            if os.path.isfile(filepath):
                tA = max(tA, os.path.getmtime(filepath))
            for trajectory in ext_current:
                if not trajectory.nframes:
                    continue
            t0[j] = max(t0[j], trajectory.frame_simulation_times[0])
    
    # iterate through all the workers and identify the states
    for h in range(nA + nB + eA + eB):
        worker_id = f'{directory}/worker{h}.run'
        
        # populate eq_current and get state info
        if h < nA + nB:
            k = h
            extend = False
            if len(eq_current) > k:
                trajectory = eq_current[k]
            else:
                trajectory = PathEnsemble()
                eq_current.append(trajectory)
            if k < nA:
                state = 'A'
            else:
                state = 'B'
            trajectory_directory = f'{directory}/equilibrium{state}'
        else:  # populate ext_current and get state info
            k = h - nA - nB
            extend = True
            if len(ext_current) > k:
                trajectory = ext_current[k]
            else:
                trajectory = PathEnsemble()
                ext_current.append(trajectory)
            if k < eA:
                state = 'A'
            else:
                state = 'B'
            trajectory_directory = f'{directory}/extend{state}'
        
        if verbose:
            if not extend:
                i = k - nA * (k >= nA)
            else:
                i = k - eA * (k >= eA)
            print(f'\n*** worker {h}: '
                  f'{state}{i}{" (extension)" if extend else ""}')
        
        # create trajectory object if necessary
        if len(trajectory.trajectory_files):
            eq_id = int(trajectory.trajectory_files[0][4:10])
        else:
            trajectory.directory = trajectory_directory
            trajectory.topology = os.path.relpath(
                topology, trajectory.directory)
            trajectory.states_function = states_function
            trajectory.descriptors_function = descriptors_function
            trajectory.values_function = values_function
            if not extend:
                eq_id = k - nA * (k >= nA) + 1
            else:
                eq_id = k - eA * (k >= eA) + 1
            trajectory._PathEnsemble__trajectory_files = [
                f'traj{eq_id:06g}.part0000{trajectory_extension}']
        
        init_frames = []
        
        # iterate through all the completed and upcoming simulations
        while True:
            
            # do you need to initialize a new simulation?
            if trajectory.trajectory_files[0] not in \
                os.listdir(trajectory.directory):
                """
                No necessity to check for tpr files in case of gromacs
                simulation cause part0000.xtc/trr will be present in
                any case! :)
                """
                
                # force extend one of previous transitions
                if extend:
                    init_frames = []
                
                # get initial frame if not having them
                if len(init_frames) < 2:
                    if not extend:
                        transition = initial_paths[
                            np.random.choice(len(initial_paths))]
                    elif len(available_transitions):
                        T = available_transitions.completion_times
                        i = np.argmax(T)
                        t1 = T[i]  # use only transitions produced after
                        if t1 > t0[0 if state == 'A' else 1]:
                            # beginning of previous extension simulation
                            transition = available_transitions[i]
                        else:  # wait a suitable available transition
                            if verbose:
                                print(f'=== no transition available yet')
                            break
                    else:  # wait a suitable available transition
                        if verbose:
                            print(f'=== no transition available yet')
                        break
                    if transition.final_states[0] == state:
                        if not extend:
                            init_frames = transition.frames([-2, -1])
                        else:
                            init_frames = transition.frames()
                            t0 = time.time()  # of current extension
                    if transition.initial_states[0] == state:
                        if not extend:
                            init_frames = transition.frames([+1, +0])
                        else:
                            init_frames = transition.frames(
                                range(transition.nframes - 1, -1, -1))
                            t0 = time.time()  # of current extension
                
                # report what you used (useful for reconstructing history)
                index = init_frames.frame_trajectory_indices[0]
                position = init_frames.frame_trajectory_positions[0]
                frame0 = f'({init_frames.filenames[index]}, {position})'
                index = init_frames.frame_trajectory_indices[-1]
                position = init_frames.frame_trajectory_positions[-1]
                frame1 = f'({init_frames.filenames[index]}, {position})'
                print(f'=== using {frame0} -> {frame1} '
                      f'for initializing {trajectory.directory}/'
                      f'{trajectory.trajectory_files[0][:10]}')
                
                # initialize and start simulation
                initialize_simulation(
                    init_frames,  # also removes garbage
                    params,
                    f'{trajectory.directory}/' +
                    f'{trajectory.trajectory_files[0][:10]}')
            
            # update trajectory
            nframes = update_equilibrium_trajectory(trajectory,
                load_h5=trajectory.nframes == 0, save_h5=save_h5)
            init_frames = check_equilibrium_stop_condition(
                trajectory, state, max_excursion_length,
                extra_extend_frames if extend else 0)
            
            # trajectory is completed (len(init_frames) > 0): go to the next
            if len(init_frames):
                stop_simulation(f'{directory}/worker{h}.run')
                print(f'=== {trajectory.directory}/'
                      f'{trajectory.trajectory_files[0][:10]} completed')
                if save_h5:
                    trajectory.save(f'{trajectory.directory}/'
                                    f'{trajectory.trajectory_files[0][:10]}.h5',
                                    directory='.')
                if not extend:
                    eq_completed.append(trajectory)
                trajectory = PathEnsemble()
                trajectory.directory = trajectory_directory
                trajectory.topology = os.path.relpath(
                    topology, trajectory.directory)
                trajectory.states_function = states_function
                trajectory.descriptors_function = descriptors_function
                trajectory.values_function = values_function
                if not extend and state == 'A':
                    eq_id += nA
                if not extend and state == 'B':
                    eq_id += nB
                if extend and state == 'A':
                    eq_id += eA
                if extend and state == 'B':
                    eq_id += eB
                trajectory._PathEnsemble__trajectory_files = [
                    f'traj{eq_id:06g}.part0000{trajectory_extension}']
                if not extend:
                    eq_current[k] = trajectory
                else:
                    ext_current[k] = trajectory
            else:
                break  # can go inspect the next trajectory
        
        # start or continue current simulation
        fname = f'{trajectory.directory}/{trajectory.trajectory_files[0][:10]}'
        if simulate and trajectory.nframes:
            continue_simulation(worker_id, fname)


def check_equilibrium_stop_condition(
    trajectory, state, max_excursion_length, extra_frames=0):
    
    # evaluate stop condition
    states = trajectory.frame_states
    target_states = [state, 'R']
    
    # stop condition: not accepted (too long)
    trajectory.split(max_excursion_length)
    if len(k := np.where(trajectory.lengths > max_excursion_length)[0]):
        index = trajectory.frame_indices(k[0])[0][0]
        if states[index] == state:  # restart from last crossing
            initial_frames = trajectory.frames([index + 1, index])
        else:
            initial_frames = trajectory.frames([index, index + 1])
        print(f'xxx stopping {trajectory.directory}/'
              f'{trajectory.trajectory_files[0][:10]} '
              f'after {trajectory.nframes} frames '
              f'because the last segment is too long')
        trajectory.are_accepted[k[0]:] = False  # no selection/reweighting
        return initial_frames
    
    # stop condition: going to other state (transition)
    """
    You always restart with R -> <target state>.
    """
    condition = np.sum([states == state for state in target_states], axis=0)
    condition = condition.astype(bool)
    if trajectory.nframes:
        condition[0] = True  # first frame does not count
    max_index = np.where(~condition)[0]
    if len(max_index):
        max_index = max_index[0]
        if max_index < len(states) - extra_frames:
            print(f'xxx stopping {trajectory.directory}/'
                  f'{trajectory.trajectory_files[0][:10]} '
                  f'because it reached state {states[max_index]} '
                  f'after {max_index + 1} frames')
            # no selection/reweighting
            trajectory.are_accepted[
                trajectory.initial_states == states[max_index]] = False
            trajectory.are_accepted[
                trajectory.internal_states == states[max_index]] = False
            """
            Remove in case of multi-stage transition where state
            C is not included in state B to allow path transfer.
            """
            trajectory.are_accepted[  # the only allowed outcomes
                (trajectory.final_states != 'A') *
                (trajectory.final_states != 'B') *
                (trajectory.final_states != 'R')] = False
            
            index = np.where(states == 'R')[0]
            index = index[index < max_index]
            if not len(index):
                print(f'!!! trajectory was never in state R '
                      f'before state {states[max_index]}')
                return [0]  # flag for using eq path for initializing
            max_index = index[-1]
            index = np.where(states == state)[0]
            index = index[index < max_index]
            if not len(index):
                print(f'!!! trajectory was never in state {state} '
                      f'before state {states[max_index]}')
                return [0]  # flag for using eq path for initializing
            index = index[-1]
            initial_frames = trajectory.frames([index + 1, index])
            # last time you exited the target state before reaching
            # the "unallowed" state, reversed
            return initial_frames
    
    if 'indicted_trajectories.log' in os.listdir(trajectory.directory):
        with open(f'{trajectory.directory}/indicted_trajectories.log') as file:
            # get indicted info: trajectory name and first indicted frame
            for line in file:
                info = line.split()
                if len(info) >= 1:
                    filename = info[0]
                else:
                    filename = ''
                if len(info) >= 2:
                    nframes = int(info[1])
                else:
                    nframes = 0
                
                # indict and do not accept beyond "nframes" for overriding
                if trajectory.nframes and (
                    filename[:10] == trajectory.trajectory_files[0][:10]):
                    print(f'xxx stopping {trajectory.directory}/{filename} '
                          f'because it has been indicted '
                          f'after {nframes} frames')
                    try:
                        # individual frame index
                        i = np.where(trajectory.
                            _PathEnsemble__frame_indices >= nframes)[0][0]
                        # first path to be indicted
                        j = np.where(np.cumsum(trajectory.lengths) >= i)[0][0]
                        print(f'xxx indicting {len(trajectory) - j} paths')
                        trajectory.are_accepted[j:] = False
                    except:
                        pass
                    return trajectory.frames([0, 1])
    
    return trajectory.frames([])


def update_pathensemble(
    directory,
    topology='run.gro',
    states_function=None,
    descriptors_function=None,
    values_function=lambda descriptors: np.repeat(0., len(descriptors)),
    trajectory_extension='.xtc',
    add_missing_paths=True,
    add_missing_frames=False,
    shooting_chains=['shots'],  # will look for progressive number from zero!
    equilibriumA=['equilibriumA'],  # directories where to search
    # must contain "equilibriumA" to be correctly recognized
    equilibriumB=['equilibriumB'],
    # must contain "equilibriumB" to be correctly recognized
    equilibriumA_states_map=[''],
    equilibriumB_states_map=[''],
    verbose=True):
    """
    States map e.g. "AB BC": changes A to B and B to C.
    "": do not change anything
    """
    
    added_nframes = []
    
    def _report(pathensemble, file=False):
        dt = convert_seconds(time.time() -
            pathensemble.completion_times[-1] if pathensemble.nframes else 0)
        print(f'*** {pathensemble.directory}{"/" if file else ""}'
              f'{pathensemble.trajectory_files[0][:10] if file else ""}: '
              f'{pathensemble}, last updated '
              f'{dt} ago')

    def _check_indicted(trajectory):
        if 'indicted_trajectories.log' in os.listdir(trajectory.directory):
            with open(f'{trajectory.directory}/'
                      f'indicted_trajectories.log') as file:
                # get indicted info: trajectory name and first indicted frame
                for line in file:
                    info = line.split()
                    if len(info) >= 1:
                        filename = info[0]
                    else:
                        filename = ''
                    if len(info) >= 2:
                        nframes = int(info[1])
                    else:
                        nframes = 0
                    
                    # indict and do not accept beyond "nframes" for overriding
                    if trajectory.nframes and (
                        filename[:10] == trajectory.trajectory_files[0][:10]):
                        if verbose:
                            print(f'xxx {trajectory.directory}/{filename} '
                                  f'has been indicted after {nframes} frames')
                        try:
                            # individual frame index
                            i = np.where(trajectory.
                                _PathEnsemble__frame_indices >= nframes)[0][0]
                            # first path to be indicted
                            j = np.where(np.cumsum(trajectory.lengths) >= i)[0][0]
                            if verbose:
                                print(f'xxx indicting {len(trajectory) - j} '
                                      f'paths')
                            trajectory.are_accepted[j:] = False
                        except:
                            pass

    def _load_equilibrium(directory, filename):
        trajectory = PathEnsemble()
        trajectory.directory = directory
        trajectory.topology = os.path.relpath(topology, directory)
        trajectory.states_function = states_function
        trajectory.descriptors_function = descriptors_function
        trajectory.values_function = values_function
        trajectory._PathEnsemble__trajectory_files = [
            f'{filename}.part0000{trajectory_extension}']
        nframes = update_equilibrium_trajectory(trajectory,
            add_missing_frames=add_missing_frames, load_h5=True,
            verbose=False)
        if len(states_map):
            states = trajectory.frame_states.copy()
            for state1, state2 in states_map.split():
                trajectory.frame_states[states == state1] = state2
        _check_indicted(trajectory)
        return trajectory, nframes
    
    # shooting chains
    chains = []
    for expression in shooting_chains:
        n = 0
        while os.path.exists(f'{directory}/{expression}{n}'):
            chain = PathEnsemble()
            nframes = update_shooting_chain(
                chain, f'{expression}{n}', directory, topology,
                states_function, descriptors_function, values_function,
                add_missing_paths=add_missing_paths, load_h5=True)
            chains.append(chain)
            added_nframes.append(nframes)
            n += 1
            if verbose:
                _report(chain)    
    
    # equilibrium A
    _equilibriumA = []
    equilibriumA_states_map = np.tile(
        equilibriumA_states_map, len(equilibriumA))
    for expression, states_map in zip(equilibriumA, equilibriumA_states_map):
        if not os.path.exists(f'{directory}/{expression}'):
            continue
        for filename in sorted(os.listdir(f'{directory}/{expression}')):
            if len(filename) != 13:
                continue
            if filename[:4] != 'traj':
                continue
            if filename[-3:] != '.h5':
                continue
            if not filename[4:10].isdigit():
                continue
            trajectory, nframes = _load_equilibrium(
                f'{directory}/{expression}', filename[:-3])
            added_nframes.append(nframes)
            _equilibriumA.append(trajectory)
            if verbose:
                _report(trajectory, True)
    
    # equilibrium B
    _equilibriumB = []
    equilibriumB_states_map = np.tile(
        equilibriumB_states_map, len(equilibriumB))
    for expression, states_map in zip(equilibriumB, equilibriumB_states_map):
        if not os.path.exists(f'{directory}/{expression}'):
            continue
        for filename in sorted(os.listdir(f'{directory}/{expression}')):
            if len(filename) != 13:
                continue
            if filename[:4] != 'traj':
                continue
            if filename[-3:] != '.h5':
                continue
            if not filename[4:10].isdigit():
                continue
            trajectory, nframes = _load_equilibrium(
                f'{directory}/{expression}', filename)
            added_nframes.append(nframes)
            _equilibriumB.append(trajectory)
            if verbose:
                _report(trajectory, True)
    
    return PathEnsemblesCollection(
        *chains, *_equilibriumA, *_equilibriumB), added_nframes


def extract_pathensembles(pathensemble, expression):
    return PathEnsemblesCollection(*[pathensemble for pathensemble in
        pathensemble.pathensembles if
        expression in pathensemble.directory.split('/')[-1]])


def scorporate_pathensembles(pathensemble):
    shots = extract_pathensembles(pathensemble, 'shots')
    equilibriumA = extract_pathensembles(pathensemble, 'equilibriumA')
    equilibriumB = extract_pathensembles(pathensemble, 'equilibriumB')
    return shots, equilibriumA, equilibriumB


def equilibrium_trajectory_names(equilibrium):
    if type(equilibrium) is PathEnsemblesCollection:
        trajectory_names = [
            trajectory.directory + '/' +
            '.'.join(trajectory.trajectory_files[0].split('.')[:-2])
            for trajectory in equilibrium.pathensembles]
    else:
        trajectory_names = ['.'.join(equilibrium.split('.')[:-2])]
    return [
        trajectory_name if trajectory_name[:2] != './'
        else trajectory_name[2:] for trajectory_name
        in trajectory_names]


def run_acceptance_rejection_on_latest_path(chain, network):
    # in case of TPS
    
    # load params at the time of SP selection
    bins, densities = load_network_and_projections(network, chain.directory)
    
    def compute_sp_bias(values, sp_value, bins, densities):
        densities = np.append(densities, [np.inf])
        values = chain.values(leading, internal=True)[0]
        biases = 1 / densities[np.digitize(values, bins) - 1]
        sp_bias = 1 / densities[np.digitize(sp_value, bins) - 1]
        sp_bias /= np.sum(biases)
        return sp_bias
    
    # ensure weight is zero
    chain.weights[-1] = 0.
    
    # get index of leading path
    leading = None
    if np.sum(chain.weights):  # leading trajectory exists
        leading = np.where(chain.weights)[0][-1]
    
    # compute acceptance probability; easy job
    if not chain.are_accepted[-1]:
        print(f'=== acceptance probability: {0:.3f} (anomaly)')
        print(f'*** rejected')
        if leading is not None:
            chain.weights[leading] += 1.
        return
    
    if not chain.are_transitions[-1]:
        print(f'=== acceptance probability: {0:.3f} (not a transition)')
        print(f'*** rejected')
        if leading is not None:
            chain.weights[leading] += 1.
        return
    
    # now a real transition
    if leading is None:
        print(f'=== acceptance probability: {np.inf:.3f} (first transition)')
        print(f'*** accepted')
        chain.weights[-1] += 1.
        return
    
    # compute acceptance probability; now for real
    keepers = [leading, -1]
    
    # get values
    chain.update_values(network, key=keepers)
    leading_values, trial_values = chain.values(keepers, internal=True)
    leading_sp_value, trial_sp_value = chain.shooting_values[keepers]
    
    # run acceptance/rejection
    leading_sp_bias = compute_sp_bias(
        leading_values, leading_sp_value, bins, densities)
    trial_sp_bias = compute_sp_bias(
        trial_values, trial_sp_value, bins, densities)
    acceptance = trial_sp_bias / leading_sp_bias
    print(f'=== acceptance probability: {acceptance:.3f} (transition)')
    if np.random.random() < acceptance:
        print(f'*** accepted')
        chain.weights[-1] += 1.
    else:
        print(f'*** rejected')
        chain.weights[leading] += 1.


def get_bins(pathensemble, nbins=10,
             cutoff_min=.5, cutoff_max=20.,
             initial_paths=None, states=False):
    """
    nbins without the additional states.
    Two additional bins when `states = True`.
    """
    
    # special case
    if not len(pathensemble) and initial_paths is None:
        if states:
            return np.array([-np.inf, +np.inf])
        return np.zeros(0)
    
    # extraction
    try:
        shots, equilibriumA, equilibriumB = scorporate_pathensembles(pathensemble)
    except:
        shots = pathensemble
        equilibriumA = pathensemble[:0]
        equilibriumB = pathensemble[:0]
    if not equilibriumA.nframes and not np.sum(pathensemble.are_transitions):
        equilibriumA = initial_paths.crop(
            frame_indices=initial_paths.frame_states =='A')
    if not equilibriumB.nframes and not np.sum(pathensemble.are_transitions):
        equilibriumB = initial_paths.crop(
            frame_indices=initial_paths.frame_states =='B')
    equilibrium = equilibriumA + equilibriumB
    pathensemble = shots + equilibrium
    
    # limit
    limit = int(np.sum(shots.are_transitions) + 1)
    
    # initialize to range
    begin = -cutoff_max
    end = +cutoff_max
    
    # (inverse) eq. crossing probability histogram from A and from B
    try:
        eA = np.sort(
            equilibrium.max_values(equilibrium.initial_states=='A'))[::-1]
        if not len(eA):
            raise
    except:
        eA = np.array([-np.inf])    
    try:
        eB = np.sort(
            equilibrium.min_values(equilibrium.initial_states=='B'))
        if not len(eB):
            raise
    except:
        eB = np.array([+np.inf])
    
    # if not crossing prob. data: just min and max value
    if eA[0] == -np.inf and pathensemble.nframes:
        try:
            eA = np.array([np.min(pathensemble.frame_values[
                           pathensemble.frame_states == 'R'])])
        except:
            eA = np.array([np.min(pathensemble.frame_values)])
    if eB[0] == +np.inf and pathensemble.nframes:
        try:
            eB = np.array([np.max(pathensemble.frame_values[
                           pathensemble.frame_states == 'R'])])
        except:
            eB = np.array([np.max(pathensemble.frame_values)])
    
    # assign
    begin = np.clip(eA[min(limit, len(eA) - 1)], begin, -cutoff_min)
    end = np.clip(eB[min(limit, len(eB) - 1)], +cutoff_min, end)
    
    # further correct (avoid empty bins)
    if eA[-1] > begin:
        begin = eA[-1]
    if eB[-1] < end:
        end = eB[-1]

    if begin < end:
        bins = np.linspace(begin, end, nbins + 1)
    else:
        bins = [(begin + end) / 2]
    
    if states:  # min and max become infinity
        bins = np.concatenate([[-np.inf], bins, [+np.inf]])
    
    return bins


def extract_frame(trajectory, position, topology):
    while True:
        try:
            shooting_point = MDATrajectory([mda.Universe(
                topology, trajectory)], [0], [position])
        except Exception as exception:
            print(f'!!! {exception}')
            sleep(.1)
            raise
        if len(shooting_point):
            break
        sleep(.1)
    return shooting_point


def add_path_to_chain(path, chain,
    states=None, descriptors=None,
    trajectory_extension='.xtc', eneconv=None):
    """
    TODO do only if path and last path in chain differ
    Otherwise there is a minor chance of path duplication in case
    an error happened right after having called this function.
    However, this chance is so minor that one may as well get along
    with it.
    """
    
    # save path
    fname = f'path{len(chain) + 1:06g}'
    filename = f'{chain.directory}/{fname}{trajectory_extension}'
    path.write(filename, joined_shooting_segments=True)
    
    # save energy
    if eneconv:
        os.system(f'{eneconv} -f {chain.directory}/back.edr '
                              f' {chain.directory}/forw.edr '
                            f'-o {chain.directory}/{fname}.edr')
    
    # add path to chain
    try:
        selection_bias = np.load(f'{chain.directory}/shoot_bias.npy')
    except:
        selection_bias = 1.
    nframes, _ = chain.add_path(f'{fname}{trajectory_extension}',
                   selection_bias=selection_bias, weight=1.,
                   states=states, descriptors=descriptors)
    if not nframes:
        raise
    
    # report: extract info
    initial_state = chain.initial_states[-1]
    internal_state = chain.internal_states[-1]
    final_state = chain.final_states[-1]
    path_length = chain.lengths[-1]
    shooting_value = chain.shooting_values[-1]
    if initial_state == 'A':
        if final_state == 'B':
            extreme = +np.inf
        else:
            extreme = np.max(chain.values(-1)[0])
    elif initial_state == 'B':
        if final_state == 'A':
            extreme = -np.inf
        else:
            extreme = np.min(chain.values(-1)[0])
    elif final_state == 'A':
        extreme = np.max(chain.values(-1)[0])
    elif final_state == 'B':
        extreme = np.min(chain.values(-1)[0])
    else:
        extreme = np.nan
    
    # report: write info
    print(f'\nObtained {filename}: '
          f'{initial_state}{internal_state}{final_state} '
          f'of {path_length} frames ({now()})')
    print(f'*** shooting point\'s estimated value: '
          f'{shooting_value:.3f}')
    print(f'*** path\'s extreme value: {extreme:.3f}')
    
    # is the path legit?
    '''
    More complex situation where state C is not included in B (see
    p116 case
    if (((chain.initial_states[-1] not in ['A', 'B']) and  # not states
        (chain.final_states[-1] not in ['A', 'B'])) or
        ((chain.initial_states[-1] == 'A') and  # skipped B (A -> C)
        (chain.final_states[-1] == 'C')) or
        ((chain.final_states[-1] == 'A') and  #  skipped B (C -> A)
        (chain.initial_states[-1] == 'C')) or
        ((chain.final_states[-1] == 'B') and  #  skipped A (B -> Z)
        (chain.initial_states[-1] == 'Z')) or
        ((chain.initial_states[-1] == 'B') and  #  skipped A (Z -> B)
        (chain.final_states[-1] == 'Z')) or
        (chain.initial_states[-1] == 'R') or  # not full excursion
        (chain.final_states[-1] == 'R')):  # not full excursion
    '''
    if ((chain.initial_states[-1] not in ['A', 'B']) or
        (chain.final_states[-1]   not in ['A', 'B']) or
        (chain.initial_states[-1] == 'R') or
        (chain.final_states[-1] == 'R') or
        (chain.internal_states[-1] != 'R')):
        print(f'!!! NOT inserting path into pool')
        chain.weights[-1] = 0.


def initialize_shooting_simulation(
    chain, pool, directory, params,
    shooting_chains=None, equilibrium=PathEnsemblesCollection()):
    
    # report info
    i = len(chain)
    descriptors_function = chain.descriptors_function
    values_function = chain.values_function
    old_fname = f'path{i:06g} -> ' if i else ''
    new_fname = f'path{i+1:06g}'
    relpath = os.path.relpath(chain.directory, directory)
    print(f'\nShooting for chain {relpath}: '
          f'{old_fname}{new_fname}  ({now()})')

    # get params
    network = params.network
    topology = params.topology
    lorentzian = params.lorentzian
    #selection_pool_size = params.selection_pool_size
    adjust_selection_in_marginal_bins = params.adjust_selection_in_marginal_bins
    equilibrium_overriding_rate = params.equilibrium_overriding_rate
    equilibrium_overriding_recovery_rate = params.equilibrium_overriding_recovery_rate
    randomize_shooting_velocities = params.randomize_shooting_velocities
    
    # load most updated params, backup in chain's directory
    bins, densities = load_network_and_projections(
        network, directory, chain.directory)
    
    # populations
    completed_shooting_descriptors = []
    current_shooting_descriptors = []
    if shooting_chains is not None:
        completed_shooting_descriptors = shooting_chains.shooting_descriptors
        for _chain in shooting_chains.pathensembles:
            _directory = _chain.directory
            _topology = _chain.topology
            for trajectory_extension in ['.xtc', '.trr']:
                fname = f'{_directory}/back{trajectory_extension}'
                if os.path.exists(fname):
                    try:
                        current_shooting_descriptors.append(
                            descriptors_function([mda.Universe(
                                f'{_directory}/{_topology}',
                                fname).trajectory[0]])[0])
                    except Exception as e:
                        print(f'!!! {e}')
                        pass
    
    descriptors = []
    if (len(completed_shooting_descriptors) and
        len(current_shooting_descriptors)):
        descriptors = np.append(completed_shooting_descriptors,
                                current_shooting_descriptors, axis=0)
    elif len(completed_shooting_descriptors):
        descriptors = completed_shooting_descriptors
    elif len(current_shooting_descriptors):
        descriptors = np.array(current_shooting_descriptors)
    
    if len(descriptors):
        populations = np.histogram(values_function(descriptors), bins)[0] + .1
    else:
        populations = np.zeros(len(bins) - 1) + .1
    
    # report bins, densities, and populations
    print(f'    bins                  {array2string(bins, 25)}')
    print(f'    densities             {array2string(densities, 25)}')
    print(f'    populations           {array2string(populations, 25)}')
    
    # lorentzian correction
    A = (bins[:-1] + bins[1:]) / 2
    if lorentzian < np.inf:
        populations *= 1 / (A ** 2 + lorentzian ** 2)
        print(f'=== Lorentzian correction {array2string(populations, 25)}')
    
    # bias by densities and populations
    correction = 1 / np.concatenate(
        [[np.inf], densities * populations, [np.inf]])
    
    # select shooting point from paths in pool
    print(f'\nShooting point selection from paths in pool {pool.directory}')
    for fname in pool.shooting_trajectory_filenames:
        _fname = os.path.relpath(f'{pool.directory}/{fname}', directory)
        print(f'    {_fname}')
    
    # update values & display preliminary statistics
    pool.update_values()
    values = pool.values(internal=True)
    states = pool.states(internal=True)
    print(f'*** current pool shooting interfaces '
          f'{array2string(pool.shooting_values, 36)}')
    histograms = [np.histogram(values, bins)[0] for values in values]
    histograms = np.array(histograms)
    histogram = np.sum(histograms, axis=0)
    print(f'*** {np.sum(histogram)} candidate points in selection pool')
    print(f'    histogram: {array2string(histogram, 14)}')
    for pool_index in range(len(pool)):
        initial_state = pool.initial_states[pool_index]
        internal_state = pool.internal_states[pool_index]
        final_state = pool.final_states[pool_index]
        print(f'        path {pool_index:02g} '
              f'({initial_state}{internal_state}{final_state}) : '
              f'{array2string(histograms[pool_index], 22)}')
    print(f'    coverage : {array2string(np.sum(histograms > 0, axis=0), 14)}')
    
    # merge empty marginal bins
    if np.isinf(bins[0]) and histogram[0] == 0 and histogram[-1] == 0:
        print(f'\n!!! merging empty marginal bins')
        begin, end = np.where(histogram)[0][[0, -1]]
        bins = np.concatenate([[bins[0]], bins[begin + 1:end + 1], [bins[-1]]])
        densities = np.concatenate([[np.sum(densities[:begin + 1])],
                                     densities[begin + 1:end],
                                    [np.sum(densities[end:])]])
        populations = np.concatenate([[np.sum(populations[:begin + 1])],
                                     populations[begin + 1:end],
                                    [np.sum(populations[end:])]])
        print(f'    bins                  {array2string(bins, 25)}')
        print(f'    densities             {array2string(densities, 25)}')
        print(f'    populations           {array2string(populations, 25)}')
        correction = np.concatenate(
            [[0.], 1 / (densities * populations), [0.]])
        histograms = [np.histogram(v, bins)[0] for v in values]
        histograms = np.array(histograms)
        histogram = np.sum(histograms, axis=0)
        print(f'*** {np.sum(histogram)} candidate points in selection pool')
        print(f'    histogram: {array2string(histogram, 14)}')
        for pool_index in range(len(pool)):
            initial_state = pool.initial_states[pool_index]
            internal_state = pool.internal_states[pool_index]
            final_state = pool.final_states[pool_index]
            print(f'        path {pool_index:02g} '
                  f'({initial_state}{internal_state}{final_state}) : '
                  f'{array2string(histograms[pool_index], 22)}')
        print(f'    coverage : '
              f'{array2string(np.sum(histograms > 0, axis=0), 14)}')

    # put it all together
    weights = []
    histograms = []
    for v in values:
        w = correction[np.digitize(v, bins)]
        norm = np.sum(w)
        if not norm:  # recover from unpleasant situation
            #   (happening only if bins[0] > -inf, bins[1] < +inf)
            w[np.argmin(np.abs(v))] = 1.
            norm = 1.
        w /= norm
        weights.append(w)
        histograms.append(np.histogram(v, bins, weights=w)[0])
    histograms = np.array(histograms) / len(histograms)
    histogram = np.sum(histograms, axis=0)
    
    # adjust selection
    if adjust_selection_in_marginal_bins:
        print(f'=== density correction by '
              f'{array2string(correction[1:-1], 25)}')
        print(f'*** bin selection probability: {array2string(histogram, 30, formatter={"float_kind":lambda x: "%.3f" % x})}')
        print(f'\nAdjusting selection in bins')
        for i in [0, len(bins) - 2]:
            expected = 1 / populations[i] / np.sum(1 / populations)
            rescale = max(1., (histogram[i] / expected))
            print(f'    bin {i}: expected {expected:.3e}, '
                  f'actual {histogram[i]:.3e}, '
                  f'rescale by {rescale:.3e}')
            correction[i + 1] /= rescale
        # in the future, you may as well resample the paths, e.g. removing
        # short excursions in excess
        
        """ THIS IS SOMETHING FOR THE FUTURE
        correction = np.sum(histograms, axis=0).astype(float)
        correction /= np.sum(correction)
        correction = 1 / correction
        #correction *= (np.ones(len(bins) - 1) * 10) ** (
        #    -np.sum(histograms > 0, axis=0) *
        #    round(100 / selection_pool_size))  # safe with floating point error
        correction *= 1 / populations
        correction = np.concatenate([[0.], correction, [0.]])
        """
        weights = []
        histograms = []
        for v in values:
            w = correction[np.digitize(v, bins)]
            norm = np.sum(w)
            if not norm:  # recover from unpleasant situation
                #   (happening only if bins[0] > -inf, bins[1] < +inf)
                w[np.argmin(np.abs(v))] = 1.
                norm = 1.
            w /= norm
            weights.append(w)
            histograms.append(np.histogram(v, bins, weights=w)[0])
        histograms = np.array(histograms) / len(histograms)
        histogram = np.sum(histograms, axis=0)
    
    # report
    print(f'=== density correction by '
              f'{array2string(correction[1:-1], 25)}')
    print(f'*** bin selection probability: {array2string(histogram, 30, formatter={"float_kind":lambda x: "%.3f" % x})}')
    for pool_index in range(len(pool)):
        initial_state = pool.initial_states[pool_index]
        internal_state = pool.internal_states[pool_index]
        final_state = pool.final_states[pool_index]
        print(f'        path {pool_index:02g} '
              f'({initial_state}{internal_state}{final_state}) : '
                  f'{array2string(histograms[pool_index], 22, formatter={"float_kind":lambda x: "%.3f" % x})}')
    
    values = np.concatenate(values)
    states = np.concatenate(states)
    weights = np.concatenate(weights)
    positions = np.concatenate(pool.trajectory_positions(internal=True))
    files = np.concatenate(pool.trajectory_filenames(internal=True))
    indices = np.repeat(np.arange(len(pool)), pool.internal_lengths)  # path id
    
    # report selection pool situation
    pool_numbers = np.array([int(file.split('/')[-1].split('.')[0][4:])
        if 'initial' not in file else 0
        for file in pool.shooting_trajectory_filenames])
    with open(f'{chain.directory}/pool.log', 'a+') as file:
        file.write(f'{len(chain)} ')
        for number in pool_numbers:
            file.write(f'{number} ')
        file.write('\n')
    
    # final weights
    weights /= np.sum(weights)
    
    # select point
    i = np.random.choice(len(values), p=weights)
    fname = files[i]
    position = positions[i]
    index = indices[i]
    value = values[i]
    k = np.digitize(value, bins) - 1
    selection_bias = correction[k + 1]
    _fname = os.path.relpath(fname, directory)
    print(f'=== selecting shooting point {_fname}, {position} '
          f'(value: {value:.2f}, bin: {k})')
    if not selection_bias:
        selection_bias = np.inf
        print(f'    pool position: {index}')
    else:
        print(f'    pool position: {index}, selection bias: {selection_bias}')
    
    # get equilibrium candidates
    if not equilibrium.nframes:
        candidates = np.zeros(0, dtype=int)
    else:
        try:
            candidates = np.concatenate(equilibrium.frame_indices(
                equilibrium.are_accepted *
                (equilibrium.internal_states == 'R'), 
                internal=True))
        except:
            candidates = []
    
    # there are candidates for overriding
    if len(candidates):
        print(f'\nAttempting overriding from {len(candidates)} '
              f'candidate equilibrium configurations')
        rate = equilibrium_overriding_rate + 0.
        if (np.digitize(pool.shooting_values[index], bins) - 1) == k and rate:
            if np.random.random() > equilibrium_overriding_recovery_rate:
                rate = 0.
                print(f'=== skipped overriding because the SP of path '
                      f'{index} in pool has the same bin {k} (recovery rate '
                      f'= {equilibrium_overriding_recovery_rate})')
            else:  # on a very rare occasion: still override
                print(f'=== rescued overriding with a recovery rate '
                      f'of {equilibrium_overriding_recovery_rate}')
        
        while np.random.random() < rate:
            rate -= 1.
            i = np.random.choice(candidates)
            value = values_function(
                equilibrium.frame_descriptors[i:i+1])[0]
            eq_fname = np.array(equilibrium.trajectory_files)[
                equilibrium.frame_trajectory_indices[i]]
            eq_position = equilibrium.frame_trajectory_positions[i]
            
            h = np.digitize(value, bins) - 1 # which bin?
            print(f'=== picking {eq_fname}, {eq_position}')
            print(f'    (value {value:.2f}, bin {h})')
            if h == k: # success
                print(f'*** accepted\n')
                fname = eq_fname
                position = eq_position
                selection_bias = 1.
                break
            else:
                print(f'*** rejected\n')
        else:
            print(f'*** no candidates for equilibrium overriding\n')
    
    # extract
    try:
        shooting_point = extract_frame(fname, position, topology)
    except:
        print('!!! Attention! Frame extraction failed. Attempting a new one')
        return initialize_shooting_simulation(
            chain, pool, directory, params,
            shooting_chains, equilibrium)
    
    # initialize simulation
    initialize_simulation(
        shooting_point,
        params,
        f'{chain.directory}/back',
        f'{chain.directory}/forw')
    
    # save
    np.save(f'{chain.directory}/shoot_bias.npy', selection_bias)
    np.save(f'{chain.directory}/pool_index.npy', index)
    print(f'\nShooting initialization completed ({now()})\n')

###############################################################################
#### Analysis #################################################################
###############################################################################

def extract_chain(pathensemble, shooting_chain_index,
                  path_index, initial_paths=None):
    shooting_chain = shooting_chain_index
    index = path_index
    directory = pathensemble.pathensembles[shooting_chain
        ].directory.split('/shots')[0]
    index = np.arange(len(pathensemble.pathensembles[shooting_chain]))[index]
    offset = int(np.sum([len(p) for p in pathensemble.
                     pathensembles[:shooting_chain]]))
    tracking = [offset + index]
    index0 = tracking[0]
    
    with open(f'{directory}/manager.log', 'r') as f:
        lines = [line for line in f]
    
    terminate = False
    for i in range(len(lines) - 1, -1 ,-1):
        if (f'Shooting for chain shots{shooting_chain}: '
            f'path{index:06g} -> path{index + 1:06g}') in lines[i]:
            j = i
            while 'Shooting initialization completed' not in lines[j]:
                j += 1
            while True:
                if '*** accepted' in lines[j]:
                    while 'traj' not in lines[j]:
                        j -= 1
                    l = lines[j]
                    file = l.split('equilibrium')[1][2:].split(',')[0]
                    folder = f'equilibrium{l.split("equilibrium")[1][0]}'
                    position = int(l.split('equilibrium')[1][2:].
                                   split(',')[1].split()[0])
                    this_offset = 0
                    for trajectory in pathensemble.pathensembles:
                        if folder not in trajectory.directory:
                            this_offset += len(trajectory)
                            continue
                        if file not in trajectory.trajectory_files:
                            this_offset += len(trajectory)
                            continue
                        file_index = trajectory.trajectory_files.index(file)
                        k = np.where((trajectory.frame_trajectory_indices
                                  == file_index) * (trajectory.
                                                    frame_trajectory_positions
                                  == position))[0]
                        tracking.append(this_offset +
                            np.where(trajectory.
                                     initial_frame_indices < k)[0][-1])
                        terminate = True
                        break
                    break
                if terminate:
                    break
                if 'selecting' in lines[j]:
                    try:
                        index = int(lines[j].split('/')[-1].split(',')[0].
                                    split('.')[0][4:]) - 1
                        tracking.append(offset + index)
                    except:
                        if initial_paths is not None:
                            tracking.append(-1 - initial_paths.
                                            trajectory_files.index(
                                lines[j].split('shooting point ')[1].
                                                split(',')[0]))
                    break
                j -= 1
    
    tracking = np.array(tracking)[::-1]
    if tracking[0] < 0:
        result = initial_paths[-tracking[0] - 1]
    else:
        result = pathensemble[tracking[0]]
    result += pathensemble[tracking[1:]]
    return result.merge()

def visualize_chain(result, _2d=False, boundaries=None):
    if _2d:
        xmin, ymin = np.min(result.frame_descriptors, axis=0)
        xmax, ymax = np.max(result.frame_descriptors, axis=0)
        old = None
        old_fname = None
        old_s = None
        for p,s,f in zip(result.descriptors(),
                         result.shooting_descriptors,
                         result.initial_trajectory_filenames):
            if old is None:
                old = p
                old_fname = f
                old_s = s
                continue
            plt.figure()
            plt.title(f'{old_fname} ->\n{f}    ')
            plt.plot(*old.T, label='old')
            plt.plot(*old_s.T, 'o', color='blue')
            plt.plot(*p.T, label='new')
            plt.plot(*s.T, 'o', color='red')
            plt.legend()
            old = p
            old_fname = f
            old_s = s
            if boundaries is not None:
                plt.xlim(boundaries[0],boundaries[2])
                plt.ylim(boundaries[1],boundaries[3])
            plt.gca().set_aspect('equal')
            plt.grid()
        return
    old = None
    old_fname = None
    old_i = None
    old_s = None
    for p,i,s,f in zip(result.values(),
                    result.shooting_indices,
                     result.shooting_values,
                     result.initial_trajectory_filenames):
        if old is None:
            old = p
            old_fname = f
            old_i = i
            old_s = s
            continue
        plt.figure()
        plt.title(f'{old_fname} ->\n{f}    ')
        plt.plot(old, label='old')
        plt.plot(old_i, old_s, 'o', color='blue')
        plt.plot(p, label='new')
        plt.plot(i, s, 'o', color='red')
        plt.legend()
        if boundaries is not None:
            plt.ylim(boundaries[0], boundaries[1])
        old = p
        old_s = s
        old_i = i
        old_fname = f
        plt.grid()    


def update_results(filename, key=[], result=[]):
    try:
        with open(filename, 'rb') as file:
            dictionary = pickle.load(file)
    except:
        dictionary = {}
    
    save = False
    for k, r in zip(key, result):
        dictionary[k] = r
        save = True
    
    if save:
        with open(filename, 'wb') as file:
            pickle.dump(dictionary, file)
    
    return dictionary

def crop_pathensemble(pathensemble, step_number):
    """
    At the time of step_number
    """
    shots, equilibriumA, equilibriumB = scorporate_pathensembles(pathensemble)
    if step_number is None:
        keepers = None
        t1 = np.inf
    else:
        completion_times = shots.completion_times
        keepers = np.argsort(completion_times)[:step_number]
        t1 = completion_times[keepers][-1]
    shots = shots[keepers]
    equilibriumA = equilibriumA.crop(tmax=t1)
    equilibriumB = equilibriumB.crop(tmax=t1)
    return shots + equilibriumA + equilibriumB


def initialize_reference_pathensemble(
    states_function, descriptors_function,
    directory='.', topology='run.gro', xy_traj='xy.xtc',
    weights=None):
    reference_pe = PathEnsemble(directory, topology,
        states_function, descriptors_function)
    reference_pe.append(xy_traj, verbose=True)
    frame_indices =  np.zeros(reference_pe.nframes * 3, dtype=int)
    frame_indices[1::3] = np.arange(reference_pe.nframes)
    reference_pe.frame_states[0] = 'Z'
    reference_pe._PathEnsemble__frame_indices = frame_indices
    reference_pe._PathEnsemble__lengths = np.ones(
        reference_pe.nframes, dtype=int) * 3
    reference_pe._PathEnsemble__shooting_indices = np.zeros(
        reference_pe.nframes, dtype=int)
    if weights:
        reference_pe._PathEnsemble__weights = nd.load(weights).ravel()
    else:
        reference_pe._PathEnsemble__weights = np.ones(reference_pe.nframes)
    reference_pe._PathEnsemble__are_accepted = np.ones(
        reference_pe.nframes, dtype=bool)  
    return reference_pe


def estimate_transition_rates_from_equilibrium(equilibrium, dt=1.):
    kAB0 = []
    kBA0 = []
    kAB0_max = []
    kBA0_max = []
    kAB0_min = []
    kBA0_min = []
    TP_lengths = []
    TP_lengths_max = []
    TP_lengths_min = []
    times0 = []
    total_tAB = []
    total_tBA = []
    current_lengths = np.zeros(0)

    transitions = equilibrium[equilibrium.are_transitions]
    shooting_trajectory_indices = equilibrium.shooting_trajectory_indices
    tr_shooting_trajectory_indices = transitions.shooting_trajectory_indices    
    
    trajectory_indices = []
    for trajectory_index in shooting_trajectory_indices:
        if trajectory_index not in trajectory_indices:
            trajectory_indices.append(trajectory_index)

    initial_times = equilibrium.initial_times
    final_times = equilibrium.final_times
    if len(trajectory_indices) == len(transitions):
        pathensemble = [transitions]
        final_times = [transitions.final_times[-1]]
    else:
        pathensemble = [
            transitions[tr_shooting_trajectory_indices == trajectory_index]
            for trajectory_index in trajectory_indices]
        final_times = [final_times[
            shooting_trajectory_indices == trajectory_index][-1] -
            initial_times[
            shooting_trajectory_indices == trajectory_index][0]
            for trajectory_index in trajectory_indices]
    
    for equilibrium, base in zip(pathensemble, padcumsum(final_times)):
        t = []
        ti = equilibrium.frame_times[0] * dt
        t0 = equilibrium.final_times * dt - ti
        lengths = equilibrium.internal_lengths * dt
        if len(times0):
            times0 += list(t0 + base * dt)
        else:
            times0 = list(t0)
        start_from_A = equilibrium.final_states[0] == 'A'
        for i in range(len(t0)):
            current_lengths = np.append(current_lengths, [lengths[i]])
            t.append(t0[i])
            if start_from_A:
                tAB = np.diff(t)[0::2]  # BA information
                tBA = np.diff(t)[1::2]  # AB information
            else:
                tAB = np.diff(t)[1::2]  # BA information
                tBA = np.diff(t)[0::2]  # AB information
            current_tAB = np.append(total_tAB, tAB)
            current_tBA = np.append(total_tBA, tBA)
            if len(current_tAB):
                kAB = 1 / np.mean(current_tAB)
                temp = []
                for bootstrapping_event in range(1000):
                    k = np.random.choice(len(current_tAB), len(current_tAB))
                    temp.append(1 / np.mean(current_tAB[k]))
                kAB_max = np.quantile(temp, .975)
                kAB_min = np.quantile(temp, .025)
            else:
                kAB = np.nan
                kAB_max = np.nan
                kAB_min = np.nan
            if len(current_tBA):
                kBA = 1 / np.mean(current_tBA)
                temp = []
                for bootstrapping_event in range(1000):
                    k = np.random.choice(len(current_tBA), len(current_tBA))
                    temp.append(1 / np.mean(current_tBA[k]))
                kBA_max = np.quantile(temp, .975)
                kBA_min = np.quantile(temp, .025)
            else:
                kBA = np.nan
                kBA_max = np.nan
                kBA_min = np.nan
            temp = []
            for bootstrapping_event in range(1000):
                k = np.random.choice(len(current_lengths), len(current_lengths))
                temp.append(np.mean(current_lengths[k]))
            TP_lengths.append(np.mean(current_lengths))
            TP_lengths_max.append(np.quantile(temp, .975))
            TP_lengths_min.append(np.quantile(temp, .025))
            kAB0.append(kAB)
            kBA0.append(kBA)
            kAB0_max.append(kAB_max)
            kBA0_max.append(kBA_max)
            kAB0_min.append(kAB_min)
            kBA0_min.append(kBA_min)
        total_tAB = current_tAB
        total_tBA = current_tBA
        
    return (np.array(kAB0), np.array(kBA0),
            np.array(kAB0_max), np.array(kBA0_max),
            np.array(kAB0_min), np.array(kBA0_min),
            np.array(TP_lengths),
            np.array(TP_lengths_max), np.array(TP_lengths_min),
            np.array(times0))


def compute_energies_and_rates(pathensemble,
                               bins=[-np.inf, +np.inf],
                               bootstrapping=0,
                               reweight_while_bootstrapping=False,
                               states='AB',
                               reweight_parameters={},
                               f=None,
                               frames=False,
                               verbose=False):
    # boostrap
    bootstrapping_results = []
    bootstrapping_k = []
    bootstrapping_e = []
    bootstrapping_z = []
    
    for _ in tqdm(range(bootstrapping), position=0, disable=not verbose, file=sys.stdout):
        k = np.random.choice(len(pathensemble), len(pathensemble))
        if reweight_while_bootstrapping:
            E = []
            Z = []
            K = []
            total_weights = pathensemble.weights * 0.
            for state in states:
                w, t1, t2, t3, z, m, e, s, t4 = pathensemble.reweight(
                    state=state,key=k,**reweight_parameters)
                E.append(e)
                Z.append(z)
                old_weights = pathensemble.weights
                weights = pathensemble.weights * 0.
                for i, kk in enumerate(k):
                    weights[kk] += w[i]
                total_weights += weights
                pathensemble.weights = weights
                K.append(1 / pathensemble.project()[0])
                pathensemble.weights = old_weights
            pathensemble.weights = total_weights
            bootstrapping_results.append(
                pathensemble.project(bins=bins, f=f, frames=frames))
            pathensemble.weights = old_weights
            bootstrapping_k.append(K)
            bootstrapping_e.append(E)
            bootstrapping_z.append(Z)
        else:
            bootstrapping_k.append([np.nan])
            bootstrapping_e.append([np.nan])
            bootstrapping_z.append([np.nan])
            bootstrapping_results.append(pathensemble.project(
                key=k, bins=bins, f=f, frames=frames))
        bootstrapping_results[-1] /= np.sum(bootstrapping_results[-1])
    bootstrapping_results = np.array(bootstrapping_results)
    bootstrapping_k = np.array(bootstrapping_k)
    bootstrapping_e = np.array(bootstrapping_e, dtype=object)
    bootstrapping_z = np.array(bootstrapping_z, dtype=object)
    
    result = pathensemble.project(bins=bins, f=f, frames=frames)
    result /= np.sum(result)
    return (result, bootstrapping_results,
            bootstrapping_k,
            bootstrapping_e,
            bootstrapping_z)


def plot_2d_energy(X, Y, F, levels,
                   X2=None, Y2=None, P=None,
                   rc_levels=None,
                   rc_labels=True,
                   cmap='magma', clabel='[kT]',
                   rotate_clabel=True,
                   xA=None, yA=None,
                   xB=None, yB=None,
                   rA=None, rB=None,
                   wrmse=0.):
    figure, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    if X2 is not None:
        drop = len(X) // (len(X2) * 2)
    else:
        drop = 0
    print(drop)
    
    X = X[drop:len(X)-drop,
                   drop:len(X[0])-drop]
    Y = Y[drop:len(Y)-drop,
                   drop:len(Y[0])-drop]
    F = F[drop:len(F)-drop,
                   drop:len(F[0])-drop]
    plt.contourf(X, Y, F, levels=levels, cmap=cmap, zorder=40)
    ax.set_aspect(abs((X[-1, -1] - X[0, 0]) / (Y[-1, -1] - Y[0, 0])))
    plt.subplots_adjust(left=0.16, bottom=0.1, right=0.8, top=1.04)
    
    if rotate_clabel:
        c = plt.colorbar(fraction=0.0452)
        plt.text(X[0,0] + (X[-1,-1] - X[0,0]) * (
            1.25 + .05 * (levels[-1] >= 10.)),
                 Y[0,0] + (Y[-1,-1] - Y[0,0]) * .94, clabel)
    else:
        plt.colorbar(fraction=0.0452,label=clabel)
    
    if xA is not None and yA is not None:
        plt.text(xA, yA, 'A',
                 ha='center',
                 va='center',
                 fontsize=20,
                 color='black',
                 path_effects=[pe.withStroke(linewidth=3, foreground="w")],
                 fontweight=750,
                 zorder=60)
    if yA is not None and yB is not None:
        plt.text(xB, yB, 'B',
                 ha='center',
                 va='center',
                 fontsize=20,
                 color='black',
                 path_effects=[pe.withStroke(linewidth=3, foreground="w")],
                 fontweight=750,
                 zorder=60)
    
    if P is not None:
        if X2 is None:
            X2 = X
            Y2 = Y
            P = P[drop:len(X)-drop,
                  drop:len(X[0])-drop]
        if len(rc_levels) > 8:
            plt.contour(X2,Y2,P,zorder=50,
                        colors='#cccccc', alpha=.84, linewidths=1.36,
                         levels=rc_levels[::2])
            c = plt.contour(X2,Y2,P,zorder=50,
                          colors='#cccccc', alpha=.84, linewidths=1.36,
                         levels=rc_levels[1::2])
        else:
            c = plt.contour(X2,Y2,P,zorder=50,
                          colors='#cccccc', alpha=.84, linewidths=1.36,
                         levels=rc_levels)
        if rc_labels:
            f=plt.clabel(c, colors='black')
            plt.setp(f, path_effects=[pe.withStroke(linewidth=3, foreground="w")])
        
    return figure, ax


def plot_1d_energy_profile(pathensemble,
                           reference,
                           nbins=100,
                           vmin=-np.inf,
                           vmax=+np.inf,
                           bootstrapping=0,
                           pathensemble_bootstrapping=500,
                           reference_bootstrapping=50,
                           reweight_while_bootstrapping=True,
                           states='AB',
                           reweight_parameters={},
                           offset=np.nan,
                           max_error=1.,
                           verbose=False,
                           base_color=plt.get_cmap('Dark2')(0),
                           SP_selection_bins=None):
    
    vmin = np.max([-30, vmin, np.min(np.concatenate(reference.values(reference.weights > 0)))])
    vmax = np.min([+30, vmax, np.max(np.concatenate(reference.values(reference.weights > 0)))])
    bins = np.linspace(vmin, vmax, nbins + 1)
    values = (bins[:-1] + bins[1:]) / 2
    
    # reference
    result, bootstrapping_results, *_ = (
        compute_energies_and_rates(
            reference, bins,
            bootstrapping=reference_bootstrapping,
            reweight_while_bootstrapping=False,
            verbose=verbose))
    
    F0 = -np.log(result)
    if reference_bootstrapping:
        F0_bootstrapping = -np.log(bootstrapping_results)
        F0_min = F0 - np.std(F0_bootstrapping, axis=0)
        F0_min = np.quantile(F0_bootstrapping, 0.025, axis=0)
        F0_max = F0 + np.std(F0_bootstrapping, axis=0)
        F0_max = np.quantile(F0_bootstrapping, 0.975, axis=0)
    else:
        F0_min = F0
        F0_max = F0

    # pe estimate
    result, bootstrapping_results, *_ = (
        compute_energies_and_rates(
            pathensemble, bins,
            bootstrapping=pathensemble_bootstrapping,
            reweight_while_bootstrapping=reweight_while_bootstrapping,
            states=states,
            reweight_parameters=reweight_parameters,
            verbose=verbose))
    
    F = -np.log(result)
    if pathensemble_bootstrapping:
        F_bootstrapping = -np.log(bootstrapping_results)
        F_min = F - np.std(F_bootstrapping, axis=0)
        F_min = np.quantile(F_bootstrapping, 0.025, axis=0)
        F_max = F + np.std(F_bootstrapping, axis=0)
        F_max = np.quantile(F_bootstrapping, 0.975, axis=0)
        k = ~np.isnan(F_max)
    else:
        F_min = F
        F_max = F
        k = ~np.isinf(F)
    
    values = values[k]
    F = F[k]
    F0 = F0[k]
    F_min = F_min[k]
    F0_min = F0_min[k]
    F0_max = F0_max[k]
    F_max = F_max[k]
        
    if not np.isnan(offset):
        center = np.argmin(np.abs(values-offset))
        F -= F0[center]
        F_min -= F0[center]
        F_max -= F0[center]
        F0_min -= F0[center]
        F0_max -= F0[center]
        F0 -= F0[center]
    
    # plot
    figure, (ax1, ax2) = plt.subplots(2,1,
                                      figsize=(4,2.7),
                                      sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})
    color = base_color

    # uncertanties
    if np.sum(np.abs(F0_max - F0_min)):
        ax1.fill_between(values, F0_min, F0_max, color='black', alpha=.25)
        ax2.fill_between(values, F0_min-F0, F0_max-F0, color='black', alpha=.25)
    if np.sum(np.abs(F_max - F_min)):
        ax1.fill_between(values, F_min, F_max, color=color2, alpha=.4)
        ax2.fill_between(values, F_min-F0, F_max-F0, color=color2, alpha=.4)
    
    # estimates
    ax1.plot(values, F, '.', color=color, markersize=2.5)
    ax1.plot(values, F, '-', color=color, lw=2.5)
    ax1.plot(values, F0, ':', color='black')

    # errors
    ax2.plot(values, F - F0, '.', color=color, markersize=2.5)
    ax2.plot(values, F - F0, color=color, lw=2.5)
    ax2.plot(values, values * 0, ':', color='black')
    
    # fix axis
    ax1.grid()
    ax2.grid()
    ax2.set_xlabel('Reaction coordinate $\lambda$')
    ax1.set_ylabel('Free energy [$k_BT$]')
    ax2.set_ylabel('Est$-$true')
    if max_error >= 1.5:
        ax2.set_yticks([-1, 0., 1], ['$-$1','0','+1'])
    elif max_error >= .75:
        ax2.set_yticks([-.5, 0., .5], ['$-$0.5','0.0','+0.5'])
    elif max_error >= .25:
        ax2.set_yticks([-.2, 0., .2], ['$-$0.2','0.0','+0.2'])
    ax2.set_ylim(-max_error, max_error)
    plt.minorticks_off()
    plt.subplots_adjust(left=.2102,
                        bottom=.1958,
                        right=.9148,
                        top=.9373,
                        hspace=0)
    
    if SP_selection_bins is not None:
        H = np.histogram(pathensemble.shooting_values[
            pathensemble.are_shot],
            SP_selection_bins)[0].astype(float)
        ylim = ax1.get_ylim()
        H /= np.max(H)
        H *= .4 * (ylim[1] - ylim[0])
        H += ylim[0] + (ylim[1] - ylim[0]) * .067
        A = np.repeat(SP_selection_bins, 2)
        H = np.repeat(H, 2)
        K = np.zeros(len(A))
        K[0] = ylim[0] + (ylim[1] - ylim[0]) * .067
        K[-1] = ylim[0] + (ylim[1] - ylim[0]) * .067
        K[1:-1] = H
        ax1.fill_between(
            A, A * 0 + ylim[0] + (ylim[1] - ylim[0]) * .067,
            K, color='black',
            alpha=.2, zorder=-10)
        ax1.set_ylim(ylim)
        
    return figure, (ax1, ax2)



def project_on_grid(pathensemble, X, Y, f=lambda x:x, frames=False):
    Z = pathensemble.project([X[0, :], Y[:, 0]], f=f, frames=frames)
    Z /= np.sum(Z)
    return Z


def plot_2d_committor_estimate_vs_reference(grid_X, grid_Y, grid_V,
                                            grid_committor_estimate,
                                            grid_committor_relaxation,
                                            lambdaA,
                                            lambdaB,
                                            xA, yA, xB, yB, radius,
                                            potential_energy_levels,
                                            grid_committor_levels,
                                            exact_levels=None,
                                            error_threshold=.125,
                                            logit=False,
                                            rescale_error=True):
    figure = plt.figure(figsize=(4/1.12, 3/1.08))
    if logit is False:
        error = (expit(grid_committor_estimate) - 
                 expit(grid_committor_relaxation))
    else:
        error = grid_committor_estimate - grid_committor_relaxation
    
    if rescale_error:
        error /= (grid_committor_relaxation *
                  (1 - grid_committor_relaxation) * 4) ** .5
    error[np.isinf(error)] = np.nan
    error[error >= +error_threshold] = + error_threshold - 1e-9
    error[error <= -error_threshold] = - error_threshold + 1e-9
    error[grid_V > potential_energy_levels[-1]] = np.nan    
    # contours
    plt.gca().set_aspect('equal')
    plt.contourf(grid_X,
                 grid_Y,
                 error,
                 levels=np.linspace(-error_threshold, error_threshold, 11),
                 cmap='RdYlGn')
    plt.colorbar(fraction=0.0452)

    if logit:
        plt.text(grid_X[-1,-1] * 1.33, grid_Y[-1,-1] * 1.05,
                 '$\lambda-\\mathrm{logit}(p_B)$', ha='right')
    else:
        plt.text(grid_X[-1,-1] * 1.33, grid_Y[-1,-1] * 1.05,
                 '$\\mathrm{expit}(\lambda)-p_B$', ha='right')
    
    plt.contour(grid_X,
                grid_Y,
                grid_V,
                levels=potential_energy_levels,
                colors='#a0a0a0',
                linewidths=1,
                alpha=.64)
    
    # validity region
    grid_validity_region = grid_X * 0.
    grid_validity_region[grid_committor_estimate < lambdaA] = 1.
    grid_validity_region[grid_committor_estimate > lambdaB] = 1.
    plt.contourf(grid_X,
                 grid_Y,
                 grid_validity_region,
                 levels=[.5, 1],
                 colors='black',
                 alpha=0.25)
    
    # states
    circleA = plt.Circle((xA, yA),
                         radius,
                         ec='black',
                         fc=(1,1,1,1),
                         zorder=20)
    circleB = plt.Circle((xB, yB),
                         radius,
                         ec='black',
                         fc=(1,1,1,1),
                         zorder=20)
    plt.text(xA, yA, 'A',
             ha='center',
             va='center',
             fontsize=20,
             color='black',
             fontweight=750,
             zorder=22)
    plt.text(xB, yB, 'B',
             ha='center',
             va='center',
             fontsize=20,
             color='black',
             fontweight=750,
             zorder=22)
    ax = plt.gca()
    ax.add_patch(circleA)
    ax.add_patch(circleB)

    if exact_levels is None:
        exact_levels = grid_committor_levels
    plt.contour(grid_X, grid_Y, grid_committor_relaxation, zorder=-10,
                 colors='#cccccc', levels=exact_levels)
    plt.contour(grid_X, grid_Y, grid_committor_estimate, zorder=-10,
                 colors='#333333', levels=grid_committor_levels)
    
    plt.xlim(grid_X[0,0],grid_X[-1,-1])
    plt.ylim(grid_Y[0,0],grid_Y[-1,-1])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_aspect(abs((xlim[-1]-xlim[0])/(ylim[-1]-ylim[0])))
    figure.subplots_adjust(
        left=.12,
        bottom=.15,
        right=.8,
        top=.92,
        hspace=0)
    figure.set_size_inches(4/1.12, 3/1.08)
    return figure, ax


def create_equilibrium_tpe():
    # TODO much easier if saved as xtc files. Keep like this for now.
    equilibrium = tps[:0]
    trajs = [f'equilibrium/{file}' for file in sorted(os.listdir('equilibrium'))
            if len(file) == 20 and file[:10] == 'transition'
            and file[-4:] == '.npy']
    for traj in tqdm(trajs[:4000], position=0, file=sys.stdout):
        t = np.load(traj)
        equilibrium._update(
            trajectory_files = equilibrium.trajectory_files + [f'{len(equilibrium)}'],
            frame_trajectory_indices = np.append(
                equilibrium.frame_trajectory_indices,
                np.repeat(len(equilibrium), len(t))),
            frame_trajectory_positions = np.append(
                equilibrium.frame_trajectory_positions,
                np.arange(len(t))),
            frame_times = np.append(
                equilibrium.frame_times,
                np.arange(len(t))),
            frame_simulation_times = np.append(
                equilibrium.frame_simulation_times,
                np.arange(len(t))),
            frame_states = np.append(
                equilibrium.frame_states,
                ['A'] + ['R'] * (len(t) - 2) + ['B']),
            frame_descriptors = np.append(
                equilibrium.frame_descriptors,
                t, axis=0) if equilibrium.nframes else t,
            frame_values = np.append(
                equilibrium.frame_values,
                np.zeros(len(t))),
            frame_indices = np.append(
                equilibrium._PathEnsemble__frame_indices,
                np.arange(len(t)) + equilibrium.nframes),
            lengths = np.append(equilibrium.lengths, [len(t)]),
            weights = np.append(equilibrium.weights, [1.]),
            shooting_indices = np.append(equilibrium.shooting_indices, [0]),
            are_accepted = np.append(equilibrium.are_accepted, [True]))
        equilibrium.save('equilibrium/pe.h5')


def initialize_results(n=4000):
    return {'step_numbers': np.zeros(n, dtype=int) + np.nan,
            'kAB': np.zeros(n) + np.nan,
            'kBA': np.zeros(n) + np.nan,
            'kAB_max': np.zeros(n) + np.nan,
            'kAB_min': np.zeros(n) + np.nan,
            'kBA_max': np.zeros(n) + np.nan,
            'kBA_min': np.zeros(n) + np.nan,
            'times': np.zeros(n) + np.nan,
            'timesA': np.zeros(n) + np.nan,
            'timesB': np.zeros(n) + np.nan,
            'timesS': np.zeros(n) + np.nan,
            'timesT': np.zeros(n) + np.nan,
            'TP_number': np.zeros(n, dtype=int) + np.nan,
            'TP_length': np.zeros(n) + np.nan,
            'TP_length_min': np.zeros(n) + np.nan,
            'TP_length_max': np.zeros(n) + np.nan,
            'channel_differences': np.zeros(n) + np.nan}


def compute_average_tps_lenghts(tps, dt=1.):
    TP_length = []
    TP_length_max = []
    TP_length_min = []
    lengths = tps.internal_lengths * dt
    weights = tps.weights
    for i in tqdm(range(len(tps)), position=0, file=sys.stdout):
        TP_length.append(
            np.average(lengths[:i + 1], weights=weights[:i + 1]))
        temp = []
        for bootstrapping_event in range(1000):
            k = np.random.choice(i + 1, i + 1)
            temp.append(np.average(lengths[k], weights=weights[k]))
        TP_length_max.append(np.quantile(temp, .975))
        TP_length_min.append(np.quantile(temp, .025))
    return (np.array(TP_length),
            np.array(TP_length_max),
            np.array(TP_length_min))

