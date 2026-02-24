"""
...
"""

# external imports
import copy
import time
import torch
import numpy as np
from math import inf
from tqdm import tqdm
from scipy.special import expit

# aimmd imports
from .utils import extract_indices_and_series
from ..core.utils import concatenate, now
from ..analysis.utils import compute_bins, merge_marginal_bins

# fit function
def fit(params,
        pathensemble,
        keys=None,
        
        # binning
        nbins=0,
        cutoff_min=0.5,
        cutoff_max=20.,
        state_bins='',
        max_adjustment_in_bin=10.0,
        
        # data augmentation
        augment='no',
        
        # learning
        lr=1e-3,
        loss_bayesian_factor=0,
        loss_smoothening_weight=0,
        loss_regularization_weight=0,
        epochs=500,
        batch_size=4096,
        stop=50.,
        train_validation_early_stopping=False,
        early_stopping_patience=10,
        early_stopping_min_samples=1000,
        early_stopping_split=0.1,
        
        # processing
        in_memory=False,
        graphs=False,

        # misc
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
    params : aimmd.Params
        params.network: contains the neural network model to train
        params.descriptor_transform: from descriptors to direct NN input
    
    pathensemble : PathEnsemble
        Collection of sampled paths, including shooting and free simulations.
        Used to build the training set. `pathensemble.params.network` contains
        the neural network model to train. `pathensemble.initial_paths` are
        used if `pathensemble` has no sampled paths.
    
    keys : array-like of int, range, or slice, optional
        Use only the paths at these indices for training. Useful for
        bootstrapping or block averaging.
    
    nbins : int, default=0
        Divide the reactive space in `nbins + 2` bins based on the input
        neural network model, and assign selection probabilities to the
        training set points such that each batch has a uniform population
        across the bins. Plus, regularize the shooting results in the bins.
        `nbins = 10` is usually a good value.
    
    cutoff_max : float, default=20.
        The `cutoff_max` parameter in `utils.compute_bins`.

    cutoff_min : float, default=0.5
        The `cutoff_min` parameter in `utils.compute_bins`.
    
    state_bins : str, default=''
        Add two additional bins for the states defined in `state_bins`.
        For example, `state_bins = 'AB'` includes both in A and in B data
        in the training set. Assign selection probabilities to the training
        set points such that each batch has a uniform population across all
        the bins. state_bins = 'all' means include all states bins.

    max_adjustment_in_bin : float, default=10
        Allow for rare shooting results to compute at most
        "max_adjustment_in_bin" more frequently in the training batch,
        counterbalancing this with a rescaling of the shooting result.
    
    augment : str, default='no'
        If 'yes', include all available data in the training set: shooting
        points, two-way shooting simulations, and free trajectories. This
        broadens coverage and improves learning. Otherwise, it just trains
        on the shooting points and their associated results.
        If 'experimental': also assign further fractional results to
        transitions to  further improve the training performance.
    
    lr : float, default=1e-3
        Learning rate. Will use an ADAM optimizer.
    
    loss_bayesian_factor : float, default=100.0
        In defining the logit committor square deviation training loss,
        necessary with discrete data. If zero: apply standard logit
        binomial loss (worse close to the states).
    
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

    stop : float, default=50.0
        Stop when the scale of the logit committor reaches this value.
        
    train_validation_early_stopping : bool, default=False
        If True, split the training set into a training and validation set,
        and use early stopping based on the validation loss.
    
    early_stopping_patience : int, default=10
        Number of epochs with no improvement on the validation loss. After this
        number of epochs, stop the training. Reset to best validation epoch 
        at the end.

    early_stopping_min_samples : int, default=1000
        Minimum number of samples in the training set to enable early stopping.

    early_stopping_split : float, default=0.1
        Fraction of the training set to use as validation set for early stopping.

    in_memory : bool, default=False
        If `True`, reload (transformed) descriptors for every batch.
        If `False`, load *all* descriptors just once.
        ATM ALL DESCRIPTORS ARE LOADED TOGETHER, TRANSFORMED LATER if True
    
    graphs : bool, default=False
        If True, the descriptors are graphs, stored in the mlcolvar DataDict
        format. If False, they are numpy arrays.
    
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
    """

    # check input parameters
    if augment not in ('no', 'yes', 'experimental'):
        raise TypeError("augment parameter must be "
                        "either 'no', 'yes', or 'experimental'")      
    
    # will periodically check
    def must_stop():
        return getattr(worker, 'termination_signal', False)
    
    # initialize
    t0 = time.time()
    losses, scales = [], []
    network = params.network
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    a, r, b = states = params.sorted_states
    descriptors_source = ('descriptors' if params.descriptors_function else
                          'positions')
    descriptor_transform = params.descriptor_transform or (lambda x:x)
    
    if graphs:
        # only need to import this torch_geometric Batch if graphs
        # are used as descriptors. Otherwise, avoid the dependency.
        from torch_geometric.data import Batch
    
    if not augment:
        th1 = None
        th2 = None
    
    # classify paths
    types = pathensemble.types()
    i, t, f, s = types.view('U1').reshape(len(types), 4).T
    # i: initial states of path
    # f: final states of path
    # s: shooting states of path
    # t: internal states path

    # collect all data
    # separate paths for convenience by internal indices
    (in1, in1_back, in1_forw, in1_descriptors,
     npaths_in1) = extract_indices_and_series(pathensemble,
        np.flatnonzero((i == r) & (t == a)), descriptors_source)
    if must_stop():
        return [], [], [], [], []
    (in2, in2_back, in2_forw, in2_descriptors,
     npaths_in2) = extract_indices_and_series(pathensemble,
        np.flatnonzero((i == r) & (t == b)), descriptors_source)
    if must_stop():
        return [], [], [], [], []
    (free1to1, free1to1_back, free1to1_forw,
     free1to1_values, free1to1_descriptors,
     npaths_free1to1) = extract_indices_and_series(pathensemble,
        np.flatnonzero((i == a) & (t == r) & (f != b) & (s == a)),
        'values', descriptors_source)
    if must_stop():
        return [], [], [], [], []
    (free2to2, free2to2_back, free2to2_forw,
     free2to2_values, free2to2_descriptors,
     npaths_free2to2) = extract_indices_and_series(pathensemble,
        np.flatnonzero((i == b) & (t == r) & (f != a) & (s == b)),
        'values', descriptors_source)
    if must_stop():
        return [], [], [], [], []
    (free1to2, free1to2_back, free1to2_forw,
     free1to2_values, free1to2_descriptors,
     npaths_free1to2) = extract_indices_and_series(pathensemble,
        np.flatnonzero((i == a) & (t == r) & (f == b) & (s == a)),
        'values', descriptors_source)
    if must_stop():
        return [], [], [], [], []
    (free2to1, free2to1_back, free2to1_forw,
     free2to1_values, free2to1_descriptors,
     npaths_free2to1) = extract_indices_and_series(pathensemble,
        np.flatnonzero((i == b) & (t == r) & (f == a) & (s == b)),
        'values', descriptors_source)
    if must_stop():
        return [], [], [], [], []
    (shot1to1, shot1to1_back, shot1to1_forw,
     shot1to1_values, shot1to1_descriptors,
     npaths_shot1to1) = extract_indices_and_series(pathensemble,
        np.flatnonzero((i == a) & (t == r) & (f != b) & (s == r)),
        'values', descriptors_source)
    if must_stop():
        return [], [], [], [], []
    (shot2to2, shot2to2_back, shot2to2_forw,
     shot2to2_values, shot2to2_descriptors,
     npaths_shot2to2) = extract_indices_and_series(pathensemble,
        np.flatnonzero((i == b) & (t == r) & (f != a) & (s == r)),
        'values', descriptors_source)
    if must_stop():
        return [], [], [], [], []
    (shot1to2, shot1to2_back, shot1to2_forw,
     shot1to2_values, shot1to2_descriptors,
     npaths_shot1to2) = extract_indices_and_series(pathensemble,
        np.flatnonzero((i == a) & (t == r) & (f == b) & (s == r)),
        'values', descriptors_source)
    if must_stop():
        return [], [], [], [], []
    (shot2to1, shot2to1_back, shot2to1_forw,
     shot2to1_values, shot2to1_descriptors,
     npaths_shot2to1) = extract_indices_and_series(pathensemble,
        np.flatnonzero((i == b) & (t == r) & (f == a) & (s == r)),
        'values', descriptors_source)
    if must_stop():
        return [], [], [], [], []
    
    # report
    lengths = [len(in1), len(in2),
               len(free1to1), len(free2to2),
               len(free1to2), len(free2to1),
               len(shot1to1), len(shot2to2),
               len(shot1to2), len(shot2to1)]
    print(f'\nCollected {lengths[0]:9} in {a} frames '
                   f'({npaths_in1:6} paths),\n'
            f'          {lengths[1]:9} in {b} frames '
                   f'({npaths_in2:6} paths),\n'
            f'          {lengths[2]:9} free {a} to {a} frames '
                   f'({npaths_free1to1:6} paths),\n'
            f'          {lengths[3]:9} free {b} to {b} frames '
                   f'({npaths_free2to2:6} paths),\n'
            f'          {lengths[4]:9} free {a} to {b} frames '
                    f'({npaths_free1to2:6} paths),\n'
            f'          {lengths[5]:9} free {b} to {a} frames '
                   f'({npaths_free2to1:6} paths),\n'
            f'          {lengths[6]:9} shot {a} to {a} frames '
                   f'({npaths_shot1to1:6} paths),\n'
            f'          {lengths[7]:9} shot {b} to {b} frames '
                   f'({npaths_shot2to2:6} paths),\n'
            f'          {lengths[8]:9} shot {a} to {b} frames '
                   f'({npaths_shot1to2:6} paths), and\n'
            f'          {lengths[9]:9} shot {b} to {a} frames '
                   f'({npaths_shot2to1:6} paths)\n'
            f'   TOTAL: {sum(lengths):9} frames')
    
    print(f'\nCalculating shooting results and sel. probabilities {now()}')
    
    # assign results
    in1_results = np.zeros((lengths[0], 2))
    in2_results = np.zeros((lengths[1], 2))
    free1to1_results = np.zeros((lengths[2], 2))
    free2to2_results = np.zeros((lengths[3], 2))
    free1to2_results = np.zeros((lengths[4], 2))
    free2to1_results = np.zeros((lengths[5], 2))
    shot1to1_results = np.zeros((lengths[6], 2))
    shot2to2_results = np.zeros((lengths[7], 2))
    shot1to2_results = np.zeros((lengths[8], 2))
    shot2to1_results = np.zeros((lengths[9], 2))

    # in 1
    in1_results[:, 0] = 1.
    
    # in 2
    in2_results[:, 1] = 1.

    # at shooting points
    if augment == 'no':
        shot1to1_results[shot1to1_back & shot1to1_forw, 0] += 2.0
        shot2to2_results[shot2to2_back & shot2to2_forw, 1] += 2.0
        shot1to2_results[shot1to2_back & shot1to2_forw] += 1.0
        shot2to1_results[shot2to1_back & shot2to1_forw] += 1.0

    # augment
    else:
        free1to1_results[:, 0] += 1.0
        free2to2_results[:, 1] += 1.0
        free1to2_results[:, 0] += 1.0
        free2to1_results[:, 1] += 1.0        
        shot1to1_results[shot1to1_back, 0] += 1.0
        shot1to1_results[shot1to1_forw, 0] += 1.0
        shot2to2_results[shot2to2_back, 1] += 1.0
        shot2to2_results[shot2to2_forw, 1] += 1.0
        shot1to2_results[shot1to2_back, 0] += 1.0
        shot1to2_results[shot1to2_forw, 1] += 1.0
        shot2to1_results[shot2to1_back, 1] += 1.0
        shot2to1_results[shot2to1_forw, 0] += 1.0

    if augment == 'experimental':
        conversion1to2 =    + expit(free1to1_values + free1to2_values).sum()
        conversion2to1 = (1 - expit(free2to2_values + free2to1_values)).sum()
        conversion1to2 /= len(shot1to2_results) + len(shot1to2_results)
        conversion2to1 /= len(shot1to2_results) + len(shot1to2_results)
        free1to1_results[:, 0] += 1.0
        free1to2_results[:, 1] += 1.0
        free2to2_results[:, 1] += 1.0
        free2to1_results[:, 0] += 1.0
        shot1to2_results[:, 1] += conversion1to2
        shot1to2_results[:, 0] += conversion2to1
        shot2to1_results[:, 1] += conversion1to2
        shot2to1_results[:, 0] += conversion2to1
        print(f'Augmentation factor from {a} to {b}: {conversion1to2:.3e}')
        print(f'Augmentation factor from {b} to {a}: {conversion2to1:.3e}')
    
    # add together
    values = concatenate([free1to1_values, free2to2_values,  # only reactive
                          free1to2_values, free2to1_values,
                          shot1to1_values, shot2to2_values,
                          shot1to2_values, shot2to1_values])
    descriptors = concatenate([in1_descriptors, in2_descriptors,
                               free1to1_descriptors, free2to2_descriptors,
                               free1to2_descriptors, free2to1_descriptors,
                               shot1to1_descriptors, shot2to2_descriptors,
                               shot1to2_descriptors, shot2to1_descriptors])
    results = concatenate([in1_results, in2_results,
                           free1to1_results, free2to2_results,
                           free1to2_results, free2to1_results,
                           shot1to1_results, shot2to2_results,
                           shot1to2_results, shot2to1_results])
    selection_probabilities = np.zeros(sum(lengths))
    if must_stop():
        return [], [], [], [], []

    # compute selection bins
    bins = compute_bins(pathensemble, max(nbins, 1),
                        cutoff_max, cutoff_min,
                        find_extremes_with='transitions',
                        source='values',
                        states=states,
                        marginal_bins='all')
    if must_stop():
        return [], [], [], [], []
    
    # uniformize in bins
    # in 1, in 2 data
    n_internal_frames = lengths[0] + lengths[1]
    if lengths[0]:
        selection_probabilities[:lengths[0]] = 1 / lengths[0]  
    if lengths[1]:
        selection_probabilities[lengths[0]:n_internal_frames
            ] = 1 / lengths[1]

    # merge sparse bins together
    print(f'\nBins {bins}')
    bins, counts = merge_marginal_bins(bins,
        values[results[n_internal_frames:, 0].astype(bool)],
        values[results[n_internal_frames:, 1].astype(bool)], min_values=3)
    if len(bins) < nbins + 1:
        print(f'*** merged {nbins + 2 - len(bins)} bins together '
              f'to avoid overfitting: {bins}')
    
    # other bins
    indices = np.digitize(values, bins) - 1
    present = results[n_internal_frames:].sum(axis=1).astype(bool)    
    for i, bin_counts in enumerate(counts):
        # which data are we talking about?
        mask = n_internal_frames + np.flatnonzero((indices == i) & present)
        if not mask.any():
            continue
        
        # correct for imbalance
        # let selection probability absorb imbalance from states 1 and 2
        bin_results = results[mask]
        mask0 = mask[bin_results.prod(axis=1) > 0]  # both
        mask1 = mask[bin_results[:, 1] == 0]  # only 1
        mask2 = mask[bin_results[:, 0] == 0]  # only 2
        adjust0 = adjust1 = adjust2 = 0.
        if mask0.size:
            adjust0 = min(max_adjustment_in_bin, mask.size / mask0.size)
            results[mask0] /= adjust0
            selection_probabilities[mask0] = adjust0
        if mask1.size:
            adjust1 = min(max_adjustment_in_bin, mask.size / mask1.size)
            results[mask1] /= adjust1
            selection_probabilities[mask1] = adjust1
        if mask2.size:
            adjust2 = min(max_adjustment_in_bin, mask.size / mask2.size)
            results[mask2] /= adjust2
            selection_probabilities[mask2] = adjust2
        results[mask] /= results[mask].mean()
        selection_probabilities[mask] /= selection_probabilities.sum()
        selection_probabilities[mask] *= bin_counts
        # while merged together, preserve measure
        
        if not verbose:
            continue
        
        r1, r2 = np.average(results[mask], axis=0,
                            weights=selection_probabilities[mask])
        r1, r2 = r1 / (r1 + r2), r2 / (r1 + r2)
        
        print(f'    bin {i}: ({expit(bins[i]):.3e}, {expit(bins[i+1]):.3e}) '
              f'[weight {bin_counts}]')
        print(f'    ... {len(mask):<9} frames')
        print(f'    ... {r1:.3e} average result to {a}')
        print(f'    ... {r2:.3e} average result to {b}')
        print(f'    ... {adjust0 if mask0.size else 0:.3e} ∝sel prob of [a,b]')
        print(f'    ... {adjust1 if mask1.size else 0:.3e} ∝sel prob of [a,0]')
        print(f'    ... {adjust2 if mask2.size else 0:.3e} ∝sel prob of [0,b]')
    
    # keep in a or in b only if required or truly necessary
    if (a not in state_bins and state_bins != 'all' and
        (results[n_internal_frames:, 0] >
         results[n_internal_frames:, 1]).any()):
        selection_probabilities[:lengths[0]] = 0.
    if (b not in state_bins and state_bins != 'all' and
        (results[n_internal_frames:, 1] >
         results[n_internal_frames:, 0]).any()):
        selection_probabilities[lengths[0]:n_internal_frames] = 0.

    # restrict to selection_probabilities > 0
    keepers = selection_probabilities > 0
    selection_probabilities = selection_probabilities[keepers]
    values = values[keepers[n_internal_frames:]]
    descriptors = descriptors[keepers]
    results = results[keepers]
    print(f'\nTraining set size {len(selection_probabilities)}')
    if not keepers.any() or must_stop():
        return [], [], [], [], []
    selection_probabilities /= selection_probabilities.sum()
    
    if in_memory:
        print(f'Transforming descriptors {now()}')
        descriptors = descriptor_transform(descriptors)
    
    """
    Training loop.
    """
    
    # initialization
    # backups for restoration
    state_dict0 = copy.deepcopy(network.state_dict())  # fixed
    state_dict1 = copy.deepcopy(network.state_dict())  # linked to min_loss1
    state_dict2 = copy.deepcopy(network.state_dict())  # linked to min_loss2
    min_loss1 = inf
    min_loss2 = inf
    min_loss_step1 = 0
    min_loss_step2 = 0

    # Early stopping setup

    # disable early stopping, if threshold of training set size is not met
    if (len(selection_probabilities) < early_stopping_min_samples and
        train_validation_early_stopping):
        train_validation_early_stopping = False
        print(f"\nDisabling early stopping since < {early_stopping_min_samples} samples.")

    if train_validation_early_stopping:
        state_dict_early_stopping = copy.deepcopy(network.state_dict())  # early stopping
        no_improvement_steps = 0
        min_validation_loss = np.inf
        validation_losses = []
        validation_set_size = int(len(selection_probabilities) * early_stopping_split)
        print(f"\nUsing early stopping with {validation_set_size} samples for validation set.")
        validation_indices = np.random.choice(
            len(selection_probabilities),
            size=validation_set_size,
            replace=False)
        # set selection probabilities of validation set to zero in training set
        selection_probabilities[validation_indices] = 0.
        selection_probabilities /= np.sum(selection_probabilities)
        
        # create validation vectors
        if not graphs:
            d_val = descriptor_transform(descriptors[validation_indices])
            d_val = torch.tensor(d_val, dtype=dtype, device=device)
            d_val.requires_grad = True
        else:
            # when using graphs, we need to process the DataDict objects
            # instead of arrays
            d_val_list = [descriptors['data_list'][i] for i in validation_indices]
            d_val = Batch.from_data_list(d_val_list).to(device).to_dict()                
        r_val = torch.tensor(results[validation_indices], dtype=dtype, device=device)
    
    print(f'Resetting the network parameters {now()}\n')
    network.reset_parameters()
    
    # actual loop
    print(f'Starting the training cycle {now()}')
    counter = tqdm(total=epochs, disable=not verbose)
    while True:
        
        if must_stop():
            return [], [], [], [], []
        
        for param_group in optimizer.param_groups:
            # slowly increase lr
            param_group['lr'] = lr * min(1, (counter.n + 1) / (epochs / 20))
        
        # sample batch
        indices = np.random.choice(len(selection_probabilities),
                                   batch_size, p=selection_probabilities)
        
        if not graphs:
            if not in_memory:  # separately to save memory
                d = descriptor_transform(descriptors[indices])
            else:
                d = descriptors[indices]
            d = torch.tensor(d, dtype=dtype, device=device)
            d.requires_grad = True
        else:
            # when using graphs, we need to process the the DataDict objects
            # instead of arrays
            if not in_memory:  # separately to save memory
                d = descriptor_transform(descriptors[indices,:])
            else:
                d = [descriptors['data_list'][i] for i in indices]
            d = Batch.from_data_list(d).to(device).to_dict()

        d = torch.flatten(d, start_dim=1)
        r = torch.tensor(results[indices], dtype=dtype, device=device)
        
        # define loss function
        # Note: refactored this to allow for computation of validation loss
        def loss_function(q, r):
            if loss_bayesian_factor:
                
                q1 = - (torch.log(1 + torch.exp(-q[:, 0])) +
                        loss_bayesian_factor)
                q2 = + (torch.log(1 + torch.exp(+q[:, 0])) +
                        loss_bayesian_factor)
                
                to1_contribution = (q[:, 0] - q1) ** 2
                to2_contribution = (q[:, 0] - q2) ** 2
                
                q_bis = q[:,0].detach()
                
                loss = torch.sum(q_bis ** 2 *
                    (r[:, 0] * to2_contribution +
                    r[:, 1] * to1_contribution))
                
                # normalize
                loss /= torch.sum(q_bis ** 2 * (r[:, 0] + r[:, 1]) *
                                loss_bayesian_factor ** 2)
                loss -= 1.0
            
            else:
                # binomial loss
                # (stable version, which will not output NaN for large |q|)
                p = torch.sigmoid(q[:, 0])
                to1_contrib = r[:, 0] * torch.log(1 - p + 1e-20)
                to2_contrib = r[:, 1] * torch.log(    p + 1e-20)
                loss = - torch.sum(to1_contrib + to2_contrib) / torch.sum(r)
            
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
            return loss

        def closure():
            optimizer.zero_grad()
            q = network(d)
            loss = loss_function(q, r)
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
        
        # update counter
        if verbose:
            counter.update(1)
        else:
            counter.n += 1
        
        # handle termination: too high scales
        if scales[-1] >= stop or np.isnan(scales[-1]):
            print(f'!!! stopping early since scale '
                  f'{scales[-1]:.3f} > {stop:.3f}')
            if counter.n < 1.25 * epochs:
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
            min_loss_step1 = counter.n - 1
            state_dict1 = copy.deepcopy(network.state_dict())
        
        # new min loss after reaching target epochs
        if counter.n >= epochs and losses[-1] <= min_loss2:
            min_loss2 = losses[-1]
            min_loss_step2 = counter.n - 1
            state_dict2 = copy.deepcopy(network.state_dict())
        
        # handle termination: lowest loss
        if counter.n >= epochs and losses[-1] <= min_loss1:
            break
        
        # new target after 1.25 * epochs
        if counter.n >= 1.25 * epochs and losses[-1] <= min_loss2:
            break
        
        # at most 1.5 * epochs
        if counter.n >= 1.5 * epochs:
            print(f'    restoring lowest loss\' ({min_loss2:.3e}) '
                  f'weights, step {min_loss_step2 + 1}')
            network.load_state_dict(state_dict2)
            break
        
        # D.append(d)
        # R.append(r)
        
        # handle early stopping with validation set
        if train_validation_early_stopping:
            # compute validation loss

            network.eval()
            with torch.no_grad():
                pred_val = network(d_val)
                val_loss = loss_function(pred_val, r_val)
                val_loss = float(val_loss.detach())
                validation_losses.append(val_loss)
            network.train()

            # early stopping logic. Only start when Range is acceptable
            if (Range[0] <= 0 and Range[1] >= 0 and scales[-1] >= 1):
                if val_loss < min_validation_loss:
                    min_validation_loss = val_loss
                    no_improvement_steps = 0
                    state_dict_early_stopping = copy.deepcopy(network.state_dict())
                else:
                    no_improvement_steps += 1
                    if verbose:
                        print(f'    Early stopping report, epoch {i}: no improvement for '
                        f'{no_improvement_steps} steps, loss {loss:.3e} validation loss {val_loss:.3e} '
                        f'(min val loss: {min_validation_loss:.3e})')
                    if no_improvement_steps >= early_stopping_patience:
                        print(f'    Early stopping triggered after {no_improvement_steps} steps without improvement, '
                            f'restoring best model from early stopping '
                            f'(val loss: {min_validation_loss:.3e})')
                        network.load_state_dict(state_dict_early_stopping)
                        # exit training loop
                        break

        # report
        if verbose and counter.n % max(epochs // 20, 1) == 0:
            print(f'\n    loss {losses[-1]:.3e}, '
                  f'scale {scales[-1]:.3f}, '
                  f'range ({Range[0]:.3f}, {Range[1]:.3f})')
    
    # close counter
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
    print(f'    {counter.n} epochs')
    print(f'    last loss {losses[-1]:.3e}')
    print(f'    last scale {scales[-1]:.3f}')
    print(f'    last range ({Range[0]:.3f}, {Range[1]:.3f})')
    return losses, scales, values, selection_probabilities, results#, D, R

#fit.__source__ = f'from aimmd.learning import fit'

def placeholder(params, pathensemble, key, verbose, worker):
    return fit(params=params,
               pathensemble=pathensemble,
               key=key,
               verbose=verbose,
               worker=worker)

#placholder.__source__ = f'from aimmd.learning import placholder'
