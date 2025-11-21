import os
import time
import torch
import numpy as np
import functools
import traceback
from contextlib import contextmanager
from ..core.utils import (now,
                          array2string,
                          load_initial_paths,
                          load_network_and_projections,
                          update_pathensemble,
                          scorporate_pathensembles,
                          get_bins)
from ..core.pathensemble import PathEnsemblesCollection

inf = float('inf')

# quick logging
print = functools.partial(print, flush=True)

kAB_current_estimate = None
kBA_current_estimate = None

def train(self, log_file=None, verbose=False, nrounds=inf, walltime=inf, pathensemble_fraction=-1):
    """nrounds: number of training rounds.
    walltime: maximum walltime in seconds.
    verbose: whether to print detailed information during training.
    log_file: file to log output to. If None, logs to stdout.
    pathensemble_fraction: fraction of the path ensemble to use for training. -1 means all paths.

    Returns
    -------
    pathensemble : aimmd.core.pathensemble.PathEnsemble
        The final path ensemble after training.
    
    """
    
    # report
    self.log_file = log_file
    print(f"Starting worker: train ({now()})")
    if not log_file:
        print(f"Press Control+C to interrupt.")
    
    # process arguments
    nrounds = float(nrounds)
    walltime = float(walltime)
    if verbose == 'False' or verbose == 'false':
        verbose = False
    else:
        verbose = bool(verbose)
    
    # bind resources
    self.bind_resources()
    
    # initialize output
    pathensemble = None
    
    # get aimmd run parameters
    print(f'\nLoading AIMMD run parameters ({now()})')
    
    # extract necessary parameters (in order of appearance)
    directory = self.directory
    topology = self.params.topology
    states_function = self.params.states_function
    descriptors_function = self.params.descriptors_function
    values_function = self.params.values_function
    network = self.params.network
    extra_equilibriumA = self.params.extra_equilibriumA
    extra_equilibriumB = self.params.extra_equilibriumB
    extra_equilibriumA_states_map = self.params.extra_equilibriumA_states_map
    extra_equilibriumB_states_map = self.params.extra_equilibriumB_states_map
    fit = self.params.fit
    nbins = self.params.nbins
    cutoff_max = self.params.cutoff_max
    rescale_committor = self.params.rescale_committor
    include_marginal_bins = self.params.include_marginal_bins
    reweight_parameters = self.params.reweight_parameters
    do_tps = self.params.do_tps
    sparse_update_max_frames = self.params.sparse_update_max_frames
    reweight_pathensemble_after_training = self.params.reweight_pathensemble_after_training
    
    print(f'\nLoading initial path(s) ({now()})')
    initial_paths = load_initial_paths(f'{directory}/initial_paths', topology,
        states_function, descriptors_function, values_function)
    assert initial_paths.nframes
    print(f'    {initial_paths}')
    
    # load the network if it is already possible
    load_network_and_projections(network, directory, wait=False)
    
    # stop condition
    t0 = time.time()
    counts = 0
    def stop_condition():
        self.termination_signal = None
        if time.time() - t0 > walltime or counts >= nrounds:
            self.termination_signal = 2   # sigint
        return bool(self.termination_signal)
    
    # main cycle
    print(f'\nStarting the main cycle ({now()})')
    while True:
        # stop?
        if stop_condition():
            return pathensemble
        
        print(f'\nLoading most recent path ensemble ({now()})')
        pathensemble, added_nframes = update_pathensemble(directory, topology,
            states_function, descriptors_function, values_function,
            add_missing_paths=False, add_missing_frames=False, verbose=True)
        shooting_chains, equilibriumA, equilibriumB = scorporate_pathensembles(
            pathensemble)
        
        if pathensemble_fraction > 0:
            # select a fraction of paths only
            shooting_chains_fraction = []
            for schain in shooting_chains.pathensembles:
                n_paths = len(schain)
                n_select = max(1, int(n_paths * pathensemble_fraction))
                schain_new = schain[:n_select]
                shooting_chains_fraction.append(schain_new)
            shooting_chains = PathEnsemblesCollection(*shooting_chains_fraction)

            # for equilibrium, just make fractions, without going into subpathensembles
            n_paths_A = len(equilibriumA)
            n_select_A = max(1, int(n_paths_A * pathensemble_fraction))
            equilibriumA = equilibriumA[:n_select_A]
            n_paths_B = len(equilibriumB)
            n_select_B = max(1, int(n_paths_B * pathensemble_fraction))
            equilibriumB = equilibriumB[:n_select_B]

            pathensemble = shooting_chains + equilibriumA + equilibriumB
            print(f"Made subset of path ensemble, now with {len(pathensemble)} paths and {pathensemble.nframes} frames.")
        # extra equilibriumA and equilibriumB
        if len(extra_equilibriumA):
            print(f'\nLoading extra free simulations around A ({now()})')
            equilibriumA += update_pathensemble(directory, topology,
                states_function, descriptors_function, values_function,
                add_missing_paths=False, add_missing_frames=False,
                shooting_chains=[],
                equilibriumA=extra_equilibriumA, equilibriumB=[],
                equilibriumA_states_map=extra_equilibriumA_states_map,
                verbose=True)[0]
        if len(extra_equilibriumB):
            print(f'\nLoading extra free simulations around B ({now()})')
            equilibriumB += update_pathensemble(directory, topology,
                states_function, descriptors_function, values_function,
                add_missing_paths=False, add_missing_frames=False,
                shooting_chains=[],
                equilibriumA=[], equilibriumB=extra_equilibriumB,
                equilibriumB_states_map=extra_equilibriumB_states_map,
                verbose=True)[0]
        pathensemble = shooting_chains + equilibriumA + equilibriumB
        
        print(f'\nTraining the network ({now()})')
        losses, *_ = fit(network, pathensemble, initial_paths=initial_paths,
                         verbose=verbose, worker=self)
        if not len(losses):
            if 'network.h5' in os.listdir(directory):
                bins, densities = load_network_and_projections(
                    network, directory, wait=False)
                print('!!! reloaded most recent network '
                      'because training failed')
            else:
                print('!!! training failed, retrying')
                continue

        # stop?
        if stop_condition():
            return pathensemble
        
        print(f'\nUpdating the path ensemble values ({now()})')
        pathensemble.update_values(sparse_update_max_frames=sparse_update_max_frames)
        initial_paths.update_values()
        
        print(f'\nObtaining the adaptation bins ({now()})')
        bins = get_bins(pathensemble, nbins,
            cutoff_max=cutoff_max,
            initial_paths=initial_paths,
            states=include_marginal_bins)
        print(f'    bins: {array2string(bins, 9)}')
        
        if reweight_pathensemble_after_training:
            print(f'\nReweighting the full path ensemble ({now()})')
            _reweight_parameters = reweight_parameters.copy()
            if 'sp_cutoff_min' not in _reweight_parameters and len(bins) > 3:
                _reweight_parameters['sp_cutoff_min'] = 2 * bins[+1] - bins[+2]
            if 'sp_cutoff_max' not in _reweight_parameters and len(bins) > 3:
                _reweight_parameters['sp_cutoff_max'] = 2 * bins[-2] - bins[-3]
            resultA = pathensemble.reweight('A', **_reweight_parameters)
            resultB = pathensemble.reweight('B', **_reweight_parameters)
            wA, xPA, extremesA = resultA[0], resultA[4], resultA[6]
            wB, xPB, extremesB = resultB[0], resultB[4], resultB[6]
            pathensemble.weights = (wA + wB) * pathensemble.are_excursions
            # bonus track: estimate rates
            kAB = np.sum(wA * pathensemble.internal_lengths)
            if kAB:
                kAB = 1 / kAB
            else:
                kAB = np.nan
            kBA = np.sum(wB * pathensemble.internal_lengths)
            if kBA:
                kBA = 1 / kBA
            else:
                kBA = np.nan
            print(f'    kAB estimate: {kAB:.3e} [1/dt]')
            print(f'    kBA estimate: {kBA:.3e} [1/dt]')

            # put this in global variables to be read by the convergence function
            global kAB_current_estimate, kBA_current_estimate
            kAB_current_estimate = kAB
            kBA_current_estimate = kBA
            
            # rescale committor: determine params
            if rescale_committor:
                print(f'\nRescaling committor to match '
                      f'the crossing probability ({now()})')
                
                # process crossing probability
                if len(xPA):
                    xPA /= xPA[-1] * expit(extremesA)[-1]
                else:
                    xPA = np.zeros(1)
                    extremesA = np.array([-np.inf])
                if len(xPB):
                    xPB /= xPB[-1] * expit(extremesB)[-1]
                else:
                    xPB = np.zeros(1)
                    extremesB = np.array([+np.inf])
                
                # determine domain of action
                if xPA[0]:
                    vmax = xPA[0]
                else:
                    vmax = 1.0
                if xPB[0]:
                    vmin = 4 / xPB[0]
                else:
                    vmin = 1.0
                
                # turn into fine grained interpolation in logit committor space
                q = np.arange(-100.00, +100.01, .01)
                
                # from A
                xp = np.maximum(extremesA, -200.00)
                fp = xPA
                if np.min(xp) > -100:
                    xp = np.insert(xp, 0, -100.00)
                    fp = np.insert(fp, 0, xPA[0])
                if np.max(xp[xp < np.inf]) < 100:
                    xp = np.insert(xp, -1, 100.00)
                    fp = np.insert(fp, -1, 1.00)  # in very good approximation
                # the space where interpolation is linear: log committor-log
                xp = np.log(expit(xp))
                # new interpolated version
                xPA = np.exp(np.interp(np.log(expit(q)), xp, np.log(fp)))
                
                # from B
                xp = np.maximum(extremesB, -200.00)
                fp = xPB
                if np.min(xp) > -100:
                    xp = np.insert(xp, 0, -100.00)
                    fp = np.insert(fp, 0, xPB[0])
                if np.max(xp[xp < np.inf]) < 100:
                    xp = np.insert(xp, -1, 100.00)
                    fp = np.insert(fp, -1, 1.00)  # in very good approximation
                # the space where interpolation is linear: log committor-log
                xp = np.log(expit(xp))
                # new interpolated version
                xPB = np.exp(np.interp(np.log(expit(q)), xp, np.log(fp)))[::-1]
                
                # TS shift and rescaling computation
                ts = 0.  # initialization
                r = 1.
                from_A_wins = xPA >= xPB
                from_B_wins = xPA <  xPB
                if xPA[0] > 2 and xPB[-1] > 2 and\
                    np.sum(from_A_wins) and np.sum(from_B_wins):
                    ts = (q[from_A_wins][-1] + q[from_B_wins][0]) / 2
                    r = 2. / xPA[from_A_wins][-1]
                elif xPA[0] > 2 and np.sum(from_A_wins):
                    ts = q[np.argmin(np.abs(xPA / 2 - 1.))]
                elif xPB[0] > 2 and np.sum(from_B_wins):
                    ts = np.clip(q[np.argmin(np.abs(xPB / 2 - 1.))], -8., 8.)
                print(f'*** transition state shift by {ts:.3f}, '
                      f'total xP rescaling by {r:.3f}')
                                
                # theoretical line
                y = np.zeros(len(q))
                y[q <= ts] = 1 / expit(q[q <= ts] - ts)
                y[q >  ts] = 4 * (1 - expit(q[q > ts] - ts))
                
                # actual line
                y0 = np.append(r * xPA[q <= ts], 4 / xPB[q > ts] / r)
                
                # determine number of knots / values
                vmin /= r
                vmax *= r
                if xPA[0] and xPB[-1]:
                    drop = xPA[0] * xPB[-1] * r ** 2
                elif xPA[0]:
                    drop = xPA[0]
                elif xPB[0]:
                    drop = xPB[-1]
                else:
                    drop = 1.0
                try:
                    n = np.clip(round(np.log(drop)), 0, 100)
                except:
                    n = 0
                if not n:
                    print(f'!!! rescaling is not possible (yet)')
                else:
                    print(f'*** generating {n} knots')
                
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
                
                for knot, value in zip(knots, values):
                    print(f'    {knot:+7.3f} -> {value:+7.3f}')
                
                # load interpolation in network
                network.rescale_knots[:len(knots)] = torch.from_numpy(knots)
                network.rescale_values[:len(knots)] = torch.from_numpy(values)
                
                # directly rescale values
                for p in pathensemble.pathensembles:
                    p.frame_values[:] = rescale(p.frame_values, knots, values)
                
                # rescaled bins
                bins = get_bins(pathensemble, nbins,
                    cutoff_max=cutoff_max,
                    initial_paths=initial_paths,
                    states=include_marginal_bins)
                print(f'    rescaled bins: {array2string(bins, 18)}')
        else:  # only TPS weights
            equilibriumA.weights = 0.
            equilibriumB.weights = 0.
        
        print(f'\nProjecting the {"T" if do_tps else ""}PE density ({now()})')
        densities = pathensemble.project(bins)
        if not hasattr(densities, '__len__'):
            densities = np.zeros(len(bins) - 1)
            densities[+0] += 1.
            densities[-1] += 1.
        densities[densities == 0.] = 1e-9
        densities /= np.sum(densities)
        print(f'    densities: {array2string(densities, 14)}')
        
        # save
        fname = f'network.h5'
        print(f'\nSaving network parameters to {directory}/{fname} ({now()})')
        torch.save(network.state_dict(), f'{directory}/{fname}')
        np.save(f'{directory}/bins.npy', bins)
        np.save(f'{directory}/densities.npy', densities)
        
        # update counts and sleep
        counts += 1
        time.sleep(1)
    
    return pathensemble

def kinetics_convergence(self, log_file=None, chunks = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                         kinetics_outfile='kinetics_convergence.txt', kinetics_convergence_plotfile='kinetics_convergence.pdf'):
    """ Perform convergence analysis of kinetics on the current path ensemble.
    This works by taking the first x fraction of the paths in the path ensemble,
    training a network for those, and computing kinetics. Requires that
    in the params reweight_pathensemble_after_training=True and
    sparse_update_max_frames=-1 (otherwise no kinetics calculation can be performed.)
    
    Parameters
    ----------
    log_file : str or None
        File to log output to. If None, logs to stdout.
    chunks : list of float
        Fractions of the path ensemble to consider for convergence analysis.
    kinetics_outfile : str
        File to save kinetics convergence data to.
    kinetics_convergence_plotfile : str
        File to save kinetics convergence plot to.

    Returns
    -------
    None
    """

    kAB_estimates = []
    kBA_estimates = []

    for fraction in chunks:
        print(f'\n=== Kinetics convergence analysis: fraction {fraction} ===\n')
        
        # train on fraction of path ensemble
        train(self, log_file=log_file, nrounds=1, walltime=inf,
                   pathensemble_fraction=fraction)
        
        # read current estimates from global variables
        global kAB_current_estimate, kBA_current_estimate
        kAB_estimates.append(kAB_current_estimate)
        kBA_estimates.append(kBA_current_estimate)
        
        print(f'Kinetics estimates at fraction {fraction}: '
              f'kAB = {kAB_current_estimate}, kBA = {kBA_current_estimate}\n')
    
    # save results to file
    with open(kinetics_outfile, 'w') as f:
        f.write('# fraction kAB[1/dt] kBA[1/dt]\n')
        for fraction, kAB, kBA in zip(chunks, kAB_estimates, kBA_estimates):
            f.write(f'{fraction} {kAB} {kBA}\n')
    print(f'Kinetics convergence data saved to {kinetics_outfile}')
    # generate convergence plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(chunks, kAB_estimates, marker='o', label='kAB')
        plt.plot(chunks, kBA_estimates, marker='o', label='kBA')
        plt.xlabel('Run progression / fraction of paths')
        plt.yscale('log')
        plt.ylabel('Kinetics Estimates [1/dt]')
        plt.title('Kinetics Convergence Analysis')
        plt.legend()
        plt.savefig(kinetics_convergence_plotfile)
        plt.close()
        print(f'Kinetics convergence plot saved to {kinetics_convergence_plotfile}')
    except ImportError:
        print('matplotlib not available, skipping convergence plot generation.')