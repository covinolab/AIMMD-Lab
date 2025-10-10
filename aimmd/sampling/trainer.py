from utils import *

directory = sys.argv[1]
params = sys.argv[2]
if len(sys.argv) > 2:
    verbose = True
else:
    verbose = False

# get aimmd run parameters
write(f'\nLoading AIMMD run parameters ({now()})')
aimmd_run_params = import_aimmd_run_params(params)

# extract necessary parameters (in order of appearance)
topology = aimmd_run_params['topology']
states_function = aimmd_run_params['states_function']
descriptors_function = aimmd_run_params['descriptors_function']
values_function = aimmd_run_params['values_function']
network = aimmd_run_params['network']
extra_equilibriumA = aimmd_run_params['extra_equilibriumA']
extra_equilibriumB = aimmd_run_params['extra_equilibriumB']
extra_equilibriumA_states_map = \
    aimmd_run_params['extra_equilibriumA_states_map']
extra_equilibriumB_states_map = \
    aimmd_run_params['extra_equilibriumB_states_map']
fit = aimmd_run_params['fit']
nbins = aimmd_run_params['nbins']
cutoff_max = aimmd_run_params['cutoff_max']
rescale_committor = aimmd_run_params['rescale_committor']
include_marginal_bins = aimmd_run_params['include_marginal_bins']
reweight_parameters = aimmd_run_params['reweight_parameters']
do_tps = aimmd_run_params['do_tps']

write(f'\nLoading initial path(s) ({now()})')
initial_path = load_initial_path(directory, topology,
    states_function, descriptors_function, values_function)
assert initial_path.nframes
write(f'    {initial_path}')

# load the network if it is already possible
load_network_and_projections(network, directory, wait=False)

while True:    
    write(f'\nLoading most recent path ensemble ({now()})')
    pathensemble, added_nframes = update_pathensemble(directory, topology,
        states_function, descriptors_function, values_function,
        add_missing_paths=False, add_missing_frames=False, verbose=True)
    shooting_chains, equilibriumA, equilibriumB = scorporate_pathensembles(
        pathensemble)

    # extra equilibriumA and equilibriumB
    if len(extra_equilibriumA):
        write(f'\nLoading extra free simulations around A ({now()})')
        equilibriumA += update_pathensemble(directory, topology,
            states_function, descriptors_function, values_function,
            add_missing_paths=False, add_missing_frames=False,
        shooting_chains=[], equilibriumA=extra_equilibriumA, equilibriumB=[],
            equilibriumA_states_map=extra_equilibriumA_states_map,
            verbose=True)[0]
    if len(extra_equilibriumB):
        write(f'\nLoading extra free simulations around B ({now()})')
        equilibriumB += update_pathensemble(directory, topology,
            states_function, descriptors_function, values_function,
            add_missing_paths=False, add_missing_frames=False,
        shooting_chains=[], equilibriumA=[], equilibriumB=extra_equilibriumB,
            equilibriumB_states_map=extra_equilibriumB_states_map,
            verbose=True)[0]
    pathensemble = shooting_chains + equilibriumA + equilibriumB
    
    write(f'\nTraining the network ({now()})')
    losses, *_ = fit(network, pathensemble, initial_path, verbose=verbose)
    if not len(losses):
        if 'network.h5' in os.listdir(directory):
            write('!!! reloaded most recent network because training failed')
            bins, densities = load_network_and_projections(network, directory)
        else:
            write('!!! training failed, retrying')
            continue
    
    write(f'\nUpdating the path ensemble values ({now()})')
    pathensemble.update_values()
    initial_path.update_values()
    
    write(f'\nObtaining the adaptation bins ({now()})')
    bins = get_bins(pathensemble, nbins,
        cutoff_max=cutoff_max,
        initial_path=initial_path,
        states=include_marginal_bins)
    write(f'    bins: {array2string(bins, 9)}')
    
    write(f'\nReweighting the full path ensemble ({now()})')
    if not do_tps:
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
        kAB = np.nan_to_num(1 / np.sum(wA * pathensemble.internal_lengths))
        kBA = np.nan_to_num(1 / np.sum(wB * pathensemble.internal_lengths))
        write(f'    kAB estimate: {kAB:.3e} [1/dt]')
        write(f'    kBA estimate: {kBA:.3e} [1/dt]')
        
        # rescale committor: determine params
        if rescale_committor:
            write(f'\nRescaling committor to match '
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
            write(f'*** transition state shift by {ts:.3f}, '
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
                write(f'!!! rescaling is not possible (yet)')
            else:
                write(f'*** generating {n} knots')
            
            # fill
            knots = np.zeros(n)
            values = np.zeros(n)
            for i, v in enumerate(np.geomspace(vmin, vmax, n + 2)[::-1][1:-1]):
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
                write(f'    {knot:+7.3f} -> {value:+7.3f}')
            
            # load interpolation in network
            network.rescale_knots[:len(knots)] = torch.from_numpy(knots)
            network.rescale_values[:len(knots)] = torch.from_numpy(values)
            
            # directly rescale values
            for p in pathensemble.pathensembles:
                p.frame_values[:] = rescale(p.frame_values, knots, values)
    
            # rescaled bins
            bins = get_bins(pathensemble, nbins,
                cutoff_max=cutoff_max,
                initial_path=initial_path,
                states=include_marginal_bins)
            write(f'    rescaled bins: {array2string(bins, 18)}')
    else:  # only TPS weights
        equilibriumA.weights = 0.
        equilibriumB.weights = 0.
    
    write(f'\nProjecting the {"T" if do_tps else ""}PE density ({now()})')
    densities = pathensemble.project(bins)
    if not hasattr(densities, '__len__'):
        densities = np.zeros(len(bins) - 1)
        densities[+0] += 1.
        densities[-1] += 1.
    densities[densities == 0.] = 1e-9
    densities /= np.sum(densities)
    write(f'    densities: {array2string(densities, 14)}')
    
    # save
    fname = f'network.h5'
    write(f'\nSaving network parameters to {fname} ({now()})')
    torch.save(network.state_dict(), f'{directory}/{fname}')
    np.save(f'{directory}/bins.npy', bins)
    np.save(f'{directory}/densities.npy', densities)
    sleep(1)
