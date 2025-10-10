import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def rsync_from_juwels(
    *runs,
    cluster_directory='scratch',
    current_directory='.',
    manager=True,
    trainer=True,
    network=True,
    pathensembles=True,
    trajectories=False,
    energy_files=False,
    include_all=False):
    
    for run in runs:
        command = f'rsync -av --prune-empty-dirs '
        command += f"--include='/{run}/' "
        command += f"--include='/{run}/*xtc' "  # initial paths
        command += f"--include='/{run}/*trr' "  # initial paths
        command += f"--include='/{run}/*/' "
        if manager or include_all:
            command += f"--include='/{run}/manager.log' "
        if trainer or include_all:
            command += f"--include='/{run}/trainer.log' "
        if network or include_all:
            command += f"--include='/{run}/network.h5' "
        if pathensembles or include_all:
            command += f"--include='/{run}/*/*.h5' "
        if trajectories or include_all:
            command += f"--include='/{run}/*/*xtc' "
            command += f"--include='/{run}/*/*trr' "
        if energy_files or include_all:
            command += f"--include='/{run}/*/*edr' "
        if include_all:  # some logs are missing
            command += f"--include='/{run}/*log' "
            command += f"--include='/{run}/*/*log' "
        command += f"--exclude='*' "
        command += f"--exclude='*/.*' "  # exclude hidden files
        command += f"lazzeri1@juwels-booster.fz-juelich.de:"
        command += f"{cluster_directory}/{run} "
        command += f'{current_directory}'
        os.system(command)


def get_rates_from_log(log_file, dt=1.):
    t = []
    kAB = []
    kBA = []
    
    measure_time = False
    with open(log_file) as f:
        for line in f:
            if 'Loading most recent path ensemble' in line:
                measure_time = True
                T = 0.0
            elif measure_time:
                try:
                    T += float(line.split('individual')[0].split(', ')[1]) * dt
                except:
                    pass
                if 'Training' in line:
                    measure_time = False
            if 'kAB estimate' in line:
                kAB.append(float(line.split('estimate: ')[-1].split()[0]) / dt)
            if 'kBA estimate' in line:
                kBA.append(float(line.split('estimate: ')[-1].split()[0]) / dt)
                t.append(T)
            if len(t) > 1 and t[-2] > t[-1]:
                t = t[:-1]
                kAB = kAB[:-1]
                kBA = kBA[:-1]
    return np.array(t), np.array(kAB), np.array(kBA)


def plot_rates_evolution(times, rates, names,
                         window=20, log_y=True, log_x=False,
                         t0 = 0., t_rescaling=1.):
    """
    Maximum four time series supported.
    """
    plt.figure(figsize=(5,4))
    x = np.linspace(max(t0, np.min(np.concatenate(times))),
                    np.max(np.concatenate(times)), 1001)[1:]
    plt.plot(x, 1/t_rescaling/x, ':', color='black')
    if log_y:
        plt.gca().set_yscale('log')
    if log_x:
        plt.gca().set_xscale('log')
    for x, y, n, c1, c2 in zip(times, rates, names,
                           ['blue', 'red', 'darkgreen', 'darkviolet'],
                           ['dodgerblue', 'tomato', 'lightgreen', 'violet']):
        k = x > t0
        plt.plot(x[k], y[k], '.', color=c2, alpha=.5, zorder=-1)
        y = np.array([np.median(
            y[max(i - window,0):i + window + 1])
            for i in range(len(y))])
        plt.plot(x[k], y[k], color=c1, label=n)
    plt.grid()
    plt.legend()
    plt.xlabel('Total simulated time [dt]')
    plt.ylabel('Rate estimate [1/dt]')
    return plt.gcf()


def analysis(*runs,
             step_number=None,
             params=['params.py'],
             states=['AB'],
             descriptors_condition=None,
             extra_equilibriumA=None,
             extra_equilibriumB=None,
             dt = 1e-4,
             cv_indices=[],
             cv_names=[],
             cv_titles=[],
             cv_bins=None,  # if previously determined, do not recompute
             cv_grid_size=20,
             cv_grid_size_2d=20,
             block=1,
             total_blocks=1,
             bootstrap=False,
             network_params=None,  # otw take network.h5
             retrain=False,
             reweight_parameters=None,  # use params' defauls
             merge=True,
             figs_prefix='.'):
    """
    If changing, e.g., fit function and network architecture,
    then work at the "params" level: create a new params file with the updates.
    
    Attention! It will modify the first descriptor dimension with committor
    value for faster free energy plots.
    """
    
    # process to ensure right length
    n = len(runs)
    if not cv_bins:
        cv_bins = [{}]
    params = (params * n)[:n]
    states = (states * n)[:n]
    if extra_equilibriumA:
        extra_equilibriumA = (extra_equilibriumA * n)[:n]
    if extra_equilibriumB:
        extra_equilibriumB = (extra_equilibriumB * n)[:n]
    i = 0
    l = len(cv_bins)
    while len(cv_bins) < (n + merge):
        cv_bins.append(cv_bins[i % l].copy())
        i += 1
    
    # initialize output
    aimmd_params = []
    pathensembles = []
    shooting_chains = []
    equilibriumA = []
    equilibriumB = []
    steps = []
    keys = []  # if not bootstrapping just a range
    extendA = []  # info of reached state !works with just one worker!
    extendB = []
    weights = []
    k12 = []
    k21 = []
    cv_rho = []

    # for cropping
    tmin = -np.inf
    tmax = +np.inf
    
    # add committor as a CV
    cv_names = ['q'] + list(cv_names)
    cv_indices = [0] + list(cv_indices)
    cv_titles = ['Logit committor'] + list(cv_titles)
    
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    for i in range(len(runs)):
        write(f'\nProcessing {runs[i]} ({now()})')
        
        # import params
        aimmd_params.append(import_aimmd_run_params(params[i]))
        if params[i] in sys.modules:
            del sys.modules[params[i]]
        
        # get info out of params
        device = next(aimmd_params[-1]['network'].parameters()).device
        topology = aimmd_params[-1]['topology']
        directory = '/'.join(params[i].split('/')[:-1])
        if directory:
            topology = f'{directory}/{topology}'
        states_function = aimmd_params[-1]['states_function']
        descriptors_function = aimmd_params[-1]['descriptors_function']
        values_function = aimmd_params[-1]['values_function']
        network = aimmd_params[-1]['network']
        fit = aimmd_params[-1]['fit']
        rescale_committor = aimmd_params[-1]['rescale_committor']
        nbins = aimmd_params[-1]['nbins']
        include_marginal_bins = aimmd_params[-1]['include_marginal_bins']
        if reweight_parameters is None:
            reweight_params = aimmd_params[-1]['reweight_parameters']
        save_interval = aimmd_params[-1]['save_interval']
        
        # get extra equilibrium info from input args or params
        if extra_equilibriumA is None:
            extraA = aimmd_params[-1]['extra_equilibriumA']
        else:
            extraA = extra_equilibriumA[i]
        if extra_equilibriumB is None:
            extraB = aimmd_params[-1]['extra_equilibriumB']
        else:
            extraB = extra_equilibriumB[i]
        extraA_states_map = aimmd_params[-1][
            'extra_equilibriumA_states_map']
        extraB_states_map = aimmd_params[-1][
            'extra_equilibriumB_states_map']

        # initialize (will load only if necessary)
        initial_path = None
        
        write(f'\nLoading most recent path ensemble ({now()})')
        pathensemble, added_nframes = update_pathensemble(runs[i], topology,
            states_function, descriptors_function, values_function,
            add_missing_paths=False, add_missing_frames=False, verbose=False)
        chains, eqA, eqB = scorporate_pathensembles(pathensemble)
        
        # extra equilibriumA
        if extraA:
            write(f'\nLoading extra free simulations around A ({now()})')
            eqA += update_pathensemble(runs[i], topology,
                states_function, descriptors_function, values_function,
                add_missing_paths=False, add_missing_frames=False,
            shooting_chains=[], equilibriumA=extraA, equilibriumB=[],
                equilibriumA_states_map=extraA_states_map,
                verbose=False)[0]
        
        # extra equilibrium B
        if extraB:
            write(f'\nLoading extra free simulations around B ({now()})')
            eqB += update_pathensemble(runs[i], topology,
                states_function, descriptors_function, values_function,
                add_missing_paths=False, add_missing_frames=False,
            shooting_chains=[], equilibriumA=[], equilibriumB=extraB,
                equilibriumB_states_map=extraB_states_map,
                verbose=False)[0]
        
        # crop?
        completion_times = np.sort(chains.completion_times)
        nS = len(completion_times)
        if total_blocks > 1:
            if tmin == -np.inf or states[0] == states[1]:
                tmin = completion_times[
                    round(nS / total_blocks * block)]
            if tmax == +np.inf or states[0] == states[1]:
                tmax = completion_times[
                    round(nS / total_blocks * (block + 1)) - 1]
        if step_number:
            if tmax == +np.inf or states[0] == states[1]:
                tmax = completion_times[
                    round(nS / total_blocks * (block + 1)) - 1]
                
        if tmax < +np.inf or tmin > -np.inf: 
            for j, p in enumerate(chains.pathensembles):
                chains.pathensembles[j] = p[
                    (p.completion_times >= tmin) *
                    (p.completion_times <= tmax)]
            eqA = eqA.crop(tmin=tmin, tmax=tmax)
            eqB = eqB.crop(tmin=tmin, tmax=tmax)
        
        steps.append(len(chains))  # so that faster

        # special extract
        if descriptors_condition is not None:
            chains = chains[
                descriptors_condition(chains.shooting_descriptors)]
            eqA = eqA[descriptors_condition(eqA.shooting_descriptors)]
            eqB = eqB[descriptors_condition(eqB.shooting_descriptors)]
        
        # all together
        pathensemble = chains + eqA + eqB
        pathensembles.append(pathensemble)
        shooting_chains.append(chains)
        equilibriumA.append(eqA)
        equilibriumB.append(eqB)
        
        # report
        nS = len(chains)
        nA = len(eqA)
        nB = len(eqB)
        tS = chains.nframes * dt
        tA = eqA.nframes * dt
        tB = eqB.nframes * dt
        nT = np.sum(chains.are_transitions)
        pT = pT = nT / len(chains)
        o = np.argsort(pathensemble.completion_times)
        t = np.where(pathensemble.are_transitions)[0]
        print(f'\n*** {states[i][0]} <-> {states[i][-1]} ***' +
              f'\n    {nS} shooting paths, simulated {tS:.03f} us'
              f'\n    {nA} equil. A trajs, simulated {tA:.03f} us'
              f'\n    {nB} equil. B trajs, simulated {tB:.03f} us'
              f'\n    {tS / (tA + tB + 1e-15) * 2:.02f} shooting / equil. balance'
              f'\n    {nT} shooting transitions ({pT * 100:.01f}%), '
              f'{len(t)} in total:\n'
              f'    ... {t[np.argsort(o[t])[-10:]]} (in order)')
        
        # plot transition production rate
        o = np.argsort(chains.completion_times)
        l = len(o)
        cumul_transitions = np.cumsum(chains.are_transitions[o])
        plt.figure(figsize=(3.25,2.5))
        plt.plot(cumul_transitions)
        plt.plot(np.arange(l), np.arange(l)*.33, label='33%')
        plt.plot(np.arange(l), np.arange(l)*.25, label='25%')
        plt.plot(np.arange(l), np.arange(l)*.15, label='15%')
        plt.plot(np.arange(l), np.arange(l)*.10, label='10%')
        plt.title(f'{states[i][0]} $\\rightleftharpoons$ {states[i][1]} '
                  f'transitions production')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if figs_prefix is not None:
            plt.savefig(f'{figs_prefix}{states[i]}_transitions_production.pdf')
        plt.close()
        
        if bootstrap:
            k = np.concatenate([
                np.random.choice(np.arange(nS), nS),
                np.random.choice(np.arange(nA), nA) + nS,
                np.random.choice(np.arange(nB), nB) + nS + nA])
        else:
            k = np.arange(nS + nA + nB)
        keys.append(k)
        
        # trajectory extensions
        print('\nTrajectory extension')
        
        # extend A
        temp = update_pathensemble(runs[i],
            shooting_chains='',
            equilibriumA=['extendA'],
            equilibriumB=[], verbose=False)[0]
        if tmax < +np.inf or tmin > -np.inf:
            temp = temp.crop(tmin=tmin, tmax=tmax)
        h = np.where((temp.initial_states == 'A') *
                     (temp.final_states != 'A') *
                     (temp.final_states != 'R'))[0]
        
        if not bootstrap:
            extendA.append(temp.final_states[h])
        else:
            extendA.append(np.random.choice(temp.final_states[h], len(h)))
        
        # extend B
        temp = update_pathensemble(runs[i],
            shooting_chains='',
            equilibriumA=[],
            equilibriumB=['extendB'], verbose=False)[0]
        if tmax < +np.inf or tmin > -np.inf:
            temp = temp.crop(tmin=tmin, tmax=tmax)
        h = np.where((temp.initial_states == 'B') *
                     (temp.final_states != 'B') *
                     (temp.final_states != 'R'))[0]
        
        if not bootstrap:
            extendB.append(temp.final_states[h])
        else:
            extendB.append(np.random.choice(temp.final_states[h], len(h)))
        
        print(f'    extendA: {extendA[-1]}')
        print(f'    extendB: {extendB[-1]}')
        
        # fit?
        if not retrain:
            if network_params:
                f = network_params
            elif step_number:
                s = max((steps[i] // save_interval), 1) * save_interval
                f = f'{runs[i]}/network{s:06g}.h5'
            else:
                f = f'{runs[i]}/network.h5'
            state_dict = torch.load(f, map_location=device)
            network.load_state_dict(state_dict)
        else:
            write(f'\nTraining the neural network model ({now()})')
            try:
                losses, scales, values, weights = fit(
                    network, pathensemble, initial_path=initial_path,
                    verbose=True, keys=keys[i], save_memory=True)
            except:
                write(f'\nLoading initial path(s) ({now()})')
                initial_path = load_initial_path(runs[i], topology,
                states_function,
                        descriptors_function, values_function, verbose=False)
                losses, scales, values, weights = fit(
                    network, pathensemble, initial_path=initial_path,
                    verbose=True, keys=keys[i], save_memory=True)
        
        write(f'\nUpdating frame values ({now()})')
        pathensemble.update_values(verbose=False)
    
        # rescale committor
        if retrain and rescale_committor:
            # first reweighting
            bins = get_bins(pathensemble, nbins,
                            states=include_marginal_bins)
            bins_centers = (bins[1:] + bins[:-1]) / 2
            bins_centers[+0] = bins_centers[+1] - (bins[+2] - bins[+1])
            bins_centers[-1] = bins_centers[-2] + (bins[-2] - bins[-3])
            reweight_params['sp_cutoff_min'] = 2 * bins[+1] - bins[+2]
            reweight_params['sp_cutoff_max'] = 2 * bins[-2] - bins[-3]
            resultA = pathensemble.reweight('A', **reweight_params)
            resultB = pathensemble.reweight('B', **reweight_params)
            wA, xPA, extremesA = resultA[0], resultA[4], resultA[6]
            wB, xPB, extremesB = resultB[0], resultB[4], resultB[6]
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
            for I, v in enumerate(np.geomspace(vmin, vmax, n + 2)[::-1][1:-1]):
                knots[I] = q[np.argmin(np.abs(v - y0))]
                values[I] = q[np.argmin(np.abs(v - y))]
            
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
    
        # get bins and populations and plot
        try:
            bins = get_bins(pathensemble, nbins, initial_path=initial_path,
                            states=include_marginal_bins)
        except:
            write(f'\nLoading initial path(s) ({now()})')
            initial_path = load_initial_path(runs[i], topology, states_function,
                    descriptors_function, values_function, verbose=False)
            bins = get_bins(pathensemble, nbins, initial_path=initial_path,
                            states=include_marginal_bins)
        bins_centers = (bins[1:] + bins[:-1]) / 2
        bins_centers[+0] = bins_centers[+1] - (bins[+2] - bins[+1])
        bins_centers[-1] = bins_centers[-2] + (bins[-2] - bins[-3])
        sp_populations = np.histogram(chains.shooting_values, bins)[0]
        plt.figure(figsize=(3.25,2.5))
        plt.plot(bins_centers, sp_populations, 'o', color='purple', label='SPs')
        print('sp populations: ', sp_populations)
        plt.xlim(bins_centers[0], bins_centers[-1])
        plt.gca().set_yscale('log')
        plt.title(f'{states[i][0]} $\\rightleftharpoons$ {states[i][-1]}'
                  f' SP selection')
        plt.grid()
        plt.xlabel('Logit committor model')
        plt.ylabel('SP population')
        plt.tight_layout()
        plt.legend()
        if figs_prefix is not None:
            plt.savefig(f'{figs_prefix}'
                    f'{states[i]}_step_{steps[i]}_SP_popul.pdf')
        plt.close()
    
        # reweight
        weights.append([])
        reweight_params['sp_cutoff_min'] = 2 * bins[+1] - bins[+2]
        reweight_params['sp_cutoff_max'] = 2 * bins[-2] - bins[-3]
        for state, _states in zip(['A', 'B'], [states[i], states[i][::-1]]):
            (w, path_indices, internal_segments, excursions,
             xP, m, extremes,
             shooting_values, factors) = pathensemble.reweight(
             state, **reweight_params)
    
            # convert weights
            www = np.zeros(len(pathensembles[i]))
            for k, ww in zip(keys[i], w):
                    www[k] += ww
            weights[i].append(www)

            if not len(xP):
                continue  # do not bother
            
            # eq. crossing probability
            e = extremes[np.isinf(shooting_values)]
            if not len(e):
                e = np.array([-np.inf])
            xP0 = np.arange(len(e), 0, -1.0)
            xP0 /= xP0[0] * xP[-1]
            xP0[e == np.max(e)] = np.max(xP0[e == np.max(e)])
            
            # crossing statistics plot
            m /= m[0]
            m /= xP[-1] * expit(extremes[-1])
            xP /= xP[-1] * expit(extremes[-1])
            xP0 /= xP[-1] * expit(extremes[-1])
            plt.figure(figsize=(3.25,2.5))
            x = np.geomspace(1/(xP[0]/xP[-1]), 1, 1001)
            plt.plot(x, 1/x, ':', color='black', label='theor.')
            plt.plot(expit(extremes), xP, color='red', label='xP', zorder=4)
            plt.plot(expit(e), xP0, color='blue', label='eq.', lw=2.5, zorder=3)
            plt.plot(expit(extremes), m, color='orange', label='m')
            plt.gca().set_xscale('log')
            plt.gca().set_yscale('log')
            plt.grid()
            plt.xlabel(f'Committor model {_states[0]} $\\rightarrow$ {_states[1]}')
            plt.ylabel(f'Crossing probability')
            plt.tight_layout()
            xlim = list(plt.gca().get_xlim())
            xlim[0] = max(xlim[0], x[0] / 10)
            plt.xlim(*xlim)
            plt.legend()
            if figs_prefix is not None:
                plt.savefig(f'{figs_prefix}{states[i]}_step_{steps[i]}'
                        f'_xP_{_states[0]}{_states[1]}.pdf')
            plt.close()

    # merged pathensemble and its weights
    if merge:
        states.append(f'{states[0][0]}{states[1][1]}')
        print(f'\nMerged {states[-1][0]} <-> {states[-1][1]} '
               f'pathensemble ({now()})')
        
        # definitive object
        pathensemble = PathEnsemblesCollection()
        chains = PathEnsemblesCollection()
        eqA = PathEnsemblesCollection()
        eqB = PathEnsemblesCollection()
        keys.append(np.zeros(0, dtype=int))
        steps.append(0)
        for i in range(len(runs)):
            keys[-1] = np.append(keys[-1], keys[i] + len(pathensemble))
            pathensemble += pathensembles[i]
            chains += shooting_chains[i]
            eqA += equilibriumA[i]
            eqB += equilibriumB[i]
            steps[-1] += steps[i]
        pathensembles.append(pathensemble)
        shooting_chains.append(chains)
        equilibriumA.append(eqA)
        equilibriumB.append(eqB)    
        i += 1
        
        # combine weights together preserving detailed balance
        if states[0][1] == states[1][0] == 'B' and states[2] == 'AC':
            # two-step transition
            pathensembles[0].weights = weights[0][1]
            C = pathensembles[0].project()[0]
            pathensembles[1].weights = weights[1][0]
            C /= pathensembles[1].project()[0]
            weights.append([
                np.append(weights[0][0], weights[1][0] * C / 2),
                np.append(weights[0][1] / 2, weights[1][1] * C)])
        else:
            
            # many equivalent runs merged together
            weights.append([
                np.concatenate([weights[0] for weights in weights]),
                np.concatenate([weights[1] for weights in weights])])
    
    # rates and projections
    for i in range(len(runs) + merge):
        if i < len(runs):
            print(f'\nRates and projections for {runs[i]} ({now()})')
        else:
            print(f'\nRates and projections for merged pathensemble ({now()})')
    
            # update values according to last available model
            if states[0][-1] == states[i][-1]:
                print(f'    re-evaluate committor values on merged pathesemble')
                pathensembles[i].update_values(verbose=False)
            else:  # shift and consider the good ones...
                v0 = np.mean(np.append(
                    equilibriumB[0].initial_values[
                    equilibriumB[0].initial_values == 'B'],
                    equilibriumB[0].final_values[
                    equilibriumB[0].final_states == 'B']))
                v1 = np.mean(np.append(
                    equilibriumA[1].initial_values[
                    equilibriumA[1].internal_states == 'A'],
                    equilibriumA[1].final_values[
                    equilibriumA[1].internal_states == 'A']))
                shift = v0 - v1
                print(f'    second pathensemble values shift by {shift:.3f}')
                for p in pathensembles[1].pathensembles:
                    p.frame_values += shift
                
                # only one model in one region
                for p1, p2 in zip(
                    equilibriumB[0].pathensembles,
                    equilibriumA[1].pathensembles):
    
                    # towards first part: first leads
                    p2.frame_values[p2.frame_states == 'A'] = \
                        p1.frame_values[p1.frame_states == 'R']
                    
                    # towards second part: second leads
                    p1.frame_values[p1.frame_states == 'B'] = \
                        p2.frame_values[p2.frame_states == 'R']
        
        # standard way of computing rates
        if i < len(runs) or not (
            states[0][1] == states[1][0] and states[0][0] != states[1][1]):
            pathensembles[i].weights = weights[i][0]
            k12.append(1 / (pathensembles[i].project()[0] * dt))
            pathensembles[i].weights = weights[i][1]
            k21.append(1 / (pathensembles[i].project()[0] * dt))
            print(f'    k{states[i]      } = {k12[i]:.3e} [1/us]')
            print(f'    k{states[i][::-1]} = {k21[i]:.3e} [1/us]')
    
        # total AC rates based on extension trajectories
        else:  #  (most sensible estimate)
            s = np.array(extendB[0])
            norm = np.sum(s == 'A') + np.sum(s == 'C')
            if norm:
                eta = np.sum(s == 'C') / norm
            else:
                eta = 0
            k12.append(k12[0] * eta)
            k21.append(k21[1] * k21[0])  # out of equilibrium
            print(f'    eta = {eta:.3f}')
            print(f'    k{states[i]} = {k12[i]:.3e} [1/us]')
        
        # project (determine CV bins)
        cv_rho.append({})
    
        # assign weights
        pathensembles[i].weights += weights[i][0] + weights[i][1]
        
        # utilities: change descriptors for values
        for p in pathensembles[i].pathensembles:
            if not p.nframes:
                continue
            descriptors = p.frame_descriptors
            descriptors[:, 0] = p.frame_values
            p.frame_descriptors = descriptors
        
        # cvs projections
        for cv_name, cv_title, cv_index in zip(
            cv_names, cv_titles, cv_indices):
            fname = (f'{figs_prefix}{states[i]}_step_{steps[i]}'
                     f'_free_{cv_name}.pdf')
            if figs_prefix is not None:
                print('   ', fname)
            if cv_name not in cv_bins[i]:
                v = pathensembles[i].frame_descriptors[:, cv_index]
                v0 = v.min()
                v1 = v.max()
                v2 = (v1 - v0) * .005
                x = np.linspace(v0 - v2, v1 + v2, cv_grid_size + 1)
                cv_bins[i][cv_name] = x
            x = cv_bins[i][cv_name]
            y = (x[:-1] + x[1:]) / 2
            z = pathensembles[i].project(x, f=lambda x:x[:, cv_index])
            z /= np.sum(z)
            cv_rho[i][cv_name] = z
            z = -np.log(z)
            plt.figure(figsize=(3.25,2.5))
            plt.plot(y, z)
            plt.grid()
            plt.xlabel(cv_title)
            plt.ylabel('Free energy [$k_BT$]')
            plt.title(f'{states[i][0]} $\\rightleftharpoons$ {states[i][1]} free energy')
            plt.tight_layout()
            if figs_prefix is not None:
                plt.savefig(fname)
            plt.close()
        
        # 2D projections
        for i1 in range(len(cv_indices) - 1):
            for i2 in range(i1 + 1, len(cv_indices)):
                fname = (f'{figs_prefix}{states[i]}_step_{steps[i]}'
                         f'_free_{cv_names[i1]}_{cv_names[i2]}.pdf')
                if figs_prefix is not None:
                    print('   ', fname)
                name = f'{cv_names[i1]}_{cv_names[i2]}'
                if name not in cv_bins[i]:
                    x1 = np.linspace(
                        cv_bins[i][cv_names[i1]][+0],
                        cv_bins[i][cv_names[i1]][-1],
                        cv_grid_size_2d + 1)
                    x2 = np.linspace(
                        cv_bins[i][cv_names[i2]][+0],
                        cv_bins[i][cv_names[i2]][-1],
                        cv_grid_size_2d + 1)
                    cv_bins[i][name] = (x1, x2)
                x1, x2 = cv_bins[i][name]
                y1 = (x1[:-1] + x1[1:]) / 2
                y2 = (x2[:-1] + x2[1:]) / 2
                z = pathensembles[i].project([x1, x2],
                    f=lambda x:x[:, [cv_indices[i1], cv_indices[i2]]])
                z /= np.sum(z)
                y1, y2 = np.meshgrid(y1, y2)
                cv_rho[i][name] = z
                z = -np.log(z)
                z -= np.min(z)
                plt.figure(figsize=(3.25,2.5))
                plt.contourf(y1, y2, z, zorder=5)
                plt.grid()
                plt.xlabel(cv_titles[i1])
                plt.ylabel(cv_titles[i2])
                plt.title(f'{states[i][0]} $\\rightleftharpoons$ {states[i][1]} free energy [$k_BT$]')
                plt.colorbar()
                plt.tight_layout()
                if figs_prefix is not None:
                    plt.savefig(fname)
                plt.close()
    
    return (pathensembles, aimmd_params,
            shooting_chains, equilibriumA, equilibriumB,
            steps, keys, extendA, extendB,
            weights, k12, k21, cv_bins, cv_rho)



def network_feature_importance(
    network, descriptors, keys=None,
    n_repeats=5, random_state=0):
    """
    Credits: fixed an awful chatGPT initial suggestion
    """
    
    # utils
    def evaluate(network, descriptors):
        device = next(network.parameters()).device
        dtype = next(network.parameters()).dtype
        network.eval()
        
        # initialize
        results = []
        
        # compute in batches
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(
                descriptors, batch_size=4096, shuffle=False):
                batch = batch.to(device=device, dtype=dtype)
                output = network(batch).detach().cpu().numpy().ravel()
                results.append(output)
        
        # return
        if len(results):
            return np.concatenate(results)
        else:
            return np.zeros(0)

    if random_state:
        np.random.seed(random_state)        
    y = evaluate(network, descriptors)
    
    keys = np.arange(descriptors.shape[1])[keys].ravel()
    importances = np.zeros(len(keys))
    n = len(descriptors)
    

    for i, k in tqdm(enumerate(keys), total=len(keys), position=0):
        diffs = []
        for _ in range(n_repeats):
            descriptors1 = descriptors.copy()
            order = np.arange(n)
            np.random.shuffle(order)
            descriptors1[:, k] = descriptors[order, k]
            y1 = evaluate(network, descriptors1)
            diff = np.mean(np.abs(y - y1))
            diffs.append(diff)
        importances[i] = np.mean(diffs)
    
    return importances
