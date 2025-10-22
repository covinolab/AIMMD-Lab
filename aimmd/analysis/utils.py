import os
import numpy as np
import matplotlib.pyplot as plt
from ..core.utils import *


def initialize_plot():
    figure, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    plt.subplots_adjust(left=0.18, bottom=0.18, right=0.99, top=0.8)
    return figure, ax


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
    for split in tqdm(progress):
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
    
    for _ in tqdm(range(bootstrapping), disable=not verbose):
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
    for traj in tqdm(trajs[:4000]):
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


def compute_average_tps_lenghts(tps, dt=1.):
    TP_length = []
    TP_length_max = []
    TP_length_min = []
    lengths = tps.internal_lengths * dt
    weights = tps.weights
    for i in tqdm(range(len(tps))):
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
