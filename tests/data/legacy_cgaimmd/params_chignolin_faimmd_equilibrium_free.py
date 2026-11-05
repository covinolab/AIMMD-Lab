"""Online F-AIMMD on solvated chignolin (CLN025, CHARMM22*/TIP3P, 340 K) with the committor-conditioned BioEmu2 as the
generator, campaign 2026-10-27_chignolin-faimmd-equilibrium-free: committor_bioemu2_aimmd/faimmd/DESIGN_FAIMMD_EQUILIBRIUM_FREE.md.

The 26 Oct sequential campaign's params (faimmd/campaign_chignolin_sequential/params_chignolin_faimmd_sequential.py;
fresh, early-stopped head and fresh trunk adapter every round, nothing carried over) with three changes, all to the
free simulations (PI, 27 Oct), so that the free-B MD samples the unfolded state instead of its boundary:
- initial_paths: one artificial path (faimmd/make_equilibrium_initial_path.py) of 6 equilibrium A frames (Q 0.87-0.92)
  and 6 equilibrium B frames (Q 0-0.04) of the pretrained head's pool (the free trajectories of the Goethe snapshot,
  results/joint/goethe_dataset.npz), full system from the Goethe source run, with one R frame between them from the
  steered pull US_chignolin_gmx_50ns_x3/rep1 (Q 0.55; only there so the file holds a transition; never seeded from);
- free_seeding_position = 'random': the first free trajectory of each state starts at a random frame of that state's
  run of the path (not at the boundary frame);
- free_restart_source = 'equilibrium' (no bias recorded, so the same as 'basin'): every later free trajectory starts
  at a frame drawn uniformly from the in-state frames the free trajectories of its state have accumulated (not at the
  last in-state frame of the previous one, 'crossing').

- network: frozen bioemu2-preview + trunk-embedding adapter + one committor head (bioemu2_committor/aimmd_flow.py);
  initial network = adapter at zero (unmodified BioEmu2) + the 23 Oct campaign's head pretrained on unbiased in-state
  frames only (faimmd/pretrain_flow_network.py): the chains' network until the first round, and the source of the
  head's frozen z-score scaler;
- flow atoms = the 166 protein atoms, whole, Angstrom (FlowDescriptors), what the network scores and generates;
- hooks: TopologyValidity (bond graph, chirality, clashes against unbiased free-MD frames), GromacsSolvator (back into
  run.gro's periodic system with its exact composition: solvate, genion, restrained EM, 20 ps restrained NVT at 340 K),
  GromacsMinimizer (restrained EM of the candidate that is shot). GROMACS gets -ntomp (JUPITER's psslurm rule).
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from chignolin_states import N_PROTEIN, states_function  # noqa: E402,F401  (AIMMD reads states_function)

from aimmd.flow.descriptors import FlowAtoms, FlowDescriptors  # noqa: E402
from aimmd.flow.minimize import GromacsMinimizer  # noqa: E402
from aimmd.flow.solvate import GromacsSolvator, NVTStage  # noqa: E402
from aimmd.flow.topology import read_masses  # noqa: E402
from aimmd.flow.validity import TopologyValidity  # noqa: E402

from bioemu2_committor.aimmd_flow import BioEmu2FlowNetwork, values_function_for  # noqa: E402

engine = "gromacs"
chain_type = "flow"
states = "ARB"                       # p_B = probability of unfolding (B) first
trajectory_extension = ".xtc"
topology = "run.gro"
gmx_grompp = "gmx grompp -maxwarn 5"
gmx_mdrun = "gmx mdrun"
gmx_eneconv = ""                     # energies are not used; avoids a machine-specific path
gmx_mdp = "run.mdp"
initial_paths = ["initial_path_equilibrium_AB.xtc"]   # 6 A + 1 R (steered pull, unused) + 6 B equilibrium frames
free_seeding_position = 'random'     # first free trajectory of each state: a random frame of its run
free_restart_source = 'equilibrium'  # later ones: uniform over the state's accumulated in-state frames
initial_network = "pretrained/networkARB.h5"
gen_temperature = 340.0
max_length = 20000                   # frames: 100 ns per two-way path at 5 ps
retry_with_state_definition_glitches = True
reweight_parameters = {"free_threshold": 20}
masses_from_topology = True
round_shooting_index = True
# n_targets 16 x 2 candidates per refill (32 was the first test's): every new network resets the pool, and a trained
# round lands every few minutes, so most of a refill is discarded (test 2473711: ~60 s generation + ~60 s polish per
# 64 candidates)
flow_parameters = {'lambda_tolerance': 1.0, 'window_after_trained_shots': 1000, 'n_targets': 16,
                   'candidates_per_target': 2, 'generate_chunk': 64, 'generation_seed': 1,
                   'generation_device': 'auto', 'values_device': 'auto', 'training_device': 'auto'}

_THREADS = int(os.environ.get('SLURM_CPUS_PER_TASK', '12'))
_GMX_MDRUN_PREP = f'gmx mdrun -ntmpi 1 -ntomp {_THREADS} -pin off'

_descriptors = FlowDescriptors(os.path.join(_here, topology), FlowAtoms(n_solute=N_PROTEIN),
                               masses=read_masses(os.path.join(_here, 'topol.top')))


def descriptors_function(trajectory):
    """The flat Angstrom coordinates of the 166 protein atoms, whole: (n_frames, 498)."""
    return _descriptors(trajectory)


class Network(BioEmu2FlowNetwork):
    """Defined here so that AIMMD's regenerated params file imports it from this module."""


network = Network(shared=False, cfg_weight=1.0)
# on the worker's GPU where there is one (each srun step sees its own GPU as cuda:0): the flow chains move the network
# themselves, but the trainer fits and evaluates it where it is, and on the CPU its BioEmu2 passes crawl (job 2473611)
import torch  # noqa: E402
if torch.cuda.is_available():
    network = network.to('cuda')
_values = values_function_for(network)


def values_function(descriptors):
    """Per-frame logit committor of the network's EMA head (float64, 1-D). A def here, not the closure itself: AIMMD's
    regenerated params file imports callables by module and name, and a closure is no module attribute."""
    return _values(descriptors)

# the validity check sees only the 166 protein atoms: protein-only topology and structure (systems/chignolin, as
# conditional_generation/prepare.py), whose atom order is run.gro's first 166 atoms; clash reference = every 50th
# frame of unbiased free MD, protein only
_SYSTEM = os.environ.get('BIOEMU2_COMMITTOR_CHIGNOLIN', '/e/project1/rnalig/lichtinger1/data/systems/chignolin')
_validity_descriptors = FlowDescriptors(os.path.join(_SYSTEM, 'conf_A_box.gro'), FlowAtoms(n_solute=N_PROTEIN))
_validity = TopologyValidity(os.path.join(_SYSTEM, 'topol.top'), _validity_descriptors,
                             node_labels=np.load(os.path.join(_here, 'validity_node_labels.npy')),
                             clash_reference=[os.path.join(_here, 'clash_reference_every50th_free_md_frame.xtc')],
                             reflect_mirror_images=False)


# the polish of conditional_generation/prepare.py (the local shooting tests'): every finite sample is minimised in
# vacuum with its heavy atoms restrained (AIMMD-Lab's own validation polish, aimmd.flow.validate._polish, with the
# protein-only GromacsMinimizer), and the polished structure is what is checked and passed on. Raw BioEmu2 samples
# carry near-contacts that the bond-graph check reads as extra bonds (bond_count: ~60 % of raw samples locally,
# ~0 % after the polish).
_POLISH_WORKERS = max(1, _THREADS)
_polisher = GromacsMinimizer(em_mdp=os.path.join(_SYSTEM, 'em.mdp'), restrain='flow_heavy',
                             grompp='gmx grompp -maxwarn 5', mdrun='gmx mdrun -ntmpi 1 -ntomp 1 -nb cpu -pin off',
                             topology=os.path.join(_SYSTEM, 'topol.top'))


class _AtomCount:
    n_atoms = N_PROTEIN                  # GromacsMinimizer asks the network only for n_atoms


def flow_validity_function(positions, context):
    """Validity of the raw sample (recorded), restrained-EM polish, validity of the polished sample (decides)."""
    import types
    import MDAnalysis as mda
    from aimmd.flow import validate
    from aimmd.flow.types import ValidityResult
    x = np.asarray(positions, dtype=np.float64)
    raw = _validity(x, context)
    raw_pos = np.asarray(raw.positions, dtype=np.float64)
    finite = np.isfinite(raw_pos).reshape(len(x), -1).all(axis=1)
    out_pos, ok = raw_pos.copy(), np.zeros(len(x), bool)
    reason = np.array(['polish: not finite'] * len(x), dtype=object)
    info = [dict(raw_ok=bool(raw.ok[i]), raw_reason=str(raw.reason[i])) for i in range(len(x))]
    idx = np.flatnonzero(finite)
    if len(idx):
        scratch = os.path.join(context.scratch, 'polish')
        os.makedirs(scratch, exist_ok=True)
        uni = mda.Universe(os.path.join(_SYSTEM, 'conf_A_box.gro'))
        p = types.SimpleNamespace(universe=uni, parent=_SYSTEM)
        res = validate._polish(raw_pos[idx], _polisher, _AtomCount(), p, uni.dimensions, scratch, _POLISH_WORKERS,
                               int(context.seed))
        done = np.array([r[1] for r in res], bool)
        for j, r in enumerate(res):
            info[idx[j]].update(polish_ok=bool(r[1]),
                                heavy_rms_displacement_A=(r[3] or {}).get('heavy_rms_displacement_A'))
            if not r[1]:
                reason[idx[j]] = f'polish: {r[2]}'
        if done.any():
            polished = np.stack([res[j][0] for j in np.flatnonzero(done)])
            again = _validity(polished, context)
            sel = idx[done]
            out_pos[sel] = np.asarray(again.positions, dtype=np.float64)
            ok[sel] = np.asarray(again.ok, bool)
            reason[sel] = np.asarray(again.reason, dtype=object)
    return ValidityResult(positions=out_pos, ok=ok, reason=reason, info=info)


# em_flow.mdp / relax_flow.mdp = the RFPS campaign's em.mdp / relax.mdp without their define = -DPOSRES: the hooks
# restrain the flow atoms with their own define and reference, and the solvator's genion grompp passes no -r (an
# active posre.itp made every candidate fail there, test job 2473269)
_solvator = GromacsSolvator(_descriptors, topology=os.path.join(_here, 'topol.top'),
                            em_mdp=os.path.join(_here, 'em_flow.mdp'),
                            nvt_stages=[NVTStage(os.path.join(_here, 'relax_flow.mdp'), 20.0, gen_vel=True)],
                            grompp='gmx grompp', mdrun=_GMX_MDRUN_PREP, maxwarn=5)


def flow_prepare_function(positions, context):
    return _solvator(positions, context)


_minimizer = GromacsMinimizer(em_mdp=os.path.join(_here, 'em_flow.mdp'), restrain='flow_heavy', restraint_k=1e4,
                              emtol=100.0, nsteps=1000, grompp='gmx grompp', mdrun=_GMX_MDRUN_PREP, maxwarn=5)


def flow_minimize_function(frame, context):
    return _minimizer(frame, context)


def fit(params, pathensemble, verbose=False, worker=None):
    """One round of the sequential recipe (fresh head, early-stopped; then a fresh adapter on its labels) on the
    campaign's own data so far."""
    from bioemu2_committor.aimmd_flow_fit import bioemu2_fit_sequential
    # the generator starts at 100 shooting points; the test job sets both variables low to exercise it quickly
    min_shots = int(os.environ.get('BIOEMU2_FIT_MIN_SHOTS', '100'))
    adapter_steps = int(os.environ.get('BIOEMU2_FIT_ADAPTER_STEPS', '3000'))
    # verbose always: each round's phases and choices go to the trainer's log (the overnight monitoring reads them)
    return bioemu2_fit_sequential(params, pathensemble, verbose=True, worker=worker,
                                  min_shots_for_generator=min_shots, adapter_steps=adapter_steps)
