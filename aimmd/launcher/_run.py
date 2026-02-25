"""
...
"""

# external
import time
import multiprocessing
from abc import ABC
from math import inf

# aimmd imports
from ..worker import Worker
from ..core.utils import now

# must be defined here else not working
def run_task(params_file, directory,
             localid, cpus_per_task, gpus_per_task,
             log_file, walltime, nsteps, nframes,
             termination_timeout, task, *args, **kwargs):
    Worker(params_file, directory,
           localid, cpus_per_task, gpus_per_task,
           log_file, walltime, nsteps, nframes,
           termination_timeout).run(task, *args, **kwargs)

# run
class LauncherRun(ABC):
    def run(self, n=1, n1=0, n2=0,
            reactive_region_mode='chain',
            state1_mode='free', state2_mode='free',
            nsteps=inf, nframes=inf, nrounds=inf, walltime=inf,
            cpus_per_task='share', gpus_per_task='share'):
        """
        Launch the simulation locally, spawning multiple processes.
        
        Parameters
        ----------
        n: default 1, number of replicas dedicated to shooting simulations
        n1: default 0, number of replicas dedicated to simulations in/around
            the initial state (specified by self.params.states[0])
        n2: default 0, number of replicas dedicated to simulations in/around
            the final state (specified by self.params.states[2])
        reactive_region_mode: default 'chain'; if 'chain': create a Markov
            chain (either TPS or RFPS algorithm); if 'sweep': shoot in order
            from the initial trajectories' configuration, repeat when done
            (useful for directly measuring committor values); if 'free': run
            free simulations instead
        state1_mode: default 'free'; if 'free': run standard free simulations
            in/around the initial state; if 'shoot': shoot in the state
            (use a single bin)
        state2_mode: default 'free'; if 'free': run standard free simulations
            in/around the final state; if 'shoot': shoot in the state
            (use a single bin)
        nsteps: default inf, maximum number of shooting simulations
        nframes: default inf, maximum number of simulated frames,
                 has priority over nsteps
        walltime: default inf, maximum number of simulation time,
                  has priority over nframes and nsteps
        cpus_per_task: str or int, defaut 'share'
            Number of CPUs to allocate per task
            if 'share': equally distribute available resources among workers
            if 'all': each worker takes them all
        gpus_per_task: str or int, default 'share'
            Number of GPUs to allocate per task
            if 'share': equally distribute available resources among workers
            if 'all': each worker takes them all

        Returns
        -------
        None
        """
        
        # update
        self._update(n, n1, n2,
                     reactive_region_mode, state1_mode, state2_mode,
                     nsteps, nframes, nrounds, walltime,
                     cpus_per_task, gpus_per_task)
        
        # initialize processes, create folders
        for args, description in zip(*self._build()):
            self._processes.add(run_task, *args, name=description)
        
        # start processes
        try:
            self._processes.run(timeout=self.termination_timeout)
            
            # wait for completion within walltime
            # stop all as soon as any other stops
            t0 = time.time()
            must_stop = not self._processes.alive.all()
            while time.time() - t0 < walltime:
                if must_stop:
                    break
                for process in self._processes:
                    exitcode = process.exitcode
                    if exitcode is None:
                        continue
                    must_stop = True
                    if exitcode:
                        raise RuntimeError('launcher run failed')
                    break
                time.sleep(.01)  # avoid freezing
        
        # catch exceptions
        except Exception as exception:
            raise exception
        
        # clear all processes
        finally:
            self._processes.clear(timeout=self.termination_timeout)
