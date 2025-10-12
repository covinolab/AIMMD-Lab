import os, sys, argparse, subprocess, time
from ..core import Params

class Generate:
    
    def __init__(self, params):
        self.params = Params
    
    def generate_slurm(directory, nsteps, n, nA, nB, eA, eB, params)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AIMMD launcher')
    parser.add_argument('directory', type=str,
        help='project directory')
    parser.add_argument('nsteps', type=int,
        help='target 2-way-shooting excursions in the transition region')
    parser.add_argument('n', type=int,
        help='number of tasks dedicated to 2-way shooting')
    parser.add_argument('nA', type=int,
        help='number of tasks dedicated to equilibrium simulations in A')
    parser.add_argument('nB', type=int,
        help='number of tasks dedicated to equilibrium simulations in B')
    parser.add_argument('-eA', '--extend_A', type=int, default=0,
        help='number of tasks dedicated to extending transitin ending in A')
    parser.add_argument('-eB', '--extend_B', type=int, default=0,
        help='number of tasks dedicated to extending transitin ending in B')
    parser.add_argument('-p', '--params', type=str, default='params.py',
        help='aimmd run parameters (will override defaults)')
    parser.add_argument('-s', '--slurm', action='store_true')
    parser.add_argument('-d', '--dependency', type=str, default='',
        help='wait for job to terminate before starting')
    args = parser.parse_args()
    directory = args.directory
    nsteps = args.nsteps
    n = args.n
    nA = args.nA
    nB = args.nB
    eA = args.extend_A
    eB = args.extend_B
    params = args.params
    slurm = args.slurm
    dependency = args.dependency

    # TODO put generate backend here as a bonus
    
    PYTHON = sys.executable
    command = (f'{PYTHON} generate_backend.py '
               f'"{directory}" {nsteps} {n} {nA} {nB} {eA} {eB} '
               f'"{params}" {slurm} "{dependency}"')
    
    try:
        text0 = open(f'{directory}/manager.log').read()
    except:
        text0 = ''
    
    if slurm:
        os.system(command)
    else:
        command = (f'nohup {command} > /dev/null 2>&1 & echo $!')
        print(command)
        process = subprocess.Popen(command, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pid = process.stdout.read().decode().strip()
        if dependency:
            print(f'To cancel process in case it has not started yet: kill {pid}')
        else:
            print(f'Please do NOT kill PID {pid}, which is handling AIMMD')
        while True:  # print the output to command line, too
            try:
                time.sleep(1.)
                os.kill(int(pid), 0)
                if not os.path.exists(f'{directory}/manager.log'):
                    continue
                text = open(f'{directory}/manager.log').read()
                if len(text0):
                    text = text.replace(text0, '')
                regex = '#' * 80
                if regex in text:
                    text = text.split(regex)
                    if len(text) < 3:
                        continue
                    print(regex.join(text[:2]) + regex)  # TODO test
                    break
            except:  # see the problem yourself!
                os.system(command[6:-27])
                break
