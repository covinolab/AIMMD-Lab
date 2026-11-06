"""The SLURM job script of ``Launcher.create_job``: environment, run-folder guard, preflight.

- Golden scripts (``tests/data/job_script_*.golden.sh``) of the toy 1-D
  C-GAIMMD campaign: one with every hook (``preamble``, the run-folder guard,
  ``preflight``) and one without, which is the script ``create_job`` wrote
  before the hooks existed.
- The run-folder guard, run in bash with a stand-in ``squeue``: a folder
  claimed by a job that still runs stops the new job; a finished job's claim
  is taken over; two jobs claiming at the same moment leave one.
- Slow lane: the whole toy campaign started through its job script, with
  local stand-ins for ``srun``, ``scancel`` and ``squeue`` (preamble,
  guard, preflight, four workers, sidecars), and a job whose preflight fails
  starts no worker.

The stand-ins come first on ``PATH`` and the job id is not a number, so no
real SLURM command can act on a real job.
"""

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

import aimmd
from aimmd import _config
from tests._cgaimmd_toy_campaign import make_trained_campaign
from tests.test_toy_1d_cgaimmd import short_sys_path        # noqa: F401 (fixture)


DATA = Path(__file__).resolve().parent / 'data'
PREAMBLE = ['module load GROMACS/2025.4-CUDA-13', 'export HF_HUB_OFFLINE=1']
JOB_ID = 'local-test-job'
BASH = shutil.which('bash')


def _campaign(tmp_path):
    folder = make_trained_campaign(tmp_path / 'campaign', 'flow_matching')
    os.chdir(folder)
    return folder


def _golden(name):
    text = (DATA / name).read_text()
    return text.replace('@PYTHON@', _config.PYTHON).replace('@WORKER@', _config.WORKER)


def test_job_script_with_every_hook(tmp_path, monkeypatch, short_sys_path):
    """Preamble, run-folder guard and preflight, in this order, before the workers (golden)."""
    monkeypatch.chdir(tmp_path)
    _campaign(tmp_path)
    aimmd.Launcher('params.py', 'run').create_job(
        'job.sh', n=1, n1=1, n2=1, nsteps=4, walltime=1800, preamble=PREAMBLE,
        preflight='--candidates 2')
    assert Path('job.sh').read_text() == _golden('job_script_with_hooks.golden.sh')


def test_job_script_without_hooks_is_unchanged(tmp_path, monkeypatch, short_sys_path):
    """Without the hooks the script is the one create_job wrote before they existed (golden)."""
    monkeypatch.chdir(tmp_path)
    _campaign(tmp_path)
    aimmd.Launcher('params.py', 'run').create_job(
        'job.sh', n=1, n1=1, n2=1, nsteps=4, walltime=1800, guard_run_folders=False)
    assert Path('job.sh').read_text() == _golden('job_script_plain.golden.sh')


def test_hook_arguments_are_checked(tmp_path, monkeypatch, short_sys_path):
    monkeypatch.chdir(tmp_path)
    _campaign(tmp_path)
    launcher = aimmd.Launcher('params.py', 'run')
    with pytest.raises(TypeError, match='preflight must be'):
        launcher.create_job('job.sh', n=1, preflight=1)
    with pytest.raises(TypeError, match='preamble must be'):
        launcher.create_job('job.sh', n=1, preamble=['ok', 3])
    assert not Path('job.sh').exists()
    # a string preamble is split into lines; the guard is on by default
    launcher.create_job('job.sh', n=1, preamble='source env/bin/activate\nexport A=1\n')
    text = Path('job.sh').read_text()
    assert '# environment\nsource env/bin/activate\nexport A=1\n\n# default names' in text
    assert 'claim_run_folder run\nsleep 2\nconfirm_run_folder run\n' in text
    assert 'aimmd.cgaimmd.preflight' not in text


def test_preflight_needs_a_cgaimmd_run(tmp_path, monkeypatch, short_sys_path):
    """The preflight checks C-GAIMMD params: a job without a C-GAIMMD run refuses it."""
    monkeypatch.chdir(tmp_path)
    folder = _campaign(tmp_path)
    params = aimmd.Params.load(str(folder / 'params.py'))
    params.chain_type = 'rfps'
    with pytest.raises(ValueError, match="no run of this job has chain_type='cgaimmd'"):
        aimmd.Launcher(params, 'run').create_job('job.sh', n=1, preflight=True)


# ---------------------------------------------------------------------------
# the run-folder guard, in bash
# ---------------------------------------------------------------------------

SQUEUE = '''#!/bin/bash
# stand-in for `squeue -h -j ID -t ... -o %i`: prints ID when it is listed in running.txt
while [[ $# -gt 0 ]]; do
    case "$1" in -j) id="$2"; shift 2;; *) shift;; esac
done
if grep -qx "${id}" "{running}" 2>/dev/null; then echo "${id}"; fi
exit 0
'''

RACING_SLEEP = '''#!/bin/bash
# stand-in for `sleep`: meanwhile another job claims the run folder
echo "999 otherhost 2026-11-06T00:00:00+00:00" > "{record}"
'''


def _guard_block(text):
    """The guard of a job script: from its comment to the blank line after it."""
    start = text.index('# one job per run folder')
    return text[start:text.index('\n\n', start)] + '\n'


@pytest.fixture
def guard(tmp_path, monkeypatch, short_sys_path):
    """Run the guard of a generated job script in bash; returns ``run(running=(), race=False)``."""
    monkeypatch.chdir(tmp_path)
    folder = _campaign(tmp_path)
    aimmd.Launcher('params.py', 'run').create_job('job.sh', n=1, n1=1, n2=1, nsteps=4)
    script = folder / 'guard.sh'
    script.write_text('set -e\n' + _guard_block((folder / 'job.sh').read_text()))
    record = folder / 'run' / 'aimmd_job.txt'
    shims = tmp_path / 'shims'
    shims.mkdir()
    (shims / 'squeue').write_text(SQUEUE.replace('{running}', str(tmp_path / 'running.txt')))
    (shims / 'squeue').chmod(0o755)

    def run(running=(), race=False, path=None):
        (tmp_path / 'running.txt').write_text(''.join(f'{job}\n' for job in running))
        sleep = shims / 'sleep'
        sleep.write_text(RACING_SLEEP.replace('{record}', str(record)) if race
                         else '#!/bin/bash\nexit 0\n')
        sleep.chmod(0o755)
        env = dict(os.environ, SLURM_JOB_ID=JOB_ID,
                   PATH=path or f'{shims}:{os.environ["PATH"]}')
        return subprocess.run([BASH, str(script)], cwd=folder, env=env, capture_output=True,
                              text=True, timeout=60)

    run.record = record
    run.shims = shims
    return run


def _owner(record):
    return record.read_text().split()[0]


def test_guard_claims_a_new_run_folder(guard):
    result = guard()
    assert result.returncode == 0, result.stderr
    assert _owner(guard.record) == JOB_ID


def test_guard_stops_while_another_job_runs_in_the_folder(guard):
    guard.record.write_text('111 node7 2026-11-06T00:00:00+00:00\n')
    result = guard(running=['111'])
    assert result.returncode == 1
    assert 'job 111 is still running in the run folder run' in result.stderr
    assert _owner(guard.record) == '111'                    # the running job keeps its claim


def test_guard_takes_over_the_folder_of_a_finished_job(guard):
    guard.record.write_text('111 node7 2026-11-06T00:00:00+00:00\n')
    result = guard(running=['222'])                          # 111 is no longer running
    assert result.returncode == 0, result.stderr
    assert _owner(guard.record) == JOB_ID
    # a job that finds its own claim (requeued under the same id) starts as well
    assert guard(running=[JOB_ID]).returncode == 0


def test_guard_two_jobs_claiming_at_the_same_moment(guard):
    """The later claim wins: this job sees job 999's claim after its own and stops."""
    result = guard(race=True)
    assert result.returncode == 1
    assert 'job 999 claimed the run folder run at the same time' in result.stderr


def test_guard_without_squeue_warns_and_claims(guard, tmp_path):
    tools = tmp_path / 'tools'
    tools.mkdir()
    for name in ('head', 'cut', 'date', 'hostname', 'mkdir', 'grep'):
        os.symlink(subprocess.run(['bash', '-c', f'command -v {name}'], capture_output=True,
                                  text=True, check=True).stdout.strip(), tools / name)
    os.symlink(guard.shims / 'sleep', tools / 'sleep')
    result = guard(path=str(tools))
    assert result.returncode == 0, result.stderr
    assert 'squeue not found' in result.stderr
    assert _owner(guard.record) == JOB_ID


# ---------------------------------------------------------------------------
# slow: the toy campaign started through its job script
# ---------------------------------------------------------------------------

SRUN = '''#!/bin/bash
# stand-in for srun: drop the srun options, keep the PID for the clean-up, run the worker
while [[ $# -gt 0 && "$1" == -* ]]; do shift; done
echo $$ >> "{pids}"
exec "$@"
'''

SCANCEL = '''#!/bin/bash
# stand-in for scancel: SIGTERM to the job script's other children (its workers)
echo "scancel $*" >> "{log}"
for pid in $(ps -o pid= --ppid "$PPID"); do
    if [[ "$pid" != "$$" ]]; then kill -TERM "$pid" 2>/dev/null; fi
done
exit 0
'''


def _run_job(folder, shims_root, timeout=900):
    """Run ``job.sh`` of `folder` with bash and the stand-ins; ``(status, output, worker PIDs)``."""
    shims = shims_root / 'shims'
    shims.mkdir()
    pids = shims_root / 'worker_pids.txt'
    for name, text in (('srun', SRUN.replace('{pids}', str(pids))),
                       ('scancel', SCANCEL.replace('{log}', str(shims_root / 'scancel.log'))),
                       ('squeue', '#!/bin/bash\nexit 0\n')):
        (shims / name).write_text(text)
        (shims / name).chmod(0o755)
    env = dict(os.environ, PATH=f'{shims}:{os.environ["PATH"]}', SLURM_JOB_ID=JOB_ID)
    env.pop('PYTHONPATH', None)
    with open(shims_root / 'job.out', 'w') as out:
        job = subprocess.Popen([BASH, 'job.sh'], cwd=folder, env=env, stdout=out,
                               stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = job.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(job.pid, signal.SIGTERM)
            status = job.wait(timeout=60)
    started = [int(p) for p in pids.read_text().split()] if pids.exists() else []
    for pid in started:                                     # only the workers this job started
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    return status, (shims_root / 'job.out').read_text(), started


@pytest.mark.slow
def test_toy_campaign_through_its_job_script(tmp_path, monkeypatch, short_sys_path):
    """The generated script runs the preamble, claims the run folder, passes the preflight and
    starts the four workers, which shoot, train and stop the job (about a minute)."""
    monkeypatch.chdir(tmp_path)
    folder = _campaign(tmp_path)
    aimmd.Launcher('params.py', 'run').create_job(
        'job.sh', n=1, n1=1, n2=1, nsteps=4, walltime=1800,
        preamble=['echo "preamble ran" > preamble.txt'], preflight=True)
    t0 = time.time()
    status, output, started = _run_job(folder, tmp_path)
    assert status == 0, output[-3000:]
    assert (folder / 'preamble.txt').read_text() == 'preamble ran\n'
    assert _owner(folder / 'run' / 'aimmd_job.txt') == JOB_ID
    assert 'preflight OK' in output, output[-3000:]
    assert len(started) == 4                                 # free A, free B, chain, trainer
    sidecars = sorted((folder / 'run').glob('chainR*/path*.xtc.cgaimmd.json'))
    assert sidecars and all(json.loads(s.read_text())['source'] == 'cgaimmd' for s in sidecars)
    assert json.loads((folder / 'run' / 'networkARB.json').read_text())['round'] >= 1
    assert time.time() - t0 < 900


@pytest.mark.slow
def test_a_failed_preflight_starts_no_worker(tmp_path, monkeypatch, short_sys_path):
    monkeypatch.chdir(tmp_path)
    folder = _campaign(tmp_path)
    aimmd.Launcher('params.py', 'run').create_job('job.sh', n=1, n1=1, n2=1, nsteps=4,
                                                  preflight=True)
    (folder / 'initial_network.h5').write_bytes(b'not a network')
    status, output, started = _run_job(folder, tmp_path)
    assert status == 1
    assert 'preflight FAILED' in output and 'failed: no worker started' in output
    assert started == []
