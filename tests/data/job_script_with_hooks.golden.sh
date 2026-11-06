#!/bin/bash -x
#SBATCH --job-name=AIMMD
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=4
#SBATCH --time=00:30:00

# environment
module load GROMACS/2025.4-CUDA-13
export HF_HUB_OFFLINE=1

# default names
PYTHON="@PYTHON@"
WORKER="@WORKER@"

# one job per run folder: claim the run folders, or stop if another
# job that is still running claimed one of them
claim_run_folder() {
    local folder="$1" record="$1/aimmd_job.txt" other=""
    mkdir -p "${folder}"
    if [[ -f "${record}" ]]; then
        other=$(head -n 1 "${record}" | cut -d ' ' -f 1)
    fi
    if [[ -n "${other}" && "${other}" != "${SLURM_JOB_ID}" ]] && \
            squeue -h -j "${other}" -t RUNNING,COMPLETING,CONFIGURING,SUSPENDED -o %i \
            2>/dev/null | grep -qx "${other}"; then
        echo "job ${other} is still running in the run folder ${folder}:" \
             "this job (${SLURM_JOB_ID}) does not start" >&2
        exit 1
    fi
    echo "${SLURM_JOB_ID} $(hostname) $(date -Iseconds)" > "${record}"
}
confirm_run_folder() {
    local folder="$1" owner=""
    owner=$(head -n 1 "${folder}/aimmd_job.txt" | cut -d ' ' -f 1)
    if [[ "${owner}" != "${SLURM_JOB_ID}" ]]; then
        echo "job ${owner} claimed the run folder ${folder} at the same time:" \
             "this job (${SLURM_JOB_ID}) does not start" >&2
        exit 1
    fi
}
if ! command -v squeue > /dev/null; then
    echo "warning: squeue not found; the run-folder guard cannot see other jobs" >&2
fi
claim_run_folder run
sleep 2
confirm_run_folder run

# preflight: check every C-GAIMMD run before any worker starts
"${PYTHON}" -m aimmd.cgaimmd.preflight params1.py --run run --candidates 2 || {
    echo "preflight of params1.py failed: no worker started" >&2
    exit 1
}

# enable job control
set -m

# workers
# "run" freeA (worker0)
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=0 \
  "${PYTHON}" "${WORKER}" "params1.py" "run" "0" "skip" "skip" "freeA/worker0.log" "inf" "inf" "inf" "59.0" "free" "A" "0" "1" "True" &

# "run" freeB (worker0)
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=0 \
  "${PYTHON}" "${WORKER}" "params1.py" "run" "0" "skip" "skip" "freeB/worker0.log" "inf" "inf" "inf" "59.0" "free" "B" "0" "1" "True" &

# "run" chainR0
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=0 \
  "${PYTHON}" "${WORKER}" "params1.py" "run" "0" "skip" "skip" "chainR0/worker.log" "inf" "inf" "inf" "59.0" "shoot" "R" "0" "" "inf" "1" &

# "run" ARB trainer
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=0 \
  "${PYTHON}" "${WORKER}" "params1.py" "run" "0" "skip" "skip" "trainARB.log" "inf" "4.0" "inf" "59.0" "train" "inf" "True" &

# wait until any process exits
wait -n
scancel ${SLURM_JOB_ID}
wait
