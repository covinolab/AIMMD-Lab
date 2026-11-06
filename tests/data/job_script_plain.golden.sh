#!/bin/bash -x
#SBATCH --job-name=AIMMD
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=4
#SBATCH --time=00:30:00

# default names
PYTHON="@PYTHON@"
WORKER="@WORKER@"

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
