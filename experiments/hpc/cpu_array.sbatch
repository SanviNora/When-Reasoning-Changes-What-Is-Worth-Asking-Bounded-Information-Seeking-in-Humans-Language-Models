#!/bin/bash
# Run a Python script once per array task on CPU; the argument @TASK@ becomes the task index.
#   sbatch --array=0-9 experiments/hpc/cpu_array.sbatch \
#       experiments/study_b/fit_llm_human_models.py fit --model gpt-oss-120b --block @TASK@ --procedure converged
# Set the job name, time and array range with sbatch options; logs go to results/slurm_<name>_<job>_<task>.log.
#SBATCH --job-name=mm-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm_%x_%A_%a.log

cd "${CCM_ROOT:-$SLURM_SUBMIT_DIR}"
CCM_ENV=${CCM_ENV:-$PWD/env}  # Python environment on the cluster
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
args=()
for a in "$@"; do args+=("${a//@TASK@/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$CCM_ENV"/bin/python -u "${args[@]}"
