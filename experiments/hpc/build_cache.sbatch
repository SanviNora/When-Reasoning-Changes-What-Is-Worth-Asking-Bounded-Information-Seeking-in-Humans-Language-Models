#!/bin/bash
# Approximation search for the bounded model, one array task per (pattern, feedback) case.
#   sbatch experiments/hpc/build_cache.sbatch                       (conservative cache, M2c / bounded model)
#   sbatch experiments/hpc/build_cache.sbatch --variant symmetric   (symmetric cache, M2s)
# Then aggregate on the login node or a laptop: python experiments/study_a/build_cache.py aggregate
#SBATCH --job-name=mm-cache
#SBATCH --array=0-54
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm_%x_%a.log

cd "${CCM_ROOT:-$SLURM_SUBMIT_DIR}"
CCM_ENV=${CCM_ENV:-$PWD/env}  # Python environment on the cluster
"$CCM_ENV"/bin/python -u experiments/study_a/build_cache.py search --case $SLURM_ARRAY_TASK_ID --iters 100000 --workers 4 "$@"
