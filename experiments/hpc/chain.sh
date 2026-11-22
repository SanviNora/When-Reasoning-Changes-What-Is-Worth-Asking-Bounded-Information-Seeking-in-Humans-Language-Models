#!/bin/bash
# Submit N short (45-minute) H200 jobs for one task, each starting after the previous one ends.
# Short jobs get scheduled into gaps in the queue; every LLM reply is cached on disk, so each job
# resumes where the last one stopped and the remaining copies finish quickly once the task is done.
#   bash experiments/hpc/chain.sh 3 b1-qwen3-32b qwen3-32b experiments/study_b/play_games.py --model qwen3-32b --mode both
N=$1; NAME=$2; shift 2
[ -f experiments/hpc/site.env ] && . experiments/hpc/site.env  # cluster account and paths (git-ignored)
dep=""
for i in $(seq 1 "$N"); do
    id=$(sbatch --parsable --job-name="$NAME" --time=00:45:00 $dep experiments/hpc/serve_and_run.sbatch "$@")
    dep="--dependency=afterany:$id"
    echo "$NAME part $i: job $id"
done
