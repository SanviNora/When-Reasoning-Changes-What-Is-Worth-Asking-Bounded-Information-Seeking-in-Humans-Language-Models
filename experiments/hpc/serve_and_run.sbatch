#!/bin/bash
# Serve one registry model with vLLM, then run an experiment script against it.
#   sbatch experiments/hpc/serve_and_run.sbatch qwen3-32b experiments/study_b/screen_models.py --model qwen3-32b
#   sbatch --gres=gpu:l40s:2 experiments/hpc/serve_and_run.sbatch ...   (when the H200 queue is long)
# The model's Hugging Face id and vLLM arguments come from configs/model_registry.json; the model is split
# across all GPUs the job receives (tensor parallelism; weights and precision unchanged).
#SBATCH --job-name=mm-run
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm_%x_%j.log

set -euo pipefail
KEY=$1; SCRIPT=$2; shift 2
cd "${CCM_ROOT:-$SLURM_SUBMIT_DIR}"
CCM_ENV=${CCM_ENV:-$PWD/env}  # Python environment on the cluster
PY="$CCM_ENV/bin/python"
export PATH="$CCM_ENV/bin:$PATH"  # vLLM JIT-compiles kernels and needs ninja from the env
# Weights location is a project setting (CCM_HF_HOME); an HF_HOME from the login environment is ignored.
export HF_HOME="${CCM_HF_HOME:-$(dirname "$PWD")/hf_cache}" HF_HUB_OFFLINE=1
read -r HF_ID VLLM_ARGS < <($PY -c "
import json; r=json.load(open('configs/model_registry.json'))['$KEY']
print(r['hf_id'], r['vllm_args'].replace(' ', '|'))")
VLLM_ARGS=${VLLM_ARGS//|/ }
GPUS=$(nvidia-smi -L | wc -l)
export GPU_NAME="${GPUS}x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
PORT=$((8000 + SLURM_JOB_ID % 1000))
nvidia-smi --query-gpu=name,memory.total --format=csv

"$CCM_ENV"/bin/vllm serve "$HF_ID" --port $PORT --tensor-parallel-size "$GPUS" \
    --gpu-memory-utilization 0.90 $VLLM_ARGS > "results/vllm_${KEY}_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap "kill $SERVER 2>/dev/null" EXIT
until curl -s localhost:$PORT/v1/models > /dev/null; do
    kill -0 $SERVER 2>/dev/null || { echo "vLLM server died"; tail -60 "results/vllm_${KEY}_${SLURM_JOB_ID}.log"; exit 1; }
    sleep 10
done
echo "server up: $HF_ID ($KEY) $(date)"
export VLLM_BASE_URL=http://localhost:$PORT/v1
$PY -u "$SCRIPT" "$@"
echo "finished $(date)"
