#!/bin/bash
# Download model weights to the shared Hugging Face cache, HF_HOME (CPU only, no GPU time).
#   sbatch experiments/hpc/download_models.sbatch Qwen/Qwen3-32B openai/gpt-oss-120b ...
#SBATCH --job-name=mm-download
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm_download_%j.log

set -euo pipefail
cd "${CCM_ROOT:-$SLURM_SUBMIT_DIR}"
CCM_ENV=${CCM_ENV:-$PWD/env}  # Python environment on the cluster
export HF_HOME="${CCM_HF_HOME:-$(dirname "$PWD")/hf_cache}"  # project setting; an HF_HOME from the login environment is ignored
export HF_HUB_ENABLE_HF_TRANSFER=0
for MODEL in "$@"; do
    echo "== $MODEL $(date)"
    "$CCM_ENV"/bin/python - "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
model = sys.argv[1]
# Skip duplicate weight copies vLLM does not need; Mistral keeps its consolidated file (vLLM loads it in Mistral format).
path = snapshot_download(model, ignore_patterns=["original/*", "metal/*", "*.pth", "*.gguf"] if "mistralai" in model else ["original/*", "metal/*", "consolidated*", "*.pth", "*.gguf"])
print("saved", model, "->", path)
EOF
done
echo "done $(date)"
