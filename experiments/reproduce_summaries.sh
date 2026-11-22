#!/bin/bash
# Regenerate every reported number and figure of Studies A–C from the saved records and fits.
# No language-model calls and no refits: the fits themselves come from the cluster jobs in experiments/hpc/
# (their commands are listed in experiments/hpc/README.md). Study C's calibration, power analysis and lock are
# not rerun: they are fixed inputs of the preregistered test. Run from the repository root:
#   bash experiments/reproduce_summaries.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { echo "== $*"; "$PY" "$@" > /dev/null; }

# Study A: behavior
run experiments/study_a/e1_reproduction.py
run experiments/study_a/e2_descriptive.py
run experiments/study_a/e3_reproduction.py
# Study A: model comparison, paper's procedure (greedy) and converged fits (primary)
for proc in greedy converged; do
  run experiments/study_a/fit_human_models.py summarize --procedure $proc
  run experiments/study_a/fit_human_models.py summarize --include-unsolved --procedure $proc
  run experiments/study_a/fit_human_models.py heldout --procedure $proc
  run experiments/study_a/fit_human_models.py halves --procedure $proc
  run experiments/study_a/fit_human_models.py cache_seed --procedure $proc
  run experiments/study_a/recover_human_models.py summarize --procedure $proc
done
run experiments/study_a/fit_human_models.py summarize --procedure wide

# Study B: B1b, models with fits (those passing the solve-rate rule), both procedures
for d in results/study_b/b1b/*_T0 results/study_b/b1b/*_T0_converged; do
  [ -f "$d/pooled.json" ] || continue
  name=$(basename "$d"); model=${name%_T0*}
  proc=greedy; [[ $name == *_converged ]] && proc=converged
  run experiments/study_b/fit_llm_human_models.py summarize --model "$model" --procedure $proc
done
# B1b cache-seed check: pooled bounded fit with the replicate cache, where it was run
for d in results/study_b/b1b/*_T0_cacherep2_converged; do
  [ -f "$d/pooled.json" ] || continue
  name=$(basename "$d"); run experiments/study_b/fit_llm_human_models.py cache_seed --model "${name%_T0_cacherep2_converged}"
done
# B1: every model with an E1 file; the solve-rate rule decides which get inferential results.
# Runs after B1b because its capability table reads the B1b summaries.
MODELS=$(ls results/study_b/b1/*_e1_T0.jsonl | xargs -n1 basename | sed 's/_e1_T0.jsonl//' | tr '\n' ' ')
run experiments/study_b/analyze_b1.py --models $MODELS
# B2/B3: error profile, intervals, recovery, the pre-stated confirmation, native choice
run experiments/study_b/fit_process_models.py profile --model qwen3-32b --conditions T0
run experiments/study_b/fit_process_models.py intervals --model qwen3-32b --condition T0
run experiments/study_b/fit_process_models.py confirm --model qwen3-32b --condition T0
run experiments/study_b/recover_process_models.py summarize
run experiments/study_b/recover_process_models.py summarize --fitted qwen3-32b_T0
run experiments/study_b/native_choice.py --model qwen3-32b
# Study C: C-a, the exploratory choice-shift look on the calibration pairs, the preregistered score (it checks the
# lock's hashes first) and choice shift, the post hoc design check, and Holm across the primary tests
run experiments/study_c/aid_effects.py
run experiments/study_c/choice_shift.py --split support_calibration choice_calibration
run experiments/study_c/prediction_test.py score
run experiments/study_c/choice_shift.py --split c_confirmation
run experiments/study_c/noise_check.py
run experiments/primary_tests.py
# Figures for the paper and poster (results/figures/)
run experiments/make_figures.py
