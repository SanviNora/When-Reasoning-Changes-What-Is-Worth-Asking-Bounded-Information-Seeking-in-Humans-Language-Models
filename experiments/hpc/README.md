# Cluster job scripts (SLURM)

Open-weight models are served with vLLM on a GPU cluster; CPU-heavy fits run as array jobs.
Submit every script from the repository root. The cluster environment needs the packages in
`requirements.txt` plus `vllm==0.31.0` and `huggingface_hub`.

Before submitting, set the allocation (and, if your layout differs from the defaults, the paths):

| Variable | Meaning | Default |
|---|---|---|
| `SBATCH_ACCOUNT` | SLURM allocation to charge (read by `sbatch` itself) | none: required |
| `CCM_ROOT` | repository path on the cluster | the submit directory |
| `CCM_ENV` | Python environment with the requirements and vLLM | `$CCM_ROOT/env` |
| `CCM_HF_HOME` | Hugging Face cache holding the model weights (jobs set `HF_HOME` from it, so an `HF_HOME` already in your login environment cannot redirect them) | `hf_cache/` next to the repository |

Put the exports in `experiments/hpc/site.env` (git-ignored; copy `site.env.example`) and
`source experiments/hpc/site.env` before `sbatch`. `chain.sh` sources it automatically.

| Script | Use |
|---|---|
| `download_models.sbatch` | download model weights (CPU only) |
| `serve_and_run.sbatch` | serve one model from `configs/model_registry.json` and run an experiment script against it |
| `chain.sh` | submit N short GPU jobs in sequence; replies are cached, so each job resumes where the last stopped |
| `build_cache.sbatch` | approximation search for the bounded model's representation cache |
| `cpu_array.sbatch` | run any Python script once per array task on CPU; the argument `@TASK@` becomes the task index |

## The runs behind the reported results

Language-model runs (`G=experiments/hpc/serve_and_run.sbatch`; some ran with `--gres=gpu:l40s:2 --mem=96G` on the L40S nodes):

```
sbatch $G <model> experiments/study_b/screen_models.py --model <model>              # screening
sbatch $G <model> experiments/study_b/play_games.py --model <model> --mode both     # B1, every admitted open-weight model
#   Qwen3-32B's B1 (rerun with the corrected client; see below). e1 under T3 ran in 6 parts and under T0 in 4,
#   on 2 L40S GPUs each, and free play ran in one job; each e1 set then ran once without --shard on CPU to assemble.
sbatch --gres=gpu:l40s:2 --mem=96G --time=24:00:00 $G qwen3-32b experiments/study_b/play_games.py --model qwen3-32b \
    --mode e1 --condition T3 --shard i --shards 6        # i = 0..5; T0: --condition T0 --shards 4
sbatch --gres=gpu:l40s:2 --mem=96G $G qwen3-32b experiments/study_b/play_games.py --model qwen3-32b --mode free
sbatch --array=0 experiments/hpc/cpu_array.sbatch experiments/study_b/play_games.py --model qwen3-32b --mode e1 --condition T3
sbatch $G qwen3-32b experiments/study_b/probe_battery.py --model qwen3-32b --condition T0 \
    --splits development processing_calibration confirmation --samples 4                              # B2 battery
sbatch $G qwen3-32b experiments/study_b/probe_battery.py --model qwen3-32b --condition T0 \
    --splits reserve --samples 4                                                                       # B3 confirmation data
sbatch $G qwen3-32b experiments/study_b/usefulness_choice.py --model qwen3-32b --condition T0 \
    --splits choice_calibration --pairs-per-family 2 --branch-samples 4 --choice-samples 8           # native choice
```

Study C (Qwen3-32B; c = T1, T2, T3 for the battery, T0 and T3 for usefulness):
```
sbatch $G qwen3-32b experiments/study_b/probe_battery.py --model qwen3-32b --condition c \
    --splits processing_calibration confirmation --samples 4                                       # C-a battery
sbatch $G qwen3-32b experiments/study_b/usefulness_choice.py --model qwen3-32b --condition c \
    --splits support_calibration --pairs-per-family 2 --branch-samples 4 --choice-samples 8        # C-c calibration
                                                                       # (and --splits choice_calibration under T3)
sbatch --gres=gpu:l40s:2 --mem=96G $G qwen3-32b experiments/study_c/cap_pilot.py --model qwen3-32b \
    --max-tokens 15000 --n-t3 200 --n-t0 60 --workers 24                                           # cap pilot
```
The confirmation run is in `experiments/study_c/README.md`.

The language-model client waits up to 4 hours for a reply and never retries on its own (`mastermind/llm.py`). An
earlier version timed out at 600 s and retried silently; a timeout aborts the request on the server, so on a busy
node long replies were redrawn until a short one finished. Every cached reply records its total time
(`duration_s`); an audit of all of them found redrawn replies only in Qwen3-32B's B1 game play and in a first cap
pilot. Both were set aside and rerun from scratch with the corrected client.
The Claude models run from a workstation (`screen_models.py` and `play_games.py` with `--model claude-haiku-5-5`
or `claude-opus-5-5`; the API key in `.env`).

CPU fits (`S=experiments/hpc/cpu_array.sbatch`). Converged fits read the greedy fits of the same data as a start,
so the greedy jobs run first; variants and recovery start after the primary converged fits.

```
sbatch --array=0-38 $S experiments/study_a/fit_human_models.py fit --participant @TASK@                       # paper's procedure
sbatch --array=0-38 $S experiments/study_a/fit_human_models.py fit --participant @TASK@ --procedure converged  # primary
#   variants: add --include-unsolved | --models bounded --half early|late | --models bounded --cache-rep 2 (each with
#   and without --procedure converged); wider limits: --procedure wide
sbatch --array=0-38 $S experiments/study_a/recover_human_models.py run --participant @TASK@ [--procedure converged]
sbatch --array=0 $S experiments/study_b/fit_llm_human_models.py fit --model <model> [--procedure converged]         # B1b pooled
sbatch --array=0-9 $S experiments/study_b/fit_llm_human_models.py fit --model <model> --block @TASK@ [--procedure converged]
sbatch --array=0-199 $S experiments/study_b/fit_llm_human_models.py fit --model <model> --boot @TASK@ [--procedure converged]
python experiments/study_b/fit_process_models.py fit --model qwen3-32b --condition T0                              # B3 fits (local)
sbatch --array=0-99 $S experiments/study_b/fit_process_models.py boot --model qwen3-32b --condition T0 --rep @TASK@0 --count 10
sbatch --array=0-19 $S experiments/study_b/recover_process_models.py run --rep @TASK@                            # B3 recovery
sbatch --array=0-49 $S experiments/study_b/recover_process_models.py run --rep @TASK@ --fitted qwen3-32b_T0       # ... at the fitted values
```
Then run locally (minutes on a laptop):

```
python experiments/study_b/fit_llm_human_models.py fit --model <model> --procedure converged --cache-rep 2   # B1b cache-seed check
python experiments/study_b/fit_process_models.py refit_m3 --model qwen3-32b --condition T0   # corrected M3 (fit + every boot resample)
python experiments/study_b/recover_process_models.py refit_m3                                # corrected M3 in the planned-design recovery
python experiments/study_b/recover_process_models.py refit_m3 --fitted qwen3-32b_T0          # ... and at the fitted values
python experiments/study_b/fit_process_models.py product_rule --model qwen3-32b --condition T0   # product-rule check of the observation model
```
Every summary is then regenerated locally with `bash experiments/reproduce_summaries.sh`.
