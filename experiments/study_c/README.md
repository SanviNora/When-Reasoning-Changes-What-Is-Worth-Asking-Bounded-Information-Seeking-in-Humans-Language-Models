# Study C: capacity manipulation

The depth model (Qwen3-32B) gets the structured reasoning aid (T3) and is compared with the standard (T0) and
length-matched (T1) prompts; a generic "check your work" prompt (T2) is an extra control. Outputs go to
`results/study_c/`; the confirmation plan is `prereg/analysis_plan.md`.

| Script | Does | Writes |
|---|---|---|
| `aid_effects.py` | C-a: how the aid changes evidence use (error rates and process-model parameters, paired family bootstrap) | `ca_aid_effects.json` |
| `cap_pilot.py` | reruns cut-off aided replies at a higher reply cap, to set the cap of the usefulness runs | `cap_pilot_*.json` |
| `prediction_test.py` | C-c: calibrate the predictors, power simulation, pair selection, lock, score | `cc_*.json`, `prereg/prediction_lock.json` |
| `draw_families.py` | fresh confirmation families (none related to an earlier family) | `data/splits/families_c_v1.json` |
| `choice_shift.py` | C-b: do the model's question choices follow the aid's change in usefulness? | `cb_choice_shift_*.json` |
| `noise_check.py` | post hoc: Γ's sampling noise on the calibration and confirmation pairs, and the power the realized design had | `cc_noise_check.json` |

The model runs use `experiments/study_b/probe_battery.py` (C-a) and `experiments/study_b/usefulness_choice.py`
(usefulness and choices; `--families` and `--pairs-file` for the confirmation).

Order: C-a battery and fits → cap pilot (kept the 6,144-token cap) → `calibrate` → `power --r2 0.20` →
`draw_families --n 120` → `select` → `lock --samples 8` (commit, then OSF) → confirmation runs under T0 and T3 →
`score` and `choice_shift.py --split c_confirmation` → `noise_check.py` (post hoc) → `experiments/primary_tests.py`
(Holm across B1, B3 and C-b).

Confirmation run, per condition c in T0, T3 (registry cap, so no `--max-tokens`). Parts i = 0 … n−1 ran in
parallel jobs on 2 L40S GPUs each: T3 in 6 parts, T0 in 4.

    sbatch --gres=gpu:l40s:2 --mem=96G --time=24:00:00 experiments/hpc/serve_and_run.sbatch qwen3-32b \
        experiments/study_b/usefulness_choice.py --model qwen3-32b --condition c \
        --families data/splits/families_c_v1.json --pairs-file data/splits/confirmation_pairs_c_v1.json \
        --splits c_confirmation --branch-samples 8 --choice-samples 8 --workers 32 --shard i --shards n

Then one run per condition, without `--shard`, reads every reply from the cache and writes the record files. It
needs no model server, so it ran on CPU (`S=experiments/hpc/cpu_array.sbatch`):

    sbatch --array=0 --mem=32G $S experiments/study_b/usefulness_choice.py --model qwen3-32b --condition c \
        --families data/splits/families_c_v1.json --pairs-file data/splits/confirmation_pairs_c_v1.json \
        --splits c_confirmation --branch-samples 8 --choice-samples 8 --workers 32
