# Study B: language models on the human paradigm

Outputs go to `results/study_b/`; the findings are in `results/study_b/README.md`. Scripts that call a model
run where it is served (`experiments/hpc/serve_and_run.sbatch`) or, for the Claude models, with the API key in `.env`.

| Script | Does | Writes |
|---|---|---|
| `screen_models.py` | screens a candidate model against the floor rule (format compliance, identification above chance) | `screening/` |
| `make_splits.py` | draws the symmetry-distinct state families and assigns the fixed splits | `data/splits/` |
| `play_games.py` | B1: continues the 132 Experiment 1 positions and plays free games | `b1/` |
| `analyze_b1.py` | B1 analysis: signature effect, hazard and mixed models, free-play measures, capability, aid arm | `b1_summary.json` |
| `fit_llm_human_models.py` | B1b: the Study A human models fitted to each model's free play, with blocks, resamples and the cache-seed check | `b1b/`, `b1b_<model>_T0*.json` |
| `probe_battery.py` | B2: probe battery (classify, single-clue classify, identify) on the state families | `battery/` |
| `fit_process_models.py` | B3: process-model fits, error profile, family-bootstrap intervals, product-rule check, M1a confirmation | `process_fits/` |
| `recover_process_models.py` | B3: parameter and model recovery on the planned design and at the fitted values | `process_recovery*/`, `process_recovery*_summary.json` |
| `usefulness_choice.py` | native question choice data: the model's choices between query pairs and its usefulness of each query | `usefulness_choice/` |
| `native_choice.py` | native question choice analysis: which query properties predict the choices | `native_choice/` |

`B3_CONFIRMATION.md` is the pre-stated confirmation test of the exploratory model M1a, committed before its data
were collected. Usage is in each script's docstring.
