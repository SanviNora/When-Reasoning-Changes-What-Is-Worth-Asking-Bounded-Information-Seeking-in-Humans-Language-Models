# Study A: human reproduction

Reproduction of Cheyette et al. (2025) from their released data (see `data/human/README.md`). Outputs go to
`results/study_a/`; the findings are in `results/study_a/README.md`.

| Script | Does | Writes |
|---|---|---|
| `e1_reproduction.py` | Experiment 1: extracts the starting positions people saw and reproduces the behavioral result | `data/stimuli/e1_stimuli.json`, `e1_reproduction.json` |
| `e2_descriptive.py` | Experiment 2 forced choice, descriptive | `e2_descriptive.json` |
| `e3_reproduction.py` | Experiment 3 free play: solve rate, guesses, information per guess, benchmarks | `e3_reproduction.json` |
| `build_cache.py` | rebuilds the bounded model's representation cache (conservative; symmetric variant for Study B; `--rep 2` for an independent replicate) | `data/model_cache/` |
| `fit_human_models.py` | fits the unbounded, leaky and bounded models to each participant and compares them; held-out, early/late, cache-seed and wider-limit checks | `e3_fits*/`, `e3_model_comparison*.json`, `e3_heldout*.json`, `e3_game_index_halves*.json`, `e3_cache_seed*.json` |
| `recover_human_models.py` | model and parameter recovery at the Experiment 3 sample size | `e3_recovery*/`, `e3_recovery_summary*.json` |

Each fitting script takes `--procedure greedy` (the paper's search, the reproduction) or `--procedure converged`
(the primary results); outputs of converged fits end in `_converged`. Usage is in each script's docstring.
