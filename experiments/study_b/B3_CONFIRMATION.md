# B3 confirmation test: two-rate noisy evidence use (M1a)

Written and committed **before** any data on the reserve families exist.

## Why
The planned B3 comparison (M0, M1, M2c, M2s, M3; Qwen3-32B, T0) was run as specified. The error profile then showed that Qwen3 rules out still-possible codes far more often than it keeps ruled-out ones, including on single clues (0.32 vs. 0.13), which none of the planned models can produce with one noise rate. M1a (`mastermind/process_models.py`) gives each clue two error rates: ν_x, wrongly ruling out a code the clue allows, and ν_r, wrongly keeping a code the clue rules out. It fits much better on the calibration families and on the held-out confirmation families. M1a was formulated after the error profile, computed on those same families, had been seen, so that result is exploratory. This test checks it on families that have never been used.

## Data
- Model and condition: Qwen3-32B, condition T0 (thinking off), the same frozen prompts (`configs/prompt_freeze_v1.json`) and decoding as the existing T0 battery.
- Families: the 40 families of the `reserve` split in `data/splits/families_v1.json` (F212–F251; sha256 `5eb3438c…bf15e`, as in `data/splits/manifest_v1.json`). They were drawn with the committed seed, are symmetry-distinct from every other split and from the Experiment 1 positions, and have not been run under any condition.
- Battery: `experiments/study_b/probe_battery.py --model qwen3-32b --condition T0 --splits reserve --samples 4` (classify, classify_single, identify; scored offline). Unparsed answers are excluded, as in every B3 fit.

## Parameters (frozen)
No model is refitted. Every model keeps the parameters estimated on the processing-calibration families and saved in `results/study_b/process_fits/qwen3-32b_T0.json`; the sha256 of those parameters (computed as in `confirm`) is `01f1ca5fcddac14c2c198992f4d1d3b421ce9d5b0e06d9dc384a95b7505b1332`.

| Model | Parameters |
|---|---|
| M0 | ζ 0.5116 |
| M1 | ν 0.4242, ζ 0.1070 |
| M2c | α 0.0002, ν 0.4242, ζ 0.1070 |
| M2s | α 0.0001, ν 0.4156, ζ 0.1107 |
| M3 | k0 37.68, k1 0.036, ζ 0.5115 |
| M1a | ν_x 0.2765, ν_r 0.0674, ζ 0.2241 |

## Test
Held-out log likelihood of the reserve answers under each model.

**M1a is confirmed if both hold:**
1. its total log likelihood is higher than that of every planned model (M0, M1, M2c, M2s, M3), and
2. the 95% family-bootstrap interval (2,000 resamples of the 40 families, seed 20261118) of the difference between M1a and the best planned model lies entirely above 0.

Otherwise M1a is reported as not confirmed, and B3's result is the planned comparison alone. Either way, the planned comparison stays the primary B3 result and is reported as specified.

Also reported (descriptive, no test): the difference to every planned model with its interval, the number of families favoring M1a, and observed vs. predicted error rates by probe type.

## Commands
```
python experiments/study_b/probe_battery.py --model qwen3-32b --condition T0 --splits reserve --samples 4   # on the cluster
python experiments/study_b/fit_process_models.py confirm --model qwen3-32b --condition T0
```
Output: `results/study_b/process_fits/qwen3-32b_T0_reserve_confirmation.json`.
