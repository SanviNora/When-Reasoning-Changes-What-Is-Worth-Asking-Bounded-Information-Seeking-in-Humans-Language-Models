# Results

Outputs of the scripts in `experiments/`, one folder per study. The findings are in `study_a/README.md`,
`study_b/README.md` and `study_c/README.md`, with every number traced to its file. The Holm correction across
the primary tests is below.

File and folder suffixes:
- none: the paper's greedy search;
- `_converged`: converged fits (primary);
- `_with_unsolved`, `_early` / `_late`, `_cacherep2` and `_wide`: sensitivity fits.

| Folder | Contents |
|---|---|
| `study_a/` | Experiment 1–3 reproduction summaries; per-participant fits of the three human models (`e3_fits*/`), recovery (`e3_recovery*/`) and their summaries |
| `study_b/screening/` | screening of the candidate models (records and summaries) |
| `study_b/b1/` | B1 game records, one file per model, task and condition |
| `study_b/b1b/` | B1b fits: pooled, 10 blocks and 200 resamples per model; summaries in `b1b_<model>_T0*.json` |
| `study_b/battery/` | B2 probe-battery records (Qwen3-32B, T0) |
| `study_b/process_fits/` | B3 fits, family-bootstrap resamples, intervals, error profile, product-rule check and M1a confirmation |
| `study_b/process_recovery*/` | B3 recovery replicates (planned design; at the fitted values) |
| `study_b/usefulness_choice/`, `study_b/native_choice/` | Native question choice: records and analysis. Usefulness and choice records under T0 and T3 for Study C, on the calibration families and on the confirmation pairs (`*_c_confirmation*`) |
| `study_b/battery/*_T1–T3_*`, `study_b/process_fits/*T1–T3*` | Study C aid conditions: probe-battery records, process-model fits and family-bootstrap resamples |
| `study_c/` | C-a aid effects (`ca_aid_effects.json`); reply-cap pilot; C-c calibration (`cc_calibration.json`), power analysis (`cc_power.json`) and preregistered score (`cc_score.json`); C-b choice shift on the confirmation pairs (preregistered) and on the calibration pairs (exploratory); post hoc design check (`cc_noise_check.json`) |
| `primary_tests.json` | Holm correction across the course's primary tests (below) |
| `figures/` | the paper's four figures (design; people vs. models; evidence use and the aid; the preregistered tests) and one supplementary figure (Study A), as PDF and PNG. Captions are in `figures/README.md` |

## Primary tests across studies (`primary_tests.json`)

The project plan names three primary tests for the course (B1, B3 and C-b) and a Holm correction across them
(restated in `prereg/analysis_plan.md`, Tests). It does not say how B1's several
models or B3's two rivals enter the family. That was fixed here, after the data:

- **B1:** one test per model that passes the solve-rate rule. Two-sided, as reported in Study B, because the
  alternative includes the ideal player's direction.
- **B3:** the best planned structured account against both unstructured rivals. The p-value is the larger of the
  two one-sided bootstrap p-values, computed on the same family resamples as the reported intervals.
- **H4:** as preregistered.

`experiments/primary_tests.py` computes the family, Holm at α = 0.05.

| Test | Estimate [95% CI] | p | Holm p | Result |
|---|---|---|---|---|
| B1 gpt-oss-120b: Max-EIG minus Simple (guesses; mixed model) | +0.225 [−0.24, 0.69] | 0.35 | 0.69 | not significant |
| B1 Claude Haiku 5.5 | −0.556 [−0.90, −0.21] | 0.0018 | 0.0072 | significant, in the ideal player's direction (faster after Max-EIG), not the human one |
| B1 Claude Opus 5.5 | −0.785 [−1.07, −0.50] | 5 × 10⁻⁸ | 2 × 10⁻⁷ | significant, in the ideal player's direction |
| B3: M2s minus M1 in held-out log likelihood (the binding rival; M2s minus M0 is 429 [333, 515], p < 0.001) | +4.5 [−1.2, 10.9] | 0.072 | 0.21 | not significant |
| H4: slope of the choice shift on measured Γ | −0.63 [−2.50, 1.32] | 0.75 | 0.75 | not significant |

**Reading the table:**
- No eligible model shows the human signature effect.
- Qwen3's evidence use is noisy, but the structured account's edge over plain noise is not reliable on held-out
  families.
- The model's choices do not track the aid's change in usefulness.

H3 (C-c) is the paper's primary test, not one of the course's three. It has its own preregistered Holm correction
across its four rivals, and it is not supported (`study_c/cc_score.json`).
