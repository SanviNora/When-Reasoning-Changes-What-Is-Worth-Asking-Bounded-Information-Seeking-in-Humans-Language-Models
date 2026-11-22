# Preregistration (Study C confirmation)

Committed before any confirmation data were collected, and registered on OSF.

| File | Contents |
|---|---|
| `analysis_plan.md` | hypotheses H3 and H4, data, measures, predictors, tests, stopping and exclusion rules, disclosures |
| `prediction_lock.json` | the predicted Γ of every predictor for all 360 confirmation pairs, the frozen calibration, the design, and SHA-256 hashes of the analysis code, families, pairs, calibration, plan, prompts and model registry |

`python experiments/study_c/prediction_test.py score` refuses to run if any hashed analysis file has changed since
the lock.
