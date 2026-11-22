# Study C confirmation: analysis plan (draft until the lock)

Written before any confirmation data exist. The cap, the number of families and the samples per branch were fixed
by the cap pilot, the calibration and the power analysis below, before `prediction_test.py lock`; after the lock
nothing here changes.

## Questions (from the approved proposal)
- **H3 (C-c).** The fitted models predict in advance how the structured aid (T3) changes each question's usefulness
  on held-out game states, better than simpler baselines based on problem difficulty.
- **H4 (C-b).** The model's own question choices shift toward the questions the aid makes more useful.

## Model, conditions, prompts
- Qwen3-32B (revision in `configs/model_registry.json`), thinking off, vLLM, temperature 0.7, top-p 0.95.
- T0 (standard) and T3 (structured aid: check every candidate against every clue, step by step); the frozen
  templates of `configs/prompt_freeze_v1.json`. The choice prompt carries the same guidance as its condition.
- Reply cap 6,144 tokens in both conditions (the registry cap of every Study B run), kept after the cap pilot
  (`cap_pilot.py`, `results/study_c/cap_pilot_qwen3-32b_cap15000.json`). At 6,144 tokens 25.2% of aided (T3) and
  9.8% of standard (T0) usefulness replies were cut off. The pilot redrew 200 T3 and 60 T0 cut-off replies with
  identical prompts at 15,000 tokens. Of the redraws that ran past 6,144 tokens, 16 of 69 (T3) and 0 of 5 (T0)
  finished by 15,000, and none of those named a still-possible code; the rest ran to 15,000 without an answer.
  A 15,000 cap would lower the T3 cut-off share to about 19%, leave T0 unchanged and leave usefulness unchanged.
  Unfinished replies are therefore treated as a property of the condition (they name nothing), and each
  condition's share of them enters the model-based predictions.

## Data
- Fresh families: `data/splits/families_c_v1.json` (`draw_families.py`, seed 20261120). None is equivalent under
  symmetry to an earlier family or an Experiment 1 position, and none shares a pair of clues with an earlier
  family.
- Pairs: `data/splits/confirmation_pairs_c_v1.json` (`prediction_test.py select`), chosen without outcomes, one
  pair per stratum per family:
  - (a) equal V*, largest predicted change;
  - (b) similar EIG (within 0.2 bits), different numbers of digits;
  - (c) predicted change near 0.
- 120 families (360 pairs, 2,713 answer branches), 8 samples per answer branch, 8 choices per pair and
  condition: 43,408 usefulness calls and 5,760 choice calls over both conditions. Fixed by the power simulation (`prediction_test.py power --r2 0.20
  --pairs-per-family 3`, `results/study_c/cc_power.json`): power 0.864 at α = 0.0125 (one-sided) to detect a
  predictor that explains 20% of the true variance of Γ (true SD of Γ 0.053, sampling-noise SD 0.065 at 4 samples
  per branch, both estimated on the 96 calibration pairs). 4 samples × 160 families has similar power (0.845) for
  fewer calls, but 8 samples halve the noise variance of each pair's Γ, which the choice test (H4) regresses on.

## Measures
- Usefulness U(q): the probability that the model names the secret after seeing q's answer,
  (1/K) Σ_a P(named code ∈ S_a). It is summed exactly over answer branches, with 8 samples per
  branch. A reply with no valid code (including an unfinished one) counts as naming nothing.
- Γ = [U(q1,T3) − U(q2,T3)] − [U(q1,T0) − U(q2,T0)] per pair.
- Choice shift: logit P(choose q1 | T3) − logit P(choose q1 | T0), adding 0.5 to the counts.

## Predictors (fitted on calibration data only; frozen in `prereg/prediction_lock.json`)
- **Primary, support_map:** the B3 process model M2s with its T0 parameters, plus one support parameter δ fitted
  on the support-calibration answers under T3. δ may be negative (the aid can make clue checking worse).
- Secondary, fitted_M2s: M2s with the parameters fitted to the T0 and T3 probe batteries.
- Exploratory, fitted_M1a: the same with the two-rate model M1a.
- Rivals:
  - difficulty_regression: ridge regression on V*, EIG, codes remaining, clues, distinct digits and predicted
    baseline usefulness, with the penalty chosen by leave-one-family-out cross-validation;
  - reporting_only: only the reporting noise changes;
  - uniform_gain: Γ = 0;
  - ideal_noise: an exact reasoner whose reporting noise differs between conditions.
- Model-based predictions include each condition's calibration share of unfinished replies.

## Tests
- **H3 (primary):** family-level mean squared prediction error (MSPE) of Γ. For each rival, the difference
  MSPE(rival) − MSPE(support_map), with a cluster bootstrap over families (10,000 resamples) and one-sided p.
  The p-values are Holm-adjusted across the four rivals at α = 0.05. H3 is supported only if support_map beats
  every rival.
- **H3, also reported:**
  - noise-corrected MSPE;
  - bias;
  - calibration slope with interval;
  - sign accuracy on the largest quarter of observed effects;
  - reversal rate (the aid flips which query is more useful);
  - results by stratum;
  - the secondary and exploratory predictors against the rivals (no Holm).
- **H4:** the OLS slope of the choice shift on measured Γ over pairs; one-sided test that it is positive, with a
  cluster bootstrap over families. Also reported: the slope on the primary predictor's Γ̂, the split-half
  reliability of measured Γ, and the slope corrected for it. H4 is one of the course's primary tests, and the
  Holm correction across B1, B3 and H4 is reported in the cross-study analysis.

## Stopping and exclusion rules
- Every planned call is run. Transport errors and server outages are retried, up to 3 times per call, through the
  reply cache, so a retry never changes a recorded reply. A reply is never rerun because of its content.
- Replies without a valid code count as naming nothing (usefulness) or as missing (choices). Unparsed choices are
  counted and reported; a pair is dropped from H4 only if a condition has no parsed choice.
- No family, pair or state is dropped for its result. Pairs with missing branches are listed, and the
  intention-to-evaluate set (all locked pairs) is reported with its missingness.
- Nothing is tuned on confirmation data, and the analysis code is the locked code: `score` checks the hashes in
  the lock.

## Disclosed before the lock
- The C-a battery ran T0–T3 on the old confirmation families. C-c uses fresh families for that reason.
- A first run of the cap pilot was discarded: its 600-second client timeout aborted long replies and redrew them,
  so the stored replies over-represented short ones. Those replies were moved out of the reply cache, the timeout
  was raised to 4 hours with no automatic retries, and the pilot was rerun (0 retries). An audit of every cached
  reply found the same problem only in Qwen3-32B's B1 game play, not in the probe batteries or the usefulness runs
  used here.
- Calibration showed that the process models do not predict baseline usefulness across queries (correlation near
  0), and that Γ is measured with low reliability (0.20 at 4 samples per branch). H3 is therefore not expected
  to be supported, and the design is sized so that a null result is informative.
