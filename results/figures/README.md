# Figures

Drawn by `python experiments/make_figures.py` from the result files named in each caption. The same command runs
at the end of `experiments/reproduce_summaries.sh`.

**The set.** Four figures for the paper, one per question in the order of the text, plus one supplementary figure.

**Formats.** Each figure is a vector PDF for the paper and a PNG for the poster and slides. All are sized for the
full width of a two-column page (6.9 in) and set in Arial at print size.

**Reproducibility.** The output is byte-identical on every run.

**Intervals.** Intervals shown only in the figures are recomputed from the records by resampling the unit of each
study: people in Study A, state families in B3 and C. Every recomputed point estimate is checked against the
result file it comes from; the script stops if one differs.

**Colors.** Each entity keeps its color in every figure: people orange, models blue, descriptive or exploratory
entries hollow gray. Every series also differs in marker shape, so the figures read in grayscale. The palette
passes the color-vision-deficiency checks on white.

## Figure 1 (`fig1_overview`): design

**One task, three studies and a preregistered prediction test.**

**(A)** The task. Each supplied guess is followed by its feedback:
- a black peg for a right color in the right place;
- a white peg for a right color in the wrong place.

With 4 positions and 4 colors there are 256 codes, so ideal inference is exact.

**(B)** The three studies:
- Study A reproduces the human findings and the comparison of the bounded, leaky and unbounded models.
- Study B puts language models on the same positions and fits the same human models and process models.
- Study C gives one model a structured reasoning aid and tests whether the fitted process model predicts its
  effects.

**(C)** The prediction test. U is the chance that the model names the secret after seeing a query's answer. Γ is
the aid's change in the difference between two queries' usefulness. It was predicted for 360 pairs and locked on
OSF before the data were collected.

## Figure 2 (`fig2_humans_vs_models`): people and language models (B1, B1b)

**People show the informativity–interpretability signature; the language models that play well do not.**

One row per agent.

**(A)** Extra guesses after informative but complex (Max-EIG) histories compared with Simple ones, on the
Experiment 1 starting positions.
- Data: people, 393 games (44 people); each model, 1,320 games (132 positions × 10).
- Unsolved games count as 16.
- Bars: 95% intervals from resampling starting positions within condition.
- Below the line are models that solve fewer than half their free-play games. They are descriptive: point
  estimates only.
- Vertical line: the ideal player.

**(B)** Guesses to solve after Max-EIG histories that leave exactly one possible code. People played 46 such games
and each model 190. The ideal player needs 1.

**(C)** The human (bounded) model on free play: its AIC advantage over the ideal observer (the unbounded model),
per decision.
- Dots: the 39 people (Study A, converged fits). The bar marks their median, 1.49.
- Models that pass the solve-rate rule: the pooled fit, with a 95% interval from 200 resamples of games. All
  three fall below almost every person.
- Models below the rule were not fitted.

*Sources:* `study_b/b1_summary.json`, `study_b/b1b_<model>_T0_converged.json` and `study_a/e3_fits_converged/`.

## Figure 3 (`fig3_evidence_use`): how Qwen3-32B uses evidence, and what the aid changes (B2/B3, C-a)

**Qwen3-32B's evidence use is noisy. The structured aid does not make it more exact: it raises the noise.**

**(A)** Error rates of the candidate judgments under the standard prompt, by number of clues, on the
processing-calibration and confirmation families:
- a still-possible code ruled out;
- a near miss kept (a code ruled out only by the newest clue);
- a far miss kept.

95% intervals from 10,000 resamples of families.

**(B)** Held-out log likelihood of each process model minus exact inference with reporting noise (M0), on the 80
held-out families.
- 95% intervals from 2,000 resamples of families.
- M1, M2c and M2s are almost identical. M1a was added after the planned comparison and is exploratory.

**(C)** Change in each rate with the structured aid (T3), against the standard prompt (T0) and against the
length-matched prompt (T1), on 140 families.
- 95% intervals from 10,000 paired resamples of families.
- Upper group: clue-checking errors (higher is worse). Lower group: naming a code.

**(D)** The fitted noise of evidence use (ν in M2s) under each prompt.
- 95% intervals from 1,000 resamples of families per prompt.
- The paired difference between the aid and the standard prompt is printed.

*Sources:* `study_b/process_fits/qwen3-32b_error_profile_T0.json`, `qwen3-32b_T0.json`,
`qwen3-32b_T0_intervals.json` and `study_c/ca_aid_effects.json`.

## Figure 4 (`fig4_prediction_test`): the preregistered tests (C-c, C-b; 360 pairs, 120 fresh families)

**The fitted model does not predict how the aid changes each question's usefulness, and the model's own choices
do not follow those changes.**

**(A)** Calibration of the support map.
- The pairs are split into four equal bins by predicted Γ.
- Each point shows the bin's mean predicted Γ and its mean observed Γ, with 95% intervals from 2,000 resamples of
  families.
- The diagonal marks perfect prediction. The slope over all 360 pairs is printed.

**(B)** H3: the family-level mean squared prediction error of each rival minus that of the support map.
- 95% intervals from 10,000 resamples of families.
- Every point estimate favors the rival, and no difference is reliable.

**(C)** H4: the shift in the model's choice of q1 (aid minus standard, in logits) against observed Γ.
- The least-squares line has a 95% band from 2,000 resamples of families.
- The preregistered slope and its interval are printed.

*Sources:* `study_c/cc_score.json`, `study_c/cb_choice_shift_c_confirmation.json` and
`prereg/prediction_lock.json`.

## Figure S1 (`figS1_human_reproduction`): Study A

**The human findings reproduce. The bounded model fits and predicts people best, by a smaller margin over the
leaky model than reported.**

**(A)** Experiment 1: guesses after the supplied history, by history type.
- People: solved games, with 95% intervals from 10,000 resamples of people.
- Hollow markers: the ideal player.

**(B)** Experiment 3: for each person, the AIC of the leaky and of the unbounded model minus that of the bounded
model (converged fits, 39 people). Bars and labels mark the means. The bounded model fits best for 33 people.

**(C)** Predicting new people: the bounded model's held-out log likelihood minus each rival's.
- Each person is predicted from the other 38 people's median parameters.
- 95% intervals from resampling people.

*Sources:* `study_a/e1_reproduction.json`, `study_a/e3_fits_converged/`,
`study_a/e3_model_comparison_converged.json` and `study_a/e3_heldout_converged.json`.
