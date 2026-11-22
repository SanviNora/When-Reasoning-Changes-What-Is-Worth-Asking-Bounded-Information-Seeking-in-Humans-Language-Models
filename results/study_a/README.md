# Study A: human reproduction and model comparison

Data: Cheyette et al., OSF `vjrqu` (E1, E2, E3).
Every result below is read from the JSON file named in its row (the paper's values, its sample sizes and the
search settings come from the paper and the code). `bash experiments/reproduce_summaries.sh`
regenerates them from the saved fits; the scripts are listed in `experiments/study_a/README.md`.

## Behavior

| Measure | Ours | Paper | File |
|---|---|---|---|
| E1 sample | 44 people, 393 games, 132 distinct starting positions | same | `e1_reproduction.json` |
| E1 guesses after the supplied history (solved games): Simple / Max-EIG / none | 4.74 / 6.32 / 7.15 | (figure only) | `e1_reproduction.json` |
| E1 regression as in the paper (guesses as made, all Simple and Max-EIG games): Simple minus Max-EIG; initial entropy (nats) | **B = −2.31, t(283) = −5.92; B = 0.62, t = 4.96** | B = −2.29, t(283) = −5.89; B = 0.62, t = 4.94 | `e1_reproduction.json` |
| E1 ideal continuation: Simple / Max-EIG / none | 2.49 / 1.91 / 3.68 | (not reported) | `e1_reproduction.json` |
| E1 Max-EIG histories that leave exactly one code (ideal needs 1 guess) | people need 5.41 (the 44 solved of 46 games) | (not reported) | `e1_reproduction.json` |
| E1 Max-EIG minus Simple, unsolved = 16, clustered by person | 1.29 [0.29, 2.29] | | `e1_reproduction.json` |
| … controlling for supplied guesses and log2 codes remaining | **2.41 [1.34, 3.48]** | | `e1_reproduction.json` |
| E2 (descriptive only; 33 of 40 people released) | easy choices rise from 0.39 (game 1) to 0.69 (game 10); easy choices predict solving (logit slope 1.79 [0.95, 2.63]) and fewer guesses (−3.84 [−5.15, −2.52]) | same direction (authors' R script) | `e2_descriptive.json` |
| E3 sample after cleaning | 39 people, 390 games | 39 | `e3_reproduction.json` |
| E3 solve rate | 0.864 | 0.86 | `e3_reproduction.json` |
| E3 guesses to solve | **7.72** (SD 3.16) | **7.5** | `e3_reproduction.json` |
| Ideal (greedy Max-EIG), all 256 secrets | 3.56 (max 4) | 3.6 | `e3_reproduction.json` |
| Random-query benchmark | 4.81 (SD 1.26, max 12) | 4.8 (SD 1.0, max 7) | `e3_reproduction.json` |
| Uncertainty before guesses 1–8 (log2 codes still possible, mean over games, 0 once solved) | 8.0, 5.51, 3.39, 1.84, 1.16, 0.57, 0.27, 0.12 | (figure only) | `e3_reproduction.json` |
| Information per guess (bits, mean) | 1.61 | | `e3_reproduction.json` |
| Guesses still possible given earlier feedback | 0.31 | | `e3_reproduction.json` |
| Guesses using 1 / 2 / 3 / 4 colors: people vs. ideal player | 0.18 / 0.21 / 0.42 / 0.18 vs. 0.004 / 0.25 / 0.66 / 0.08 | | `e3_reproduction.json` |
| First guesses using a single color | 0.45 | | `e3_reproduction.json` |

## Model comparison (E3, solved games, per participant)

The three models are fitted to each person's free-play guesses in two ways, and both are reported:

- **Paper's procedure** (greedy stochastic search, as in the paper): the reproduction.
- **Converged fits** (primary): the paper's search often stops before the maximum. With it, 9 of 39 leaky fits
  have a lower log likelihood than the unbounded model nested in them, which cannot happen at the maximum. The
  converged procedure keeps the same models, likelihood and parameter limits, starts from the greedy-search fit
  and from nested and other fixed starting points, and polishes each start with bounded Nelder-Mead until a rerun
  gains less than 0.01. Its rules are written in `mastermind/human_models.py` and were committed before any
  converged fit was run. Converged fits are never below the greedy fit they start from (checked for every file).
  Average gain in log likelihood over the greedy search: unbounded 2.3, leaky 41.9, bounded 14.0 per person.

| Measure | Converged (primary) | Paper's procedure | Paper | Files |
|---|---|---|---|---|
| Mean ΔAIC, bounded vs. unbounded | **128.1** | 104.6 | 82.5 | `e3_model_comparison[_converged].json` |
| Mean ΔAIC, bounded vs. leaky | **27.8** | 83.5 | 72.3 | same |
| People best fit by bounded / leaky / unbounded | **33 / 6 / 0** | 34 / 5 / 0 | 37 of 39 favor bounded | same |
| Held-out participants (each person predicted from the median parameters of the other 38): total log likelihood bounded / leaky / unbounded | **−12,036 / −12,438 / −13,283** | −13,082 / −13,429 / −13,407 | | `e3_heldout[_converged].json` |
| … bounded minus leaky (bootstrap over people, 95% CI) | **402 [110, 692]**, 26 of 39 favor bounded | 347 [−87, 726], 29 of 39 | | same |
| … bounded minus unbounded | **1,247 [794, 1,661]**, 33 of 39 | 325 [−103, 701], 28 of 39 | | same |
| … best held-out model per person, bounded / leaky / unbounded | 20 / 13 / 6 | 28 / 6 / 5 | | same |
| Game-index drift (static bounded fit, per-decision log likelihood vs. game 1–10) | +0.076 per game (SE 0.022, p = 0.0006) | +0.075 (SE 0.023, p = 0.0009) | | same |
| Sensitivity: including unsolved games | 122.7 / 36.1; best 35 / 4 / 0 | 98.4 / 90.2; best 37 / 2 / 0 | 37 of 39 | `e3_model_comparison_with_unsolved[_converged].json` |
| Sensitivity: bounded fitted to games 1–5 and 6–10 separately (8 parameters) vs. one static fit (4) | separate fits better for **29 of 39** (sign test p = 0.003); mean AIC gain 13.6, median 3.6; ε slightly lower late (median −0.02, Wilcoxon p = 0.02), which does not survive Holm correction across the 4 parameters | 21 of 39 (p = 0.75); mean 9.7, median 0.7 | | `e3_game_index_halves[_converged].json` |
| Cache seed: bounded refitted with an independently rebuilt cache | 123.1 / 22.8; best 29 / 10 / 0; bounded AIC r = 0.996 across caches; α Spearman 0.79 | 108.4 / 87.2; best 35 / 4 / 0; r = 0.986; α Spearman 0.55 | | `e3_cache_seed[_converged].json` |
| Sensitivity: wider parameter limits (τ down to 10⁻⁶, λ up to 10⁵) | 146.9 / 16.1; best 30 / 9 / 0 | | | `e3_model_comparison_converged_wide.json` |
| Recovery: each model simulates every person (their fitted parameters, same number of solved games); all three refitted | bounded 32/39 (0.82), unbounded 29/39 (0.74), leaky 34/39 (0.87); bounded data come back as leaky for 7, leaky data as bounded for 5; α Spearman 0.95 | bounded 35/39 (0.90), unbounded 29/39 (0.74), leaky 28/39 (0.72); α Spearman 0.91 | | `e3_recovery_summary[_converged].json` |

**Parameters sit at their limits.** In the converged fits the choice temperature τ is at its lower limit (10⁻³), or within 1%
of the log search range from it, for 30 of 39 bounded and 37 of 39 leaky fits, and the noise ε is near its upper limit (0.5) for
many leaky fits (median ε 0.45, τ 0.001; bounded median α 0.14). The unbounded model's τ is at its upper limit (100)
for 35 of 39 (`parameters_at_a_limit` in `e3_model_comparison_converged.json`). With wider limits the likelihood keeps rising toward ε → 0.5 and
τ → 0: each piece of feedback carries almost no evidence, and choices are very sharp. The model ranking is
stable across limits, caches and data subsets; the individual parameter values are not interpretable on their own,
so people and models are compared on ΔAIC per decision rather than on raw parameters.

**What holds and what does not.** The unbounded (ideal-observer) model is never the best in-sample fit, for any
person under either procedure or any check; held out, it predicts 6 of 39 people best (5 with the paper's
procedure). The bounded model is best for most people in every check and, with converged fits, predicts held-out
people better than the leaky model (interval above 0; with the paper's procedure the interval includes 0), but
its margin over the leaky model is well under half of the paper's (27.8 vs. 72.3) and shrinks further with wider limits (16.1). Recovery shows the same weak spot: the
converged procedure recovers each model for 74–87% of simulated people, and the bounded–leaky pair is the one it
confuses most (7 of 39 bounded data sets come back as leaky, 5 of 39 leaky data sets as bounded).

## Discrepancies to report

1. **E3 guesses to solve:** 7.72 vs. the paper's 7.5, from the same cleaned 39 participants.
2. **The paper's search does not reach the maximum likelihood** for many people (above). With it, our ΔAIC values
   are 104.6 / 83.5 (paper 82.5 / 72.3) and 34 of 39 favor bounded (37 with unsolved games, as in the paper). Our
   implementation of that search (one parameter changed at a time, several starting points, stop after 100 proposals
   without a gain of 1) also stops after 3,000 likelihood evaluations in total, shared by its starting points; the
   paper gives no such cap. It binds for 9 of 39 bounded fits (4 of 39 with unsolved games; `evaluation_cap_reached`
   in `e3_model_comparison[_with_unsolved].json`).
   With converged fits the advantage over unbounded grows (128.1) and over leaky shrinks (27.8). Remaining
   differences from the paper may also come from search choices the paper does not specify for the cache (chain
   settings, proposal correction, size cap; recorded in the cache manifest), since the authors' cache is unreleased.
3. **People change over the session.** Later games fit the static model better, and with converged fits separate
   early/late parameters improve the fit for 29 of 39 people. The static model is the paper's and is kept as the
   summary, with this caveat.
4. **Random-query benchmark:** mean matches; the spread does not. No simple rule reproduces SD 1.0 and max 7
   (without repeats: 4.74 / 1.11 / 10; possible codes only: 3.78 / 0.78 / 6).
5. **E1 analysis:** our pipeline is Python. The authors' E1 R script calls `mm_plot` (line 63) before defining it
   (line 109); we did not use it.
