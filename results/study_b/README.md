# Study B: language models on the human paradigm

Every number below is read from the JSON file named in its row. `bash experiments/reproduce_summaries.sh`
regenerates all of them from the saved records and fits; the scripts are listed in `experiments/study_b/README.md`.

## Models

Admitted by the floor rule fixed before any B1 run (screening on 20 development states: format compliance
≥ 90%, identification above chance 0.03). Versions and decoding settings are in `configs/model_registry.json`.

| Model | Role | Screening: format / identification | Admitted |
|---|---|---|---|
| Qwen3-32B (thinking off) | depth model (B2, B3, native choice) and panel | 0.95 / 0.175 | yes |
| gpt-oss-120b (reasoning model) | panel, largest open-weight | 0.96 / 0.825 | yes |
| Phi-4 | panel, smallest | 1.00 / 0.075 | yes |
| OLMo-2-32B | panel | 1.00 / 0.05 | yes |
| Mistral-Small-24B | panel candidate | 0.57 / 0.275 | **no** (format below 90%) |
| Claude Haiku 5.5 (thinking off: standard mode) | panel, proprietary | 0.98 / 0.90 | yes |
| Claude Opus 5.5 (cannot switch thinking off; low effort: reasoning mode) | panel, proprietary | 1.00 / 1.00 | yes |

Files: `screening/<model>_summary.json`; the panel is recorded in `configs/model_selection.json`. Open-weight
models run on vLLM at temperature 0.7, top-p 0.95; the Claude API takes no sampling parameters, so its defaults
apply. Both proprietary models are Claude models: the plan asks for one model per family, but also for proprietary
models in standard and reasoning modes, which these two cover.

## Rules fixed before the analyses (disclosed)

1. **Solve-rate rule.** Inferential B1 results and B1b are reported only for models that solve at least half of
   their 100 free-play games, the same rule the paper used to exclude people. Models below it are reported
   descriptively. The rule was stated after the solve rates were seen and before any contrast was run.
2. **Two fitting procedures for the human models** (B1b), as in Study A: the paper's greedy search and converged
   fits (primary). People and models are always compared within one procedure.
3. **Not done** (in the detailed plan but not the proposal): a small probe-battery subset for the panel models, and
   hand-coding of probe errors as arithmetic vs. logic.

## B1: the human signature effect (`b1_summary.json`)

Each model continues each of the 132 Experiment 1 starting positions 10 times (1,320 games; people: 393 games,
44 people), under T0. Measure: guesses after the supplied history, unsolved games counted as 16 (one more than the
15-guess limit, which counts only the player's own guesses, for people and models alike). Signature effect: Max-EIG
minus Simple histories, 95% CI from 20,000 resamples of stimuli within condition (people are resampled the same way
here, so their interval differs slightly from the person-clustered one in Study A, 1.29 [0.29, 2.29]). The ideal
player is faster after Max-EIG histories (−0.59).

| | Max-EIG minus Simple | Relative to ideal | Mixed model (random intercept per stimulus) | Minus people | Ratio to people | Hazard check: log odds of solving per turn after Max-EIG | Max-EIG histories that leave one code: mean guesses |
|---|---|---|---|---|---|---|---|
| People | **1.29 [0.29, 2.28]** | 1.88 | 1.265 [0.28, 2.25] | | | −0.47 (p = 0.003) | 5.87 (46 games; 5.41 over the 44 solved, as in Study A) |
| gpt-oss-120b | 0.23 [−0.23, 0.69] | 0.81 | 0.225 [−0.24, 0.69] | −1.07 [−2.16, 0.03] | 0.17 [−0.24, 0.98] | −0.12 (p = 0.46) | 2.49 (190 games) |
| Claude Haiku 5.5 | −0.56 [−0.90, −0.21] | 0.03 | −0.556 [−0.90, −0.21] | −1.85 [−2.89, −0.79] | −0.43 [−1.71, −0.13] | +0.47 (p = 0.006) | 1.53 |
| Claude Opus 5.5 | −0.79 [−1.07, −0.51] | −0.20 | −0.785 [−1.07, −0.50] | −2.08 [−3.11, −1.03] | −0.61 [−2.37, −0.29] | +1.20 (p < 0.001) | 1.02 |
| Ideal player | −0.59 | 0 | | | | | 1 |

- **No eligible model shows the human signature.** People are slower after informative but complex histories;
  Haiku and Opus are faster after them, like the ideal player; gpt-oss lies between (its difference from people,
  −1.07 [−2.16, 0.03], just includes 0). On histories that leave a single code, people need 5.87 guesses, the
  models 1.02–2.49. Ratios between effects of opposite sign (Haiku, Opus) carry little meaning.
- **People's unsolved games.** People's 23 unsolved games ended before the limit (after 0 to 11 of their own
  guesses), so for them 16 is an imputed value, not a count. Over solved games only, people's effect is 1.58 (models:
  gpt-oss 0.14, Haiku −0.56, Opus −0.79). The hazard check censors every unsolved game after its last guess.
- **Models below the solve-rate rule** (descriptive: point estimates only, no intervals or tests). Qwen3-32B and
  Phi-4 are slower after Max-EIG histories, the human direction: Max-EIG minus Simple 1.86 and 1.94 (over solved
  games 0.97 and 2.18; hazard log odds −0.48 and −0.72), and they solve fewer games after them (Qwen3 0.414 vs. 0.564,
  Phi-4 0.182 vs. 0.327). OLMo-2-32B almost never solves (free play 0.03; Max-EIG minus Simple 0.02). At these solve rates the measure mostly counts
  unsolved games, which is why the rule keeps them out of inference.
- **Invalid turns** (no valid guess after 2 re-asks): none for gpt-oss, Haiku, Opus, Phi-4 or OLMo; Qwen3-32B 7 of
  14,710 turns (4 after Max-EIG, 2 after no history, 1 after Simple). Re-asks were needed on 4.5–7.3% of Qwen3's
  turns and on at most 0.3% of every other model's.

**Free play** (100 seeded games per model; the Study A measures, computed by the same code for people and models;
guesses to solve over solved games only; EIG and V* of each guess relative to the best available query):

| | Solve rate | Guesses to solve | Bits per guess | Guesses still possible | EIG relative to best | V* relative to best | Guesses using 1 / 2 / 3 / 4 colors | First guess uses one color |
|---|---|---|---|---|---|---|---|---|
| Phi-4 | 0.19 | 8.11 | 1.215 | 0.066 | 0.478 | 0.571 | 0.002 / 0.271 / 0.267 / 0.460 | 0.00 |
| OLMo-2-32B | 0.03 | 10.33 | 1.547 | 0.041 | 0.585 | 0.662 | 0.048 / 0.445 / 0.364 / 0.143 | 0.00 |
| People (E3, 390 games) | 0.864 | 7.72 | 1.611 | 0.312 | 0.652 | 0.635 | 0.184 / 0.213 / 0.421 / 0.183 | 0.45 |
| Qwen3-32B | 0.44 | 7.70 | 1.724 | 0.154 | 0.686 | 0.678 | 0.006 / 0.125 / 0.366 / 0.503 | 0.00 |
| gpt-oss-120b | 0.98 | 4.64 | 2.260 | 0.504 | 0.848 | 0.788 | 0.006 / 0.445 / 0.297 / 0.252 | 0.00 |
| Claude Haiku 5.5 | 1.00 | 4.48 | 2.402 | 0.635 | 0.904 | 0.859 | 0.007 / 0.333 / 0.455 / 0.205 | 0.01 |
| Claude Opus 5.5 | 1.00 | 3.69 | 2.623 | 0.683 | 0.942 | 0.885 | 0.005 / 0.393 / 0.463 / 0.138 | 0.00 |
| Ideal player (256 secrets) | 1.00 | 3.56 | 2.825 | 0.793 | 1.00 | 0.998 | 0.004 / 0.254 / 0.659 / 0.082 | 0.00 |

Rows are ordered by EIG relative to the best query (the `capability` section of `b1_summary.json`). People's
simplest query, one color, is 18% of their guesses and 45% of their first guesses; every model almost never uses
it (at most 5% of guesses, OLMo). Two-color guesses are more common than in people in every model except Qwen3-32B.

**Capability** (`capability` in `b1_summary.json`; descriptive). Among the models that pass the solve-rate rule,
the signature effect falls as query quality rises (gpt-oss 0.23, Haiku −0.56, Opus −0.79), and none shows the
human pattern. A human-sized effect appears only below the rule, in the two models that solve some games (Qwen3-32B,
whose query quality is closest to people's, 0.686 vs. 0.652, and Phi-4), and not in OLMo-2-32B, which almost
never solves. This is the pattern the plan predicted (human-like at intermediate capability, none for models that
mostly guess or that approach the ideal player), but it rests on point estimates from models outside the rule, so it
is not tested here. B1b's per-decision advantage of the bounded model shows no trend with capability (gpt-oss
0.159, Haiku 0.364, Opus 0.160, converged fits).

**Aid arm** (`aid_arm_T3` in `b1_summary.json`): Qwen3-32B under T3 (structured constraint checking) on the same
positions, 1,320 games; descriptive, since Qwen3 is below the solve-rate rule. With the aid it solves fewer games
after every kind of history than under T0 (Simple 0.411 vs. 0.564, Max-EIG 0.267 vs. 0.414, none 0.242 vs. 0.363)
and needs more guesses (11.64 vs. 10.50, 13.18 vs. 12.37, 13.65 vs. 13.05); Max-EIG minus Simple is 1.54 vs. 1.86.
The aided replies are longer and more often reach the reply cap before naming a guess: re-asks on 9.0–14.5% of
turns (T0: 4.5–7.3%) and 54 invalid turns of 15,734 (T0: 7 of 14,710).

## B1b: the human models fitted to model free play (`b1b_<model>_T0[_converged].json`)

The three Study A models are fitted once to each eligible model's solved free-play games (one agent; each person's
fit uses their solved games, 5 to 10, median 9, with 42 to 88 decisions, median 67), with the same likelihood,
limits and procedures. For the comparison with people, ΔAIC is divided by the number of decisions, and the model is
placed among the 39 per-person fits of the same procedure (percentile: share of people with a smaller value).
Intervals: 200 resamples of games. Blocks: the same fits on 10 blocks of 10 games (33 to 56 decisions), about one
person's amount of data.

Converged fits (primary):

| | Games / decisions | Best model | ΔAIC per decision, bounded vs. unbounded | bounded vs. leaky | Bounded best in resamples | Blocks of 10 games best fit by bounded | Percentile among people (vs. unbounded / vs. leaky) | α [95% CI] (percentile) |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 98 / 455 | bounded | 0.159 [−0.005, 0.542] | 0.041 [−0.005, 0.119] | 0.795 | 8 of 10 | 0.000 / 0.179 | 0.0283 [0.0001, 0.0430] (0.231) |
| Claude Haiku 5.5 | 100 / 448 | bounded | 0.364 [0.144, 0.616] | 0.029 [−0.005, 0.133] | 0.910 | 5 of 10 | 0.051 / 0.179 | 0.0226 [0.0001, 0.0562] (0.205) |
| Claude Opus 5.5 | 100 / 369 | bounded | 0.160 [0.059, 0.399] | 0.148 [0.047, 0.382] | 0.995 | 7 of 10 | 0.000 / 0.385 | 0.0226 [0.0223, 0.0229] (0.205) |
| People (median of 39) | 5–10 solved games each | bounded for 33 | 1.485 | 0.292 | | | | 0.1356 |

Paper's procedure:

| | Games / decisions | Best model | ΔAIC per decision, bounded vs. unbounded | bounded vs. leaky | Bounded best in resamples | Blocks of 10 games best fit by bounded | Percentile among people (vs. unbounded / vs. leaky) | α [95% CI] (percentile) |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 98 / 455 | unbounded | −0.032 [−0.099, 0.233] | 0.010 [−0.067, 0.277] | 0.410 | 4 of 10 | 0.000 / 0.128 | 0.0150 [0.0001, 0.0546] (0.051) |
| Claude Haiku 5.5 | 100 / 448 | unbounded | −0.052 [−0.074, 0.225] | −0.046 [−0.066, 0.233] | 0.125 | 4 of 10 | 0.000 / 0.077 | 0.0001 [0.0001, 0.2108] (0.000) |
| Claude Opus 5.5 | 100 / 369 | bounded | 0.059 [0.025, 0.377] | 0.051 [0.016, 0.337] | 0.980 | 5 of 10 | 0.051 / 0.154 | 0.0151 [0.0079, 0.0393] (0.051) |
| People (median of 39) | 5–10 solved games each | bounded for 34 | 1.220 | 0.826 | | | | 3.3414 |

- **With converged fits, the bounded model is best for all three models, as for most people, but by far less.** Its
  advantage over the unbounded (ideal-observer) model is 0.16–0.36 per decision against a human median of 1.49
  (intervals above 0 for Haiku and Opus, −0.005 to 0.542 for gpt-oss): gpt-oss and Opus are below every person,
  Haiku above 2 of 39. Its advantage over the leaky model (0.03–0.15) lies
  in the lower part of the human range (percentiles 0.18–0.39); the interval excludes 0 for Opus and just includes
  it for gpt-oss and Haiku. Opus's margins shrink to 0.03 and 0.02 with the replicate cache (Cache seed, below).
- **With the paper's procedure, the unbounded model is best for gpt-oss and Haiku.** The converged fits of the same
  data reverse this: as in Study A, the greedy search stops early on the leaky and bounded models. Its
  3,000-evaluation cap (Study A, Discrepancies, item 2) binds for the pooled bounded fits of gpt-oss and Haiku, and
  for 191 and 154 of their 211 bounded fits (pooled, blocks and resamples; Opus 32 of 211, not the pooled fit;
  `evaluation_cap_reached` in `b1b_<model>_T0.json`). Opus is fitted best by the bounded model under both procedures.
- **Cache seed** (the Study A check; `b1b_<model>_T0_cache_seed_converged.json`). With the bounded model refitted on
  the independently rebuilt cache, it stays best for all three models. For gpt-oss and Haiku the best model and the conclusions
  do not change (0.181 and 0.363 per decision against unbounded, 0.062 and 0.028 against leaky). For Opus its advantage shrinks to 0.032
  against unbounded and 0.020 against leaky (from 0.160 and 0.148; bounded log likelihood −1,313.7 instead of
  −1,290.0, α 0.0087 instead of 0.0226): Opus's cache-1 optimum is a narrow peak in α (resample interval 0.0223 to
  0.0229) that the replicate cache does not have, so its margin depends on the cache. People refitted with the same
  cache: medians 1.376 and 0.234, 29 of 39 best fit by the bounded model; against them Opus's advantage over leaky
  lies at percentile 0.256 (0.385 with cache 1).
- **Per-decision ΔAIC includes the AIC penalty**, which per decision is smaller for a model (369–455 decisions) than
  for a person (median 67). Without it (twice the log-likelihood difference per decision) the percentiles change
  little: Haiku 0.026 instead of 0.051 against unbounded and 0.154 instead of 0.179 against leaky, Opus 0.308 instead
  of 0.385 against leaky, the rest of the converged fits unchanged (`percentile_2dLL_per_decision`).
- **Parameters.** For gpt-oss and Haiku the converged leaky and bounded fits sit at the limits (ε 0.5, τ 0.001,
  λ 1000): the weak-evidence ridge found for many people in Study A. Opus's fit is interior (ε 0.021, α 0.023,
  λ 4.15, τ 0.13). α lies at percentile 0.205–0.231 of people's; its interval is very narrow for Opus (a narrow peak
  of the cache-1 likelihood; see Cache seed) and reaches the lower limit for the other two. As in Study A, people and models are therefore compared on ΔAIC per decision,
  not on raw parameters. The sampling temperature (0.7 for gpt-oss, the provider default for the Claude models; see
  Models) affects the fitted τ.

## B2 and B3: how the depth model uses evidence (Qwen3-32B, T0)

Probe battery on 24 development, 60 processing-calibration and 80 confirmation families, 4 samples per prompt. A
reply either judges a list of codes at once (up to 16: still possible, near misses, far misses; single-clue probes up
to 12) or names one code. 2,689 of 2,724 replies parsed (all 35 failures are identification replies), and 26,866 of
26,892 code judgments were answered (`replies_by_split` in the error-profile file). Models are fitted on the
processing-calibration families and compared on the held-out confirmation families. The battery ran partly on two
L40S GPUs and partly on one H200, with the same weights and precision.

**Error profile** (`process_fits/qwen3-32b_error_profile_T0.json`; calibration and confirmation families). Qwen3 rules
out still-possible codes far more often than it keeps ruled-out ones: false exclusion 0.569 on multi-clue and 0.323
on single-clue probes; false retention of near misses 0.245 and 0.125, of far misses 0.134 (balanced accuracy 0.616
and 0.776). Identification (naming a still-possible code): 0.24.
- By number of clues (1 / 2 / 3): false exclusion 0.323 / 0.576 / 0.510, near-miss retention 0.125 / 0.235 / 0.342.
- By complexity of the clue's query on single-clue probes (1 / 2 / 3 / 4 distinct digits): false exclusion 0.25 /
  0.30 / 0.34 / 0.37, near-miss retention 0.00 / 0.11 / 0.14 / 0.10 (only 96 judgments with one digit). False
  exclusions are higher after two or three clues than after one and grow with complexity; false retentions grow with count but hardly with complexity, the
  reverse of the conservative human account (M2c), whose errors are false retentions that grow with complexity.
- Clue age: near misses are ruled out only by the newest clue, far misses by an earlier one; far misses are kept less
  often (0.134 vs. 0.245), so the older clues are not the ones ignored (far misses can also break several clues).

**Product rule** (`process_fits/qwen3-32b_T0_product_rule.json`). The observation model combines
clues by multiplying per-clue checks, which predicts false exclusions rising with the number of clues. Against the
prespecified alternatives (one global check: the minimum or the k-th root of the per-clue checks), it fits far
better for M1 and M2s on the calibration families (AIC lower by 460–521) and the held-out families (log likelihood
higher by 352–389). It predicts false exclusions of 0.24 / 0.39 / 0.51 for 1 / 2 / 3 clues (observed 0.32 / 0.58 /
0.51), where the alternatives predict 0.25–0.29. The planned observation model is kept.

**Planned comparison** (`process_fits/qwen3-32b_T0.json`, `qwen3-32b_T0_intervals.json`):

| Model | Account | AIC (calibration) | Held-out log likelihood | Parameters |
|---|---|---|---|---|
| M0 | exact inference, reporting noise only | 13,387 | −9,352 | ζ 0.51 |
| M1 | noisy evidence use | 12,831 | −8,928 | ν 0.42, ζ 0.11 |
| M2c | representational approximation, conservative cache | 12,833 | −8,928 | α 0.0002 (near the lower limit 0.0001), ν 0.42, ζ 0.11 |
| M2s | representational approximation, symmetric cache | **12,814** | **−8,923** | α 0.0001 (lower limit), ν 0.42, ζ 0.11 |
| M3 | clue retention (forgetting) | 13,236 | −9,260 | k0 2.80, k1 −1.72 (older clues kept less often), ζ 0.47 |

M3 uses the corrected likelihood (`M3_corrected`; Departures, item 9).

- **Noisy evidence use beats reporting noise alone:** M2s minus M0 on held-out families 429 [333, 515]
  (family bootstrap, 80 families, 69 favor M2s); ΔAIC of M0 over M2s 573 on the calibration families, 95% interval
  [403, 763] across 1,000 resamples of them. Forgetting (M3) improves on M0 but stays far behind: M2s minus M3 on
  held-out families 337 [219, 443] (62 of 80), ΔAIC 421 [211, 652]; M3 is never best in a resample.
- **The noisy-evidence models are close.** M2s minus M1 on held-out families 4.5 [−1.2, 10.9] (40 of 80 families
  favor M2s); ΔAIC M1 − M2s 16.1 [−4.8, 41.3]; M2s has the lowest AIC in 92% of resamples, M1 in 6%, M2c in 2%. At
  Qwen3's fitted values, none of 50 data sets generated by M1 (or M0, M2c, M3) gives M2s an AIC edge as large as the
  observed 16.1, while 42% of those generated by M2s do, and 8% of those generated by the exploratory M1a
  (`dAIC_M1_minus_M2s` in `process_recovery_fitted_qwen3-32b_T0_summary.json`). So on the calibration families the
  symmetric approximation fits better than plain noise could produce; on the held-out families the difference is
  not reliable. Its α is at the lower limit, where the grammar's simplicity carries no weight and only approximations
  close to the exact clue keep weight (errors on codes that nearly fit or nearly break a clue), not simple rules.
- **Related families.** Families are symmetry-distinct, but the same single clues recur across splits (26 of the 30
  single clues probed alone in the confirmation families were also probed alone in calibration), and 1 confirmation
  family shares a two-clue sub-history with a calibration family. On the multi-clue replies of the other 79
  confirmation families, M2s minus M0 is 414 [322, 504] and M2s minus M1 2.4 [−3.3, 8.9]
  (`robustness_multi_clue_unrelated` in the intervals file).
- **Recovery** on the planned design (20 replicates): M0 0.75, M1 0.95, M2c, M2s and M3 1.00; parameters recovered
  with correlations 0.89–1.00 (`process_recovery_summary.json`). **At Qwen3's fitted values** (50 simulations each on
  the same items; planned models only, `process_recovery_fitted_qwen3-32b_T0_summary.json`): M0 0.80, M1 0.94, M2s
  0.98 and M3 1.00 are recovered, with their parameters (ν and ζ within 0.01; M3 k1 −1.72 recovered as −1.69), but
  data generated by M2c come back as M1 (M2c 0.12): at these values M2c is M1.

Reading (as planned): the human account (M2c) does not transfer. Qwen3's evidence use is noisy, with errors in both
directions that are more frequent after several clues than after one; among the planned models the symmetric approximation (M2s) fits best,
a comparative boundary of the human account, but its edge over plain noise does not carry over reliably to held-out
families.

## Native question choice (Qwen3-32B, T0)

`native_choice/qwen3-32b_T0.json`. 24 choice-calibration families, 2 pairs each (a simpler query, fewer distinct
digits, against a more informative one: the contrast of the human Experiment 2), 8 choices per pair with the order
randomized (no scores shown); 370 choices parsed, 14 not.

- Qwen3 chooses the **simpler** query in **0.31** of choices (the first-listed query in 0.50). People in
  Experiment 2 chose the easier query in 0.39 of their choices in the first game, rising to 0.69 in the tenth
  (`study_a/e2_descriptive.json`).
- Readouts `P(choose A) = ε/2 + (1 − ε)·σ[b + β·(X(A) − X(B))]` (A listed first), fitted by maximum likelihood with
  |b| and |β| at most 50 per SD of the predictor difference, compared by held-out log likelihood (leave one family
  out) relative to side bias alone: query structure (choosing the more complex query) +27.3 [9.0, 45.4]; EIG +22.4
  [6.1, 38.6]; V* +17.2 [0.7, 33.4]; its own predicted usefulness under the B3 model (M2s), the plan's readout, +15.9
  [−1.5, 33.7]; its own measured usefulness of the query −3.8 [−8.3, 0.4].
- For EIG and for measured usefulness the likelihood keeps rising as β grows (β at the limit in 24 and 23 of the 24
  folds): the best readout is a step, always the query with the higher value apart from lapses.
- Its choices follow informativeness and structure, not how useful a query actually is to it: in 29% of pairs the
  simpler query was more useful to Qwen3 itself, and its own predicted usefulness is not reliably better than side
  bias. In this pair design the more complex query never has the lower EIG or V* (ties: EIG in 2 of 48 pairs, V* in
  8), so those predictors cannot be separated from each other.

## Exploratory, not in the proposal: two error rates per clue (M1a)

Added after the error profile was seen, so reported separately from the planned comparison. M1a gives each clue
two error rates: ν_x for ruling out a code the clue allows and ν_r for keeping a code it rules out.

- Calibration fit ν_x 0.28 [0.24, 0.32], ν_r 0.07 [0.05, 0.09] (1,000 family resamples); lowest AIC in every
  resample; held-out confirmation families: M1a minus M2s 236 [165, 306].
- **Pre-stated test on 40 fresh families** (`experiments/study_b/B3_CONFIRMATION.md`, committed before the data were
  collected; `process_fits/qwen3-32b_T0_reserve_confirmation.json`): **confirmed.** M1a has the highest log
  likelihood (−4,138 vs. −4,271 for M2s, the best planned model); M1a minus M2s 133 [80, 189], 32 of 40 families.
  Observed vs. predicted false exclusion: 0.53 vs. 0.50 (multi-clue), 0.34 vs. 0.33 (single clue); M1a
  under-predicts near-miss retention on multi-clue probes (0.15 vs. 0.24 observed) and over-predicts it on single
  clues (0.16 vs. 0.105).
- **What the confirmation rests on** (post hoc; `robustness_multi_clue_unrelated` in the same file). The reserve
  families had never been run, but 26 of their 27 single clues had been probed alone in other splits, and 5 of the
  40 families share a two-clue sub-history with a family of another split. On the multi-clue replies of the other
  35 families, M1a and M2s do not differ (2.0 [−19.4, 25.4]). The confirmation therefore rests on the single-clue
  probes: new answers to clue types already seen when M1a was formulated.
- The pre-stated test used frozen parameters, at which the old and corrected M3 likelihoods agree. M1a also beats
  the corrected M3 on the reserve families (305 [195, 412]; `M3_corrected_check`).
- Recovery at the fitted values: M1a 1.00; data generated by M1a look like an M2s (33/50) or M1 (17/50) win when
  only the planned models are fitted, which matches the planned result; M1a wins wrongly on 1 of 50 M0, 6 of 50 M1
  and 7 of 50 M2c simulations.

## Departures from the plan

1. **Fitting procedure (B1b).** The plan fits the human models to model free play with the paper's greedy search.
   Those fits are reported, but the converged fits are primary, for the reason found in Study A (the greedy search
   stops early). People and models are always compared within one procedure.
2. **Solve-rate rule.** Added after the solve rates were seen, before any contrast was run (Rules, item 1). It
   leaves gpt-oss-120b, Claude Haiku 5.5 and Claude Opus 5.5 for inference; the depth model Qwen3-32B is reported
   descriptively in B1, so its aid arm is descriptive as well. The two models below the rule that solve some games
   show the human direction descriptively (B1).
3. **Panel.** Two of the admitted models are Claude models (one per family was planned), chosen to cover the
   standard and reasoning modes the plan asks for. Mistral-Small-24B failed the floor rule.
4. **Decoding of the Claude models.** The API takes no sampling parameters, so their temperature is the provider
   default rather than the fixed 0.7 of the open-weight models; Opus 5.5 cannot switch thinking off and runs at low
   effort (`configs/model_registry.json`).
5. **Not done:** the small probe-battery subset for the panel models and hand-coding of probe errors (Rules,
   item 3).
6. **Added and labeled exploratory:** the two-rate model M1a, with its pre-stated confirmation on fresh families.
7. **Native choice.** In the Experiment 2 pair design the more complex query is never the less informative one, so
   EIG and V* cannot be separated as predictors of choice. The plan's "difficulty covariates" are not a separate
   readout: the state's difficulty (codes remaining, number of clues) is the same for both queries of a pair, so it
   cannot predict which one is chosen; a query's own difficulty enters through query structure, V* and the model's
   measured usefulness. The readout parameters are bounded (|b|, |β| ≤ 50 per SD) because for some predictors the
   likelihood keeps rising toward a step function.
8. **Several codes per reply (B2).** The plan asks about one code per call; the battery asks about up to 16 codes in
   one reply. The process models treat the judgments in a reply as independent given the clues, so AIC differences
   overstate the evidence; the family-bootstrap intervals allow for dependence within families.
9. **M3 likelihood corrected.** The first version drew the kept-clue subset separately for every code judged
   in a reply, while its simulator, like the plan ("exact over the 2^k kept subsets"), keeps one subset per reply.
   Every M3 result above uses the corrected likelihood (fit, 1,000 resamples, both recoveries). The pre-stated reserve
   test keeps its frozen parameters, at which the two versions agree (every clue kept).
10. **Related families.** Families are symmetry-distinct and split by family, but the same single clues recur across
    splits and a few families share a two-clue sub-history with a family of another split (the plan keeps
    descendants in one family). Each held-out result is also reported on the multi-clue replies of unrelated families.
11. **B1 test.** The plan's single censoring-aware mixed model is reported as two analyses: a linear mixed model with a
    random intercept per stimulus (unsolved = 16) and a discrete-time hazard model in which unsolved games are
    censored after their last guess (standard errors clustered by stimulus).
12. **Records.** The B1 records of gpt-oss-120b, OLMo-2-32B and Phi-4 (the first runs) lack the revision and decoding
    fields of later records; their settings are those in the registry, which their screening records also carry.
    Records keep the last 300 characters of each reply; the full replies, re-asks and timestamps are in the
    language-model caches, which are not in the repository. Serving details (vLLM 0.31.0, bf16, quantization,
    hardware) are under `serving` in `configs/model_registry.json`.
13. **Qwen3-32B's B1 games were rerun.** The language-model client first timed out at 600 s and retried
    silently. A timeout aborts the request on the server, so on a busy node long replies were redrawn until a
    short one finished. An audit of every cached reply found it only in Qwen3-32B's game play and in a first Study C cap pilot, which
    was discarded. Cached replies that took over 600 s in total, the mark of a redraw: 72 in Experiment 1 under T0,
    15 in free play and 1,417 in the aid arm (`b1/qwen3-32b_rerun_audit.json`). All three sets were set aside and replayed
    from scratch with the corrected client, which waits up to 4 hours and never retries. Same positions, secrets
    and settings; no reply was reused. Every Qwen3 B1 number above comes from the rerun.

    No other model's records changed. Regenerating `b1_summary.json` leaves every non-Qwen3 entry identical,
    including every interval and test. Qwen3's free-play solve rate (0.44, first run 0.41) stays below the
    solve-rate rule. The descriptive changes are small under T0. In the aid arm, the first run had understated how
    often aided replies run out of room: re-asks rose from 3.1–5.5% to 9.0–14.5% of turns, invalid turns from 6 to 54,
    and the aid arm's Max-EIG minus Simple fell from 2.15 to 1.54.
14. **Preference for interpretable queries.** The proposal asks whether models prefer interpretable queries, as
    people do. Every model's free play reports the structure of its queries (colors per guess, first guesses; the
    free-play table). The Experiment 2-style choice between a simpler and a more informative query was run only
    for the depth model, Qwen3-32B (Native question choice).

Bootstrap resamples that ran past their time limit on the cluster were rerun with the same seeds; every summary
uses all of its resamples (200 refits per B1b fit, 1,000 refits for the B3 calibration bootstrap).
