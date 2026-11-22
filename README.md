# When Reasoning Changes What Is Worth Asking

**Bounded Information Seeking in Humans and Language Models**

People playing Mastermind trade informativity for interpretability: they favor queries whose feedback they can
interpret, even when other queries are more informative (Cheyette et al., 2025). This project asks whether
language models show the same trade-off, and whether independent measures of how they process evidence can
predict how a reasoning aid changes which questions are worth asking. All studies use 4-position, 4-color
Mastermind (256 codes), where ideal Bayesian inference is exact.

1. **Study A, human modeling:** reproduce the published behavioral findings and the comparison of the
   bounded-representation model with noisy-inference and ideal-observer accounts.
2. **Study B, language models:** run language models on the positions people saw and on full games, fit the same
   human models to their play, and compare process models of how one model uses feedback.
3. **Study C, capacity manipulation:** give that model a structured reasoning aid and test whether the fitted models
   predict how the aid changes the usefulness of each question.

This repository contains all three studies. Study C's confirmation test was preregistered: the analysis plan and
prediction lock in `prereg/` were committed and registered on OSF before any confirmation data were collected.

## Main results

**Study A** (`results/study_a/README.md`)
- The behavioral findings reproduce. Experiment 1: after Simple histories people need fewer guesses than after
  Max-EIG histories (B = −2.31, t(283) = −5.92; paper −2.29, −5.89). Experiment 3: solve rate 0.864, 7.72 guesses
  (paper 0.86, 7.5).
- The bounded model fits best for 33 of 39 people (converged fits; 34 with the paper's search, paper 37) and, with
  converged fits, predicts held-out people better than the leaky and unbounded models. Its margin over the leaky
  model is smaller than reported (mean ΔAIC 27.8 vs. 72.3).

**Study B** (`results/study_b/README.md`)
- **B1:** people need 1.29 [0.29, 2.28] (resampling starting positions) more guesses after informative but complex histories than after simple ones.
  None of the models that solve at least half their games (gpt-oss-120b, Claude Haiku 5.5, Claude Opus 5.5) shows
  this (−0.79 to 0.23); Haiku and Opus are faster after them, like the ideal player. Two weaker models show the
  human direction descriptively.
- **B1b:** the bounded model fits these models' free play best, but its advantage over the ideal observer is far
  smaller than in people (0.16–0.36 per decision against a human median of 1.49; for Opus the margin depends on
  the representation cache).
- **B2/B3** (Qwen3-32B): its evidence use is noisy, mostly ruling out codes that are still possible, with more errors
  after several clues than after one. The human account (a conservative approximation) does not transfer; a
  symmetric approximation fits best among the planned models, but its edge over plain noise is not reliable on
  held-out families.
- **Native question choice:** Qwen3's choices follow the informativeness and structure of queries, not how useful
  a query is to it.

**Study C** (`results/study_c/README.md`; Qwen3-32B with a structured reasoning aid)
- **C-a:** the aid does not make evidence use more exact. Clue-checking errors do not fall, and the fitted noise
  of evidence use rises (ν 0.42 → 0.52). Finished replies identify a still-possible code more often, but far more
  replies run out of room. A generic "check your work" prompt does at least as well.
- **C-c (H3, preregistered): not supported.**
  - The fitted model predicts how the aid changes each question's usefulness no better than any of four simpler
    rivals; every point estimate favors the rival.
  - Across 360 pairs in 120 fresh families, the aid's effect varies beyond sampling noise, but no predictor
    anticipates it.
- **C-b (H4, preregistered): not supported.** The model's own question choices do not follow the aid's change in
  usefulness (slope −0.63 [−2.50, 1.32]).
- **Design check (post hoc).** The confirmation data were noisier than the power analysis assumed (realized power
  about 0.50 at the planned smallest effect, not 0.86). Even so, the point estimates point away from the
  hypotheses.

**Primary tests across studies** (`results/README.md`): Holm across B1, B3 and C-b. Only Haiku's and Opus's B1
effects are significant, and they go in the ideal player's direction, not the human one.

## What the proposal set out to do, and where it is

Every commitment of the approved proposal, in its order, with the section that reports it. Status: ✓ done as
proposed; ◐ done with a stated limit.

| | Proposal | Status | Where |
|---|---|---|---|
| A1 | Use the released Mastermind data from three experiments | ✓ | `results/study_a/README.md`, Behavior (E1, E2, E3) |
| A2 | Reproduce: people solve games faster after simpler guesses | ✓ | Study A, Behavior (E1 regression and signature effect) |
| A3 | Reproduce: people choose simpler, less informative queries | ✓ | Study A, Behavior (E2) |
| A4 | Reproduce the free-play comparison of the bounded model with noisy-inference and ideal-observer accounts | ✓ | Study A, Model comparison (both fitting procedures, held-out people, recovery) |
| B1 | Open-weight models across capabilities, possibly proprietary ones | ✓ | `results/study_b/README.md`, Models (four open-weight, two proprietary) |
| B2 | Continue games from the participants' starting positions, after informative but complex and after simpler guesses | ✓ | Study B, B1 |
| B3 | Play full games, each guess scored against the ideal Bayesian player | ✓ | Study B, B1 free play (EIG and V* of every guess relative to the best) |
| B4 | Fit the same human model to their play and compare it with the rival accounts by AIC | ◐ | Study B, B1b. Fitted for the three models that solve at least half their games; the rule was set after the solve rates were seen (Departures, item 2) |
| B5 | Compare the fitted parameters with people's | ✓ | Study B, B1b (α and ΔAIC per decision placed among the 39 people) |
| B6 | Do the models play better after simpler guesses, as people do? | ✓ | Study B, B1 |
| B7 | Do they prefer interpretable queries, as people do? | ◐ | Query structure in every model's free play; the Experiment 2-style choice only for Qwen3-32B (Departures, item 14) |
| B8 | How these patterns change with capability | ✓ | Study B, Capability (descriptive) |
| B9 | One model in depth: probes of which codes are still possible | ✓ | Study B, B2 (Qwen3-32B) |
| B10 | Compare exact-inference, noisy-evidence and representational-approximation accounts | ✓ | Study B, B3 |
| B11 | Parameter and model recovery | ✓ | Study B, B3 recovery (planned design and fitted values) |
| C1 | A structured aid that checks candidate codes against each clue step by step | ✓ | `results/study_c/README.md` (T3) |
| C2 | Compare it with standard and length-matched prompts; measure how it changes evidence use | ✓ | Study C, C-a |
| C3 | Usefulness: the probability that the model names the secret after a question's answer | ✓ | Study C, C-c Measure |
| C4 | Do the model's own choices shift toward the questions the aid makes more useful? | ✓ | Study C, C-b (H4) |
| C5 | Can the fitted models predict how the aid changes each question's usefulness on held-out states, preregistered, against difficulty baselines? | ✓ | Study C, C-c (H3); `prereg/` (OSF) |

## Repository layout

```
mastermind/          core package: game engine and ideal observer, human models, process models,
                     usefulness, prompts, language-model client, symmetry utilities
experiments/
  study_a/           human reproduction, representation cache, model fits, recovery
  study_b/           model screening, game play (B1), human models fitted to model play (B1b),
                     probe battery (B2), process-model comparison (B3), native question choice
  study_c/           aid effects (C-a), reply-cap pilot, fresh families, prediction test (C-c), choice shift (C-b)
  primary_tests.py   Holm correction across the primary tests of the three studies
  make_figures.py    the paper's figures, from the saved results (results/figures/)
  hpc/               SLURM job scripts and the list of cluster runs
  reproduce_summaries.sh   regenerates every reported number from the saved records and fits
prereg/              Study C analysis plan and prediction lock
configs/             model registry, model selection and frozen prompt templates
data/
  stimuli/           the Experiment 1 starting positions, derived from the human data
  splits/            state families and their fixed splits
  model_cache/       rebuilt representation caches for the bounded model
  human/             where to put the human data (not redistributed; see its README)
results/             outputs of every script and the write-up of each study
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # API key, only needed for the Claude models; .env is git-ignored
```

Download the human data as described in `data/human/README.md`. Open-weight models are served with vLLM on a GPU
cluster (`experiments/hpc/`); everything else runs on a laptop.

## Reproducing the results

```
bash experiments/reproduce_summaries.sh     # every summary and figure, from the saved records and fits; no model calls or refits
```

The model fits run on a cluster; `experiments/hpc/README.md` lists every job. The human models are fitted in two
ways, both reported: the paper's greedy search (the reproduction) and converged fits (the primary results; rules
in `mastermind/human_models.py`). For one participant:

```
python experiments/study_a/fit_human_models.py fit --participant 0                          # paper's procedure
python experiments/study_a/fit_human_models.py fit --participant 0 --procedure converged    # converged fits
```

## References

- Cheyette, S. J., Callaway, F., Bramley, N. R., Nelson, J. D., & Tenenbaum, J. B. (2025). Representational constraints shape human information-seeking. *PsyArXiv*. https://doi.org/10.31234/osf.io/w4b3y
- Rothe, A., Lake, B. M., & Gureckis, T. M. (2018). Do people ask good questions? *Computational Brain & Behavior, 1*, 69–89.
