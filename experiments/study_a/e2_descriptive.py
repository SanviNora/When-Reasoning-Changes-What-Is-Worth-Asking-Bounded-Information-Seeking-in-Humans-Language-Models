"""Experiment 2, descriptive only (the released data lack the offered pair for some choices).

Mirrors the authors' mainAnalyses.R for the Mastermind forced-choice phase, in Python:
  choice trend   P(choose the easy guess) ~ trial number              (logistic)
  solving        P(solved the game) ~ share of easy choices in it      (logistic, one row per game)
  attempts       guesses used ~ share of easy choices                  (linear)
Standard errors are clustered by participant (the R script uses plain GLM errors; both are reported).

Writes results/study_a/e2_descriptive.json.
Run: .venv/bin/python experiments/study_a/e2_descriptive.py
"""
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
E2 = ROOT / "data/human/cheyette2025_osf_vjrqu/Experiments/E2/mastermind_E2.csv"
OUT = ROOT / "results/study_a/e2_descriptive.json"


def coef(fit_plain, fit_cluster, name):
    lo, hi = fit_cluster.conf_int().loc[name]
    return {"estimate": round(float(fit_plain.params[name]), 4), "p_glm": round(float(fit_plain.pvalues[name]), 4),
            "p_clustered": round(float(fit_cluster.pvalues[name]), 4), "ci95_clustered": [round(float(lo), 4), round(float(hi), 4)]}


def main():
    mm = pd.read_csv(E2)
    forced = mm[mm.guess_phase == "forced"].copy()
    per_game = forced.sort_values("id").groupby(["uniqueid", "trial_id"], as_index=False).last()
    out = {"participants": int(mm.uniqueid.nunique()), "forced_choices": len(forced), "games": len(per_game),
           "p_easy_overall": round(float(forced.easy_guess.mean()), 3),
           "p_easy_by_trial": {int(t): round(float(v), 3) for t, v in forced.groupby("trial_id").easy_guess.mean().items()}}
    for name, formula, data, family, term in [
        ("choice_trend", "easy_guess ~ trial_id", forced, "logit", "trial_id"),
        ("solving_vs_p_easy", "ever_corr ~ p_easy", per_game, "logit", "p_easy"),
        ("attempts_vs_p_easy", "n_attempts ~ p_easy", per_game, "ols", "p_easy"),
    ]:
        model = smf.logit if family == "logit" else smf.ols
        groups = data.uniqueid.astype("category").cat.codes
        plain = model(formula, data=data).fit(disp=0) if family == "logit" else model(formula, data=data).fit()
        clus = (model(formula, data=data).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups}) if family == "logit"
                else model(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": groups}))
        out[name] = coef(plain, clus, term)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
