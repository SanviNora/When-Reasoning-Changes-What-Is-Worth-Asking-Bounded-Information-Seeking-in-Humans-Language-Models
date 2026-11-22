"""Experiment 1: extract the starting positions people saw and
reproduce the behavioral result (people continue faster after Simple histories than after
Max-EIG histories, although an ideal player is faster after Max-EIG histories).

Writes:
  data/stimuli/e1_stimuli.json          one entry per distinct (condition, supplied guesses, secret)
  results/study_a/e1_reproduction.json  summary numbers
Run: .venv/bin/python experiments/study_a/e1_reproduction.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind.engine import get_game  # noqa: E402

E1 = ROOT / "data/human/cheyette2025_osf_vjrqu/Experiments/E1/mastermind_E1.csv"
STIMULI = ROOT / "data/stimuli/e1_stimuli.json"
SUMMARY = ROOT / "results/study_a/e1_reproduction.json"
CONDITIONS = ["Simple", "Max-EIG", "None"]


def load_games(mm):
    """One record per game: condition, supplied history, secret and the human continuation."""
    df = pd.read_csv(E1, dtype={"guess": str, "true_code": str})
    df["condition"] = df["filter_cost"].fillna("None")
    games = []
    for (uid, trial), g in df.groupby(["uniqueid", "trial_id"], sort=True):
        g = g.sort_values("guess_number")
        moves = [(mm.index(r.guess), (int(r.n_green), int(r.n_yellow)), r.whose_guess) for r in g.itertuples()]
        games.append({
            "uniqueid": uid,
            "trial_id": int(trial),
            "condition": g["condition"].iloc[0],
            "secret": mm.index(g["true_code"].iloc[0]),
            "supplied": [(q, fb) for q, fb, who in moves if who == "Model"],
            "human": [(q, fb) for q, fb, who in moves if who == "Human"],
            "solved": int(g["solved"].max()),
        })
    return games


def controlled_effect(df):
    """The Max-EIG minus Simple difference in guesses, controlling for the number of supplied
    guesses and the uncertainty they leave (log2 codes remaining). Unsolved games count as 16 (the
    convention of analyze_b1). OLS with standard errors clustered by participant."""
    import statsmodels.formula.api as smf

    d = df[df.condition.isin(["Simple", "Max-EIG"])].copy()
    d["guesses"] = np.where(d.solved == 1, d.n_human, 16)
    out = {"n_supplied_by_condition": {c: sorted(map(int, d[d.condition == c].n_supplied.unique())) for c in ["Simple", "Max-EIG"]}}
    for name, f in [("raw", "guesses ~ C(condition, Treatment('Simple'))"),
                    ("controlled", "guesses ~ C(condition, Treatment('Simple')) + n_supplied + log2_remaining"),
                    ("controlled_plus_ideal", "guesses ~ C(condition, Treatment('Simple')) + n_supplied + log2_remaining + ideal")]:
        fit = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.uniqueid.astype("category").cat.codes})
        k = "C(condition, Treatment('Simple'))[T.Max-EIG]"
        lo, hi = fit.conf_int().loc[k]
        out[name] = {"max_eig_minus_simple": round(float(fit.params[k]), 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
                     "p": round(float(fit.pvalues[k]), 4)}
    # The paper's own regression (Results, Experiment 1): guesses after the supplied history, as made (unsolved games
    # not imputed), on condition and initial entropy (natural log of the codes remaining), ordinary least squares,
    # all Simple and Max-EIG games.
    d["entropy_nats"] = d.log2_remaining * np.log(2)
    fit = smf.ols("n_human ~ C(condition, Treatment('Max-EIG')) + entropy_nats", data=d).fit()
    k = "C(condition, Treatment('Max-EIG'))[T.Simple]"
    out["paper_specification"] = {
        "games": int(len(d)), "df_resid": int(fit.df_resid),
        "simple_minus_max_eig": {"B": round(float(fit.params[k]), 2), "t": round(float(fit.tvalues[k]), 2)},
        "initial_entropy_nats": {"B": round(float(fit.params["entropy_nats"]), 2), "t": round(float(fit.tvalues["entropy_nats"]), 2)},
        "paper": {"simple_minus_max_eig": {"B": -2.29, "t": -5.89, "df": 283}, "initial_entropy": {"B": 0.62, "t": 4.94, "df": 283}}}
    return out


def main():
    mm = get_game(4, 4)
    games = load_games(mm)

    stimuli = {}
    for g in games:
        key = (g["condition"], tuple(g["supplied"]), g["secret"])
        if key not in stimuli:
            remaining = int(mm.consistent(g["supplied"]).sum()) if g["supplied"] else mm.N
            ideal = mm.play_greedy_eig(g["secret"], history=g["supplied"])
            stimuli[key] = {
                "stimulus_id": f"E1-{len(stimuli):03d}",
                "condition": g["condition"],
                "secret": mm.to_str(g["secret"]),
                "supplied": [{"guess": mm.to_str(q), "black": b, "white": w} for q, (b, w) in g["supplied"]],
                "codes_remaining": remaining,
                "ideal_guesses_to_solve": len(ideal),
                "n_human_games": 0,
            }
        stimuli[key]["n_human_games"] += 1
        g["stimulus"] = stimuli[key]

    games_df = pd.DataFrame([{
        "uniqueid": g["uniqueid"],
        "condition": g["condition"],
        "n_supplied": len(g["supplied"]),
        "log2_remaining": float(np.log2(g["stimulus"]["codes_remaining"])),
        "n_human": len(g["human"]),
        "solved": g["solved"],
        "ideal": g["stimulus"]["ideal_guesses_to_solve"],
        "one_code": g["stimulus"]["codes_remaining"] == 1,
    } for g in games])

    by_cond = {}
    for c in CONDITIONS:
        d = games_df[games_df.condition == c]
        by_cond[c] = {
            "games": int(len(d)),
            "stimuli": sum(s["condition"] == c for s in stimuli.values()),
            "solve_rate": round(float(d.solved.mean()), 3),
            "human_guesses_solved_games": round(float(d[d.solved == 1].n_human.mean()), 2),
            "human_guesses_all_games": round(float(d.n_human.mean()), 2),
            "ideal_guesses": round(float(d.ideal.mean()), 2),
        }
    # The key subset: Max-EIG histories that leave exactly one code, so an ideal player needs one guess.
    one_all = games_df[games_df.one_code & (games_df.condition == "Max-EIG")]
    one = one_all[one_all.solved == 1]
    summary = {
        "participants": int(pd.read_csv(E1, usecols=["uniqueid"]).uniqueid.nunique()),
        "controlled_effect": controlled_effect(games_df),
        "games": len(games),
        "stimuli": len(stimuli),
        "by_condition": by_cond,
        "one_code_subset_max_eig": {
            "games": int(len(one_all)),
            "solve_rate": round(float(one_all.solved.mean()), 3),
            "human_guesses_solved_games": round(float(one.n_human.mean()), 2),
            "ideal_guesses": 1,
            "simple_games_also_one_code": int((games_df.one_code & (games_df.condition == "Simple")).sum()),
        },
    }

    STIMULI.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    STIMULI.write_text(json.dumps(list(stimuli.values()), indent=1))
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
