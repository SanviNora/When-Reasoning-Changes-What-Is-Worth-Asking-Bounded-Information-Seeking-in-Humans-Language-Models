"""B1 analysis, every model against people on the same 132 Experiment 1 positions.

Effect measure: guesses after the supplied history, unsolved games counted as 16
(one more than the 15-guess limit), so failures are not dropped.
  signature effect   Max-EIG minus Simple (positive = worse after informative but complex histories,
                     the human pattern); bootstrap 95% CI resampling stimuli within condition
  relative to ideal  the same effect minus the ideal player's effect (ideal is faster after Max-EIG)
  hazard check       discrete-time logistic model of solving at each turn, condition + turn,
                     cluster-robust SEs by stimulus (censoring-aware test of the contrast: an unsolved game
                     is censored after the last guess actually made; people sometimes stopped before the limit)
  one-code subset    Max-EIG histories that leave one code (ideal needs 1 guess)
  mixed model        game-level linear mixed model, guesses ~ Max-EIG, random intercept per stimulus
  free play          the Study A (E3) measures, computed by the same code for people and models: solve rate,
                     guesses to solve, bits per guess, entropy curve, consistent guesses, distinct colors per
                     guess (query simplicity); every guess scored against Max-EIG and V* (V* = m/K: the chance
                     of naming the secret right after the answer, with exact inference)
  aid arm            the depth model under T3 on the same stimuli, next to T0 (descriptive)
The human-model difference is reported as an estimate with a CI, not a test. Bootstrap intervals use 20,000
resamples, so their bounds do not move with the random seed at the reported precision.

Solve-rate rule (stated after the solve rates were seen but before any contrast was run):
the inferential results (effect CI, hazard check, difference from people) and B1b are reported only for
models that solve at least half of their free-play games. Models below it are reported descriptively
(solve rates, mean guesses by condition, free play, and point estimates of the contrast without intervals or tests).

  python experiments/study_b/analyze_b1.py --models qwen3-32b gpt-oss-120b phi-4 olmo2-32b
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind.engine import get_game  # noqa: E402
from mastermind.states import load_human_games, n_distinct  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments/study_a"))
from e1_reproduction import load_games  # noqa: E402
from e3_reproduction import entropy_curve  # noqa: E402

B1 = ROOT / "results/study_b/b1"
UNSOLVED = 16
MIN_FREE_SOLVE = 0.5  # solve-rate rule


def human_e1(mm, stimuli):
    key = {(s["condition"], tuple((g["guess"], g["black"], g["white"]) for g in s["supplied"]), s["secret"]): s for s in stimuli}
    rows = []
    for g in load_games(mm):
        k = (g["condition"], tuple((mm.to_str(q), b, w) for q, (b, w) in g["supplied"]), mm.to_str(g["secret"]))
        s = key[k]
        rows.append({"agent": "humans", "stimulus": s["stimulus_id"], "condition": g["condition"],
                     "guesses": len(g["human"]) if g["solved"] else UNSOLVED, "solved": bool(g["solved"]),
                     "turns": len(g["human"]),  # guesses actually made (people sometimes stopped before the limit)
                     "ideal": s["ideal_guesses_to_solve"], "codes_remaining": s["codes_remaining"]})
    return pd.DataFrame(rows)


def model_e1(model, stimuli, condition="T0"):
    by_id = {s["stimulus_id"]: s for s in stimuli}
    rows = []
    for line in open(B1 / f"{model}_e1_{condition}.jsonl"):
        g = json.loads(line)
        sid = g["game_id"].split(":")[0]
        s = by_id[sid]
        rows.append({"agent": model, "stimulus": sid, "condition": s["condition"],
                     "guesses": g["own_guesses"] if g["solved"] else UNSOLVED, "solved": g["solved"],
                     "turns": g["own_guesses"],  # turns played, invalid ones included (15 when unsolved)
                     "ideal": s["ideal_guesses_to_solve"], "codes_remaining": s["codes_remaining"],
                     "invalid_turns": g["invalid_turns"],
                     "reask_turns": sum(t["attempts"] > 1 for t in g["turns"])})
    return pd.DataFrame(rows)


def effect(df):
    m = df.groupby("condition")["guesses"].mean()
    return float(m.get("Max-EIG", np.nan) - m.get("Simple", np.nan))


def effect_solved(df):
    """The same contrast over solved games only (descriptive)."""
    return effect(df[df.solved])


def boot_ci(df, rng, n=20000):
    """Bootstrap over stimuli (within condition), keeping all games of a resampled stimulus; the effect of a
    resample is the game-weighted mean of Max-EIG minus that of Simple."""
    means = {}
    for c in ("Max-EIG", "Simple"):
        g = df[df.condition == c].groupby("stimulus").guesses.agg(["sum", "size"])
        s, k = g["sum"].to_numpy(float), g["size"].to_numpy(float)
        idx = rng.integers(len(s), size=(n, len(s)))
        means[c] = s[idx].sum(1) / k[idx].sum(1)
    out = means["Max-EIG"] - means["Simple"]
    return [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)], out


def hazard_test(df):
    """Person-period data: one row per turn at risk; solved on that turn or not. An unsolved game is censored
    after the last guess actually made (15 for the models; people sometimes stopped earlier)."""
    rows = []
    for gid, r in enumerate(df[df.condition.isin(["Simple", "Max-EIG"])].itertuples()):
        last = r.turns
        for t in range(1, last + 1):
            rows.append({"game": gid, "stimulus": r.stimulus, "turn": t, "maxeig": int(r.condition == "Max-EIG"),
                         "event": int(r.solved and t == last)})
    pp = pd.DataFrame(rows)
    X = sm.add_constant(pd.get_dummies(pp["turn"].clip(upper=10), prefix="t", drop_first=True, dtype=float).assign(maxeig=pp["maxeig"]))
    res = sm.GLM(pp["event"], X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pp["stimulus"])[0]})
    b, se = res.params["maxeig"], res.bse["maxeig"]
    return {"log_odds_maxeig": round(float(b), 3), "se": round(float(se), 3), "z": round(float(b / se), 2),
            "p": float(res.pvalues["maxeig"]), "note": "negative = slower to solve after Max-EIG histories"}


def mixed_test(df):
    """Game-level linear mixed model on Simple and Max-EIG games: guesses (unsolved = 16) ~ Max-EIG,
    random intercept for each stimulus."""
    import statsmodels.formula.api as smf

    d = df[df.condition.isin(["Simple", "Max-EIG"])].assign(maxeig=lambda x: (x.condition == "Max-EIG").astype(float))
    res = smf.mixedlm("guesses ~ maxeig", d, groups=d["stimulus"]).fit(reml=False)
    b, se = float(res.params["maxeig"]), float(res.bse["maxeig"])
    return {"maxeig_minus_simple": round(b, 3), "se": round(se, 3), "ci95": [round(b - 1.96 * se, 2), round(b + 1.96 * se, 2)],
            "p": float(res.pvalues["maxeig"]), "stimulus_variance": round(float(res.cov_re.iloc[0, 0]), 3),
            "games": int(len(d)), "stimuli": int(d.stimulus.nunique())}


def v_star(mm, mask):
    """V*(q) = m/K for every query: K codes still possible, m distinct answers q can get among them."""
    t = mm.table[:, mask]
    return np.array([np.unique(row).size for row in t]) / mask.sum()


def play_measures(mm, games, max_guesses=15):
    """The Study A free-play measures for a list of games [(query, (black, white)), ...] (people or models)."""
    solved = [g for g in games if g and g[-1][1][0] == mm.P]
    bits, consistent, distinct, first, eig_rel, eig_top, v_rel, v_top = [], [], [], [], [], [], [], []
    for g in games:
        if not g:
            continue
        first.append(n_distinct(mm, g[0][0]))
        mask = np.ones(mm.N, bool)
        for t, (q, (b, w)) in enumerate(g):
            distinct.append(n_distinct(mm, q))
            if mask.sum() > 1:
                bits.append(mm.info_gain(q, mm.encode(b, w), mask))
                if t > 0:
                    consistent.append(bool(mask[q]))
                e = mm.eig(mask)
                v = v_star(mm, mask)
                eig_rel.append(e[q] / e.max())
                eig_top.append(e[q] >= e.max() - 1e-9)
                v_rel.append(v[q] / v.max())
                v_top.append(v[q] >= v.max() - 1e-9)
            mask = mm.consistent([(q, (b, w))], mask)

    def share(xs):
        return {k: round(xs.count(k) / len(xs), 3) for k in (1, 2, 3, 4)}

    n = np.array([len(g) for g in solved])
    return {"games": len(games), "solve_rate": round(len(solved) / len(games), 3),
            "guesses_to_solve": {"mean": round(float(n.mean()), 2), "sd": round(float(n.std(ddof=1)), 2)} if len(n) > 1 else None,
            "bits_per_guess_mean": round(float(np.mean(bits)), 3),
            "entropy_curve_mean": [round(float(x), 3) for x in np.mean([entropy_curve(mm, g) for g in games if g], axis=0)],
            "consistent_guess_rate": round(float(np.mean(consistent)), 3),
            "distinct_colors_per_guess": share(distinct), "distinct_colors_first_guess": share(first),
            "eig_relative_to_max_mean": round(float(np.mean(eig_rel)), 3), "share_at_max_eig": round(float(np.mean(eig_top)), 3),
            "vstar_relative_to_max_mean": round(float(np.mean(v_rel)), 3), "share_at_max_vstar": round(float(np.mean(v_top)), 3)}


def free_play_games(mm, model, condition="T0"):
    """A model's free-play games as move lists; invalid turns (no guess, no feedback) are left out."""
    path = B1 / f"{model}_free_{condition}.jsonl"
    if not path.exists():
        return None, None
    recs = [json.loads(line) for line in open(path)]
    games = [[(mm.index(t["guess"]), (t["black"], t["white"])) for t in r["turns"] if t.get("guess") is not None] for r in recs]
    return recs, games


def free_play(mm, model, condition="T0"):
    """Solve rate and guesses from the game records (an invalid turn counts as a guess, as in the runner),
    plus the per-guess Study A measures from `play_measures`."""
    recs, games = free_play_games(mm, model, condition)
    if recs is None:
        return None
    out = play_measures(mm, games)
    solved = [r for r in recs if r["solved"]]
    n = np.array([r["own_guesses"] for r in solved])
    out.update({"games": len(recs), "solve_rate": round(len(solved) / len(recs), 3),
                "guesses_to_solve": {"mean": round(float(n.mean()), 2), "sd": round(float(n.std(ddof=1)), 2)} if len(n) > 1 else None,
                "invalid_turn_rate": round(sum(r["invalid_turns"] for r in recs) / max(1, sum(r["own_guesses"] for r in recs)), 3)})
    return out


def capability(out, models):
    """How the B1 and B1b patterns change with capability (secondary, descriptive): models ordered by
    free-play query quality (mean EIG of their guesses relative to the best available query; ideal = 1)."""
    rows = []
    for m in [m for m in models if m in out]:
        r, fp = out[m], out[m]["free_play"]
        row = {"model": m, "eig_relative_to_max": fp["eig_relative_to_max_mean"], "free_play_solve_rate": fp["solve_rate"],
               "guesses_to_solve": fp["guesses_to_solve"]["mean"] if fp["guesses_to_solve"] else None,
               "eligible_for_inference": r["eligible_for_inference"],
               # below the solve-rate rule: the descriptive point estimate, without an interval
               "e1_maxeig_minus_simple": r.get("signature_effect_maxeig_minus_simple", r.get("descriptive", {}).get("maxeig_minus_simple")),
               "e1_ci95": r.get("ci95"),
               "one_code_mean_guesses": r["one_code_subset"]["mean_guesses"],
               "first_guess_single_color": fp["distinct_colors_first_guess"][1]}
        for proc, suf in (("greedy", ""), ("converged", "_converged")):
            f = ROOT / f"results/study_b/b1b_{m}_T0{suf}.json"
            if f.exists():
                b = json.loads(f.read_text())
                row[f"b1b_{proc}"] = {"best_model": b["pooled"]["best_model"], "dAIC_per_decision": b["pooled"]["dAIC_per_decision"]}
        rows.append(row)
    h, hf = out["humans"], out["humans_free_play_E3"]
    people = {"eig_relative_to_max": hf["eig_relative_to_max_mean"], "free_play_solve_rate": hf["solve_rate"],
              "guesses_to_solve": hf["guesses_to_solve"]["mean"], "e1_maxeig_minus_simple": h["signature_effect_maxeig_minus_simple"],
              "e1_ci95": h["ci95"], "one_code_mean_guesses": h["one_code_subset"]["mean_guesses"],
              "first_guess_single_color": hf["distinct_colors_first_guess"][1]}
    return {"measure": "free-play mean EIG of a guess relative to the best available query (ideal player = 1)",
            "note": "descriptive; B1b per-decision dAIC from the B1b summaries (both fitting procedures)",
            "people": people, "ideal": {"eig_relative_to_max": out["ideal_free_play"]["eig_relative_to_max_mean"]},
            "models": sorted(rows, key=lambda x: x["eig_relative_to_max"])}


def turn_counts(df):
    """Invalid turns (no valid guess after 2 re-asks) and turns that needed a re-ask, per condition, out of the
    turns actually played."""
    by = df.groupby("condition")
    return {"invalid_turn_rate": round(float(df.invalid_turns.sum() / max(1, df.turns.sum())), 4),
            "invalid_turns_by_condition": {c: int(d.invalid_turns.sum()) for c, d in by},
            "invalid_turn_rate_by_condition": {c: round(float(d.invalid_turns.sum() / max(1, d.turns.sum())), 4) for c, d in by},
            "reask_share_by_condition": {c: round(float(d.reask_turns.sum() / max(1, d.turns.sum())), 3) for c, d in by},
            "turns_played": int(df.turns.sum())}


def summarize(df, rng, label):
    ci, draws = boot_ci(df, rng)
    one = df[(df.condition == "Max-EIG") & (df.codes_remaining == 1)]
    ideal_effect = float(df[df.condition == "Max-EIG"].ideal.mean() - df[df.condition == "Simple"].ideal.mean())
    return {
        "games": int(len(df)),
        "solve_rate_by_condition": df.groupby("condition").solved.mean().round(3).to_dict(),
        "mean_guesses_by_condition": df.groupby("condition").guesses.mean().round(2).to_dict(),
        "signature_effect_maxeig_minus_simple": round(effect(df), 2), "ci95": ci,
        "maxeig_minus_simple_solved_games_only": round(effect_solved(df), 2),
        "unsolved_games": int((~df.solved).sum()),
        "guesses_made_in_unsolved_games": ({"min": int(df[~df.solved].turns.min()), "max": int(df[~df.solved].turns.max())}
                                           if (~df.solved).any() else None),
        "ideal_effect": round(ideal_effect, 2),
        "effect_relative_to_ideal": round(effect(df) - ideal_effect, 2),
        "one_code_subset": {"games": int(len(one)), "mean_guesses": round(float(one.guesses.mean()), 2) if len(one) else None,
                            "solve_rate": round(float(one.solved.mean()), 3) if len(one) else None},
        "hazard_check": hazard_test(df),
        "mixed_model": mixed_test(df),
    }, draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    mm = get_game(4, 4)
    stimuli = json.loads((ROOT / "data/stimuli/e1_stimuli.json").read_text())
    rng = np.random.default_rng(0)  # people; every model gets its own stream, so results do not depend on the model order

    human = human_e1(mm, stimuli)
    out = {"measure": f"guesses after the supplied history; unsolved = {UNSOLVED}",
           "rule": f"inferential results only for models solving >= {MIN_FREE_SOLVE:.0%} of free-play games"}
    out["humans"], h_draws = summarize(human, rng, "humans")
    out["humans_free_play_E3"] = play_measures(mm, load_human_games(mm))
    out["ideal_free_play"] = play_measures(mm, [mm.play_greedy_eig(sec) for sec in range(mm.N)])
    for m in args.models:
        if not (B1 / f"{m}_e1_T0.jsonl").exists():
            continue
        df = model_e1(m, stimuli)
        fp = free_play(mm, m)
        m_rng = np.random.default_rng([0, zlib.crc32(m.encode())])
        res, draws = summarize(df, m_rng, m)
        res["completions_per_stimulus"] = int(df.groupby("stimulus").size().median())
        res.update(turn_counts(df))
        res["free_play"] = fp
        res["eligible_for_inference"] = bool(fp is not None and fp["solve_rate"] >= MIN_FREE_SOLVE)
        if res["eligible_for_inference"]:
            m_eff, h_eff = effect(df), effect(human)  # unrounded
            diff = draws - h_draws
            res["difference_from_humans"] = {"estimate": round(m_eff - h_eff, 2),
                                             "ci95": [round(float(np.percentile(diff, 2.5)), 2), round(float(np.percentile(diff, 97.5)), 2)]}
            ratio = draws / h_draws
            res["ratio_to_humans"] = {"estimate": round(m_eff / h_eff, 2),
                                      "ci95": [round(float(np.percentile(ratio, 2.5)), 2), round(float(np.percentile(ratio, 97.5)), 2)],
                                      "note": "LLM effect / human effect; unstable where the human draw is near 0"}
        else:  # descriptive only: point estimates, no intervals or tests
            res["descriptive"] = {"maxeig_minus_simple": res["signature_effect_maxeig_minus_simple"],
                                  "maxeig_minus_simple_solved_games_only": res["maxeig_minus_simple_solved_games_only"],
                                  "hazard_log_odds_maxeig": res["hazard_check"]["log_odds_maxeig"],
                                  "note": "below the solve-rate rule: point estimates only, no intervals or tests"}
            for k in ("signature_effect_maxeig_minus_simple", "maxeig_minus_simple_solved_games_only", "ci95",
                      "effect_relative_to_ideal", "hazard_check", "mixed_model"):
                res.pop(k, None)
        if (B1 / f"{m}_e1_T3.jsonl").exists():  # aid arm (secondary), same stimuli
            d3 = model_e1(m, stimuli, "T3")
            res["aid_arm_T3"] = {"games": int(len(d3)), "eligible_for_inference": res["eligible_for_inference"],
                                 "solve_rate_by_condition": d3.groupby("condition").solved.mean().round(3).to_dict(),
                                 "mean_guesses_by_condition": d3.groupby("condition").guesses.mean().round(2).to_dict(),
                                 "maxeig_minus_simple_T3": round(effect(d3), 2), "maxeig_minus_simple_T0": round(effect(df), 2),
                                 **turn_counts(d3)}
            if res["eligible_for_inference"]:
                res["aid_arm_T3"]["ci95_T3"], _ = boot_ci(d3, m_rng)
        out[m] = res
    out["capability"] = capability(out, args.models)
    (ROOT / "results/study_b/b1_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
