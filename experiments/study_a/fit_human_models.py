"""Fit the unbounded, leaky and bounded models to each Experiment 3 participant
(solved games, as in the paper) and compare them by AIC.

  python experiments/study_a/fit_human_models.py fit --participant 0     # one participant, all 3 models
  python experiments/study_a/fit_human_models.py summarize               # all participants -> table
  add --include-unsolved to both for the sensitivity fit on all games
  python experiments/study_a/fit_human_models.py heldout                 # held-out participants + game-index trend
  python experiments/study_a/fit_human_models.py fit --participant 0 --models bounded --half early   (then late; then `halves`)
  python experiments/study_a/fit_human_models.py fit --participant 0 --models bounded --cache-rep 2  (then `cache_seed`):
      refit the bounded model with an independently rebuilt cache (build_cache.py --rep 2)
Targets (paper, Mastermind): bounded beats unbounded by mean dAIC 82.5 and leaky by 72.3; 37/39 favor bounded.

--procedure (every command): greedy = the paper's greedy search (the reproduction, default); converged = the
converged fits (primary result; rules in mastermind/human_models.py); wide = converged with wider parameter
limits (sensitivity check). Converged fits start from the greedy fits of the same data, so run greedy first.
Outputs of converged and wide runs go to folders and files ending in _converged and _converged_wide.
"""
import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import human_models as hm  # noqa: E402

E3 = ROOT / "data/human/cheyette2025_osf_vjrqu/Experiments/E3"
CACHE = ROOT / "data/model_cache/bounded_cache_v1.npz"
OUT = ROOT / "results/study_a/e3_fits"


def participant_games(mm, solved_only=True):
    keep = {r["uniqueid"] for r in csv.DictReader(open(E3 / "analysis/e3_human_preprocessed.csv")) if r["game"] == "Mastermind"}
    games = defaultdict(list)
    for r in csv.DictReader(open(E3 / "mastermind_E3.csv")):
        if r["uniqueid"] in keep:
            games[(r["uniqueid"], r["trial_id"])].append(r)
    by = defaultdict(list)
    for (uid, _), rows in sorted(games.items()):
        rows.sort(key=lambda r: int(r["guess_number"]))
        g = [(mm.index(ast.literal_eval(r["guess"])), (int(r["n_green"]), int(r["n_yellow"]))) for r in rows]
        if not solved_only or g[-1][1][0] == mm.P:
            by[uid].append(g)
    return dict(sorted(by.items()))


SUFFIX = {"greedy": "", "converged": "_converged", "wide": "_converged_wide"}


def out_dir(include_unsolved=False, half=None, cache_rep=1, procedure="greedy"):
    return OUT.with_name(OUT.name + ("_with_unsolved" if include_unsolved else "") + (f"_{half}" if half else "")
                         + (f"_cacherep{cache_rep}" if cache_rep != 1 else "") + SUFFIX[procedure])


def cache_path(rep=1):
    return CACHE if rep == 1 else CACHE.with_name(f"{CACHE.stem}_rep{rep}.npz")


def fit_one(idx, models, seed, include_unsolved=False, half=None, cache_rep=1, procedure="greedy"):
    task = hm.Task()
    cache = hm.BoundedCache(cache_path(cache_rep)) if "bounded" in models else None
    uid, games = list(participant_games(task.mm, solved_only=not include_unsolved).items())[idx]
    if half:  # game-index sensitivity: session games 1-5 vs 6-10
        order = trial_order()
        games = [g for g in games if (order[(uid, tuple(q for q, _ in g))] <= 5) == (half == "early")]
    out = {"participant": idx, "uniqueid": uid, "games": len(games), "decisions": sum(len(g) for g in games), "fits": {}}
    if procedure == "greedy":
        for m in models:
            out["fits"][m] = hm.fit(task, m, games, cache=cache, seed=seed + idx)
    else:
        greedy = out_dir(include_unsolved, half, cache_rep) / f"participant_{idx:02d}.json"
        paper = {}
        if greedy.exists():
            g = json.loads(greedy.read_text())
            assert g["uniqueid"] == uid and g["games"] == len(games), "greedy fit is for different data"
            paper = g["fits"]
        extra = {}  # variants and the wide check also start from the person's primary converged fit
        primary = out_dir(procedure="converged") / f"participant_{idx:02d}.json"
        if (procedure == "wide" or include_unsolved or half or cache_rep != 1) and primary.exists():
            extra = {m: [f["params"]] for m, f in json.loads(primary.read_text())["fits"].items()}
        bounds = hm.WIDE_BOUNDS if procedure == "wide" else hm.BOUNDS
        out["fits"] = hm.fit_converged_models(task, models, games, cache, paper, bounds, extra, seed + idx)
    out["procedure"] = procedure
    for m in out["fits"]:
        print(idx, m, round(out["fits"][m]["loglik"], 2), out["fits"][m]["params"], flush=True)
    d = out_dir(include_unsolved, half, cache_rep, procedure)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"participant_{idx:02d}.json").write_text(json.dumps(out, indent=1))


def summarize(include_unsolved=False, procedure="greedy"):
    fits = [json.loads(p.read_text()) for p in sorted(out_dir(include_unsolved, procedure=procedure).glob("participant_*.json"))]
    aic = {m: np.array([f["fits"][m]["aic"] for f in fits]) for m in hm.MODELS}
    d_unb = aic["unbounded"] - aic["bounded"]
    d_leaky = aic["leaky"] - aic["bounded"]
    summary = {
        "procedure": procedure,
        "participants": len(fits),
        "mean_dAIC_bounded_vs_unbounded": round(float(d_unb.mean()), 1),
        "mean_dAIC_bounded_vs_leaky": round(float(d_leaky.mean()), 1),
        "participants_favoring_bounded_over_both": int(((d_unb > 0) & (d_leaky > 0)).sum()),
        "best_model_counts": {m: int(c) for m, c in zip(hm.MODELS, np.bincount(np.argmin(np.stack([aic[m] for m in hm.MODELS]), 0), minlength=3))},
        "paper": {"dAIC_vs_unbounded": 82.5, "dAIC_vs_leaky": 72.3, "favor_bounded": "37/39"},
        "median_params": {m: {k: round(float(np.median([f["fits"][m]["params"][k] for f in fits])), 4)
                              for k in hm.MODELS[m]} for m in hm.MODELS},
    }
    if procedure == "greedy":  # the search stops after hm.fit's max_evals likelihood evaluations, shared by all starts
        summary["evaluation_cap_reached"] = {m: sum(f["fits"][m]["evals"] >= 3000 for f in fits) for m in hm.MODELS}
    if procedure != "greedy":
        summary["parameters_at_a_limit"] = {m: {k: sum(k in f["fits"][m]["at_limit"] for f in fits) for k in hm.MODELS[m]}
                                            for m in hm.MODELS}
        summary["gain_over_greedy_search_loglik"] = {
            m: {"mean": round(float(np.mean([f["fits"][m]["loglik"] - f["fits"][m]["greedy_search_loglik"] for f in fits])), 2),
                "max": round(float(np.max([f["fits"][m]["loglik"] - f["fits"][m]["greedy_search_loglik"] for f in fits])), 2)}
            for m in hm.MODELS}
    name = f"e3_model_comparison{'_with_unsolved' if include_unsolved else ''}{SUFFIX[procedure]}.json"
    (ROOT / "results/study_a" / name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def heldout(seed=0, procedure="greedy"):
    """Held-out participants: predict each person's games from the median fitted parameters
    of the other 38 (per-parameter median; nothing from the held-out person is used). Also the
    game-index check: does a static model's per-decision fit drift across a person's 10 games?"""
    import statsmodels.formula.api as smf

    task = hm.Task()
    cache = hm.BoundedCache(CACHE)
    fits = [json.loads(p.read_text()) for p in sorted(out_dir(procedure=procedure).glob("participant_*.json"))]
    people = list(participant_games(task.mm).items())
    order = trial_order()
    ll = {m: np.zeros(len(fits)) for m in hm.MODELS}
    rows = []
    for i, ((uid, games), f) in enumerate(zip(people, fits)):
        assert f["uniqueid"] == uid
        for m in hm.MODELS:
            others = [g["fits"][m]["params"] for j, g in enumerate(fits) if j != i]
            p = {k: float(np.median([o[k] for o in others])) for k in hm.MODELS[m]}
            ll[m][i] = hm.loglik(task, m, p, games, cache)
        tables = hm._tables(task, "bounded", f["fits"]["bounded"]["params"], cache)
        for g in games:
            lg = hm.game_loglik(task, "bounded", f["fits"]["bounded"]["params"], g, cache, tables)
            rows.append({"pid": i, "game_index": order[(uid, tuple(q for q, _ in g))], "ll_per_decision": lg / len(g)})
    n_dec = sum(len(g) for _, gs in people for g in gs)
    rng = np.random.default_rng(seed)
    boot = rng.integers(len(fits), size=(5000, len(fits)))

    def diff(a, b):
        d = ll[a] - ll[b]
        lo, hi = np.percentile(d[boot].sum(1), [2.5, 97.5])
        return {"total": round(float(d.sum()), 1), "ci95": [round(float(lo), 1), round(float(hi), 1)],
                "participants_favoring_first": int((d > 0).sum())}

    import pandas as pd
    trend = smf.ols("ll_per_decision ~ game_index + C(pid)", data=pd.DataFrame(rows)).fit(cov_type="cluster", cov_kwds={"groups": pd.DataFrame(rows)["pid"]})
    out = {
        "procedure": procedure,
        "method": "leave-one-participant-out; per-parameter median of the other 38 participants' fits; solved games",
        "heldout_loglik_total": {m: round(float(v.sum()), 1) for m, v in ll.items()},
        "heldout_loglik_per_decision": {m: round(float(v.sum() / n_dec), 4) for m, v in ll.items()},
        "best_heldout_counts": {m: int(c) for m, c in zip(hm.MODELS, np.bincount(np.argmax(np.stack([ll[m] for m in hm.MODELS]), 0), minlength=3))},
        "bounded_minus_unbounded": diff("bounded", "unbounded"),
        "bounded_minus_leaky": diff("bounded", "leaky"),
        "game_index_trend_bounded": {"slope_ll_per_decision_per_game": round(float(trend.params["game_index"]), 4),
                                     "se": round(float(trend.bse["game_index"]), 4), "p": round(float(trend.pvalues["game_index"]), 4)},
    }
    (ROOT / f"results/study_a/e3_heldout{SUFFIX[procedure]}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def halves(procedure="greedy"):
    """Static vs. game-index model (sensitivity): the bounded model fitted to all solved games
    (4 parameters) against separate fits to games 1-5 and 6-10 (8 parameters), by AIC per participant."""
    static = {json.loads(p.read_text())["participant"]: json.loads(p.read_text())
              for p in sorted(out_dir(procedure=procedure).glob("participant_*.json"))}
    rows = []
    for p in sorted(out_dir(half="early", procedure=procedure).glob("participant_*.json")):
        e = json.loads(p.read_text())
        late_path = out_dir(half="late", procedure=procedure) / p.name
        if not late_path.exists():
            continue
        lt = json.loads(late_path.read_text())
        i = e["participant"]
        split_aic = e["fits"]["bounded"]["aic"] + lt["fits"]["bounded"]["aic"]
        rows.append({"participant": i, "d_aic_static_minus_split": static[i]["fits"]["bounded"]["aic"] - split_aic,
                     "early": e["fits"]["bounded"]["params"], "late": lt["fits"]["bounded"]["params"]})
    d = np.array([r["d_aic_static_minus_split"] for r in rows])
    out = {
        "procedure": procedure,
        "participants": len(rows),
        "mean_dAIC_static_minus_split": round(float(d.mean()), 1),
        "median_dAIC_static_minus_split": round(float(np.median(d)), 1),
        "participants_favoring_split": int((d > 0).sum()),
        "sign_test_p": round(float(stats.binomtest(int((d > 0).sum()), len(d)).pvalue), 4),
        "wilcoxon_late_minus_early": {k: {"median_diff": round(float(np.median([r["late"][k] - r["early"][k] for r in rows])), 4),
                                          "p": round(float(stats.wilcoxon([r["late"][k] for r in rows], [r["early"][k] for r in rows]).pvalue), 4)}
                                      for k in hm.MODELS["bounded"]},
        "median_params_early": {k: round(float(np.median([r["early"][k] for r in rows])), 4) for k in hm.MODELS["bounded"]},
        "median_params_late": {k: round(float(np.median([r["late"][k] for r in rows])), 4) for k in hm.MODELS["bounded"]},
    }
    (ROOT / f"results/study_a/e3_game_index_halves{SUFFIX[procedure]}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def cache_seed(rep=2, procedure="greedy"):
    """Does the model comparison depend on one run of the random cache search? Same people, same optimizer
    seeds; only the bounded model's cache differs (rebuilt with new search seeds)."""
    main = {json.loads(p.read_text())["participant"]: json.loads(p.read_text())
            for p in sorted(out_dir(procedure=procedure).glob("participant_*.json"))}
    other = {json.loads(p.read_text())["participant"]: json.loads(p.read_text())
             for p in sorted(out_dir(cache_rep=rep, procedure=procedure).glob("participant_*.json"))}
    ids = sorted(set(main) & set(other))
    b1 = np.array([main[i]["fits"]["bounded"]["aic"] for i in ids])
    b2 = np.array([other[i]["fits"]["bounded"]["aic"] for i in ids])
    unb = np.array([main[i]["fits"]["unbounded"]["aic"] for i in ids])
    leaky = np.array([main[i]["fits"]["leaky"]["aic"] for i in ids])

    def comparison(b):
        best = np.argmin(np.stack([unb, leaky, b]), 0)
        return {"mean_dAIC_bounded_vs_unbounded": round(float((unb - b).mean()), 1),
                "mean_dAIC_bounded_vs_leaky": round(float((leaky - b).mean()), 1),
                "best_model_counts": {"unbounded": int((best == 0).sum()), "leaky": int((best == 1).sum()), "bounded": int((best == 2).sum())}}

    a1 = [main[i]["fits"]["bounded"]["params"]["alpha"] for i in ids]
    a2 = [other[i]["fits"]["bounded"]["params"]["alpha"] for i in ids]
    out = {
        "procedure": procedure, "participants": len(ids), "replicate": rep,
        "cache_1": comparison(b1), f"cache_{rep}": comparison(b2),
        "bounded_aic_difference": {"mean": round(float((b2 - b1).mean()), 1), "median_abs": round(float(np.median(np.abs(b2 - b1))), 1),
                                   "max_abs": round(float(np.abs(b2 - b1).max()), 1),
                                   "pearson_r": round(float(np.corrcoef(b1, b2)[0, 1]), 4)},
        "alpha_spearman": round(float(stats.spearmanr(a1, a2).statistic), 3),
        "median_bounded_params": {f"cache_{r}": {k: round(float(np.median([d[i]["fits"]["bounded"]["params"][k] for i in ids])), 4)
                                                 for k in hm.MODELS["bounded"]} for r, d in ((1, main), (rep, other))},
    }
    (ROOT / f"results/study_a/e3_cache_seed{SUFFIX[procedure]}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def trial_order():
    """(uniqueid, guess sequence) -> game position 1..10 within the person's session (numeric trial_id)."""
    mm = hm.Task().mm
    games = defaultdict(list)
    for r in csv.DictReader(open(E3 / "mastermind_E3.csv")):
        games[(r["uniqueid"], int(r["trial_id"]))].append(r)
    order, per = {}, defaultdict(list)
    for (uid, t), rows in games.items():
        per[uid].append(t)
        rows.sort(key=lambda r: int(r["guess_number"]))
        order[(uid, tuple(mm.index(ast.literal_eval(r["guess"])) for r in rows))] = t
    rank = {uid: {t: k + 1 for k, t in enumerate(sorted(ts))} for uid, ts in per.items()}
    return {key: rank[key[0]][t] for key, t in order.items()}


def main():
    ap = argparse.ArgumentParser()
    proc = argparse.ArgumentParser(add_help=False)
    proc.add_argument("--procedure", choices=list(SUFFIX), default="greedy",
                      help="greedy (paper reproduction), converged (primary) or wide (sensitivity)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", parents=[proc])
    f.add_argument("--participant", type=int, required=True)
    f.add_argument("--models", nargs="+", default=list(hm.MODELS))
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--include-unsolved", action="store_true", help="sensitivity: fit on all games")
    f.add_argument("--half", choices=["early", "late"], help="sensitivity: fit games 1-5 or 6-10 only")
    f.add_argument("--cache-rep", type=int, default=1, help="use the independently rebuilt cache (build_cache.py --rep N)")
    s = sub.add_parser("summarize", parents=[proc])
    s.add_argument("--include-unsolved", action="store_true")
    sub.add_parser("heldout", parents=[proc])
    sub.add_parser("halves", parents=[proc])
    c = sub.add_parser("cache_seed", parents=[proc])
    c.add_argument("--rep", type=int, default=2)
    args = ap.parse_args()
    if args.cmd == "fit":
        fit_one(args.participant, args.models, args.seed, args.include_unsolved, args.half, args.cache_rep, args.procedure)
    elif args.cmd == "heldout":
        heldout(procedure=args.procedure)
    elif args.cmd == "halves":
        halves(args.procedure)
    elif args.cmd == "cache_seed":
        cache_seed(args.rep, args.procedure)
    else:
        summarize(args.include_unsolved, args.procedure)


if __name__ == "__main__":
    main()
