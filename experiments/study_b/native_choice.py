"""Native question choice (Study B): does the depth model's own choice between two
queries follow how useful each query is to it?

Data: Qwen3-32B, condition T0, the 24 choice-calibration families, 2 query pairs per family (a simpler
query with fewer distinct digits against a more informative one, the contrast of the human Experiment 2),
8 choices per pair, order randomized (the first-listed option is called A), no scores shown (collected with usefulness_choice.py).
Unparsed choices are excluded and counted.

Readout: P(choose A) = eps/2 + (1 - eps) * sigmoid(b + beta * (X(A) - X(B))), with b the bias for
the first-listed option, beta the weight on the query predictor X and eps a lapse rate; maximum
likelihood within fixed bounds (`fit_readout`). One readout per predictor X:
  usefulness_predicted  the model's own predicted usefulness U(q) under its fitted B3 process model (the
                        best planned model on the held-out families; parameters from the
                        processing-calibration families): the planned readout
  eig                   expected information gain (bits)
  vstar                 V* = m/K, usefulness under exact inference
  simple_structure      minus the number of distinct digits (a preference for simpler queries)
  usefulness_observed   the model's measured usefulness of q from its own branch runs (its identification
                        accuracy after q's answer: how hard q's answers are for it to use)
State-level difficulty (codes remaining, clue count) is the same for both queries of a pair, so it cannot
predict which one is chosen; query-level difficulty enters through simple_structure, vstar and
usefulness_observed.
  side_only             no query predictor (b and eps only)
Comparison: AIC on all pairs, and held-out log likelihood by leave-one-family-out cross-validation (each
family's choices predicted from fits to the other 23: test on held-out histories), with family-bootstrap
95% intervals of the differences to side_only. Also reported: the share of choices for the simpler query
(the Experiment 2 measure).

  python experiments/study_b/native_choice.py --model qwen3-32b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/study_b")]
import fit_process_models as F  # noqa: E402
from mastermind import human_models as hm  # noqa: E402
from mastermind import usefulness as us  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from mastermind.states import n_distinct  # noqa: E402

DATA = ROOT / "results/study_b/usefulness_choice"
OUT = ROOT / "results/study_b/native_choice"
PREDICTORS = ["usefulness_predicted", "eig", "vstar", "simple_structure", "usefulness_observed"]


def load(model, condition="T0", split="choice_calibration"):
    tag = f"{model}_{condition}_{split}"
    pairs = {p["pair_id"]: p for p in json.loads((DATA / f"{tag}_pairs.json").read_text())}
    rows = [json.loads(line) for line in open(DATA / f"{tag}.jsonl")]
    return pairs, rows


def query_features(mm, model, condition, pairs, rows):
    """Predictors for both queries of every pair: {pair_id: {"q_simple": {...}, "q_complex": {...}}}."""
    fams = {f["family_id"]: f for f in json.loads(F.ROOT.joinpath("data/splits/families_v1.json").read_text())}
    saved = json.loads((F.OUT / f"{model}_{condition}.json").read_text())
    proc = saved["best_heldout"]  # best planned model on the held-out families
    params = saved["fits"][proc]["params"]
    caches = {k: hm.BoundedCache(v) for k, v in F.CACHES.items()}
    feats = {}
    for pid, p in pairs.items():
        hist = [(mm.index(g), (b, w)) for g, b, w in fams[p["family_id"]]["history"]]
        mask = mm.consistent(hist)
        eig = mm.eig(mask)
        feats[pid] = {}
        for j, role in enumerate(("q_simple", "q_complex")):
            q = mm.index(p[role])
            assert abs(eig[q] - p["eig"][j]) < 1e-3, "EIG differs from the pair manifest"
            branch_rows = [r for r in rows if r["kind"] == "branch" and r["pair_id"] == pid and r["query"] == role]
            feats[pid][role] = {
                "code": p[role],
                "usefulness_predicted": us.predicted_usefulness(mm, proc, params, hist, q, caches),
                "eig": float(eig[q]),
                "vstar": float(p["vstar"][j]),
                "simple_structure": -float(n_distinct(mm, q)),
                "usefulness_observed": us.observed_usefulness(branch_rows),
            }
    return feats, proc


def design(feats, rows):
    """One row per parsed choice: family, y = 1 if the first-listed option (A) was chosen, X(A) - X(B)."""
    fam, y, dx, chose_simple, unparsed = [], [], [], [], 0
    for r in rows:
        if r["kind"] != "choice":
            continue
        if not r["parsed"]:
            unparsed += 1
            continue
        f = feats[r["pair_id"]]
        a = "q_simple" if f["q_simple"]["code"] == r["A"] else "q_complex"
        b = "q_complex" if a == "q_simple" else "q_simple"
        assert f[b]["code"] == r["B"]
        fam.append(r["pair_id"].split(":")[0])
        y.append(int(r["choice"] == "A"))
        dx.append([f[a][k] - f[b][k] for k in PREDICTORS])
        chose_simple.append(r["chose"] == "q_simple")
    return np.array(fam), np.array(y), np.array(dx), np.array(chose_simple), unparsed


def nll(theta, x, y):
    b, beta, eps = theta[0], theta[1], expit(theta[2])
    p = eps / 2 + (1 - eps) * expit(b + beta * x)
    return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


BOUND = 50.0  # |b| and |beta| (per SD of the predictor difference); logit(eps) within [-12, 12]


def fit_readout(x, y, with_x=True):
    """Maximum likelihood within fixed bounds, by differential evolution (3 seeds) polished with L-BFGS-B.
    For some predictors the likelihood keeps rising as beta grows: the readout tends to a step function of the
    predictor plus lapses, so the fit stops at |beta| = BOUND (flagged `beta_at_bound`); at that value the
    sigmoid is already a step for all but the smallest differences."""
    if with_x:
        f, bounds = (lambda t: nll(t, x, y)), [(-BOUND, BOUND), (-BOUND, BOUND), (-12.0, 12.0)]
    else:
        f, bounds = (lambda t: nll([t[0], 0.0, t[1]], x, y)), [(-BOUND, BOUND), (-12.0, 12.0)]
    best = None
    for seed in range(3):
        r = differential_evolution(f, bounds, seed=seed, tol=1e-10, maxiter=3000, polish=True)
        if best is None or r.fun < best.fun:
            best = r
    t = best.x if with_x else np.array([best.x[0], 0.0, best.x[1]])
    return {"b": float(t[0]), "beta": float(t[1]), "eps": float(expit(t[2])), "loglik": -float(best.fun), "k": 3 if with_x else 2,
            "beta_at_bound": bool(with_x and abs(t[1]) > BOUND - 1e-3)}


def run(model, condition="T0", n_boot=2000, seed=20261119):
    mm = get_game(4, 4)
    pairs, rows = load(model, condition)
    feats, proc = query_features(mm, model, condition, pairs, rows)
    fam, y, dx, chose_simple, unparsed = design(feats, rows)
    sd = dx.std(0)
    sd[sd == 0] = 1.0
    z = dx / sd  # predictor differences per SD (beta is per SD of the difference)
    names = PREDICTORS + ["side_only"]
    fams = sorted(set(fam))
    fits, heldout, at_bound = {}, {n: np.zeros(len(fams)) for n in names}, {n: 0 for n in names}
    for j, n in enumerate(names):
        x = z[:, j] if n != "side_only" else np.zeros(len(y))
        f = fit_readout(x, y, with_x=n != "side_only")
        f["aic"] = 2 * f["k"] - 2 * f["loglik"]
        fits[n] = f
        for i, g in enumerate(fams):  # leave one family out
            tr, te = fam != g, fam == g
            ft = fit_readout(x[tr], y[tr], with_x=n != "side_only")
            at_bound[n] += ft["beta_at_bound"]
            heldout[n][i] = -nll([ft["b"], ft["beta"], np.log(ft["eps"] / (1 - ft["eps"]))], x[te], y[te])
    rng = np.random.default_rng(seed)
    boots = rng.integers(len(fams), size=(n_boot, len(fams)))

    def diff(a, b):
        d = heldout[a] - heldout[b]
        return {"difference": round(float(d.sum()), 2), "ci95": np.percentile(d[boots].sum(1), [2.5, 97.5]).round(2).tolist(),
                "families_favoring_first": int((d > 0).sum())}

    by_pair = {}
    for pid, f in feats.items():
        by_pair[pid] = {r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f[r].items()} for r in f}
    simple_more_useful = {k: float(np.mean([f["q_simple"][k] > f["q_complex"][k] for f in feats.values()]))
                          for k in ("usefulness_predicted", "eig", "vstar", "usefulness_observed")}
    out = {
        "model": model, "condition": condition, "split": "choice_calibration", "families": len(fams), "pairs": len(pairs),
        "choices_parsed": int(len(y)), "choices_unparsed": unparsed,
        "process_model_for_predicted_usefulness": proc,
        "share_choosing_simpler_query": round(float(chose_simple.mean()), 3),
        "share_choosing_first_listed": round(float(y.mean()), 3),
        "share_of_pairs_where_simpler_query_scores_higher": {k: round(v, 3) for k, v in simple_more_useful.items()},
        "predictor_sd_of_differences": dict(zip(PREDICTORS, sd.round(4).tolist())),
        "fits_all_pairs": {n: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()} for n, f in fits.items()},
        "readout_bounds": {"b_and_beta_abs_max": BOUND, "logit_eps": [-12.0, 12.0]},
        "heldout_folds_beta_at_bound": {n: int(v) for n, v in at_bound.items()},
        "heldout_loglik_leave_one_family_out": {n: round(float(v.sum()), 2) for n, v in heldout.items()},
        "heldout_minus_side_only": {n: diff(n, "side_only") for n in PREDICTORS},
        "heldout_minus_usefulness_predicted": {n: diff("usefulness_predicted", n) for n in names if n != "usefulness_predicted"},
        "best_heldout": max(names, key=lambda n: heldout[n].sum()),
        "bootstrap": {"resamples": n_boot, "unit": "family", "seed": seed},
        "query_features": by_pair,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}_{condition}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "query_features"}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--condition", default="T0")
    a = ap.parse_args()
    run(a.model, a.condition)


if __name__ == "__main__":
    main()
