"""C-c: can the fitted models predict, before the run, how the aid changes each question's usefulness?

Target, for a query pair (q1, q2) on a state h:
  Gamma(h) = [U(q1, T3) - U(q2, T3)] - [U(q1, T0) - U(q2, T0)],
  U(q) = P(the model names the secret after seeing q's answer) = (1/K) sum_a P(d in S_a).
A reply that is unfinished or has no valid code names nothing and counts as a failure (the proposal's definition).

Predictors, all fitted on calibration data only and frozen in prereg/prediction_lock.json before the run:
  support_map       PRIMARY. The B3 process model (the best planned model on held-out families, M2s, with its T0
                    parameters) plus one support parameter for the aid, l' = (1 - delta) l + delta c, fitted on the
                    support-calibration identification answers under T3. delta > 0 moves clue weights toward the
                    exact constraint, delta < 0 away from it (C-a shows the aid makes clue checking noisier).
  fitted_M2s        secondary: the same process model with the parameters fitted to the T0 and T3 probe batteries
                    (C-a); nothing fitted to usefulness
  fitted_M1a        exploratory: as fitted_M2s with the two-rate model M1a
  difficulty_regression  rival (the proposal's "simpler baselines based on problem difficulty"): ridge regression of
                    each query's observed usefulness change (T3 - T0) on V*, EIG, codes remaining, clues, distinct
                    digits and predicted baseline usefulness; penalty chosen by leave-one-family-out cross-validation
  reporting_only    rival: only the reporting noise changes under the aid (zeta refitted on T3 answers)
  uniform_gain      rival: the aid changes every query's usefulness by the same amount, so Gamma = 0
  ideal_noise       rival: an exact reasoner whose reporting noise differs between conditions
Every model-based prediction includes each condition's share of unfinished replies (calibration runs): the process
models describe finished answers, while U counts unfinished ones as failures.

Commands, in order; nothing before `score` reads confirmation outcomes:
  calibrate  fit every predictor on the calibration usefulness runs (support- and choice-calibration families)
  power      simulation that fixes the number of confirmation families and the samples per branch
  select     choose confirmation pairs on the fresh families (draw_families.py), without outcomes, in three strata
  lock       write prereg/prediction_lock.json (every prediction; hashes of the data, code and analysis plan). Commit
             it and register it on OSF before the confirmation run
  score      after the run: family-level MSPE per predictor; the primary model against each rival (paired, cluster
             bootstrap over families, Holm across the four rivals); noise-corrected MSPE, bias, calibration slope,
             sign accuracy for large effects, reversal rate, missing pairs

  python experiments/study_c/prediction_test.py calibrate --cap 15000
  python experiments/study_c/prediction_test.py power --cap 15000
  python experiments/study_c/prediction_test.py select --cap 15000
  python experiments/study_c/prediction_test.py lock --cap 15000 --samples 16
  python experiments/study_c/prediction_test.py score --cap 15000
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import human_models as hm  # noqa: E402
from mastermind import process_models as pm  # noqa: E402
from mastermind import usefulness as us  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from mastermind.states import n_distinct  # noqa: E402

FAMILIES_V1 = ROOT / "data/splits/families_v1.json"
FAMILIES_C = ROOT / "data/splits/families_c_v1.json"
MANIFEST_C = ROOT / "data/splits/manifest_c_v1.json"
PAIRS = ROOT / "data/splits/confirmation_pairs_c_v1.json"
UC = ROOT / "results/study_b/usefulness_choice"
FITS = ROOT / "results/study_b/process_fits"
OUT = ROOT / "results/study_c"
CAL = OUT / "cc_calibration.json"
LOCK = ROOT / "prereg/prediction_lock.json"
PLAN = ROOT / "prereg/analysis_plan.md"
CACHES = {"M2c": ROOT / "data/model_cache/bounded_cache_v1.npz", "M2s": ROOT / "data/model_cache/symmetric_cache_v1.npz"}
BASE_CAP = 6144
CAL_SPLITS = ("support_calibration", "choice_calibration")
CONF_SPLIT = "c_confirmation"
PRIMARY = "support_map"
SECONDARY = ["fitted_M2s", "fitted_M1a"]
RIVALS = ["difficulty_regression", "reporting_only", "uniform_gain", "ideal_noise"]
PREDICTORS = [PRIMARY] + SECONDARY + RIVALS
FEATURES = ["vstar", "eig", "K", "n_clues", "distinct_digits", "pred_U_T0"]
CODE = ["experiments/study_c/prediction_test.py", "experiments/study_c/choice_shift.py", "mastermind/usefulness.py",
        "mastermind/process_models.py", "experiments/study_b/usefulness_choice.py", "mastermind/prompts.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tag(model, condition, split, cap):
    return f"{model}_{condition}_{split}" + ("" if cap == BASE_CAP else f"_cap{cap}")


def families(mm):
    out = {}
    for path in (FAMILIES_V1, FAMILIES_C):
        if path.exists():
            out.update({f["family_id"]: [(mm.index(g), (b, w)) for g, b, w in f["history"]] for f in json.loads(path.read_text())})
    return out


def records(model, condition, split, cap):
    """(branch rows, choice rows, pairs) of one usefulness run."""
    t = tag(model, condition, split, cap)
    pairs = {p["pair_id"]: p for p in json.loads((UC / f"{t}_pairs.json").read_text())}
    rows = [json.loads(line) for line in open(UC / f"{t}.jsonl")]
    return [r for r in rows if r["kind"] == "branch"], [r for r in rows if r["kind"] == "choice"], pairs


def usefulness(rows, rng=None):
    """U from the branch rows of one (pair, query, condition): sum_a (K_a / K) mean(Y_a), Y = 1/K_a if the named code is
    in the branch, else 0 (unparsed and unfinished replies included). With `rng`, samples are resampled within
    each branch (sampling noise)."""
    by, ka = {}, {}
    for r in rows:
        o = tuple(r["outcome"])
        ka[o] = r["K_a"]
        by.setdefault(o, []).append(r["Y"])
    K = sum(ka.values())
    return sum(ka[o] * np.mean(rng.choice(v, len(v)) if rng is not None else v) for o, v in by.items()) / K


def observed_gamma(groups, pairs, n_noise=0, seed=0):
    """{pair_id: (Gamma, noise variance)} for pairs with both queries in both conditions. groups: {(cond, pair, query): rows}."""
    rng = np.random.default_rng(seed)
    out = {}
    for pid in pairs:
        keys = [(c, pid, q) for c in ("T0", "T3") for q in ("q_simple", "q_complex")]
        if not all(k in groups for k in keys):
            continue

        def gam(r=None):
            u = {k: usefulness(groups[k], r) for k in keys}
            return (u[keys[2]] - u[keys[3]]) - (u[keys[0]] - u[keys[1]])
        noise = float(np.var([gam(rng) for _ in range(n_noise)])) if n_noise else None
        out[pid] = (float(gam()), noise)
    return out


def group(rows_by_condition):
    g = {}
    for c, rows in rows_by_condition.items():
        for r in rows:
            g.setdefault((c, r["pair_id"], r["query"]), []).append(r)
    return g


class Branches:
    """Parsed identification answers in branches, pre-processed for fast likelihoods over delta and zeta."""

    def __init__(self, mm, fams, rows, pairs, model, theta, caches):
        self.items = []
        for r in rows:
            if r["answer"] is None:
                continue
            p = pairs[r["pair_id"]]
            hist = fams[p["family_id"]] + [(mm.index(p[r["query"]]), tuple(r["outcome"]))]
            clues = [(h, mm.encode(b, w)) for h, (b, w) in hist]
            c = np.stack([mm.table[h] == o for h, o in clues]).astype(np.float64)
            item = pm.Item.__new__(pm.Item)
            item.clues, item.c = clues, c
            base = c if model == "M0" else pm.clue_weights(model, theta, item, caches)
            self.items.append((base, c, mm.index(r["answer"])))

    def loglik(self, delta, zeta):
        ll = 0.0
        for base, c, d in self.items:
            g = np.prod(np.clip((1 - delta) * base + delta * c, 0.0, 1.0), axis=0)
            ll += np.log((1 - zeta) * g[d] / g.sum() + zeta / g.size)
        return ll


def features(mm, hist, q, u0_pred):
    mask = mm.consistent(hist)
    return [len(set(mm.table[q, mask].tolist())) / mask.sum(), float(mm.eig(mask, [q])[0]),
            float(mask.sum()), float(len(hist)), float(n_distinct(mm, q)), u0_pred]


class Predictor:
    """Predicted U and Gamma for every predictor, with memoized usefulness computations."""

    def __init__(self, mm, cal, caches):
        self.mm, self.cal, self.caches, self.memo = mm, cal, caches, {}

    def U(self, fam, hist, q, model, params, delta=0.0):
        key = (fam, q, model, json.dumps(params, sort_keys=True), round(delta, 12))
        if key not in self.memo:
            self.memo[key] = us.predicted_usefulness(self.mm, model, params, hist, q, self.caches, delta)
        return self.memo[key]

    def condition_u(self, name, fam, hist, q):
        """(U under T0, U under T3) for a model-based predictor, unfinished replies included."""
        cal = self.cal
        f0, f3 = 1 - cal["unfinished"]["T0"], 1 - cal["unfinished"]["T3"]
        proc, th0 = cal["process_model"], cal["theta"]["T0"][cal["process_model"]]
        if name == "support_map":
            u0, u3 = self.U(fam, hist, q, proc, th0), self.U(fam, hist, q, proc, th0, cal["support_map"]["delta"])
        elif name == "fitted_M2s":
            u0, u3 = self.U(fam, hist, q, proc, th0), self.U(fam, hist, q, proc, cal["theta"]["T3"][proc])
        elif name == "fitted_M1a":
            u0, u3 = self.U(fam, hist, q, "M1a", cal["theta"]["T0"]["M1a"]), self.U(fam, hist, q, "M1a", cal["theta"]["T3"]["M1a"])
        elif name == "reporting_only":
            u0, u3 = self.U(fam, hist, q, proc, th0), self.U(fam, hist, q, proc, {**th0, "zeta": cal["reporting_only"]["zeta_T3"]})
        elif name == "ideal_noise":
            u0 = self.U(fam, hist, q, "M0", {"zeta": cal["ideal_noise"]["zeta_T0"]})
            u3 = self.U(fam, hist, q, "M0", {"zeta": cal["ideal_noise"]["zeta_T3"]})
        else:
            raise ValueError(name)
        return f0 * u0, f3 * u3

    def gamma(self, fam, hist, q1, q2, names=PREDICTORS):
        out = {}
        for name in names:
            if name == "uniform_gain":
                out[name] = 0.0
            elif name == "difficulty_regression":
                dr = self.cal["difficulty_regression"]
                proc = self.cal["process_model"]
                th0 = self.cal["theta"]["T0"][proc]

                def dU(q):
                    x = (np.array(features(self.mm, hist, q, self.U(fam, hist, q, proc, th0))) - np.array(dr["mean"])) / np.array(dr["sd"])
                    return float(np.r_[1.0, x] @ np.array(dr["beta"]))
                out[name] = dU(q1) - dU(q2)
            else:
                a0, a3 = self.condition_u(name, fam, hist, q1)
                b0, b3 = self.condition_u(name, fam, hist, q2)
                out[name] = (a3 - b3) - (a0 - b0)
        return out


def ridge(Z, y, lam):
    pen = lam * np.diag([0.0] + [1.0] * (Z.shape[1] - 1))
    return np.linalg.solve(Z.T @ Z + pen, Z.T @ y)


def calibrate(model, cap):
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    fams = families(mm)
    fit = {c: json.loads((FITS / f"{model}_{c}.json").read_text()) for c in ("T0", "T3")}
    proc = fit["T0"]["best_heldout"]
    assert proc in pm.PLANNED, proc
    theta = {c: {m: fit[c]["fits"][m]["params"] for m in (proc, "M1a")} for c in ("T0", "T3")}
    runs = {(c, s): records(model, c, s, cap) for c in ("T0", "T3") for s in CAL_SPLITS}
    unfinished = {c: float(np.mean([not r["parsed"] for s in CAL_SPLITS for r in runs[(c, s)][0]])) for c in ("T0", "T3")}

    rows3, _, pairs3 = runs[("T3", "support_calibration")]
    rows0, _, pairs0 = runs[("T0", "support_calibration")]
    br3 = Branches(mm, fams, rows3, pairs3, proc, theta["T0"][proc], caches)
    z0 = theta["T0"][proc]["zeta"]
    d = minimize_scalar(lambda x: -br3.loglik(x, z0), bounds=(-1.0, 1.0), method="bounded", options={"xatol": 1e-6})
    zeta_rep = minimize_scalar(lambda z: -br3.loglik(0.0, z), bounds=(1e-4, 0.999), method="bounded").x
    ex0 = Branches(mm, fams, rows0, pairs0, "M0", {}, caches)
    ex3 = Branches(mm, fams, rows3, pairs3, "M0", {}, caches)
    zeta_ideal = {c: minimize_scalar(lambda z: -b.loglik(0.0, z), bounds=(1e-4, 0.999), method="bounded").x
                  for c, b in (("T0", ex0), ("T3", ex3))}

    # Difficulty regression: each query's observed change in U (T3 - T0) on both calibration splits.
    pred = Predictor(mm, {"process_model": proc, "theta": theta}, caches)
    X, y, fam_of = [], [], []
    for s in CAL_SPLITS:
        g = group({"T0": runs[("T0", s)][0], "T3": runs[("T3", s)][0]})
        pairs = runs[("T0", s)][2]
        for (c, pid, q), rows in g.items():
            if c != "T3" or ("T0", pid, q) not in g:
                continue
            fam = pairs[pid]["family_id"]
            qi = mm.index(pairs[pid][q])
            X.append(features(mm, fams[fam], qi, pred.U(fam, fams[fam], qi, proc, theta["T0"][proc])))
            y.append(usefulness(rows) - usefulness(g[("T0", pid, q)]))
            fam_of.append(fam)
    X, y, fam_of = np.array(X), np.array(y), np.array(fam_of)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    grid = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4]
    cv = {}
    for lam in grid:
        err = []
        for f in np.unique(fam_of):
            tr, te = fam_of != f, fam_of == f
            err += list((Z[te] @ ridge(Z[tr], y[tr], lam) - y[te]) ** 2)
        cv[lam] = float(np.mean(err))
    lam = min(cv, key=cv.get)
    beta = ridge(Z, y, lam)

    gam = {s: observed_gamma(group({"T0": runs[("T0", s)][0], "T3": runs[("T3", s)][0]}), runs[("T0", s)][2], n_noise=200)
           for s in CAL_SPLITS}
    obs = np.array([v[0] for s in CAL_SPLITS for v in gam[s].values()])
    noise = np.array([v[1] for s in CAL_SPLITS for v in gam[s].values()])
    samples = sorted({r["sample"] for r in runs[("T0", "support_calibration")][0]})
    cal = {
        "model": model, "cap": cap, "process_model": proc, "theta": theta, "unfinished": unfinished,
        "support_map": {"delta": float(d.x), "at_bound": bool(abs(abs(d.x) - 1) < 1e-3), "zeta": z0,
                        "loglik": float(-d.fun), "loglik_delta0": float(br3.loglik(0.0, z0))},
        "reporting_only": {"zeta_T0": z0, "zeta_T3": float(zeta_rep)},
        "ideal_noise": {"zeta_T0": float(zeta_ideal["T0"]), "zeta_T3": float(zeta_ideal["T3"])},
        "difficulty_regression": {"features": FEATURES, "mean": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist(),
                                  "ridge_lambda": lam, "cv_mse": {str(k): v for k, v in cv.items()}, "queries": int(len(y))},
        "observed_gamma": {"pairs": int(len(obs)), "mean": float(obs.mean()), "sd": float(obs.std()),
                           "noise_sd": float(np.sqrt(noise.mean())),
                           "true_sd": float(np.sqrt(max(obs.var() - noise.mean(), 0.0))),
                           "samples_per_branch": len(samples), "branches_per_query": float(np.mean(
                               [len({tuple(r["outcome"]) for r in rows}) for rows in group(
                                   {"T0": [r for s in CAL_SPLITS for r in runs[("T0", s)][0]]}).values()]))},
        "data": {"answers_T3_support": len(br3.items), "answers_T0_support": len(ex0.items),
                 "files": {tag(model, c, s, cap): sha(UC / f"{tag(model, c, s, cap)}.jsonl") for c in ("T0", "T3") for s in CAL_SPLITS},
                 "fits": {c: sha(FITS / f"{model}_{c}.json") for c in ("T0", "T3")}},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    CAL.write_text(json.dumps(cal, indent=2))
    print(json.dumps({k: v for k, v in cal.items() if k not in ("theta", "data")}, indent=2))


def power(r2, pairs_per_family, n_sims=2000, alpha=0.05, seed=0):
    """Smallest effect of interest: a predictor that explains r2 of the true variance of Gamma (MSPE lower than the
    zero predictor by r2 * tau^2). True Gamma ~ N(0, tau^2) and sampling noise ~ N(0, sigma^2 * S0 / S) per pair, both
    estimated on the calibration pairs; the predictor is the calibrated linear predictor with that r2. Power is the
    chance that the paired family-level MSPE difference is significant at alpha / 4 (one-sided; Holm's first step
    across the four rivals). Also: the 95% interval half-width of that difference, in units of tau^2."""
    cal = json.loads(CAL.read_text())
    og = cal["observed_gamma"]
    tau2, sig2, s0, nb = og["true_sd"] ** 2, og["noise_sd"] ** 2, og["samples_per_branch"], og["branches_per_query"]
    rng = np.random.default_rng(seed)
    z = norm.ppf(1 - alpha / 4)
    rows = []
    for S in (4, 8, 16, 24, 32):
        for N in (40, 60, 80, 120, 160):
            gt = rng.normal(0, np.sqrt(tau2), (n_sims, N, pairs_per_family))
            m = r2 * gt + np.sqrt(r2 * (1 - r2) * tau2) * rng.normal(size=gt.shape)
            obs = gt + rng.normal(0, np.sqrt(sig2 * s0 / S), gt.shape)
            d = (obs ** 2 - (m - obs) ** 2).mean(2)  # per family: zero predictor minus model
            mean, se = d.mean(1), d.std(1, ddof=1) / np.sqrt(N)
            calls = N * pairs_per_family * 2 * nb * S * 2
            rows.append({"samples_per_branch": S, "families": N, "power": round(float(np.mean(mean - z * se > 0)), 3),
                         "ci95_halfwidth_over_tau2": round(float(np.median(1.96 * se) / tau2), 3), "usefulness_calls": int(calls)})
    res = {"tau_sd": np.sqrt(tau2), "noise_sd_at_calibration": np.sqrt(sig2), "calibration_samples_per_branch": s0,
           "branches_per_query": nb, "smallest_effect_r2": r2, "alpha_per_test": alpha / 4, "pairs_per_family": pairs_per_family,
           "design": rows}
    (OUT / "cc_power.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def select(model, pairs_per_stratum, seed=11, candidates=60):
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    cal = json.loads(CAL.read_text())
    pred = Predictor(mm, cal, caches)
    fams = json.loads(FAMILIES_C.read_text())
    rng = np.random.default_rng(seed)
    rows = []
    for f in fams:
        fid = f["family_id"]
        hist = [(mm.index(g), (b, w)) for g, b, w in f["history"]]
        mask = mm.consistent(hist)
        eig = mm.eig(mask)
        probes = [int(q) for q in np.flatnonzero(~mask & (eig > 0))]  # no immediate win, informative
        cands = sorted({tuple(sorted(rng.choice(probes, 2, replace=False).tolist())) for _ in range(candidates)})
        scored = []
        for q1, q2 in cands:
            g = pred.gamma(fid, hist, q1, q2, names=[PRIMARY])[PRIMARY]
            m1, m2 = (len(set(mm.table[q, mask].tolist())) for q in (q1, q2))
            scored.append({"q1": q1, "q2": q2, "gamma_hat": g, "vstar_equal": m1 == m2,
                           "eig_gap": abs(float(eig[q1] - eig[q2])), "digit_gap": abs(n_distinct(mm, q1) - n_distinct(mm, q2))})
        strata = {
            "a_matched_vstar": sorted([s for s in scored if s["vstar_equal"]], key=lambda s: -abs(s["gamma_hat"])),
            "b_similar_eig_diff_burden": sorted([s for s in scored if s["eig_gap"] < 0.2 and s["digit_gap"] >= 1],
                                                key=lambda s: -abs(s["gamma_hat"])),
            "c_predicted_zero": sorted(scored, key=lambda s: abs(s["gamma_hat"])),
        }
        used = set()
        for name, lst in strata.items():
            k = 0
            for s in lst:
                if k == pairs_per_stratum:
                    break
                if (s["q1"], s["q2"]) in used:
                    continue
                used.add((s["q1"], s["q2"]))
                q1, q2 = (s["q1"], s["q2"]) if s["gamma_hat"] >= 0 else (s["q2"], s["q1"])
                rows.append({"pair_id": f"{fid}:{name[0]}{k}", "family_id": fid, "stratum": name,
                             "q_simple": mm.to_str(q1), "q_complex": mm.to_str(q2),
                             "eig": [round(float(eig[q1]), 4), round(float(eig[q2]), 4)],
                             "distinct_digits": [n_distinct(mm, q1), n_distinct(mm, q2)],
                             "note": "q_simple = q1 and q_complex = q2 of the Gamma definition (field names of the runner)"})
                k += 1
    PAIRS.write_text(json.dumps(rows, indent=1))
    print(json.dumps({"pairs": len(rows), "families": len(fams),
                      "by_stratum": {s: sum(r["stratum"] == s for r in rows) for s in sorted({r["stratum"] for r in rows})}}, indent=2))


def lock(model, samples, choice_samples):
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    cal = json.loads(CAL.read_text())
    assert PLAN.exists(), "write prereg/analysis_plan.md first"
    pred = Predictor(mm, cal, caches)
    fams = families(mm)
    preds = []
    for p in json.loads(PAIRS.read_text()):
        g = pred.gamma(p["family_id"], fams[p["family_id"]], mm.index(p["q_simple"]), mm.index(p["q_complex"]))
        preds.append({"pair_id": p["pair_id"], "family_id": p["family_id"], "stratum": p["stratum"],
                      "gamma_hat": {k: round(float(v), 8) for k, v in g.items()}})
    doc = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model, "conditions": ["T0", "T3"],
        "design": {"reply_cap": cal["cap"], "branch_samples": samples, "choice_samples": choice_samples,
                   "families": len({p["family_id"] for p in preds}), "pairs": len(preds), "split": CONF_SPLIT},
        "primary": PRIMARY, "secondary": SECONDARY, "rivals": RIVALS,
        "primary_endpoint": "family-level MSPE of Gamma; MSPE(rival) - MSPE(support_map) for each rival, cluster bootstrap "
                            "over families; Holm across the four rivals at alpha 0.05; H3 is supported if support_map beats "
                            "every rival",
        "hashes": {**{p: sha(ROOT / p) for p in CODE},
                   "families_v1": sha(FAMILIES_V1), "families_c_v1": sha(FAMILIES_C), "manifest_c_v1": sha(MANIFEST_C),
                   "pairs": sha(PAIRS), "calibration": sha(CAL), "analysis_plan": sha(PLAN),
                   "prompt_freeze_v1": sha(ROOT / "configs/prompt_freeze_v1.json"),
                   "model_registry": sha(ROOT / "configs/model_registry.json")},
        "calibration": cal, "predictions": preds}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(doc, indent=1))
    print(f"wrote {LOCK}: {len(preds)} predictions, sha256 {sha(LOCK)}. Commit it and register it on OSF before the run.")


def score(model, cap, n_boot=10000, seed=0):
    doc = json.loads(LOCK.read_text())
    assert doc["design"]["reply_cap"] == cap
    for p, h in doc["hashes"].items():  # the analysis must use the locked files
        if p in CODE:
            assert sha(ROOT / p) == h, f"{p} changed after the lock"
    b0, c0, pairs = records(model, "T0", CONF_SPLIT, cap)
    b3, c3, _ = records(model, "T3", CONF_SPLIT, cap)
    gam = observed_gamma(group({"T0": b0, "T3": b3}), pairs, n_noise=300, seed=seed)
    rows, missing = [], []
    for pr in doc["predictions"]:
        if pr["pair_id"] not in gam:
            missing.append(pr["pair_id"])
            continue
        obs, noise = gam[pr["pair_id"]]
        rows.append({"pair_id": pr["pair_id"], "family": pr["family_id"], "stratum": pr["stratum"], "obs": obs, "noise": noise,
                     **pr["gamma_hat"]})
    fam_ids = sorted({r["family"] for r in rows})
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(fam_ids), size=(n_boot, len(fam_ids)))

    def per_family(f):
        return np.array([np.mean([f(r) for r in rows if r["family"] == fam]) for fam in fam_ids])
    sq = {k: per_family(lambda r, k=k: (r[k] - r["obs"]) ** 2) for k in PREDICTORS}
    noise_f = per_family(lambda r: r["noise"])

    def compare(model_name, rivals):
        out = {}
        for k in rivals:
            d = sq[k] - sq[model_name]
            boots = d[idx].mean(1)
            out[k] = {"mspe_rival_minus_model": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                      "p_one_sided": float(np.mean(boots <= 0)), "families_favoring_model": int((d > 0).sum())}
        ps = sorted((v["p_one_sided"], k) for k, v in out.items())  # Holm
        running = 0.0
        for i, (p, k) in enumerate(ps):
            running = max(running, min(1.0, (len(ps) - i) * p))
            out[k]["p_holm"] = running
        return out

    obs = np.array([r["obs"] for r in rows])
    fam_of = np.array([fam_ids.index(r["family"]) for r in rows])
    big = np.abs(obs) >= np.percentile(np.abs(obs), 75)
    by_pred = {}
    for k in PREDICTORS:
        p = np.array([r[k] for r in rows])
        slope_boot = []
        for b in idx[:2000]:
            sel = np.concatenate([np.flatnonzero(fam_of == i) for i in b])
            if np.var(p[sel]) > 0:
                slope_boot.append(np.polyfit(p[sel], obs[sel], 1)[0])
        by_pred[k] = {"mspe": float(sq[k].mean()), "mspe_noise_corrected": float(sq[k].mean() - noise_f.mean()),
                      "bias": float(np.mean(p - obs)),
                      "calibration_slope": float(np.polyfit(p, obs, 1)[0]) if np.var(p) > 0 else None,
                      "calibration_slope_ci95": [float(np.percentile(slope_boot, 2.5)), float(np.percentile(slope_boot, 97.5))] if slope_boot else None,
                      "sign_accuracy_large_effects": float(np.mean(np.sign(p[big]) == np.sign(obs[big]))) if np.any(p != 0) else None}

    # Reversals: the aid flips which query of a pair is more useful.
    g = group({"T0": b0, "T3": b3})
    flips = []
    for r_ in rows:
        pid = r_["pair_id"]
        d0 = usefulness(g[("T0", pid, "q_simple")]) - usefulness(g[("T0", pid, "q_complex")])
        d3 = usefulness(g[("T3", pid, "q_simple")]) - usefulness(g[("T3", pid, "q_complex")])
        flips.append(float(np.sign(d0) * np.sign(d3) < 0))
    flips = np.array(flips)
    flip_f = np.array([flips[fam_of == i].mean() for i in range(len(fam_ids))])
    res = {"pairs_locked": len(doc["predictions"]), "pairs_scored": len(rows), "missing_pairs": missing,
           "families": len(fam_ids), "observed_gamma": {"mean": float(obs.mean()), "sd": float(obs.std()),
                                                        "noise_sd": float(np.sqrt(np.mean([r["noise"] for r in rows])))},
           "primary_vs_rivals": compare(PRIMARY, RIVALS),
           "secondary_vs_rivals": {m: compare(m, RIVALS) for m in SECONDARY},
           "by_predictor": by_pred,
           "reversal_rate": {"observed": float(flip_f.mean()), "ci95": [float(np.percentile(flip_f[idx].mean(1), 2.5)),
                                                                         float(np.percentile(flip_f[idx].mean(1), 97.5))]},
           "by_stratum": {s: {"pairs": int(sum(r["stratum"] == s for r in rows)),
                              "mean_observed_gamma": float(np.mean([r["obs"] for r in rows if r["stratum"] == s])),
                              "mspe": {k: float(np.mean([(r[k] - r["obs"]) ** 2 for r in rows if r["stratum"] == s])) for k in PREDICTORS}}
                          for s in sorted({r["stratum"] for r in rows})},
           "choices": {"T0": len(c0), "T3": len(c3)}}
    (OUT / "cc_score.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "power", "select", "lock", "score"):
        s = sub.add_parser(name)
        s.add_argument("--model", default="qwen3-32b")
        s.add_argument("--cap", type=int, default=BASE_CAP, help="reply cap of the usefulness runs")
        if name == "power":
            s.add_argument("--r2", type=float, default=0.20, help="smallest effect of interest: share of the true variance of Gamma")
            s.add_argument("--pairs-per-family", type=int, default=3)
        if name == "select":
            s.add_argument("--pairs-per-stratum", type=int, default=1)
        if name == "lock":
            s.add_argument("--samples", type=int, required=True, help="samples per branch in the confirmation run")
            s.add_argument("--choice-samples", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "calibrate":
        calibrate(args.model, args.cap)
    elif args.cmd == "power":
        power(args.r2, args.pairs_per_family)
    elif args.cmd == "select":
        select(args.model, args.pairs_per_stratum)
    elif args.cmd == "lock":
        lock(args.model, args.samples, args.choice_samples)
    else:
        score(args.model, args.cap)


if __name__ == "__main__":
    main()
