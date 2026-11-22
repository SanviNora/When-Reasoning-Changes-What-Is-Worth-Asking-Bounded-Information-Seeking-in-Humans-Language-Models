"""B3: fit the process models to the battery and compare them.

  fit       fit M0-M3 on the processing-calibration families; score held-out log likelihood on the
            confirmation families; AIC on calibration. One condition per call.
  profile   error profile: false exclusions (saying a still-possible code is ruled out) and false
            retentions (near misses / far misses called possible), by clue count and query complexity,
            for the multi-clue and single-clue probes separately, plus identification accuracy. Uses the
            processing-calibration and confirmation families only (the same families in every condition;
            the development families were run at T0 only).

  confirm   confirmation test of the exploratory model M1a on the reserve families (never used before it,
            plan in experiments/study_b/B3_CONFIRMATION.md): every model keeps the parameters saved by `fit`
            (processing-calibration families); held-out log likelihood per family, family bootstrap.

  boot      family bootstrap of the calibration fits: resample the processing-calibration families with
            replacement and refit every model (one resample per call; run many as a job array).
  intervals uncertainty for B3: family-bootstrap intervals of the held-out differences between models
            (confirmation families, saved parameters, no refit), and, from the `boot` resamples, parameter
            intervals and how often each model is best by AIC.
  refit_m3  refit M3 with its corrected likelihood (one kept-clue subset per reply, as in its simulator; the
            first fits drew the subset separately for every code judged in a reply) on the calibration families
            and on every `boot` resample, stored as `M3_corrected`. `fits` is left unchanged: it holds the frozen
            parameters of the pre-stated reserve test (B3_CONFIRMATION.md), at which every clue is kept and the
            two likelihoods agree. `intervals` uses `M3_corrected` for M3.

  python experiments/study_b/fit_process_models.py fit --model qwen3-32b --condition T0
  python experiments/study_b/fit_process_models.py boot --model qwen3-32b --condition T0 --rep 0 [--count 10]
  python experiments/study_b/fit_process_models.py intervals --model qwen3-32b --condition T0
  python experiments/study_b/fit_process_models.py profile --model qwen3-32b --conditions T0 T1 T2 T3
  python experiments/study_b/fit_process_models.py confirm --model qwen3-32b --condition T0
  python experiments/study_b/fit_process_models.py refit_m3 --model qwen3-32b --condition T0
  python experiments/study_b/fit_process_models.py product_rule --model qwen3-32b --condition T0

  product_rule  check of the observation model's product rule: g(s) = prod_i l_i(s)
            assumes independent per-clue checks, so false exclusions rise with the number of clues. It is compared
            with the prespecified alternatives, one global check: the min rule g = min_i l_i and the k-th root
            g = (prod_i l_i)^(1/k). M1 and M2s (the best planned model) are refitted under each rule on the
            processing-calibration families (same starts as `fit`) and scored on the confirmation families;
            observed and predicted false exclusions by clue count are reported next to them.
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import human_models as hm  # noqa: E402
from mastermind import process_models as pm  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from mastermind.states import n_distinct  # noqa: E402

BATTERY = ROOT / "results/study_b/battery"
OUT = ROOT / "results/study_b/process_fits"
PROFILE_SPLITS = ("processing_calibration", "confirmation")
CACHES = {"M2c": ROOT / "data/model_cache/bounded_cache_v1.npz", "M2s": ROOT / "data/model_cache/symmetric_cache_v1.npz"}


def load_records(model, condition):
    rows = []
    for path in sorted(glob.glob(str(BATTERY / f"{model}_{condition}_*.jsonl"))):
        if "_cap" in Path(path).name:  # runs at a different reply cap are a different setting, never mixed in
            continue
        rows += [json.loads(line) for line in open(path)]
    seen, out = set(), []
    for r in rows:  # the same item can appear in two files (e.g. development run)
        key = (r["family_id"], r["task"], r["clue_index"], r["sample"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def to_items(mm, records):
    items, fmt = [], {"answered": 0, "unparsed": 0}
    for r in records:
        hist = [(mm.index(g), (b, w)) for g, b, w in r["history"]]
        if r["task"] == "identify":
            if r["answer"] is None:
                fmt["unparsed"] += 1
                continue
            items.append(("identify", r["split"], pm.Item(mm, hist, "identify", mm.index(r["answer"])), r["family_id"]))
        else:
            ans = [(mm.index(c), v) for c, v in (r.get("labels") or {}).items() if v is not None]
            if not ans:
                fmt["unparsed"] += 1
                continue
            items.append(("classify", r["split"], pm.Item(mm, hist, "classify", ans), r["family_id"]))
        fmt["answered"] += 1
    return items, fmt


def _exact_gap(cache, alpha, mm):
    """Largest |pi - exact| over every query and outcome: 0 means the representation is exact inference."""
    rep = cache.representation(alpha)
    gap = 0.0
    for q in range(mm.N):
        for o in set(mm.table[q].tolist()):
            gap = max(gap, float(np.abs(rep[q, o] - (mm.table[q] == o)).max()))
    return gap


def boundary_notes(m, params, caches, mm, max_clues=6):
    """Fitted values at a boundary where the model reduces to a simpler one, so that its fit is not evidence
    for its own mechanism. For M2c/M2s this is checked directly: is the fitted representation exact?"""
    notes = []
    if m in ("M2c", "M2s"):
        gap = _exact_gap(caches[m], params["alpha"], mm)
        if gap < 0.01:
            notes.append(f"representation at the fitted alpha is exact (max gap {gap:.3g}): the model reduces to M1")
        elif params["alpha"] <= 1e-3:
            kind = "false retentions" if m == "M2c" else "false exclusions and retentions"
            notes.append(f"alpha at or near its lower bound (<= 1e-3), but the representation is not exact (max gap {gap:.2f}): "
                         f"with the distance weight fixed at 1, near-exact approximations keep some weight as alpha -> 0 ({kind})")
    if m == "M3":
        r = expit(params["k0"] + params["k1"] * np.arange(max_clues))
        if r.min() > 0.999:
            notes.append("every clue retained (r > 0.999): the model reduces to M0")
    for k in ("nu", "nu_x", "nu_r", "zeta"):
        if k in params and (params[k] < 1e-3 or params[k] > 0.999):
            notes.append(f"{k} at a boundary ({params[k]:.4g})")
    return notes


def fit(model, condition):
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    items, fmt = to_items(mm, load_records(model, condition))
    cal = [it for _, s, it, _ in items if s == "processing_calibration"]
    held = [it for _, s, it, _ in items if s == "confirmation"]
    res = {"model": model, "condition": condition, "items_calibration": len(cal), "items_heldout": len(held), "format": fmt, "fits": {}}
    for m in pm.MODELS:
        f = pm.fit(m, cal, caches)
        f["heldout_loglik"] = pm.loglik(m, f["params"], held, caches)
        f["boundary"] = boundary_notes(m, f["params"], caches, mm)
        res["fits"][m] = f
        print(condition, m, round(f["aic"], 1), round(f["heldout_loglik"], 1), f["params"], flush=True)
    # the planned comparison (M0-M3) is the primary result; exploratory models are labelled separately
    res["planned_models"] = pm.PLANNED
    res["best_aic"] = min(pm.PLANNED, key=lambda m: res["fits"][m]["aic"])
    res["best_heldout"] = max(pm.PLANNED, key=lambda m: res["fits"][m]["heldout_loglik"])
    res["exploratory"] = {"models": pm.EXPLORATORY,
                          "note": "added after the planned comparison, motivated by the error profile; "
                                  "confirmation test on the reserve families: `confirm`",
                          "best_aic_all": min(res["fits"], key=lambda m: res["fits"][m]["aic"]),
                          "best_heldout_all": max(res["fits"], key=lambda m: res["fits"][m]["heldout_loglik"])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}_{condition}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "fits"}, indent=2))


def profile(model, conditions):
    mm = get_game(4, 4)
    out = {}
    for c in conditions:
        allrecs = load_records(model, c)
        replies = {}
        for r in allrecs:  # every reply (prompt x sample), by split: parsed or not; code judgments asked and answered
            x = replies.setdefault(r["split"], {"replies": 0, "parsed": 0, "unparsed_by_task": {}, "judgments_asked": 0, "judgments_answered": 0})
            x["replies"] += 1
            x["parsed"] += bool(r.get("parsed"))
            if not r.get("parsed"):
                x["unparsed_by_task"][r["task"]] = x["unparsed_by_task"].get(r["task"], 0) + 1
            if r["task"] in ("classify", "classify_single"):
                x["judgments_asked"] += len(r["items"])
                x["judgments_answered"] += sum(v is not None for v in (r.get("labels") or {}).values())
        every = [r for r in allrecs if r["split"] in PROFILE_SPLITS]
        recs = [r for r in every if r["task"] in ("classify", "classify_single") and r.get("labels")]
        ident = [r for r in every if r["task"] == "identify" and r.get("parsed")]
        cells = {}
        for r in recs:
            n_clues = len(r["history"])
            newest = mm.index(r["history"][-1][0])
            for (code, kind) in r["items"]:
                said = r["labels"].get(code)
                if said is None:
                    continue
                key = (r["task"], kind, n_clues, n_distinct(mm, newest))
                cells.setdefault(key, []).append(said if kind != "possible" else not said)  # error indicator
        rows = [{"task": t, "kind": k, "clues": n, "query_distinct_digits": d, "n": len(v), "error_rate": round(float(np.mean(v)), 3)}
                for (t, k, n, d), v in sorted(cells.items())]
        def rate(kind, task=None):
            v = [x for (t, k, _, _), vals in cells.items() for x in vals if k == kind and (task is None or t == task)]
            return round(float(np.mean(v)), 3) if v else None
        def pooled(select):  # rate per group over the cells select() maps to a group (None = skip)
            acc = {}
            for (t, k, n, d), vals in cells.items():
                g = select(t, k, n, d)
                if g is not None:
                    acc.setdefault(g[0], {}).setdefault(g[1], []).extend(vals)
            return {a: {str(b): {"n": len(v), "rate": round(float(np.mean(v)), 3)} for b, v in sorted(x.items())} for a, x in sorted(acc.items())}

        def balanced(task):
            sens = 1 - rate("possible", task)
            wrong = [x for (t, k, _, _), vals in cells.items() for x in vals if t == task and k != "possible"]
            spec = 1 - float(np.mean(wrong))
            return {"sensitivity": round(sens, 3), "specificity": round(spec, 3), "balanced_accuracy": round((sens + spec) / 2, 3)}

        out[c] = {"multi_clue": {k: rate(k, "classify") for k in ("possible", "near", "far")},
                  "single_clue": {k: rate(k, "classify_single") for k in ("possible", "near", "far")},
                  "balanced_accuracy": {"multi_clue": balanced("classify"), "single_clue": balanced("classify_single")},
                  "replies_by_split": replies,
                  "by_clue_count": pooled(lambda t, k, n, d: (k, n)),
                  "by_query_distinct_digits": {"single_clue": pooled(lambda t, k, n, d: (k, d) if t == "classify_single" else None),
                                               "multi_clue": pooled(lambda t, k, n, d: (k, d) if t == "classify" else None)},
                  "breakdown_note": "clue count = clues in the probe's history (single-clue probes have 1); query distinct "
                                    "digits = complexity of the newest clue's query. Near misses are ruled out only by the "
                                    "newest clue, far misses by an earlier one (so near vs. far also contrasts clue age).",
                  "identification": round(float(np.mean([bool(r["correct"]) for r in ident])), 3) if ident else None,
                  "note": "rates are error rates: 'possible' = false exclusion, 'near'/'far' = false retention",
                  "cells": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}_error_profile_{'+'.join(conditions)}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({c: {k: v for k, v in d.items() if k != "cells"} for c, d in out.items()}, indent=2))


def _p_possible(m, p, item, caches):
    """P(answer "still possible") for every code, as in pm.item_loglik (for M3: the marginal over kept-clue
    subsets of one code, which gives the expected error rate whether the subset is drawn per reply or per code)."""
    if m == "M3":
        r = expit(p["k0"] + p["k1"] * np.arange(len(item.clues))[::-1])
        g = np.prod(r[:, None] * item.c + (1 - r[:, None]), axis=0)
    else:
        g = np.prod(pm.clue_weights(m, p, item, caches), axis=0)
    return (1 - p["zeta"]) * g + p["zeta"] / 2


def confirm(model, condition, split="reserve", n_boot=2000, seed=20261118):
    """Pre-stated test (B3_CONFIRMATION.md): with the saved calibration parameters, M1a must have a higher
    held-out log likelihood on the reserve families than every planned model, and the family-bootstrap 95%
    interval of the difference to the best planned model must lie above 0."""
    mm = get_game(4, 4)
    fit_path = OUT / f"{model}_{condition}.json"
    saved = json.loads(fit_path.read_text())
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    recs = [r for r in load_records(model, condition) if r["split"] == split]
    if not recs:
        raise SystemExit(f"no {split} records for {model} {condition}")
    items, fmt = to_items(mm, recs)
    fams = sorted({f for *_, f in items})
    models = list(pm.MODELS)
    ll = np.zeros((len(fams), len(models)))  # held-out log likelihood per family and model
    index = {f: i for i, f in enumerate(fams)}
    for _, _, it, fam in items:
        for j, m in enumerate(models):
            ll[index[fam], j] += pm.item_loglik(m, saved["fits"][m]["params"], it, caches)
    total = dict(zip(models, ll.sum(0).round(2).tolist()))
    best_planned = max(pm.PLANNED, key=total.get)
    rng = np.random.default_rng(seed)
    boots = rng.integers(len(fams), size=(n_boot, len(fams)))
    diffs = {}
    for m in pm.PLANNED:
        d = ll[:, models.index("M1a")] - ll[:, models.index(m)]
        bs = d[boots].sum(1)
        diffs[m] = {"difference": round(float(d.sum()), 2), "ci95": np.percentile(bs, [2.5, 97.5]).round(2).tolist(),
                    "families_favoring_M1a": int((d > 0).sum())}
    passed = all(total["M1a"] > total[m] for m in pm.PLANNED) and diffs[best_planned]["ci95"][0] > 0
    corrected = None
    if "M3_corrected" in saved:  # not part of the pre-stated test: M3 refitted with the corrected likelihood
        d = ll[:, models.index("M1a")].copy()
        for _, _, it, fam in items:
            d[index[fam]] -= pm.item_loglik("M3", saved["M3_corrected"]["params"], it, caches)
        corrected = {"note": "not part of the pre-stated test (frozen parameters); M3 refitted with the corrected "
                             "likelihood (`refit_m3`), compared with M1a on the same reserve families",
                     "heldout_loglik_M3_corrected": round(float(total["M1a"] - d.sum()), 2),
                     "M1a_minus_M3_corrected": {"difference": round(float(d.sum()), 2),
                                                "ci95": np.percentile(d[boots].sum(1), [2.5, 97.5]).round(2).tolist(),
                                                "families_favoring_M1a": int((d > 0).sum())}}

    # descriptive: observed vs predicted error rates by probe type (no test)
    cells = {}
    for r in recs:
        if r["task"] not in ("classify", "classify_single") or not r.get("labels"):
            continue
        it = pm.Item(mm, [(mm.index(g), (b, w)) for g, b, w in r["history"]], "classify", [])
        p_yes = {m: _p_possible(m, saved["fits"][m]["params"], it, caches) for m in models}
        for code, kind in r["items"]:
            said = r["labels"].get(code)
            if said is None:
                continue
            x, err = mm.index(code), kind == "possible"  # error = "ruled out" for possible codes, "possible" otherwise
            c = cells.setdefault(f"{r['task']}:{kind}", {"n": 0, "observed": 0.0, **{m: 0.0 for m in models}})
            c["n"] += 1
            c["observed"] += (not said) if err else said
            for m in models:
                c[m] += (1 - p_yes[m][x]) if err else p_yes[m][x]
    error_rates = {k: {"n": c["n"], **{m: round(c[m] / c["n"], 3) for m in ["observed"] + models}} for k, c in sorted(cells.items())}
    res = {"model": model, "condition": condition, "split": split, "families": len(fams), "items": len(items),
           "format": fmt, "parameters_from": fit_path.name,
           "parameters_sha256": hashlib.sha256(json.dumps({m: saved["fits"][m]["params"] for m in models},
                                                          sort_keys=True).encode()).hexdigest(),
           "heldout_loglik": total, "best_planned": best_planned, "M1a_minus_planned": diffs,
           "bootstrap": {"resamples": n_boot, "unit": "family", "seed": seed}, "confirmed": bool(passed),
           "error_rates": error_rates}
    if corrected:
        res["M3_corrected_check"] = corrected
    others = [s_ for s_ in ("development", "processing_calibration", "choice_calibration", "support_calibration", "confirmation")]
    related, rrecs = unrelated_multi_clue(model, condition, split, others)
    sub, _ = to_items(mm, rrecs)
    n_f, rd = _family_diffs(sub, {m: saved["fits"][m]["params"] for m in models}, caches, "M1a", list(pm.PLANNED), seed)
    res["robustness_multi_clue_unrelated"] = {
        "note": "not part of the pre-stated test: reserve families sharing no sub-history of "
                ">= 2 clues with a family of any other split, multi-clue replies only (single clues recur across splits)",
        "families_excluded_as_related": related, "families": n_f, "items": len(sub),
        "single_clue_probes": single_clue_overlap(model, condition, split, others), "M1a_minus_planned": rd}
    (OUT / f"{model}_{condition}_{split}_confirmation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def boot(model, condition, rep, seed=20261119):
    """One family-bootstrap resample of the calibration fits (all models refitted)."""
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    items, _ = to_items(mm, load_records(model, condition))
    by = {}
    for _, s, it, fam in items:
        if s == "processing_calibration":
            by.setdefault(fam, []).append(it)
    fams = sorted(by)
    rng = np.random.default_rng(seed + rep)
    pick = [fams[i] for i in rng.integers(len(fams), size=len(fams))]
    sample = [it for f in pick for it in by[f]]
    fits = {m: pm.fit(m, sample, caches) for m in pm.MODELS}
    out = {"model": model, "condition": condition, "rep": rep, "seed": seed + rep, "families": pick, "items": len(sample),
           "fits": {m: {k: f[k] for k in ("params", "loglik", "k", "aic")} for m, f in fits.items()}}
    d = OUT / f"boot_{model}_{condition}"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"rep_{rep:03d}.json.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(d / f"rep_{rep:03d}.json")
    print(rep, {m: round(f["aic"], 1) for m, f in fits.items()}, flush=True)


def intervals(model, condition, n_boot=2000, seed=20261118):
    """Held-out differences with family-bootstrap intervals (saved parameters), plus the `boot` refits."""
    mm = get_game(4, 4)
    saved = json.loads((OUT / f"{model}_{condition}.json").read_text())
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    items, _ = to_items(mm, [r for r in load_records(model, condition) if r["split"] == "confirmation"])
    fams = sorted({f for *_, f in items})
    models = list(pm.MODELS)
    index = {f: i for i, f in enumerate(fams)}
    ll = np.zeros((len(fams), len(models)))
    for _, _, it, fam in items:
        for j, m in enumerate(models):
            ll[index[fam], j] += pm.item_loglik(m, planned_fit(saved, m)["params"], it, caches)
    total = dict(zip(models, ll.sum(0).round(2).tolist()))
    assert all(abs(total[m] - planned_fit(saved, m)["heldout_loglik"]) < 0.05 for m in models), "held-out totals differ from the fit"
    rng = np.random.default_rng(seed)
    boots = rng.integers(len(fams), size=(n_boot, len(fams)))

    def diff(a, b):
        d = ll[:, models.index(a)] - ll[:, models.index(b)]
        return {"difference": round(float(d.sum()), 2), "ci95": np.percentile(d[boots].sum(1), [2.5, 97.5]).round(2).tolist(),
                "families_favoring_first": int((d > 0).sum()), "families": len(fams)}

    best = max(pm.PLANNED, key=total.get)
    out = {"model": model, "condition": condition, "heldout_split": "confirmation", "families": len(fams), "items": len(items),
           "M3": "corrected likelihood (M3_corrected)" if "M3_corrected" in saved else "as fitted",
           "heldout_loglik": total, "best_planned_heldout": best,
           "best_planned_minus_other_planned": {m: diff(best, m) for m in pm.PLANNED if m != best},
           "exploratory_M1a_minus_planned": {m: diff("M1a", m) for m in pm.PLANNED},
           "bootstrap": {"resamples": n_boot, "unit": "family", "seed": seed}}
    related, recs = unrelated_multi_clue(model, condition, "confirmation", ("processing_calibration",))
    sub, _ = to_items(mm, recs)
    n_f, diffs = _family_diffs(sub, {m: planned_fit(saved, m)["params"] for m in models}, caches, best,
                               [m for m in pm.PLANNED if m != best], seed)
    out["robustness_multi_clue_unrelated"] = {
        "note": "post hoc: confirmation families sharing no sub-history of >= 2 clues with a calibration "
                "family, multi-clue replies only (single clues recur across splits)",
        "families_excluded_as_related": related, "families": n_f, "items": len(sub),
        "single_clue_probes": single_clue_overlap(model, condition, "confirmation", ("processing_calibration",)),
        "best_planned_minus_other_planned": diffs}
    reps = [json.loads(q.read_text()) for q in sorted((OUT / f"boot_{model}_{condition}").glob("rep_*.json"))]
    if "M3_corrected" in saved:
        assert all("M3_corrected" in r for r in reps), "run refit_m3: some resamples lack the corrected M3 fit"
        for r in reps:
            r["fits"]["M3"] = r["M3_corrected"]
    if reps:
        ci = lambda xs: np.percentile(xs, [2.5, 50, 97.5]).round(4).tolist()  # noqa: E731
        best_aic = [min(pm.PLANNED, key=lambda m: r["fits"][m]["aic"]) for r in reps]
        best_all = [min(r["fits"], key=lambda m: r["fits"][m]["aic"]) for r in reps]
        out["calibration_bootstrap"] = {
            "resamples": len(reps), "unit": "processing-calibration family",
            "params_2.5_50_97.5": {m: {k: ci([r["fits"][m]["params"][k] for r in reps]) for k in pm.MODELS[m]} for m in models},
            "share_best_aic_planned": {m: round(best_aic.count(m) / len(reps), 3) for m in pm.PLANNED},
            "share_best_aic_all": {m: round(best_all.count(m) / len(reps), 3) for m in models},
            "dAIC_vs_" + saved["best_aic"]: {m: ci([r["fits"][m]["aic"] - r["fits"][saved["best_aic"]]["aic"] for r in reps])
                                              for m in models if m != saved["best_aic"]},
        }
    (OUT / f"{model}_{condition}_intervals.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "calibration_bootstrap"}, indent=2))


RULES = ("product", "min", "kth_root")


def _combine(l_, rule):
    if rule == "product":
        return np.prod(l_, axis=0)
    if rule == "min":
        return l_.min(axis=0)
    return np.prod(l_, axis=0) ** (1.0 / l_.shape[0])


def rule_loglik(m, p, items, caches, rule):
    total = 0.0
    for it in items:
        g = _combine(pm.clue_weights(m, p, it, caches), rule)
        z = p["zeta"]
        if it.task == "classify":
            xs = np.array([x for x, _ in it.answer])
            said = np.array([v for _, v in it.answer], bool)
            py = (1 - z) * g[xs] + z / 2
            total += float(np.where(said, np.log(py), np.log1p(-py)).sum())
        else:
            total += float(np.log((1 - z) * g[it.answer] / g.sum() + z / g.size))
    return total


def product_rule(model, condition):
    from scipy.optimize import minimize

    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    items, _ = to_items(mm, load_records(model, condition))
    cal = [it for _, s, it, _ in items if s == "processing_calibration"]
    held = [it for _, s, it, _ in items if s == "confirmation"]
    saved = json.loads((OUT / f"{model}_{condition}.json").read_text())
    starts = [pm.START] + [{**pm.START, "alpha": a, "zeta": z} for a in (0.03, 3.0) for z in (0.02, 0.3)]  # as pm.fit
    res = {"model": model, "condition": condition, "rules": list(RULES), "fits": {}}
    for m in ("M1", "M2s"):
        for rule in RULES:
            best = None
            for st in starts:
                v0 = pm._to_vec(m, {k: st[k] for k in pm.MODELS[m]})
                r = minimize(lambda v: -rule_loglik(m, pm._from_vec(m, v), cal, caches, rule), v0, method="Nelder-Mead",
                             options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400})
                if best is None or r.fun < best.fun:
                    best = r
            prm = pm._from_vec(m, best.x)
            k = len(pm.MODELS[m])
            f = {"params": prm, "loglik": -float(best.fun), "aic": 2 * k + 2 * float(best.fun),
                 "heldout_loglik": rule_loglik(m, prm, held, caches, rule)}
            if rule == "product":  # must reproduce the planned fit
                assert abs(f["loglik"] - saved["fits"][m]["loglik"]) < 0.05, (m, f["loglik"], saved["fits"][m]["loglik"])
            res["fits"][f"{m}:{rule}"] = f
            print(m, rule, round(f["aic"], 1), round(f["heldout_loglik"], 1), prm, flush=True)
    # observed and predicted false exclusion (a still-possible code called ruled out) by number of clues
    by_k = {}
    for it in cal + held:
        if it.task != "classify":
            continue
        k = len(it.clues)
        exact = np.prod(it.c, axis=0) > 0.5
        for x, said in it.answer:
            if not exact[x]:
                continue
            c = by_k.setdefault(k, {"n": 0, "observed": 0.0, **{key: 0.0 for key in res["fits"]}})
            c["n"] += 1
            c["observed"] += not said
            for key, f in res["fits"].items():
                m, rule = key.split(":")
                g = _combine(pm.clue_weights(m, f["params"], it, caches), rule)
                c[key] += 1 - ((1 - f["params"]["zeta"]) * g[x] + f["params"]["zeta"] / 2)
    res["false_exclusion_by_clue_count"] = {str(k): {"n": c["n"], **{key: round(v / c["n"], 3) for key, v in c.items() if key != "n"}}
                                            for k, c in sorted(by_k.items())}
    for m in ("M1", "M2s"):
        best_aic = min(RULES, key=lambda r: res["fits"][f"{m}:{r}"]["aic"])
        best_held = max(RULES, key=lambda r: res["fits"][f"{m}:{r}"]["heldout_loglik"])
        res[f"{m}_best_rule"] = {"aic": best_aic, "heldout": best_held}
    res["product_rule_holds"] = all(res[f"{m}_best_rule"] == {"aic": "product", "heldout": "product"} for m in ("M1", "M2s"))
    (OUT / f"{model}_{condition}_product_rule.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "fits"}, indent=2))


def _sub_keys(history):
    """Canonical keys of every sub-history of >= 2 clues (the full history included)."""
    from itertools import combinations

    from mastermind.symmetry import family_key
    mm = get_game(4, 4)
    h = [(mm.index(g), (b, w)) for g, b, w in history]
    return {family_key(list(c)) for n in range(2, len(h) + 1) for c in combinations(h, n)}


def unrelated_multi_clue(model, condition, split, reference_splits):
    """Post hoc robustness check: families of `split` that share no sub-history of >= 2 clues
    (under symmetry) with any family of `reference_splits`, and only their multi-clue replies (classify with the full
    history, identify); single-clue probes are left out because the same single clues recur across splits."""
    fams = json.loads((ROOT / "data/splits/families_v1.json").read_text())
    ref = set().union(*(_sub_keys(f["history"]) for f in fams if f["split"] in reference_splits))
    related = sorted(f["family_id"] for f in fams if f["split"] == split and _sub_keys(f["history"]) & ref)
    every = load_records(model, condition)
    recs = [r for r in every if r["split"] == split and r["task"] in ("classify", "identify") and r["family_id"] not in related]
    return related, recs


def single_clue_overlap(model, condition, split, reference_splits):
    """Distinct single clues (under symmetry) probed alone in `split`, and how many were also probed alone in a
    reference split."""
    from mastermind.symmetry import family_key
    mm = get_game(4, 4)

    def keys(splits):
        out = set()
        for r in load_records(model, condition):
            if r["task"] == "classify_single" and r["split"] in splits:
                g, b_, w = r["history"][-1] if len(r["history"]) == 1 else r["history"][r["clue_index"]]
                out.add(family_key([(mm.index(g), (b_, w))]))
        return out
    mine, ref = keys((split,)), keys(reference_splits)
    return {"distinct_single_clues": len(mine), "also_probed_alone_in_reference_splits": len(mine & ref)}


def _family_diffs(items, params, caches, a, others, boots_seed, n_boot=2000):
    fams = sorted({f for *_, f in items})
    index = {f: i for i, f in enumerate(fams)}
    ll = {m: np.zeros(len(fams)) for m in [a, *others]}
    for _, _, it, fam in items:
        for m in ll:
            ll[m][index[fam]] += pm.item_loglik(m, params[m], it, caches)
    boots = np.random.default_rng(boots_seed).integers(len(fams), size=(n_boot, len(fams)))
    out = {}
    for m in others:
        d = ll[a] - ll[m]
        out[m] = {"difference": round(float(d.sum()), 2), "ci95": np.percentile(d[boots].sum(1), [2.5, 97.5]).round(2).tolist(),
                  "families_favoring_first": int((d > 0).sum())}
    return len(fams), out


def planned_fit(saved, m):
    """The fit used for model m in the comparison: M3 refitted with the corrected likelihood when available."""
    return saved["M3_corrected"] if m == "M3" and "M3_corrected" in saved else saved["fits"][m]


def refit_m3(model, condition):
    """See `refit_m3` in the module docstring."""
    mm = get_game(4, 4)
    items, _ = to_items(mm, load_records(model, condition))
    cal = [it for _, s, it, _ in items if s == "processing_calibration"]
    held = [it for _, s, it, _ in items if s == "confirmation"]
    path = OUT / f"{model}_{condition}.json"
    saved = json.loads(path.read_text())
    f = pm.fit("M3", cal)
    f["heldout_loglik"] = pm.loglik("M3", f["params"], held)
    f["boundary"] = boundary_notes("M3", f["params"], None, mm)
    f["note"] = ("M3 refitted with the corrected likelihood (one kept-clue subset per reply); `fits.M3` keeps the "
                 "frozen parameters of the pre-stated reserve test")
    saved["M3_corrected"] = f
    path.write_text(json.dumps(saved, indent=2))
    print("M3 corrected", round(f["aic"], 1), round(f["heldout_loglik"], 1), f["params"], flush=True)
    by = {}
    for _, s, it, fam in items:
        if s == "processing_calibration":
            by.setdefault(fam, []).append(it)
    for q in sorted((OUT / f"boot_{model}_{condition}").glob("rep_*.json")):
        r = json.loads(q.read_text())
        sample = [it for fam in r["families"] for it in by[fam]]
        assert len(sample) == r["items"], q.name
        assert abs(pm.loglik("M0", r["fits"]["M0"]["params"], sample) - r["fits"]["M0"]["loglik"]) < 1e-6, q.name
        g = pm.fit("M3", sample)
        r["M3_corrected"] = {k: g[k] for k in ("params", "loglik", "k", "aic")}
        tmp = q.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(r, indent=1))
        tmp.replace(q)
    print("bootstrap resamples refitted", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--model", required=True)
    f.add_argument("--condition", default="T0")
    p = sub.add_parser("profile")
    p.add_argument("--model", required=True)
    p.add_argument("--conditions", nargs="+", default=["T0"])
    c = sub.add_parser("confirm")
    c.add_argument("--model", required=True)
    c.add_argument("--condition", default="T0")
    b = sub.add_parser("boot")
    b.add_argument("--model", required=True)
    b.add_argument("--condition", default="T0")
    b.add_argument("--rep", type=int, required=True, help="first resample")
    b.add_argument("--count", type=int, default=1, help="number of resamples from --rep on")
    i = sub.add_parser("intervals")
    i.add_argument("--model", required=True)
    i.add_argument("--condition", default="T0")
    pr = sub.add_parser("product_rule")
    pr.add_argument("--model", required=True)
    pr.add_argument("--condition", default="T0")
    r3 = sub.add_parser("refit_m3")
    r3.add_argument("--model", required=True)
    r3.add_argument("--condition", default="T0")
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.model, args.condition)
    elif args.cmd == "profile":
        profile(args.model, args.conditions)
    elif args.cmd == "boot":
        for r in range(args.rep, args.rep + args.count):
            boot(args.model, args.condition, r)
    elif args.cmd == "intervals":
        intervals(args.model, args.condition)
    elif args.cmd == "product_rule":
        product_rule(args.model, args.condition)
    elif args.cmd == "refit_m3":
        refit_m3(args.model, args.condition)
    else:
        confirm(args.model, args.condition)


if __name__ == "__main__":
    main()
