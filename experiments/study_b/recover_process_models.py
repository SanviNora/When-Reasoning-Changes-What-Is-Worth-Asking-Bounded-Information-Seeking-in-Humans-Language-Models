"""Parameter and model recovery for the process models M0-M3 on the planned B2 design
(processing-calibration families, the battery's item mix, 4 samples per item).

Each replicate draws parameters for every generating model (jittered around plausible values),
simulates the battery, and fits all five models. The confusion matrix counts how often the generating
model has the lowest AIC; models the data cannot tell apart are merged before B3 is interpreted.

  python experiments/study_b/recover_process_models.py run --rep 0
  python experiments/study_b/recover_process_models.py summarize

At the fitted values (add --fitted qwen3-32b_T0 to both): every model, the exploratory M1a included,
generates answers with the parameters fitted to that model's battery, on the same calibration items it
answered; all models are refitted. This shows which models the real data could tell apart.

  python experiments/study_b/recover_process_models.py refit_m3 [--fitted qwen3-32b_T0]

M3's likelihood was corrected after the first recovery runs (one kept-clue subset per reply, as in `pm.simulate`).
`refit_m3` regenerates every data set exactly (same seeds and order; each is checked against its stored M0 fit),
refits M3 with the corrected likelihood and updates the best model. At the fitted values, M3 now generates at
its corrected fit (`M3_corrected` in the fit file), so every model is refitted on those data sets.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/study_b"))
from mastermind import human_models as hm  # noqa: E402
from mastermind import process_models as pm  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from probe_battery import build_jobs  # noqa: E402

OUT = ROOT / "results/study_b/process_recovery"
CACHES = {"M2c": ROOT / "data/model_cache/bounded_cache_v1.npz", "M2s": ROOT / "data/model_cache/symmetric_cache_v1.npz"}
GEN = {"M0": {"zeta": 0.05}, "M1": {"nu": 0.2, "zeta": 0.05}, "M2c": {"alpha": 1.0, "nu": 0.05, "zeta": 0.05},
       "M2s": {"alpha": 1.0, "nu": 0.05, "zeta": 0.05}, "M3": {"k0": 1.5, "k1": -1.0, "zeta": 0.05}}


def design_items(mm, samples=4):
    fams = [f for f in json.loads((ROOT / "data/splits/families_v1.json").read_text()) if f["split"] == "processing_calibration"]
    items = []
    for j in build_jobs(mm, fams, "T0", samples):
        m = j["meta"]
        hist = [(mm.index(g), (b, w)) for g, b, w in m["history"]]
        if m["task"] == "identify":
            items.append(pm.Item(mm, hist, "identify", 0))
        else:
            items.append(pm.Item(mm, hist, "classify", [(mm.index(c), False) for c, _ in m["items"]]))
    return items


def jitter(p, rng):
    out = {}
    for k, v in p.items():
        if k in ("zeta", "nu"):
            out[k] = float(np.clip(v * np.exp(rng.normal(0, 0.4)), 0.005, 0.45))
        elif k == "alpha":
            out[k] = float(v * np.exp(rng.normal(0, 1.0)))
        else:
            out[k] = float(v + rng.normal(0, 0.5))
    return out


def run(rep):
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    items = design_items(mm)
    rng = np.random.default_rng(100 + rep)
    out = {"rep": rep, "items": len(items), "generated": {}}
    for gen, base in GEN.items():
        p = jitter(base, rng)
        sim = pm.simulate(gen, p, items, caches, rng)
        fits = {m: pm.fit(m, sim, caches) for m in pm.MODELS}
        out["generated"][gen] = {"true": p, "fits": fits, "best": min(fits, key=lambda m: fits[m]["aic"])}
        print(rep, gen, "->", out["generated"][gen]["best"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rep_{rep:02d}.json").write_text(json.dumps(out, indent=1))


def run_fitted(name, rep, seed=300):
    sys.path.insert(0, str(ROOT / "experiments/study_b"))
    import fit_process_models as F

    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    model, condition = name.rsplit("_", 1)
    saved = json.loads((F.OUT / f"{name}.json").read_text())
    items, _ = F.to_items(mm, F.load_records(model, condition))
    cal = [it for _, s, it, _ in items if s == "processing_calibration"]
    rng = np.random.default_rng(seed + rep)
    out = {"rep": rep, "fitted_to": name, "items": len(cal), "generated": {}}
    for gen in pm.MODELS:
        p = saved["fits"][gen]["params"]
        sim = pm.simulate(gen, p, cal, caches, rng)
        fits = {m: pm.fit(m, sim, caches) for m in pm.MODELS}
        out["generated"][gen] = {"true": p, "fits": fits, "best": min(fits, key=lambda m: fits[m]["aic"]),
                                 "best_planned": min(pm.PLANNED, key=lambda m: fits[m]["aic"])}
        print(rep, gen, "->", out["generated"][gen]["best"], flush=True)
    d = OUT.with_name(f"process_recovery_fitted_{name}")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"rep_{rep:02d}.json").write_text(json.dumps(out, indent=1))


def _replace_m3(gen_rec, sim, caches):
    old = gen_rec["fits"]["M3"]
    gen_rec["fits"]["M3"] = pm.fit("M3", sim, caches)
    gen_rec["M3_before_correction"] = {k: old[k] for k in ("params", "loglik", "aic")}


def _check(gen_rec, sim, caches, label):
    m0 = gen_rec["fits"]["M0"]
    assert abs(pm.loglik("M0", m0["params"], sim, caches) - m0["loglik"]) < 1e-6, f"{label}: data set not reproduced"


def _best(gen_rec):
    fits = gen_rec["fits"]
    gen_rec["best"] = min(fits, key=lambda m: fits[m]["aic"])
    if "best_planned" in gen_rec:
        gen_rec["best_planned"] = min(pm.PLANNED, key=lambda m: fits[m]["aic"])


def refit_m3_rep(args):
    rep, fitted = args
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    if fitted:
        sys.path.insert(0, str(ROOT / "experiments/study_b"))
        import fit_process_models as F
        path = OUT.with_name(f"process_recovery_fitted_{fitted}") / f"rep_{rep:02d}.json"
        rec = json.loads(path.read_text())
        if rec.get("m3_corrected"):
            return rep
        saved = json.loads((F.OUT / f"{fitted}.json").read_text())
        model, condition = fitted.rsplit("_", 1)
        items, _ = F.to_items(mm, F.load_records(model, condition))
        cal = [it for _, s, it, _ in items if s == "processing_calibration"]
        rng = np.random.default_rng(300 + rep)
        for gen in pm.MODELS:
            g = rec["generated"][gen]
            p = saved["M3_corrected"]["params"] if gen == "M3" else saved["fits"][gen]["params"]
            sim = pm.simulate(gen, p, cal, caches, rng)  # the number of draws does not depend on the parameters
            if gen == "M3":
                g["true"] = p
                g["fits"] = {m: pm.fit(m, sim, caches) for m in pm.MODELS}
            else:
                _check(g, sim, caches, f"{path.name} {gen}")
                _replace_m3(g, sim, caches)
            _best(g)
    else:
        path = OUT / f"rep_{rep:02d}.json"
        rec = json.loads(path.read_text())
        if rec.get("m3_corrected"):
            return rep
        items = design_items(mm)
        rng = np.random.default_rng(100 + rep)
        for gen, base in GEN.items():
            g = rec["generated"][gen]
            p = jitter(base, rng)
            assert all(abs(p[k] - g["true"][k]) < 1e-12 for k in p), f"{path.name} {gen}: parameters not reproduced"
            sim = pm.simulate(gen, p, items, caches, rng)
            _check(g, sim, caches, f"{path.name} {gen}")
            _replace_m3(g, sim, caches)
            _best(g)
    rec["m3_corrected"] = True
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(path)
    print(rep, "done", flush=True)
    return rep


def refit_m3(fitted, workers=8):
    from multiprocessing import Pool

    d = OUT.with_name(f"process_recovery_fitted_{fitted}") if fitted else OUT
    reps = sorted(int(q.stem.split("_")[1]) for q in d.glob("rep_*.json"))
    with Pool(workers) as pool:
        for _ in pool.imap_unordered(refit_m3_rep, [(r, fitted) for r in reps]):
            pass


def summarize_fitted(name):
    rows = [json.loads(p.read_text()) for p in sorted(OUT.with_name(f"process_recovery_fitted_{name}").glob("rep_*.json"))]
    names = list(pm.MODELS)
    conf = np.zeros((len(names), len(names)), int)
    conf_p = np.zeros((len(names), len(pm.PLANNED)), int)
    par = {}
    for r in rows:
        for i, gen in enumerate(names):
            g = r["generated"][gen]
            conf[i, names.index(g["best"])] += 1
            conf_p[i, pm.PLANNED.index(g["best_planned"])] += 1
            for k, v in g["true"].items():
                par.setdefault(f"{gen}.{k}", []).append((v, g["fits"][gen]["params"][k]))
    summary = {
        "fitted_to": name, "replicates": len(rows),
        "confusion_all_models": {g: dict(zip(names, map(int, conf[i]))) for i, g in enumerate(names)},
        "confusion_planned_only": {g: dict(zip(pm.PLANNED, map(int, conf_p[i]))) for i, g in enumerate(names)},
        "recovery_rate_all_models": {g: round(conf[i, i] / max(1, conf[i].sum()), 3) for i, g in enumerate(names)},
        "parameter_true_vs_median_refit": {k: [round(v[0][0], 4), round(float(np.median([b for _, b in v])), 4)] for k, v in par.items()},
    }
    # how the observed calibration difference between M1 and M2s compares with data sets each model generates
    saved = json.loads((ROOT / f"results/study_b/process_fits/{name}.json").read_text())
    obs = saved["fits"]["M1"]["aic"] - saved["fits"]["M2s"]["aic"]
    summary["dAIC_M1_minus_M2s"] = {
        "observed": round(obs, 2),
        "by_generating_model": {gen: {"median": round(float(np.median(v)), 2), "range": [round(float(min(v)), 2), round(float(max(v)), 2)],
                                      "share_at_least_observed": round(float(np.mean(np.array(v) >= obs)), 3)}
                                for gen in names
                                for v in [[r["generated"][gen]["fits"]["M1"]["aic"] - r["generated"][gen]["fits"]["M2s"]["aic"] for r in rows]]}}
    summary["recovery_rate_planned_only"] = {g: round(conf_p[i, pm.PLANNED.index(g)] / max(1, conf_p[i].sum()), 3)
                                             for i, g in enumerate(names) if g in pm.PLANNED}
    (ROOT / f"results/study_b/process_recovery_fitted_{name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def summarize():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("rep_*.json"))]
    names = list(rows[0]["generated"])  # the models these replicates generated and fitted (the planned M0-M3)
    conf = np.zeros((len(names), len(names)), int)
    par = {}
    for r in rows:
        for i, gen in enumerate(names):
            g = r["generated"][gen]
            conf[i, names.index(g["best"])] += 1
            for k, v in g["true"].items():
                par.setdefault(f"{gen}.{k}", []).append((v, g["fits"][gen]["params"][k]))
    summary = {
        "replicates": len(rows),
        "confusion_rows_generating_cols_best": {g: dict(zip(names, map(int, conf[i]))) for i, g in enumerate(names)},
        "recovery_rate": {g: round(conf[i, i] / max(1, conf[i].sum()), 3) for i, g in enumerate(names)},
        "parameter_recovery_corr": {k: round(float(np.corrcoef(np.array(v).T)[0, 1]), 3) for k, v in par.items() if len(v) > 2},
    }
    (ROOT / "results/study_b/process_recovery_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rep", type=int, required=True)
    r.add_argument("--fitted", help="e.g. qwen3-32b_T0: generate at that fit's parameters, on its calibration items")
    s = sub.add_parser("summarize")
    s.add_argument("--fitted")
    m3 = sub.add_parser("refit_m3")
    m3.add_argument("--fitted")
    m3.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "refit_m3":
        refit_m3(args.fitted, args.workers)
    elif args.cmd == "run":
        run_fitted(args.fitted, args.rep) if args.fitted else run(args.rep)
    else:
        summarize_fitted(args.fitted) if args.fitted else summarize()


if __name__ == "__main__":
    main()
