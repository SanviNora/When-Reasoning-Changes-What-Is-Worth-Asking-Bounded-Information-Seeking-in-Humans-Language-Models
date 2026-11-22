"""Model recovery for the three human models at the Experiment 3 sample size.

For participant i, each model generates synthetic play with that participant's fitted parameters
(same number of games, secrets drawn at random; only solved games kept, as in the fits). All three
models are then refitted to the synthetic data. The confusion matrix counts how often the generating
model has the lowest AIC.

  python experiments/study_a/recover_human_models.py run --participant 0
  python experiments/study_a/recover_human_models.py summarize
  add --procedure converged to both for the converged fits (generating parameters from the converged
  fits; every refit by the converged procedure; outputs end in _converged)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import human_models as hm  # noqa: E402

FITS = ROOT / "results/study_a/e3_fits"
CACHE = ROOT / "data/model_cache/bounded_cache_v1.npz"
OUT = ROOT / "results/study_a/e3_recovery"
SUFFIX = {"greedy": "", "converged": "_converged"}


def run(idx, seed=0, procedure="greedy"):
    task = hm.Task()
    cache = hm.BoundedCache(CACHE)
    fits_dir = FITS.with_name(FITS.name + SUFFIX[procedure])
    fit = json.loads((fits_dir / f"participant_{idx:02d}.json").read_text())
    rng = np.random.default_rng(seed + 1000 * idx)
    out = {"participant": idx, "generated": {}}
    for gen in hm.MODELS:
        p = fit["fits"][gen]["params"]
        tables = hm._tables(task, gen, p, cache)
        games = []
        while len(games) < fit["games"]:
            g = hm.simulate_game(task, gen, p, int(rng.integers(task.mm.N)), rng, cache, tables)
            if g[-1][1][0] == task.mm.P:
                games.append(g)
        if procedure == "greedy":
            refits = {m: hm.fit(task, m, games, cache=cache, seed=seed + idx) for m in hm.MODELS}
        else:  # no start uses the generating parameters, so no model gets an advantage
            refits = hm.fit_converged_models(task, list(hm.MODELS), games, cache, seed=seed + idx)
        out["generated"][gen] = {"true_params": p, "refits": refits,
                                 "best": min(refits, key=lambda m: refits[m]["aic"])}
        print(idx, gen, "->", out["generated"][gen]["best"], flush=True)
    out["procedure"] = procedure
    d = OUT.with_name(OUT.name + SUFFIX[procedure])
    d.mkdir(parents=True, exist_ok=True)
    (d / f"participant_{idx:02d}.json").write_text(json.dumps(out, indent=1))


def summarize(procedure="greedy"):
    rows = [json.loads(p.read_text()) for p in sorted(OUT.with_name(OUT.name + SUFFIX[procedure]).glob("participant_*.json"))]
    names = list(hm.MODELS)
    conf = np.zeros((3, 3), int)
    alpha_true, alpha_fit = [], []
    for r in rows:
        for i, gen in enumerate(names):
            g = r["generated"][gen]
            conf[i, names.index(g["best"])] += 1
            if gen == "bounded":
                alpha_true.append(g["true_params"]["alpha"])
                alpha_fit.append(g["refits"]["bounded"]["params"]["alpha"])
    summary = {
        "procedure": procedure,
        "participants": len(rows),
        "confusion_rows_generating_cols_best": {gen: dict(zip(names, map(int, conf[i]))) for i, gen in enumerate(names)},
        "recovery_rate": {gen: round(conf[i, i] / max(1, conf[i].sum()), 3) for i, gen in enumerate(names)},
        "alpha_recovery_spearman": round(float(np.corrcoef(np.argsort(np.argsort(alpha_true)), np.argsort(np.argsort(alpha_fit)))[0, 1]), 3) if len(rows) > 2 else None,
    }
    (ROOT / f"results/study_a/e3_recovery_summary{SUFFIX[procedure]}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--participant", type=int, required=True)
    r.add_argument("--procedure", choices=list(SUFFIX), default="greedy")
    s = sub.add_parser("summarize")
    s.add_argument("--procedure", choices=list(SUFFIX), default="greedy")
    args = ap.parse_args()
    run(args.participant, procedure=args.procedure) if args.cmd == "run" else summarize(args.procedure)


if __name__ == "__main__":
    main()
