"""C-b: do the model's own question choices shift toward the questions the aid makes more useful?

For each query pair (q1, q2) the model chooses between the two queries under T0 and under T3 (the choice prompt
carries the same guidance as the condition, so the chooser is the supported reasoner; order randomized, no scores
shown), and its usefulness of each query is measured under both conditions (usefulness_choice.py).
  choice shift   Delta logit P(choose q1) = logit P_T3 - logit P_T0, with 0.5 added to the counts (Haldane-Anscombe)
  usefulness     Gamma = [U(q1,T3) - U(q2,T3)] - [U(q1,T0) - U(q2,T0)], measured (primary) or predicted by the locked
                 primary model of C-c (secondary)
Test (H4, one-sided): the slope of the choice shift on Gamma is positive. OLS over pairs, 95% interval and p from a
cluster bootstrap over families. Gamma is measured with noise, which pulls the slope toward 0; its split-half
reliability and the slope corrected for it are reported next to the test.

  python experiments/study_c/choice_shift.py --cap 15000 --split c_confirmation           # confirmatory (locked pairs)
  python experiments/study_c/choice_shift.py --split support_calibration choice_calibration   # exploratory look
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/study_c"))
import prediction_test as P  # noqa: E402

OUT = ROOT / "results/study_c"


def logit(k, n):
    p = (k + 0.5) / (n + 1.0)
    return float(np.log(p / (1 - p)))


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0]) if np.var(x) > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--cap", type=int, default=P.BASE_CAP)
    ap.add_argument("--split", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    halves = {"a": {}, "b": {}}
    predicted = {}
    if args.split == [P.CONF_SPLIT] and P.LOCK.exists():
        lock = json.loads(P.LOCK.read_text())
        predicted = {p["pair_id"]: p["gamma_hat"][lock["primary"]] for p in lock["predictions"]}
    for split in args.split:
        b0, c0, pairs = P.records(args.model, "T0", split, args.cap)
        b3, c3, _ = P.records(args.model, "T3", split, args.cap)
        groups = P.group({"T0": b0, "T3": b3})
        gam = P.observed_gamma(groups, pairs)
        samples = sorted({r["sample"] for r in b0})
        half = {"a": set(samples[: len(samples) // 2]), "b": set(samples[len(samples) // 2:])}
        for h, keep in half.items():  # split-half Gamma for the reliability of the measure
            sub = P.group({"T0": [r for r in b0 if r["sample"] in keep], "T3": [r for r in b3 if r["sample"] in keep]})
            halves[h].update({f"{split}:{k}": v[0] for k, v in P.observed_gamma(sub, pairs).items()})
        for pid, p in pairs.items():
            ch = {c: [r for r in rows_c if r["pair_id"] == pid and r["parsed"]] for c, rows_c in (("T0", c0), ("T3", c3))}
            if pid not in gam or not ch["T0"] or not ch["T3"]:
                continue
            k = {c: sum(r["chose"] == "q_simple" for r in ch[c]) for c in ch}
            rows.append({"pair": f"{split}:{pid}", "family": p["family_id"], "gamma_obs": gam[pid][0],
                         "gamma_pred": predicted.get(pid), "n_T0": len(ch["T0"]), "n_T3": len(ch["T3"]),
                         "p_q1_T0": k["T0"] / len(ch["T0"]), "p_q1_T3": k["T3"] / len(ch["T3"]),
                         "shift": logit(k["T3"], len(ch["T3"])) - logit(k["T0"], len(ch["T0"]))})

    fams = sorted({r["family"] for r in rows})
    fam_of = np.array([fams.index(r["family"]) for r in rows])
    x, y = np.array([r["gamma_obs"] for r in rows]), np.array([r["shift"] for r in rows])
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(len(fams), size=(args.n_boot, len(fams)))
    members = [np.flatnonzero(fam_of == i) for i in range(len(fams))]

    def boot(xv):
        out = []
        for b in idx:
            sel = np.concatenate([members[i] for i in b])
            out.append(slope(xv[sel], y[sel]))
        out = np.array(out)
        return out[np.isfinite(out)]

    bo = boot(x)
    common = sorted(set(halves["a"]) & set(halves["b"]))
    r = float(np.corrcoef([halves["a"][k] for k in common], [halves["b"][k] for k in common])[0, 1]) if len(common) > 2 else float("nan")
    rel = 2 * r / (1 + r) if np.isfinite(r) and r > 0 else float("nan")
    res = {
        "model": args.model, "cap": args.cap, "splits": args.split, "pairs": len(rows), "families": len(fams),
        "choices_parsed": {"T0": int(sum(r["n_T0"] for r in rows)), "T3": int(sum(r["n_T3"] for r in rows))},
        "mean_p_choose_q1": {"T0": float(np.mean([r["p_q1_T0"] for r in rows])), "T3": float(np.mean([r["p_q1_T3"] for r in rows]))},
        "mean_choice_shift_logit": float(y.mean()),
        "primary_measured_gamma": {"slope": slope(x, y), "ci95": [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                                   "p_one_sided": float(np.mean(bo <= 0)), "spearman": float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])},
        "gamma_reliability": {"split_half_r": r, "spearman_brown": rel,
                              "slope_corrected_for_reliability": slope(x, y) / rel if np.isfinite(rel) and rel > 0 else None},
    }
    if predicted:
        xp = np.array([r["gamma_pred"] for r in rows], float)
        bp = boot(xp)
        res["secondary_predicted_gamma"] = {"slope": slope(xp, y), "ci95": [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))],
                                            "p_one_sided": float(np.mean(bp <= 0))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cb_choice_shift_{'+'.join(args.split)}{'' if args.cap == P.BASE_CAP else f'_cap{args.cap}'}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
