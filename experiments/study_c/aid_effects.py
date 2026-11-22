"""C-a: how the aid changes the way the depth model uses evidence.

The probe battery (classify, single-clue classify, identify) ran under T0 (standard), T1 (length-matched control),
T2 (generic "check your work"; an extra control, not in the proposal) and T3 (the structured aid) on the same
processing-calibration and confirmation families. Error measures are those of the Study B error profile:
  false exclusion   a still-possible code called ruled out (multi-clue and single-clue probes)
  false retention   a ruled-out code called possible (near misses: ruled out only by the newest clue; far misses)
  identification    a named code is still possible (parsed replies); unfinished = no parseable code
Contrasts: T3 - T0 and T3 - T1 (the proposal's comparisons), T3 - T2 (structure beyond generic guidance), with 95%
intervals from 10,000 paired family-bootstrap resamples (the same families in every condition).
Parameter shifts: the process models fitted per condition (fit_process_models.py fit --condition c); intervals
from the `boot` resamples, which use the same family draws in every condition (same seeds), so they are paired.

  python experiments/study_c/aid_effects.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/study_b"))
import fit_process_models as F  # noqa: E402

MODEL = "qwen3-32b"
CONDITIONS = ("T0", "T1", "T2", "T3")
CONTRASTS = (("T3", "T0"), ("T3", "T1"), ("T3", "T2"))
MEASURES = ("false_exclusion_multi", "false_exclusion_single", "false_retention_near_multi", "false_retention_far_multi",
            "false_retention_near_single", "identification", "identify_unfinished")
PARAMS = {"M2s": ("nu", "zeta", "alpha"), "M1": ("nu", "zeta"), "M1a": ("nu_x", "nu_r", "zeta")}
OUT = ROOT / "results/study_c"


def family_counts(condition):
    """{family: {measure: [errors, n]}} over the processing-calibration and confirmation families."""
    out = {}
    for r in F.load_records(MODEL, condition):
        if r["split"] not in F.PROFILE_SPLITS:
            continue
        f = out.setdefault(r["family_id"], {m: [0, 0] for m in MEASURES})
        if r["task"] == "identify":
            f["identify_unfinished"][0] += not r.get("parsed")
            f["identify_unfinished"][1] += 1
            if r.get("parsed"):
                f["identification"][0] += bool(r["correct"])
                f["identification"][1] += 1
            continue
        if not r.get("labels"):
            continue
        multi = r["task"] == "classify"
        for code, kind in r["items"]:
            said = r["labels"].get(code)
            if said is None:
                continue
            if kind == "possible":
                m = "false_exclusion_multi" if multi else "false_exclusion_single"
                err = not said
            elif kind == "near":
                m = "false_retention_near_multi" if multi else "false_retention_near_single"
                err = said
            elif multi:
                m, err = "false_retention_far_multi", said
            else:
                continue
            f[m][0] += err
            f[m][1] += 1
    return out


def main(n_boot=10000, seed=0):
    counts = {c: family_counts(c) for c in CONDITIONS}
    fams = sorted(set.intersection(*(set(v) for v in counts.values())))
    arr = {c: {m: np.array([counts[c][f][m] for f in fams], float) for m in MEASURES} for c in CONDITIONS}  # [fam, 2]
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(fams), size=(n_boot, len(fams)))

    def rate(c, m, sel=None):
        a = arr[c][m] if sel is None else arr[c][m][sel]
        return a[..., 0].sum(-1) / a[..., 1].sum(-1)

    res = {"model": MODEL, "families": len(fams), "rates": {c: {m: round(float(rate(c, m)), 4) for m in MEASURES} for c in CONDITIONS},
           "contrasts": {}}
    for a, b in CONTRASTS:
        res["contrasts"][f"{a}-{b}"] = {}
        for m in MEASURES:
            d = rate(a, m) - rate(b, m)
            boots = rate(a, m, idx) - rate(b, m, idx)
            res["contrasts"][f"{a}-{b}"][m] = {"difference": round(float(d), 4),
                                               "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}

    fits = {c: json.loads((F.OUT / f"{MODEL}_{c}.json").read_text()) for c in CONDITIONS}
    res["parameters"] = {c: {m: {k: round(fits[c]["fits"][m]["params"][k], 4) for k in ks} for m, ks in PARAMS.items()} for c in CONDITIONS}
    res["best_model"] = {c: {"aic": fits[c]["best_aic"], "heldout_planned": fits[c]["best_heldout"]} for c in CONDITIONS}
    boots = {}
    for c in CONDITIONS:
        d = F.OUT / f"boot_{MODEL}_{c}"
        boots[c] = {int(p.stem.split("_")[1]): json.loads(p.read_text()) for p in sorted(d.glob("rep_*.json"))} if d.exists() else {}
    res["parameter_shifts"] = {}
    for a, b in CONTRASTS:
        reps = sorted(set(boots[a]) & set(boots[b]))
        if not reps:
            res["parameter_shifts"][f"{a}-{b}"] = "needs the boot resamples of both conditions"
            continue
        assert all(boots[a][r]["families"] == boots[b][r]["families"] for r in reps), "resamples are not paired"
        shift = {}
        for m, ks in PARAMS.items():
            for k in ks:
                dv = np.array([boots[a][r]["fits"][m]["params"][k] - boots[b][r]["fits"][m]["params"][k] for r in reps])
                shift[f"{m}.{k}"] = {"difference": round(float(fits[a]["fits"][m]["params"][k] - fits[b]["fits"][m]["params"][k]), 4),
                                     "ci95": [round(float(np.percentile(dv, 2.5)), 4), round(float(np.percentile(dv, 97.5)), 4)]}
        res["parameter_shifts"][f"{a}-{b}"] = {"resamples": len(reps), **shift}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ca_aid_effects.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
