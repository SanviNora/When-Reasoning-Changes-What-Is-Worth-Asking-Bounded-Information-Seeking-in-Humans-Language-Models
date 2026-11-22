"""Holm correction across the course's primary tests: B1, B3 and H4.

The project plan names three primary tests for the course and a Holm correction across them
(restated in prereg/analysis_plan.md, Tests). H3 (C-c) is the paper's
primary test and is not in this family: it has its own preregistered Holm correction across its four rivals
(experiments/study_c/prediction_test.py score). It is listed in the output next to the family.

The family has 5 p-values, Holm at alpha 0.05:
  B1  Max-EIG minus Simple within each model that passes the solve-rate rule (gpt-oss-120b, Claude Haiku 5.5,
      Claude Opus 5.5): mixed model with a random intercept per stimulus, two-sided p, as reported in Study B.
      The alternative includes the ideal player's direction.
  B3  held-out comparison (H2: structured accounts beat the unstructured rivals). The best planned account
      against M0 (exact inference + reporting noise) and against M1 (noisy evidence use). The p-value is
      one-sided, from the family bootstrap, and is the larger of the two, since H2 needs the account to beat both.
  H4  slope of the choice shift on measured Gamma: one-sided cluster-bootstrap p (choice_shift.py).
The B3 p-values use the same 2,000 family resamples and seed as the intervals in qwen3-32b_T0_intervals.json. This
script recomputes those intervals and checks that they match.

  python experiments/primary_tests.py      ->  results/primary_tests.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/study_b"))
import fit_process_models as F  # noqa: E402

ALPHA = 0.05
OUT = ROOT / "results/primary_tests.json"


def b1_tests():
    s = json.loads((ROOT / "results/study_b/b1_summary.json").read_text())
    tests = []
    for m, d in s.items():
        if not isinstance(d, dict) or not d.get("eligible_for_inference") or "mixed_model" not in d:
            continue
        mx = d["mixed_model"]
        tests.append({"test": f"B1 {m}", "claim": "slower after Max-EIG than after Simple histories (the human signature)",
                      "estimate": mx["maxeig_minus_simple"], "ci95": mx["ci95"], "p": mx["p"], "sided": "two",
                      "source": "results/study_b/b1_summary.json (mixed_model)"})
    return tests


def heldout_by_family(model="qwen3-32b", condition="T0"):
    """B3 held-out log likelihood per (confirmation family, model) at the saved parameters, and the family
    resamples of the intervals file (same seed). Returns (models, ll [family, model], boots, intervals file)."""
    iv = json.loads((F.OUT / f"{model}_{condition}_intervals.json").read_text())
    saved = json.loads((F.OUT / f"{model}_{condition}.json").read_text())
    mm = F.get_game(4, 4)
    caches = {k: F.hm.BoundedCache(v) for k, v in F.CACHES.items()}
    items, _ = F.to_items(mm, [r for r in F.load_records(model, condition) if r["split"] == "confirmation"])
    fams = sorted({f for *_, f in items})
    models = list(F.pm.MODELS)
    index = {f: i for i, f in enumerate(fams)}
    ll = np.zeros((len(fams), len(models)))
    for _, _, it, fam in items:
        for j, m in enumerate(models):
            ll[index[fam], j] += F.pm.item_loglik(m, F.planned_fit(saved, m)["params"], it, caches)
    rng = np.random.default_rng(iv["bootstrap"]["seed"])  # the same resamples as the intervals file
    boots = rng.integers(len(fams), size=(iv["bootstrap"]["resamples"], len(fams)))
    return models, ll, boots, iv


def b3_test(model="qwen3-32b", condition="T0", rivals=("M0", "M1")):
    models, ll, boots, iv = heldout_by_family(model, condition)
    best = iv["best_planned_heldout"]
    comp = {}
    for r in rivals:
        d = ll[:, models.index(best)] - ll[:, models.index(r)]
        sums = d[boots].sum(1)
        ci = np.percentile(sums, [2.5, 97.5]).round(2).tolist()
        saved_ci = iv["best_planned_minus_other_planned"][r]["ci95"]
        assert ci == saved_ci and round(float(d.sum()), 2) == iv["best_planned_minus_other_planned"][r]["difference"], \
            f"{best} - {r}: recomputed {ci}, saved {saved_ci}"
        comp[r] = {"difference": round(float(d.sum()), 2), "ci95": ci, "p_one_sided": float(np.mean(sums <= 0))}
    worst = max(comp, key=lambda r: comp[r]["p_one_sided"])
    return {"test": "B3", "claim": f"the best planned structured account ({best}) beats both unstructured rivals "
                                   f"({', '.join(rivals)}) in held-out log likelihood",
            "estimate": comp[worst]["difference"], "ci95": comp[worst]["ci95"], "p": comp[worst]["p_one_sided"],
            "sided": "one", "comparisons": comp, "binding_rival": worst,
            "source": f"results/study_b/process_fits/{model}_{condition}_intervals.json (recomputed resamples)"}


def h4_test():
    c = json.loads((ROOT / "results/study_c/cb_choice_shift_c_confirmation.json").read_text())
    p = c["primary_measured_gamma"]
    return {"test": "H4", "claim": "the choice shift rises with the measured change in usefulness (slope > 0)",
            "estimate": round(p["slope"], 3), "ci95": [round(x, 3) for x in p["ci95"]], "p": p["p_one_sided"],
            "sided": "one", "source": "results/study_c/cb_choice_shift_c_confirmation.json"}


def holm(tests):
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    running, m = 0.0, len(tests)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * tests[i]["p"]))
        tests[i]["p_holm"] = running
        tests[i]["rejected"] = running < ALPHA
    return tests


def main():
    tests = holm(b1_tests() + [b3_test(), h4_test()])
    sc = json.loads((ROOT / "results/study_c/cc_score.json").read_text())
    h3 = {"test": "H3", "claim": "support_map has lower family-level MSPE of Gamma than every rival",
          "p_holm_by_rival": {k: v["p_holm"] for k, v in sc["primary_vs_rivals"].items()},
          "supported": all(v["p_holm"] < ALPHA for v in sc["primary_vs_rivals"].values()),
          "note": "not in the family: preregistered with its own Holm correction across the four rivals",
          "source": "results/study_c/cc_score.json"}
    res = {"alpha": ALPHA, "method": "Holm step-down across the course's primary tests (B1 per eligible model, B3, H4)",
           "family": tests, "outside_family": [h3]}
    OUT.write_text(json.dumps(res, indent=2))
    for t in tests:
        print(f"{t['test']:22s} est {t['estimate']:>8} p {t['p']:.3g}  Holm {t['p_holm']:.3g}  "
              f"{'rejected' if t['rejected'] else 'not rejected'}")
    print(f"H3 supported: {h3['supported']}")


if __name__ == "__main__":
    main()
