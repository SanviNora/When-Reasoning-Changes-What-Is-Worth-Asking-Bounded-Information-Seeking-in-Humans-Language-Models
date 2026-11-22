"""Post hoc: why the confirmation's Gamma was noisier than the power analysis assumed (not part of the locked analysis).

Gamma's sampling noise comes from the branch samples. U = (1/K) sum_a p_a, where p_a is the share of a branch's
samples that name a code in that branch. So Var(U) = (1/K^2) sum_a p_a (1 - p_a) / n per query. For the
calibration pairs (4 samples per branch) and the confirmation pairs (8), this script reports:
  - the noise as the locked code estimates it: resampling within branches, a plug-in that is low by (n - 1) / n;
  - the same noise corrected by n / (n - 1);
  - its parts: samples per branch, K, branches per query, mean p_a (1 - p_a);
  - the true SD of Gamma, sqrt(observed variance - noise variance), under both noise estimates.
It then runs the power simulation of `prediction_test.py power` with the confirmation's own values (corrected
noise): the power the realized design had at the planned smallest effect, r2 = 0.20.

  python experiments/study_c/noise_check.py      ->  results/study_c/cc_noise_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/study_c"))
import prediction_test as P  # noqa: E402

SETS = {"calibration": ["support_calibration", "choice_calibration"], "confirmation": ["c_confirmation"]}


def describe(splits):
    rows = {c: [r for s in splits for r in P.records("qwen3-32b", c, s, P.BASE_CAP)[0]] for c in ("T0", "T3")}
    g = P.group(rows)
    gam = P.observed_gamma(g, sorted({k[1] for k in g}), n_noise=300, seed=0)
    obs = np.array([v[0] for v in gam.values()])
    boot = float(np.mean([v[1] for v in gam.values()]))
    n, K, nb, pq = [], [], [], []
    for pid in gam:
        for c in ("T0", "T3"):
            for q in ("q_simple", "q_complex"):
                by, ka = {}, {}
                for r in g[(c, pid, q)]:
                    o = tuple(r["outcome"])
                    ka[o] = r["K_a"]
                    by.setdefault(o, []).append(r["Y"] > 0)
                K.append(sum(ka.values()))
                nb.append(len(by))
                for ys in by.values():
                    n.append(len(ys))
                    pq.append(np.mean(ys) * (1 - np.mean(ys)))
    s = float(np.mean(n))
    corrected = boot * s / (s - 1)
    return {"pairs": len(gam), "samples_per_branch": s, "mean_K": round(float(np.mean(K)), 3),
            "mean_inverse_K_squared": round(float(np.mean(1 / np.square(K))), 5),
            "branches_per_query": round(float(np.mean(nb)), 3), "mean_p_1_minus_p": round(float(np.mean(pq)), 4),
            "observed_sd": round(float(obs.std()), 4), "noise_sd_locked_estimate": round(float(np.sqrt(boot)), 4),
            "noise_sd_corrected": round(float(np.sqrt(corrected)), 4),
            "true_sd_locked_estimate": round(float(np.sqrt(max(obs.var() - boot, 0))), 4),
            "true_sd_corrected": round(float(np.sqrt(max(obs.var() - corrected, 0))), 4)}


def realized_power(tau, sigma, families=120, pairs_per_family=3, r2=0.20, n_sims=2000, alpha=0.05, seed=0):
    """The simulation of prediction_test.power at one design, with given tau and noise SD."""
    rng = np.random.default_rng(seed)
    z = norm.ppf(1 - alpha / 4)
    gt = rng.normal(0, tau, (n_sims, families, pairs_per_family))
    m = r2 * gt + np.sqrt(r2 * (1 - r2)) * tau * rng.normal(size=gt.shape)
    obs = gt + rng.normal(0, sigma, gt.shape)
    d = (obs ** 2 - (m - obs) ** 2).mean(2)
    mean, se = d.mean(1), d.std(1, ddof=1) / np.sqrt(families)
    return round(float(np.mean(mean - z * se > 0)), 3)


def main():
    res = {k: describe(v) for k, v in SETS.items()}
    cal, conf = res["calibration"], res["confirmation"]
    planned = json.loads((ROOT / "results/study_c/cc_power.json").read_text())
    res["planned"] = {"tau_sd": round(planned["tau_sd"], 4),
                      "noise_sd_at_8_samples": round(planned["noise_sd_at_calibration"] * np.sqrt(4 / 8), 4),
                      "power": next(r["power"] for r in planned["design"] if r["samples_per_branch"] == 8 and r["families"] == 120)}
    res["calibration_projected_to_8_samples_corrected"] = round(cal["noise_sd_corrected"] * np.sqrt(4 / 8), 4)
    res["realized_power_at_r2_0.20"] = realized_power(conf["true_sd_corrected"], conf["noise_sd_corrected"])
    (ROOT / "results/study_c/cc_noise_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
