"""Predicted and observed usefulness of a query (native question choice in Study B; the aid in Study C).

A query q on family h splits the codes still possible, S(h), into branches S_a. Its usefulness to a
receiver is U(q) = (1/K) sum_a P(d in S_a | h + (q, a)), the chance of naming the secret after
seeing q's answer.

Predicted U under a process model with the support map:
  l'_i(s) = (1 - delta) l_i(s) + delta c_i(s)     (delta = 0 in the baseline condition)
  P(d | clues) = (1 - zeta) b(d) + zeta / 256,     b proportional to prod_i l'_i(d)
The support map moves every clue weight toward the exact constraint (delta > 0) or away from it (delta < 0;
weights are kept in [0, 1]), so the aid changes usefulness most where the base model distorts most, which
differs across queries (this is what makes Gamma selective).
"""
from itertools import product

import numpy as np
from scipy.special import expit

from mastermind import process_models as pm


def identification_probs(model, p, clues, c, caches, delta=0.0):
    """P(d) over all codes for a history given as clue rows c[i, s] and (q, o) pairs."""
    n = c.shape[1]
    if model == "M3":
        k = c.shape[0]
        r = np.clip((1 - delta) * expit(p["k0"] + p["k1"] * np.arange(k)[::-1]) + delta, 0.0, 1.0)
        b = np.zeros(n)
        for kept in product([0, 1], repeat=k):
            kept = np.array(kept, bool)
            w = np.prod(np.where(kept, r, 1 - r))
            g = np.prod(c[kept], axis=0) if kept.any() else np.ones(n)
            b += w * g / g.sum()
    else:
        item = pm.Item.__new__(pm.Item)
        item.clues, item.c = clues, c
        l_ = pm.clue_weights(model, p, item, caches)
        l_ = np.clip((1 - delta) * l_ + delta * c, 0.0, 1.0)  # delta < 0 moves weights away from the exact clue
        g = np.prod(l_, axis=0)
        b = g / g.sum()
    return (1 - p["zeta"]) * b + p["zeta"] / n


def predicted_usefulness(mm, model, p, history, q, caches, delta=0.0):
    hist_clues = [(h, mm.encode(b, w)) for h, (b, w) in history]
    mask = mm.consistent(history)
    K = mask.sum()
    total = 0.0
    for o in np.unique(mm.table[q, mask]):
        clues = hist_clues + [(q, int(o))]
        c = np.stack([mm.table[h] == oo for h, oo in clues]).astype(np.float64)
        pd_ = identification_probs(model, p, clues, c, caches, delta)
        branch = mask & (mm.table[q] == o)
        total += pd_[branch].sum()
    return float(total / K)


def observed_usefulness(rows):
    """Exact branch-sum estimate from usefulness runs: U = sum_a (K_a / K) mean(Y_a) = (1/K) sum_a P(d in S_a).
    rows: branch records for one (pair, query, condition) with fields outcome, K_a, Y."""
    by_branch = {}
    for r in rows:
        by_branch.setdefault(tuple(r["outcome"]), []).append(r)
    K = sum(v[0]["K_a"] for v in by_branch.values())
    return float(sum(v[0]["K_a"] * np.mean([x["Y"] for x in v]) for v in by_branch.values()) / K)
