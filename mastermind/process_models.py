"""Process models of how a reasoner uses Mastermind feedback, fitted to the B2 battery.

Each model gives a clue weight l_i(s) for every clue i of a history; compatibility g(s) = prod_i l_i(s)
and belief b(s) proportional to g(s) (uniform prior). One shared observation model:
  classify ("is x still possible?"):  P(yes) = (1 - zeta) g(x) + zeta / 2
  identify (name one code):           P(d)   = (1 - zeta) b(d) + zeta / 256
Unparsed answers are format failures, counted separately and left out of the likelihood.
Decision temperature is fixed at 1 (not identifiable from final codes under a uniform prior).

  M0   exact inference + reporting noise    l = c                                   params: zeta
  M1   noisy evidence use                   l = (1 - nu) c + nu / 2                 params: nu, zeta
  M2c  conservative approximation (human)   l = (1 - nu) pi_c(alpha) + nu / 2       params: alpha, nu, zeta
  M2s  symmetric approximation              l = (1 - nu) pi_s(alpha) + nu / 2       params: alpha, nu, zeta
  M3   evidence retention                   clue i kept with r_i = sigmoid(k0 + k1 * age_i),
                                            age 0 = newest; exact over the 2^k kept subsets   params: k0, k1, zeta
                                            One subset per reply: every code judged in one reply uses the same kept
                                            clues, as in `simulate`.

Exploratory (added after the planned comparison, motivated by the error profile; not part of it):
  M1a  two-rate noisy evidence use          l = (1 - nu_x) c + nu_r (1 - c)                    params: nu_x, nu_r, zeta
       nu_x: a clue wrongly rules out a code it allows; nu_r: a clue wrongly keeps a code it rules out.
       nu_x = nu_r = nu / 2 is M1; nu_x = 0 is the one-sided (conservative) error pattern.
"""
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from mastermind.engine import get_game

PLANNED = ["M0", "M1", "M2c", "M2s", "M3"]
EXPLORATORY = ["M1a"]
MODELS = {"M0": ["zeta"], "M1": ["nu", "zeta"], "M2c": ["alpha", "nu", "zeta"],
          "M2s": ["alpha", "nu", "zeta"], "M3": ["k0", "k1", "zeta"], "M1a": ["nu_x", "nu_r", "zeta"]}
TRANSFORM = {"zeta": "unit", "nu": "unit", "alpha": "log", "k0": "real", "k1": "real", "nu_x": "unit", "nu_r": "unit"}
START = {"zeta": 0.1, "nu": 0.1, "alpha": 0.3, "k0": 2.0, "k1": -0.5, "nu_x": 0.1, "nu_r": 0.1}


class Item:
    """One battery response, pre-processed: clue indicator rows c[i, s] and the observed answer."""

    def __init__(self, mm, history, task, answer):
        self.task = task
        self.clues = [(q, mm.encode(b, w)) for q, (b, w) in history]
        self.c = np.stack([mm.table[q] == o for q, o in self.clues]).astype(np.float64)  # [k, 256]
        self.answer = answer  # classify: list of (code_idx, said_possible); identify: code_idx


def clue_weights(model, p, item, caches):
    """l[i, s] for every clue of the item (M3 handled separately)."""
    if model == "M0":
        return item.c
    if model == "M1":
        return (1 - p["nu"]) * item.c + p["nu"] / 2
    if model == "M1a":
        return (1 - p["nu_x"]) * item.c + p["nu_r"] * (1 - item.c)
    rep = caches["M2c" if model == "M2c" else "M2s"].representation(p["alpha"])
    pi = np.stack([rep[q, o] for q, o in item.clues])
    return (1 - p["nu"]) * pi + p["nu"] / 2


def _subsets(k):
    """Every kept/forgotten pattern of k clues, [2^k, k] booleans, in itertools.product order."""
    return np.array(list(product([0, 1], repeat=k)), bool)


def _m3_table(item):
    """Per kept-clue subset S of one reply: classify -> number of answers that agree with exact inference from
    the clues in S (g_S(x) = 1 said possible, g_S(x) = 0 said ruled out), and the number of answers;
    identify -> b_S(d), the probability of the named code under uniform belief over the codes S allows.
    Cached on the item (the answers never change)."""
    t = getattr(item, "_m3", None)
    if t is None:
        S = _subsets(len(item.clues))
        if item.task == "classify":
            xs = np.array([x for x, _ in item.answer])
            said = np.array([s for _, s in item.answer], bool)
            g = np.array([np.prod(item.c[kept][:, xs], axis=0) if kept.any() else np.ones(len(xs)) for kept in S]) > 0.5
            t = ((g == said[None, :]).sum(1).astype(float), float(len(xs)))
        else:
            gs = [np.prod(item.c[kept], axis=0) if kept.any() else np.ones(item.c.shape[1]) for kept in S]
            t = (np.array([g[item.answer] / g.sum() for g in gs]),)
        item._m3 = t
    return t


def m3_loglik(p, items):
    """M3 log likelihood, exact: one kept-clue subset per reply (the same subset for every code judged in it,
    as in `simulate`), marginalized over the 2^k subsets. A reply's answers are independent given the subset."""
    zeta = p["zeta"]
    groups = {}
    for it in items:
        groups.setdefault((it.task, len(it.clues)), []).append(it)
    total = 0.0
    with np.errstate(divide="ignore"):
        for (task, k), its in groups.items():
            S = _subsets(k)
            r = expit(p["k0"] + p["k1"] * np.arange(k)[::-1])  # newest clue has age 0
            logw = np.where(S, np.log(r), np.log1p(-r)).sum(1)  # [2^k]
            if task == "classify":
                agree = np.array([_m3_table(it)[0] for it in its])  # [n, 2^k]
                n = np.array([_m3_table(it)[1] for it in its])[:, None]
                ll = logw[None, :] + agree * np.log1p(-zeta / 2) + (n - agree) * np.log(zeta / 2)
                m = ll.max(1, keepdims=True)
                total += float((m[:, 0] + np.log(np.exp(ll - m).sum(1))).sum())
            else:
                bS = np.array([_m3_table(it)[0] for it in its])  # [n, 2^k]
                b = bS @ np.exp(logw)
                total += float(np.log((1 - zeta) * b + zeta / its[0].c.shape[1]).sum())
    return total


def item_loglik(model, p, item, caches):
    zeta = p["zeta"]
    if model == "M3":
        return m3_loglik(p, [item])
    else:
        l_ = clue_weights(model, p, item, caches)
        g = np.prod(l_, axis=0)
        if item.task == "identify":
            b = g / g.sum()
    if item.task == "classify":
        ll = 0.0
        for x, said in item.answer:
            py = (1 - zeta) * g[x] + zeta / 2
            ll += np.log(py if said else 1 - py)
        return ll
    return float(np.log((1 - zeta) * b[item.answer] + zeta / item.c.shape[1]))


def loglik(model, p, items, caches=None):
    if model == "M3":
        return m3_loglik(p, items)
    return sum(item_loglik(model, p, it, caches) for it in items)


def _to_vec(model, p):
    out = []
    for k in MODELS[model]:
        t = TRANSFORM[k]
        out.append(logit(p[k]) if t == "unit" else np.log(p[k]) if t == "log" else p[k])
    return np.array(out)


def _from_vec(model, v):
    p = {}
    for k, x in zip(MODELS[model], v):
        t = TRANSFORM[k]
        p[k] = float(expit(x)) if t == "unit" else float(np.exp(np.clip(x, -9.2, 4.6))) if t == "log" else float(x)
    return p


def fit(model, items, caches=None, starts=None):
    """Maximum likelihood with Nelder-Mead on transformed parameters, several starts."""
    if starts is None:
        starts = [START] + [{**START, "alpha": a, "zeta": z} for a in (0.03, 3.0) for z in (0.02, 0.3)]
        if "nu_x" in MODELS[model]:  # one-sided starting points for the two noise rates
            starts += [{**START, "nu_x": 0.3, "nu_r": 0.05}, {**START, "nu_x": 0.05, "nu_r": 0.3}]
    best = None
    for s in starts:
        v0 = _to_vec(model, {k: s[k] for k in MODELS[model]})
        res = minimize(lambda v: -loglik(model, _from_vec(model, v), items, caches), v0, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
        if len(MODELS[model]) == 1:
            break
    k = len(MODELS[model])
    ll = -float(best.fun)
    return {"model": model, "params": _from_vec(model, best.x), "loglik": ll, "k": k, "aic": 2 * k - 2 * ll}


def simulate(model, p, items, caches, rng):
    """Synthetic answers from a model on the same items (for parameter and model recovery)."""
    mm = get_game(4, 4)
    out = []
    for it in items:
        zeta = p["zeta"]
        if model == "M3":
            k = len(it.clues)
            r = expit(p["k0"] + p["k1"] * np.arange(k)[::-1])
            kept = rng.random(k) < r
            g = np.prod(it.c[kept], axis=0) if kept.any() else np.ones(mm.N)
        else:
            g = np.prod(clue_weights(model, p, it, caches), axis=0)
        new = Item.__new__(Item)
        new.task, new.clues, new.c = it.task, it.clues, it.c
        if it.task == "classify":
            new.answer = [(x, bool(rng.random() < (1 - zeta) * g[x] + zeta / 2)) for x, _ in it.answer]
        else:
            probs = (1 - zeta) * g / g.sum() + zeta / mm.N
            new.answer = int(rng.choice(mm.N, p=probs / probs.sum()))
        out.append(new)
    return out
