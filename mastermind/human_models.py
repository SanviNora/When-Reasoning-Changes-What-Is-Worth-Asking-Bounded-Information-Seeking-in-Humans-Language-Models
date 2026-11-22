"""The three models of Cheyette et al. (2025, Methods and Eqs. 4, 6-8), fitted to free-play guesses.

  unbounded  exact belief updating; noise only in the choice           params: lam, tau
  leaky      belief updating with a Bernoulli likelihood, eps noise    params: eps, lam, tau
  bounded    leaky model whose likelihood for each piece of feedback is
             the alpha-weighted mixture of cached approximations       params: eps, alpha, lam, tau
             (beta = 1, not fitted)

Utility of query q given belief P over codes:
  U(q) = lam * P(q) + sum_o P(o | q) * KL[ P*(H | o) || P(H) ]      (bits)
where P(o | q) uses the true feedback function and P* is the model's updated belief.
Choice: P(choose q) proportional to exp(U(q) / tau), over all 256 codes.
The likelihood of feedback under a representation pi(h) in [0, 1] is (1 - eps) * pi(h) + eps * (1 - pi(h)),
so the bounded model reduces to the leaky model when every approximation is exact.
"""
import numpy as np
from scipy.special import logsumexp, softmax

from mastermind.engine import get_game

N_OUT = 25  # outcome codes black * 5 + white for 4 positions

MODELS = {
    "unbounded": ["lam", "tau"],
    "leaky": ["eps", "lam", "tau"],
    "bounded": ["eps", "alpha", "lam", "tau"],
}


class Task:
    """Feedback tables shared by all models."""

    def __init__(self):
        self.mm = get_game(4, 4)
        F = self.mm.table.astype(np.int64)  # F[q, h]
        self.onehot = np.zeros((self.mm.N, N_OUT, self.mm.N), bool)
        self.onehot[np.arange(self.mm.N)[:, None], F, np.arange(self.mm.N)[None, :]] = True
        self.onehot_flat = self.onehot.reshape(self.mm.N * N_OUT, self.mm.N).astype(np.float64)


class BoundedCache:
    """Rebuilt approximation cache (experiments/study_a/build_cache.py)."""

    def __init__(self, path):
        z = np.load(path)
        self.mm = get_game(4, 4)
        self.off = z["case_offsets"]
        self.ext = z["ext"].astype(np.float64)
        self.logp, self.dist = z["logp"], z["dist"].astype(np.float64)
        self.case_pattern = z["case_pattern"]
        self.case_feedback = z["case_feedback"]
        self.query_pattern, self.query_perm = z["query_pattern"], z["query_perm"]
        self.patterns = ["1111", "1112", "1122", "1123", "1234"]
        self._memo = {}

    def representation(self, alpha):
        """pi[q, o, h]: probability that code h is kept after query q gets outcome o, at cost alpha."""
        key = round(float(alpha), 10)
        if key in self._memo:
            return self._memo[key]
        mm = self.mm
        per_case = np.zeros((len(self.case_pattern), mm.N))
        for c in range(len(self.case_pattern)):
            a, b = self.off[c], self.off[c + 1]
            w = softmax(alpha * self.logp[a:b] - self.dist[a:b])
            per_case[c] = w @ self.ext[a:b]
        index = {(str(p), int(fb[0]), int(fb[1])): c for c, (p, fb) in enumerate(zip(self.case_pattern, self.case_feedback))}
        pi = np.zeros((mm.N, N_OUT, mm.N))
        for q in range(mm.N):
            pat = self.patterns[self.query_pattern[q]]
            T = self.query_perm[q]
            for o in set(mm.table[q].tolist()):
                b, w = mm.decode(o)
                pi[q, o] = per_case[index[(pat, b, w)]][T]
        if len(self._memo) > 64:
            self._memo.clear()
        self._memo[key] = pi
        return pi


def representation(task, model, p, cache=None):
    if model == "bounded":
        return cache.representation(p["alpha"])
    return task.onehot.astype(np.float64)


def _tables(task, model, p, cache):
    """Likelihood L[q, o, h] and L log2 L, flattened to (q*o, h) for matrix-vector products."""
    rep = representation(task, model, p, cache)
    eps = p.get("eps", 0.0)
    lik = (1 - eps) * rep + eps * (1 - rep)
    with np.errstate(divide="ignore", invalid="ignore"):
        liklog = np.where(lik > 0, lik * np.log2(lik), 0.0)
    n = task.mm.N
    return lik.reshape(n * N_OUT, n), liklog.reshape(n * N_OUT, n)


def game_loglik(task, model, p, game, cache=None, tables=None):
    """Sum of log P(choose q_t) over a game [(q, (black, white)), ...].

    With posterior P*(h | o) = P(h) L(o | h) / z(o) and z(o) = sum_h P(h) L(o | h),
      KL[P* || P] = (sum_h P(h) L log2 L) / z  -  log2 z,
    so each decision needs only three matrix-vector products."""
    mm = task.mm
    n = mm.N
    lik, liklog = tables if tables is not None else _tables(task, model, p, cache)
    onehot = task.onehot_flat
    belief = np.full(n, 1.0 / n)
    total = 0.0
    for q_t, (b, w) in game:
        o_t = mm.encode(b, w)
        p_out = (onehot @ belief).reshape(n, N_OUT)
        z = (lik @ belief).reshape(n, N_OUT)
        a = (liklog @ belief).reshape(n, N_OUT)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = np.where(z > 0, a / z - np.log2(z), 0.0)
        utility = p["lam"] * belief + (p_out * kl).sum(-1)
        logits = utility / p["tau"]
        total += logits[q_t] - logsumexp(logits)
        belief = belief * lik[q_t * N_OUT + o_t]
        belief /= belief.sum()
    return float(total)


def loglik(task, model, p, games, cache=None):
    tables = _tables(task, model, p, cache)
    return sum(game_loglik(task, model, p, g, cache, tables) for g in games)


def simulate_game(task, model, p, secret, rng, cache=None, tables=None, max_guesses=15):
    """Play one game by sampling each guess from the model's choice rule (for recovery checks)."""
    mm = task.mm
    n = mm.N
    lik, liklog = tables if tables is not None else _tables(task, model, p, cache)
    onehot = task.onehot_flat
    belief = np.full(n, 1.0 / n)
    game = []
    for _ in range(max_guesses):
        p_out = (onehot @ belief).reshape(n, N_OUT)
        z = (lik @ belief).reshape(n, N_OUT)
        a = (liklog @ belief).reshape(n, N_OUT)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = np.where(z > 0, a / z - np.log2(z), 0.0)
        logits = (p["lam"] * belief + (p_out * kl).sum(-1)) / p["tau"]
        q = int(rng.choice(n, p=softmax(logits)))
        b, w = mm.decode(mm.table[q, secret])
        game.append((q, (b, w)))
        if b == mm.P:
            break
        belief = belief * lik[q * N_OUT + mm.encode(b, w)]
        belief /= belief.sum()
    return game


# ---- fitting (paper: greedy stochastic search) ------------------------------
BOUNDS = {"lam": (1e-3, 1e3), "tau": (1e-3, 1e2), "alpha": (1e-4, 100.0), "eps": (1e-6, 0.5)}


def _propose(name, value, rng, scale=0.1):
    if name == "eps":
        x = np.log(value / (1 - value)) + rng.normal(0, scale)
        v = 1 / (1 + np.exp(-x))
    else:
        v = float(np.exp(np.log(value) + rng.normal(0, scale)))
    lo, hi = BOUNDS[name]
    return float(np.clip(v, lo, hi))


def fit(task, model, games, cache=None, starts=None, seed=0, patience=100, min_gain=1.0, max_evals=3000):
    """Maximum-likelihood parameters by the paper's greedy stochastic search, from several starts.
    Stops a run when the log-likelihood improves by less than `min_gain` over `patience` proposals."""
    rng = np.random.default_rng(seed)
    names = MODELS[model]
    starts = starts or [dict(zip(names, s)) for s in _default_starts(model)]
    best_p, best_ll, evals = None, -np.inf, 0
    for start in starts:
        p = {k: start[k] for k in names}
        ll = loglik(task, model, p, games, cache)
        evals += 1
        anchor = ll
        since = 0
        while evals < max_evals:
            name = names[rng.integers(len(names))]
            cand = dict(p)
            cand[name] = _propose(name, p[name], rng)
            cl = loglik(task, model, cand, games, cache)
            evals += 1
            since += 1
            if cl > ll:
                p, ll = cand, cl
            if since >= patience:
                if ll - anchor < min_gain:
                    break
                anchor, since = ll, 0
        if ll > best_ll:
            best_p, best_ll = p, ll
    k = len(names)
    return {"model": model, "params": best_p, "loglik": best_ll, "k": k, "aic": 2 * k - 2 * best_ll, "evals": evals}


def _default_starts(model):
    if model == "unbounded":
        return [(1.0, 0.5), (5.0, 0.2), (0.3, 1.0)]
    if model == "leaky":
        return [(0.01, 1.0, 0.5), (0.1, 5.0, 0.2), (0.001, 0.3, 1.0)]
    return [(0.01, a, 1.0, 0.5) for a in (0.01, 0.3, 3.0)] + [(0.1, 1.0, 5.0, 0.2)]


# ---- fitting, converged version ----------------------------------------------
# The greedy search above often stops before the maximum (9 of 39 leaky fits had a lower
# log-likelihood than the unbounded fit nested in them). The converged fits are the primary Study A and
# B1b results; the greedy-search fits are kept as the reproduction of the paper's procedure.
# Rules, fixed before any converged fit was run:
#   1. Same models, likelihood, choice rule and parameter limits (BOUNDS). WIDE_BOUNDS (tau down to 1e-6,
#      lam up to 1e5) is a sensitivity check only.
#   2. Search on log lam, log tau, log alpha and logit eps, with the limits enforced on that scale.
#   3. Starts: the greedy-search fit of the same model to the same data; for the unbounded model also
#      the greedy search's default starts; the nested start (leaky: the unbounded optimum with eps at its
#      lower limit; bounded: the leaky optimum with alpha 0.001, 0.1 and 1); one start on the weak-evidence
#      ridge (eps 0.45, lam 0.3, tau 0.002, alpha 1); any extra starts the caller passes (for example the
#      primary fit, when refitting with wider limits).
#   4. From each start: bounded Nelder-Mead (initial step 0.5 on the search scale), rerun from its own
#      result until a rerun gains less than 0.01 log-likelihood (at most 10 runs). The best start wins.
#   5. Check: the leaky fit is at least the unbounded fit minus 0.01 (the nested start ensures it);
#      otherwise the fit stops with an error.
#   6. A parameter within 1% of the limit range (on the search scale) of a limit is listed in "at_limit".
WIDE_BOUNDS = {**BOUNDS, "lam": (1e-3, 1e5), "tau": (1e-6, 1e2)}
NEST_TOL = 0.01
RIDGE_START = {"eps": 0.45, "alpha": 1.0, "lam": 0.3, "tau": 2e-3}
NESTED = {"leaky": "unbounded", "bounded": "leaky"}


def _to_x(name, v):
    return float(np.log(v / (1 - v))) if name == "eps" else float(np.log(v))


def _from_x(name, x):
    return float(1 / (1 + np.exp(-x))) if name == "eps" else float(np.exp(x))


def _polish(task, model, games, cache, start, limits, step=0.5, tol=0.01, max_runs=10):
    """Bounded Nelder-Mead on the search scale, rerun from its own result until a rerun gains < tol."""
    from scipy.optimize import minimize

    names = MODELS[model]
    lo = np.array([_to_x(k, limits[k][0]) for k in names])
    hi = np.array([_to_x(k, limits[k][1]) for k in names])
    evals = 0

    def nll(x):
        nonlocal evals
        evals += 1
        v = -loglik(task, model, {k: _from_x(k, xi) for k, xi in zip(names, x)}, games, cache)
        return v if np.isfinite(v) else 1e12

    x = np.clip([_to_x(k, start[k]) for k in names], lo, hi)
    best = nll(x)
    runs = 0
    while runs < max_runs:
        runs += 1
        simplex = [x.copy()]
        for i in range(len(names)):
            v = x.copy()
            v[i] = v[i] + step if v[i] + step <= hi[i] else v[i] - step
            simplex.append(v)
        r = minimize(nll, x, method="Nelder-Mead", bounds=list(zip(lo, hi)),
                     options={"initial_simplex": np.array(simplex), "xatol": 1e-4, "fatol": 1e-4, "maxfev": 4000})
        gain = best - r.fun
        if r.fun < best:
            x, best = np.clip(r.x, lo, hi), float(r.fun)
        if gain < tol:
            break
    params = {k: _from_x(k, xi) for k, xi in zip(names, x)}
    pos = (x - lo) / (hi - lo)
    at_limit = [k for k, f in zip(names, pos) if f < 0.01 or f > 0.99]
    return params, -best, evals, runs, at_limit


def fit_converged(task, model, games, cache=None, paper=None, nested=None, bounds=None, extra_starts=(), seed=0):
    """Maximum-likelihood fit by the rules above.
    paper: the greedy-search fit of this model to the same data (run here when not given);
    nested: the converged parameters of the next-simpler model (unbounded for leaky, leaky for bounded);
    extra_starts: further parameter dicts to start from."""
    limits = bounds or BOUNDS
    names = MODELS[model]
    if paper is None:
        paper = fit(task, model, games, cache=cache, seed=seed)
    starts = [("greedy_search", paper["params"])]
    if model == "unbounded":
        starts += [("default", dict(zip(names, s))) for s in _default_starts(model)]
    if nested is not None and model == "leaky":
        starts.append(("nested", {**nested, "eps": limits["eps"][0]}))
    if nested is not None and model == "bounded":
        starts += [("nested", {**nested, "alpha": a}) for a in (1e-3, 0.1, 1.0)]
    if model != "unbounded":
        starts.append(("ridge", {k: RIDGE_START[k] for k in names}))
    starts += [("extra", s) for s in extra_starts]
    tried, best, evals = [], None, 0
    for label, s in starts:
        p, ll, ev, runs, at_limit = _polish(task, model, games, cache, s, limits)
        evals += ev
        tried.append({"start": label, "loglik": ll, "runs": runs})
        if best is None or ll > best[1]:
            best = (p, ll, at_limit)
    p, ll, at_limit = best
    k = len(names)
    return {"model": model, "procedure": "converged", "params": p, "loglik": ll, "k": k, "aic": 2 * k - 2 * ll,
            "evals": evals, "at_limit": at_limit, "limits": {n: list(limits[n]) for n in names},
            "greedy_search_loglik": paper["loglik"], "starts": tried}


def fit_converged_models(task, models, games, cache=None, paper=None, bounds=None, extra=None, seed=0):
    """Converged fits of `models` in nesting order, each using the previous model's optimum as a start.
    paper / extra: {model: greedy-search fit} / {model: [parameter dicts]} for the same data."""
    paper, extra, out = paper or {}, extra or {}, {}
    for m in [m for m in MODELS if m in models]:
        prev = NESTED.get(m)
        out[m] = fit_converged(task, m, games, cache, paper.get(m), out[prev]["params"] if prev in out else None,
                               bounds, extra.get(m, ()), seed)
    if "unbounded" in out and "leaky" in out and out["leaky"]["loglik"] < out["unbounded"]["loglik"] - NEST_TOL:
        raise RuntimeError(f"leaky fit ({out['leaky']['loglik']:.3f}) below the nested unbounded fit "
                           f"({out['unbounded']['loglik']:.3f})")
    return out
