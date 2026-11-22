"""Rebuild the bounded-representation model's approximation cache.

The released code has the grammar, the prior, the proposal moves and the likelihood, but not the
cache of approximations used to fit the model. Following the manuscript's Methods:
  for each (query, feedback) pair, run 4 chains of 100,000 iterations of a Metropolis-Hastings search
  with tree-regeneration proposals from the grammar prior; keep every sampled approximation; then,
  at each alpha on a log grid from 1e-4 to 100, keep the top 10 approximations (ties included) and
  add the exact expression for the feedback.
An approximation pi is scored at cost alpha by  alpha * log p_G(pi) - D(pi), where D counts codes pi
keeps that the feedback rules out and is infinite if pi rules out a code the feedback allows
(the conservative distance, Eq. 5). This is the same score the released model.py uses.

Symmetry: the grammar treats digits and positions alike, so the 256 queries reduce to 5 patterns
(1111, 1112, 1122, 1123, 1234) and 55 (pattern, feedback) cases. Other queries reuse a case through
the digit/position permutation that maps them onto its pattern.

Search details not given in the paper (our choices, recorded in the manifest): chain alphas,
starting points, and the proposal correction (exponent alpha - 1 on the prior ratio).

  python experiments/study_a/build_cache.py search --case 0     # one case, 4 chains (parallel)
  python experiments/study_a/build_cache.py aggregate           # all cases -> data/model_cache/
  add --rep 2 to both for an independent replicate (new seeds, separate search folder and cache file),
  used to check that the fits do not depend on one run of the random search
"""
import argparse
import hashlib
import json
import math
import pickle
import random
import sys
import time
from itertools import permutations
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUTHORS = ROOT / "data/human/cheyette2025_osf_vjrqu/Models/Mastermind"
sys.path.insert(0, str(ROOT))
from mastermind.engine import get_game  # noqa: E402

OUT = ROOT / "data/model_cache"
SEARCH = OUT / "search"
SEARCH_SYM = OUT / "search_symmetric"  # M2s: approximations may also rule out allowed codes
PATTERNS = ["1111", "1112", "1122", "1123", "1234"]
ALPHA_GRID = [round(10 ** e, 6) for e in np.arange(-4, 2.0001, 0.25)]
CHAIN_ALPHAS = [0.001, 0.1, 1.0, 10.0]
TOP_K = 10
REP_SEED_OFFSET = 100_000  # replicate r uses seeds 1000 * case + chain + (r - 1) * REP_SEED_OFFSET


def search_dir(variant, rep=1):
    base = SEARCH if variant == "conservative" else SEARCH_SYM
    return base if rep == 1 else base.with_name(f"{base.name}_rep{rep}")


def cache_name(variant, rep=1):
    name = "bounded_cache_v1" if variant == "conservative" else "symmetric_cache_v1"
    return name if rep == 1 else f"{name}_rep{rep}"


def cases(mm):
    out = []
    for pat in PATTERNS:
        q = mm.index(pat)
        for o in sorted(set(mm.table[q].tolist())):
            out.append((pat, mm.decode(o)))
    return out  # (pattern, (black, white))


# ---- search (runs inside the authors' module namespace) ------------------
def _authors():
    sys.path.insert(0, str(AUTHORS))
    import grammar as G  # noqa: E402  (authors' module; star-imports primitives and utils)
    return G


def _evaluate(expr, codes, truth, conservative=True):
    """Extension of expr over all codes. In the conservative variant, None if it rules out an allowed
    code (D infinite); allowed codes are checked first so most failing proposals exit early."""
    ext = np.zeros(len(codes), bool)
    try:
        for i in np.flatnonzero(truth):
            ext[i] = bool(expr.evaluate([codes[i]]))
            if conservative and not ext[i]:
                return None
        for i in np.flatnonzero(~truth):
            ext[i] = bool(expr.evaluate([codes[i]]))
    except Exception:
        return None
    return ext


def _distance(ext, truth, conservative):
    return int((ext & ~truth).sum()) if conservative else int((ext != truth).sum())


def run_chain(args):
    pattern, black, white, alpha, start, seed, iters, variant = args
    conservative = variant == "conservative"
    random.seed(seed)
    np.random.seed(seed)
    G = _authors()
    grammar = G.make_grammar(4, 4)
    codes = G.get_all_codes(4, 4)
    mm = get_game(4, 4)
    truth = mm.table[mm.index(pattern)] == mm.encode(black, white)
    query = codes[mm.index(pattern)]
    feedback = (white, black)  # authors' order is (yellow, green) = (white, black)

    exact = G.find_disjunctive_representation(G.find_mastermind_possibilities(codes, query, feedback), grammar)
    exact_ext = _evaluate(exact, codes, truth, conservative)
    assert exact_ext is not None and (exact_ext == truth).all(), "exact expression does not match the feedback"

    max_size = max(int(1.5 * exact.size), 50)  # keeps chains at small alpha from growing without bound
    expr = exact if start == "exact" else G.sample(grammar)
    ext = _evaluate(expr, codes, truth, conservative)
    score = -math.inf if ext is None else alpha * expr.log_probability - _distance(ext, truth, conservative)
    found = {}
    accepted = 0
    for _ in range(iters):
        prop = G.resample_random_subtree(expr, grammar)
        if prop.size > max_size:
            continue
        pext = _evaluate(prop, codes, truth, conservative)
        if pext is None:
            continue
        d = _distance(pext, truth, conservative)
        pscore = alpha * prop.log_probability - d
        key = (np.packbits(pext).tobytes(), round(prop.log_probability, 6))
        if key not in found:
            found[key] = (prop.log_probability, d, prop.size, str(prop))
        log_ratio = pscore - score - (prop.log_probability - expr.log_probability)  # prior-proposal correction
        if score == -math.inf or math.log(random.random() + 1e-300) < log_ratio:
            expr, score = prop, pscore
            accepted += 1
    return {"alpha": alpha, "start": start, "seed": seed, "accepted": accepted, "found": found,
            "exact": (exact.log_probability, 0, exact.size, str(exact))}


def search(case_idx, iters, workers, variant, rep=1):
    mm = get_game(4, 4)
    pattern, (black, white) = cases(mm)[case_idx]
    starts = ["exact", "exact", "prior", "prior"]
    seed0 = 1000 * case_idx + (rep - 1) * REP_SEED_OFFSET
    jobs = [(pattern, black, white, a, s, seed0 + k, iters, variant) for k, (a, s) in enumerate(zip(CHAIN_ALPHAS, starts))]
    t0 = time.time()
    with Pool(workers) as pool:
        chains = pool.map(run_chain, jobs)
    folder = search_dir(variant, rep)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"case_{case_idx:02d}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"case": case_idx, "pattern": pattern, "feedback": (black, white), "iters": iters, "chains": chains}, f)
    n = len({k for c in chains for k in c["found"]})
    print(f"case {case_idx} {pattern} b{black} w{white}: {n} distinct approximations, "
          f"accept {[c['accepted'] for c in chains]}, {time.time() - t0:.0f}s", flush=True)


# ---- aggregation ----------------------------------------------------------
def symmetry_maps(mm):
    """For every query q: (pattern index, code permutation T) with ext_q[s] = ext_pattern[T[s]]."""
    pats = {mm.index(p): i for i, p in enumerate(PATTERNS)}
    maps = {}
    for sym in permutations(range(1, 5)):
        relabel = np.array([0, *sym])
        for pos in permutations(range(4)):
            moved = relabel[mm.codes[:, list(pos)]]
            T = np.array([mm.index(c) for c in moved])
            for q in range(mm.N):
                if q not in maps and T[q] in pats:
                    maps[q] = (pats[T[q]], T)
        if len(maps) == mm.N:
            break
    return maps


def aggregate(variant="conservative", rep=1):
    mm = get_game(4, 4)
    folder = search_dir(variant, rep)
    name = cache_name(variant, rep)
    all_cases = cases(mm)
    per_case = []
    for i, (pattern, fb) in enumerate(all_cases):
        d = pickle.load(open(folder / f"case_{i:02d}.pkl", "rb"))
        pool = {}
        for c in d["chains"]:
            pool.update(c["found"])
        items = list(pool.items())
        logp = np.array([v[0] for _, v in items])
        dist = np.array([v[1] for _, v in items])
        keep = set()
        for a in ALPHA_GRID:
            s = a * logp - dist
            kth = np.sort(s)[::-1][min(TOP_K, len(s)) - 1]
            keep.update(np.flatnonzero(s >= kth - 1e-9).tolist())
        exact = d["chains"][0]["exact"]
        truth = mm.table[mm.index(pattern)] == mm.encode(*fb)
        filters = [(np.unpackbits(np.frombuffer(items[j][0][0], np.uint8))[: mm.N].astype(bool), *items[j][1]) for j in sorted(keep)]
        if not any(f[2] == 0 and np.isclose(f[1], exact[0]) for f in filters):
            filters.append((truth.copy(), *exact))
        per_case.append({"pattern": pattern, "feedback": fb, "filters": filters})

    maps = symmetry_maps(mm)
    # Check: the symmetry maps send true extensions to true extensions for every query and feedback.
    case_index = {(c["pattern"], c["feedback"]): k for k, c in enumerate(per_case)}
    for q, (p, T) in maps.items():
        for o in set(mm.table[q].tolist()):
            ext_c = mm.table[mm.index(PATTERNS[p])] == o
            assert ((mm.table[q] == o) == ext_c[T]).all()
            assert (PATTERNS[p], mm.decode(o)) in case_index

    np.savez_compressed(
        OUT / f"{name}.npz",
        case_pattern=np.array([c["pattern"] for c in per_case]),
        case_feedback=np.array([c["feedback"] for c in per_case]),
        case_offsets=np.cumsum([0] + [len(c["filters"]) for c in per_case]),
        ext=np.array([f[0] for c in per_case for f in c["filters"]]),
        logp=np.array([f[1] for c in per_case for f in c["filters"]]),
        dist=np.array([f[2] for c in per_case for f in c["filters"]]),
        size=np.array([f[3] for c in per_case for f in c["filters"]]),
        query_pattern=np.array([maps[q][0] for q in range(mm.N)]),
        query_perm=np.array([maps[q][1] for q in range(mm.N)]),
    )
    (OUT / f"{name}_expressions.json").write_text(json.dumps(
        [{"pattern": c["pattern"], "feedback": c["feedback"], "filters": [[f[1], f[2], f[4]] for f in c["filters"]]} for c in per_case], indent=0))
    src_hash = hashlib.sha256(b"".join((AUTHORS / f).read_bytes() for f in ("grammar.py", "primitives.py", "utils.py"))).hexdigest()
    first = pickle.load(open(folder / "case_00.pkl", "rb"))
    manifest = {
        "version": 1, "variant": variant,
        "distance": "false retentions; infinite if an allowed code is ruled out" if variant == "conservative" else "false retentions + false exclusions",
        "procedure": "Manuscript Methods: 4 MH chains per (query, feedback) case with tree-regeneration proposals; top 10 per alpha on a log grid 1e-4..100 plus the exact expression",
        "cases": len(per_case), "filters_total": int(sum(len(c["filters"]) for c in per_case)),
        "iterations_per_chain": first["iters"], "chain_alphas": CHAIN_ALPHAS, "chain_starts": ["exact", "exact", "prior", "prior"],
        "alpha_grid": ALPHA_GRID, "top_k": TOP_K, "seeds": f"1000 * case + chain + {(rep - 1) * REP_SEED_OFFSET}", "replicate": rep,
        "authors_code_sha256": src_hash,
        "our_choices": "chain alphas, starting points, proposal correction (prior ratio exponent alpha - 1) and a size cap of max(1.5 x exact size, 50) nodes are not specified in the paper",
    }
    (OUT / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--case", type=int, required=True)
    s.add_argument("--iters", type=int, default=100_000)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--variant", choices=["conservative", "symmetric"], default="conservative")
    s.add_argument("--rep", type=int, default=1)
    g = sub.add_parser("aggregate")
    g.add_argument("--variant", choices=["conservative", "symmetric"], default="conservative")
    g.add_argument("--rep", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "search":
        search(args.case, args.iters, args.workers, args.variant, args.rep)
    else:
        aggregate(args.variant, args.rep)


if __name__ == "__main__":
    main()
