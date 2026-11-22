"""Mid-game states for probing, and candidate questions for each state.

A state is a history [(guess_idx, (black, white)), ...] that leaves at least `min_remaining`
codes possible. States come from two sources:
  - human: prefixes of real games from Cheyette et al. (2025), Experiment 3 (free play)
  - random: games where each guess is drawn uniformly from all codes
"""
import ast
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HUMAN_E3 = ROOT / "data/human/cheyette2025_osf_vjrqu/Experiments/E3/mastermind_E3.csv"
HUMAN_E3_CLEAN = ROOT / "data/human/cheyette2025_osf_vjrqu/Experiments/E3/analysis/e3_human_preprocessed.csv"


def load_human_games(mm, path=HUMAN_E3, clean_path=HUMAN_E3_CLEAN):
    """Human Mastermind games as lists of (guess_idx, (black, white)), restricted to the
    participants retained in the authors' cleaned file."""
    keep = None
    if clean_path.exists():
        with open(clean_path) as f:
            keep = {r["uniqueid"] for r in csv.DictReader(f) if r["game"] == "Mastermind"}
    games = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if keep is not None and r["uniqueid"] not in keep:
                continue
            games.setdefault((r["pid"], r["trial_id"]), []).append(r)
    out = []
    for rows in games.values():
        rows.sort(key=lambda r: int(r["guess_number"]))
        out.append([(mm.index(ast.literal_eval(r["guess"])), (int(r["n_green"]), int(r["n_yellow"]))) for r in rows])
    return out


def human_states(mm, rng, n, max_turns=3, min_remaining=6):
    """Sample n distinct prefixes (1..max_turns guesses) of human games."""
    pool = []
    for game in load_human_games(mm):
        for t in range(1, min(max_turns, len(game)) + 1):
            hist = game[:t]
            if mm.consistent(hist).sum() >= min_remaining:
                pool.append(tuple(hist))
    pool = sorted(set(pool))
    pick = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [list(pool[i]) for i in pick]


def random_states(mm, rng, n, max_turns=3, min_remaining=6):
    states, seen = [], set()
    while len(states) < n:
        secret = rng.integers(mm.N)
        hist = []
        for _ in range(rng.integers(1, max_turns + 1)):
            q = int(rng.integers(mm.N))
            hist.append((q, mm.decode(mm.table[q, secret])))
        key = tuple(hist)
        if key not in seen and mm.consistent(hist).sum() >= min_remaining:
            seen.add(key)
            states.append(hist)
    return states


def n_distinct(mm, q):
    return len(set(mm.codes[q].tolist()))


def candidate_questions(mm, history, n_top=3, n_simple=3):
    """Top-EIG queries plus simpler queries (fewer distinct digits) with non-trivial EIG.

    The 'simple' set is a placeholder for the bounded-learner complexity measure; with
    few distinct digits the feedback is easier to interpret (Cheyette et al., 2023)."""
    mask = mm.consistent(history)
    eig = mm.eig(mask)
    order = np.argsort(-eig, kind="stable")
    top = [int(q) for q in order[:n_top]]
    informative = eig[eig > 0]
    floor = np.quantile(informative, 0.25) if len(informative) else 0
    pool = [int(q) for q in order if q not in top and eig[q] >= floor and eig[q] > 0]
    pool.sort(key=lambda q: (n_distinct(mm, q), -eig[q]))
    simple = pool[:n_simple]
    return top + simple, eig
