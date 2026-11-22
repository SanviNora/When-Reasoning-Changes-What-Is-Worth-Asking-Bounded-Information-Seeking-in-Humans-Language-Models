"""Experiment 3 (free play): solve rate, guesses to solve, information per
guess and entropy curves for people, against the ideal (greedy Max-EIG) and random-query
benchmarks from the paper (3.6 and 4.8 guesses).

Writes results/study_a/e3_reproduction.json.
Run: .venv/bin/python experiments/study_a/e3_reproduction.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind.engine import get_game  # noqa: E402
from mastermind.states import load_human_games, n_distinct  # noqa: E402

SUMMARY = ROOT / "results/study_a/e3_reproduction.json"
MAX_GUESSES = 15


def random_player(mm, secret, rng):
    """Paper's random benchmark: guess uniformly from all codes until one code remains."""
    mask, n = np.ones(mm.N, bool), 0
    while True:
        n += 1
        if mask.sum() == 1:
            return n
        q = int(rng.integers(mm.N))
        if q == secret:
            return n
        mask &= mm.table[q] == mm.table[q, secret]


def entropy_curve(mm, game, length=8):
    """log2 of the number of codes still possible before each guess (0 once solved)."""
    out, mask = [], np.ones(mm.N, bool)
    for t in range(length):
        out.append(float(np.log2(mask.sum())) if t < len(game) else 0.0)
        if t < len(game):
            mask = mm.consistent([game[t]], mask)
    return out


def main():
    mm = get_game(4, 4)
    games = load_human_games(mm)
    solved = [g for g in games if g[-1][1][0] == mm.P]
    n_guesses = np.array([len(g) for g in solved])

    bits = []
    for g in games:
        mask = np.ones(mm.N, bool)
        for q, (b, w) in g:
            if mask.sum() > 1:
                bits.append(mm.info_gain(q, mm.encode(b, w), mask))
            mask = mm.consistent([(q, (b, w))], mask)

    # Query structure: how often a guess is still possible given the feedback so far, and
    # how many distinct colors guesses use, for people and the ideal player.
    consistent, distinct_h, first_h = [], [], []
    for g in games:
        mask = np.ones(mm.N, bool)
        first_h.append(n_distinct(mm, g[0][0]))
        for t, (q, fb) in enumerate(g):
            distinct_h.append(n_distinct(mm, q))
            if t > 0 and mask.sum() > 1:
                consistent.append(bool(mask[q]))
            mask = mm.consistent([(q, fb)], mask)
    ideal_games = [mm.play_greedy_eig(s) for s in range(mm.N)]
    distinct_i = [n_distinct(mm, q) for g in ideal_games for q, _ in g]

    def share(xs):
        return {k: round(xs.count(k) / len(xs), 3) for k in (1, 2, 3, 4)}

    rng = np.random.default_rng(0)
    ideal = np.array([len(g) for g in ideal_games])
    rand = np.array([random_player(mm, int(rng.integers(mm.N)), rng) for _ in range(1000)])

    summary = {
        "participants_cleaned": 39,
        "games": len(games),
        "solve_rate": round(len(solved) / len(games), 3),
        "human_guesses_solved_games": {"mean": round(float(n_guesses.mean()), 2), "sd": round(float(n_guesses.std(ddof=1)), 2)},
        "paper_reports": {"solve_rate": 0.86, "human_guesses": 7.5, "max_eig": 3.6, "random": 4.8},
        "ideal_max_eig_all_256_secrets": {"mean": round(float(ideal.mean()), 2), "max": int(ideal.max())},
        "random_query_1000_games": {"mean": round(float(rand.mean()), 2), "sd": round(float(rand.std(ddof=1)), 2), "max": int(rand.max()),
                                    "note": "mean matches the paper (4.8); the spread does not (paper SD 1.0, max 7). Drawing without "
                                            "repeats gives 4.74 / 1.11 / 10 and drawing only possible codes gives 3.78 / 0.78 / 6, so no "
                                            "simple rule matches all three; reported as a minor discrepancy in a benchmark only."},
        "query_structure": {
            "human_consistent_guess_rate": round(float(np.mean(consistent)), 3),
            "distinct_colors_per_guess_human": share(distinct_h),
            "distinct_colors_per_guess_ideal": share(distinct_i),
            "distinct_colors_first_guess_human": share(first_h),
        },
        "human_bits_per_guess_mean": round(float(np.mean(bits)), 3),
        "entropy_curve_human_mean": [round(float(x), 3) for x in np.mean([entropy_curve(mm, g) for g in games], axis=0)],
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
