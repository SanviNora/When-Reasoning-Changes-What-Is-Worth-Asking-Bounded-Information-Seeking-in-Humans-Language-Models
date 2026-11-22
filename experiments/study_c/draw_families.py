"""Fresh state families for the C-c confirmation run.

The confirmation families of families_v1.json were run under T0-T3 for C-a, and the reserve families hold the
Study B confirmation test, so C-c is confirmed on families drawn here, with their own committed seed. Drawn exactly
as in experiments/study_b/make_splits.py (histories of 2-3 clues, no clue solved, 4-16 codes still possible),
and in addition:
  - not equivalent under symmetry to any families_v1 family or any Experiment 1 starting position;
  - sharing no sub-history of >= 2 clues (under symmetry) with any families_v1 family, so no confirmation family
    contains a clue pair seen before (single clues can recur: there are only 256 x 14 of them).

Writes data/splits/families_c_v1.json (split "c_confirmation") and data/splits/manifest_c_v1.json.
Commit both before the confirmation run.

  python experiments/study_c/draw_families.py --n 120
"""
import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind.engine import get_game  # noqa: E402
from mastermind.symmetry import family_key  # noqa: E402

OUT = ROOT / "data/splits"
SEED = 20261120
SPLIT = "c_confirmation"


def sub_keys(hist):
    """Canonical keys of every sub-history of >= 2 clues (the full history included)."""
    return {family_key(list(c)) for n in range(2, len(hist) + 1) for c in combinations(hist, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of families (from the power simulation)")
    args = ap.parse_args()

    mm = get_game(4, 4)
    v1 = json.loads((OUT / "families_v1.json").read_text())
    e1 = json.loads((ROOT / "data/stimuli/e1_stimuli.json").read_text())
    e1_keys = {family_key([(mm.index(g["guess"]), (g["black"], g["white"])) for g in s["supplied"]])
               for s in e1 if len(s["supplied"]) >= 2}
    v1_hist = [[(mm.index(g), (b, w)) for g, b, w in f["history"]] for f in v1]
    seen = e1_keys | {f["key"] for f in v1}
    v1_sub = set().union(*(sub_keys(h) for h in v1_hist))

    rng = np.random.default_rng(SEED)
    families, tried, rejected_related = [], 0, 0
    while len(families) < args.n:
        tried += 1
        secret = int(rng.integers(mm.N))
        guesses = rng.choice(mm.N, size=int(rng.integers(2, 4)), replace=False)
        hist = [(int(q), mm.decode(mm.table[q, secret])) for q in guesses]
        if any(b == mm.P for _, (b, _) in hist):
            continue
        k = int(mm.consistent(hist).sum())
        if not 4 <= k <= 16:
            continue
        key = family_key(hist)
        if key in seen:
            continue
        if sub_keys(hist) & v1_sub:
            rejected_related += 1
            continue
        seen.add(key)
        families.append({"family_id": f"C{len(families):03d}", "split": SPLIT, "key": key, "K": k, "n_clues": len(hist),
                         "history": [[mm.to_str(q), b, w] for q, (b, w) in hist]})

    blob = json.dumps(families, indent=1)
    (OUT / "families_c_v1.json").write_text(blob)
    manifest = {"version": 1, "seed": SEED, "split": SPLIT, "families": len(families), "candidates_drawn": tried,
                "rejected_sharing_a_clue_pair_with_v1": rejected_related, "K_range": [4, 16], "clues": [2, 3],
                "mean_K": round(float(np.mean([f["K"] for f in families])), 2),
                "excluded": "families_v1 (all splits), Experiment 1 starting positions, any >= 2-clue sub-history of a v1 family",
                "families_v1_sha256": hashlib.sha256((OUT / "families_v1.json").read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(blob.encode()).hexdigest()}
    (OUT / "manifest_c_v1.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
