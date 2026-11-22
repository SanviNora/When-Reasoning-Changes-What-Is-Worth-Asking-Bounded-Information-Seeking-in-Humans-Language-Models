"""Generate symmetry-canonical state families and assign them to fixed splits.

A family is a history of 2-3 clues that leaves 4-16 codes possible. Families are deduplicated by
their canonical key (mastermind/symmetry.py), so no two splits share an equivalent problem, and
families equivalent to an Experiment 1 starting position are excluded (that set stays separate).

Writes data/splits/families_v1.json and data/splits/manifest_v1.json. Commit both before any
confirmation run.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind.engine import get_game  # noqa: E402
from mastermind.symmetry import family_key  # noqa: E402

OUT = ROOT / "data/splits"
SPLITS = [("development", 24), ("processing_calibration", 60), ("choice_calibration", 24),
          ("support_calibration", 24), ("confirmation", 80), ("reserve", 40)]
SEED = 20261117


def main():
    mm = get_game(4, 4)
    e1 = json.loads((ROOT / "data/stimuli/e1_stimuli.json").read_text())
    excluded = {family_key([(mm.index(g["guess"]), (g["black"], g["white"])) for g in s["supplied"]])
                for s in e1 if len(s["supplied"]) >= 2}

    rng = np.random.default_rng(SEED)
    need = sum(n for _, n in SPLITS)
    families, seen, tried = [], set(excluded), 0
    while len(families) < need:
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
        seen.add(key)
        families.append({"key": key, "K": k, "n_clues": len(hist),
                         "history": [[mm.to_str(q), b, w] for q, (b, w) in hist]})

    order = rng.permutation(len(families))
    out, i = [], 0
    for name, n in SPLITS:
        for j in order[i:i + n]:
            out.append({"family_id": f"F{len(out):03d}", "split": name, **families[j]})
        i += n

    OUT.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(out, indent=1)
    (OUT / "families_v1.json").write_text(blob)
    manifest = {
        "version": 1, "seed": SEED, "families": len(out), "candidates_drawn": tried,
        "splits": {name: n for name, n in SPLITS},
        "K_range": [4, 16], "clues": [2, 3], "excluded_e1_equivalents": len(excluded),
        "sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "K_by_split": {name: round(float(np.mean([f["K"] for f in out if f["split"] == name])), 2) for name, _ in SPLITS},
    }
    (OUT / "manifest_v1.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
