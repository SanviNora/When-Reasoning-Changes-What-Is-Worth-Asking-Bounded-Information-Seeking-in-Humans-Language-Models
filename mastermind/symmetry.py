"""Symmetry canonicalization.

Mastermind's rules do not change if the digits are renamed or the positions are reordered, as long as
the same change is applied to every guess and to the secret. A state family is therefore keyed by the
lexicographically smallest form of its set of clues over all 24 x 24 = 576 such transforms, so
equivalent problems, re-orderings of the same clues, and their relabelings share one key and can never
fall into different data splits.
"""
from functools import lru_cache
from itertools import permutations

import numpy as np

from mastermind.engine import get_game


@lru_cache(maxsize=None)
def transforms():
    """Code-index maps T for all digit relabelings x position permutations (4 x 4 game)."""
    mm = get_game(4, 4)
    maps = []
    for sym in permutations(range(1, 5)):
        relabel = np.array([0, *sym])
        for pos in permutations(range(4)):
            moved = relabel[mm.codes[:, list(pos)]]
            maps.append(np.array([mm.index(c) for c in moved]))
    return np.stack(maps)  # (576, 256)


def family_key(history):
    """Canonical key of a set of clues [(guess_idx, (black, white)), ...], order ignored."""
    mm = get_game(4, 4)
    best = None
    for T in transforms():
        form = tuple(sorted((mm.to_str(int(T[q])), b, w) for q, (b, w) in history))
        if best is None or form < best:
            best = form
    return "|".join(f"{g}:{b}{w}" for g, b, w in best)
