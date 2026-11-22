"""Exact Mastermind solver: codes, feedback, consistent sets, information gain.

Defaults match Cheyette et al. (2025): 4 positions x 4 digits (1-4) = 256 codes.
Feedback is (black, white): black = right digit in the right position,
white = right digit in the wrong position (the "green" and "yellow" of the human data).
"""
from functools import lru_cache
from itertools import product

import numpy as np


class Mastermind:
    def __init__(self, n_positions=4, n_colors=4):
        self.P = n_positions
        self.C = n_colors
        self.codes = np.array(list(product(range(1, n_colors + 1), repeat=n_positions)), dtype=np.int8)
        self.N = len(self.codes)
        self._index = {tuple(c): i for i, c in enumerate(self.codes)}
        self.table = self._feedback_table()  # table[q, s] = encoded feedback of guess q against secret s
        self.n_outcomes = (self.P + 1) ** 2

    # ---- encoding -------------------------------------------------------
    def encode(self, black, white):
        return black * (self.P + 1) + white

    def decode(self, f):
        return divmod(int(f), self.P + 1)

    def index(self, code):
        """Index of a code given as tuple, list, array, or string like '1123'."""
        if isinstance(code, str):
            code = tuple(int(ch) for ch in code.strip("[]").replace(",", "").replace(" ", ""))
        return self._index[tuple(int(x) for x in code)]

    def to_str(self, i):
        return "".join(str(int(x)) for x in self.codes[i])

    # ---- feedback -------------------------------------------------------
    def feedback(self, guess, secret):
        g, s = np.asarray(guess), np.asarray(secret)
        black = int((g == s).sum())
        common = sum(min((g == c).sum(), (s == c).sum()) for c in range(1, self.C + 1))
        return black, int(common - black)

    def _feedback_table(self):
        codes = self.codes
        black = (codes[:, None, :] == codes[None, :, :]).sum(-1)
        counts = np.stack([(codes == c).sum(1) for c in range(1, self.C + 1)], axis=1)
        common = np.minimum(counts[:, None, :], counts[None, :, :]).sum(-1)
        return (black * (self.P + 1) + (common - black)).astype(np.int16)

    # ---- beliefs --------------------------------------------------------
    def consistent(self, history, mask=None):
        """Boolean mask of codes consistent with history [(guess_idx, (black, white)), ...]."""
        m = np.ones(self.N, bool) if mask is None else mask.copy()
        for q, (b, w) in history:
            m &= self.table[q] == self.encode(b, w)
        return m

    def outcome_probs(self, q, mask):
        """P(feedback | q) under a uniform belief over the codes in mask."""
        counts = np.bincount(self.table[q, mask], minlength=self.n_outcomes)
        return counts / max(mask.sum(), 1)

    def eig(self, mask, queries=None):
        """Expected information gain (bits) of each query under a uniform belief over mask.
        Feedback is deterministic given the secret, so EIG = entropy of the outcome distribution."""
        queries = np.arange(self.N) if queries is None else np.asarray(queries)
        sub = self.table[np.ix_(queries, np.flatnonzero(mask))]
        n = sub.shape[1]
        out = np.empty(len(queries))
        for k, row in enumerate(sub):
            p = np.bincount(row, minlength=self.n_outcomes) / n
            p = p[p > 0]
            out[k] = -(p * np.log2(p)).sum()
        return out

    def info_gain(self, q, outcome, mask):
        """Bits gained by observing an encoded outcome for query q: log2|H| - log2|H ∩ a|."""
        n_before = mask.sum()
        n_after = (mask & (self.table[q] == outcome)).sum()
        return float(np.log2(n_before) - np.log2(max(n_after, 1)))

    # ---- play -----------------------------------------------------------
    def play_greedy_eig(self, secret, max_guesses=15, first=None, history=None):
        """Greedy Max-EIG player (guesses only from all codes, prefers consistent codes on ties).
        With `history`, continues a game from that position; the returned list holds only
        the new guesses."""
        mask = self.consistent(history) if history else np.ones(self.N, bool)
        new = []
        for t in range(max_guesses):
            if t == 0 and first is not None:
                q = first
            elif mask.sum() == 1:
                q = int(np.flatnonzero(mask)[0])
            else:
                e = self.eig(mask) + 1e-9 * mask  # tie-break toward consistent codes
                q = int(np.argmax(e))
            fb = self.decode(self.table[q, secret])
            new.append((q, fb))
            if fb[0] == self.P:
                return new
            mask = self.consistent([(q, fb)], mask)
        return new


@lru_cache(maxsize=None)
def get_game(n_positions=4, n_colors=4):
    return Mastermind(n_positions, n_colors)
