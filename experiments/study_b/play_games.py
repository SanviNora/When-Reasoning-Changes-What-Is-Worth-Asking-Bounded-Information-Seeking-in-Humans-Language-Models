"""B1: language models play Mastermind like the human participants.

  e1    continue each of the 132 Experiment 1 starting positions (supplied guesses shown as already
        played, as on the human board), `--completions` times each
  free  play full games from the start against `--games` seeded secrets (Experiment 3 style)

Each turn is a fresh call with the full board. The player gets up to 15 guesses of its own, as people
did. An unparseable or invalid guess gets up to 2 re-asks with a one-line format reminder; after
that the turn counts as an invalid guess with no feedback. Every guess is scored offline by the oracle.

Run inside a job where the model is served:
  python experiments/study_b/play_games.py --model qwen3-32b --mode e1 --completions 10
Long runs can be split across jobs: `--shard i --shards n` plays every n-th game (a partial file under
b1/shards/); a final run without --shard then reads every reply from the cache and writes the single file.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import prompts  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from mastermind.llm import Client  # noqa: E402

OUT = ROOT / "results/study_b/b1"
MAX_OWN_GUESSES = 15
REMINDER = ('\n\nYour previous reply did not end with a valid guess. End your reply with one line of JSON '
            'such as {"guess": "1234"}, using four digits from 1 to 4.')


def play(client, mm, secret, supplied, condition, game_id):
    history = list(supplied)
    turns = []
    for own in range(MAX_OWN_GUESSES):
        mask = mm.consistent(history) if history else np.ones(mm.N, bool)
        prompt, th = prompts.build(mm, "guess", condition, history, n_supplied=len(supplied))
        q, rec, attempts = None, None, 0
        for attempt in range(3):
            rec = client.complete(prompt + (REMINDER if attempt else ""), sample=f"{game_id}:{own}:{attempt}")
            attempts += 1
            q = prompts.parse(mm, "guess", rec["reply"])
            if q is not None:
                break
        turn = {"turn": own + 1, "attempts": attempts, "template": th, "codes_remaining": int(mask.sum()),
                "completion_tokens": (rec.get("usage") or {}).get("completion"), "reply_tail": rec["reply"][-300:]}
        if q is None:
            turn["guess"] = None
            turns.append(turn)
            continue
        b, w = mm.decode(mm.table[q, secret])
        eig = mm.eig(mask, [q])[0]
        turn.update({"guess": mm.to_str(q), "black": b, "white": w, "consistent": bool(mask[q]),
                     "eig": round(float(eig), 4), "max_eig": round(float(mm.eig(mask).max()), 4)})
        turns.append(turn)
        history.append((q, (b, w)))
        if b == mm.P:
            break
    solved = bool(turns and turns[-1].get("black") == mm.P)
    return {"game_id": game_id, "secret": mm.to_str(secret), "supplied": [[mm.to_str(q), b, w] for q, (b, w) in supplied],
            "solved": solved, "own_guesses": len(turns), "invalid_turns": sum(t["guess"] is None for t in turns),
            "turns": turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["e1", "free", "both"], required=True)
    ap.add_argument("--condition", default="T0")
    ap.add_argument("--completions", type=int, default=10)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0,
                    help="pilot: play an evenly spaced subset of N games (same game ids, so a later full run reuses them)")
    ap.add_argument("--shard", type=int, default=0, help="this job's part (0-based) when the games are split")
    ap.add_argument("--shards", type=int, default=1, help="number of jobs the games are split across")
    args = ap.parse_args()
    if not 0 <= args.shard < args.shards:
        ap.error("--shard must be between 0 and --shards - 1")

    for mode in (["e1", "free"] if args.mode == "both" else [args.mode]):
        run(args, mode)


def run(args, mode):
    mm = get_game(4, 4)
    client = Client(args.model, condition=args.condition)
    jobs = []
    if mode == "e1":
        for st in json.loads((ROOT / "data/stimuli/e1_stimuli.json").read_text()):
            supplied = [(mm.index(g["guess"]), (g["black"], g["white"])) for g in st["supplied"]]
            for c in range(args.completions):
                jobs.append((mm.index(st["secret"]), supplied, f"{st['stimulus_id']}:{st['condition']}:{c}"))
    else:
        rng = np.random.default_rng(20261117)
        for g, s in enumerate(rng.integers(mm.N, size=args.games)):
            jobs.append((int(s), [], f"free:{g}"))
    if args.limit:
        jobs = jobs[::max(1, len(jobs) // args.limit)][:args.limit]
    out_dir = OUT / "pilots" if args.limit else OUT  # pilots (cost and format checks) are kept apart from results
    name = f"{args.model}_{mode}_{args.condition}{'_pilot' if args.limit else ''}"
    if args.shards > 1:  # every n-th game, so each part gets a similar mix of starting positions
        jobs = jobs[args.shard::args.shards]
        out_dir, name = OUT / "shards", f"{name}_shard{args.shard}of{args.shards}"

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    tmp = path.with_suffix(f".tmp{os.getpid()}")  # one writer per file; the finished file replaces the old one
    with ThreadPoolExecutor(args.workers) as ex, open(tmp, "w") as f:
        for n, rec in enumerate(ex.map(lambda j: play(client, mm, j[0], j[1], args.condition, j[2]), jobs), 1):
            rec.update({"model": args.model, "mode": mode, "condition": args.condition,
                        "decoding": client.decoding, "extra": client.extra, "revision": client.revision})
            f.write(json.dumps(rec) + "\n")
            if n % 50 == 0:
                print(f"{n}/{len(jobs)} games", flush=True)
    os.replace(tmp, path)
    games = [json.loads(line) for line in open(path)]
    print(json.dumps({"model": args.model, "mode": mode, "games": len(games),
                      "solve_rate": round(float(np.mean([g["solved"] for g in games])), 3),
                      "mean_own_guesses_solved": round(float(np.mean([g["own_guesses"] for g in games if g["solved"]] or [0])), 2),
                      "invalid_turn_rate": round(sum(g["invalid_turns"] for g in games) / max(1, sum(g["own_guesses"] for g in games)), 3)}, indent=2))


if __name__ == "__main__":
    main()
