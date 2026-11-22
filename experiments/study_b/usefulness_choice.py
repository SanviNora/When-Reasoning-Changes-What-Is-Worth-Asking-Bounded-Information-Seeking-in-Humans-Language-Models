"""Usefulness and native question choice for the depth model.

Used by Study B (native choice: the T0 runs on the choice-calibration families) and by Study C
(the aid conditions and the C-c confirmation runs; those outputs are kept with Study C, not in this
repository).

For each family h and query pair (q1, q2) of nonterminal probes (q not in S(h), so no immediate win):
  usefulness  U(q) = (1/K) sum_a P(d in S_a): the model identifies a code after h + (q, a), summed
              exactly over every branch a, `--branch-samples` completions per branch; each answer is
              scored Y = 1/K_a if d is in S_a
  choice      the model picks A or B; order and labels randomized; no scores shown
Pairs pit a simpler query (fewer distinct digits) against a more informative one, the contrast in the
human Experiment 2; pair metadata records EIG, V* = m/K and digit counts for both.

  python experiments/study_b/usefulness_choice.py --model qwen3-32b --condition T0 \
      --splits choice_calibration --pairs-per-family 2 --branch-samples 4 --choice-samples 8
Long runs can be split across jobs: `--shard i --shards n` makes every n-th call (disjoint parts, so no reply is
ever generated twice) and only fills the reply cache; a final run without --shard reads every reply from the
cache and writes the files.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import prompts  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from mastermind.llm import Client  # noqa: E402
from mastermind.states import n_distinct  # noqa: E402

FAMILIES = ROOT / "data/splits/families_v1.json"
OUT = ROOT / "results/study_b/usefulness_choice"


def v_star(mm, q, mask):
    return len(set(mm.table[q, mask].tolist())) / mask.sum()


def candidate_pairs(mm, hist, rng, n_pairs):
    """Pairs (simple, complex): the simple query has fewer distinct digits and lower EIG."""
    mask = mm.consistent(hist)
    eig = mm.eig(mask)
    probes = np.flatnonzero(~mask & (eig > 0))
    simple = [q for q in probes if n_distinct(mm, q) <= 2]
    complex_ = [q for q in probes if n_distinct(mm, q) >= 3]
    pairs = []
    for _ in range(50 * n_pairs):
        if len(pairs) == n_pairs or not simple or not complex_:
            break
        a, b = int(rng.choice(simple)), int(rng.choice(complex_))
        if eig[b] > eig[a] and (a, b) not in pairs:
            pairs.append((a, b))
    return [{"q_simple": mm.to_str(a), "q_complex": mm.to_str(b),
             "eig": [round(float(eig[a]), 4), round(float(eig[b]), 4)],
             "vstar": [round(v_star(mm, a, mask), 4), round(v_star(mm, b, mask), 4)],
             "distinct_digits": [n_distinct(mm, a), n_distinct(mm, b)]} for a, b in pairs]


def branches(mm, hist, q):
    mask = mm.consistent(hist)
    return {int(o): int(n) for o, n in zip(*np.unique(mm.table[q, mask], return_counts=True))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", default="T0")
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--pairs-per-family", type=int, default=2)
    ap.add_argument("--branch-samples", type=int, default=4)
    ap.add_argument("--choice-samples", type=int, default=8)
    ap.add_argument("--pairs-file", help="fixed pair manifest (C-c confirmation); otherwise pairs are drawn")
    ap.add_argument("--families", default=str(FAMILIES), help="family file (default: data/splits/families_v1.json)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, help="reply cap for every call (default: the registry value); added to the output name")
    ap.add_argument("--shard", type=int, default=0, help="this job's part (0-based) when the calls are split across jobs")
    ap.add_argument("--shards", type=int, default=1, help="number of jobs the calls are split across")
    args = ap.parse_args()
    if not 0 <= args.shard < args.shards:
        ap.error("--shard must be between 0 and --shards - 1")

    mm = get_game(4, 4)
    rng = np.random.default_rng(7)
    fams = [f for f in json.loads(Path(args.families).read_text()) if f["split"] in args.splits]
    fixed = json.loads(Path(args.pairs_file).read_text()) if args.pairs_file else None

    pair_rows, jobs = [], []
    for fam in fams:
        hist = [(mm.index(g), (b, w)) for g, b, w in fam["history"]]
        pairs = [p for p in fixed if p["family_id"] == fam["family_id"]] if fixed else candidate_pairs(mm, hist, rng, args.pairs_per_family)
        for k, pr in enumerate(pairs):
            pid = pr.get("pair_id", f"{fam['family_id']}:P{k}")
            pair_rows.append({"pair_id": pid, "family_id": fam["family_id"], "split": fam["split"], **pr})
            for which in ("q_simple", "q_complex"):
                q = mm.index(pr[which])
                for o, n in branches(mm, hist, q).items():
                    b, w = mm.decode(o)
                    prompt, th = prompts.build(mm, "identify", args.condition, hist + [(q, (b, w))])
                    for s in range(args.branch_samples):
                        jobs.append({"kind": "branch", "pair_id": pid, "query": which, "outcome": [b, w], "K_a": n,
                                     "prompt": prompt, "template": th, "sample": s})
            for s in range(args.choice_samples):
                flip = bool(rng.random() < 0.5)
                a_q, b_q = (pr["q_complex"], pr["q_simple"]) if flip else (pr["q_simple"], pr["q_complex"])
                prompt, th = prompts.build(mm, "choice", args.condition, hist, a=mm.index(a_q), b=mm.index(b_q))
                jobs.append({"kind": "choice", "pair_id": pid, "A": a_q, "B": b_q, "prompt": prompt, "template": th, "sample": s})

    client = Client(args.model, condition=args.condition, max_tokens=args.max_tokens)
    print(f"{len(fams)} families, {len(pair_rows)} pairs, {len(jobs)} calls", flush=True)
    calls = [(j["prompt"], f"{j['pair_id']}:{j['kind']}:{j.get('query', '')}:{j.get('outcome', '')}:{j['sample']}") for j in jobs]
    if args.shards > 1:  # a part only fills the reply cache; the final run without --shard writes the files
        part = calls[args.shard::args.shards]
        client.batch(part, workers=args.workers)
        print(f"part {args.shard} of {args.shards}: {len(part)} calls cached", flush=True)
        return
    recs = client.batch(calls, workers=args.workers)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.condition}_{'+'.join(args.splits)}" + (f"_cap{args.max_tokens}" if args.max_tokens else "")
    (OUT / f"{tag}_pairs.json").write_text(json.dumps(pair_rows, indent=1))
    fam_hist = {f["family_id"]: [(mm.index(g), (b, w)) for g, b, w in f["history"]] for f in fams}
    tmp = OUT / f"{tag}.jsonl.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        for j, r in zip(jobs, recs):
            row = {k: v for k, v in j.items() if k != "prompt"}
            row.update({"condition": args.condition, "prompt_sha": r["prompt_sha"],
                        "completion_tokens": (r.get("usage") or {}).get("completion"), "reply_tail": r["reply"][-300:]})
            if j["kind"] == "branch":
                fam_id = j["pair_id"].split(":")[0] if not fixed else next(p["family_id"] for p in pair_rows if p["pair_id"] == j["pair_id"])
                q = mm.index(next(p for p in pair_rows if p["pair_id"] == j["pair_id"])[j["query"]])
                mask = mm.consistent(fam_hist[fam_id] + [(q, tuple(j["outcome"]))])
                d = prompts.parse(mm, "identify", r["reply"])
                row.update({"parsed": d is not None, "answer": None if d is None else mm.to_str(d),
                            "in_branch": d is not None and bool(mask[d]), "Y": (1 / j["K_a"]) if d is not None and mask[d] else 0.0})
            else:
                c = prompts.parse(mm, "choice", r["reply"])
                row.update({"parsed": c is not None, "choice": c,
                            "chose": None if c is None else ("q_simple" if (j["A"] if c == "A" else j["B"]) == next(
                                p for p in pair_rows if p["pair_id"] == j["pair_id"])["q_simple"] else "q_complex")})
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, OUT / f"{tag}.jsonl")  # one writer per file; the finished file replaces the old one
    print(json.dumps({"model": args.model, "condition": args.condition, "pairs": len(pair_rows), "calls": len(jobs)}, indent=2))


if __name__ == "__main__":
    main()
