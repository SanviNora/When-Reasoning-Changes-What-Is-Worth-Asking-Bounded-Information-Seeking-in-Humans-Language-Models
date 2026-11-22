"""Reply-cap pilot for the Study C usefulness runs.

At the registry cap (6,144 tokens) about a quarter of the aided (T3) identification replies in the usefulness
calibration runs are cut off, against about a tenth under T0. This pilot reruns a random sample of those cut-off
replies, with the identical prompts, at a higher cap and reports how many now finish. It decides whether the
Study C usefulness runs use a higher cap (the same for every condition).

The calls use the same cache keys as `usefulness_choice.py --max-tokens N`, so a full rerun at that cap reuses
these replies.

Run inside a job where the model is served:
  python experiments/study_c/cap_pilot.py --model qwen3-32b --max-tokens 15000 --n-t3 200 --n-t0 60
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import prompts  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from mastermind.llm import Client  # noqa: E402

UC = ROOT / "results/study_b/usefulness_choice"
OUT = ROOT / "results/study_c"
SPLITS = ("support_calibration", "choice_calibration")
OLD_CAP = 6144


def loops(text):
    """The reply repeats one line at least 3 times in its last 300 characters (e.g. 'Already tried.')."""
    lines = [x.strip() for x in text[-300:].split("\n") if x.strip()]
    return bool(lines) and collections.Counter(lines).most_common(1)[0][1] >= 3


def capped_items(mm, model, condition):
    fams = {f["family_id"]: [(mm.index(g), (b, w)) for g, b, w in f["history"]]
            for f in json.loads((ROOT / "data/splits/families_v1.json").read_text())}
    items = []
    for split in SPLITS:
        pairs = {p["pair_id"]: p for p in json.loads((UC / f"{model}_{condition}_{split}_pairs.json").read_text())}
        for line in open(UC / f"{model}_{condition}_{split}.jsonl"):
            r = json.loads(line)
            if r["kind"] != "branch" or (r.get("completion_tokens") or 0) < OLD_CAP:
                continue
            p = pairs[r["pair_id"]]
            q = mm.index(p[r["query"]])
            hist = fams[p["family_id"]] + [(q, tuple(r["outcome"]))]
            prompt, _ = prompts.build(mm, "identify", condition, hist)
            assert hashlib.sha256(prompt.encode()).hexdigest()[:16] == r["prompt_sha"], "prompt differs from the original run"
            key = f"{r['pair_id']}:branch:{r['query']}:{r['outcome']}:{r['sample']}"  # as in usefulness_choice.py
            items.append({"split": split, "record": r, "prompt": prompt, "key": key, "mask": mm.consistent(hist)})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--max-tokens", type=int, default=15000)
    ap.add_argument("--n-t3", type=int, default=200)
    ap.add_argument("--n-t0", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    mm = get_game(4, 4)
    rng = np.random.default_rng(args.seed)
    rows, summary = [], {"model": args.model, "old_cap": OLD_CAP, "new_cap": args.max_tokens, "seed": args.seed}
    for condition, n in (("T3", args.n_t3), ("T0", args.n_t0)):
        items = capped_items(mm, args.model, condition)
        pick = [items[i] for i in sorted(rng.choice(len(items), min(n, len(items)), replace=False))]
        client = Client(args.model, condition=condition, max_tokens=args.max_tokens)
        recs = client.batch([(it["prompt"], it["key"]) for it in pick], workers=args.workers)
        for it, rec in zip(pick, recs):
            tokens = (rec.get("usage") or {}).get("completion") or 0
            d = prompts.parse(mm, "identify", rec["reply"])
            rows.append({"condition": condition, "split": it["split"], "pair_id": it["record"]["pair_id"],
                         "query": it["record"]["query"], "outcome": it["record"]["outcome"], "sample": it["record"]["sample"],
                         "old_loop": loops(it["record"]["reply_tail"]), "completion_tokens": tokens,
                         "at_new_cap": tokens >= args.max_tokens, "new_loop": loops(rec["reply"]), "parsed": d is not None,
                         "in_branch": d is not None and bool(it["mask"][d])})
        r = [x for x in rows if x["condition"] == condition]
        fin = [x for x in r if x["parsed"]]
        summary[condition] = {
            "capped_at_old_cap": len(items), "rerun": len(r),
            "finished_and_parsed": round(len(fin) / len(r), 3),
            "still_at_new_cap": round(float(np.mean([x["at_new_cap"] for x in r])), 3),
            "looping_at_old_cap": round(float(np.mean([x["old_loop"] for x in r])), 3),
            "finished_among_old_loops": round(float(np.mean([x["parsed"] for x in r if x["old_loop"]] or [np.nan])), 3),
            "finished_among_old_nonloops": round(float(np.mean([x["parsed"] for x in r if not x["old_loop"]] or [np.nan])), 3),
            "in_branch_among_finished": round(float(np.mean([x["in_branch"] for x in fin] or [np.nan])), 3),
            "median_tokens_finished": float(np.median([x["completion_tokens"] for x in fin])) if fin else None,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"cap_pilot_{args.model}_cap{args.max_tokens}"
    stem.with_suffix(".jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
