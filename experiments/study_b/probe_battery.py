"""B2: the diagnostic battery on the fixed state families.

Tasks per family:
  classify         full history; test codes = still possible (up to 6), near misses ruled out only
                   by the newest clue (up to 6), far misses ruled out earlier (4)
  classify_single  each clue on its own, same mix (single-clue decoding)
  identify         full history; name one code that could be the secret
Responses are scored offline; the process models (mastermind/process_models.py) are fitted to them.

  python experiments/study_b/probe_battery.py --model qwen3-32b --condition T0 \
      --splits processing_calibration confirmation --samples 4
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

FAMILIES = ROOT / "data/splits/families_v1.json"
OUT = ROOT / "results/study_b/battery"


def test_codes(mm, history, rng, n_pos=6, n_near=6, n_far=4):
    before = mm.consistent(history[:-1]) if len(history) > 1 else np.ones(mm.N, bool)
    after = mm.consistent(history)
    groups = {"possible": np.flatnonzero(after), "near": np.flatnonzero(before & ~after), "far": np.flatnonzero(~before)}
    want = {"possible": n_pos, "near": n_near, "far": n_far}
    items = []
    for kind, idx in groups.items():
        for i in rng.choice(idx, size=min(want[kind], len(idx)), replace=False):
            items.append((int(i), kind))
    order = rng.permutation(len(items))
    return [items[j] for j in order]


def build_jobs(mm, families, condition, samples, seed=0):
    rng = np.random.default_rng(seed)
    jobs = []
    for fam in families:
        hist = [(mm.index(g), (b, w)) for g, b, w in fam["history"]]
        variants = [("classify", hist, None)] + [("classify_single", [c], k) for k, c in enumerate(hist)] + [("identify", hist, None)]
        for task, h, clue in variants:
            items = test_codes(mm, h, rng) if task.startswith("classify") else []
            prompt, th = prompts.build(mm, "classify" if items else "identify", condition, h,
                                       **({"codes": [i for i, _ in items]} if items else {}))
            for s in range(samples):
                jobs.append({"prompt": prompt, "sample": s, "meta": {
                    "family_id": fam["family_id"], "split": fam["split"], "task": task, "clue_index": clue,
                    "history": [[mm.to_str(q), b, w] for q, (b, w) in h], "items": [[mm.to_str(i), kind] for i, kind in items],
                    "template": th, "condition": condition}})
    return jobs


def score(mm, meta, reply):
    hist = [(mm.index(g), (b, w)) for g, b, w in meta["history"]]
    mask = mm.consistent(hist)
    if meta["task"] == "identify":
        d = prompts.parse(mm, "identify", reply)
        return {"parsed": d is not None, "answer": None if d is None else mm.to_str(d), "correct": d is not None and bool(mask[d])}
    codes = [mm.index(c) for c, _ in meta["items"]]
    got = prompts.parse(mm, "classify", reply, codes) or {}
    labels = {mm.to_str(i): got.get(i) for i in codes}
    return {"parsed": bool(got), "complete": len(got) == len(codes), "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", default="T0")
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, help="reply cap for every call (default: the registry value); added to the output name")
    args = ap.parse_args()

    mm = get_game(4, 4)
    families = [f for f in json.loads(FAMILIES.read_text()) if f["split"] in args.splits]
    jobs = build_jobs(mm, families, args.condition, args.samples)
    client = Client(args.model, condition=args.condition, max_tokens=args.max_tokens)
    print(f"{len(families)} families, {len(jobs)} calls", flush=True)
    recs = client.batch([(j["prompt"], f"{j['meta']['family_id']}:{j['meta']['task']}:{j['meta']['clue_index']}:{j['sample']}") for j in jobs],
                        workers=args.workers)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (f"{args.model}_{args.condition}_{'+'.join(args.splits)}" + (f"_cap{args.max_tokens}" if args.max_tokens else "") + ".jsonl")
    parsed = 0
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        for j, r in zip(jobs, recs):
            s = score(mm, j["meta"], r["reply"])
            parsed += s["parsed"]
            f.write(json.dumps({**j["meta"], "sample": j["sample"], **s, "prompt_sha": r["prompt_sha"],
                                "completion_tokens": (r.get("usage") or {}).get("completion"),
                                "reply_tail": r["reply"][-300:]}) + "\n")
    os.replace(tmp, path)
    print(json.dumps({"model": args.model, "condition": args.condition, "calls": len(jobs),
                      "format_compliance": round(parsed / len(jobs), 3), "out": str(path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
