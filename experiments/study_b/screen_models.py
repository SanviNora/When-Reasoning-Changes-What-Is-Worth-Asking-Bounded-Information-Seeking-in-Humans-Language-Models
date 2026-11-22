"""Screening of a candidate model for the panel, on 20 development states (T0).

Three short tasks per state, 2 samples each:
  classify  label 10 codes as still possible / ruled out (balanced accuracy)
  identify  name one code that could still be the secret (correct if it is in the survivor set)
  guess     next guess on E1 starting positions (valid code; consistent with the feedback)
Floor rule (fixed before any B1 run): format compliance >= 90% and identification above chance.

Run inside a job where the model is served:  python experiments/study_b/screen_models.py --model qwen3-32b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import prompts  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from mastermind.llm import Client  # noqa: E402

OUT = ROOT / "results/study_b/screening"


def dev_states(mm, n=20, seed=20261117):
    """Random histories of 1-3 guesses leaving 4-16 codes (development set only)."""
    rng = np.random.default_rng(seed)
    states = []
    while len(states) < n:
        secret = int(rng.integers(mm.N))
        hist = []
        for _ in range(int(rng.integers(1, 4))):
            q = int(rng.integers(mm.N))
            hist.append((q, mm.decode(mm.table[q, secret])))
        k = int(mm.consistent(hist).sum())
        if 4 <= k <= 16:
            states.append(hist)
    return states


def classify_codes(mm, hist, rng, n_pos=5, n_neg=5):
    mask = mm.consistent(hist)
    pos = rng.choice(np.flatnonzero(mask), size=min(n_pos, mask.sum()), replace=False)
    neg = rng.choice(np.flatnonzero(~mask), size=n_neg, replace=False)
    codes = [int(i) for i in np.concatenate([pos, neg])]
    rng.shuffle(codes)
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    mm = get_game(4, 4)
    rng = np.random.default_rng(0)
    client = Client(args.model)
    states = dev_states(mm)
    stimuli = json.loads((ROOT / "data/stimuli/e1_stimuli.json").read_text())[::13][:10]

    jobs, meta = [], []
    for sid, hist in enumerate(states):
        codes = classify_codes(mm, hist, rng)
        for task, fields in (("classify", {"codes": codes}), ("identify", {})):
            prompt, th = prompts.build(mm, task, "T0", hist, **fields)
            for s in range(args.samples):
                jobs.append((prompt, s))
                meta.append({"task": task, "state": sid, "history": hist, "codes": codes, "template": th})
    for st in stimuli:
        hist = [(mm.index(g["guess"]), (g["black"], g["white"])) for g in st["supplied"]]
        prompt, th = prompts.build(mm, "guess", "T0", hist, n_supplied=len(hist))
        for s in range(args.samples):
            jobs.append((prompt, s))
            meta.append({"task": "guess", "state": st["stimulus_id"], "history": hist, "codes": None, "template": th})

    recs = client.batch(jobs, workers=args.workers)

    rows = []
    for m, r in zip(meta, recs):
        mask = mm.consistent(m["history"]) if m["history"] else np.ones(mm.N, bool)
        ans = prompts.parse(mm, m["task"], r["reply"], m["codes"])
        row = {"task": m["task"], "state": m["state"], "K": int(mask.sum()), "parsed": ans is not None}
        if m["task"] == "classify":
            truth = {i: bool(mask[i]) for i in m["codes"]}
            got = ans or {}
            pos = [i for i in truth if truth[i]]
            neg = [i for i in truth if not truth[i]]
            row["balanced_acc"] = float(np.mean([np.mean([got.get(i) is True for i in pos]),
                                                 np.mean([got.get(i) is False for i in neg])]))
            row["complete"] = len(got) == len(truth)
        elif m["task"] == "identify":
            row["correct"] = ans is not None and bool(mask[ans])
            row["chance"] = row["K"] / mm.N
        else:
            row["consistent"] = ans is not None and bool(mask[ans])
        row["completion_tokens"] = (r.get("usage") or {}).get("completion")
        rows.append(row)
        r.update({k: v for k, v in m.items() if k != "history"})
        r.update({"history": [[mm.to_str(q), b, w] for q, (b, w) in m["history"]], "parsed_ok": row["parsed"]})

    def rate(task, key):
        v = [x[key] for x in rows if x["task"] == task]
        return round(float(np.mean(v)), 3)

    ident = [x for x in rows if x["task"] == "identify"]
    summary = {
        "model": args.model,
        "calls": len(rows),
        "format_compliance": round(float(np.mean([x["parsed"] for x in rows])), 3),
        "format_by_task": {t: rate(t, "parsed") for t in ("classify", "identify", "guess")},
        "classify_balanced_accuracy": rate("classify", "balanced_acc"),
        "identify_accuracy": rate("identify", "correct"),
        "identify_chance": round(float(np.mean([x["chance"] for x in ident])), 3),
        "guess_consistent_rate": rate("guess", "consistent"),
        "median_completion_tokens": float(np.median([x["completion_tokens"] or 0 for x in rows])),
    }
    summary["passes_floor"] = summary["format_compliance"] >= 0.9 and summary["identify_accuracy"] > summary["identify_chance"]

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{args.model}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    (OUT / f"{args.model}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
