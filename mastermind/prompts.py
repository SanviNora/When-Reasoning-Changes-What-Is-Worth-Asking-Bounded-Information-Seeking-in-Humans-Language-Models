"""Prompt templates for every task and condition.

One shared rules text, fixed feedback wording ("black B, white W"), and one template per
task x condition. Every reply ends with one line of JSON; free-text reasoning before it is
allowed in every condition. Templates are hashed into each record, and prompts never contain
the secret, the codes still possible, EIG, V* or scores.

Tasks:  guess     next guess in a game          {"guess": "1234"}
        classify  which listed codes are still possible   {"1234": "possible", ...}
        identify  name one code that could be the secret  {"code": "1234"}
        choice    which of two guesses to make next       {"choice": "A"}
Conditions:  T0 standard, T1 length-matched neutral text, T2 generic "check your work",
             T3 structured checking, T4 = T0 text with the model's native thinking mode on.
"""
import hashlib
import json
import re

RULES = (
    "We are playing Mastermind. The secret code has {P} positions, each a digit from 1 to {C}, "
    "and digits may repeat. After each guess the player is told two numbers:\n"
    "- black: how many digits are correct AND in the correct position\n"
    "- white: how many digits are in the code but in the wrong position\n"
    "Each digit of the secret is counted at most once across black and white. "
    "For example, if the secret is 1122 and the guess is 1213, the feedback is black 1, white 2."
)

GUIDANCE = {
    "T0": "",
    "T1": (
        "The information above describes the game so far. The guesses are listed in the order in which "
        "they were made, and each line shows the guess followed by the feedback it received. The question "
        "below uses the same rules that were described at the start of this message, and the required "
        "format of your answer is given at the end of the question."
    ),
    "T2": "Reason carefully and check your work before you answer.",
    "T3": (
        "Before you answer, check each candidate code against every piece of feedback, one guess at a time: "
        "(1) count the digits in the right position (black); (2) count the digits the two codes share in total, "
        "counting each digit at most once; (3) the white count is the shared total minus black; "
        "(4) keep a code only if both counts match the feedback for every guess."
    ),
    "T4": "",
}

TASKS = {
    "guess": (
        "What is your next guess? You may reason first. "
        'End your reply with one line of JSON, for example {{"guess": "{ex}"}}.'
    ),
    "classify": (
        "Using ALL the feedback above, decide for each code below whether it could still be the secret "
        "code (possible) or is ruled out (impossible).\nCodes: {codes}\n"
        "You may reason first. End your reply with one line of JSON that maps every listed code to "
        '"possible" or "impossible", in the form {{"code": "possible", "code": "impossible", ...}} '
        "with the listed codes as keys."
    ),
    "identify": (
        "Name one code that could still be the secret code, given ALL the feedback above. "
        'You may reason first. End your reply with one line of JSON, for example {{"code": "{ex}"}}.'
    ),
    "choice": (
        "You can make one of these two guesses next:\nA: {a}\nB: {b}\n"
        "Which guess would you make? You may reason first. "
        'End your reply with one line of JSON, for example {{"choice": "A"}}.'
    ),
}


def board(mm, history, n_supplied=0):
    lines = []
    if n_supplied:
        lines.append(f"The first {n_supplied} guess{'es were' if n_supplied > 1 else ' was'} made for you.")
    lines.append("Guesses so far:" if history else "No guesses have been made yet.")
    for t, (q, (b, w)) in enumerate(history, 1):
        lines.append(f"{t}. guess {mm.to_str(q)} -> black {b}, white {w}")
    return "\n".join(lines)


def build(mm, task, condition, history, n_supplied=0, **fields):
    """Return (prompt, template_hash). `fields` fill the task template (codes, a, b)."""
    parts = [RULES.format(P=mm.P, C=mm.C), board(mm, history, n_supplied)]
    if GUIDANCE[condition]:
        parts.append(GUIDANCE[condition])
    example = "1234"[: mm.P]
    if task == "classify":
        codes = [mm.to_str(i) for i in fields["codes"]]
        text = TASKS[task].format(codes=", ".join(codes))
    elif task == "choice":
        text = TASKS[task].format(a=mm.to_str(fields["a"]), b=mm.to_str(fields["b"]))
    else:
        text = TASKS[task].format(ex=example)
    parts.append(text)
    return "\n\n".join(parts), template_hash(task, condition)


def template_hash(task, condition):
    blob = json.dumps([RULES, GUIDANCE[condition], TASKS[task]])
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---- parsing ------------------------------------------------------------
def _last_json(text):
    for blob in reversed(re.findall(r"\{[^{}]*\}", text or "", flags=re.S)):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    return None


def _code(mm, value):
    s = re.sub(r"\D", "", str(value))
    if len(s) == mm.P and all(1 <= int(ch) <= mm.C for ch in s):
        return mm.index(s)
    return None


def parse(mm, task, text, codes=None):
    """Parsed answer, or None when the reply has no valid answer.
    guess/identify -> code index; choice -> "A"/"B"; classify -> {code_idx: bool} for listed codes."""
    d = _last_json(text)
    if not isinstance(d, dict):
        return None
    if task in ("guess", "identify"):
        return _code(mm, d.get("guess" if task == "guess" else "code"))
    if task == "choice":
        v = str(d.get("choice", "")).strip().upper()
        return v if v in ("A", "B") else None
    if task == "classify":
        out = {}
        for k, v in d.items():
            i = _code(mm, k)
            v = str(v).lower()
            if i is not None and i in set(codes) and v.startswith(("possible", "impossible")):
                out[i] = v.startswith("possible")
        return out or None
    raise ValueError(task)
