"""Figures for the paper and poster, drawn from the saved result files (no model calls, no refits).

  python experiments/make_figures.py      ->  results/figures/fig*.pdf and .png

The paper's four figures and one supplementary figure:
  fig1_overview             the task, the three studies and how the prediction test is built (a schematic)
  fig2_humans_vs_models     B1 and B1b, one row per agent: the human signature effect, histories that leave one
                            code, and the bounded model's advantage on free play against the 39 people's
  fig3_evidence_use         B2/B3 (top): judgment errors by number of clues, process models on held-out families;
                            C-a (bottom): what the structured aid changes, the noise of evidence use by prompt
  fig4_prediction_test      C-c and C-b: calibration of the predicted change in usefulness, MSPE of each rival minus
                            the support map, choice shift against the measured change
  figS1_human_reproduction  Study A: E1 guesses after the supplied history, per-person AIC differences, held-out people

Intervals that only the figures show are recomputed here from the records, resampling the unit of the study:
people for Study A, state families for B3 and C. Every recomputed point estimate is checked against the result
file it comes from. Study A needs the human data (data/human/README.md).
"""
import glob
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)  # font-subsetting notices, not errors
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "experiments/study_a", ROOT / "experiments/study_b", ROOT / "experiments/study_c"):
    sys.path.insert(0, str(p))
import fit_process_models as F  # noqa: E402
import prediction_test as P  # noqa: E402
import primary_tests as PT  # noqa: E402
from mastermind.engine import get_game  # noqa: E402

OUT = ROOT / "results/figures"
R = ROOT / "results"
N_BOOT = 10000

# Colors: categorical slots in fixed order (validated for color-vision deficiency on white), ink and chrome.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, PANEL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f6f5f2"
PEOPLE, MODEL = S2, S1          # the same entity keeps its color in every figure
FULL_WIDTH, HEIGHT = 6.9, 2.45  # inches; the full width of a two-column page

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7.5,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.cal": "Arial", "mathtext.sf": "Arial", "mathtext.tt": "Arial",
    "axes.labelsize": 7.5, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.fontsize": 6.8, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "lines.solid_capstyle": "round",
})


# ---------------------------------------------------------------- helpers
def load(path):
    return json.loads((R / path).read_text())


def grid(ax, axis):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def zero_line(ax, vertical=True, at=0.0):
    (ax.axvline if vertical else ax.axhline)(at, color=AXIS, linewidth=0.8, zorder=1)


def point(ax, x, y, color, hollow=False, marker="o", size=4.2, xerr=None, yerr=None):
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, markersize=size, color=color, ecolor=color,
                elinewidth=1.0, capsize=0, markeredgewidth=1.0,
                markerfacecolor="white" if hollow else color, zorder=3)


def ci_err(est, ci):
    return [[est - ci[0]], [ci[1] - est]]


def minus(text):
    """Typographic minus signs in annotation text."""
    return text.replace("-", "−")


def note(ax, x, y, text, ha="left", va="top", **kw):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6.5, color=INK2, **kw)


def group_rule(ax, y, label):
    """A hairline between groups of rows, with the next group's name above it."""
    ax.axhline(y, color=GRID, linewidth=0.8, zorder=1)
    if label:
        ax.text(1.0, y + 0.06, label, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.3,
                color=MUTED, style="italic")


def cluster_boot(units, n_boot=N_BOOT, seed=0):
    """Resample the units (people, families) with replacement: an index array [n_boot, n_units]."""
    return np.random.default_rng(seed).integers(len(units), size=(n_boot, len(units)))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})  # reproducible bytes
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- overview (schematic)
DIGIT = {1: S1, 2: S2, 3: S3, 4: S4}


def peg(ax, x, y, d, r=0.115):
    ax.add_patch(Circle((x, y), r, facecolor=DIGIT[d], edgecolor="white", linewidth=0.8, zorder=3))
    ax.text(x, y - 0.004, str(d), ha="center", va="center", fontsize=6.3, fontweight="bold",
            color=INK if d == 4 else "white", zorder=4)


def feedback(ax, x, y, black, white, r=0.034, gap=0.105):
    spots = [(x, y + gap / 2), (x + gap, y + gap / 2), (x, y - gap / 2), (x + gap, y - gap / 2)]
    for i, (px, py) in enumerate(spots):
        if i < black:
            ax.add_patch(Circle((px, py), r, facecolor=INK, edgecolor=INK, linewidth=0.5, zorder=3))
        elif i < black + white:
            ax.add_patch(Circle((px, py), r, facecolor="white", edgecolor=INK, linewidth=0.6, zorder=3))
        else:
            ax.add_patch(Circle((px, py), r * 0.45, facecolor=AXIS, edgecolor="none", zorder=3))


def box(ax, x, y, w, h, title, lines, accent):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", facecolor=PANEL,
                                edgecolor=GRID, linewidth=0.8, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), 0.035, h, boxstyle="square,pad=0", facecolor=accent, edgecolor="none", zorder=2))
    ax.text(x + 0.11, y + h - 0.08, title, ha="left", va="top", fontsize=7.2, fontweight="bold", color=INK)
    ax.text(x + 0.11, y + h - 0.25, lines, ha="left", va="top", fontsize=6.3, color=INK2, linespacing=1.3)


def arrow(ax, x, y0, y1, label):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>,head_length=2.6,head_width=1.6",
                                 color=MUTED, linewidth=0.9, zorder=2, shrinkA=0, shrinkB=0))
    ax.text(x + 0.07, (y0 + y1) / 2, label, ha="left", va="center", fontsize=6.1, color=MUTED, style="italic")


def fig1_overview():
    """One canvas in inches, so every mark and label is placed at its printed size."""
    mm = get_game(4, 4)
    secret = mm.index("2143")
    W, H, B = FULL_WIDTH, 2.9, 0.24  # B: empty margin cut from the bottom
    fig = plt.figure(figsize=(W, H - B))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(B, H)
    ax.set_aspect("equal")
    ax.axis("off")
    title = dict(fontsize=8.5, fontweight="bold", color=INK, va="top", ha="left")

    # A: the task
    ax.text(0.02, H - 0.05, "A  The task", **title)
    ax.text(0.04, 2.47, "Supplied guesses and feedback", fontsize=6.2, color=MUTED, style="italic", va="center")
    for i, g in enumerate(["1122", "3141"]):
        y = 2.2 - i * 0.33
        black, white = mm.decode(mm.table[mm.index(g), secret])
        for j, ch in enumerate(g):
            peg(ax, 0.17 + j * 0.28, y, int(ch))
        feedback(ax, 1.33, y, black, white)
    y = 2.2 - 2 * 0.33
    for j in range(4):
        ax.add_patch(Circle((0.17 + j * 0.28, y), 0.1, facecolor="white", edgecolor=AXIS, linewidth=0.8, zorder=3))
        ax.text(0.17 + j * 0.28, y - 0.004, "?", ha="center", va="center", fontsize=6.3, color=MUTED, fontweight="bold")
    ax.text(1.3, y, "next guess", ha="left", va="center", fontsize=6.1, color=MUTED)
    for k, (fill, lab) in enumerate([(INK, "right color, right place"), ("white", "right color, wrong place")]):
        yy = 1.13 - k * 0.19
        ax.add_patch(Circle((0.1, yy), 0.034, facecolor=fill, edgecolor=INK, linewidth=0.6))
        ax.text(0.2, yy, lab, va="center", fontsize=6.2, color=INK2)
    ax.text(0.04, 0.62, "4 positions, 4 colors: 256 codes,\nso the ideal observer is exact.",
            fontsize=6.2, color=INK2, va="top", linespacing=1.3)

    # B: the three studies
    ax.text(2.0, H - 0.05, "B  Three studies", **title)
    bx, bw, bh = 2.0, 2.78, 0.6
    tops = [2.55, 1.7, 0.85]
    box(ax, bx, tops[0] - bh, bw, bh, "Study A  ·  People",
        "Reproduce Experiments 1–3 (Cheyette et al., 2025)\nFit three human models; predict held-out people", PEOPLE)
    box(ax, bx, tops[1] - bh, bw, bh, "Study B  ·  Language models, same task",
        "B1 signature effect · B1b human models on free play\nB2/B3 process models of evidence use (Qwen3-32B)", MODEL)
    box(ax, bx, tops[2] - bh, bw, bh, "Study C  ·  A structured reasoning aid",
        "C-a evidence use · C-c predicted change in usefulness\n(preregistered) · C-b do the model's choices follow?", S3)
    arrow(ax, bx + 0.25, tops[0] - bh, tops[1], "same positions, models and fitting")
    arrow(ax, bx + 0.25, tops[1] - bh, tops[2], "fitted process model makes the predictions")

    # C: the prediction test
    cx0 = 5.0
    ax.text(cx0, H - 0.05, "C  The prediction test", **title)
    gx, cw, chh, gtop = 5.62, 0.62, 0.3, 2.28
    for j, lab in enumerate(["standard (T0)", "aid (T3)"]):
        ax.text(gx + j * cw + cw / 2, gtop + 0.05, lab, ha="center", va="bottom", fontsize=6.2, color=INK2)
    for i, q in enumerate(["1", "2"]):
        yy = gtop - (i + 1) * chh
        ax.text(gx - 0.05, yy + chh / 2, f"query $q_{q}$", ha="right", va="center", fontsize=6.3, color=INK2)
        for j, tt in enumerate(["0", "3"]):
            ax.add_patch(FancyBboxPatch((gx + j * cw + 0.02, yy + 0.02), cw - 0.04, chh - 0.04,
                                        boxstyle="round,pad=0,rounding_size=0.03", facecolor=PANEL, edgecolor=GRID, linewidth=0.8))
            ax.text(gx + j * cw + cw / 2, yy + chh / 2, f"$U(q_{q},T_{tt})$", ha="center", va="center", fontsize=6.6, color=INK)
    ax.text(cx0 + 0.02, 1.45, r"$\Gamma = [U(q_1,T_3) - U(q_2,T_3)]$", fontsize=7, color=INK, va="top")
    ax.text(cx0 + 0.02, 1.22, r"$\qquad - \,[U(q_1,T_0) - U(q_2,T_0)]$", fontsize=7, color=INK, va="top")
    ax.text(cx0 + 0.02, 0.88, "U: the chance the model names the secret\nafter seeing the query's answer.\n"
            "Γ was predicted for 360 pairs and locked\non OSF before any data were collected.",
            fontsize=6.2, color=INK2, va="top", linespacing=1.3)
    save(fig, "fig1_overview")


# ---------------------------------------------------------------- Study A
def figS1_human_reproduction():
    import e1_reproduction as E1
    mm = get_game(4, 4)
    e1 = load("study_a/e1_reproduction.json")["by_condition"]
    games = E1.load_games(mm)
    people = sorted({g["uniqueid"] for g in games})
    pidx = {u: i for i, u in enumerate(people)}
    boots = cluster_boot(people)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL_WIDTH, HEIGHT), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 0.95, 1.05]})
    conds = [("Simple", "Simple"), ("Max-EIG", "Max-EIG"), ("None", "None")]
    for i, (k, _) in enumerate(conds):
        tot, cnt = np.zeros(len(people)), np.zeros(len(people))
        for g in games:
            if g["condition"] == k and g["solved"]:
                tot[pidx[g["uniqueid"]]] += len(g["human"])
                cnt[pidx[g["uniqueid"]]] += 1
        est = tot.sum() / cnt.sum()
        assert round(est, 2) == e1[k]["human_guesses_solved_games"], (k, est)
        bs = tot[boots].sum(1) / cnt[boots].sum(1)
        point(a, i - 0.12, est, PEOPLE, yerr=ci_err(est, np.percentile(bs, [2.5, 97.5])))
        point(a, i + 0.12, e1[k]["ideal_guesses"], MUTED, hollow=True)
    a.set_xticks(range(3), ["Simple", "Max-EIG", "No\nhistory"])
    a.set_xlim(-0.5, 2.5)
    a.set_ylim(0, 9)
    a.set_ylabel("Guesses after the supplied history")
    a.set_title("A  Experiment 1")
    a.legend([plt.Line2D([], [], marker="o", color=PEOPLE, linestyle="", markersize=4.2),
              plt.Line2D([], [], marker="o", color=MUTED, markerfacecolor="white", linestyle="", markersize=4.2)],
             ["People (solved games)", "Ideal player"], loc="upper left")
    grid(a, "y")

    d = []
    for f in sorted(glob.glob(str(R / "study_a/e3_fits_converged/participant_*.json"))):
        x = json.loads(Path(f).read_text())["fits"]
        d.append((x["leaky"]["aic"] - x["bounded"]["aic"], x["unbounded"]["aic"] - x["bounded"]["aic"]))
    d = np.array(d)
    comp = load("study_a/e3_model_comparison_converged.json")
    assert abs(d[:, 0].mean() - comp["mean_dAIC_bounded_vs_leaky"]) < 0.05
    assert abs(d[:, 1].mean() - comp["mean_dAIC_bounded_vs_unbounded"]) < 0.05
    rng = np.random.default_rng(0)
    for i in range(2):
        b.scatter(i + rng.uniform(-0.13, 0.13, len(d)), d[:, i], s=10, facecolor=PEOPLE, edgecolor="white",
                  linewidth=0.4, alpha=0.85, zorder=3)
        b.plot([i - 0.24, i + 0.24], [d[:, i].mean()] * 2, color=INK, linewidth=1.4, zorder=4)
        b.text(i + 0.27, d[:, i].mean(), f"{d[:, i].mean():.1f}", va="center", fontsize=6.4, color=INK)
    zero_line(b, vertical=False)
    b.set_xticks([0, 1], ["Leaky", "Unbounded"])
    b.set_xlim(-0.5, 1.6)
    b.set_ylim(-20, 360)
    b.set_ylabel("AIC above the bounded model, per person")
    b.set_title("B  Experiment 3 fits")
    note(b, 0.02, 0.99, f"bounded best for {comp['best_model_counts']['bounded']} of {comp['participants']}")
    grid(b, "y")

    ho = load("study_a/e3_heldout_converged.json")
    rows = [("bounded_minus_leaky", "vs. leaky"), ("bounded_minus_unbounded", "vs. unbounded")]
    for y, (k, lab) in zip([1, 0], rows):
        v = ho[k]
        point(c, v["total"], y, PEOPLE, xerr=ci_err(v["total"], v["ci95"]))
        c.text(v["total"], y + 0.17, f"{v['participants_favoring_first']} of 39 people", ha="center", va="bottom",
               fontsize=6.3, color=INK2)
    zero_line(c)
    c.set_yticks([1, 0], [lab for _, lab in rows])
    c.set_ylim(-0.6, 1.7)
    c.set_xlabel("Bounded model's held-out log likelihood\nminus the rival's")
    c.set_title("C  Predicting new people")
    grid(c, "x")
    save(fig, "figS1_human_reproduction")


# ---------------------------------------------------------------- B1
def fig2_humans_vs_models():
    s = load("study_b/b1_summary.json")
    rows = [("People", "humans"), ("gpt-oss-120b", "gpt-oss-120b"), ("Claude Haiku 5.5", "claude-haiku-5-5"),
            ("Claude Opus 5.5", "claude-opus-5-5"), ("Qwen3-32B", "qwen3-32b"), ("Phi-4", "phi-4"),
            ("OLMo-2-32B", "olmo2-32b")]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL_WIDTH, HEIGHT + 0.25), sharey=True,
                                  gridspec_kw={"wspace": 0.1, "width_ratios": [1.15, 0.8, 1]})
    ys = np.arange(len(rows))[::-1]
    ideal = s["humans"]["ideal_effect"]
    for y, (label, key) in zip(ys, rows):
        d = s[key]
        kind = "people" if key == "humans" else "model" if d.get("eligible_for_inference") else "below"
        color, hollow = {"people": (PEOPLE, False), "model": (MODEL, False), "below": (MUTED, True)}[kind]
        if kind == "below":
            point(a, d["descriptive"]["maxeig_minus_simple"], y, color, hollow=True)
        else:
            est = d["signature_effect_maxeig_minus_simple"]
            point(a, est, y, color, xerr=ci_err(est, d["ci95"]))
        point(b, d["one_code_subset"]["mean_guesses"], y, color, hollow=hollow)
    for ax in (a, b, c):
        group_rule(ax, 2.5, "")
    a.text(2.32, 2.43, "below the solve-rate rule", ha="right", va="top", fontsize=6.3, color=MUTED, style="italic")
    zero_line(a)
    a.axvline(ideal, color=MUTED, linewidth=0.8, zorder=1)
    a.text(ideal, len(rows) - 0.42, "ideal player", ha="center", va="bottom", fontsize=6.4, color=INK2)
    a.text(2.35, len(rows) - 0.42, "human direction →", ha="right", va="bottom", fontsize=6.3, color=MUTED, style="italic")
    a.set_yticks(ys, [r[0] for r in rows])
    a.set_ylim(-0.6, len(rows) - 0.1)
    a.set_xlim(-1.25, 2.4)
    a.set_xlabel("Extra guesses after Max-EIG\nvs. Simple histories")
    a.set_title("A  The human signature effect")
    grid(a, "x")
    b.axvline(1, color=MUTED, linewidth=0.8, zorder=1)
    b.text(1.4, len(rows) - 0.42, "ideal", ha="left", va="bottom", fontsize=6.4, color=INK2)
    b.set_xlim(0, 16.5)
    b.set_xlabel("Guesses to solve\n(unsolved = 16)")
    b.set_title("B  One code left")
    b.tick_params(axis="y", length=0)
    grid(b, "x")
    # C: B1b. The bounded model's AIC advantage over the unbounded (ideal-observer) model per decision: each of the
    # 39 people (Study A, converged fits) and each model that passes the solve-rate rule (pooled fit, 95% interval
    # from 200 resamples of games).
    ppl = []
    for f in sorted(glob.glob(str(R / "study_a/e3_fits_converged/participant_*.json"))):
        x = json.loads(Path(f).read_text())
        ppl.append((x["fits"]["unbounded"]["aic"] - x["fits"]["bounded"]["aic"]) / x["decisions"])
    ppl = np.array(ppl)
    ref = load("study_b/b1b_gpt-oss-120b_T0_converged.json")["human_comparison"]["human_dAIC_per_decision_median"]
    assert abs(np.median(ppl) - ref["vs_unbounded"]) < 5e-4, (np.median(ppl), ref)
    c.scatter(ppl, ys[0] + np.random.default_rng(0).uniform(-0.22, 0.22, len(ppl)), s=9, facecolor=PEOPLE,
              edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
    c.plot([np.median(ppl)] * 2, [ys[0] - 0.32, ys[0] + 0.32], color=INK, linewidth=1.3, zorder=4)
    c.text(np.median(ppl), ys[0] + 0.36, "median", ha="center", va="bottom", fontsize=6.2, color=INK2)
    for y, (label, key) in zip(ys, rows):
        if key != "humans" and s[key].get("eligible_for_inference"):
            bb = load(f"study_b/b1b_{key}_T0_converged.json")
            e = bb["pooled"]["dAIC_per_decision"]["vs_unbounded"]
            point(c, e, y, MODEL, xerr=ci_err(e, bb["bootstrap_over_games"]["dAIC_per_decision_ci95"]["vs_unbounded"]))
    c.text(2.6, 1.0, "not fitted\n(below the rule)", ha="center", va="center", fontsize=6.3, color=MUTED, style="italic")
    zero_line(c)
    c.set_xlim(-0.25, 5.4)
    c.set_xlabel("Bounded model's AIC advantage over\nthe ideal observer, per decision")
    c.set_title("C  The human model on free play")
    c.tick_params(axis="y", length=0)
    grid(c, "x")
    handles = [plt.Line2D([], [], marker="o", color=PEOPLE, linestyle="", markersize=4.2),
               plt.Line2D([], [], marker="o", color=MODEL, linestyle="", markersize=4.2),
               plt.Line2D([], [], marker="o", color=MUTED, markerfacecolor="white", linestyle="", markersize=4.2)]
    fig.legend(handles, ["People", "Models that solve at least half their free-play games",
                         "Models below that rule (descriptive)"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.14), handletextpad=0.3, columnspacing=1.6)
    save(fig, "fig2_humans_vs_models")


# ---------------------------------------------------------------- B2/B3
def error_by_clues(model="qwen3-32b", condition="T0"):
    """Error rate by (kind, number of clues) with family-bootstrap intervals, as in the error profile."""
    counts = {}
    for r in F.load_records(model, condition):
        if r["split"] not in F.PROFILE_SPLITS or r["task"] not in ("classify", "classify_single") or not r.get("labels"):
            continue
        n_clues = len(r["history"])
        for code, kind in r["items"]:
            said = r["labels"].get(code)
            if said is None:
                continue
            err = (not said) if kind == "possible" else said
            c = counts.setdefault((kind, n_clues), {}).setdefault(r["family_id"], [0, 0])
            c[0] += err
            c[1] += 1
    out = {}
    for key, fam in counts.items():
        arr = np.array(list(fam.values()), float)
        boots = cluster_boot(fam)
        bs = arr[boots, 0].sum(1) / arr[boots, 1].sum(1)
        out[key] = (arr[:, 0].sum() / arr[:, 1].sum(), np.percentile(bs, [2.5, 97.5]), len(fam))
    return out


def panels_b3(a, b):
    prof = load("study_b/process_fits/qwen3-32b_error_profile_T0.json")["T0"]["by_clue_count"]
    est = error_by_clues()
    series = [("possible", "Still-possible code ruled out", S1, "o", -0.06),
              ("near", "Near miss kept", S2, "s", 0.0),
              ("far", "Far miss kept", S3, "D", 0.06)]
    for key, label, color, marker, dx in series:
        xs = sorted(n for (k, n) in est if k == key)
        ys = [est[(key, n)][0] for n in xs]
        for n, y in zip(xs, ys):
            assert round(y, 3) == prof[key][str(n)]["rate"], (key, n, y)
        lo = [est[(key, n)][0] - est[(key, n)][1][0] for n in xs]
        hi = [est[(key, n)][1][1] - est[(key, n)][0] for n in xs]
        a.plot(np.array(xs) + dx, ys, color=color, linewidth=1.5, zorder=3)
        a.errorbar(np.array(xs) + dx, ys, yerr=[lo, hi], fmt=marker, markersize=4.2, color=color, ecolor=color,
                   elinewidth=1.0, capsize=0, label=label, zorder=4)
    a.set_xticks([1, 2, 3])
    a.set_xlim(0.75, 3.25)
    a.set_ylim(0, 0.95)
    a.set_xlabel("Clues in the history")
    a.set_ylabel("Error rate")
    a.set_title("A  Judgment errors (standard prompt)")
    a.legend([plt.Line2D([], [], color=c, marker=m, linewidth=1.5, markersize=4.2) for _, _, c, m, _ in series],
             [lab for _, lab, _, _, _ in series], loc="upper left", handlelength=1.8)
    grid(a, "y")

    models, ll, boots, iv = PT.heldout_by_family()
    base = models.index("M0")
    names = {"M1": "M1 noisy evidence", "M2c": "M2c conservative (human)", "M2s": "M2s symmetric",
             "M3": "M3 forgetting", "M1a": "M1a two error rates"}
    order = ["M1", "M2c", "M2s", "M3", "M1a"]
    ys = np.array([4.2, 3.2, 2.2, 1.2, 0])
    for y, m in zip(ys, order):
        dif = ll[:, models.index(m)] - ll[:, base]
        sums = dif[boots].sum(1)
        e, ci = float(dif.sum()), np.percentile(sums, [2.5, 97.5])
        if m == "M2s":  # the reported interval of the best planned model against M0
            saved = iv["best_planned_minus_other_planned"]["M0"]
            assert round(e, 2) == saved["difference"] and ci.round(2).tolist() == saved["ci95"], (e, ci, saved)
        point(b, e, y, MUTED if m == "M1a" else MODEL, hollow=(m == "M1a"), xerr=ci_err(e, ci))
    group_rule(b, 0.6, "exploratory")
    zero_line(b)
    b.set_yticks(ys, [names[m] for m in order])
    b.set_ylim(-0.6, 4.8)
    b.set_xlabel("Held-out log likelihood minus M0 (exact + noise)")
    b.set_title("B  Process models, 80 held-out families")
    grid(b, "x")


# ---------------------------------------------------------------- C-a
def panels_ca(a, b):
    ca = load("study_c/ca_aid_effects.json")
    measures = [("false_exclusion_multi", "Possible code ruled out (multi-clue)"),
                ("false_exclusion_single", "Possible code ruled out (one clue)"),
                ("false_retention_near_multi", "Near miss kept (multi-clue)"),
                ("false_retention_far_multi", "Far miss kept (multi-clue)"),
                ("false_retention_near_single", "Near miss kept (one clue)"),
                ("identification", "Named code possible (finished replies)"),
                ("identify_unfinished", "Reply unfinished")]
    ys = np.array([7.6, 6.6, 5.6, 4.6, 3.6, 1.6, 0.6])
    for contrast, color, marker, dy in (("T3-T0", S1, "o", 0.15), ("T3-T1", S2, "s", -0.15)):
        for y, (m, _) in zip(ys, measures):
            cc = ca["contrasts"][contrast][m]
            point(a, cc["difference"], y + dy, color, marker=marker, size=3.8, xerr=ci_err(cc["difference"], cc["ci95"]))
    group_rule(a, 2.55, "naming a code")
    a.text(1.0, 8.25, "checking clues (errors)", transform=a.get_yaxis_transform(), ha="right", va="bottom",
           fontsize=6.3, color=MUTED, style="italic")
    zero_line(a)
    a.set_yticks(ys, [lab for _, lab in measures])
    a.set_ylim(0, 8.6)
    a.set_xlabel("Change in rate with the aid (95% CI)")
    a.set_title("C  What the structured aid changes")
    a.legend([plt.Line2D([], [], marker="o", color=S1, linestyle="", markersize=3.8),
              plt.Line2D([], [], marker="s", color=S2, linestyle="", markersize=3.8)],
             ["Aid minus standard prompt (T3 − T0)", "Aid minus length-matched prompt (T3 − T1)"],
             loc="upper center", ncol=1, bbox_to_anchor=(0.45, -0.17), handletextpad=0.3)
    grid(a, "x")

    conds = [("T0", "Standard"), ("T1", "Length-\nmatched"), ("T2", "Generic\ncheck"), ("T3", "Structured\naid")]
    for i, (cnd, _) in enumerate(conds):
        nus = [json.loads(Path(f).read_text())["fits"]["M2s"]["params"]["nu"]
               for f in glob.glob(str(R / f"study_b/process_fits/boot_qwen3-32b_{cnd}/rep_*.json"))]
        e = ca["parameters"][cnd]["M2s"]["nu"]
        lo, hi = np.percentile(nus, [2.5, 97.5])
        point(b, i, e, MODEL, yerr=[[e - lo], [hi - e]])
    sh = ca["parameter_shifts"]["T3-T0"]["M2s.nu"]
    note(b, 0.02, 0.99, minus(f"aid minus standard: +{sh['difference']:.2f} [{sh['ci95'][0]:.2f}, {sh['ci95'][1]:.2f}]"))
    b.set_xticks(range(len(conds)), [lab for _, lab in conds])
    b.set_xlim(-0.5, len(conds) - 0.5)
    b.set_ylim(0.3, 0.75)
    b.set_ylabel("Noise of evidence use (M2s)")
    b.set_title("D  Fitted noise by prompt")
    grid(b, "y")


def fig3_evidence_use():
    fig = plt.figure(figsize=(FULL_WIDTH, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1.2], wspace=0.7, hspace=0.55)
    a, b, c, d = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
    panels_b3(a, b)
    panels_ca(c, d)
    save(fig, "fig3_evidence_use")


# ---------------------------------------------------------------- C-c and C-b
def pair_data():
    """Per confirmation pair: family, stratum, predicted Gamma (support map), observed Gamma, choice shift."""
    lock = json.loads(P.LOCK.read_text())
    b0, c0, pairs = P.records("qwen3-32b", "T0", P.CONF_SPLIT, P.BASE_CAP)
    b3, c3, _ = P.records("qwen3-32b", "T3", P.CONF_SPLIT, P.BASE_CAP)
    gam = P.observed_gamma(P.group({"T0": b0, "T3": b3}), pairs)

    def logit(k, n):
        p = (k + 0.5) / (n + 1.0)
        return float(np.log(p / (1 - p)))
    rows = []
    for p in lock["predictions"]:
        pid = p["pair_id"]
        ch = {c: [r for r in rows_c if r["pair_id"] == pid and r["parsed"]] for c, rows_c in (("T0", c0), ("T3", c3))}
        k = {c: sum(r["chose"] == "q_simple" for r in ch[c]) for c in ch}
        rows.append({"family": p["family_id"], "stratum": p["stratum"], "pred": p["gamma_hat"][lock["primary"]],
                     "obs": gam[pid][0], "shift": logit(k["T3"], len(ch["T3"])) - logit(k["T0"], len(ch["T0"]))})
    return rows


def fig4_prediction_test():
    sc = load("study_c/cc_score.json")
    cb = load("study_c/cb_choice_shift_c_confirmation.json")
    rows = pair_data()
    pred, obs, shift = (np.array([r[k] for r in rows]) for k in ("pred", "obs", "shift"))
    fams = sorted({r["family"] for r in rows})
    fam_of = np.array([fams.index(r["family"]) for r in rows])
    members = [np.flatnonzero(fam_of == i) for i in range(len(fams))]
    boots = cluster_boot(fams, n_boot=2000)
    sel = [np.concatenate([members[i] for i in bt]) for bt in boots]
    cal = sc["by_predictor"]["support_map"]
    assert abs(np.polyfit(pred, obs, 1)[0] - cal["calibration_slope"]) < 1e-9
    assert abs(np.polyfit(obs, shift, 1)[0] - cb["primary_measured_gamma"]["slope"]) < 1e-9
    assert len(rows) == sc["pairs_scored"]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL_WIDTH, HEIGHT + 0.1),
                                  gridspec_kw={"wspace": 0.95, "width_ratios": [1, 0.92, 1]})
    # A: calibration. Pairs in four equal bins of predicted Gamma (90 pairs each); mean observed Gamma per bin, with
    # family-bootstrap intervals (bin membership fixed).
    bins = np.array_split(np.argsort(pred, kind="stable"), 4)
    lim = max(pred.max(), 0.012) * 1.08
    a.plot([0, lim], [0, lim], color=INK2, linewidth=0.9, zorder=2)
    a.text(lim * 0.98, lim * 0.98 + 0.003, "perfect prediction", ha="right", va="bottom", fontsize=6.2, color=INK2, rotation=0)
    for idx in bins:
        mx, my = pred[idx].mean(), obs[idx].mean()
        inbin = np.zeros(len(rows), bool)
        inbin[idx] = True
        bs = [obs[s][inbin[s]].mean() for s in sel if inbin[s].any()]
        point(a, mx, my, MODEL, yerr=ci_err(my, np.percentile(bs, [2.5, 97.5])))
    zero_line(a, vertical=False)
    a.set_xlim(-0.0005, lim)
    a.set_ylim(-0.05, 0.05)
    a.set_xlabel("Predicted Γ (support map)")
    a.set_ylabel("Observed Γ (mean of 90 pairs)")
    a.set_title("A  Calibration")
    note(a, 0.03, 0.99, minus(f"slope, all 360 pairs: {cal['calibration_slope']:.2f}\n"
                              f"[{cal['calibration_slope_ci95'][0]:.2f}, {cal['calibration_slope_ci95'][1]:.2f}]"))
    grid(a, "both")

    rivals = [("difficulty_regression", "Difficulty regression"), ("reporting_only", "Reporting only"),
              ("uniform_gain", "Uniform gain (Γ = 0)"), ("ideal_noise", "Ideal + noise")]
    ys = np.arange(len(rivals))[::-1]
    for y, (k, _) in zip(ys, rivals):
        v = sc["primary_vs_rivals"][k]
        e, ci = v["mspe_rival_minus_model"] * 1e4, [x * 1e4 for x in v["ci95"]]
        point(b, e, y, MODEL, xerr=ci_err(e, ci))
    zero_line(b)
    b.set_yticks(ys, [lab for _, lab in rivals])
    b.set_ylim(-0.6, len(rivals) - 0.15)
    b.set_xlabel("MSPE(rival) − MSPE(support map), $\\times 10^{-4}$")
    b.set_title("B  H3: support map vs. rivals")
    note(b, 0.99, 0.99, "above 0 favors\nthe support map", ha="right")
    grid(b, "x")

    c.scatter(obs, shift, s=8, facecolor=MODEL, edgecolor="white", linewidth=0.35, alpha=0.55, zorder=3)
    xs = np.linspace(obs.min(), obs.max(), 50)
    k1, k0 = np.polyfit(obs, shift, 1)
    lines = np.array([np.polyval(np.polyfit(obs[s], shift[s], 1), xs) for s in sel])
    c.fill_between(xs, *np.percentile(lines, [2.5, 97.5], axis=0), color=INK, alpha=0.12, linewidth=0, zorder=2)
    c.plot(xs, k0 + k1 * xs, color=INK, linewidth=1.2, zorder=4)
    zero_line(c)
    zero_line(c, vertical=False)
    pm = cb["primary_measured_gamma"]
    note(c, 0.02, 0.99, minus(f"slope {pm['slope']:.2f}\n[{pm['ci95'][0]:.2f}, {pm['ci95'][1]:.2f}]"))
    c.set_xlabel("Observed Γ")
    c.set_ylabel("Choice shift (Δ logit, aid − standard)")
    c.set_title("C  H4: choices vs. Γ")
    grid(c, "both")
    save(fig, "fig4_prediction_test")


if __name__ == "__main__":
    for f in (fig1_overview, fig2_humans_vs_models, fig3_evidence_use, fig4_prediction_test, figS1_human_reproduction):
        f()
