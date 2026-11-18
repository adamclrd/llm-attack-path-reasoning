"""
Builds every figure of the paper, without titles (the captions carry the
explanation), as PNG at 300 dpi plus a PDF copy.

    python make_figures.py --outdir figures

File names follow the figure numbers of the paper. Two panels shown side by
side get a letter:

    fig01      the three architectures, with the one evaluated here framed
    fig02a/b   C1 (single prompt) and C2 (agent with symbolic verification)
    fig03      how one instance is generated, checked and scored
    fig04      effective branching factor, read from scenarios_v2.json
    fig05      chain discovery rate against chain depth, main grid
    fig06      chain discovery rate against distractor count, main grid
    fig07      chain discovery rate on the low grid
    fig08a/b   C1 against C2 on the 32 paired instances

Figures 5 to 8 are drawn from the counts reported in the result tables of the
paper (RESULTS below), so they always match the tables. Error bars are 95%
Wilson score intervals computed from the same counts.
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

INK = "#1b1b1b"
GREY = "#8a8a8a"
LLM = "#c9dcf0"
SYM = "#f6dfc0"
DATA = "#e2e2e2"
OUT = "#d7ecd9"
FRAME = "#d62728"

# series colours, identical to the earlier version of the figures
C_G31 = "#1f77b4"    # gemini-3.1-flash-lite
C_G35 = "#ff7f0e"    # gemini-3.5-flash-lite
C_B1 = "#2ca02c"     # breadth-first search
C_B3 = "#d62728"     # random choice
C_QWEN = "#9467bd"   # qwen2.5:14b

# ------------------------------------------------------------------ results
# Counts (successes, runs) from the result tables of the revised paper, which
# analysis/tables.py computes from logs/final.csv; each grid uses only the
# instances of its own scenario file. B3 is given as a rate (five runs per instance).

RESULTS = {
    "by_depth": {                       # main grid, pooled over distractors
        "depths": [4, 5, 6, 7],
        "gemini-3.1-flash-lite": [(40, 40), (36, 40), (33, 40), (29, 40)],
        "gemini-3.5-flash-lite": [(37, 40), (26, 40), (29, 40), (25, 40)],
        "B3": [9.5, 3.0, 2.0, 1.5],
    },
    "by_distractors": {                 # main grid, pooled over depths
        "k": [20, 30, 40, 50, 60],
        "gemini-3.1-flash-lite": [(31, 32), (27, 32), (27, 32), (27, 32),
                                  (26, 32)],
        "gemini-3.5-flash-lite": [(26, 32), (28, 32), (26, 32), (22, 32),
                                  (15, 32)],
        "B3": [6.9, 6.2, 3.8, 1.9, 1.2],
    },
    "low_grid": {                       # depths 2 to 4, pooled over depths
        "k": [5, 10, 20],
        "qwen2.5:14b": [(18, 24), (20, 24), (13, 24)],
        "gemini-3.5-flash-lite": [(24, 24), (24, 24), (23, 24)],
        "B3": [67.5, 56.7, 36.7],
    },
    "paired": {                         # 32 paired instances, C1 against C2
        "discovery": {"C1": (25, 32), "C2": (27, 32)},
        # inapplicable steps per instance: proposed by the model, and left
        # in the chain that is returned (C2 rejects them before they enter)
        "proposed": {"C1": 0.84, "C2": 0.56},
        "returned": {"C1": 0.84, "C2": 0.0},
    },
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


# ------------------------------------------------------------------ helpers

def box(ax, x, y, w, h, text, fc, fontsize=8.5, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                linewidth=0.9, edgecolor=INK, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, linespacing=1.35)


def arrow(ax, p, q, style="-|>", ls="-", color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, linestyle=ls,
                                 color=color, linewidth=0.9,
                                 mutation_scale=9,
                                 connectionstyle=f"arc3,rad={rad}"))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}.png and {path}.pdf")


def rate_axes(ax, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Chain discovery rate (%)")
    ax.set_ylim(-4, 104)
    ax.grid(alpha=0.3)


def model_line(ax, xs, counts, fmt, color, label):
    vals = [wilson(k, n) for k, n in counts]
    ys = [v[0] for v in vals]
    err = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
    ax.errorbar(xs, ys, yerr=err, fmt=fmt, color=color, label=label,
                markersize=5, linewidth=1.3, elinewidth=0.8, capsize=2.5,
                alpha=0.95)


# ------------------------------------------------------------------ figure 1

def solution_approach(path):
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    blank(ax, (0, 11.4), (0, 6.2))

    rows = [
        ("A. Symbolic", 4.5, "Manual\ntranslator", SYM, "Planner", SYM, True,
         "existing work:\nMulVAL, ChainReactor"),
        ("B. Hybrid", 2.75, "LLM", LLM, "Planner", SYM, True,
         "configuration C4\n(specified, not run)"),
        ("C. Generative", 1.0, "LLM", LLM, "LLM", LLM, False,
         "configurations C1, C2"),
    ]

    ax.text(0.1, 5.9, "Environment\n(raw host and network data)",
            fontsize=8, ha="left", va="center", color=GREY)
    ax.text(3.1, 5.9, "model construction", fontsize=8, ha="center",
            color=GREY, style="italic")
    ax.text(5.5, 5.9, "formal model", fontsize=8, ha="center",
            color=GREY, style="italic")
    ax.text(7.7, 5.9, "search", fontsize=8, ha="center", color=GREY,
            style="italic")
    ax.text(9.85, 5.9, "output", fontsize=8, ha="center", color=GREY,
            style="italic")

    for label, y, mc, mc_c, srch, srch_c, formal, note in rows:
        ax.text(0.1, y + 0.42, label, fontsize=9.5, weight="bold", va="center")
        ax.text(0.1, y - 0.08, note, fontsize=7.2, va="center", color=GREY,
                linespacing=1.4)
        box(ax, 2.4, y - 0.32, 1.4, 0.72, mc, mc_c)
        if formal:
            box(ax, 4.8, y - 0.32, 1.4, 0.72, "PDDL /\nDatalog", DATA)
            arrow(ax, (3.8, y + 0.04), (4.8, y + 0.04))
            arrow(ax, (6.2, y + 0.04), (7.0, y + 0.04))
        else:
            ax.text(5.5, y + 0.32, "none", fontsize=8, ha="center",
                    color=GREY, style="italic")
            arrow(ax, (3.8, y + 0.04), (7.0, y + 0.04), ls=(0, (4, 2)))
        box(ax, 7.0, y - 0.32, 1.4, 0.72, srch, srch_c)
        box(ax, 9.1, y - 0.32, 1.5, 0.72, "attack\nchain", OUT)
        arrow(ax, (8.4, y + 0.04), (9.1, y + 0.04))
        arrow(ax, (2.1, y + 0.04), (2.4, y + 0.04))

    ax.plot([2.05, 2.05], [0.55, 5.5], color=GREY, linewidth=0.7)

    # the architecture evaluated in this study
    ax.add_patch(Rectangle((0.02, 0.5), 10.78, 1.27, fill=False,
                           edgecolor=FRAME, linewidth=1.6))
    ax.text(10.74, 1.62, "evaluated in this study", fontsize=7.6,
            color=FRAME, ha="right", va="center", weight="bold")

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                   markerfacecolor=LLM, markeredgecolor=INK,
                   label="language model"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                   markerfacecolor=SYM, markeredgecolor=INK,
                   label="symbolic component"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
              fontsize=8, bbox_to_anchor=(0.55, -0.10))
    save(fig, path)


# ------------------------------------------------------------------ figure 2

def c1_panel(path):
    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    blank(ax, (0.2, 4.6), (0.55, 4.25))
    box(ax, 0.35, 3.25, 1.8, 0.75, "scenario\n(state, vulns, goal)", DATA)
    box(ax, 2.6, 3.25, 1.8, 0.75, "language model", LLM)
    box(ax, 1.45, 1.9, 1.8, 0.75, "full chain", OUT)
    box(ax, 1.45, 0.7, 1.8, 0.75, "verifier\n(scoring only)", SYM)
    arrow(ax, (2.15, 3.63), (2.6, 3.63))
    arrow(ax, (3.5, 3.25), (2.7, 2.65), rad=-0.15)
    arrow(ax, (2.35, 1.9), (2.35, 1.45))
    save(fig, path)


def c2_panel(path):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    blank(ax, (5.45, 10.95), (0.55, 4.25))
    box(ax, 5.65, 3.25, 1.9, 0.75, "current state\n(held by verifier)", DATA)
    box(ax, 8.0, 3.25, 1.8, 0.75, "language model", LLM)
    box(ax, 8.0, 1.9, 1.8, 0.75, "one proposed\nvulnerability", OUT)
    box(ax, 5.65, 1.9, 1.9, 0.75, "verifier\napplies or rejects", SYM)

    arrow(ax, (7.55, 3.63), (8.0, 3.63))
    arrow(ax, (8.9, 3.25), (8.9, 2.65))
    arrow(ax, (8.0, 2.27), (7.55, 2.27))
    arrow(ax, (6.6, 2.65), (6.6, 3.25))
    ax.text(6.5, 2.95, "accepted:\neffects applied", fontsize=7, ha="right",
            va="center", color=GREY)

    # rejection loop, routed outside the boxes so it crosses nothing
    ax.plot([6.6, 6.6, 10.55, 10.55], [1.9, 1.35, 1.35, 3.63],
            color=GREY, linewidth=0.9, linestyle=(0, (3, 2)))
    arrow(ax, (10.55, 3.63), (9.85, 3.63), color=GREY, ls=(0, (3, 2)))
    ax.text(8.6, 1.16, "rejected: the missing preconditions are returned",
            fontsize=7, ha="center", va="top", color=GREY)
    save(fig, path)


# ------------------------------------------------------------------ figure 3

def pipeline(path):
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    blank(ax, (0, 11.0), (0.2, 5.5))

    # ------------------------------------------------ generation, top row
    gy, gh = 4.55, 0.8
    gen = [(0.2, 1.9, "draw reference\nchain of depth d", DATA),
           (2.35, 1.9, "add k distractors\nT1 T2 T3 T4", DATA),
           (4.5, 1.9, "BFS depth check\nshortest = d ?", SYM),
           (6.65, 1.9, "replace identifiers\nwith opaque tokens", DATA)]
    for x, w, t, c in gen:
        box(ax, x, gy, w, gh, t, c)
    for i in range(len(gen) - 1):
        arrow(ax, (gen[i][0] + gen[i][1], gy + gh / 2),
              (gen[i + 1][0], gy + gh / 2))
    box(ax, 8.8, gy, 1.9, gh, "instance\n+ metadata", OUT)
    arrow(ax, (8.55, gy + gh / 2), (8.8, gy + gh / 2))

    # rejection loop, routed below the top row
    ax.plot([5.45, 5.45, 3.3, 3.3], [gy, gy - 0.42, gy - 0.42, gy],
            color=GREY, linewidth=0.9, linestyle=(0, (3, 2)))
    arrow(ax, (3.3, gy - 0.1), (3.3, gy), color=GREY)
    ax.text(4.38, gy - 0.55, "shorter route found: discard and redraw",
            fontsize=7.2, ha="center", va="top", color=GREY, style="italic")

    # ------------------------------------------------ evaluation, bottom row
    sy, sh = 1.85, 0.85
    bus_in, bus_out = 3.20, 1.30
    sysb = [(0.35, 1.95, "C1\nsingle prompt", LLM),
            (2.60, 1.95, "C2\nagent + verifier", LLM),
            (4.85, 1.95, "B1\nbreadth-first", SYM),
            (7.10, 1.95, "B3\nrandom choice", SYM)]

    ax.plot([9.75, 9.75], [gy, bus_in], color=GREY, linewidth=0.9)
    ax.plot([1.325, 9.75], [bus_in, bus_in], color=GREY, linewidth=0.9)
    for x, w, t, c in sysb:
        box(ax, x, sy, w, sh, t, c)
        cx = x + w / 2
        arrow(ax, (cx, bus_in), (cx, sy + sh), color=GREY)
        ax.plot([cx, cx], [sy, bus_out], color=GREY, linewidth=0.9)

    ax.plot([1.325, 9.75], [bus_out, bus_out], color=GREY, linewidth=0.9)
    box(ax, 8.8, 0.30, 1.9, 0.8, "symbolic verifier\nand metrics", SYM)
    arrow(ax, (9.75, bus_out), (9.75, 1.10), color=GREY)

    ax.text(1.0, 3.42, "every system sees the same instance", fontsize=7.2,
            ha="left", color=GREY, style="italic")
    ax.text(1.0, 1.06, "every produced chain is checked the same way",
            fontsize=7.2, ha="left", va="top", color=GREY, style="italic")
    save(fig, path)


# ------------------------------------------------------------------ figure 4

def branching(path, scenarios_file):
    if not os.path.exists(scenarios_file):
        print(f"  (skipping {path}: {scenarios_file} not found)")
        return
    with open(scenarios_file) as f:
        scen = json.load(f)
    cells = {}
    for s in scen:
        cells.setdefault((s["depth"], s["nominal_k"]), []).append(
            s["branching_mean"])

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    depths = sorted({d for d, _ in cells})
    marks = ["o-", "s--", "^:", "d-."]
    for i, d in enumerate(depths):
        ks = sorted(k for dd, k in cells if dd == d)
        ys = [sum(cells[(d, k)]) / len(cells[(d, k)]) for k in ks]
        ax.plot(ks, ys, marks[i % len(marks)], label=f"depth {d}",
                markersize=4, linewidth=1.1)
    ax.set_xlabel("Nominal distractor count")
    ax.set_ylabel("Mean applicable actions per state")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    save(fig, path)


# ------------------------------------------------------------------ figures 5-7

def by_depth(path):
    r = RESULTS["by_depth"]
    xs = r["depths"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    model_line(ax, xs, r["gemini-3.1-flash-lite"], "o-", C_G31,
               "C1, gemini-3.1-flash-lite")
    model_line(ax, xs, r["gemini-3.5-flash-lite"], "s--", C_G35,
               "C1, gemini-3.5-flash-lite")
    ax.plot(xs, [100] * len(xs), "^:", color=C_B1, label="B1 (BFS)",
            markersize=5, linewidth=1.6)
    ax.plot(xs, r["B3"], "v-.", color=C_B3, label="B3 (random)",
            markersize=5, linewidth=1.3)
    ax.set_xticks(xs)
    rate_axes(ax, "Chain depth")
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, loc="center left",
              bbox_to_anchor=(0.01, 0.36))
    save(fig, path)


def by_distractors(path):
    r = RESULTS["by_distractors"]
    xs = r["k"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    model_line(ax, xs, r["gemini-3.1-flash-lite"], "o-", C_G31,
               "C1, gemini-3.1-flash-lite")
    model_line(ax, xs, r["gemini-3.5-flash-lite"], "s--", C_G35,
               "C1, gemini-3.5-flash-lite")
    ax.plot(xs, [100] * len(xs), "^:", color=C_B1, label="B1 (BFS)",
            markersize=5, linewidth=1.6)
    ax.plot(xs, r["B3"], "v-.", color=C_B3, label="B3 (random)",
            markersize=5, linewidth=1.3)
    ax.set_xticks(xs)
    rate_axes(ax, "Nominal distractor count")
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, loc="center left")
    save(fig, path)


def low_grid(path):
    r = RESULTS["low_grid"]
    xs = r["k"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    model_line(ax, xs, r["qwen2.5:14b"], "o-", C_QWEN, "C1, qwen2.5:14b")
    model_line(ax, xs, r["gemini-3.5-flash-lite"], "s--", C_G35,
               "C1, gemini-3.5-flash-lite")
    ax.plot(xs, r["B3"], "v-.", color=C_B3, label="B3 (random)",
            markersize=5, linewidth=1.3)
    ax.set_xticks(xs)
    rate_axes(ax, "Nominal distractor count")
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, loc="lower left")
    save(fig, path)


# ------------------------------------------------------------------ figure 8

def paired_discovery(path):
    r = RESULTS["paired"]["discovery"]
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    for i, (name, col) in enumerate([("C1", C_G31), ("C2", C_G35)]):
        p, lo, hi = wilson(*r[name])
        ax.bar(i, p, width=0.55, color=col, edgecolor=INK, linewidth=0.8)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color=INK,
                    capsize=4, linewidth=0.9)
        ax.text(i, hi + 1.5, f"{p:.0f}%", ha="center", va="bottom",
                fontsize=8.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["C1", "C2"])
    ax.set_ylabel("Chain discovery rate (%)")
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.6, 1.6)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    save(fig, path)


def paired_steps(path):
    r = RESULTS["paired"]
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    groups = ["proposed by\nthe model", "left in the\nreturned chain"]
    w = 0.34
    for j, (name, col) in enumerate([("C1", C_G31), ("C2", C_G35)]):
        vals = [r["proposed"][name], r["returned"][name]]
        xs = [g + (j - 0.5) * w for g in range(len(groups))]
        ax.bar(xs, vals, width=w, color=col, edgecolor=INK, linewidth=0.8,
               label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.02, f"{v:.2f}" if v else "0", ha="center",
                    va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("Inapplicable steps per instance")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    save(fig, path)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--scenarios", default="scenarios_v2.json")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda name: os.path.join(args.outdir, name)
    solution_approach(out("fig01"))
    c1_panel(out("fig02a"))
    c2_panel(out("fig02b"))
    pipeline(out("fig03"))
    branching(out("fig04"), args.scenarios)
    by_depth(out("fig05"))
    by_distractors(out("fig06"))
    low_grid(out("fig07"))
    paired_discovery(out("fig08a"))
    paired_steps(out("fig08b"))


if __name__ == "__main__":
    main()
