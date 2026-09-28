"""
Produces the tables of Section 4.8 from a results file.

    python tables.py --results results.csv --out tables.md

Handles the new column layout, so it works with several models and with C2 in
the same file. Everything it prints is also written to the output file, ready
to paste into the document.
"""

import argparse
import csv
import json
import math
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def cell(rows):
    if not rows:
        return "  –  "
    k = sum(int(r["success"]) for r in rows)
    p, lo, hi = wilson(k, len(rows))
    return f"{k}/{len(rows)} {p*100:.0f}% [{lo*100:.0f}–{hi*100:.0f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--scenarios", default="scenarios_v2.json")
    ap.add_argument("--out", default="tables.md")
    args = ap.parse_args()

    with open(args.results, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not r.get("error")]
    for r in rows:
        for k in ("depth", "distractors", "success", "valid_steps",
                  "invalid_steps", "chain_len", "ref_len", "exact_match",
                  "in_tokens", "out_tokens", "turns", "repeat_proposals"):
            r[k] = int(r[k] or 0)
        r["time_s"] = float(r["time_s"] or 0)

    # a system is identified by its label plus, for model systems, the model
    systems = sorted({(r["system"], r["model"]) for r in rows})
    depths = sorted({r["depth"] for r in rows})
    ks = sorted({r["distractors"] for r in rows})
    out = []

    def add(line=""):
        out.append(line)

    add("## Chain discovery rate by chain depth, pooled over distractor counts\n")
    add("| System | " + " | ".join(f"d={d}" for d in depths) + " |")
    add("|---" * (len(depths) + 1) + "|")
    for sysname, model in systems:
        cells = [cell([r for r in rows if r["system"] == sysname
                       and r["model"] == model and r["depth"] == d])
                 for d in depths]
        add(f"| {sysname} ({model}) | " + " | ".join(cells) + " |")

    add("\n## Chain discovery rate by distractor count, pooled over depths\n")
    add("| System | " + " | ".join(f"k={k}" for k in ks) + " |")
    add("|---" * (len(ks) + 1) + "|")
    for sysname, model in systems:
        cells = [cell([r for r in rows if r["system"] == sysname
                       and r["model"] == model and r["distractors"] == k])
                 for k in ks]
        add(f"| {sysname} ({model}) | " + " | ".join(cells) + " |")

    add("\n## Depth against distractor count, model systems only\n")
    for sysname, model in systems:
        if sysname.startswith("B"):
            continue
        add(f"\n**{sysname} — {model}**\n")
        add("| | " + " | ".join(f"k={k}" for k in ks) + " |")
        add("|---" * (len(ks) + 1) + "|")
        for d in depths:
            cells = [cell([r for r in rows if r["system"] == sysname
                           and r["model"] == model and r["depth"] == d
                           and r["distractors"] == k]) for k in ks]
            add(f"| d={d} | " + " | ".join(cells) + " |")

    add("\n## Secondary metrics by system\n")
    add("| System | Valid transition rate | Exact match | Invalid steps/run | "
        "Turns/run | Repeat proposals/run | Mean time (s) | Mean tokens |")
    add("|---|---|---|---|---|---|---|---|")
    for sysname, model in systems:
        sel = [r for r in rows if r["system"] == sysname and r["model"] == model]
        steps = sum(r["valid_steps"] + r["invalid_steps"] for r in sel)
        vtr = sum(r["valid_steps"] for r in sel) / steps if steps else 1.0
        n = len(sel)
        add(f"| {sysname} ({model}) | {vtr*100:.1f}% | "
            f"{sum(r['exact_match'] for r in sel)/n*100:.0f}% | "
            f"{sum(r['invalid_steps'] for r in sel)/n:.2f} | "
            f"{sum(r['turns'] for r in sel)/n:.1f} | "
            f"{sum(r['repeat_proposals'] for r in sel)/n:.2f} | "
            f"{sum(r['time_s'] for r in sel)/n:.3f} | "
            f"{sum(r['in_tokens']+r['out_tokens'] for r in sel)/n:.0f} |")

    add("\n## Produced chain length against reference length, model systems\n")
    add("| System | Distractors | Runs | Reference length | Longer | Shorter | Empty |")
    add("|---|---|---|---|---|---|---|")
    for sysname, model in systems:
        if sysname.startswith("B"):
            continue
        for k in ks:
            sel = [r for r in rows if r["system"] == sysname
                   and r["model"] == model and r["distractors"] == k]
            if not sel:
                continue
            same = sum(1 for r in sel if r["chain_len"] == r["ref_len"])
            longer = sum(1 for r in sel if r["chain_len"] > r["ref_len"])
            shorter = sum(1 for r in sel if 0 < r["chain_len"] < r["ref_len"])
            empty = sum(1 for r in sel if r["chain_len"] == 0)
            add(f"| {sysname} ({model}) | {k} | {len(sel)} | {same} | "
                f"{longer} | {shorter} | {empty} |")

    # ---- distractor types selected in failures, read from the scenario file
    try:
        with open(args.scenarios) as f:
            roles = {s["id"]: s.get("roles", {}) for s in json.load(f)}
    except OSError:
        roles = {}

    if roles:
        add("\n## Distractor type selected in unsuccessful runs\n")
        labels = ["T1-inert", "T2-nearmiss", "T3-decoy", "T4-deadend"]
        add("| System | Failures | " + " | ".join(labels) + " |")
        add("|---" * (len(labels) + 2) + "|")
        for sysname, model in systems:
            if sysname.startswith("B"):
                continue
            fails = [r for r in rows if r["system"] == sysname
                     and r["model"] == model and not r["success"]]
            c = defaultdict(int)
            for r in fails:
                picked = {roles.get(r["scenario"], {}).get(v)
                          for v in r["chain"].split("|") if v}
                for lab in labels:
                    if lab in picked:
                        c[lab] += 1
            add(f"| {sysname} ({model}) | {len(fails)} | "
                + " | ".join(str(c[lab]) for lab in labels) + " |")

    text = "\n".join(out)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
