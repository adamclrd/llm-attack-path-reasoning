"""
Exports each scenario as a PDDL domain and problem, so that baseline B2 can be
run instead of only specified.

Every ground predicate of the scenario becomes a nullary PDDL predicate and
every vulnerability becomes a nullary action. This is a propositional encoding:
it does not lift the predicates into schemata with typed objects, which a
hand-authored domain would do. That is deliberate. The point of B2 is to
measure what a production planner costs on the same instances, not to compare
domain-engineering styles, and lifting would introduce a hand-written domain,
which is exactly the artefact the paper argues is the bottleneck.

Usage:
    python to_pddl.py --scenarios scenarios.json --outdir pddl

Then, with Fast Downward on the path:
    for p in pddl/*_problem.pddl; do
        d="${p%_problem.pddl}_domain.pddl"
        /usr/bin/time -f "%e" fast-downward.py "$d" "$p" \
            --search "astar(lmcut())" > "${p%.pddl}.log" 2>&1
    done
    python to_pddl.py --collect pddl --out b2_results.csv
"""

import argparse
import csv
import json
import os
import re


def sanitise(fact):
    """PDDL identifiers cannot contain brackets or commas."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", fact).strip("_").lower()


def domain(scenario):
    preds = set(scenario["init"]) | set(scenario["goal"])
    for v in scenario["vulns"].values():
        preds |= set(v["pre"]) | set(v["eff"])
    lines = [f"(define (domain chain-{sanitise(scenario['id'])})",
             "  (:requirements :strips)",
             "  (:predicates"]
    for p in sorted(preds):
        lines.append(f"    ({sanitise(p)})")
    lines.append("  )")
    for vid, v in scenario["vulns"].items():
        pre = " ".join(f"({sanitise(p)})" for p in v["pre"])
        eff = " ".join(f"({sanitise(e)})" for e in v["eff"])
        lines += [f"  (:action {sanitise(vid)}",
                  f"    :parameters ()",
                  f"    :precondition (and {pre})",
                  f"    :effect (and {eff})",
                  "  )"]
    lines.append(")")
    return "\n".join(lines)


def problem(scenario):
    init = " ".join(f"({sanitise(f)})" for f in scenario["init"])
    goal = " ".join(f"({sanitise(f)})" for f in scenario["goal"])
    return "\n".join([
        f"(define (problem {sanitise(scenario['id'])})",
        f"  (:domain chain-{sanitise(scenario['id'])})",
        f"  (:init {init})",
        f"  (:goal (and {goal}))",
        ")"])


def collect(outdir, out):
    """Reads the planner logs back into the same CSV layout as run.py, so B2
    lands in the same table as everything else."""
    rows = []
    for name in sorted(os.listdir(outdir)):
        if not name.endswith("_problem.log"):
            continue
        sid = name[: -len("_problem.log")]
        text = open(os.path.join(outdir, name)).read()
        plan = re.findall(r"^([a-z0-9_-]+) \(1\)$", text, re.MULTILINE)
        m = re.search(r"Search time: ([0-9.]+)s", text)
        t = float(m.group(1)) if m else 0.0
        solved = "Solution found" in text
        rows.append({"scenario": sid, "system": "B2", "provider": "none",
                     "model": "fast-downward", "run": 1,
                     "success": int(solved), "chain_len": len(plan),
                     "time_s": t, "in_tokens": 0, "out_tokens": 0,
                     "chain": "|".join(plan), "error": ""})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} planner runs collected into {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="scenarios.json")
    ap.add_argument("--outdir", default="pddl")
    ap.add_argument("--collect", help="read planner logs from this directory")
    ap.add_argument("--out", default="b2_results.csv")
    args = ap.parse_args()

    if args.collect:
        collect(args.collect, args.out)
        return

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.scenarios) as f:
        scenarios = json.load(f)
    for s in scenarios:
        base = os.path.join(args.outdir, s["id"])
        with open(base + "_domain.pddl", "w") as f:
            f.write(domain(s))
        with open(base + "_problem.pddl", "w") as f:
            f.write(problem(s))
    print(f"{2 * len(scenarios)} PDDL files written to {args.outdir}/")


if __name__ == "__main__":
    main()
