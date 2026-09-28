"""
Runs the experiment.

Covers four systems:

  C1  single prompt. The model sees the whole scenario once and returns a full
      chain. State tracking and action selection are both its job.
  C2  iterative agent with symbolic verification. The model proposes one
      vulnerability per turn; a code verifier checks it, applies the effects if
      it holds, and otherwise returns the list of preconditions that were
      missing. State tracking is therefore correct and external, which leaves
      action selection as the model's only task. The C1 against C2 contrast is
      what the paper's architectural argument needs and did not have.
  B1  breadth-first search. Deterministic reference.
  B3  random choice among applicable vulnerabilities under a step budget.

Examples:
    python run.py --systems C1 C2 --provider mock --model mock --runs 3
    python run.py --systems C1 C2 --provider ollama --model llama3.1:8b --runs 5
    python run.py --systems C1 --provider gemini --model <flash-id> --delay 7 --resume

One row per run in the output CSV, with the model and the system in their own
columns so that several models can be appended to the same file and compared.
"""

import argparse
import csv
import json
import os
import random
import re
import time

from engine import (applicable, missing_predicates, verify_chain, bfs_shortest,
                    random_baseline, branching_along)
from models import get_client, list_gemini_models, ModelError

FIELDS = ["scenario", "depth", "distractors", "nominal_k", "branching_mean",
          "minimal_chains", "system", "provider", "model", "run", "success",
          "valid_steps", "invalid_steps", "repeat_proposals", "turns",
          "chain_len", "ref_len", "exact_match", "time_s", "in_tokens",
          "out_tokens", "chain", "error"]


# ------------------------------------------------------------------ prompts

C1_PROMPT = """You are analysing a network security scenario.

You are given the facts that are true at the start, a list of vulnerabilities, \
and a goal. Each vulnerability has preconditions (all must be true before it can \
be used) and effects (facts that become true after it is used). Facts are only \
ever added, never removed.

Find an ordered sequence of vulnerabilities leading from the initial state to the goal.

INITIAL STATE:
{init}

AVAILABLE VULNERABILITIES:
{vulns}

GOAL:
{goal}

Answer with a JSON object and nothing else:
{{"chain": ["ID-1", "ID-2"], "reasoning": "one or two sentences"}}

If no valid sequence exists, answer {{"chain": [], "reasoning": "..."}}."""


C2_PROMPT = """You are analysing a network security scenario, one step at a time.

Each vulnerability has preconditions (all must be true before it can be used) \
and effects (facts that become true after it is used). Facts are only ever \
added, never removed.

The CURRENT STATE below is maintained for you and is always correct. You do not \
have to track it yourself. Choose the single vulnerability to use next.

CURRENT STATE:
{state}

AVAILABLE VULNERABILITIES:
{vulns}

GOAL:
{goal}
{feedback}
Answer with one identifier, as a JSON object and nothing else:
{{"next": "ID", "reasoning": "one sentence"}}

If no sequence can reach the goal from the current state, answer \
{{"next": "NONE", "reasoning": "..."}}."""


def fmt_vulns(scenario):
    lines = []
    for vid, v in scenario["vulns"].items():
        lines.append(f"  {vid}")
        lines.append(f"    preconditions: {', '.join(v['pre'])}")
        lines.append(f"    effects: {', '.join(v['eff'])}")
    return "\n".join(lines)


def c1_prompt(scenario):
    return C1_PROMPT.format(
        init="\n".join(f"  {f}" for f in scenario["init"]),
        vulns=fmt_vulns(scenario),
        goal="\n".join(f"  {f}" for f in scenario["goal"]))


def c2_prompt(scenario, state, feedback, vulns_block):
    fb = ""
    if feedback:
        fb = "\nLAST PROPOSAL REJECTED:\n" + feedback + "\n"
    return C2_PROMPT.format(
        state="\n".join(f"  {f}" for f in sorted(state)),
        vulns=vulns_block,
        goal="\n".join(f"  {f}" for f in scenario["goal"]),
        feedback=fb)


# ------------------------------------------------------------------ parsing

def _json_blob(text):
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None, "no JSON object found"
    try:
        return json.loads(m.group(0)), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


def parse_chain(text):
    data, err = _json_blob(text)
    if err:
        return None, err
    chain = data.get("chain")
    if not isinstance(chain, list):
        return None, "no 'chain' list in response"
    return [str(x).strip() for x in chain], None


def parse_next(text):
    data, err = _json_blob(text)
    if err:
        return None, err
    nxt = data.get("next")
    if not isinstance(nxt, str):
        return None, "no 'next' string in response"
    return nxt.strip(), None


# ------------------------------------------------------------------ systems

def run_c1(scenario, call, model, transcript_path):
    prompt = c1_prompt(scenario)
    t0 = time.time()
    raw, in_tok, out_tok = call(prompt, model)
    elapsed = time.time() - t0
    chain, err = parse_chain(raw)
    if chain is None:
        chain = []
    if transcript_path:
        with open(transcript_path, "w") as f:
            f.write(prompt + "\n\n===== RESPONSE =====\n" + raw)
    res = verify_chain(chain, scenario)
    res.update(chain=chain, turns=1, repeat_proposals=0, time_s=elapsed,
               in_tokens=in_tok, out_tokens=out_tok, error=err or "")
    return res


def run_c2(scenario, call, model, transcript_path, budget=None):
    """The iterative agent. The verifier owns the state; the model only picks.

    Terminates when the goal is reached, when the model answers NONE, or when
    the step budget is spent. The budget is a multiple of the reference depth
    so that an instance cannot run indefinitely if the model loops.

    Re-proposing the identifier that was just rejected is counted separately:
    it is a documented agent failure mode and it is invisible in a plain
    invalid-step count.
    """
    budget = budget or (3 * scenario["depth"] + 5)
    state = set(scenario["init"])
    goal = set(scenario["goal"])
    vulns_block = fmt_vulns(scenario)

    chain, log = [], []
    valid = invalid = repeats = turns = 0
    in_tok = out_tok = 0
    feedback = ""
    last_rejected = None
    err = ""
    t0 = time.time()

    while turns < budget:
        turns += 1
        prompt = c2_prompt(scenario, state, feedback, vulns_block)
        try:
            raw, i_t, o_t = call(prompt, model)
        except Exception as exc:                       # noqa: BLE001
            err = f"call failed: {exc}"
            break
        in_tok += i_t
        out_tok += o_t
        log.append(f"--- turn {turns} ---\n{prompt}\n\n>>> {raw}\n")

        vid, perr = parse_next(raw)
        if vid is None:
            invalid += 1
            feedback = f"  the answer could not be parsed ({perr})"
            continue

        if vid.upper() == "NONE":
            break

        v = scenario["vulns"].get(vid)
        if v is None:
            invalid += 1
            if vid == last_rejected:
                repeats += 1
            last_rejected = vid
            feedback = f"  {vid} is not one of the available identifiers"
            continue

        if applicable(v, state):
            state |= set(v["eff"])
            chain.append(vid)
            valid += 1
            feedback = ""
            last_rejected = None
            if goal.issubset(state):
                break
        else:
            invalid += 1
            if vid == last_rejected:
                repeats += 1
            last_rejected = vid
            miss = missing_predicates(v, state)
            feedback = (f"  {vid} cannot be used yet. These preconditions are "
                        f"not satisfied:\n" +
                        "\n".join(f"    {m}" for m in miss))

    elapsed = time.time() - t0
    if transcript_path:
        with open(transcript_path, "w") as f:
            f.write("\n".join(log))

    return {
        "success": goal.issubset(state),
        "valid_steps": valid,
        "invalid_steps": invalid,
        "repeat_proposals": repeats,
        "turns": turns,
        "chain": chain,
        "time_s": elapsed,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "error": err,
    }


# ------------------------------------------------------------------ main

def row(scenario, system, provider, model, run, res):
    chain = res["chain"]
    return {
        "scenario": scenario["id"],
        "depth": scenario["depth"],
        "distractors": scenario["distractors"],
        "nominal_k": scenario.get("nominal_k", scenario["distractors"]),
        "branching_mean": scenario.get("branching_mean", ""),
        "minimal_chains": scenario.get("minimal_chains", ""),
        "system": system, "provider": provider, "model": model, "run": run,
        "success": int(res["success"]),
        "valid_steps": res["valid_steps"],
        "invalid_steps": res["invalid_steps"],
        "repeat_proposals": res.get("repeat_proposals", 0),
        "turns": res.get("turns", 1),
        "chain_len": len(chain),
        "ref_len": scenario["depth"],
        "exact_match": int(list(chain) == list(scenario["reference_chain"])),
        "time_s": round(res.get("time_s", 0.0), 4),
        "in_tokens": res.get("in_tokens", 0),
        "out_tokens": res.get("out_tokens", 0),
        "chain": "|".join(chain),
        "error": res.get("error", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["C1", "B1", "B3"],
                    choices=["C1", "C2", "B1", "B3"])
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--scenarios", default="scenarios.json")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds between model calls; use 7 on the Gemini "
                         "free tier to stay under ten requests a minute")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--transcripts", default="transcripts")
    ap.add_argument("--resume", action="store_true",
                    help="keep the rows already in the output file and run "
                         "only what is missing, so a daily quota can be "
                         "worked through over several sessions")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        list_gemini_models()
        return

    call = get_client(args.provider)
    with open(args.scenarios) as f:
        scenarios = json.load(f)
    os.makedirs(args.transcripts, exist_ok=True)

    rows, done = [], set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                if not r.get("error"):
                    done.add((r["scenario"], r["system"], r["model"], r["run"]))
        print(f"Resuming: {len(rows)} rows present, {len(done)} usable.\n")

    rng = random.Random(42)

    def flush():
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    for s in scenarios:
        # ---------------------------------------------------- B1, deterministic
        if "B1" in args.systems and (s["id"], "B1", "bfs", "1") not in done:
            t0 = time.time()
            chain, _ = bfs_shortest(s)
            el = time.time() - t0
            chain = chain if isinstance(chain, list) else []
            res = verify_chain(chain, s)
            res.update(chain=chain, time_s=el, turns=1, repeat_proposals=0,
                       in_tokens=0, out_tokens=0, error="")
            rows.append(row(s, "B1", "none", "bfs", 1, res))

        # ---------------------------------------------------- B3, random
        if "B3" in args.systems:
            for run in range(1, args.runs + 1):
                if (s["id"], "B3", "random", str(run)) in done:
                    continue
                chain = random_baseline(s, 3 * s["depth"] + 5, rng)
                res = verify_chain(chain, s)
                res.update(chain=chain, time_s=0.0, turns=len(chain),
                           repeat_proposals=0, in_tokens=0, out_tokens=0,
                           error="")
                rows.append(row(s, "B3", "none", "random", run, res))

        # ---------------------------------------------------- model systems
        for system in [x for x in args.systems if x in ("C1", "C2")]:
            for run in range(1, args.runs + 1):
                if (s["id"], system, args.model, str(run)) in done:
                    print(f"{s['id']:16s} {system} run {run}  skipped")
                    continue
                # Windows forbids : * ? " < > | / \\ in file names, and model
                # ids such as "llama3.1:8b" contain a colon, so sanitise.
                safe_model = re.sub(r"[^A-Za-z0-9._-]", "-", args.model)
                tpath = os.path.join(
                    args.transcripts,
                    f"{s['id']}_{system}_{safe_model}_r{run}.txt")
                res = None
                for attempt in range(args.retries + 1):
                    try:
                        if system == "C1":
                            res = run_c1(s, call, args.model, tpath)
                        else:
                            res = run_c2(s, call, args.model, tpath)
                        break
                    except (ModelError, Exception) as exc:   # noqa: BLE001
                        msg = str(exc)
                        # Hosted endpoints fail transiently: 429 is a rate
                        # limit, 5xx and read timeouts are the provider being
                        # briefly unavailable. None of these say anything about
                        # the model's ability, so retrying is the correct
                        # response; recording them as failures would bias the
                        # results downwards.
                        transient = ("429", "500", "502", "503", "504",
                                     "timed out", "timeout", "Connection")
                        if any(t in msg for t in transient) \
                                and attempt < args.retries:
                            wait = 15 * (2 ** attempt)
                            print(f"    transient error ({msg[:60]}), "
                                  f"retrying in {wait}s")
                            time.sleep(wait)
                            continue
                        res = {"success": False, "valid_steps": 0,
                               "invalid_steps": 0, "repeat_proposals": 0,
                               "turns": 0, "chain": [], "time_s": 0.0,
                               "in_tokens": 0, "out_tokens": 0,
                               "error": f"call failed: {msg}"}
                        break
                rows.append(row(s, system, args.provider, args.model, run, res))
                print(f"{s['id']:16s} {system} run {run}  "
                      f"success={res['success']} turns={res.get('turns')} "
                      f"chain={res['chain']}")
                flush()
                if args.delay:
                    time.sleep(args.delay)

    flush()
    print(f"\n{len(rows)} rows written to {args.out}")


if __name__ == "__main__":
    main()
