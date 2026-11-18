"""Replay C2 transcripts against the scenario files and classify each run."""
import sys, os, re, json, csv
art = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, art)  # engine.py only (pure functions), read by path
import importlib.util
spec = importlib.util.spec_from_file_location("engine", os.path.join(art, "engine.py"))
engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
scen = {}
for f, g in (("scenarios_v2.json", "main"), ("scenarios_low.json", "low")):
    for s in json.load(open(os.path.join(art, f))):
        s["grid"] = g; scen[s["id"]] = s
out = []
tdir = os.path.join(art, "transcripts")
for fn in sorted(os.listdir(tdir)):
    m = re.match(r"(d\d+_k\d+_i\d+)_C2_(.+)_r(\d+)\.txt$", fn)
    if not m: continue
    sid, model, run = m.groups()
    s = scen[sid]
    text = open(os.path.join(tdir, fn), encoding="utf-8", errors="replace").read()
    turns = re.split(r"^--- turn (\d+) ---$", text, flags=re.M)
    # turns -> ['', '1', body, '2', body...]
    state = set(s["init"]); goal = set(s["goal"]); budget = 3*s["depth"]+5
    chain=[]; invalid=0; repeats_noop=0; nturn=0; outcome=None; last_rej=None; rej_repeat=0
    props=[]
    for i in range(1, len(turns), 2):
        nturn = int(turns[i]); body = turns[i+1]
        r = body.split("\n>>> ", 1)
        if len(r) < 2:
            outcome = "no_response"; break
        resp = r[1]
        mm = re.search(r'"next"\s*:\s*"([^"]*)"', resp)
        if not mm:
            invalid += 1; props.append("?"); continue
        vid = mm.group(1).strip(); props.append(vid)
        if vid.upper() == "NONE":
            outcome = "gave_up"; break
        v = s["vulns"].get(vid)
        if v is None:
            invalid += 1
            if vid == last_rej: rej_repeat += 1
            last_rej = vid; continue
        if engine.applicable(v, state):
            if set(v["eff"]).issubset(state): repeats_noop += 1
            state |= set(v["eff"]); chain.append(vid); last_rej=None
            if goal.issubset(state):
                outcome = "success"; break
        else:
            invalid += 1
            if vid == last_rej: rej_repeat += 1
            last_rej = vid
    if outcome is None:
        outcome = "budget" if nturn >= budget else "truncated"
    out.append(dict(scenario=sid, grid=s["grid"], depth=s["depth"], k=s["nominal_k"], model=model,
                    turns=nturn, budget=budget, outcome=outcome, success=int(outcome=="success"),
                    invalid=invalid, noop=repeats_noop, rej_repeat=rej_repeat, chain="|".join(chain),
                    exact=int(chain==s["reference_chain"])))
w = csv.DictWriter(open(sys.argv[2] if len(sys.argv) > 2 else "analysis/c2_replay.csv", "w", newline=""), fieldnames=list(out[0]))
w.writeheader(); w.writerows(out)
