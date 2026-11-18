"""Run-of-record table from the original run logs.

final.csv is the file the submitted tables were computed from. The only change
is rule 2: the gemini-3.1-flash-lite answer that could not be parsed (logged
twice in results_main.csv, both times the same truncated output) is scored as
a failure instead of being dropped. Attempts that never completed (API errors
after all retries) are kept as rows with scored=False so they can be counted.
"""
import sys, os, json, pandas as pd
# run from the folder that holds engine.py:  python analysis/build_runs.py
art, logs, out = ".", "logs", "analysis/runs_of_record.csv"
scen = {}
for f, g in (("scenarios_v2.json", "main"), ("scenarios_low.json", "low")):
    for s in json.load(open(os.path.join(art, f))): s["grid"] = g; scen[s["id"]] = s
def rd(name): 
    d = pd.read_csv(os.path.join(logs, name), dtype={"error": str}); d["error"] = d["error"].fillna(""); return d
final = rd("final.csv"); final["origin"] = "final.csv"
main = rd("results_main.csv")
bad = main[(main.model == "gemini-3.1-flash-lite") & (main.error.str.contains("JSON"))].tail(1).copy()
bad["error"] = ""; bad["success"] = 0; bad["origin"] = "results_main.csv, unparseable answer scored as failure"
runs = pd.concat([final, bad], ignore_index=True); runs["scored"] = True
# attempts that never completed: in a log, errored, and no scored row for that (system, model, scenario)
att = pd.concat([rd(n) for n in ("results.csv", "results_c2.csv", "results_c2b.csv", "results_main.csv")])
att = att[att.error != ""]
have = set(zip(runs.system, runs.model, runs.scenario))
miss = att[[k not in have for k in zip(att.system, att.model, att.scenario)]].drop_duplicates(["system", "model", "scenario"], keep="last").copy()
miss["scored"] = False; miss["origin"] = "never completed: " + miss.error.str[13:40]
runs = pd.concat([runs, miss], ignore_index=True)
runs["grid"] = runs.scenario.map(lambda x: scen[x]["grid"]); runs["k"] = runs.nominal_k
runs["tokens"] = runs.in_tokens + runs.out_tokens
runs = runs.rename(columns={"valid_steps": "valid", "invalid_steps": "invalid", "exact_match": "exact"})
runs["chain"] = runs.chain.fillna("")
runs.to_csv(out, index=False)
print(runs.groupby(["system", "model", "grid", "scored"]).size().to_string())
print("scored", runs.scored.sum(), "not scored", (~runs.scored).sum())
