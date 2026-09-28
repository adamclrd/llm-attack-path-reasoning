# Evaluating LLM Agents for Multi-Step Attack-Path Reasoning

Code, scenario sets, results and transcripts for the paper *Evaluating LLM Agents
for Multi-Step Attack-Path Reasoning: Problem Formulation and Experimental Design*.

This is a **defensive** study on a **synthetic** benchmark. Scenarios are generated,
vulnerabilities are described by preconditions and effects, and nothing is exploited.
No exploit code is produced, no command is executed and no real system is contacted.
The output of every run is an ordered list of identifiers.

## What the study measures

Vulnerability chaining is treated as a discrete search problem. A scenario gives an
initial state, a set of vulnerabilities with explicit preconditions and effects, and
a goal. The system under test returns an ordered list of vulnerabilities. A chain is
correct when every step's preconditions hold at the moment it is used and the goal
predicates hold at the end.

Difficulty is varied along two axes:

* **chain depth** — the length of the reference solution;
* **distractor count** — how many irrelevant vulnerabilities are added, typed by the
  reasoning needed to rule each one out (T1 non-applicable, T2 near-miss, T3 decoy,
  T4 dead-end).

Four systems are compared on identical instances:

| System | What it is |
| --- | --- |
| `C1` | single prompt: the model tracks the state and picks the actions in one answer |
| `C2` | agent loop: a symbolic verifier holds the state and rejects proposals whose preconditions are missing |
| `B1` | breadth-first search, deterministic, the upper reference |
| `B3` | uniform random choice among applicable actions, the lower reference |

## Requirements

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

`numpy` is needed by `stats.py`, `matplotlib` by `make_figures.py`. `engine.py`,
`generate.py`, `run.py` (for `B1`/`B3`), `tables.py` and `to_pddl.py` use the
standard library only.

## API keys

No key is stored in this repository. Every client reads its key from the
environment:

```bash
export GEMINI_API_KEY=...        # hosted Gemini models
export OPENAI_API_KEY=...        # optional
export ANTHROPIC_API_KEY=...     # optional
export OLLAMA_HOST=http://localhost:11434   # optional, this is the default
export OLLAMA_NUM_CTX=4096                  # raise it if prompts are truncated
```

`.env` is in `.gitignore`. Do not commit one.

The free tier of the Gemini API caps both requests per minute and requests per day,
per project. `--delay 7` keeps the runner inside the per-minute cap (10 requests per
minute). The per-day cap resets at midnight Pacific time; `--resume` is there so a
campaign can be continued in the next quota window.

## Reproducing the results

The symbolic parts need no account and no network, so they reproduce exactly:

```bash
# breadth-first and random baselines over both scenario sets
python run.py --systems B1 B3 --scenarios scenarios_v2.json  --out results_main.csv
python run.py --systems B1 B3 --scenarios scenarios_low.json --out results_low.csv
```

The model runs need a key (or a local Ollama):

```bash
# single prompt, one hosted model, main grid
python run.py --systems C1 --provider gemini --model gemini-3.5-flash-lite \
              --scenarios scenarios_v2.json --out results_main.csv \
              --delay 7 --resume --transcripts transcripts

# agent with symbolic verification, on a subset
python run.py --systems C2 --provider gemini --model gemini-3.5-flash-lite \
              --scenarios scenarios_v2.json --out results_c2.csv --delay 7 --resume

# local model, low grid
python run.py --systems C1 --provider ollama --model qwen2.5:14b \
              --scenarios scenarios_low.json --out results_low.csv --transcripts transcripts
```

Then the analysis:

```bash
python tables.py --results results_main.csv --scenarios scenarios_v2.json --out tables.md
python stats.py  --results results_main.csv --system C1 --by-model
python make_figures.py --outdir figures --scenarios scenarios_v2.json
```

`probe.py` reports which models actually answer and at what latency and token cost,
which is worth running before a campaign: a model listed as available can still
return 404 on every call.

```bash
python probe.py --provider gemini --all --scenarios scenarios_v2.json
```

## Regenerating the scenarios

The released sets are the ones the paper uses. To build new ones:

```bash
python generate.py --instances 8 --depths 4,5,6,7 --k 20,30,40,50,60 \
                   --identifiers neutral --seed 20260828 --out scenarios_new.json
```

Generation runs backwards from the goal, so the reference solution and its depth are
guaranteed by construction. After distractors are added, a breadth-first check
redraws the instance if a shorter route appeared, which is why a failed run is a
search failure and not an error in the ground truth.

## Files

| File | Contents |
| --- | --- |
| `engine.py` | Applicability, chain verification, breadth-first search, counting of shortest solutions, branching factor, path accuracy. No model calls, no network. |
| `generate.py` | Scenario generation and verification, both identifier regimes, distractor mix as a parameter. |
| `models.py` | One client per provider behind a single signature. Keys come from the environment. |
| `probe.py` | Sends one real prompt to each candidate model and reports what came back. |
| `run.py` | Runs C1, C2, B1 and B3, one row per run, resumable across quota windows. |
| `stats.py` | The logistic regressions, with cluster-robust standard errors and likelihood-ratio tests. Needs numpy. |
| `tables.py` | Builds the result tables from a results file. |
| `to_pddl.py` | PDDL export for the B2 baseline, and collection of planner logs into the same results format. |
| `make_figures.py` | Figures 1 to 8 (`fig01` to `fig08b`), PNG at 300 dpi plus a PDF copy. |
| `scenarios_v2.json` | Main grid, 160 instances, with per-instance metadata. |
| `scenarios_low.json` | Low grid, 72 instances, with per-instance metadata. |
| `results.csv` | One row per scored run (see the note below). |
| `transcripts/` | One transcript per run. |
| `figure-data/` | The counts behind each figure, one file per figure, plus `results_complete.csv` (this file plus the C1 runs recovered from the transcripts). See `figure-data/README.md`. |

## What `results.csv` contains

One row per run, with the scenario, depth, nominal and realised distractor count,
branching factor, system, provider, model, success flag, step counts, chain length,
timing and token counts. A run is *scored* when its `error` column is empty; rows
carrying an HTTP error are kept so that the coverage gaps stay visible.

This export covers:

* `B1` (232 runs) and `B3` (1160 runs) on both grids;
* `C1` on `gemini-3.5-flash-lite` (232), `gemini-3.5-flash` (26) and `qwen2.5:14b` (97);
* `C2` on `gemini-3.5-flash-lite` (36 rows, 12 of them scored).

It does **not** contain the `gemini-3.1-flash-lite` runs, and its `C2` coverage is
smaller than the paired set reported in the paper. Tables and figures that involve
those runs cannot be reproduced from this file alone.

`make_figures.py` does not read `results.csv`: Figures 5 to 8 are drawn from the
counts reported in the result tables of the paper, which are kept in the `RESULTS`
dictionary at the top of that file, so a figure cannot drift from its table.
Figure 4 is read from `scenarios_v2.json`.

## Licence and permitted use

MIT, with a use restriction: see [LICENSE](LICENSE). Use is limited to lawful
security testing, academic research and evaluation by security teams.

## Citation

```bibtex
@misc{clouard2026llmattackpath,
  author = {Clouard, Adam},
  title  = {Evaluating LLM Agents for Multi-Step Attack-Path Reasoning:
            Problem Formulation and Experimental Design},
  year   = {2026}
}
```
