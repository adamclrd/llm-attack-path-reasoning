# Evaluating LLM Agents for Multi-Step Attack-Path Reasoning

Code, data and figures for the paper *Evaluating LLM Agents for Multi-Step
Attack-Path Reasoning: Problem Formulation and Experimental Design*
(Adam Clouard, 2026).

This is a defensive study on a synthetic benchmark: scenarios are generated,
vulnerabilities are described by preconditions and effects, and nothing is
exploited. The output of every run is an ordered list of identifiers.

## What is here

| Folder | Contents |
| --- | --- |
| `02_figures/` | The eight figures of the paper, PNG at 300 dpi and vector PDF. Figures 2 and 8 have two panels each. |
| `03_figure-data/` | The counts behind each figure, one CSV per figure. See its README. |
| `04_code-and-artefacts/llm-attack-path-reasoning/` | The code (scenario generator, runner, symbolic engine, analysis, figures), the two scenario sets, the original run logs and the transcripts. |

## Where the numbers come from

Every table and figure in the paper is computed from one run-level file,
`04_code-and-artefacts/llm-attack-path-reasoning/logs/final.csv`, with one
change: a gemini-3.1-flash-lite answer that could not be parsed is scored as a
failure instead of being dropped. The scripts in `analysis/` apply the four
counting rules of Section 4.5 of the paper and rebuild every table:

```bash
cd 04_code-and-artefacts/llm-attack-path-reasoning
pip install -r requirements.txt pandas
python analysis/build_runs.py      # -> analysis/runs_of_record.csv
python analysis/tables.py          # Tables 6 to 9
python analysis/regress.py         # Table 13 (and the submitted fits, for comparison)
python analysis/metrics.py         # Tables 10 and 11: exact match, path accuracy
python make_figures.py --outdir figures --scenarios scenarios_v2.json
```

See `04_code-and-artefacts/llm-attack-path-reasoning/README.md` for how to run
the experiment itself.

## Licence

MIT, with a use restriction: see the LICENSE file in the code folder.
