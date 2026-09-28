# Figure data

The numbers behind every figure in *Evaluating LLM Agents for Multi-Step
Attack-Path Reasoning: Problem Formulation and Experimental Design*, one file
per figure, so the figures can be redrawn to the journal's guidelines without
running the experiment again.

All files are UTF-8 CSV with a header row.

## Which figures have data

| Figure | What it shows | Data file |
| --- | --- | --- |
| 1 | The three candidate architectures | none — schematic drawing, no data |
| 2 | C1 and C2 as block diagrams | none — schematic drawing, no data |
| 3 | How one instance is built, checked and scored | none — schematic drawing, no data |
| 4 | Effective branching factor | `fig04_branching.csv` |
| 5 | Discovery rate against chain depth, main grid | `fig05_discovery_by_depth.csv` |
| 6 | Discovery rate against distractor count, main grid | `fig06_discovery_by_distractors.csv` |
| 7 | Discovery rate on the low grid | `fig07_low_grid.csv` |
| 8a | C1 against C2, discovery rate | `fig08a_paired_discovery.csv` |
| 8b | C1 against C2, inapplicable steps | `fig08b_inapplicable_steps.csv` |

Figures 1 to 3 are drawn entirely in code from fixed coordinates in
`make_figures.py`. There is no dataset behind them.

## Columns

**`fig04_branching.csv`** — one row per (depth, distractor count) cell of the
main grid, computed from `scenarios_v2.json`.

| Column | Meaning |
| --- | --- |
| `chain_depth` | Length of the reference solution |
| `nominal_distractors` | Distractors requested by the generator |
| `n_instances` | Instances in that cell |
| `mean_applicable_actions_per_state` | Mean applicable actions per state along the reference path, averaged over the cell. This is the plotted value. |
| `min_instance_mean`, `max_instance_mean` | Spread across the instances of the cell, for error bars if the journal wants them |

**`fig05`, `fig06`, `fig07`** — one row per series per x value.

| Column | Meaning |
| --- | --- |
| `series` | System and model, e.g. `C1 gemini-3.1-flash-lite` |
| `chain_depth` / `nominal_distractors` | The x value |
| `successes`, `runs` | Raw counts. Plot these, not the percentage, if the journal wants its own interval method. |
| `discovery_rate_pct` | `100 * successes / runs` |
| `wilson_95_low`, `wilson_95_high` | 95% Wilson score interval, the method used in the paper |
| `data_source` | `recorded`, `rebuilt from transcripts`, or `mixed` — see below |

**`fig08a`, `fig08b`** — the paired C1 against C2 comparison. `fig08b` splits
inapplicable steps into those the model proposed and those left in the chain
that is returned; C2 returns none by construction, because the verifier rejects
them before they enter.

## Where the numbers come from

`results_complete.csv` in this folder is the run-level source for Figures 5, 6
and 7. It is the released `results.csv` plus 177 C1 runs recovered from the
saved transcripts, with a `source` column marking which is which. A run counts
as scored when its `error` column is empty.

The recovery was necessary because the released `results.csv` does not contain
the `gemini-3.1-flash-lite` runs, which carry one of the two curves in
Figures 5 and 6. Each C1 transcript holds the prompt and the model's raw
answer, so the proposed chain can be read back and re-verified against the
scenario with the study's own `engine.verify_chain`.

The method was validated before being used: all 351 C1 runs that appear both in
`results.csv` and in the transcripts were rebuilt the same way, and success,
valid steps, invalid steps and the full chain agree in every one of them. The
rebuilt `gemini-3.1-flash-lite` rows then reproduce the paper's reported figures
exactly — 159 runs at 86.8%, 40/40, 36/40, 33/39 and 29/40 by depth, and 31/32,
27/32, 27/32, 27/32 and 26/31 by distractor count.

Every count in `fig05`, `fig06` and `fig07` was then checked against the
`RESULTS` dictionary that `make_figures.py` draws from. All 12 series match
exactly, so these files and the published figures carry the same numbers.

Rebuilt rows have no `time_s`, `in_tokens` or `out_tokens`: a transcript does
not record them. Those columns feed Table 11 only, never a figure.

## One caveat on Figure 8

Figure 8 is the exception. Its numbers come from the paper's Table 10 and
**cannot be recomputed from the released artefacts**, for two reasons:

* `results.csv` holds 36 C2 rows for `gemini-3.5-flash-lite`, 24 of them failed
  with HTTP 429, leaving 12 scored — against the 21 the paper reports;
* replaying the C2 transcripts the same way the C1 runs were rebuilt yields 27
  scoreable runs at 55.6% for that model and 13 at 76.9% for
  `gemini-flash-lite-latest`, against the 21 at 81.0% and 11 at 90.9% reported.

The C2 transcripts truncated by a quota failure are the likely reason the two
disagree, but the artefacts released here do not settle it. The values in
`fig08a` and `fig08b` are therefore transcribed from the paper's table and
labelled as such in the `data_source` column, rather than recomputed. Anyone
redrawing Figure 8 should use them as given, and the C2 run set should be
re-exported from the original run before the paper is submitted.

## Redrawing the figures

`make_figures.py` in the repository reproduces the published figures as they
stand, at 300 dpi plus a PDF copy:

```bash
python make_figures.py --outdir figures --scenarios scenarios_v2.json
```

To redraw to different guidelines, take the counts from the CSVs here rather
than from the script's `RESULTS` dictionary: they are the same numbers, in a
form that does not depend on the script.
