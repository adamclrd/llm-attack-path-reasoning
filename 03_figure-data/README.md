# Figure data

The numbers behind each figure of the paper, one CSV per figure, so the
figures can be redrawn without running the experiment again. All files are
UTF-8 CSV with a header row.

| Figure | What it shows | Data file |
| --- | --- | --- |
| 1 | The three candidate architectures | none (schematic drawing) |
| 2 | C1 and C2 as block diagrams | none (schematic drawing) |
| 3 | How one instance is built, checked and scored | none (schematic drawing) |
| 4 | Effective branching factor | `fig04_branching.csv` |
| 5 | Discovery rate against chain depth, main grid | `fig05_discovery_by_depth.csv` |
| 6 | Discovery rate against distractor count, main grid | `fig06_discovery_by_distractors.csv` |
| 7 | Discovery rate on the low grid | `fig07_low_grid.csv` |
| 8a | C1 against C2, discovery rate | `fig08a_paired_discovery.csv` |
| 8b | C1 against C2, inapplicable steps | `fig08b_inapplicable_steps.csv` |

## Where the numbers come from

- `fig04` is computed from `scenarios_v2.json`.
- `fig05`, `fig06` and `fig07` are written by
  `analysis/figure_data.py` from `analysis/runs_of_record.csv`, which is
  `logs/final.csv` plus the one unparseable gemini-3.1-flash-lite run scored as
  a failure. Each grid uses only the instances of its own scenario file
  (160 for the main grid, 72 for the low grid).
- `fig08a` and `fig08b` come from the 32 paired C1/C2 instances in
  `logs/final.csv`.

## Columns

**fig04**: `chain_depth`, `nominal_distractors`, `n_instances`,
`mean_applicable_actions_per_state` (the plotted value: mean applicable actions
per state along the reference path), `min_instance_mean` and
`max_instance_mean` (spread across the instances of the cell).

**fig05, fig06, fig07**: one row per series and x value. `successes` and `runs`
are raw counts; `discovery_rate_pct` is their ratio; `wilson_95_low` and
`wilson_95_high` are the 95% Wilson score interval used in the paper.

**fig08a**: one row per configuration, with `successes` out of `instances`.

**fig08b**: inapplicable steps per instance, proposed by the model and left in
the returned chain, for C1 and C2.
