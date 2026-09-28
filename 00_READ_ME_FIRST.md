# Evaluating LLM Agents for Multi-Step Attack-Path Reasoning — submission folder

Adam Clouard — 27 September 2026

This folder answers the request to share the data behind the figures, and
carries the revised paper with the reviewers' comments applied.

## What is here

| Folder | Contents |
| --- | --- |
| `01_paper/` | The revised paper, as .docx and as PDF. Every change made in this round is in blue; untouched text is black and word for word what it was. |
| `02_figures/` | The eight figures as separate files. `png/` at 300 dpi, `pdf/` as vector for typesetting. Figures 2 and 8 are two panels each, `fig02a`/`fig02b` and `fig08a`/`fig08b`. No titles are printed inside the figures; the captions carry the explanation. |
| `03_figure-data/` | **The data behind each figure**, one CSV per figure, plus a README describing every column and where each number comes from. |
| `04_code-and-artefacts/` | The full repository: scenario generator, experiment runner, symbolic engine, analysis code, the figure script, both scenario sets, the run-level results and 613 transcripts. |

## The figure data, in short

`03_figure-data/` has one file per figure, with the raw `successes` and `runs`
counts rather than percentages only, so intervals can be recomputed with
whatever method the journal prefers. The paper uses 95% Wilson score intervals,
and those are in the files too.

Figures 1, 2 and 3 are schematic drawings with no dataset behind them.

Figures 5, 6 and 7 are reproduced exactly from `results_complete.csv` in that
folder: all twelve series match the published figures count for count. That
file is the run-level export plus 177 runs recovered from the saved
transcripts, and it carries a `source` column saying which rows are which. The
recovery method was validated against the 351 runs present in both, which agree
on success, step counts and the full chain.

Figure 8 is the one exception, and the README in that folder explains why: the
released C2 runs do not reproduce the paired comparison in Table 10, so its two
files carry the paper's own numbers, labelled as transcribed rather than
recomputed. That is flagged here rather than smoothed over, because it is
something to settle before submission.

## Regenerating the figures

From `04_code-and-artefacts/llm-attack-path-reasoning/`:

```bash
pip install -r requirements.txt
python make_figures.py --outdir figures --scenarios scenarios_v2.json
```

That reproduces the figures as published, at 300 dpi plus a PDF copy. To redraw
them to different guidelines, take the counts from `03_figure-data/` instead of
from the script's internal `RESULTS` dictionary — they are the same numbers, in
a form that does not depend on the script.

The symbolic parts of the study need no API key and no network, so they
reproduce exactly:

```bash
python run.py --systems B1 B3 --scenarios scenarios_v2.json --out results_main.csv
```

## Two placeholders still in the paper

Both are marked `[À CONFIRMER]` in the text so they are easy to find:

1. How many times the daily quota interrupted the evaluation. The paper gave
   two different numbers in three places, so all three now carry the
   placeholder. The transcript timestamps show four separate run sessions, one
   of which resumes at 00:19 Pacific, just after a daily quota reset.
2. The repository URL in Appendix D, pending the GitHub account name.
