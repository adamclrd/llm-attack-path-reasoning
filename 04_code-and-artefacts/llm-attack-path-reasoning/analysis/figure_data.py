"""Write the counts behind Figures 5 to 7 (03_figure-data/fig05-07) from analysis/runs_of_record.csv."""
import sys, math, csv, pandas as pd
out = sys.argv[1] if len(sys.argv) > 1 else "../../03_figure-data"
r = pd.read_csv("analysis/runs_of_record.csv"); r = r[r.scored & (r.system != "C2")]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1)
def write(name, col, by, grid, levels, models):
    with open(f"{out}/{name}", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["series", col, "successes", "runs", "discovery_rate_pct", "wilson_95_low", "wilson_95_high", "data_source"])
        for label, sysm in models:
            g = r[(r.grid == grid) & (r.model == sysm)]
            for v in levels:
                x = g[g[by] == v]; k, n = int(x.success.sum()), len(x); lo, hi = wilson(k, n)
                w.writerow([label if sysm != "random" else "B3 random", v, k, n, round(100 * k / n + 1e-9, 1), lo, hi, "logs/final.csv (+ one unparseable run scored as a failure)"])
main = [("C1 gemini-3.1-flash-lite", "gemini-3.1-flash-lite"), ("C1 gemini-3.5-flash-lite", "gemini-3.5-flash-lite"), ("B3 random", "random")]
low = [("C1 qwen2.5:14b", "qwen2.5:14b"), ("C1 gemini-3.5-flash-lite", "gemini-3.5-flash-lite"), ("B3 random", "random")]
write("fig05_discovery_by_depth.csv", "chain_depth", "depth", "main", [4, 5, 6, 7], main)
write("fig06_discovery_by_distractors.csv", "nominal_distractors", "k", "main", [20, 30, 40, 50, 60], main)
write("fig07_low_grid.csv", "nominal_distractors", "k", "low", [5, 10, 20], low)
print("wrote figure data to", out)
