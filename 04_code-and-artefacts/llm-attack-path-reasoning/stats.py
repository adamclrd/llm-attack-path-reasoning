"""
Formal analysis of the effects of chain depth and distractor count.

Why this file exists: percentages with confidence intervals describe each cell
but cannot say whether depth and distractor count matter once both are in the
model together, cannot separate them when they are varied at the same time, and
cannot test their interaction. A logistic regression can.

Model:      logit P(success) = b0 + b1*depth + b2*(distractors/10) [+ b3*depth*(distractors/10)]

Both predictors are centred, so b0 is the log-odds of success at the mean design
point rather than at depth zero, which does not exist. The distractor count is
divided by ten so that b2 reads as the effect of ten more distractors, which is
a step the design actually contains.

Standard errors are clustered on the instance. Runs of the same instance are not
independent: with temperature zero and a fixed prompt, two runs of one instance
are close to a repeated measurement, so treating them as two independent
observations would understate the standard errors. Clustering is what keeps the
repetition count from inflating the apparent precision.

Statsmodels is not required. Only numpy is used, so the script runs anywhere the
rest of the pipeline runs.

Usage:
    python stats.py --results results.csv --system C1
    python stats.py --results results.csv --system C1 --by-model
"""

import argparse
import csv
import math

import numpy as np


# ------------------------------------------------------------------ fitting

def irls(X, y, max_iter=100, tol=1e-10):
    """Logistic regression by iteratively reweighted least squares.

    Returns (beta, loglik, converged). A ridge term of 1e-8 keeps the step
    solvable when a cell is perfectly separated, which happens easily with
    small samples: for instance if every run at the lowest distractor level
    succeeds, the maximum-likelihood estimate is infinite.
    """
    n, p = X.shape
    beta = np.zeros(p)
    ll_old = -np.inf
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        WX = X * w[:, None]
        A = X.T @ WX + 1e-8 * np.eye(p)
        beta_new = np.linalg.solve(A, WX.T @ z)
        eta = np.clip(X @ beta_new, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        ll = float(np.sum(y * np.log(np.clip(mu, 1e-12, 1)) +
                          (1 - y) * np.log(np.clip(1 - mu, 1e-12, 1))))
        if abs(ll - ll_old) < tol:
            return beta_new, ll, True
        beta, ll_old = beta_new, ll
    return beta, ll_old, False


def cluster_robust_cov(X, y, beta, clusters):
    """Sandwich covariance, clustered on the instance."""
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-9, None)
    bread = np.linalg.inv(X.T @ (X * w[:, None]) + 1e-8 * np.eye(X.shape[1]))
    resid = (y - mu)[:, None] * X
    meat = np.zeros((X.shape[1], X.shape[1]))
    uniq = sorted(set(clusters))
    for c in uniq:
        idx = [i for i, cc in enumerate(clusters) if cc == c]
        s = resid[idx].sum(axis=0)[:, None]
        meat += s @ s.T
    g = len(uniq)
    n, p = X.shape
    if g > 1 and n > p:
        adj = (g / (g - 1.0)) * ((n - 1.0) / (n - p))
    else:
        adj = 1.0
    return adj * (bread @ meat @ bread)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def chi2_sf(x, df):
    """Upper tail of the chi-square distribution, for the likelihood-ratio test.
    Only df 1 and 2 are needed here, so the closed forms are enough."""
    if x <= 0:
        return 1.0
    if df == 1:
        return 2 * (1 - norm_cdf(math.sqrt(x)))
    if df == 2:
        return math.exp(-x / 2)
    # regularised upper incomplete gamma, series form, sufficient for small df
    k = df / 2.0
    t = x / 2.0
    term = 1.0 / math.gamma(k + 1)
    s = term
    for i in range(1, 200):
        term *= t / (k + i)
        s += term
    return max(0.0, min(1.0, 1 - s * math.exp(-t) * t ** k))


# ------------------------------------------------------------------ reporting

def report(rows, label, interaction=True):
    y = np.array([float(r["success"]) for r in rows])
    d = np.array([float(r["depth"]) for r in rows])
    k = np.array([float(r["distractors"]) / 10.0 for r in rows])
    clusters = [r["scenario"] for r in rows]

    n, g = len(y), len(set(clusters))
    print(f"\n{'=' * 68}\n{label}\n{'=' * 68}")
    print(f"observations {n}, instances {g}, successes {int(y.sum())} "
          f"({100 * y.mean():.1f}%)")
    print(f"depth {d.min():.0f} to {d.max():.0f}, "
          f"distractors {10*k.min():.0f} to {10*k.max():.0f}")

    if y.sum() in (0, n):
        print("\nEvery run has the same outcome, so no effect can be estimated.")
        return
    if len(set(d)) < 2 or len(set(k)) < 2:
        print("\nOnly one level of a predictor is present; nothing to estimate.")
        return

    dc, kc = d - d.mean(), k - k.mean()

    designs = [("depth + distractors", np.column_stack([np.ones(n), dc, kc]),
                ["intercept", "depth", "distractors (per 10)"])]
    if interaction:
        designs.append(("depth + distractors + interaction",
                        np.column_stack([np.ones(n), dc, kc, dc * kc]),
                        ["intercept", "depth", "distractors (per 10)",
                         "depth x distractors"]))

    fits = []
    for name, X, names in designs:
        beta, ll, ok = irls(X, y)
        cov = cluster_robust_cov(X, y, beta, clusters)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        fits.append((name, X, names, beta, se, ll, ok))

    for name, X, names, beta, se, ll, ok in fits:
        print(f"\n-- {name}{'' if ok else '   (did not converge; separation)'}")
        print(f"{'term':<24}{'coef':>9}{'SE':>9}{'z':>8}{'p':>9}"
              f"{'odds ratio':>12}  95% CI")
        for i, nm in enumerate(names):
            b, s = beta[i], se[i]
            z = b / s if s > 0 else float("nan")
            p = 2 * (1 - norm_cdf(abs(z))) if s > 0 else float("nan")
            lo, hi = b - 1.96 * s, b + 1.96 * s
            if nm == "intercept":
                print(f"{nm:<24}{b:>9.3f}{s:>9.3f}{z:>8.2f}{p:>9.3f}"
                      f"{'':>12}")
            else:
                print(f"{nm:<24}{b:>9.3f}{s:>9.3f}{z:>8.2f}{p:>9.3f}"
                      f"{math.exp(b):>12.3f}  "
                      f"[{math.exp(lo):.3f}, {math.exp(hi):.3f}]")
        print(f"log-likelihood {ll:.3f}")

    # ---- likelihood-ratio tests against reduced models
    print("\n-- likelihood-ratio tests")
    X_full = fits[0][1]
    ll_full = fits[0][5]
    for drop, nm in ((1, "depth"), (2, "distractors")):
        keep = [j for j in range(X_full.shape[1]) if j != drop]
        _, ll_red, _ = irls(X_full[:, keep], y)
        stat = 2 * (ll_full - ll_red)
        print(f"  dropping {nm:<22} chi2(1) = {stat:6.2f}   "
              f"p = {chi2_sf(stat, 1):.4f}")
    if interaction and len(fits) > 1:
        stat = 2 * (fits[1][5] - ll_full)
        print(f"  adding the interaction        chi2(1) = {stat:6.2f}   "
              f"p = {chi2_sf(stat, 1):.4f}")

    # ---- how much data would settle it
    print("\n-- note on power")
    per_cell = n / max(1, len(set(zip(d.tolist(), k.tolist()))))
    print(f"  {per_cell:.1f} observations per design cell on average. "
          f"With a binary outcome,\n  a rule of thumb is ten events per "
          f"estimated coefficient, so a three-term\n  model wants at least "
          f"thirty successes and thirty failures overall.")
    print(f"  present count: {int(y.sum())} successes, {int(n - y.sum())} failures.")


def load(path, system, provider=None):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["system"] == system and not r.get("error")]
    if provider:
        rows = [r for r in rows if r.get("model") == provider]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--system", default="C1")
    ap.add_argument("--by-model", action="store_true",
                    help="fit one model per LLM, then report whether the "
                         "coefficients differ across them")
    ap.add_argument("--no-interaction", action="store_true")
    args = ap.parse_args()

    rows = load(args.results, args.system)
    if not rows:
        print(f"No usable rows for system {args.system} in {args.results}.")
        return

    if args.by_model:
        models = sorted({r.get("model", "?") for r in rows})
        for m in models:
            sub = [r for r in rows if r.get("model") == m]
            report(sub, f"{args.system} — model {m}",
                   interaction=not args.no_interaction)
        if len(models) > 1:
            print(f"\n{'=' * 68}\nall models pooled, with a model indicator"
                  f"\n{'=' * 68}")
            print("Fit the pooled model in the same way and compare the "
                  "coefficients above:\nif the distractor coefficient has the "
                  "same sign and a similar magnitude for\nevery model, the "
                  "effect is a property of the task rather than of one model.")
    else:
        report(rows, f"{args.system} — all models pooled",
               interaction=not args.no_interaction)


if __name__ == "__main__":
    main()
