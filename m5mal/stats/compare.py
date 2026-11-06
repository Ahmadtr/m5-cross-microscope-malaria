"""Seed-level significance tests for the paper's comparisons.

    python -m m5mal.stats.compare [--csv results/per_seed.csv] [--comparisons configs/comparisons.yaml]

For each comparison (method A vs method B, same setting and split) over independent training seeds:
  * Welch's t-test (unequal variances), two-sided;
  * Hedges' g (bias-corrected standardised mean difference);
  * 95% bootstrap confidence interval of the difference in means (20,000 resamples, seed 0).
With 3-5 seeds per method these tests have little power; only large, consistent gaps reach p < 0.05.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

REPO = Path(__file__).resolve().parents[2]


def hedges_g(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else np.inf
    return float(d * (1 - 3 / (4 * (n1 + n2) - 9)))


def bootstrap_ci(a, b, n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def compare(a, b) -> dict:
    t, p = stats.ttest_ind(a, b, equal_var=False)
    lo, hi = bootstrap_ci(a, b)
    return {"mean_a": np.mean(a), "sd_a": np.std(a, ddof=1), "n_a": len(a), "mean_b": np.mean(b), "sd_b": np.std(b, ddof=1),
            "n_b": len(b), "diff": np.mean(a) - np.mean(b), "ci_lo": lo, "ci_hi": hi, "p": p, "g": hedges_g(a, b)}


def load(csv_path: Path) -> list:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def values(rows, setting, method, split, metric):
    return [float(r[metric]) for r in rows if r["setting"] == setting and r["method"] == method and r["split"] == split and r[metric]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=REPO / "results" / "per_seed.csv")
    ap.add_argument("--comparisons", type=Path, default=REPO / "configs" / "comparisons.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "stats.csv")
    a = ap.parse_args()
    rows = load(a.csv)
    out = []
    for c in yaml.safe_load(a.comparisons.read_text()):
        metric = c.get("metric", "mAP50")
        va, vb = (values(rows, c["setting"], m, c["split"], metric) for m in (c["a"], c["b"]))
        if len(va) < 2 or len(vb) < 2:
            print(f"skip {c['name']}: need >= 2 seeds per method (have {len(va)}, {len(vb)})")
            continue
        r = compare(va, vb) | {"name": c["name"], "metric": metric}
        out.append(r)
        verdict = "significant" if r["p"] < 0.05 else "not significant"
        print(f"{c['name']:52s} {r['mean_a']:5.1f}±{r['sd_a']:.1f} (n={r['n_a']}) vs {r['mean_b']:5.1f}±{r['sd_b']:.1f} (n={r['n_b']})  "
              f"diff {r['diff']:+5.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  p={r['p']:.4f}  g={r['g']:+.2f}  {verdict}")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "metric", "mean_a", "sd_a", "n_a", "mean_b", "sd_b", "n_b", "diff", "ci_lo", "ci_hi", "p", "g"])
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in out)
    print("written", a.out)


if __name__ == "__main__":
    main()
