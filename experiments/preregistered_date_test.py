"""
PRE-REGISTERED confirmatory test — registered 2026-11-10, before any data it scores exists.

Origin: experiments/mechanism_battery.py (exploratory, 54 tests) found NJ_Pick6 draws contain
the day-of-month number less often than chance (8.9% vs 13.0%, p=0.0035 nominal; family-wise
NOT significant; did not replicate across halves).  One of 54 tests, so ~2.7 nominal hits are
expected by chance.  This file fixes the confirmatory test so it cannot be tuned afterwards.

HYPOTHESIS (frozen):
  H_A: P(draw contains the number equal to its calendar day-of-month) < K/N,
       for draws with day-of-month <= N.
  Primary:   NJ_Pick6, one-sided (less), alpha = 0.05, exact binomial.
  Secondary: Pick6 + Powerball pooled (same direction), alpha = 0.05.
  Only draws dated STRICTLY AFTER 2026-11-10 count.  Nothing earlier may be used.
  Do not change the hypothesis, direction, alpha or cutoff.  Read the result once the
  sample is adequate; do not peek-and-stop (that inflates false positives).

Power note: at the observed ratio 0.69 (p0=0.13) ~330 eligible draws give ~80% power;
Pick6 draws ~3/week, so expect to wait ~2 years.  A smaller true effect needs more.
Reporting before ~330 eligible Pick6 draws is informational only.

Run: python experiments/preregistered_date_test.py
"""

import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import randomness_battery as rb  # noqa: E402

CUTOFF = pd.Timestamp("2026-11-10")  # frozen; draws on/before this date are excluded
ALPHA = 0.05
TARGET_ELIGIBLE = 330


def score(game):
    d, dt, lo, hi, k = rb.load_game(game)
    dt = pd.DatetimeIndex(dt)
    keep = dt > CUTOFF
    d, dom = d[keep], dt[keep].day.values
    elig = (dom >= lo) & (dom <= hi)
    hits = int(((d == dom[:, None]).any(1) & elig).sum())
    return hits, int(elig.sum()), k / (hi - lo + 1)


def main():
    res = {g: score(g) for g in ("NJ_Pick6", "Powerball")}
    for g, (h, n, p0) in res.items():
        print(f"{g:10s} eligible draws after {CUTOFF.date()}: {n:4d}  hits {h:3d}  chance {p0:.3f}")

    h, n, p0 = res["NJ_Pick6"]
    if n == 0:
        print("\nPrimary: no post-registration draws yet.")
        return
    p = stats.binomtest(h, n, p0, alternative="less").pvalue
    state = "ADEQUATE" if n >= TARGET_ELIGIBLE else f"INFORMATIONAL ONLY ({n}/{TARGET_ELIGIBLE})"
    print(f"\nPrimary (Pick6, one-sided less): rate {h / n:.3f} vs {p0:.3f}  p={p:.4f}  [{state}]")

    # secondary: pooled via sum of observed minus expected hits (Poisson-binomial normal approx)
    import numpy as np
    obs = sum(v[0] for v in res.values())
    exp = sum(v[1] * v[2] for v in res.values())
    var = sum(v[1] * v[2] * (1 - v[2]) for v in res.values())
    if var > 0:
        z = (obs - exp) / np.sqrt(var)
        print(f"Secondary (pooled): obs {obs} vs exp {exp:.1f}  z={z:+.2f}  p(one-sided)={stats.norm.cdf(z):.4f}")
    print(f"Decision rule: reject chance only if primary p < {ALPHA} AND sample ADEQUATE.")


if __name__ == "__main__":
    main()
