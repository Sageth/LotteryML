"""
Mechanism-driven battery: tests chosen for *plausible real-world flaws*, not generic
omnibus power.  Single/low-df statistics so each has more power than a 45-cell chi-square.

  M1 trend      frequency vs ball number (software-RNG modulo/range bias)
  M2 digit      last-digit, mod-3, mod-5 uniformity (physical ball printing/weight)
  M3 block      worst 100-draw block ball-frequency chi2 vs MC max (ball-set rotation)
  M4 overlap    overlap with draws t-1..t-5 vs hypergeometric (pooled memory)
  M5 volatility Ljung-Box on squared sum deviations (variance clustering)
  M6 date       ball==day-of-month hits; draw mean by day-of-month (date-seeded RNG)
  M7 period     Fisher g on sum periodogram (hidden cycles), permutation null
  M8 spread     (max-min) decile chi2 vs exact null (order-statistic flaw)

One Holm family across games x tests (full sample).  Exploratory H1 hits must replicate in H2.
Per-mechanism Fisher pooling across games gives extra power for effects shared across games.
Run: python experiments/mechanism_battery.py [GAME ...]
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import randomness_battery as rb  # noqa: E402

NSIM = 400
rng = rb.rng


def counts_of(draws, lo, hi):
    return np.bincount((draws - lo).ravel(), minlength=hi - lo + 1).astype(float)


def mc_p(obs, simfn, two_sided=False, mu=None):
    null = np.array([simfn() for _ in range(NSIM)])
    if two_sided:
        mu = null.mean() if mu is None else mu
        return (1 + (abs(null - mu) >= abs(obs - mu)).sum()) / (NSIM + 1)
    return (1 + (null >= obs).sum()) / (NSIM + 1)


def trend_stat(draws, lo, hi):
    c = counts_of(draws, lo, hi)
    return abs(np.corrcoef(c, np.arange(len(c)))[0, 1])


def resid_chi(draws, lo, hi, mod):
    vals = np.arange(lo, hi + 1) % mod
    c = counts_of(draws, lo, hi)
    obs = np.bincount(vals, weights=c, minlength=mod)
    exp = np.bincount(vals, minlength=mod) / len(vals) * c.sum()
    return stats.chisquare(obs, exp).pvalue


def block_stat(draws, lo, hi, k, size=100):
    span = hi - lo + 1
    best = 0
    for s in range(0, len(draws) - size + 1, size):
        c = counts_of(draws[s:s + size], lo, hi)
        e = k * size / span
        best = max(best, ((c - e) ** 2 / e).sum())
    return best


def overlap_p(draws, lo, hi, k, lags=5):
    span = hi - lo + 1
    m = np.zeros((len(draws), span), bool)
    np.put_along_axis(m, draws - lo, True, axis=1)
    h = stats.hypergeom(span, k, k)
    q = 0.0
    for lag in range(1, lags + 1):
        ov = (m[lag:] & m[:-lag]).sum(1)
        z = (ov.mean() - h.mean()) / (h.std() / np.sqrt(len(ov)))
        q += z * z
    return stats.chi2.sf(q, lags)


def fisher_g(x):
    x = x - x.mean()
    pg = np.abs(np.fft.rfft(x)[1:]) ** 2
    return pg.max() / pg.sum()


def date_p(draws, dates, lo, hi, k):
    d = pd.DatetimeIndex(dates)
    dom = d.day.values
    hit = (draws == dom[:, None]).any(1)
    elig = (dom >= lo) & (dom <= hi)
    p_dom = stats.binomtest(int(hit[elig].sum()), int(elig.sum()), k / (hi - lo + 1)).pvalue
    g = [v for _, v in pd.Series(draws.mean(1)).groupby(dom) if len(v) >= 5]
    return p_dom, (stats.kruskal(*g).pvalue if len(g) > 1 else np.nan)


def spread_p(draws, lo, hi, k, ref):
    r = draws[:, -1] - draws[:, 0]
    edges = np.quantile(ref, np.linspace(0, 1, 11)[1:-1])
    obs = np.bincount(np.searchsorted(edges, r, side="right"), minlength=10)
    # discrete range => tied quantile edges => bins are NOT equiprobable; use true probs
    prob = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=10) / len(ref)
    return stats.chisquare(obs, prob * len(r)).pvalue


def mech(draws, dates, lo, hi, k):
    n = len(draws)
    out = {}
    out["M1_trend"] = mc_p(trend_stat(draws, lo, hi),
                           lambda: trend_stat(rb.sim_draws(n, k, lo, hi), lo, hi))
    out["M2_lastdigit"] = resid_chi(draws, lo, hi, 10)
    out["M2_mod3"] = resid_chi(draws, lo, hi, 3)
    out["M2_mod5"] = resid_chi(draws, lo, hi, 5)
    if n >= 200:
        out["M3_block"] = mc_p(block_stat(draws, lo, hi, k),
                               lambda: block_stat(rb.sim_draws(n, k, lo, hi), lo, hi, k))
    out["M4_overlap"] = overlap_p(draws, lo, hi, k)
    s = draws.sum(1).astype(float)
    out["M5_volatility"] = rb.ljung_box((s - s.mean()) ** 2)
    out["M6_ball_eq_day"], out["M6_dom_mean"] = date_p(draws, dates, lo, hi, k)
    out["M7_period"] = mc_p(fisher_g(s), lambda: fisher_g(rng.permutation(s)))
    ref = np.diff(rb.sim_draws(20000, k, lo, hi)[:, [0, -1]], axis=1).ravel()
    out["M8_spread"] = spread_p(draws, lo, hi, k, ref)
    return out


def main(games):
    full, halves = {}, {}
    for g in games:
        d, dt, lo, hi, k = rb.load_game(g)
        n = len(d)
        print(f"{g}: {n} draws")
        for t, p in mech(d, dt, lo, hi, k).items():
            full[(g, t)] = p
        h = n // 2
        for tag, sl in (("H1", slice(0, h)), ("H2", slice(h, n))):
            for t, p in mech(d[sl], dt[sl], lo, hi, k).items():
                halves[(g, t, tag)] = p

    fam = [(f"{g}:{t}", p) for (g, t), p in full.items() if not np.isnan(p)]
    rej = rb.holm(fam)
    print(f"\n=== {len(fam)} mechanism tests, Holm α=0.05 ===")
    print(f"chance-expected nominal hits {0.05 * len(fam):.1f}; observed {sum(p < .05 for _, p in fam)}; "
          f"Holm survivors: {rej}")
    for k_, p in sorted(fam, key=lambda kv: kv[1])[:10]:
        print(f"  {k_:36s} p={p:.4f}")

    print("\n=== Fisher-pooled across games (per mechanism) ===")
    tests = sorted({t for _, t in full})
    pooled = []
    for t in tests:
        ps = [p for (g, tt), p in full.items() if tt == t and not np.isnan(p)]
        if len(ps) > 1:
            pp = stats.chi2.sf(-2 * np.log(np.clip(ps, 1e-12, 1)).sum(), 2 * len(ps))
            pooled.append((t, pp))
    for t, pp in sorted(pooled, key=lambda x: x[1]):
        print(f"  {t:18s} pooled p={pp:.4f}   Bonferroni({len(pooled)})={min(1, pp * len(pooled)):.4f}")

    disc = [(g, t) for (g, t, tag), p in halves.items() if tag == "H1" and p < 0.05]
    rep = [(g, t) for g, t in disc if halves.get((g, t, "H2"), 1) < 0.05]
    print(f"\n=== SPLIT-HALF === H1 hits {len(disc)}, replicate in H2 {len(rep)} (chance ~{0.05 * len(disc):.1f})")
    for g, t in disc:
        print(f"  {g}:{t:16s} H1={halves[(g, t, 'H1')]:.4f} H2={halves[(g, t, 'H2')]:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:] or rb.DEFAULT_GAMES)
