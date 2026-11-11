"""
Randomness battery — the tests edge_audit.py does NOT cover.

Complements edge_audit.py (frequency, positional marginals, lag-1 repeat, lag-1
sum autocorrelation, pair dispersion) with:

  R1  Wald-Wolfowitz runs test on draw sum (above/below median)
  R2  Wald-Wolfowitz runs test on odd-ball count
  L1  Ljung-Box on draw sum, lags 1..10
  L2  Ljung-Box on odd-ball count, lags 1..10
  S1  adjacent-gap distribution of sorted draws vs Monte-Carlo uniform null
  S2  rate of draws containing a consecutive pair vs Monte-Carlo null
  K1  calendar: draw-sum by weekday (Kruskal-Wallis; daily/multi-day games)
  K2  drift: draw-sum by calendar year (Kruskal-Wallis)
  B2  per-ball repeat at lags 2..5 (pooled binomial z, exact base rate K/N)

Every test is also run on the first and second halves of the history.
Holm-Bonferroni is applied across the FULL family (all games x tests, full
sample only).  Anything with p<0.05 in the first half is then checked on the
second half — the Texas-sharpshooter guard: a real effect must replicate
without refitting.  A positive control (synthetic machine with a 1.5x hot ball
plus a sticky-sum AR(1) process) confirms the battery can actually fire.

Run:  python experiments/randomness_battery.py [GAME ...]
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config.loader import evaluate_config, load_config  # noqa: E402
from lib.data.io import load_data  # noqa: E402

DEFAULT_GAMES = ["NJ_Pick6", "NJ_Cash5", "Powerball", "Megamillions", "NJ_Millionaire4Life"]
N_SIM = 1000
rng = np.random.default_rng(20261110)


# ── helpers ────────────────────────────────────────────────────────────────
def runs_test(x):
    """Wald-Wolfowitz on above/below median (ties dropped). Returns p."""
    med = np.median(x)
    s = x[x != med] > med
    n1, n2 = s.sum(), (~s).sum()
    if n1 < 2 or n2 < 2:
        return np.nan
    runs = 1 + np.count_nonzero(s[1:] != s[:-1])
    n = n1 + n2
    mu = 2 * n1 * n2 / n + 1
    var = (mu - 1) * (mu - 2) / (n - 1)
    return 2 * stats.norm.sf(abs((runs - mu) / np.sqrt(var)))


def ljung_box(x, lags=10):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    denom = (x**2).sum()
    r = np.array([(x[k:] * x[:-k]).sum() / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * (r**2 / (n - np.arange(1, lags + 1))).sum()
    return stats.chi2.sf(q, lags)


def sim_draws(n_draws, n_balls, low, high, size=None):
    span = high - low + 1
    r = rng.random((n_draws, span))
    idx = np.argsort(r, axis=1)[:, :n_balls] + low
    return np.sort(idx, axis=1)


def gap_hist(draws, span):
    gaps = np.diff(draws, axis=1).ravel()
    return np.bincount(np.minimum(gaps, 12), minlength=13)[1:]  # gap 1..12+


def consec_rate(draws):
    return (np.diff(draws, axis=1) == 1).any(axis=1).mean()


def mc_p_gap(draws, n_balls, low, high):
    obs = gap_hist(draws, high - low + 1).astype(float)
    sims = np.array(
        [gap_hist(sim_draws(len(draws), n_balls, low, high), 0) for _ in range(N_SIM)], float
    )
    mu, sd = sims.mean(0), sims.std(0) + 1e-9
    stat = lambda h: (((h - mu) / sd) ** 2).sum()  # noqa: E731
    o = stat(obs)
    null = np.array([stat(s) for s in sims])
    return (1 + (null >= o).sum()) / (N_SIM + 1)


def mc_p_consec(draws, n_balls, low, high):
    o = consec_rate(draws)
    null = np.array(
        [consec_rate(sim_draws(len(draws), n_balls, low, high)) for _ in range(N_SIM)]
    )
    mu = null.mean()
    return (1 + (abs(null - mu) >= abs(o - mu)).sum()) / (N_SIM + 1)


def repeat_lag_p(draws, low, high, n_balls, lag):
    span = high - low + 1
    m = np.zeros((len(draws), span), bool)
    np.put_along_axis(m, draws - low, True, axis=1)
    a, b = m[:-lag], m[lag:]
    hits = (a & b).sum()
    trials = a.sum()
    p0 = n_balls / span
    return stats.binomtest(int(hits), int(trials), p0).pvalue if trials else np.nan


def battery(draws, dates, low, high, n_balls, include_mc=True):
    out = {}
    sums = draws.sum(1).astype(float)
    odd = (draws % 2).sum(1).astype(float)
    out["R1_runs_sum"] = runs_test(sums)
    out["R2_runs_odd"] = runs_test(odd)
    out["L1_ljungbox_sum"] = ljung_box(sums)
    out["L2_ljungbox_odd"] = ljung_box(odd)
    if include_mc:
        out["S1_gap_dist"] = mc_p_gap(draws, n_balls, low, high)
        out["S2_consecutive"] = mc_p_consec(draws, n_balls, low, high)
    if dates is not None:
        d = pd.Series(sums, index=pd.DatetimeIndex(dates))
        if d.index.dayofweek.nunique() > 1:
            g = [v.values for _, v in d.groupby(d.index.dayofweek.values)]
            out["K1_weekday_sum"] = stats.kruskal(*g).pvalue
        g = [v.values for _, v in d.groupby(d.index.year.values) if len(v) >= 20]
        if len(g) > 1:
            out["K2_year_drift_sum"] = stats.kruskal(*g).pvalue
    for lag in (2, 3, 4, 5):
        out[f"B2_repeat_lag{lag}"] = repeat_lag_p(draws, low, high, n_balls, lag)
    return out


def holm(items, alpha=0.05):
    items = sorted(items, key=lambda kv: kv[1])
    m, rejected, still = len(items), [], True
    for i, (k, p) in enumerate(items):
        still = still and p <= alpha / (m - i)
        if still:
            rejected.append(k)
    return rejected


def load_game(game):
    cfg = evaluate_config(load_config(game))
    data = load_data(game)
    cols = [f"Ball{b}" for b in cfg["game_balls"]]
    lo, hi = cfg["ball_game_range_low"], cfg["ball_game_range_high"]
    data = data[((data[cols] >= lo) & (data[cols] <= hi)).all(axis=1)].reset_index(drop=True)
    data["Date"] = pd.to_datetime(data["Date"])  # string dates sort lexicographically!
    data = data.sort_values("Date", kind="stable").reset_index(drop=True)
    return np.sort(data[cols].values.astype(int), axis=1), data["Date"].values, lo, hi, len(cols)


# ── positive control ───────────────────────────────────────────────────────
def positive_control():
    print("\n=== POSITIVE CONTROL (synthetic broken machines) ===")
    n, k, lo, hi = 3000, 5, 1, 45
    # (a) sticky-sum machine: resample until sum correlates with previous draw
    d = [sim_draws(1, k, lo, hi)[0]]
    for _ in range(n - 1):
        best = None
        for _ in range(4):
            c = sim_draws(1, k, lo, hi)[0]
            if best is None or abs(c.sum() - d[-1].sum()) < abs(best.sum() - d[-1].sum()):
                best = c
        d.append(best)
    res = battery(np.array(d), None, lo, hi, k, include_mc=False)
    print("sticky-sum machine :", {a: f"{p:.1e}" for a, p in res.items() if a[0] in "RL"})
    # (b) consecutive-avoiding machine (rejects draws with adjacent numbers 70%)
    d = []
    while len(d) < n:
        c = sim_draws(1, k, lo, hi)[0]
        if (np.diff(c) == 1).any() and rng.random() < 0.7:
            continue
        d.append(c)
    res = battery(np.array(d), None, lo, hi, k)
    print("consec-avoiding    :", {a: f"{p:.1e}" for a, p in res.items() if a[0] == "S"})


# ── main ───────────────────────────────────────────────────────────────────
def main(games):
    full, halves = {}, {}
    for g in games:
        draws, dates, lo, hi, k = load_game(g)
        n = len(draws)
        print(f"{g}: {n} draws, {k}/{hi - lo + 1}, {str(dates[0])[:10]} → {str(dates[-1])[:10]}")
        for name, p in battery(draws, dates, lo, hi, k).items():
            full[(g, name)] = p
        h = n // 2
        for tag, sl in (("H1", slice(0, h)), ("H2", slice(h, n))):
            for name, p in battery(draws[sl], dates[sl], lo, hi, k).items():
                halves[(g, name, tag)] = p

    fam = [(f"{g}:{t}", p) for (g, t), p in full.items() if not np.isnan(p)]
    rej = holm(fam)
    print(f"\n=== FULL-SAMPLE: {len(fam)} tests, Holm family-wise α=0.05 ===")
    print(f"expected nominal p<0.05 by chance alone: {0.05 * len(fam):.1f}")
    print(f"observed nominal p<0.05:                 {sum(p < 0.05 for _, p in fam)}")
    print(f"survive Holm correction:                 {len(rej)} {rej}")
    print("\nsmallest p-values:")
    for k_, p in sorted(fam, key=lambda kv: kv[1])[:8]:
        print(f"  {k_:42s} p={p:.4f}  Holm threshold={0.05 / len(fam):.5f}")

    # replication: discovered in H1 (p<.05) → does it hold in H2?
    disc = [(g, t) for (g, t, tag), p in halves.items() if tag == "H1" and p < 0.05]
    rep = [(g, t) for (g, t) in disc if halves.get((g, t, "H2"), 1) < 0.05]
    print(f"\n=== SPLIT-HALF REPLICATION ===")
    print(f"significant in H1: {len(disc)}   replicate in H2 (p<.05): {len(rep)}   "
          f"(chance replication rate = 5% → ~{0.05 * len(disc):.1f})")
    for g, t in disc:
        print(f"  {g}:{t:20s} H1 p={halves[(g, t, 'H1')]:.4f}  H2 p={halves[(g, t, 'H2')]:.4f}")

    positive_control()


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_GAMES)
