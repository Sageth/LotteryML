"""
Model-free statistical hypothesis tests on raw NJ Pick 6 draw history.

Tests three hypotheses:
  H1: Serial autocorrelation in draw sums (Ljung-Box + lag-1 Pearson r)
  H2: Hot/cold ball effect — does recent appearance predict next appearance?
  H3: Positional bias — is each ball position's distribution uniform?

Reads data via the existing io/features pipeline. Does NOT modify any production code.
Run: python experiments/hypothesis_tests.py
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config.loader import load_config, evaluate_config
from lib.data.io import load_data

GAMEDIR = "NJ_Pick6"
N_PERMUTATIONS = 10_000
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────
def load_raw():
    config = evaluate_config(load_config(GAMEDIR))
    data = load_data(GAMEDIR)
    ball_cols = [f"Ball{b}" for b in config["game_balls"]]
    lo = config["ball_game_range_low"]
    hi = config["ball_game_range_high"]
    mask = (data[ball_cols] >= lo).all(axis=1) & (data[ball_cols] <= hi).all(axis=1)
    data = data[mask].reset_index(drop=True)
    return data, config, ball_cols


# ──────────────────────────────────────────────────────────────
# H1: Sum autocorrelation
# ──────────────────────────────────────────────────────────────
def analyze_sum_autocorrelation(data, ball_cols):
    print("\n" + "=" * 60)
    print("H1: Serial autocorrelation in draw sums")
    print("=" * 60)

    sums = data[ball_cols].sum(axis=1).values.astype(float)
    n = len(sums)

    # Lag-k Pearson r for lags 1..10
    print(f"\n  {'Lag':>4}  {'r':>8}  {'p (t-test)':>12}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*12}")
    sig_lags = []
    for lag in range(1, 11):
        x, y = sums[:-lag], sums[lag:]
        r, p = stats.pearsonr(x, y)
        sig = " *" if p < 0.05 else ("  ~" if p < 0.10 else "")
        print(f"  {lag:>4}  {r:>8.4f}  {p:>12.4f}{sig}")
        if p < 0.05:
            sig_lags.append(lag)

    # Ljung-Box statistic (manual, avoids statsmodels dependency)
    # Q = n(n+2) * sum_{k=1}^{m} r_k^2 / (n-k)
    m = 10
    autocorrs = [np.corrcoef(sums[:-k], sums[k:])[0, 1] for k in range(1, m + 1)]
    Q = n * (n + 2) * sum(r**2 / (n - k) for k, r in enumerate(autocorrs, 1))
    p_lb = 1 - stats.chi2.cdf(Q, df=m)
    print(f"\n  Ljung-Box Q({m}) = {Q:.4f},  p = {p_lb:.4f}  "
          f"({'significant' if p_lb < 0.05 else 'not significant'} at α=0.05)")

    # Permutation test on lag-1 r
    r_obs, _ = stats.pearsonr(sums[:-1], sums[1:])
    perm_rs = np.array([
        stats.pearsonr(rng.permutation(sums[:-1]), sums[1:])[0]
        for _ in range(N_PERMUTATIONS)
    ])
    p_perm = (np.abs(perm_rs) >= np.abs(r_obs)).mean()
    print(f"\n  Permutation test (lag-1, n={N_PERMUTATIONS:,}): "
          f"r_obs={r_obs:.4f}, p={p_perm:.4f}  "
          f"({'significant' if p_perm < 0.05 else 'not significant'})")

    # Summary stats on sum distribution
    print(f"\n  Sum distribution: mean={sums.mean():.2f}, std={sums.std():.2f}, "
          f"skew={stats.skew(sums):.3f}, kurtosis={stats.kurtosis(sums):.3f}")
    _, p_norm = stats.shapiro(sums[:5000] if len(sums) > 5000 else sums)
    print(f"  Shapiro-Wilk normality test: p={p_norm:.4f}  "
          f"({'normal-ish' if p_norm > 0.05 else 'non-normal'})")

    return {"sig_lags": sig_lags, "ljung_box_p": p_lb, "lag1_r": r_obs, "lag1_p_perm": p_perm}


# ──────────────────────────────────────────────────────────────
# H2: Hot/cold ball effect
# ──────────────────────────────────────────────────────────────
def analyze_hot_cold(data, ball_cols, config):
    print("\n" + "=" * 60)
    print("H2: Hot/cold ball effect (serial autocorrelation per ball)")
    print("=" * 60)

    lo = config["ball_game_range_low"]
    hi = config["ball_game_range_high"]
    ball_nums = np.arange(lo, hi + 1)
    n_draws = len(data)

    # Build appearance matrix: (n_draws, n_numbers) boolean
    balls_arr = data[ball_cols].values  # (n, 6)
    appears = np.zeros((n_draws, hi + 1), dtype=bool)
    for col in range(balls_arr.shape[1]):
        appears[np.arange(n_draws), balls_arr[:, col].astype(int)] = True
    appears = appears[:, lo:]  # trim to valid range

    base_rate = appears.mean()  # expected p(ball appears) per draw

    # For each window w, compute P(appears | appeared in last w draws) vs base_rate
    print(f"\n  Base appearance rate per ball per draw: {base_rate:.4f}")
    print(f"  (Expected for uniform draw: {len(ball_cols) / len(ball_nums):.4f})\n")

    results = {}
    print(f"  {'Window':>8}  {'P(hot)':>8}  {'P(cold)':>9}  {'OR_hot':>8}  {'p_hot':>8}  {'p_cold':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}")

    for w in [1, 3, 5, 10, 20]:
        # For each position (draw, ball_num): was ball_num hot (appeared in last w)?
        # appeared_recently[i, b] = any(appears[max(0,i-w):i, b])
        hot_appeared = np.zeros(n_draws - w, dtype=float)
        cold_appeared = np.zeros(n_draws - w, dtype=float)
        n_hot = 0
        n_cold = 0

        for i in range(w, n_draws):
            for b_idx in range(len(ball_nums)):
                was_hot = appears[i - w:i, b_idx].any()
                did_appear = appears[i, b_idx]
                if was_hot:
                    hot_appeared[i - w] += did_appear
                    n_hot += 1
                else:
                    cold_appeared[i - w] += did_appear
                    n_cold += 1

        # Aggregate: what fraction of hot/cold slots had the ball appear?
        # This is flattened; compute per-draw average
        p_hot_arr = np.zeros(n_draws - w)
        p_cold_arr = np.zeros(n_draws - w)
        for i in range(w, n_draws):
            hot_mask = appears[i - w:i].any(axis=0)
            cold_mask = ~hot_mask
            if hot_mask.any():
                p_hot_arr[i - w] = appears[i][hot_mask].mean()
            if cold_mask.any():
                p_cold_arr[i - w] = appears[i][cold_mask].mean()

        p_hot = p_hot_arr.mean()
        p_cold = p_cold_arr.mean()
        or_hot = (p_hot / (1 - p_hot)) / (base_rate / (1 - base_rate)) if p_hot < 1 else np.nan

        # Permutation test: shuffle draw order, recompute p_hot - p_cold diff
        obs_diff = p_hot - p_cold
        perm_diffs = []
        perm_idx = np.arange(n_draws)
        for _ in range(min(1000, N_PERMUTATIONS)):  # fewer perms for speed
            shuffled = appears[rng.permutation(perm_idx)]
            ph = np.array([shuffled[i][shuffled[i-w:i].any(axis=0)].mean()
                          if shuffled[i-w:i].any(axis=0).any() else base_rate
                          for i in range(w, n_draws)]).mean()
            pc = np.array([shuffled[i][~shuffled[i-w:i].any(axis=0)].mean()
                          if (~shuffled[i-w:i].any(axis=0)).any() else base_rate
                          for i in range(w, n_draws)]).mean()
            perm_diffs.append(ph - pc)

        p_hot_perm = (np.array(perm_diffs) >= obs_diff).mean()
        p_cold_perm = (np.array(perm_diffs) <= obs_diff).mean()

        sig_hot = " *" if p_hot_perm < 0.05 else ""
        sig_cold = " *" if p_cold_perm < 0.05 else ""

        print(f"  {w:>8}  {p_hot:>8.4f}  {p_cold:>9.4f}  {or_hot:>8.3f}  "
              f"{p_hot_perm:>8.4f}{sig_hot}  {p_cold_perm:>8.4f}{sig_cold}")
        results[w] = {"p_hot": p_hot, "p_cold": p_cold, "or_hot": or_hot,
                      "obs_diff": obs_diff, "p_perm_hot": p_hot_perm, "p_perm_cold": p_cold_perm}

    print("\n  OR_hot > 1.0 = hot hand (recent balls more likely to repeat)")
    print("  OR_hot < 1.0 = gambler's fallacy (recent balls less likely to repeat)")
    return results


# ──────────────────────────────────────────────────────────────
# H3: Positional bias
# ──────────────────────────────────────────────────────────────
def analyze_positional_bias(data, ball_cols, config):
    print("\n" + "=" * 60)
    print("H3: Positional bias — is each position's distribution uniform?")
    print("=" * 60)

    lo = config["ball_game_range_low"]
    hi = config["ball_game_range_high"]
    n_values = hi - lo + 1

    # Expected under uniform: each value has prob 1/n_values
    # But these are ordered statistics — ball positions are sorted, so the
    # marginal distribution of Ball1 is the distribution of the minimum of
    # 6 draws from {lo,...,hi} without replacement. This is NOT uniform.
    # We test against the EMPIRICAL null by permutation.

    print(f"\n  Range: {lo}–{hi} ({n_values} possible values)")
    print(f"  Under IID uniform draws: Ball1 (min) should skew low, Ball6 (max) skew high\n")

    # Kolmogorov-Smirnov test vs. uniform [lo, hi]
    print(f"  {'Position':>10}  {'Mean':>7}  {'Std':>6}  {'KS stat':>8}  {'KS p':>8}  {'Skew':>7}  {'Verdict'}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*20}")

    positional_results = {}
    for col in ball_cols:
        vals = data[col].values.astype(float)
        ks_stat, ks_p = stats.kstest(vals, 'uniform', args=(lo, n_values))
        skewness = stats.skew(vals)
        verdict = "uniform-ish" if ks_p > 0.05 else "NON-UNIFORM *"
        print(f"  {col:>10}  {vals.mean():>7.2f}  {vals.std():>6.2f}  "
              f"{ks_stat:>8.4f}  {ks_p:>8.4f}  {skewness:>7.3f}  {verdict}")
        positional_results[col] = {"mean": vals.mean(), "std": vals.std(),
                                   "ks_p": ks_p, "skew": skewness}

    # Also: are there any individual numbers that appear significantly more/less often?
    print(f"\n  Top 5 most frequent balls across all positions:")
    all_balls = data[ball_cols].values.ravel()
    counts = pd.Series(all_balls.astype(int)).value_counts().head(5)
    expected = len(all_balls) / n_values
    for num, cnt in counts.items():
        chi2 = (cnt - expected) ** 2 / expected
        print(f"    Ball {num:>3}: {cnt} appearances (expected {expected:.1f}, chi2={chi2:.2f})")

    print(f"\n  Bottom 5 least frequent balls:")
    counts_asc = pd.Series(all_balls.astype(int)).value_counts().tail(5)
    for num, cnt in counts_asc.items():
        chi2 = (cnt - expected) ** 2 / expected
        print(f"    Ball {num:>3}: {cnt} appearances (expected {expected:.1f}, chi2={chi2:.2f})")

    # Global chi-squared over all numbers
    obs_counts = np.bincount(all_balls.astype(int), minlength=hi + 1)[lo:hi + 1]
    chi2_total, p_chi2 = stats.chisquare(obs_counts)
    print(f"\n  Global chi-squared across all {n_values} numbers: "
          f"χ²={chi2_total:.2f}, p={p_chi2:.4f}  "
          f"({'significant' if p_chi2 < 0.05 else 'not significant'})")

    return positional_results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    data, config, ball_cols = load_raw()
    print(f"Loaded {len(data)} draws after filtering.")

    h1 = analyze_sum_autocorrelation(data, ball_cols)
    h2 = analyze_hot_cold(data, ball_cols, config)
    h3 = analyze_positional_bias(data, ball_cols, config)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nH1 Sum autocorrelation:")
    print(f"  Significant lags (α=0.05): {h1['sig_lags'] if h1['sig_lags'] else 'none'}")
    print(f"  Ljung-Box p={h1['ljung_box_p']:.4f}  |  lag-1 permutation p={h1['lag1_p_perm']:.4f}")

    print(f"\nH2 Hot/cold effect:")
    for w, r in h2.items():
        direction = "hot hand" if r["or_hot"] > 1 else "gambler's fallacy"
        sig = "*" if r["p_perm_hot"] < 0.05 or r["p_perm_cold"] < 0.05 else "ns"
        print(f"  Window {w:>2}: OR={r['or_hot']:.3f} ({direction}), p={min(r['p_perm_hot'], r['p_perm_cold']):.4f} [{sig}]")

    print(f"\nH3 Positional bias:")
    for col, r in h3.items():
        sig = "*" if r["ks_p"] < 0.05 else "ns"
        print(f"  {col}: KS p={r['ks_p']:.4f} [{sig}], skew={r['skew']:.3f}")
