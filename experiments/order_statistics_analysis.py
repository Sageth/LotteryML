"""
Order statistics analysis and high-ball underrepresentation investigation.

Key question: Is the underrepresentation of balls 43-46 a data artifact
from game range changes, or genuine physical bias?

The game had multiple range eras:
  1981-1983: 1-36
  1984-1985: 1-39
  1986-1988: 1-42
  1989-1999: 1-46
  2000-2022: 1-49  (782 draws with balls 47-49 were filtered out)
  2023-now:  1-46

The prior chi-squared test computed expected = total_balls / 46, which is
wrong because balls 43-46 literally didn't exist before 1989. This test
uses era-adjusted expected counts as the proper null.

Also compares per-position empirical distributions against the theoretical
order statistics of a uniform draw (the correct positional null).

Run: python experiments/order_statistics_test.py
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GAMEDIR = "NJ_Pick6"
BALL_COLS = ["Ball1", "Ball2", "Ball3", "Ball4", "Ball5", "Ball6"]
N_BALLS = 6

# Game range eras — determined from yearly max-ball analysis
ERAS = [
    ("1981-1983", 1981, 1983, 36),
    ("1984-1985", 1984, 1985, 39),
    ("1986-1988", 1986, 1988, 42),
    ("1989-1999", 1989, 1999, 46),
    ("2000-2022", 2000, 2022, 49),  # 782 draws with 47-49 were filtered
    ("2023-now",  2023, 9999, 46),
]


def load_raw():
    df = pd.read_csv(f"{GAMEDIR}/source/nj-pick6.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df[BALL_COLS] = df[BALL_COLS].astype(int)
    df["max_ball"] = df[BALL_COLS].max(axis=1)
    df["year"] = df["Date"].dt.year
    return df


def assign_era(df):
    df = df.copy()
    df["era"] = "unknown"
    df["era_range_high"] = 0
    for label, y0, y1, hi in ERAS:
        mask = (df["year"] >= y0) & (df["year"] <= y1)
        df.loc[mask, "era"] = label
        df.loc[mask, "era_range_high"] = hi
    return df


def order_stat_pmf(v, m, N, k):
    """P(m-th order statistic = v) for drawing k from {1..N} without replacement."""
    if v < m or v > N - k + m:
        return 0.0
    num = comb(v - 1, m - 1) * comb(N - v, k - m)
    den = comb(N, k)
    return num / den


# ──────────────────────────────────────────────────────────────
# Section 1: Era-adjusted expected frequency (proper null)
# ──────────────────────────────────────────────────────────────
def analyze_era_adjusted_frequency(df):
    print("\n" + "=" * 60)
    print("Section 1: High-ball underrepresentation — is it an artifact?")
    print("=" * 60)

    df_era = assign_era(df)

    # For each era, compute how many draws included each ball value.
    # A ball value v is eligible in draws where era_range_high >= v AND max_ball <= era_range_high
    # (draws within the era's valid range that were not filtered out).
    # For filtered draws (draws where max_ball > current config range 46), they're excluded.
    filtered = df_era[df_era["max_ball"] <= 46].copy()  # matches what engineer_features keeps

    print(f"\n  Draws in filtered dataset: {len(filtered)}")
    print(f"  {'Era':<12} {'Draws':>7}  {'Range'}")
    for label, y0, y1, hi in ERAS:
        era_rows = filtered[filtered["era"] == label]
        print(f"  {label:<12} {len(era_rows):>7}  1–{hi}")

    # Compute era-adjusted expected count for each ball value 1..46
    expected = np.zeros(47, dtype=float)  # index = ball value
    for label, y0, y1, hi in ERAS:
        era_draws = filtered[filtered["era"] == label]
        n_era = len(era_draws)
        if n_era == 0:
            continue
        era_hi = min(hi, 46)  # cap at 46 (our config range)
        era_lo = 1
        eligible_range = era_hi - era_lo + 1
        # Under uniform draw without replacement: each eligible ball appears
        # N_BALLS / eligible_range times per draw on average
        per_draw = N_BALLS / eligible_range
        for v in range(era_lo, era_hi + 1):
            expected[v] += n_era * per_draw

    # Empirical counts
    all_balls = filtered[BALL_COLS].values.ravel()
    observed = np.bincount(all_balls, minlength=47).astype(float)

    print(f"\n  Ball-level comparison (balls 38–46):")
    print(f"  {'Ball':>6}  {'Observed':>10}  {'Expected':>10}  {'Ratio':>7}  {'Residual':>9}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*9}")
    for v in range(38, 47):
        obs = observed[v]
        exp = expected[v]
        ratio = obs / exp if exp > 0 else float("nan")
        residual = (obs - exp) / (exp ** 0.5) if exp > 0 else float("nan")
        flag = " *" if abs(residual) > 2 else ""
        print(f"  {v:>6}  {obs:>10.0f}  {exp:>10.1f}  {ratio:>7.3f}  {residual:>9.2f}{flag}")

    # Chi-squared on balls 1-46 with era-adjusted expected
    obs_vec = observed[1:47]
    exp_vec = expected[1:47]
    # Only include values with expected > 5 for chi-squared validity
    valid = exp_vec > 5
    chi2 = ((obs_vec[valid] - exp_vec[valid]) ** 2 / exp_vec[valid]).sum()
    df_chi = valid.sum() - 1
    p = 1 - stats.chi2.cdf(chi2, df_chi)
    print(f"\n  Era-adjusted chi-squared: χ²={chi2:.2f}, df={df_chi}, p={p:.4f}")
    print(f"  {'Significant — residual bias beyond range-change artifact' if p < 0.05 else 'Not significant — underrepresentation fully explained by range changes'}")

    return observed, expected


# ──────────────────────────────────────────────────────────────
# Section 2: Order statistics — empirical vs theoretical
# ──────────────────────────────────────────────────────────────
def analyze_order_statistics(df):
    print("\n" + "=" * 60)
    print("Section 2: Per-position order statistics (1-46 era only)")
    print("=" * 60)

    df_era = assign_era(df)
    # Use only draws from the 1-46 range eras for a clean comparison
    clean = df_era[df_era["era_range_high"] == 46].copy()
    print(f"\n  Using {len(clean)} draws from 1-46 eras (1989-1999, 2023-now)")

    N = 46
    k = N_BALLS

    print(f"\n  Theoretical E[Ball_m] under uniform draw from {{1..{N}}}, pick {k}:")
    print(f"  Formula: E[X_(m)] = m * (N+1) / (k+1)")
    print()
    print(f"  {'Position':>10}  {'Theoretical E':>14}  {'Empirical E':>12}  {'Std(emp)':>9}  {'KS p':>8}  {'Verdict'}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*12}  {'-'*9}  {'-'*8}  {'-'*20}")

    results = []
    for pos_idx, col in enumerate(BALL_COLS):
        m = pos_idx + 1
        theoretical_mean = m * (N + 1) / (k + 1)

        vals = clean[col].values.astype(int)
        emp_mean = vals.mean()
        emp_std = vals.std()

        # Build full theoretical PMF and CDF for KS test
        pmf = np.array([order_stat_pmf(v, m, N, k) for v in range(1, N + 1)])
        cdf_theoretical = np.cumsum(pmf)

        # Empirical CDF
        counts = np.bincount(vals, minlength=N + 1)[1:N + 1]
        cdf_empirical = np.cumsum(counts) / counts.sum()

        # KS statistic
        ks_stat = np.max(np.abs(cdf_empirical - cdf_theoretical))
        # KS p-value via simulation (exact for discrete case)
        n_sim = 5000
        rng = np.random.default_rng(42)
        sim_ks = []
        for _ in range(n_sim):
            sim_draws = np.array([
                np.sort(rng.choice(N, k, replace=False) + 1)
                for _ in range(len(clean))
            ])
            sim_counts = np.bincount(sim_draws[:, pos_idx], minlength=N + 1)[1:N + 1]
            sim_cdf = np.cumsum(sim_counts) / sim_counts.sum()
            sim_ks.append(np.max(np.abs(sim_cdf - cdf_theoretical)))
        ks_p = (np.array(sim_ks) >= ks_stat).mean()

        verdict = "matches theory" if ks_p > 0.05 else "DEVIATES *"
        print(f"  {col:>10}  {theoretical_mean:>14.2f}  {emp_mean:>12.2f}  "
              f"{emp_std:>9.2f}  {ks_p:>8.4f}  {verdict}")
        results.append({"col": col, "m": m, "theoretical_mean": theoretical_mean,
                        "emp_mean": emp_mean, "ks_p": ks_p, "ks_stat": ks_stat})

    return results


# ──────────────────────────────────────────────────────────────
# Section 3: Era 2000-2022 selection bias quantification
# ──────────────────────────────────────────────────────────────
def analyze_selection_bias(df):
    print("\n" + "=" * 60)
    print("Section 3: Selection bias from filtered 2000-2022 draws")
    print("=" * 60)

    df_era = assign_era(df)
    era_49 = df_era[df_era["era"] == "2000-2022"].copy()
    kept_49 = era_49[era_49["max_ball"] <= 46]
    excl_49 = era_49[era_49["max_ball"] > 46]

    print(f"\n  2000-2022 draws total:    {len(era_49)}")
    print(f"  Kept (max_ball ≤ 46):     {len(kept_49)}  ({100*len(kept_49)/len(era_49):.1f}%)")
    print(f"  Excluded (had 47-49):     {len(excl_49)}  ({100*len(excl_49)/len(era_49):.1f}%)")

    if len(excl_49) > 0 and len(kept_49) > 0:
        # In kept draws: how often does ball 43-46 appear vs in excluded?
        print(f"\n  Frequency of balls 43-46 in kept vs excluded 2000-2022 draws:")
        print(f"  (Expected: excluded draws should have FEWER 43-46 since they had 47-49 taking those high-value slots)")
        print(f"\n  {'Ball':>6}  {'Kept (per draw)':>16}  {'Excl (per draw)':>16}  {'Ratio kept/excl':>16}")
        print(f"  {'-'*6}  {'-'*16}  {'-'*16}  {'-'*16}")
        for v in range(43, 50):
            kept_rate = (kept_49[BALL_COLS] == v).sum().sum() / max(len(kept_49), 1)
            excl_rate = (excl_49[BALL_COLS] == v).sum().sum() / max(len(excl_49), 1)
            ratio = kept_rate / excl_rate if excl_rate > 0 else float("nan")
            note = " [kept]" if v <= 46 else " [excluded]"
            print(f"  {v:>6}  {kept_rate:>16.4f}  {excl_rate:>16.4f}  {ratio:>16.3f}{note}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading raw data...")
    df = load_raw()
    print(f"Total draws in CSV: {len(df)}")

    observed, expected = analyze_era_adjusted_frequency(df)
    ks_results = analyze_order_statistics(df)
    analyze_selection_bias(df)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    sig_positions = [r["col"] for r in ks_results if r["ks_p"] < 0.05]
    print(f"\nPositions that deviate from theoretical order statistics: "
          f"{sig_positions if sig_positions else 'none'}")
    print("\nConclusion on high-ball underrepresentation:")
    print("  See Section 1 chi-squared p-value above.")
    print("  p > 0.05 → artifact of range changes (no physical bias)")
    print("  p < 0.05 → genuine bias beyond what range changes explain")
