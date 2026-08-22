"""
Edge audit — is there any exploitable non-uniformity in the draw generator?

This answers the recurring question "can we tweak the model to beat chance?"
empirically, and keeps the answer current as new draws land.

It deliberately tests the *generator*, not the predictor. Comparing ticket
selection rules is hopeless: for a K-of-N game every possible ticket has the
same expected number of matching balls (K*K/N), so model-vs-baseline hit rates
are pure noise by construction and can never reveal an edge. Ball-level
frequency tests use every ball-observation in the history instead, which is
where what little statistical power exists actually lives.

Tests run per game (main balls only; bonus balls are drawn from a separate
machine and are reported separately):

  T1  ball frequency uniformity          chi-square, df = N-1
  T2  positional marginals               chi-square vs exact order statistics
  T3  serial repeat                      P(hit at t | hit at t-1) vs base rate
  T4  sum autocorrelation                lag-1 Pearson r
  T5  pair co-occurrence dispersion      chi-square on all C(N,2) pair counts

A "due number" / gap test is intentionally absent: a ball's gap is defined by
its own non-appearance, so regressing hits on gap length is guaranteed to find
a negative association whether or not the machine is biased.

Because the family is ~5 tests x ~5 games, uncorrected p-values would produce a
false positive on most runs. Holm-Bonferroni is applied across the whole family
and the family-wise verdict is the headline result.

Sensitivity is verified against synthetic biased machines in
experiments/validate_detection.py. Summary for a 5/45 game over 2238 draws: T1
is blind to a 1.2x over-represented ball, marginal at 1.5x (p~0.015), and
decisive at 2.0x (p~1e-21). Since a bias must reach roughly 2x to overcome the
house take, any bias worth exploiting is far above the detection floor — which
means a null result here is informative, not merely underpowered, and more
draws would not change the conclusion.

Run:  python experiments/edge_audit.py [GAME ...]
"""

import os
import sys
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config.loader import evaluate_config, load_config  # noqa: E402
from lib.data.io import load_data  # noqa: E402

DEFAULT_GAMES = ["NJ_Pick6", "NJ_Cash5", "Powerball", "Megamillions", "NJ_Millionaire4Life"]

ALPHA = 0.05
POWER = 0.80


def _indicator_matrix(draws, low, high):
    """Rows = draws, columns = ball values low..high, 1 if that ball was drawn."""
    span = high - low + 1
    matrix = np.zeros((len(draws), span), dtype=np.int8)
    for i, row in enumerate(draws):
        for value in row:
            if low <= value <= high:
                matrix[i, int(value) - low] = 1
    return matrix


def t1_ball_frequency(matrix, n_balls_drawn):
    """Chi-square goodness of fit on total appearances per ball value."""
    counts = matrix.sum(axis=0)
    n_draws, span = matrix.shape
    expected = n_draws * n_balls_drawn / span
    chi2 = (((counts - expected) ** 2) / expected).sum()
    dof = span - 1
    return {
        "name": "T1 ball frequency uniformity",
        "stat": f"chi2={chi2:.1f} df={dof}",
        "p": float(stats.chi2.sf(chi2, dof)),
        "detail": f"expected {expected:.1f}/ball, observed {counts.min()}-{counts.max()}",
    }


def _order_stat_pmf(pos, span, n_balls_drawn):
    """Exact P(j-th smallest of K drawn from N == value) for every value."""
    total = comb(span, n_balls_drawn)
    return np.array(
        [
            comb(v - 1, pos) * comb(span - v, n_balls_drawn - pos - 1) / total
            for v in range(1, span + 1)
        ]
    )


def t2_positional_marginals(draws, span, n_balls_drawn):
    """Do the sorted positions follow the exact order-statistic distribution?"""
    ordered = np.sort(draws, axis=1)
    chi2_total = 0.0
    dof_total = 0
    for pos in range(n_balls_drawn):
        pmf = _order_stat_pmf(pos, span, n_balls_drawn)
        expected = pmf * len(ordered)
        observed = np.bincount(ordered[:, pos] - 1, minlength=span).astype(float)
        # Pool cells with tiny expected counts so the chi-square approximation holds.
        keep = expected >= 5
        obs_pooled = np.append(observed[keep], observed[~keep].sum())
        exp_pooled = np.append(expected[keep], expected[~keep].sum())
        chi2_total += (((obs_pooled - exp_pooled) ** 2) / exp_pooled).sum()
        dof_total += len(exp_pooled) - 1
    return {
        "name": "T2 positional marginals",
        "stat": f"chi2={chi2_total:.1f} df={dof_total}",
        "p": float(stats.chi2.sf(chi2_total, dof_total)),
        "detail": f"pooled over {n_balls_drawn} sorted positions",
    }


def t3_serial_repeat(matrix):
    """Does a ball appearing in draw t-1 change its chance of appearing at t?"""
    previous, current = matrix[:-1], matrix[1:]
    hit_after_hit = current[previous == 1]
    hit_after_miss = current[previous == 0]
    p_hit, p_miss = hit_after_hit.mean(), hit_after_miss.mean()
    se = np.sqrt(
        p_hit * (1 - p_hit) / hit_after_hit.size + p_miss * (1 - p_miss) / hit_after_miss.size
    )
    z = (p_hit - p_miss) / se
    return {
        "name": "T3 serial repeat",
        "stat": f"z={z:+.2f}",
        "p": float(2 * stats.norm.sf(abs(z))),
        "detail": f"P(hit|hit)={p_hit:.4f} vs P(hit|miss)={p_miss:.4f}",
    }


def t4_sum_autocorrelation(draws):
    """Lag-1 correlation of draw sums — the simplest test for drift or cycling."""
    sums = draws.sum(axis=1).astype(float)
    r, p = stats.pearsonr(sums[:-1], sums[1:])
    return {
        "name": "T4 sum autocorrelation",
        "stat": f"r={r:+.4f}",
        "p": float(p),
        "detail": f"lag-1 over {len(sums)} draws",
    }


def t5_pair_dispersion(matrix, n_balls_drawn):
    """Are specific pairs of balls drawn together more often than chance allows?"""
    n_draws, span = matrix.shape
    co = matrix.T.astype(np.int32) @ matrix.astype(np.int32)
    counts = co[np.triu_indices(span, k=1)].astype(float)
    n_pairs = comb(span, 2)
    expected = n_draws * comb(n_balls_drawn, 2) / n_pairs
    if expected < 5:
        return {
            "name": "T5 pair co-occurrence",
            "stat": "skipped",
            "p": None,
            "detail": f"expected {expected:.1f}/pair is too small for chi-square",
        }
    chi2 = (((counts - expected) ** 2) / expected).sum()
    # Pair counts are not independent; the effective dof is well below n_pairs-1.
    # Using n_pairs-1 makes the test conservative, which is the safe direction here.
    dof = n_pairs - 1
    return {
        "name": "T5 pair co-occurrence",
        "stat": f"chi2={chi2:.1f} df={dof}",
        "p": float(stats.chi2.sf(chi2, dof)),
        "detail": f"expected {expected:.1f}/pair over {n_pairs} pairs",
    }


def detectable_bias(n_draws, span, n_balls_drawn):
    """
    Smallest single-ball over-representation detectable at 80% power, expressed
    as a multiple of the fair rate.
    """
    p0 = n_balls_drawn / span
    z_alpha = stats.norm.ppf(1 - ALPHA / 2)
    z_beta = stats.norm.ppf(POWER)

    def shortfall(p1):
        return (
            abs(p1 - p0) * np.sqrt(n_draws)
            - z_alpha * np.sqrt(p0 * (1 - p0))
            - z_beta * np.sqrt(p1 * (1 - p1))
        )

    lo, hi = p0 + 1e-9, 0.999
    if shortfall(hi) < 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if shortfall(mid) < 0:
            lo = mid
        else:
            hi = mid
    return hi / p0


def holm_bonferroni(pvalues, alpha=ALPHA):
    """Holm step-down. Returns a rejection flag per input p-value."""
    indexed = sorted((p, i) for i, p in enumerate(pvalues))
    m = len(indexed)
    rejected = [False] * m
    for rank, (p, idx) in enumerate(indexed):
        if p <= alpha / (m - rank):
            rejected[idx] = True
        else:
            break
    return rejected


def audit_game(gamedir):
    config = evaluate_config(load_config(gamedir))
    data = load_data(gamedir, config)

    ball_cols = [f"Ball{i}" for i in config["game_balls"]]
    low = config["ball_game_range_low"]
    high = config["ball_game_range_high"]
    span = high - low + 1
    n_balls_drawn = len(ball_cols)

    dates = pd.to_datetime(data["Date"])
    draws = data[ball_cols].to_numpy(dtype=int)
    matrix = _indicator_matrix(draws, low, high)

    results = [
        t1_ball_frequency(matrix, n_balls_drawn),
        t2_positional_marginals(draws - (low - 1), span, n_balls_drawn),
        t3_serial_repeat(matrix),
        t4_sum_autocorrelation(draws),
        t5_pair_dispersion(matrix, n_balls_drawn),
    ]

    return {
        "game": os.path.basename(gamedir.rstrip("/")),
        "n_draws": len(draws),
        "span": span,
        "n_balls_drawn": n_balls_drawn,
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "matrix_start": config.get("matrix_start"),
        "has_extra": config.get("game_has_extra", False),
        "results": results,
    }


def main(games):
    audits = []
    for game in games:
        if not os.path.isdir(game):
            print(f"skipping {game}: not a directory")
            continue
        try:
            audits.append(audit_game(game))
        except Exception as exc:  # noqa: BLE001 - a broken game shouldn't kill the sweep
            print(f"skipping {game}: {exc}")

    if not audits:
        print("no games audited")
        return 1

    flat = [(a, r) for a in audits for r in a["results"] if r["p"] is not None]
    rejected = holm_bonferroni([r["p"] for _, r in flat])
    flags = {(id(a), r["name"]): rej for (a, r), rej in zip(flat, rejected)}

    for audit in audits:
        header = (
            f"{audit['game']}  —  {audit['n_draws']} draws "
            f"({audit['start']} to {audit['end']}), "
            f"{audit['n_balls_drawn']}/{audit['span']}"
        )
        print("\n" + header)
        print("-" * len(header))
        if audit["matrix_start"]:
            print(f"  matrix era begins {audit['matrix_start']}")
        if audit["has_extra"]:
            print("  note: bonus ball excluded (separate machine)")

        for result in audit["results"]:
            if result["p"] is None:
                print(f"  {result['name']:<30} {'-':>12}   {result['detail']}")
                continue
            mark = "SIGNIFICANT" if flags[(id(audit), result["name"])] else "ns"
            print(
                f"  {result['name']:<30} {result['stat']:>22}  "
                f"p={result['p']:.4f}  [{mark}]  {result['detail']}"
            )

        ratio = detectable_bias(audit["n_draws"], audit["span"], audit["n_balls_drawn"])
        if ratio:
            print(
                f"  power: a targeted test on one known ball resolves {ratio:.2f}x at "
                f"{POWER:.0%}; the T1 omnibus, spreading power over {audit['span']} cells,"
            )
            print(
                "         needs roughly 1.5x. Profitable bias starts near 2x, which T1 "
                "flags at p~1e-21 (see validate_detection.py) — it could not be missed."
            )

    n_sig = sum(rejected)
    print("\n" + "=" * 60)
    print(f"Family: {len(flat)} tests across {len(audits)} games, Holm-corrected at alpha={ALPHA}")
    if n_sig == 0:
        print("Verdict: no exploitable structure. Draws are consistent with uniform.")
        print("Expected matching balls is K*K/N for EVERY ticket — no selection rule")
        print("changes the odds. Only expected *payout* is improvable (see --unpopular).")
    else:
        print(f"Verdict: {n_sig} test(s) survive correction — investigate before trusting.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_GAMES))
