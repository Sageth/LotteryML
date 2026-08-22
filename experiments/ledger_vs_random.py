"""
Is the live prediction ledger beating chance?

Scores every prediction file that is provably out-of-sample (its git add-time
precedes the draw it names) against two Monte Carlo nulls:

  A. unconstrained  - N uniform tickets per draw, no duplicates
  B. sum-filtered   - the same, but each ticket must pass the identical
                      mean/mode/stddev sum filter the pipeline applied

Null B is the one that matters. The pipeline only emits sum-filtered tickets,
and sum-filtered tickets sit in a denser region of ticket space, which lifts
average balls-matched without improving the chance of any prize. If a result is
significant under A but not under B, the "edge" is the filter and it is worth
nothing.

Balls matched is also reported alongside prize tiers. Tiers are the honest
test: for a fair draw they cannot be improved, whereas average balls matched
can be nudged by ticket placement alone.

Run: python experiments/ledger_vs_random.py [GAMEDIR ...]
"""

import json
import os
import subprocess
import sys
from collections import Counter
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config.loader import load_config, evaluate_config
from lib.data.io import load_data

N_SIMS = 20000
RNG = np.random.default_rng(20260822)


# ------------------------------------------------------------
# Leak control
# ------------------------------------------------------------
def git_add_time(path):
    """Author date of the commit that first added `path`, or None if untracked."""
    out = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%aI", "-1", "--", path],
        capture_output=True, text=True,
    ).stdout.strip()
    return pd.to_datetime(out) if out else None


def is_provably_prior(path, draw_date):
    """
    True if the file was committed before its draw could have happened.

    Draws are in the evening; anything committed on or before the draw date is
    prior to the draw. A file committed after its named date cannot be trusted
    as out-of-sample and is dropped.
    """
    t = git_add_time(path)
    if t is None:
        return False
    return t.tz_localize(None).normalize() <= pd.Timestamp(draw_date).normalize()


# ------------------------------------------------------------
# Scoring
# ------------------------------------------------------------
def _random_subsets(m, lo, hi, k):
    """m uniform k-subsets of [lo, hi], vectorised."""
    pool_n = hi - lo + 1
    return np.argsort(RNG.random((m, pool_n)), axis=1)[:, :k] + lo


def filtered_pool(sum_filter, lo, hi, k, target=40000, batch=40000, max_batches=40):
    """A pool of uniform k-subsets that pass `sum_filter` (all subsets if None)."""
    if sum_filter is None:
        return _random_subsets(target, lo, hi, k)
    keep = []
    got = 0
    for _ in range(max_batches):
        cand = _random_subsets(batch, lo, hi, k)
        ok = np.array([sum_filter(v) for v in cand.sum(axis=1)])
        keep.append(cand[ok])
        got += int(ok.sum())
        if got >= target:
            break
    pool = np.concatenate(keep)
    return pool[:target] if len(pool) else _random_subsets(target, lo, hi, k)


def hit_pmf_from_pool(pool, actual, k):
    """Empirical pmf of balls-matched for one draw, over a ticket pool."""
    hits = np.isin(pool, np.fromiter(actual, dtype=pool.dtype)).sum(axis=1)
    return np.bincount(hits, minlength=k + 1) / len(hits)


def exact_hypergeom_pmf(lo, hi, k):
    """Exact pmf of balls-matched for one uniform unconstrained ticket."""
    from math import comb
    n = hi - lo + 1
    total = comb(n, k)
    return np.array([comb(k, h) * comb(n - k, k - h) / total for h in range(k + 1)])


def best_of_n_pmf(pmf, n):
    """pmf of max over n iid draws: F_best(h) = F(h)**n."""
    cdf = np.cumsum(pmf)
    cdf_best = cdf ** n
    return np.diff(np.concatenate([[0.0], cdf_best]))


def filter_key(sum_filter, rec):
    if sum_filter is None:
        return None
    return (round(rec.get("mean_sum", 0), 2), rec.get("mode_sum"), round(rec.get("stddev", 0), 2))


def run_null(scored, per_draw_pmf, k, label):
    """Monte Carlo the mean best-of-N across draws from exact per-draw pmfs."""
    n_draws = len(scored)
    best_pmfs = np.stack([
        best_of_n_pmf(per_draw_pmf[i], scored[i]["n_tickets"]) for i in range(n_draws)
    ])
    vals = np.arange(k + 1)
    exp_mean = float((best_pmfs * vals).sum(axis=1).mean())

    # sample one "best" per draw per sim, vectorised via inverse-CDF
    cdfs = np.cumsum(best_pmfs, axis=1)
    u = RNG.random((N_SIMS, n_draws))
    draws = (u[:, :, None] > cdfs[None, :, :]).sum(axis=2)
    sim_mean = draws.mean(axis=1)
    return exp_mean, sim_mean


def tier_expectation(scored, per_draw_pmf, k):
    """Expected number of match-h tickets across the whole ledger."""
    exp = np.zeros(k + 1)
    for i, s in enumerate(scored):
        exp += per_draw_pmf[i] * s["n_tickets"]
    return exp


def make_sum_filter(rec, config):
    """Rebuild the exact filter predictor.py applied, from the recorded stats."""
    mean, mode, std = rec.get("mean_sum"), rec.get("mode_sum"), rec.get("stddev")
    if mean is None or mode is None or std is None:
        return None
    ma = config["mean_allowance"]
    mo = config["mode_allowance"]
    return lambda s: (
        mean * (1 - ma) <= s <= mean * (1 + ma)
        and mode * (1 - mo) <= s <= mode * (1 + mo)
        and (mean - std) <= s <= (mean + std)
    )


def analyse(gamedir):
    config = evaluate_config(load_config(gamedir))
    lo, hi = config["ball_game_range_low"], config["ball_game_range_high"]
    ball_cols = [f"Ball{i}" for i in config["game_balls"]]
    k = len(ball_cols)

    df = load_data(gamedir)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed").dt.strftime("%Y-%m-%d")
    actual_by_date = {r["Date"]: {int(r[c]) for c in ball_cols} for _, r in df.iterrows()}

    pred_dir = os.path.join(gamedir, "predictions")
    files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".json"))

    scored, no_draw, leak = [], 0, 0
    for fn in files:
        date = fn[:-5]
        if date not in actual_by_date:
            no_draw += 1
            continue
        path = os.path.join(pred_dir, fn)
        if not is_provably_prior(path, date):
            leak += 1
            continue
        with open(path) as fh:
            runs = json.load(fh)
        tickets = [r["predicted"] for r in runs if r.get("predicted")]
        if not tickets:
            continue
        actual = actual_by_date[date]
        rec = next((r for r in runs if "mean_sum" in r), {})
        scored.append({
            "date": date,
            "n_tickets": len(tickets),
            "hits": [len(actual & set(t)) for t in tickets],
            "filter": make_sum_filter(rec, config),
            "fkey": filter_key(make_sum_filter(rec, config), rec),
        })

    if not scored:
        print(f"{gamedir}: nothing scoreable")
        return

    obs_best = float(np.mean([max(s["hits"]) for s in scored]))
    obs_tiers = Counter(h for s in scored for h in s["hits"] if h >= 3)
    n_tix = sum(s["n_tickets"] for s in scored)

    print(f"\n{'=' * 74}")
    print(f"{gamedir}   {k} of {hi}   (jackpot odds 1 in {comb(hi - lo + 1, k):,})")
    print("=" * 74)
    print(f"prediction files                 : {len(files)}")
    print(f"  no matching draw yet           : {no_draw}")
    print(f"  DROPPED, not provably pre-draw : {leak}")
    print(f"  scored, leak-proof             : {len(scored)}"
          f"  [{scored[0]['date']} -> {scored[-1]['date']}]  {n_tix} tickets")
    print(f"\nOBSERVED  mean best-of-N hits    : {obs_best:.4f}")
    print(f"OBSERVED  prize tiers            : "
          + (", ".join(f"match-{h}: {c}" for h, c in sorted(obs_tiers.items(), reverse=True))
             or "none above match-2"))

    # ---- Null A: unconstrained uniform tickets -----------------------------
    pmf_a = exact_hypergeom_pmf(lo, hi, k)
    per_draw_a = [pmf_a] * len(scored)

    # ---- Null B: tickets passing the same sum filter the pipeline applied ---
    pools = {}
    per_draw_b = []
    for s in scored:
        key = s["fkey"]
        if key not in pools:
            pools[key] = filtered_pool(s["filter"], lo, hi, k)
        per_draw_b.append(hit_pmf_from_pool(pools[key], actual_by_date[s["date"]], k))

    results = {}
    for label, per_draw in [("A  unconstrained uniform", per_draw_a),
                            ("B  same sum filter      ", per_draw_b)]:
        exp_mean, sim = run_null(scored, per_draw, k, label)
        p = (np.sum(sim >= obs_best) + 1) / (N_SIMS + 1)
        exp_tiers = tier_expectation(scored, per_draw, k)
        print(f"\nNULL {label}")
        print(f"    expected mean best-of-N      : {exp_mean:.4f}   "
              f"(sim sd {sim.std():.4f})")
        print(f"    observed - expected          : {obs_best - exp_mean:+.4f}")
        print(f"    one-sided p                  : {p:.4f}"
              + ("   <-- significant at 0.05" if p < 0.05 else ""))
        print(f"    expected prize tiers         : "
              + ", ".join(f"match-{h}: {exp_tiers[h]:.2f}" for h in range(k, 2, -1)))
        results[label.strip()] = p

    print(f"\nVERDICT: ", end="")
    pa, pb = results["A  unconstrained uniform"], results["B  same sum filter"]
    if pb < 0.05:
        print("survives the sum-filtered null - worth investigating further.")
    elif pa < 0.05:
        print("significant ONLY against unconstrained tickets. The apparent edge\n"
              "         is the sum filter placing tickets in a denser region, not\n"
              "         predictive skill. It buys no additional prize probability.")
    else:
        print("indistinguishable from chance under both nulls.")
    return results


if __name__ == "__main__":
    for g in (sys.argv[1:] or ["NJ_Cash5"]):
        analyse(g)
