"""
Ticket selection that maximises expected payout, not probability of winning.

This module cannot improve your odds. For NJ Cash 5 they are 1 in 1,221,759
per ticket, and nothing here changes that number. What it changes is how much
you collect *if* you win.

Every Jersey Cash 5 prize is pari-mutuel: a tier's pool is split among the
tickets that reach it, so a jackpot won by two people pays each of them half.
Human number choice is famously non-uniform -- calendar dates dominate, which
massively over-plays 1-31 and especially 1-12; consecutive runs, arithmetic
patterns, straight lines on the play slip and previously-drawn combinations
all attract far more play than chance. Picking a combination other people
avoid does not make it more likely to come up, it just means fewer people to
split with when it does.

Sizing, from FY2025 sales of $176M over 365 draws at $2 per play:

    ~241,000 tickets per draw / 1,221,759 combinations = 0.197 expected
    jackpot winners per draw at uniform play. Expected share of a jackpot is
    E[1/(1+K)] for K ~ Poisson(lambda), so at a $1.96M jackpot:

        all balls <= 31 (birthday combo)   68.0% return per $2 staked
        average ticket                     79.6%
        2+ balls > 31, no patterns         83.2%

That is a 1.22x improvement in expected return, worth about $0.30 on a $2
ticket. It is still a losing bet -- 83% is not 100% -- and it is a variance
reduction in what you collect, not an edge on what gets drawn.

Deliberately does NOT use the trained models. Across 445 leak-proof live draws
the ensemble is indistinguishable from chance (pooled Stouffer Z = -0.33,
p = 0.63; see experiments/ledger_vs_random.py), and a chi-squared test over
NJ Cash 5's current matrix finds ball frequencies uniform (chi2 = 38.62,
p = 0.70). Sampling uniformly and selecting for unpopularity is therefore both
simpler and strictly better than re-ranking model output, which would inherit
the sum filter -- and the sum filter is actively EV-negative, because sums near
the mean are exactly where human pickers cluster.
"""

import numpy as np

# Weights are ordered by the strength of the evidence behind each effect.
# Calendar bias is by far the best-documented and gets the largest weight.
DEFAULT_WEIGHTS = {
    "above_calendar": 3.0,   # balls > 31 cannot be a day-of-month
    "above_twelve": 1.0,     # balls > 12 cannot be a month
    "no_consecutive": 1.5,   # consecutive pairs are over-played
    "no_arithmetic": 1.0,    # evenly-spaced picks (5,10,15,20,25) are over-played
    "sum_extremity": 1.0,    # mid-range sums are the crowded region
    "spread": 0.5,           # clustered picks mirror play-slip geometry
    "not_a_past_draw": 2.0,  # previously-drawn combinations get replayed
    "few_single_digits": 1.0,  # single digits are over-played beyond calendar bias
}


def _consecutive_pairs(t):
    s = np.sort(t)
    return int(np.sum(np.diff(s) == 1))


def _arithmetic_runs(t):
    """Count 3-term evenly-spaced runs, e.g. 5-10-15 within the ticket."""
    s = np.sort(t)
    d = np.diff(s)
    return int(np.sum(d[:-1] == d[1:]))


def score_ticket(ticket, config, past_draws=None, weights=None):
    """
    Unpopularity score in [0, 1]; higher means fewer people share your prize.

    `past_draws` is a set of sorted tuples of previously-drawn combinations.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    t = np.asarray(ticket, dtype=int)
    k = len(t)
    lo, hi = config["ball_game_range_low"], config["ball_game_range_high"]

    parts = {}

    # Calendar bias: the single largest effect. Day-of-month picks cap at 31,
    # month picks cap at 12.
    parts["above_calendar"] = min(np.sum(t > 31), 2) / 2.0 if hi > 31 else 1.0
    parts["above_twelve"] = min(np.sum(t > 12), 3) / 3.0 if hi > 12 else 1.0

    # Ball 1 and other single digits carry extra popularity beyond the calendar
    # effect (lucky numbers, top-left of the play slip).
    parts["few_single_digits"] = 1.0 - min(np.sum(t < 10), 2) / 2.0

    parts["no_consecutive"] = 1.0 if _consecutive_pairs(t) == 0 else 0.0
    parts["no_arithmetic"] = 1.0 if _arithmetic_runs(t) == 0 else 0.0

    # Sum extremity: distance from the centre of the sum distribution, scaled
    # so a ticket one sd out scores ~1. Human picks cluster at the centre.
    centre = k * (lo + hi) / 2.0
    sd = np.sqrt(k * ((hi - lo + 1) ** 2 - 1) / 12.0 * (1 - (k - 1) / (hi - lo)))
    parts["sum_extremity"] = min(abs(t.sum() - centre) / max(sd, 1e-9), 1.0)

    # Spread: tightly bunched tickets echo play-slip geometry (a column or a
    # block of adjacent numbers). Saturates at 60% of the range -- the aim is
    # to rule out clusters, not to force 1 and 45 into every ticket, which
    # would make all selected tickets near-identical and drag in ball 1, one
    # of the most heavily played numbers there is.
    parts["spread"] = min((t.max() - t.min()) / (0.6 * (hi - lo)), 1.0)

    parts["not_a_past_draw"] = (
        0.0 if past_draws and tuple(np.sort(t)) in past_draws else 1.0
    )

    total_w = sum(w[key] for key in parts)
    return float(sum(w[key] * v for key, v in parts.items()) / total_w), parts


def past_draw_set(data, config):
    """Set of sorted tuples of every combination already drawn in `data`."""
    cols = [f"Ball{i}" for i in config["game_balls"]]
    return {tuple(sorted(int(v) for v in row)) for row in data[cols].to_numpy()}


def generate_unpopular_tickets(config, n_tickets, past_draws=None,
                               candidates=200_000, rng=None, weights=None,
                               beta=12.0):
    """
    Sample `candidates` tickets uniformly, return `n_tickets` unpopular ones.

    Uniform sampling is the point: every combination is equally likely to be
    drawn, so restricting the candidate pool would only forfeit coverage. The
    selection step never changes which combinations *can* win, only which of
    the equally-likely ones we buy.

    Selection is stochastic -- weighted by score**beta -- rather than a strict
    top-N. Taking the argmax returns near-duplicate tickets that share most of
    their balls, which concentrates the whole batch on one region of ticket
    space and forfeits lower-tier coverage. Sampling keeps the batch diverse
    while still drawing overwhelmingly from the unpopular tail.
    """
    rng = rng or np.random.default_rng()
    lo, hi = config["ball_game_range_low"], config["ball_game_range_high"]
    k = len(config["game_balls"])

    pool = np.argsort(rng.random((candidates, hi - lo + 1)), axis=1)[:, :k] + lo
    scores = np.array([score_ticket(t, config, past_draws, weights)[0] for t in pool])

    w = scores ** beta
    w = w / w.sum()
    picked, seen = [], set()
    for idx in rng.choice(len(pool), size=min(len(pool), n_tickets * 50),
                          replace=False, p=w):
        t = sorted(int(v) for v in pool[idx])
        key = tuple(t)
        if key in seen:
            continue
        seen.add(key)
        picked.append((float(scores[idx]), t))
        if len(picked) == n_tickets:
            break
    return picked
