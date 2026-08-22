"""Tests for expected-payout ticket selection."""

import numpy as np
import pandas as pd
import pytest

from lib.models.unpopularity import (
    generate_unpopular_tickets,
    past_draw_set,
    score_ticket,
)

CONFIG = {"ball_game_range_low": 1, "ball_game_range_high": 45,
          "game_balls": [1, 2, 3, 4, 5]}


def s(ticket, past=None):
    return score_ticket(ticket, CONFIG, past or set())[0]


def test_score_is_bounded():
    for t in ([1, 2, 3, 4, 5], [2, 17, 29, 38, 44], [41, 42, 43, 44, 45]):
        assert 0.0 <= s(t) <= 1.0


def test_birthday_combination_scores_below_a_high_ball_combination():
    """All balls <= 31 is the single most over-played shape."""
    assert s([3, 11, 19, 24, 28]) < s([3, 11, 19, 38, 44])


def test_consecutive_runs_are_penalised():
    assert s([4, 5, 18, 33, 41]) < s([4, 9, 18, 33, 41])


def test_arithmetic_progressions_are_penalised():
    """5-15-25-35-45 is evenly spaced; the comparison ticket has no 3-term run.
    Both hold two balls above 31 and one single digit, so only the
    progression term differs."""
    assert s([5, 15, 25, 35, 45]) < s([5, 16, 26, 35, 45])


def test_previously_drawn_combination_is_penalised():
    ticket = [2, 17, 29, 38, 44]
    assert s(ticket, past={(2, 17, 29, 38, 44)}) < s(ticket, past=set())


def test_past_draw_lookup_is_order_independent():
    past = {(2, 17, 29, 38, 44)}
    assert s([44, 2, 38, 17, 29], past=past) == s([2, 17, 29, 38, 44], past=past)


def test_past_draw_set_from_dataframe():
    df = pd.DataFrame([[5, 3, 1, 4, 2]], columns=[f"Ball{i}" for i in range(1, 6)])
    assert past_draw_set(df, CONFIG) == {(1, 2, 3, 4, 5)}


def test_generate_returns_valid_distinct_tickets():
    tickets = generate_unpopular_tickets(
        CONFIG, 10, past_draws=set(), candidates=20000,
        rng=np.random.default_rng(0),
    )
    assert len(tickets) == 10
    seen = set()
    for score, t in tickets:
        assert len(t) == 5 and len(set(t)) == 5
        assert all(1 <= b <= 45 for b in t)
        assert t == sorted(t)
        assert tuple(t) not in seen
        seen.add(tuple(t))


def test_generated_tickets_favour_balls_above_the_calendar_range():
    """The whole point: over-weight the numbers birthdays cannot reach."""
    tickets = generate_unpopular_tickets(
        CONFIG, 20, past_draws=set(), candidates=40000,
        rng=np.random.default_rng(1),
    )
    balls = [b for _, t in tickets for b in t]
    frac_high = sum(b > 31 for b in balls) / len(balls)
    assert frac_high > 14 / 45  # chance rate for balls 32-45


def test_generated_tickets_are_not_near_duplicates():
    """Strict top-N selection used to return tickets sharing 4 of 5 balls."""
    tickets = generate_unpopular_tickets(
        CONFIG, 10, past_draws=set(), candidates=40000,
        rng=np.random.default_rng(2),
    )
    for i, (_, a) in enumerate(tickets):
        for _, b in tickets[i + 1:]:
            assert len(set(a) & set(b)) <= 3


def test_selection_never_restricts_the_reachable_ball_range():
    """Every ball must remain reachable -- restricting coverage forfeits wins."""
    tickets = generate_unpopular_tickets(
        CONFIG, 200, past_draws=set(), candidates=60000,
        rng=np.random.default_rng(3),
    )
    used = {b for _, t in tickets for b in t}
    assert min(used) <= 12 and max(used) == 45
