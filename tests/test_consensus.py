# tests/test_consensus.py

from lib.models.predictor import _consensus_prediction

BASE_CONFIG = {
    "game_balls": [1, 2, 3, 4, 5],
    "no_duplicates": True,
    "ball_game_range_low": 1,
    "ball_game_range_high": 60,
}


def _runs(*predictions):
    return [{"predicted": list(p)} for p in predictions]


def test_simple_majority_no_conflict():
    runs = _runs([1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 4, 5])
    assert _consensus_prediction(runs, BASE_CONFIG) == [1, 2, 3, 4, 5]


def test_no_duplicates_picks_next_most_voted():
    # Position 3's majority value (35) is already used by position 2;
    # consensus must fall back to position 3's runner-up (40).
    runs = _runs(
        [19, 30, 35, 35, 33],
        [19, 30, 35, 40, 33],
        [19, 30, 35, 35, 33],
    )
    result = _consensus_prediction(runs, BASE_CONFIG)
    assert result == [19, 30, 35, 40, 33]
    assert len(set(result)) == 5


def test_duplicates_allowed_when_config_off():
    config = dict(BASE_CONFIG, no_duplicates=False)
    runs = _runs([19, 30, 35, 35, 33], [19, 30, 35, 35, 33])
    assert _consensus_prediction(runs, config) == [19, 30, 35, 35, 33]


def test_extra_ball_may_repeat_main_ball():
    # 6 values = 5 mains + extra; extra (position 5) is a separate pool
    config = dict(BASE_CONFIG)
    runs = _runs([1, 2, 3, 4, 5, 3], [1, 2, 3, 4, 5, 3])
    result = _consensus_prediction(runs, config)
    assert result == [1, 2, 3, 4, 5, 3]


def test_all_candidates_used_falls_back_to_pool_then_range():
    # Every vote at every position is the same number — positions after the
    # first must fall back to unused numbers, never duplicating.
    runs = _runs([7, 7, 7, 7, 7], [7, 7, 7, 7, 7])
    result = _consensus_prediction(runs, BASE_CONFIG)
    assert result[0] == 7
    assert len(set(result)) == 5
    assert all(BASE_CONFIG["ball_game_range_low"] <= v <= BASE_CONFIG["ball_game_range_high"]
               for v in result)
