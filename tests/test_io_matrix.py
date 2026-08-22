"""Tests for load_data's matrix-era truncation and duplicate removal."""

import json
import os

import pandas as pd
import pytest

from lib.data.io import load_data


def _make_game(tmp_path, rows, config=None):
    """Write a minimal game directory with one source CSV and optional config."""
    gamedir = tmp_path / "TestGame"
    (gamedir / "source").mkdir(parents=True)
    pd.DataFrame(rows, columns=["Date", "Ball1", "Ball2", "Ball3"]).to_csv(
        gamedir / "source" / "draws.csv", index=False
    )
    if config is not None:
        (gamedir / "config").mkdir()
        (gamedir / "config" / "config.json").write_text(json.dumps(config))
    return str(gamedir)


ROWS = [
    ["01/01/2019", 1, 2, 3],   # old matrix
    ["06/15/2020", 4, 5, 6],   # old matrix
    ["06/29/2020", 7, 8, 9],   # first draw of current matrix
    ["07/01/2020", 10, 11, 12],
]


def test_matrix_start_drops_previous_matrix(tmp_path):
    gamedir = _make_game(tmp_path, ROWS, {"matrix_start": "2020-06-29"})
    data = load_data(gamedir)
    assert len(data) == 2
    assert data["Date"].tolist() == ["06/29/2020", "07/01/2020"]
    # index is reset so downstream positional logic stays valid
    assert data.index.tolist() == [0, 1]


def test_matrix_start_is_inclusive_of_the_cutoff_date(tmp_path):
    gamedir = _make_game(tmp_path, ROWS, {"matrix_start": "2020-06-29"})
    assert "06/29/2020" in load_data(gamedir)["Date"].tolist()


def test_no_matrix_start_keeps_all_rows(tmp_path):
    gamedir = _make_game(tmp_path, ROWS, {"ball_game_range_low": 1})
    assert len(load_data(gamedir)) == len(ROWS)


def test_missing_config_file_keeps_all_rows(tmp_path):
    gamedir = _make_game(tmp_path, ROWS, config=None)
    assert len(load_data(gamedir)) == len(ROWS)


def test_explicit_config_argument_overrides_config_file(tmp_path):
    gamedir = _make_game(tmp_path, ROWS, {"matrix_start": "2020-06-29"})
    data = load_data(gamedir, config={"matrix_start": "2019-01-01"})
    assert len(data) == len(ROWS)


def test_exact_duplicate_rows_are_dropped(tmp_path):
    rows = ROWS + [["07/01/2020", 10, 11, 12]]  # exact repeat of the last draw
    gamedir = _make_game(tmp_path, rows, {"matrix_start": "2020-06-29"})
    assert len(load_data(gamedir)) == 2


def test_same_date_different_numbers_is_kept(tmp_path):
    """Multi-draw games legitimately have two different draws on one date."""
    rows = ROWS + [["07/01/2020", 13, 14, 15]]
    gamedir = _make_game(tmp_path, rows, {"matrix_start": "2020-06-29"})
    assert len(load_data(gamedir)) == 3


@pytest.mark.parametrize(
    "gamedir,expected_high",
    [("NJ_Cash5", 45), ("NJ_Pick6", 46), ("Powerball", 69), ("Megamillions", 70)],
)
def test_real_games_contain_only_current_matrix_balls(gamedir, expected_high):
    """Regression guard: no shipped game may train on a previous ball matrix."""
    if not os.path.isdir(gamedir):
        pytest.skip(f"{gamedir} not present")
    config = json.load(open(os.path.join(gamedir, "config", "config.json")))
    data = load_data(gamedir)
    main_cols = [f"Ball{i}" for i in config["game_balls"]]
    assert data[main_cols].to_numpy().max() <= expected_high
    assert data[main_cols].to_numpy().min() >= config["ball_game_range_low"]
    assert config["ball_game_range_high"] == expected_high
