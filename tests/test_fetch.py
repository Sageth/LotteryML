import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import lib.data.fetch as fetch
from lib.data.fetch import fetch_new_draws

ET = ZoneInfo("America/New_York")

PICK6_CONFIG = {
    "fetch_game_name": "Pick 6",
    "game_balls": [1, 2, 3, 4, 5, 6],
    "game_has_extra": False,
}

POWERBALL_CONFIG = {
    "fetch_game_name": "Powerball",
    "game_balls": [1, 2, 3, 4, 5],
    "game_has_extra": True,
    "game_extra_col": "BallExtra",
}

CASH5_CONFIG = {
    "fetch_game_name": "Cash 5",
    "game_balls": [1, 2, 3, 4, 5],
    "game_has_extra": False,
}


def _draw_time_ms(year, month, day, hour=22):
    return int(datetime(year, month, day, hour, tzinfo=ET).timestamp() * 1000)


def _make_gamedir(tmp_path, header, rows):
    source = tmp_path / "source"
    source.mkdir()
    csv = source / "game.csv"
    csv.write_text("\n".join([header] + rows) + "\n")
    return tmp_path, csv


def _patch_api(monkeypatch, payload):
    calls = []

    def fake_get(url):
        calls.append(url)
        return payload

    monkeypatch.setattr(fetch, "_http_get_json", fake_get)
    return calls


def test_appends_new_pick6_draw(tmp_path, monkeypatch, dummy_log):
    gamedir, csv = _make_gamedir(
        tmp_path, "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Ball6",
        ["05/25/2026,01,02,03,04,05,06"],
    )
    payload = {"draws": [
        # closed for sales, not yet drawn — must be skipped
        {"gameName": "Pick 6", "drawTime": _draw_time_ms(2026, 6, 8)},
        {"gameName": "Pick 6", "drawTime": _draw_time_ms(2026, 6, 4), "results": [
            # Double Play listed first — fetcher must pick the Regular result
            {"drawType": "Double Play", "primary": ["1", "2", "3", "4", "5", "6", "M-00"]},
            {"drawType": "Regular", "primary": ["15", "9", "24", "34", "38", "41", "M-00"]},
        ]},
    ]}
    _patch_api(monkeypatch, payload)

    appended = fetch_new_draws(str(gamedir), PICK6_CONFIG, dummy_log)

    assert appended == 1
    lines = csv.read_text().strip().splitlines()
    # mains sorted ascending, zero-padded, dated in Eastern time
    assert lines[-1] == "06/04/2026,09,15,24,34,38,41"
    # round-trips through pandas
    df = pd.read_csv(csv)
    assert len(df) == 2


def test_extra_ball_game_maps_prefixed_token(tmp_path, monkeypatch, dummy_log):
    gamedir, csv = _make_gamedir(
        tmp_path, "Date,Ball1,Ball2,Ball3,Ball4,Ball5,BallExtra",
        ["05/25/2026,01,02,03,04,05,06"],
    )
    payload = {"draws": [
        {"gameName": "Powerball", "drawTime": _draw_time_ms(2026, 6, 8), "results": [
            {"drawType": "Regular", "primary": ["3", "24", "34", "43", "49", "M-03", "PB-20"]},
        ]},
    ]}
    _patch_api(monkeypatch, payload)

    assert fetch_new_draws(str(gamedir), POWERBALL_CONFIG, dummy_log) == 1
    assert csv.read_text().strip().splitlines()[-1] == "06/08/2026,03,24,34,43,49,20"


def test_bullseye_token_skipped_without_extra_col(tmp_path, monkeypatch, dummy_log):
    gamedir, csv = _make_gamedir(
        tmp_path, "Date,Ball1,Ball2,Ball3,Ball4,Ball5",
        ["05/25/2026,01,02,03,04,05"],
    )
    payload = {"draws": [
        {"gameName": "Cash 5", "drawTime": _draw_time_ms(2026, 6, 7), "results": [
            {"drawType": "Regular", "primary": ["6", "29", "40", "43", "44", "B-43", "M-03"]},
        ]},
    ]}
    _patch_api(monkeypatch, payload)

    assert fetch_new_draws(str(gamedir), CASH5_CONFIG, dummy_log) == 1
    assert csv.read_text().strip().splitlines()[-1] == "06/07/2026,06,29,40,43,44"


def test_draws_on_or_before_last_date_filtered(tmp_path, monkeypatch, dummy_log):
    gamedir, csv = _make_gamedir(
        tmp_path, "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Ball6",
        ["05/25/2026,01,02,03,04,05,06"],
    )
    payload = {"draws": [
        {"gameName": "Pick 6", "drawTime": _draw_time_ms(2026, 5, 25), "results": [
            {"drawType": "Regular", "primary": ["1", "2", "3", "4", "5", "6"]},
        ]},
    ]}
    _patch_api(monkeypatch, payload)

    assert fetch_new_draws(str(gamedir), PICK6_CONFIG, dummy_log) == 0
    assert len(csv.read_text().strip().splitlines()) == 2  # header + original row


def test_multiple_new_draws_appended_in_date_order(tmp_path, monkeypatch, dummy_log):
    gamedir, csv = _make_gamedir(
        tmp_path, "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Ball6",
        ["05/25/2026,01,02,03,04,05,06"],
    )
    payload = {"draws": [  # API returns newest first
        {"gameName": "Pick 6", "drawTime": _draw_time_ms(2026, 6, 6), "results": [
            {"drawType": "Regular", "primary": ["7", "8", "9", "10", "11", "12"]},
        ]},
        {"gameName": "Pick 6", "drawTime": _draw_time_ms(2026, 6, 4), "results": [
            {"drawType": "Regular", "primary": ["1", "2", "3", "4", "5", "6"]},
        ]},
    ]}
    _patch_api(monkeypatch, payload)

    assert fetch_new_draws(str(gamedir), PICK6_CONFIG, dummy_log) == 2
    lines = csv.read_text().strip().splitlines()
    assert lines[-2].startswith("06/04/2026")
    assert lines[-1].startswith("06/06/2026")


def test_wrong_ball_count_skipped(tmp_path, monkeypatch, dummy_log):
    gamedir, _ = _make_gamedir(
        tmp_path, "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Ball6",
        ["05/25/2026,01,02,03,04,05,06"],
    )
    payload = {"draws": [
        {"gameName": "Pick 6", "drawTime": _draw_time_ms(2026, 6, 4), "results": [
            {"drawType": "Regular", "primary": ["1", "2", "3"]},  # malformed
        ]},
    ]}
    _patch_api(monkeypatch, payload)

    assert fetch_new_draws(str(gamedir), PICK6_CONFIG, dummy_log) == 0


def test_network_error_returns_zero(tmp_path, monkeypatch, dummy_log):
    gamedir, csv = _make_gamedir(
        tmp_path, "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Ball6",
        ["05/25/2026,01,02,03,04,05,06"],
    )

    def boom(url):
        raise OSError("connection refused")

    monkeypatch.setattr(fetch, "_http_get_json", boom)

    assert fetch_new_draws(str(gamedir), PICK6_CONFIG, dummy_log) == 0
    assert len(csv.read_text().strip().splitlines()) == 2


def test_missing_fetch_game_name_returns_zero(tmp_path, dummy_log):
    gamedir, _ = _make_gamedir(
        tmp_path, "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Ball6",
        ["05/25/2026,01,02,03,04,05,06"],
    )
    config = dict(PICK6_CONFIG)
    del config["fetch_game_name"]
    assert fetch_new_draws(str(gamedir), config, dummy_log) == 0


def test_multiple_source_csvs_raises(tmp_path, dummy_log):
    gamedir, _ = _make_gamedir(
        tmp_path, "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Ball6",
        ["05/25/2026,01,02,03,04,05,06"],
    )
    (tmp_path / "source" / "other.csv").write_text("Date,Ball1\n01/01/2020,1\n")
    with pytest.raises(ValueError, match="Multiple source CSVs"):
        fetch_new_draws(str(gamedir), PICK6_CONFIG, dummy_log)


def test_append_to_file_without_trailing_newline(tmp_path, monkeypatch, dummy_log):
    gamedir, csv = _make_gamedir(
        tmp_path, "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Ball6",
        ["05/25/2026,01,02,03,04,05,06"],
    )
    csv.write_text(csv.read_text().rstrip("\n"))  # strip trailing newline
    payload = {"draws": [
        {"gameName": "Pick 6", "drawTime": _draw_time_ms(2026, 6, 4), "results": [
            {"drawType": "Regular", "primary": ["1", "2", "3", "4", "5", "6"]},
        ]},
    ]}
    _patch_api(monkeypatch, payload)

    assert fetch_new_draws(str(gamedir), PICK6_CONFIG, dummy_log) == 1
    df = pd.read_csv(csv)  # must not glue rows together
    assert len(df) == 2
    assert df.iloc[0].tolist() == ["05/25/2026", 1, 2, 3, 4, 5, 6]


def test_already_current_skips_fetch(tmp_path, monkeypatch, dummy_log):
    today = datetime.now(ET).strftime("%m/%d/%Y")
    gamedir, _ = _make_gamedir(
        tmp_path, "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Ball6",
        [f"{today},01,02,03,04,05,06"],
    )
    calls = _patch_api(monkeypatch, {"draws": []})

    assert fetch_new_draws(str(gamedir), PICK6_CONFIG, dummy_log) == 0
    assert calls == []  # no API request made
