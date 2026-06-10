from datetime import date

import pytest

from lib.data.schedule import next_draw_dates, prediction_start_date


# 2026-06-09 is a Tuesday
TUESDAY = date(2026, 6, 9)
PICK6_CONFIG = {"draw_days": ["Monday", "Thursday", "Saturday"]}


def test_next_draw_dates_respects_draw_days():
    dates = next_draw_dates(PICK6_CONFIG, 3, start=TUESDAY)
    # From Tuesday: Thursday 11th, Saturday 13th, Monday 15th
    assert dates == ["2026-06-11", "2026-06-13", "2026-06-15"]


def test_start_on_draw_day_is_included():
    monday = date(2026, 6, 8)
    dates = next_draw_dates(PICK6_CONFIG, 1, start=monday)
    assert dates == ["2026-06-08"]


def test_no_draw_days_means_daily():
    dates = next_draw_dates({}, 3, start=TUESDAY)
    assert dates == ["2026-06-09", "2026-06-10", "2026-06-11"]


def test_zero_or_negative_n_returns_empty():
    assert next_draw_dates(PICK6_CONFIG, 0, start=TUESDAY) == []
    assert next_draw_dates(PICK6_CONFIG, -1, start=TUESDAY) == []


def test_case_insensitive_weekday_names():
    dates = next_draw_dates({"draw_days": ["monday", "THURSDAY"]}, 2, start=TUESDAY)
    assert dates == ["2026-06-11", "2026-06-15"]


def test_invalid_weekday_raises():
    with pytest.raises(ValueError, match="Funday"):
        next_draw_dates({"draw_days": ["Funday"]}, 1, start=TUESDAY)


def test_defaults_to_today_without_start():
    dates = next_draw_dates({}, 1)
    assert dates == [date.today().isoformat()]


def test_prediction_start_is_today_when_data_is_behind():
    # last recorded draw was days ago — predictions start today
    assert prediction_start_date(date(2026, 6, 5), today=TUESDAY) == TUESDAY


def test_prediction_start_skips_already_recorded_draw():
    # today's draw is already in the data — start tomorrow
    assert prediction_start_date(TUESDAY, today=TUESDAY) == date(2026, 6, 10)


def test_prediction_start_defaults_to_today():
    assert prediction_start_date(date(2020, 1, 1)) == date.today()
