# lib/data/schedule.py

from datetime import date, timedelta


def prediction_start_date(last_draw_date, today=None):
    """
    First date eligible for a prediction: today, unless the source data
    already contains today's draw (e.g. an evening run after results were
    published), in which case start the day after the last recorded draw.
    Prevents exporting hindsight "predictions" for already-known outcomes.
    """
    today = today or date.today()
    return max(today, last_draw_date + timedelta(days=1))

_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def next_draw_dates(config, n, start=None):
    """
    Return the next n scheduled draw dates as ISO strings (YYYY-MM-DD),
    starting from `start` (inclusive; defaults to today).

    Uses the `draw_days` config field — a list of weekday names like
    ["Monday", "Thursday", "Saturday"]. If absent, every day is a draw day.
    """
    if n < 1:
        return []

    start = start or date.today()

    draw_days = config.get("draw_days")
    if draw_days:
        allowed = set()
        for day in draw_days:
            index = _WEEKDAY_INDEX.get(day.strip().lower())
            if index is None:
                raise ValueError(f"Invalid weekday in draw_days: {day!r}")
            allowed.add(index)
    else:
        allowed = set(range(7))

    dates = []
    current = start
    while len(dates) < n:
        if current.weekday() in allowed:
            dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates
