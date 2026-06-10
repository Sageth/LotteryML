# lib/data/fetch.py
#
# Fetch the latest winning numbers from the NJ Lottery public API and append
# them to the game's source CSV, so results no longer need manual copy/paste.
#
# The API (https://www.njlottery.com/api/v1/draw-games/draws/page) covers all
# games this project tracks, including the multi-state ones (Powerball,
# Mega Millions, Cash4Life). Each draw's `results` list contains a "Regular"
# entry whose `primary` tokens are the winning numbers:
#   - plain numeric tokens are the main balls (e.g. "9", "15", "24")
#   - "M-03" is the multiplier marker (always skipped)
#   - "PB-20"/"CB-04"/"MB-12" style tokens carry the extra ball
#   - "B-43" is Cash 5's Bullseye (skipped — no extra column for that game)
# Draws that are CLOSED for sales but not yet drawn have no `results` key.

import glob
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

NJ_API_URL = "https://www.njlottery.com/api/v1/draw-games/draws/page"
_REQUEST_TIMEOUT = 30
# The API rejects requests without a browser-like User-Agent.
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
# NJ draws happen in Eastern Time; drawTime epoch millis must be converted
# in that zone or late-evening draws shift to the next day.
_EASTERN_OFFSET_FALLBACK = timezone(timedelta(hours=-5))


def _eastern_zone():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/New_York")
    except Exception:
        return _EASTERN_OFFSET_FALLBACK


def _http_get_json(url):
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_draws(game_name, date_from_ms, date_to_ms):
    """Query the NJ API for CLOSED draws of `game_name` in the given window."""
    params = urllib.parse.urlencode({
        "game-names": game_name,
        "status": "CLOSED",
        "date-from": date_from_ms,
        "date-to": date_to_ms,
        "size": 200,
        "page": 0,
    })
    payload = _http_get_json(f"{NJ_API_URL}?{params}")
    return payload.get("draws", [])


def _parse_draw(draw, config, tz):
    """
    Convert one API draw dict to {"Date": "MM/DD/YYYY", "Ball1": n, ...},
    or None if the draw has no published results or doesn't match the
    game's expected shape.
    """
    results = draw.get("results")
    if not results:
        return None  # closed for sales, not yet drawn

    regular = next((r for r in results if r.get("drawType") == "Regular"), None)
    if regular is None and len(results) == 1:
        regular = results[0]
    if regular is None or not regular.get("primary"):
        return None

    mains, extra = [], None
    for token in regular["primary"]:
        token = str(token)
        if "-" in token:
            prefix, _, value = token.partition("-")
            if prefix == "M":
                continue  # multiplier marker
            if config.get("game_has_extra", False) and extra is None:
                extra = int(value)
            # otherwise: auxiliary token with no CSV column (e.g. Bullseye)
        else:
            mains.append(int(token))

    num_main = len(config["game_balls"])
    if len(mains) != num_main:
        return None
    if config.get("game_has_extra", False) and extra is None:
        return None

    draw_date = datetime.fromtimestamp(draw["drawTime"] / 1000, tz=tz)
    row = {"Date": draw_date.strftime("%m/%d/%Y")}
    for i, value in enumerate(sorted(mains), start=1):
        row[f"Ball{i}"] = value
    if config.get("game_has_extra", False):
        row[config.get("game_extra_col", "BallExtra")] = extra
    row["_draw_date"] = draw_date.date()
    return row


def _source_csv_path(gamedir):
    csv_files = sorted(glob.glob(os.path.join(gamedir, "source", "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {gamedir}/source")
    if len(csv_files) > 1:
        raise ValueError(
            f"Multiple source CSVs in {gamedir}/source — "
            f"can't determine which to append to: {csv_files}"
        )
    return csv_files[0]


def fetch_new_draws(gamedir, config, log):
    """
    Fetch winning numbers newer than the last row of the game's source CSV
    and append them. Returns the number of rows appended (0 on failure —
    fetch errors are logged, not raised, so the pipeline can proceed with
    existing data).
    """
    game_name = config.get("fetch_game_name")
    if not game_name:
        log.warning("No fetch_game_name in config — skipping data update.")
        return 0

    csv_path = _source_csv_path(gamedir)
    existing = pd.read_csv(csv_path)
    last_date = pd.to_datetime(existing["Date"]).max().date()

    tz = _eastern_zone()
    now = datetime.now(tz)
    date_from = datetime.combine(last_date + timedelta(days=1),
                                 datetime.min.time(), tzinfo=tz)
    if date_from >= now:
        log.info(f"Source data already current (last draw {last_date}).")
        return 0

    log.info(f"Fetching {game_name} results after {last_date}...")
    try:
        draws = _fetch_draws(game_name,
                             int(date_from.timestamp() * 1000),
                             int(now.timestamp() * 1000))
    except Exception as e:
        log.warning(f"Fetch failed ({e}) — continuing with existing data.")
        return 0

    rows = [r for r in (_parse_draw(d, config, tz) for d in draws) if r]
    rows = [r for r in rows if r["_draw_date"] > last_date]
    if not rows:
        log.info("No new draws published yet.")
        return 0

    rows.sort(key=lambda r: r["_draw_date"])
    for r in rows:
        r.pop("_draw_date")

    # Append in the CSV's existing column order, zero-padding ball numbers
    # to match the established format (e.g. "01,13,15,...").
    columns = list(existing.columns)
    with open(csv_path, "a") as f:
        for row in rows:
            values = [row["Date"]] + [f"{int(row[c]):02d}" for c in columns[1:]]
            f.write(",".join(values) + "\n")

    log.info(f"Appended {len(rows)} new draws to {csv_path} "
             f"(through {rows[-1]['Date']}).")
    return len(rows)
