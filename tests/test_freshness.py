# tests/test_freshness.py
#
# The nightly run retrains only when the source data has draws newer than the
# ones the saved models were trained through (--retrain-on-new-data).

import datetime

from lib.models.freshness import MARKER_NAME, mark_trained, models_are_stale

CONFIG = {"model_save_path": "models"}


def test_stale_when_never_marked(tmp_path):
    assert models_are_stale(str(tmp_path), CONFIG, datetime.date(2026, 10, 24))


def test_fresh_after_marking_same_draw(tmp_path):
    mark_trained(str(tmp_path), CONFIG, datetime.date(2026, 10, 24))
    assert not models_are_stale(str(tmp_path), CONFIG, datetime.date(2026, 10, 24))


def test_stale_when_newer_draw_arrives(tmp_path):
    mark_trained(str(tmp_path), CONFIG, datetime.date(2026, 10, 24))
    assert models_are_stale(str(tmp_path), CONFIG, datetime.date(2026, 10, 26))


def test_mark_creates_models_dir_and_overwrites(tmp_path):
    mark_trained(str(tmp_path), CONFIG, datetime.date(2026, 10, 24))
    mark_trained(str(tmp_path), CONFIG, datetime.date(2026, 10, 26))
    assert (tmp_path / "models" / MARKER_NAME).read_text().strip() == "2026-10-26"
