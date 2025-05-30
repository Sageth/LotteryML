# tests/test_loader.py

import os
import pytest
from lib.config.loader import load_config, evaluate_config

def test_load_config_success():
    # Test loading existing config
    config = load_config("NJ_Pick6")
    assert isinstance(config, dict), "Loaded config should be a dictionary"
    assert "game_balls" in config, "Config missing 'game_balls' key"

def test_load_config_failure():
    # Test loading a non-existent game dir
    with pytest.raises(FileNotFoundError):
        load_config("NON_EXISTENT_GAME")

def test_evaluate_config_range_conversion():
    # Test that 'game_balls' string is converted properly
    dummy_config = {
        "game_balls": "range(1, 7)"
    }
    evaluated = evaluate_config(dummy_config)
    assert isinstance(evaluated["game_balls"], range), "'game_balls' should be converted to a range"
    assert list(evaluated["game_balls"]) == [1, 2, 3, 4, 5, 6], "Incorrect range conversion"

def test_evaluate_config_preserves_other_fields():
    dummy_config = {
        "game_balls": "range(1, 4)",
        "ball_game_range_low": 1,
        "ball_game_range_high": 10
    }
    evaluated = evaluate_config(dummy_config)
    assert evaluated["ball_game_range_low"] == 1
    assert evaluated["ball_game_range_high"] == 10
