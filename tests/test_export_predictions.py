# tests/test_export_predictions.py

import os
import json
import shutil
from datetime import datetime
from lib.models.predictor import export_predictions

def test_export_predictions_creates_file(tmp_path):
    # Setup temp game dir
    gamedir = tmp_path / "test_game"
    predictions_dir = gamedir / "predictions"
    os.makedirs(predictions_dir, exist_ok=True)

    # Dummy predictions
    dummy_predictions = [
        {
            "run": 1,
            "date": datetime.now().strftime('%Y-%m-%d'),
            "predicted": [1, 2, 3, 4, 5, 6],
            "accuracy_scores": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            "predicted_sum": 21,
            "mean_sum": 20.0,
            "mode_sum": 19,
            "stddev": 5.0,
            "pass_checks": {
                "accuracy": True,
                "mean": True,
                "mode": True,
                "stddev": True
            }
        }
    ]

    # Dummy logger
    class DummyLog:
        def info(self, msg): print(msg)
        def warning(self, msg): print(msg)
        def error(self, msg): print(msg)

    log = DummyLog()

    # Run export_predictions
    export_predictions(dummy_predictions, str(gamedir), log)

    # Check file created
    today_str = datetime.now().strftime('%Y-%m-%d')
    expected_file = predictions_dir / f"{today_str}.json"

    assert expected_file.exists(), "Prediction JSON file was not created!"

    # Validate file content
    with open(expected_file, "r") as f:
        loaded = json.load(f)

    assert isinstance(loaded, list), "Exported predictions should be a list!"
    assert len(loaded) == 1, "Expected exactly 1 prediction in export!"

    print("✅ test_export_predictions_creates_file passed!")
