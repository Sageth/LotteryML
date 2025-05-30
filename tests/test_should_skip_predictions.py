# tests/test_should_skip_predictions.py

import os
from datetime import datetime
from lib.models.predictor import should_skip_predictions

def test_should_skip_predictions(tmp_path):
    # Setup temp game dir
    gamedir = tmp_path / "test_game"
    predictions_dir = gamedir / "predictions"
    os.makedirs(predictions_dir, exist_ok=True)

    today_str = datetime.now().strftime('%Y-%m-%d')
    prediction_file = predictions_dir / f"{today_str}.json"

    # Dummy logger
    class DummyLog:
        def info(self, msg): print(msg)
        def warning(self, msg): print(msg)
        def error(self, msg): print(msg)

    log = DummyLog()

    # 1️⃣ First run → file does not exist → should return False
    assert not should_skip_predictions(str(gamedir), log), "Expected should_skip_predictions to return False when file does not exist!"

    # 2️⃣ Create dummy prediction file
    prediction_file.write_text("[]")

    # 3️⃣ Second run → file exists → should return True
    assert should_skip_predictions(str(gamedir), log), "Expected should_skip_predictions to return True when file exists!"

    print("✅ test_should_skip_predictions passed!")
