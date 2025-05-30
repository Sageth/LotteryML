import os
import json
import shutil
import pandas as pd
from lib.models.predictor import should_skip_predictions

def test_should_skip_predictions():
    gamedir = "test_game"
    os.makedirs(os.path.join(gamedir, "predictions"), exist_ok=True)

    # Dummy logger
    class DummyLog:
        def info(self, msg): print(msg)
        def warning(self, msg): print(msg)
        def error(self, msg): print(msg)

    log = DummyLog()

    # Ensure no prediction file exists yet
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    prediction_path = os.path.join(gamedir, "predictions", f"{today_str}.json")

    if os.path.exists(prediction_path):
        os.remove(prediction_path)

    # Should return False first time
    assert not should_skip_predictions(gamedir, log), "should_skip_predictions should return False when no file exists"

    # Now create dummy prediction file
    with open(prediction_path, "w") as f:
        json.dump([], f)

    # Should return True now
    assert should_skip_predictions(gamedir, log), "should_skip_predictions should return True when file exists"

    print("✅ test_should_skip_predictions passed!")

    # Cleanup
    shutil.rmtree(gamedir)
