def test_conftest_dummy_log(dummy_log):
    dummy_log.info("Testing dummy_log fixture!")
    assert True

def test_conftest_test_config(test_config):
    assert "game_balls" in test_config

def test_conftest_clean_model_dir(clean_model_dir, test_config):
    import os
    assert os.path.exists(test_config["model_save_path"])

def test_conftest_dummy_data(dummy_data):
    assert len(dummy_data) == 100
    assert "Date" in dummy_data.columns