def test_conftest_dummy_log(dummy_log):
    dummy_log.info("Testing dummy_log fixture!")
    assert True

def test_conftest_test_config(test_config):
    assert "game_balls" in test_config

def test_conftest_clean_model_dir(clean_model_dir, test_config, game_dir):
    model_dir = game_dir / test_config["model_save_path"]
    assert model_dir.exists()

def test_conftest_dummy_data(dummy_data):
    assert len(dummy_data) == 350
    assert "Date" in dummy_data.columns