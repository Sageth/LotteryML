import json
import os

def load_config(gamedir: str) -> dict:
    with open(os.path.join(gamedir, "config/config.json"), 'r') as f:
        return json.load(f)

def evaluate_config(config: dict) -> dict:
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("range(") and value.endswith(")"):
            config[key] = eval(value)
    return config
