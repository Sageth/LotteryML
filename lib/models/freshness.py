"""Track which draw the saved models were last trained through.

Lets the nightly run retrain only when new draws have landed, instead of
every night (a plain --force-retrain also disables the "prediction already
exists" skip and would re-export on non-draw days).
"""
import os

MARKER_NAME = "trained_through.txt"


def _marker_path(gamedir, config):
    return os.path.join(gamedir, config["model_save_path"], MARKER_NAME)


def models_are_stale(gamedir, config, last_draw):
    """True when the models were not trained through `last_draw` (or never marked)."""
    try:
        with open(_marker_path(gamedir, config)) as fh:
            return fh.read().strip() != str(last_draw)
    except OSError:
        return True


def mark_trained(gamedir, config, last_draw):
    path = _marker_path(gamedir, config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"{last_draw}\n")
