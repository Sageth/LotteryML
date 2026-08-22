# lib/data/io.py

import glob
import json
import os
import pandas as pd


def _matrix_start(gamedir: str, config: dict | None):
    """
    Return the game's `matrix_start` cutoff as a Timestamp, or None.

    Lottery games periodically change their ball matrix (NJ Cash 5 has run as
    5/38, 5/40, 5/43 and now 5/45). Draws from an earlier matrix are samples
    from a *different distribution*: high balls look artificially cold because
    they were impossible, and the sum statistics that drive the prediction
    filter are centred on the wrong value. `matrix_start` truncates history to
    the first draw of the current matrix.
    """
    if config is None:
        cfg_path = os.path.join(gamedir, "config", "config.json")
        if not os.path.exists(cfg_path):
            return None
        with open(cfg_path) as fh:
            config = json.load(fh)

    raw = config.get("matrix_start")
    return pd.to_datetime(raw) if raw else None


def load_data(gamedir: str, config: dict | None = None) -> pd.DataFrame:
    """
    Load all CSV files from gamedir/source/, enforce deterministic ordering,
    validate schema consistency, and return a clean DataFrame.

    If the game config defines `matrix_start`, draws from before that date are
    dropped so the model only ever sees the current ball matrix. Exact
    duplicate rows are also removed. `config` is read from the game directory
    when not supplied.
    """

    source_dir = os.path.join(gamedir, "source")
    csv_files = sorted(glob.glob(os.path.join(source_dir, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    # Load all CSVs
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)

        # Basic validation: must contain Date + at least one Ball column
        required_prefix = "Ball"
        if "Date" not in df.columns:
            raise ValueError(f"File {f} is missing required 'Date' column")

        ball_cols = [c for c in df.columns if c.startswith(required_prefix)]
        if not ball_cols:
            raise ValueError(f"File {f} contains no Ball columns (expected Ball1, Ball2, ...)")

        dfs.append(df)

    # Concatenate with index reset
    data = pd.concat(dfs, ignore_index=True)

    # Enforce Date as string (feature engineering will convert to datetime)
    data["Date"] = data["Date"].astype(str)

    # Enforce numeric types for ball columns
    for col in data.columns:
        if col.startswith("Ball"):
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows where any ball is NaN after coercion
    before = len(data)
    data = data.dropna(subset=[c for c in data.columns if c.startswith("Ball")])
    after = len(data)

    if after < before:
        print(f"[load_data] Dropped {before - after} rows due to invalid numeric ball values")

    # Exact duplicate draws (same date, same numbers) are source-data errors.
    # They double-count in every frequency feature, so drop them.
    before = len(data)
    data = data.drop_duplicates()
    if len(data) < before:
        print(f"[load_data] Dropped {before - len(data)} exact duplicate rows")

    # Truncate to the current ball matrix, if the game declares one.
    cutoff = _matrix_start(gamedir, config)
    if cutoff is not None:
        parsed = pd.to_datetime(data["Date"], format="mixed")
        before = len(data)
        data = data[parsed >= cutoff]
        if len(data) < before:
            print(
                f"[load_data] Dropped {before - len(data)} rows from before "
                f"matrix_start={cutoff.date()} (previous ball matrix)"
            )

    return data.reset_index(drop=True)
