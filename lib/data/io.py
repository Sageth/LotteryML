# lib/data/io.py

import glob
import os
import pandas as pd


def load_data(gamedir: str) -> pd.DataFrame:
    """
    Load all CSV files from gamedir/source/, enforce deterministic ordering,
    validate schema consistency, and return a clean DataFrame.
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

    return data
