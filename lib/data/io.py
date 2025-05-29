import os
import glob
import pandas as pd

def load_data(gamedir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(gamedir, "./source/*.csv"))
    return pd.concat([pd.read_csv(f) for f in csv_files])
