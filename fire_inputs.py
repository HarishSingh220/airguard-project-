"""
fire_inputs.py
Daily FIRMS stubble-fire features, read from the precomputed fire_daily_features.csv
(MODIS from 2022-01-01, VIIRS from 2024-01-17; columns before that are NaN, not zero).

The *_ti columns are the transport-weighted fire indices already computed with the day's wind.
They are SAME-DAY values here; build_features.py lags them, so the wind that weights a fire
never reaches a forecast of the same day.

This replaces the fire_features.fire_daily(...) call. The raw FIRMS archives are no longer
read by the feature build.
"""
from pathlib import Path

import pandas as pd

from io_utils import read_csv_checked

FIRE_DAILY_CSV = Path("fire_daily_features.csv")
FAMILY = "Stubble fires"


def load_fire_daily(path=FIRE_DAILY_CSV, use_viirs=True):
    """Return (daily dataframe indexed by date, {column: feature family})."""
    f = read_csv_checked(path, required_cols=["date"], parse_dates=["date"])
    f = f.set_index("date").sort_index()
    if f.index.duplicated().any():
        raise ValueError(f"{path} has duplicate dates")
    if not use_viirs:
        f = f[[c for c in f.columns if not c.startswith("viirs_")]]
    return f, {col: FAMILY for col in f.columns}
