"""
cams_inputs.py
Daily CAMS aerosol and composition features from cams_3h_ist.csv (3-hourly, IST, labelled
at :30, samples at 02:30, 05:30, ... 23:30).

Everything returned is a SAME-DAY value for each IST day. build_features.py lags it.
cams_daily_ist.csv is a plain daily mean of the 3-hourly file and is used only as a check.

Note: the supplied cams_pm25 averages about 267 and cams_pm10 about 378, which is far above
the station PM2.5 and PM10. Tree models do not care about scale, but check the unit of the
CAMS extraction before reading any CAMS number as ug/m3.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import read_csv_checked

CAMS_3H_CSV = Path("cams_3h_ist.csv")
CAMS_DAILY_CSV = Path("cams_daily_ist.csv")

VARS = ["cams_aod550", "cams_pm25", "cams_pm10", "cams_no2_col", "cams_o3_col_du"]
MAX_VARS = ["cams_aod550", "cams_pm25", "cams_no2_col"]
MIN_SAMPLES = 6             # of 8 per day
FAMILY = "Aerosol (CAMS)"


def build_cams_daily():
    """Return (daily dataframe indexed by IST date, {column: feature family})."""
    c = read_csv_checked(CAMS_3H_CSV, required_cols=["timestamp"] + VARS, parse_dates=["timestamp"])
    c = c.sort_values("timestamp")
    c["date"] = c["timestamp"].dt.floor("D")
    hr = c["timestamp"].dt.hour
    g = c.groupby("date")

    out = pd.DataFrame(index=g.size().index)
    for v in VARS:
        out[f"{v}_mean"] = g[v].mean()
    for v in MAX_VARS:
        out[f"{v}_max"] = g[v].max()
    out["cams_pm25_pm10_ratio"] = out["cams_pm25_mean"] / out["cams_pm10_mean"].where(out["cams_pm10_mean"] > 0)
    morning = c[hr.isin([2, 5, 8])].groupby("date")["cams_pm25"].mean()
    evening = c[hr.isin([17, 20, 23])].groupby("date")["cams_pm25"].mean()
    out["cams_pm25_eve_minus_morn"] = evening - morning     # build-up (+) or clearing (-) through the day

    out.loc[g["cams_pm25"].count() < MIN_SAMPLES, :] = np.nan
    out.index.name = "date"

    if CAMS_DAILY_CSV.exists():
        try:
            ref = pd.read_csv(CAMS_DAILY_CSV, parse_dates=["date"]).set_index("date")
            j = pd.concat([out["cams_pm25_mean"], ref["cams_pm25"]], axis=1, keys=["a", "b"]).dropna()
            if len(j) and (j["a"] - j["b"]).abs().max() > 1e-2:
                print(f"WARNING: cams_pm25 daily mean differs from {CAMS_DAILY_CSV.name}.")
        except Exception as exc:
            # Consistency check only, not a pipeline input -- don't fail the build over it.
            print(f"WARNING: could not read {CAMS_DAILY_CSV} for the consistency check ({exc}); skipping it.")
    return out, {col: FAMILY for col in out.columns}
