"""
rebuild_fire_daily.py
Concatenate each historical FIRMS archive with its live export from
export_for_modeling.py (live_fire_modis.csv, live_fire_viirs.csv), then call
fire_features.fire_daily() -- your existing file, unmodified -- to regenerate
fire_daily_features.csv including live days. This keeps the transport-
weighting and belt/near-fire logic in exactly one place (fire_features.py).

Run from the repository root, after export_for_modeling.py:
    python ingestion/export_for_modeling.py
    python ingestion/rebuild_fire_daily.py
    python build_features.py

Two things this script does NOT get for free from fire_features.py's own
defaults, both handled explicitly below:

1. End date. fire_daily()'s default end="2025-12-31" would silently cut off
   any live data past that day. The end date here is taken instead from
   final_delhi_dataset.csv's latest timestamp (the file export_for_modeling.py
   just wrote), so the fire calendar always matches what build_features.py
   will use.

2. Transport direction. Your existing fire_daily_features.csv has modis_ti and
   viirs_ti columns, which fire_daily() only computes when given
   transport_from_by_date (the daily direction the 925 hPa wind blows FROM,
   in degrees) -- calling it with None, as fire_features.py's own __main__
   block does, would drop both columns from the rebuild. This script derives
   that series from era5_inputs.build_era5_daily() (requires
   era5_surface_hourly_ist.csv to already be exported) the same way the
   original pipeline that built fire_daily_features.csv did, using the same
   sign convention era5_inputs.py documents for transport_from_sin/cos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # repo root, where fire_features.py / era5_inputs.py live
sys.path.insert(0, str(ROOT))
import era5_inputs as ei     # noqa: E402  (import after sys.path edit, by necessity)
import fire_features as ff   # noqa: E402

PERIOD_START = "2022-01-01"


def combine(historical_path, live_path, combined_path):
    hist = pd.read_csv(historical_path) if Path(historical_path).exists() else pd.DataFrame()
    live = pd.read_csv(live_path) if Path(live_path).exists() else pd.DataFrame()
    if hist.empty and live.empty:
        print(f"SKIP: neither {historical_path} nor {live_path} found")
        return None
    combined = pd.concat([hist, live], ignore_index=True)
    dedupe_cols = [c for c in ["satellite", "instrument", "acq_date", "acq_time", "latitude", "longitude"]
                  if c in combined.columns]
    combined = combined.drop_duplicates(subset=dedupe_cols or None)
    combined.to_csv(combined_path, index=False)
    print(f"{combined_path}: {len(combined)} rows ({len(hist)} historical + {len(live)} live, before dedup)")
    return combined_path


def period_end():
    try:
        d = pd.read_csv("final_delhi_dataset.csv", usecols=["timestamp"], parse_dates=["timestamp"])
        return d["timestamp"].max().strftime("%Y-%m-%d")
    except Exception as exc:
        print(f"Could not read final_delhi_dataset.csv ({exc}); falling back to fire_features.py's own default end date.")
        return "2025-12-31"


def transport_from_degrees(start, end):
    """Daily direction the 925 hPa wind blows FROM (degrees), the same series the
    original pipeline passed to fire_daily() to get modis_ti/viirs_ti. Requires
    era5_surface_hourly_ist.csv (and the 6h/12h pressure-level CSVs) to already
    be in the repo root, i.e. run this after export_for_modeling.py."""
    era5, _ = ei.build_era5_daily()
    cal = pd.date_range(start, end, freq="D", name="date")
    era5 = era5.reindex(cal)
    deg = np.degrees(np.arctan2(era5["transport_from_sin"], era5["transport_from_cos"])) % 360
    return deg


def main():
    modis_path = combine(ff.MODIS_CSV, "live_fire_modis.csv", "combined_fire_modis.csv")
    viirs_path = combine(ff.VIIRS_CSV, "live_fire_viirs.csv", "combined_fire_viirs.csv")
    end = period_end()
    print(f"Fire daily calendar: {PERIOD_START} to {end}")

    try:
        transport = transport_from_degrees(PERIOD_START, end)
    except Exception as exc:
        print(f"Could not compute transport direction ({exc}); modis_ti/viirs_ti will be omitted from this run. "
              f"Check era5_surface_hourly_ist.csv was exported first.")
        transport = None

    fd = ff.fire_daily(transport_from_by_date=transport, start=PERIOD_START, end=end,
                       modis_path=modis_path or ff.MODIS_CSV, viirs_path=viirs_path or ff.VIIRS_CSV)
    ff.diagnostics(fd)
    fd.to_csv("fire_daily_features.csv")
    print("Wrote fire_daily_features.csv")


if __name__ == "__main__":
    main()
