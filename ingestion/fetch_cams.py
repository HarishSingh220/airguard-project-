"""
fetch_cams.py
Pull the CAMS global atmospheric composition forecast for the Delhi grid point
nearest DELHI_LAT/DELHI_LON from the Copernicus Atmosphere Data Store (ADS) and
upsert into cams_hourly.

Dataset: cams-global-atmospheric-composition-forecasts (~40 km, 3-hourly steps,
issued once daily, out to 120 hours). This is the forecast counterpart of the
CAMS reanalysis your historical cams_3h_ist.csv was built from.

Requires CDSAPI_URL and CDSAPI_KEY (your existing ADS credentials). The cdsapi
client normally reads these from ~/.cdsapirc; the GitHub Actions workflow
writes that file from the two secrets before this script runs, so it works
the same locally if you already have ~/.cdsapirc set up (in which case
CDSAPI_URL/CDSAPI_KEY need not be set as environment variables).

If the exact variable or dataset names below have changed on ADS (Copernicus
has renamed things before), run `cdsapi` against the ADS web form's "Show API
request" for this dataset and compare; this script isolates that request in
one place (REQUEST, below) so fixing it means editing only this file.
"""
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr

from config import DELHI_LAT, DELHI_LON
from db import get_engine, init_schema, log_run, upsert_dataframe

DATASET = "cams-global-atmospheric-composition-forecasts"
LEADTIME_HOURS = [str(h) for h in range(0, 121, 3)]
VARIABLES = ["particulate_matter_2.5um", "particulate_matter_10um", "total_aerosol_optical_depth_550nm",
            "nitrogen_dioxide", "total_column_ozone"]
# CAMS variable -> our column. total_aerosol_optical_depth_550nm -> cams_aod550, etc.
VAR_COLUMN = {
    "pm2p5": "cams_pm25", "pm10": "cams_pm10", "aod550": "cams_aod550",
    "no2": "cams_no2_col", "gtco3": "cams_o3_col_du",
}
DOWNLOAD_PATH = Path("/tmp/cams_forecast.nc")


def build_request():
    today = datetime.now(timezone.utc).date().isoformat()
    box = [DELHI_LAT + 0.5, DELHI_LON - 0.5, DELHI_LAT - 0.5, DELHI_LON + 0.5]  # N, W, S, E
    return {
        "variable": VARIABLES, "date": [f"{today}/{today}"], "time": ["00:00"],
        "leadtime_hour": LEADTIME_HOURS, "type": ["forecast"], "format": "netcdf",
        "area": box,
    }


def _forecast_init_time():
    """This run's forecast issuance time: 00:00 UTC today, matching build_request()'s
    "date"/"time". Used to compute each row's actual lead_time_hours, independent of
    when the script happens to run."""
    return pd.Timestamp.now(tz="UTC").normalize()


def fetch(request):
    client = cdsapi.Client()   # reads ~/.cdsapirc (URL and key), written by the workflow from secrets
    client.retrieve(DATASET, request).download(str(DOWNLOAD_PATH))
    return DOWNLOAD_PATH


def to_dataframe(nc_path, run_time, init_time):
    ds = xr.open_dataset(nc_path)
    point = ds.sel(latitude=DELHI_LAT, longitude=DELHI_LON, method="nearest")
    df = point.to_dataframe().reset_index()
    time_col = "valid_time" if "valid_time" in df.columns else ("time" if "time" in df.columns else "forecast_time")
    out = pd.DataFrame({"valid_time": pd.to_datetime(df[time_col], utc=True)})
    for src, col in VAR_COLUMN.items():
        if src in df.columns:
            out[col] = df[src]
    out = out.groupby("valid_time", as_index=False).mean(numeric_only=True)
    out["lead_time_hours"] = ((out["valid_time"] - init_time).dt.total_seconds() / 3600).round().astype("Int64")
    # Every row here is a CAMS forecast product, full stop -- is_forecast is not derived
    # from whether valid_time has now passed (a forecast's valid time being in the past
    # does not make it a measurement). is_best_estimate instead flags the short-lead rows
    # around "now" at fetch time, which export_for_modeling.py uses as the freshest
    # available model estimate -- still clearly a forecast, not an observation.
    out["is_forecast"] = True
    out["is_best_estimate"] = out["valid_time"] <= run_time
    out["run_time"] = run_time
    out["source"] = "cams-ads"
    return out


def main():
    run_time = datetime.now(timezone.utc)
    init_time = _forecast_init_time()
    engine = get_engine()
    init_schema(engine)
    started = datetime.now(timezone.utc)
    try:
        nc_path = fetch(build_request())
        df = to_dataframe(nc_path, run_time, init_time)
        df = df.dropna(how="all", subset=list(VAR_COLUMN.values()))
        n = upsert_dataframe(engine, "cams_hourly", df, pk_cols=["valid_time", "source"])
        log_run(engine, "fetch_cams", started, datetime.now(timezone.utc), n, "ok")
        print(f"cams_hourly: upserted {n} rows")
    except Exception as exc:
        log_run(engine, "fetch_cams", started, datetime.now(timezone.utc), 0, "error", str(exc))
        raise
    finally:
        nc_path_obj = DOWNLOAD_PATH
        if nc_path_obj.exists():
            nc_path_obj.unlink()


if __name__ == "__main__":
    main()
