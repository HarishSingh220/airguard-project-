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
import tempfile
import zipfile
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
            "total_column_nitrogen_dioxide", "total_column_ozone"]
# CAMS variable -> our column. total_aerosol_optical_depth_550nm -> cams_aod550, etc.
# NOTE: "nitrogen_dioxide" (no "total_column_" prefix) is a DIFFERENT, multi-level
# variable (kg/kg mixing ratio) that ADS rejects with a 400 "invalid combination of
# values" unless a model_level/pressure_level is also given -- don't swap this back
# without adding one. total_column_nitrogen_dioxide (kg/m2) is the single-level field
# this project actually wants (see cams_no2_col below), same shape as total_column_ozone.
VAR_COLUMN = {
    "pm2p5": "cams_pm25", "pm10": "cams_pm10", "aod550": "cams_aod550",
    "tcno2": "cams_no2_col", "gtco3": "cams_o3_col_du",
}
DOWNLOAD_PATH = Path("/tmp/cams_forecast.zip")


def build_request():
    today = datetime.now(timezone.utc).date().isoformat()
    box = [DELHI_LAT + 0.5, DELHI_LON - 0.5, DELHI_LAT - 0.5, DELHI_LON + 0.5]  # N, W, S, E
    return {
        "variable": VARIABLES, "date": [f"{today}/{today}"], "time": ["00:00"],
        "leadtime_hour": LEADTIME_HOURS, "type": ["forecast"],
        # The current ADS API has two separate keys where the old API had one
        # combined "format" value: data_format ("netcdf"/"grib") and download_format
        # ("zip"/"unarchived"). "netcdf_zip" isn't a valid data_format -- the client
        # silently drops it back to plain "netcdf", which is indistinguishable from
        # never asking for zipping at all and is exactly what kept 400ing. This
        # request's variables don't share one GRIB stepType/levtype (surface fields
        # like PM2.5 vs column fields like total_column_ozone), so the server can't
        # flatten them into a single NetCDF; download_format=zip is what tells it
        # that's fine and to hand back one .nc per group inside a zip instead.
        "data_format": "netcdf", "download_format": "zip",
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


def _frame_from_nc(nc_path):
    """One .nc file -> a (valid_time, <our columns present in this file>) frame.
    Returns None if this file has none of VAR_COLUMN's variables (e.g. a metadata-
    only member, if ADS ever adds one) so the caller can skip it cleanly."""
    ds = xr.open_dataset(nc_path)
    try:
        point = ds.sel(latitude=DELHI_LAT, longitude=DELHI_LON, method="nearest")
        df = point.to_dataframe().reset_index()
        time_col = "valid_time" if "valid_time" in df.columns else ("time" if "time" in df.columns else "forecast_time")
        frame = pd.DataFrame({"valid_time": pd.to_datetime(df[time_col], utc=True)})
        found = [src for src in VAR_COLUMN if src in df.columns]
        for src in found:
            frame[VAR_COLUMN[src]] = df[src]
        return (frame, found) if found else (None, [])
    finally:
        ds.close()


def to_dataframe(zip_path, run_time, init_time):
    with tempfile.TemporaryDirectory() as extract_dir:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        nc_paths = sorted(Path(extract_dir).glob("*.nc"))
        if not nc_paths:
            raise RuntimeError(f"no .nc files found inside the downloaded {zip_path} -- "
                                f"ADS may have changed this dataset's zip layout")
        frames, found_overall = [], set()
        for nc_path in nc_paths:
            frame, found = _frame_from_nc(nc_path)
            if frame is not None:
                frames.append(frame)
                found_overall.update(found)
    missing = [src for src in VAR_COLUMN if src not in found_overall]
    if missing:
        print(f"WARNING: expected CAMS variable(s) {missing} not found in any of the "
              f"{len(nc_paths)} file(s) inside the downloaded zip. "
              f"ADS may have renamed the short name(s) -- update VAR_COLUMN in fetch_cams.py.")
    if not frames:
        raise RuntimeError(f"none of VAR_COLUMN's expected variables {list(VAR_COLUMN)} were "
                            f"found in any of the {len(nc_paths)} file(s) inside the downloaded zip")
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on="valid_time", how="outer")
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
        zip_path = fetch(build_request())
        df = to_dataframe(zip_path, run_time, init_time)
        df = df.dropna(how="all", subset=list(VAR_COLUMN.values()))
        n = upsert_dataframe(engine, "cams_hourly", df, pk_cols=["valid_time", "source"])
        log_run(engine, "fetch_cams", started, datetime.now(timezone.utc), n, "ok")
        print(f"cams_hourly: upserted {n} rows")
    except Exception as exc:
        log_run(engine, "fetch_cams", started, datetime.now(timezone.utc), 0, "error", str(exc))
        raise
    finally:
        if DOWNLOAD_PATH.exists():
            DOWNLOAD_PATH.unlink()


if __name__ == "__main__":
    main()