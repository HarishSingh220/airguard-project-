"""
cds_backfill.py
The only two things in the whole feature set that genuinely have no
Open-Meteo path and must go through CDS/ADS's queue:

  era5-pressure-qw   specific_humidity (q) at 925 hPa, vertical_velocity (w)
                     at 850/700 hPa -- ua_q925_lag1d, ua_w850_lag1d,
                     ua_w700_lag1d (+ their chg1d/roll3d derivatives) in
                     features/feature_list.txt. Confirmed: Open-Meteo offers
                     neither q nor w at any pressure level, on any endpoint.
                     Dataset: reanalysis-era5-pressure-levels (real CDS,
                     cds.climate.copernicus.eu -- a DIFFERENT credential/URL
                     than the ADS one your other scripts use for CAMS; see
                     the note in config.py).

  cams-no2-o3        cams_no2_col, cams_o3_col_du -- total COLUMN NO2/O3.
                     Open-Meteo's Air Quality API only has SURFACE NO2/O3
                     (a different physical quantity -- see
                     backfill_openmeteo_airquality.py's docstring). CAMS's
                     reanalysis (EAC4) is the obvious next choice, but its
                     own ADS page states "Temporal coverage: 2003 to 2025"
                     and "Update frequency: Twice a year with 4-6 month
                     delay" -- it does not cover 2026 yet and won't for a
                     while. So this backfills from the SAME dataset
                     fetch_cams.py already uses live
                     (cams-global-atmospheric-composition-forecasts), just
                     with a past `date` range instead of today -- that
                     dataset's forecast-cycle archive goes back to August
                     2022, well past the gap.

Both still go through MARS, so: async submit (no blocking retrieve()),
monthly chunks, resumable state file. See the module docstring pattern notes
from the earlier draft of this file -- the exact polling/status method names
on your installed `cdsapi` version should be checked against its docs before
relying on poll_and_download(); the submit-then-poll SHAPE is what matters.

Usage (run repeatedly until "all done"):
    python ingestion/cds_backfill.py era5-pressure-qw 2026-01-01 2026-10-11
    python ingestion/cds_backfill.py cams-no2-o3      2026-01-01 2026-10-11
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr

from config import DELHI_LAT, DELHI_LON
from db import get_engine, init_schema, log_run, upsert_dataframe

STATE_PATH = Path("ingestion/.cds_backfill_state.json")
DOWNLOAD_DIR = Path("backfill_raw")

# (dataset, climate-data-store-vs-ads) -- era5-pressure-qw needs the actual CDS
# (cds.climate.copernicus.eu), cams-no2-o3 needs ADS (ads.atmosphere.copernicus.eu,
# the one config.CDSAPI_URL/KEY already point to per fetch_cams.py).
DATASETS = {
    "era5-pressure-qw": "reanalysis-era5-pressure-levels",
    "cams-no2-o3": "cams-global-atmospheric-composition-forecasts",
}


def load_state():
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state):
    STATE_PATH.parent.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def month_chunks(start, end):
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    for m in months:
        lo = max(m.start_time, pd.Timestamp(start))
        hi = min(m.end_time, pd.Timestamp(end))
        yield lo.date().isoformat(), hi.date().isoformat()


def build_request(kind, lo, hi):
    if kind == "era5-pressure-qw":
        days = pd.date_range(lo, hi, freq="D").strftime("%d").unique().tolist()
        years = sorted({lo[:4], hi[:4]})
        months = sorted({lo[5:7], hi[5:7]})
        area = [29.1, 76.7, 28.1, 77.7]  # N, W, S, E -- small box around Delhi, not the full NCR fire bbox
        return {
            "product_type": "reanalysis", "format": "netcdf", "area": area,
            "pressure_level": ["700", "850", "925"],  # extra levels per variable are harmless, just unused
            "variable": ["specific_humidity", "vertical_velocity"],
            "year": years, "month": months, "day": days,
            "time": ["00:00", "06:00", "12:00", "18:00"],  # matches era5_pressure_levels_6h_ist.csv's own grid
        }
    if kind == "cams-no2-o3":
        box = [DELHI_LAT + 0.5, DELHI_LON - 0.5, DELHI_LAT - 0.5, DELHI_LON + 0.5]
        return {
            "variable": ["nitrogen_dioxide", "total_column_ozone"],
            "date": [f"{lo}/{hi}"], "time": ["00:00"],
            # Only this day's own near-term hours, not the full 120h lead forecast --
            # for a backfill we want each day's own best estimate, not next week's forecast of it.
            "leadtime_hour": [str(h) for h in range(0, 24, 3)],
            "type": ["forecast"], "format": "netcdf", "area": box,
        }
    raise ValueError(kind)


def submit_missing(kind, start, end, client):
    state = load_state()
    state.setdefault(kind, {})
    dataset = DATASETS[kind]
    for lo, hi in month_chunks(start, end):
        key = f"{lo}:{hi}"
        if key in state[kind]:
            continue
        req = build_request(kind, lo, hi)
        result = client.retrieve(dataset, req)
        state[kind][key] = {"request_id": result.reply["request_id"], "status": "queued", "path": None}
        print(f"[{kind}] submitted {lo}..{hi} -> request_id={result.reply['request_id']}")
        save_state(state)


def poll_and_download(kind, client):
    state = load_state()
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    pending = False
    for key, job in state.get(kind, {}).items():
        if job["status"] == "done":
            continue
        status = client.client.status(job["request_id"]) if hasattr(client.client, "status") else "unknown"
        if status in ("completed", "successful"):
            out_path = DOWNLOAD_DIR / f"{kind}_{key.replace(':', '_')}.nc"
            client.download(job["request_id"], str(out_path))
            job["status"] = "done"
            job["path"] = str(out_path)
            print(f"[{kind}] {key}: downloaded -> {out_path}")
        else:
            pending = True
            print(f"[{kind}] {key}: still {status}")
    save_state(state)
    return pending


# ---------------------------------------------------------------------------
# Load downloaded NetCDF into Postgres -- same xr.open_dataset + nearest-point
# pattern fetch_cams.py already uses, so this stays consistent with the rest
# of the ingestion code rather than introducing a second parsing style.
# ---------------------------------------------------------------------------
def load_era5_pressure_qw():
    state = load_state()
    engine = get_engine()
    init_schema(engine)
    total = 0
    for key, job in state.get("era5-pressure-qw", {}).items():
        if job["status"] != "done" or job.get("loaded"):
            continue
        ds = xr.open_dataset(job["path"])
        point = ds.sel(latitude=DELHI_LAT, longitude=DELHI_LON, method="nearest")
        df = point.to_dataframe().reset_index()
        time_col = "valid_time" if "valid_time" in df.columns else "time"
        out = pd.DataFrame({
            "valid_time": pd.to_datetime(df[time_col], utc=True),
            "pressure_hpa": df["pressure_level"].astype(int),
            "specific_humidity": df.get("q"),
            "vertical_velocity": df.get("w"),
        })
        out = out.dropna(subset=["specific_humidity", "vertical_velocity"], how="all")
        out["is_forecast"] = False
        out["run_time"] = pd.Timestamp.now(tz="UTC")
        out["source"] = "era5-cds-backfill"
        n = upsert_dataframe(engine, "weather_pressure_hourly", out,
                              pk_cols=["valid_time", "pressure_hpa", "source"])
        total += n
        job["loaded"] = True
        print(f"[era5-pressure-qw] {key}: loaded {n} rows")
    save_state(state)
    print(f"Total specific_humidity/vertical_velocity rows loaded: {total}")


def load_cams_no2_o3():
    state = load_state()
    engine = get_engine()
    init_schema(engine)
    total = 0
    for key, job in state.get("cams-no2-o3", {}).items():
        if job["status"] != "done" or job.get("loaded"):
            continue
        ds = xr.open_dataset(job["path"])
        point = ds.sel(latitude=DELHI_LAT, longitude=DELHI_LON, method="nearest")
        df = point.to_dataframe().reset_index()
        time_col = "valid_time" if "valid_time" in df.columns else ("time" if "time" in df.columns else "forecast_time")
        out = pd.DataFrame({"valid_time": pd.to_datetime(df[time_col], utc=True)})
        if "no2" in df.columns:
            out["cams_no2_col"] = df["no2"]
        if "gtco3" in df.columns:
            out["cams_o3_col_du"] = df["gtco3"]
        out = out.groupby("valid_time", as_index=False).mean(numeric_only=True)
        # Same product line as fetch_cams.py (cams-global-atmospheric-composition-forecasts,
        # type=forecast) -- these rows are forecast output, not observations, regardless of
        # how long ago their valid_time now is. build_request() already restricts this to each
        # day's own near-term leadtime_hour window (see module docstring), so every row here
        # IS that day's best estimate -- is_best_estimate=True, not an is_forecast=False
        # relabeling of a forecast as an observed measurement.
        day_init = pd.to_datetime(key.split(":")[0], utc=True)
        out["lead_time_hours"] = ((out["valid_time"] - day_init).dt.total_seconds() / 3600).round().astype("Int64")
        out["is_forecast"] = True
        out["is_best_estimate"] = True
        out["run_time"] = pd.Timestamp.now(tz="UTC")
        out["source"] = "cams-ads-backfill"
        n = upsert_dataframe(engine, "cams_hourly", out, pk_cols=["valid_time", "source"])
        total += n
        job["loaded"] = True
        print(f"[cams-no2-o3] {key}: loaded {n} rows")
    save_state(state)
    print(f"Total cams_no2_col/cams_o3_col_du rows loaded: {total}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=DATASETS.keys())
    p.add_argument("start")
    p.add_argument("end")
    p.add_argument("--url", default=None,
                    help="CDS API root, e.g. https://cds.climate.copernicus.eu/api "
                         "(required for era5-pressure-qw unless ~/.cdsapirc already points there)")
    p.add_argument("--key", default=None,
                    help="CDS API personal access token matching --url")
    args = p.parse_args()

    # cdsapi.Client() previously always read ~/.cdsapirc regardless of these flags,
    # even though the docstring/config.py told users to pass --url/--key -- so setting
    # them did nothing. Only override the client's url/key when actually given, since
    # cams-no2-o3 is meant to keep using the ADS ~/.cdsapirc the rest of the ingestion
    # scripts already write.
    client_kwargs = {"wait_until_complete": False}
    if args.url:
        client_kwargs["url"] = args.url
    if args.key:
        client_kwargs["key"] = args.key
    client = cdsapi.Client(**client_kwargs)

    # Logged the same way the scheduled fetch_*.py scripts are, so a manual backfill
    # shows up in ingestion_log too -- useful for an audit trail, even though nothing
    # currently reads this job's rows back out (check_retrain_due.py only looks at
    # job='retrain_models'). Like fetch_cams.py/fetch_firms.py (and unlike
    # train_production_models.py's separate best-effort log_completion() wrapper), a
    # log_run() failure here is allowed to propagate as this run's own failure --
    # get_engine()/init_schema() having already succeeded above, a failure in log_run
    # would itself be a new, real DB problem worth surfacing, not something to swallow.
    engine = get_engine()
    init_schema(engine)
    started = datetime.now(timezone.utc)
    try:
        submit_missing(args.kind, args.start, args.end, client)
        pending = poll_and_download(args.kind, client)

        if args.kind == "era5-pressure-qw":
            load_era5_pressure_qw()
        else:
            load_cams_no2_o3()

        status = "ok (chunks still pending, re-run to continue)" if pending else "ok"
        log_run(engine, f"cds_backfill_{args.kind}", started, datetime.now(timezone.utc),
                None, status, f"{args.start}..{args.end}")
    except Exception as exc:
        log_run(engine, f"cds_backfill_{args.kind}", started, datetime.now(timezone.utc),
                None, "error", str(exc))
        raise

    if pending:
        print(f"\nSome chunks still queued. Re-run this same command later -- "
              f"state is saved in {STATE_PATH}, nothing is re-submitted or re-loaded.")
        sys.exit(1)
    print(f"\nAll {args.kind} chunks for {args.start}..{args.end} downloaded and loaded.")


if __name__ == "__main__":
    main()
