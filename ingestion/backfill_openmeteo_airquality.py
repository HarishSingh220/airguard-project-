"""
backfill_openmeteo_airquality.py
Backfills cams_hourly for the historical/live gap using Open-Meteo's Air
Quality API (air-quality-api.open-meteo.com) instead of ADS's CAMS
reanalysis. This is literally the same CAMS global atmospheric composition
product fetch_cams.py already pulls from ADS directly -- Open-Meteo just
repackages it with instant, no-key access and a historical archive back to
~August 2022 -- so it's not a lower-quality substitute, just a faster path
to the same underlying data for a backfill.

Column mapping to cams_hourly -- ONLY these three, deliberately (confirmed
against both schema.sql and fetch_cams.py's actual ADS request):
    pm2_5                 -> cams_pm25
    pm10                  -> cams_pm10
    aerosol_optical_depth -> cams_aod550
These are genuinely the same physical quantity on both sides: near-surface
PM mass concentration and total-column AOD.

cams_no2_col and cams_o3_col_du are deliberately NOT fetched here. Checked
fetch_cams.py: it requests ADS's "nitrogen_dioxide" and "total_column_ozone"
variables, which are TOTAL ATMOSPHERIC COLUMN quantities (kg/m2, Dobson
Units respectively). Open-Meteo's Air Quality API "nitrogen_dioxide"/"ozone"
are explicitly surface concentrations (ug/m3, "10 meter above ground") --
a different physical quantity, not an approximation of the same one. Writing
Open-Meteo's values into cams_no2_col/cams_o3_col_du would silently insert
numbers on the wrong scale and of the wrong kind, not just lower-fidelity
ones. These two columns stay NULL for this backfill; fetch them with
cds_backfill.py's "cams-eac4-col" kind instead, which requests the matching
column quantities from ADS's EAC4 reanalysis.

Usage:
    python ingestion/backfill_openmeteo_airquality.py 2026-01-01 2026-10-11
"""
import argparse
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from config import DELHI_LAT, DELHI_LON
from db import get_engine, init_schema, log_run, upsert_dataframe
from backfill_openmeteo import date_chunks
from http_utils import get_with_retry

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
IST = "Asia/Kolkata"
VARS = ["pm2_5", "pm10", "aerosol_optical_depth"]
RENAME = {"pm2_5": "cams_pm25", "pm10": "cams_pm10", "aerosol_optical_depth": "cams_aod550"}


def _parse_ist_grid(time_strings):
    naive = pd.to_datetime(time_strings)
    return naive.tz_localize(IST, ambiguous="infer", nonexistent="shift_forward").tz_convert("UTC")


def fetch_chunk(start_date, end_date):
    resp = get_with_retry(URL, params={
        "latitude": DELHI_LAT, "longitude": DELHI_LON, "hourly": ",".join(VARS),
        "start_date": start_date, "end_date": end_date, "timezone": IST, "domains": "cams_global",
    }, timeout=120)
    resp.raise_for_status()
    try:
        body = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Open-Meteo response was not valid JSON ({exc}): {resp.text[:200]!r}") from exc
    if "hourly" not in body:
        raise RuntimeError(f"Open-Meteo response has no 'hourly' key: {list(body)}")
    return body["hourly"]


def build_rows(hourly, run_time):
    valid_time = _parse_ist_grid(hourly["time"])
    df = pd.DataFrame({"valid_time": valid_time})
    for v in VARS:
        df[RENAME[v]] = hourly.get(v)
    # Same CAMS model-output product as fetch_cams.py/cds_backfill.py -- keep is_forecast=True
    # and use is_best_estimate rather than re-deriving is_forecast from valid_time vs run_time
    # (see fetch_cams.py's comment). This script only ever backfills past date ranges, so every
    # row here is this date's best-available CAMS estimate; Open-Meteo's archive endpoint
    # doesn't expose the original lead time, so lead_time_hours stays NULL.
    df["is_forecast"] = True
    df["is_best_estimate"] = True
    df["run_time"] = run_time
    df["source"] = "open-meteo-aq"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    args = ap.parse_args()

    try:
        chunks = list(date_chunks(args.start, args.end))
    except ValueError as exc:
        sys.exit(f"Could not parse start/end date ({args.start!r}, {args.end!r}) as dates: {exc}")
    if not chunks:
        sys.exit(f"No date range to backfill: start ({args.start}) is after end ({args.end}).")

    engine = get_engine()
    init_schema(engine)
    run_time = datetime.now(timezone.utc)
    total = 0
    failed_chunks = []

    for start_date, end_date in chunks:
        started = datetime.now(timezone.utc)
        try:
            hourly = fetch_chunk(start_date, end_date)
            df = build_rows(hourly, run_time).dropna(subset=["cams_pm25"])
            n = upsert_dataframe(engine, "cams_hourly", df, pk_cols=["valid_time", "source"])
            total += n
            log_run(engine, "backfill_openmeteo_airquality", started, datetime.now(timezone.utc), n, "ok")
            print(f"{start_date}..{end_date}: {n} rows")
        except Exception as exc:
            log_run(engine, "backfill_openmeteo_airquality", started, datetime.now(timezone.utc), 0, "error", str(exc))
            print(f"{start_date}..{end_date}: FAILED -- {exc}")
            failed_chunks.append(f"{start_date}..{end_date}: {exc}")
        time.sleep(1)

    print(f"\nTotal: cams_hourly +{total} rows")
    if failed_chunks:
        print(f"\n{len(failed_chunks)} chunk(s) failed and were NOT loaded:")
        for line in failed_chunks:
            print(f"  - {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()
