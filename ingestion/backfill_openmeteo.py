"""
backfill_openmeteo.py
One-time backfill of surface + pressure-level weather for the gap between
final_delhi_dataset.csv's end date and when fetch_openmeteo.py started
running live, using Open-Meteo's Historical Weather API
(archive-api.open-meteo.com) instead of the forecast API.

This covers every era5_* column EXCEPT specific_humidity (q) and
vertical_velocity (w) -- those two have no Open-Meteo equivalent at any
endpoint and are backfilled separately by cds_backfill.py. Everything here
reuses fetch_openmeteo.py's build_surface()/build_pressure() unchanged: same
variable list, same IST-grid parsing, same table shape -- only the HTTP call
(archive endpoint, start_date/end_date instead of past_days/forecast_days)
differs.

No key, no request queue -- this runs end-to-end in minutes, not hours.

Usage:
    python ingestion/backfill_openmeteo.py 2026-01-01 2026-10-11
"""
import argparse
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from config import DELHI_LAT, DELHI_LON, PRESSURE_LEVELS
from db import get_engine, init_schema, log_run, upsert_dataframe
from fetch_openmeteo import SURFACE_VARS, PRESSURE_VARS, build_surface, build_pressure
from http_utils import get_with_retry

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
IST = "Asia/Kolkata"
CHUNK_DAYS = 90   # keep each request a manageable size; the archive API has no hard
                  # documented cap, but smaller requests are easier to retry on a timeout


def date_chunks(start, end, days=CHUNK_DAYS):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cur = start
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=days - 1), end)
        yield cur.date().isoformat(), chunk_end.date().isoformat()
        cur = chunk_end + pd.Timedelta(days=1)


def fetch_chunk(start_date, end_date):
    hourly = SURFACE_VARS + [f"{v}_{lvl}hPa" for lvl in PRESSURE_LEVELS for v in PRESSURE_VARS]
    resp = get_with_retry(ARCHIVE_URL, params={
        "latitude": DELHI_LAT, "longitude": DELHI_LON, "hourly": ",".join(hourly),
        "start_date": start_date, "end_date": end_date, "timezone": IST,
        # Match fetch_openmeteo.py / the ERA5 m/s convention -- see its comment.
        "wind_speed_unit": "ms",
    }, timeout=120)
    resp.raise_for_status()
    try:
        body = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Open-Meteo response was not valid JSON ({exc}): {resp.text[:200]!r}") from exc
    if "hourly" not in body:
        raise RuntimeError(f"Open-Meteo response has no 'hourly' key: {list(body)}")
    units = body.get("hourly_units", {})
    bad = {k: v for k, v in units.items() if k.startswith("wind_speed") and v not in ("m/s", "ms")}
    if bad:
        raise RuntimeError(f"Open-Meteo returned unexpected wind-speed units: {bad}")
    return body["hourly"]


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
    run_time = datetime.now(timezone.utc)   # all valid_time here is in the past -> is_forecast=False regardless
    total_surface = total_pressure = 0
    failed_chunks = []

    for start_date, end_date in chunks:
        started = datetime.now(timezone.utc)
        try:
            hourly = fetch_chunk(start_date, end_date)
            surface = build_surface(hourly, run_time).dropna(subset=["temperature_2m"])
            pressure = build_pressure(hourly, run_time).dropna(subset=["temperature"])
            surface["source"] = "open-meteo-archive"
            pressure["source"] = "open-meteo-archive"
            n1 = upsert_dataframe(engine, "weather_surface_hourly", surface, pk_cols=["valid_time", "source"])
            n2 = upsert_dataframe(engine, "weather_pressure_hourly", pressure,
                                   pk_cols=["valid_time", "pressure_hpa", "source"])
            total_surface += n1
            total_pressure += n2
            log_run(engine, "backfill_openmeteo", started, datetime.now(timezone.utc), n1 + n2, "ok")
            print(f"{start_date}..{end_date}: surface={n1} pressure={n2}")
        except Exception as exc:
            log_run(engine, "backfill_openmeteo", started, datetime.now(timezone.utc), 0, "error", str(exc))
            print(f"{start_date}..{end_date}: FAILED -- {exc}")
            failed_chunks.append(f"{start_date}..{end_date}: {exc}")
        time.sleep(1)   # polite pause between chunks; Open-Meteo's free tier has a rate limit

    print(f"\nTotal: weather_surface_hourly +{total_surface} rows, weather_pressure_hourly +{total_pressure} rows")
    if failed_chunks:
        print(f"\n{len(failed_chunks)} chunk(s) failed and were NOT loaded:")
        for line in failed_chunks:
            print(f"  - {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()
