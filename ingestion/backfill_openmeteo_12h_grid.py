"""
backfill_openmeteo_12h_grid.py
Backfills the 4-grid-point, 700/800/900 hPa box era5_pressure_levels_12h_ist.csv
historically provided (used for levels12_daily()'s spatial temperature
gradients) -- via Open-Meteo's historical archive API using its multi-location
query support (comma-separated lat/lon), instead of CDS/ADS. Only t/u/v/r are
needed here (see era5_inputs.levels12_daily()), and Open-Meteo provides all
four, so this file has no CDS-only dependency at all.

Writes into weather_pressure12h_hourly (new table -- see schema.sql), which
export_for_modeling.py reads for the live tail alongside era5_pressure12h_hist.

Usage:
    python ingestion/backfill_openmeteo_12h_grid.py 2026-01-01 2026-10-11
"""
import argparse
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from db import get_engine, init_schema, log_run, upsert_dataframe
from backfill_openmeteo import date_chunks
from http_utils import get_with_retry

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
IST = "Asia/Kolkata"
LEVELS = [700, 800, 900]
VARS = ["temperature", "wind_speed", "wind_direction", "relative_humidity"]
# Same 0.25-degree box as era5_pressure_levels_12h_ist.csv (verified against the
# historical file's own distinct lat/lon pairs).
GRID_POINTS = [(28.50, 77.00), (28.50, 77.25), (28.75, 77.00), (28.75, 77.25)]


def _parse_ist_grid(time_strings):
    naive = pd.to_datetime(time_strings)
    return naive.tz_localize(IST, ambiguous="infer", nonexistent="shift_forward").tz_convert("UTC")


def fetch_chunk(start_date, end_date):
    hourly = [f"{v}_{lvl}hPa" for lvl in LEVELS for v in VARS]
    lats = ",".join(str(lat) for lat, lon in GRID_POINTS)
    lons = ",".join(str(lon) for lat, lon in GRID_POINTS)
    resp = get_with_retry(ARCHIVE_URL, params={
        "latitude": lats, "longitude": lons, "hourly": ",".join(hourly),
        "start_date": start_date, "end_date": end_date, "timezone": IST,
        # Match fetch_openmeteo.py / the ERA5 m/s convention -- see its comment.
        "wind_speed_unit": "ms",
    }, timeout=120)
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Open-Meteo response was not valid JSON ({exc}): {resp.text[:200]!r}") from exc
    # Multiple locations -> a list of per-location response objects, one per GRID_POINTS entry, in order.
    responses = data if isinstance(data, list) else [data]
    if len(responses) != len(GRID_POINTS):
        raise RuntimeError(
            f"Expected {len(GRID_POINTS)} per-location responses from Open-Meteo, got {len(responses)}."
        )
    for r in responses:
        units = r.get("hourly_units", {})
        bad = {k: v for k, v in units.items() if k.startswith("wind_speed") and v not in ("m/s", "ms")}
        if bad:
            raise RuntimeError(f"Open-Meteo returned unexpected wind-speed units: {bad}")
    return responses


def build_rows(per_location_responses, run_time):
    rows = []
    for (lat, lon), resp in zip(GRID_POINTS, per_location_responses):
        hourly = resp["hourly"]
        valid_time = _parse_ist_grid(hourly["time"])
        for lvl in LEVELS:
            d = pd.DataFrame({"valid_time": valid_time, "latitude": lat, "longitude": lon, "pressure_hpa": lvl})
            for v in VARS:
                d[v] = hourly.get(f"{v}_{lvl}hPa")
            rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["is_forecast"] = df["valid_time"] > run_time
    df["run_time"] = run_time
    df["source"] = "open-meteo-archive"
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
            responses = fetch_chunk(start_date, end_date)
            df = build_rows(responses, run_time).dropna(subset=["temperature"])
            n = upsert_dataframe(engine, "weather_pressure12h_hourly", df,
                                  pk_cols=["valid_time", "latitude", "longitude", "pressure_hpa", "source"])
            total += n
            log_run(engine, "backfill_openmeteo_12h_grid", started, datetime.now(timezone.utc), n, "ok")
            print(f"{start_date}..{end_date}: {n} rows")
        except Exception as exc:
            log_run(engine, "backfill_openmeteo_12h_grid", started, datetime.now(timezone.utc), 0, "error", str(exc))
            print(f"{start_date}..{end_date}: FAILED -- {exc}")
            failed_chunks.append(f"{start_date}..{end_date}: {exc}")
        time.sleep(1)

    print(f"\nTotal: weather_pressure12h_hourly +{total} rows")
    if failed_chunks:
        print(f"\n{len(failed_chunks)} chunk(s) failed and were NOT loaded:")
        for line in failed_chunks:
            print(f"  - {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()
