"""
migrate_historical.py
One-time load of the five historical CSVs into Postgres (the "_hist" tables in
schema.sql), so the database holds the full timeline and GitHub Actions does
not need the CSVs at all going forward.

Run this once, locally, from the repository root (where the CSVs live):

    export DATABASE_URL=postgresql://...
    python ingestion/migrate_historical.py

Safe to re-run: every table is upserted on its primary key, so re-running after
fixing a CSV just overwrites the affected rows.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from db import get_engine, init_schema, log_run, upsert_dataframe

STATION_COLS = ["PM25", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "O3", "Benzene", "Toluene",
               "temperature", "humidity", "wind_speed", "wind_direction", "pressure",
               "solar_radiation", "total_rainfall"]


IST = "Asia/Kolkata"


def load_csv(path, parse_dates, tz_cols=()):
    """Read a CSV, and for each column in tz_cols, localize it as IST (the
    timezone these historical files are documented and known to be in) and
    convert it to a real UTC instant before it goes anywhere near Postgres.

    Without this, a naive "2022-01-01 00:00:00" gets inserted into a
    TIMESTAMPTZ column and is silently interpreted in whatever timezone the
    database session happens to be in (commonly UTC on managed providers) --
    i.e. as if it meant 2022-01-01 00:00:00 UTC instead of IST. Every
    historical row would then be off by the full 5:30 IST offset relative to
    the live fetchers, which already convert correctly to UTC. The effect is
    a ~5.5 hour step in the data at exactly the historical/live boundary.
    """
    if not Path(path).exists():
        print(f"SKIP: {path} not found")
        return None
    df = pd.read_csv(path, parse_dates=parse_dates)
    for col in tz_cols:
        if col in df.columns:
            df[col] = df[col].dt.tz_localize(IST, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert("UTC")
    return df


def migrate_station(engine):
    df = load_csv("final_delhi_dataset.csv", ["timestamp"], tz_cols=["timestamp"])
    if df is None:
        return
    cols = ["station", "timestamp"] + [c for c in STATION_COLS if c in df.columns]
    n = upsert_dataframe(engine, "station_hourly_hist", df[cols], pk_cols=["station", "timestamp"])
    print(f"station_hourly_hist: {n} rows")


def migrate_era5_surface(engine):
    df = load_csv("era5_surface_hourly_ist.csv", ["timestamp"], tz_cols=["timestamp"])
    if df is None:
        return
    n = upsert_dataframe(engine, "era5_surface_hist", df, pk_cols=["timestamp"])
    print(f"era5_surface_hist: {n} rows")


def migrate_era5_pressure6h(engine):
    df = load_csv("era5_pressure_levels_6h_ist.csv", ["timestamp"], tz_cols=["timestamp"])
    if df is None:
        return
    n = upsert_dataframe(engine, "era5_pressure6h_hist", df, pk_cols=["timestamp", "pressure_hpa"])
    print(f"era5_pressure6h_hist: {n} rows")


def migrate_era5_pressure12h(engine):
    df = load_csv("era5_pressure_levels_12h_ist.csv", ["timestamp"], tz_cols=["timestamp"])
    if df is None:
        return
    n = upsert_dataframe(engine, "era5_pressure12h_hist", df,
                         pk_cols=["timestamp", "latitude", "longitude", "pressure_hpa"])
    print(f"era5_pressure12h_hist: {n} rows")


def migrate_cams(engine):
    df = load_csv("cams_3h_ist.csv", ["timestamp"], tz_cols=["timestamp"])
    if df is None:
        return
    n = upsert_dataframe(engine, "cams_3h_hist", df, pk_cols=["timestamp"])
    print(f"cams_3h_hist: {n} rows")


def main():
    engine = get_engine()
    init_schema(engine)
    started = datetime.now(timezone.utc)
    try:
        migrate_station(engine)
        migrate_era5_surface(engine)
        migrate_era5_pressure6h(engine)
        migrate_era5_pressure12h(engine)
        migrate_cams(engine)
        log_run(engine, "migrate_historical", started, datetime.now(timezone.utc), None, "ok")
    except Exception as exc:
        log_run(engine, "migrate_historical", started, datetime.now(timezone.utc), None, "error", str(exc))
        raise


if __name__ == "__main__":
    if "--yes" not in sys.argv:
        print("This loads final_delhi_dataset.csv, era5_surface_hourly_ist.csv, "
              "era5_pressure_levels_6h_ist.csv, era5_pressure_levels_12h_ist.csv and cams_3h_ist.csv "
              "(current directory) into DATABASE_URL. Re-run with --yes to proceed.")
        sys.exit(0)
    main()
