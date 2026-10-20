"""
fetch_openmeteo.py
Pull Delhi surface and pressure-level weather from the Open-Meteo Forecast API
(ECMWF-blended, no key needed) and upsert into weather_surface_hourly and
weather_pressure_hourly.

past_days=PAST_DAYS gives recently-observed hours, used to keep the daily
feature pipeline current between ERA5 reanalysis releases (ERA5 proper lags by
several days). forecast_days=FORECAST_DAYS gives the hours ahead. Rows are
split on valid_time <= this run's time (is_forecast = false) vs after
(is_forecast = true), so a later run simply overwrites earlier forecast rows
with observed ones as time passes (the upsert in db.py is keyed on valid_time).

Column-name note: these tables deliberately keep Open-Meteo's own field names
(temperature_2m, boundary_layer_height, ...), NOT the era5_* names
era5_inputs.py reads from era5_surface_hourly_ist.csv. export_for_modeling.py
derives the era5_* names (including era5_inv_surface, era5_inv_deep,
era5_lapse_925_700, era5_ventilation, u10/v10/u925/v925), with the formulas
documented there, because the original CSVs' own builder was not available
here to copy from directly.
"""
from datetime import datetime, timezone

import pandas as pd
import requests

from config import DELHI_LAT, DELHI_LON, PRESSURE_LEVELS
from db import get_engine, init_schema, log_run, upsert_dataframe
from http_utils import get_with_retry

IST = "Asia/Kolkata"
URL = "https://api.open-meteo.com/v1/forecast"
PAST_DAYS = 5
FORECAST_DAYS = 10

SURFACE_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
                "wind_direction_10m", "boundary_layer_height", "shortwave_radiation", "precipitation"]
SURFACE_RENAME = {"surface_pressure": "surface_pressure_hpa"}
PRESSURE_VARS = ["temperature", "wind_speed", "wind_direction", "relative_humidity", "geopotential_height"]


def fetch():
    # timezone=Asia/Kolkata (not UTC): the historical data, build_features.py
    # and the now-IST-floored CPCB live rows are all on an IST-hourly grid
    # (00:00, 01:00, ... IST). Requesting UTC here would return a grid offset
    # by 5:30 from that, so every weather-backfill merge in this file would
    # silently match nothing (or the wrong hour). Open-Meteo returns naive
    # local-time labels when a timezone is given; build_surface/build_pressure
    # localize them to IST and convert to real UTC instants explicitly.
    hourly = SURFACE_VARS + [f"{v}_{lvl}hPa" for lvl in PRESSURE_LEVELS for v in PRESSURE_VARS]
    resp = get_with_retry(URL, params={
        "latitude": DELHI_LAT, "longitude": DELHI_LON, "hourly": ",".join(hourly),
        "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS, "timezone": IST,
        # Without this, Open-Meteo defaults to km/h, while the historical ERA5 wind
        # columns and every derived wind/ventilation feature are in m/s -- a silent
        # ~3.6x unit mismatch at the historical/live boundary if left unset.
        "wind_speed_unit": "ms",
    }, timeout=60)
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


def _parse_ist_grid(time_strings):
    """Open-Meteo's 'time' values are naive local-time labels when timezone
    is requested explicitly (no UTC offset in the string). Localize as IST,
    then convert to UTC so this matches every other table's instant convention."""
    naive = pd.to_datetime(time_strings)
    return naive.tz_localize(IST, ambiguous="infer", nonexistent="shift_forward").tz_convert("UTC")


def build_surface(hourly, run_time):
    valid_time = _parse_ist_grid(hourly["time"])
    df = pd.DataFrame({"valid_time": valid_time})
    for v in SURFACE_VARS:
        df[SURFACE_RENAME.get(v, v)] = hourly.get(v)
    df["is_forecast"] = df["valid_time"] > run_time
    df["run_time"] = run_time
    df["source"] = "open-meteo"
    return df


def build_pressure(hourly, run_time):
    valid_time = _parse_ist_grid(hourly["time"])
    rows = []
    for lvl in PRESSURE_LEVELS:
        d = pd.DataFrame({"valid_time": valid_time, "pressure_hpa": lvl})
        for v in PRESSURE_VARS:
            d[v] = hourly.get(f"{v}_{lvl}hPa")
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["is_forecast"] = df["valid_time"] > run_time
    df["run_time"] = run_time
    df["source"] = "open-meteo"
    return df


def main():
    run_time = datetime.now(timezone.utc)
    engine = get_engine()
    init_schema(engine)
    started = datetime.now(timezone.utc)
    try:
        hourly = fetch()
        surface = build_surface(hourly, run_time)
        pressure = build_pressure(hourly, run_time)
        # keep only rows with a usable temperature, so a partial API response does not write all-NaN rows
        surface = surface.dropna(subset=["temperature_2m"])
        pressure = pressure.dropna(subset=["temperature"])

        n1 = upsert_dataframe(engine, "weather_surface_hourly", surface, pk_cols=["valid_time", "source"])
        n2 = upsert_dataframe(engine, "weather_pressure_hourly", pressure, pk_cols=["valid_time", "pressure_hpa", "source"])
        log_run(engine, "fetch_openmeteo", started, datetime.now(timezone.utc), n1 + n2, "ok")
        print(f"weather_surface_hourly: {n1} rows | weather_pressure_hourly: {n2} rows "
              f"({PAST_DAYS} past days, {FORECAST_DAYS} forecast days)")
    except Exception as exc:
        # Covers failures in fetch()/build_* as well as the upserts, so a dropped
        # connection or a malformed API response is logged to ingestion_log the
        # same way a database error would be, instead of only failing silently
        # before the try block (and before anything was written to the log).
        log_run(engine, "fetch_openmeteo", started, datetime.now(timezone.utc), 0, "error", str(exc))
        raise


if __name__ == "__main__":
    main()
