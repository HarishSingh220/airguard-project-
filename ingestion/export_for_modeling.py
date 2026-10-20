"""
export_for_modeling.py
Query Postgres (historical + live tables) and write the same CSVs
build_features.py, era5_inputs.py, cams_inputs.py and fire_features.py already
read, so the feature pipeline, the v9 notebook and the dashboard do not need to
change at all. Run this before build_features.py in the daily rebuild workflow.

What each output covers:
    final_delhi_dataset.csv            historical station_hourly_hist, extended
                                       with cpcb_hourly reshaped to the same wide
                                       columns, up to "now"
    era5_surface_hourly_ist.csv        historical era5_surface_hist, extended with
                                       weather_surface_hourly (observed hours only,
                                       is_forecast = false) renamed to era5_* columns
    era5_pressure_levels_6h_ist.csv    historical era5_pressure6h_hist, extended with
                                       weather_pressure_hourly (observed hours only).
                                       q (specific humidity) and w (vertical velocity)
                                       are NOT available from Open-Meteo's general
                                       forecast model and are left NaN in the live
                                       tail; features built only from them
                                       (q925_lag1d, ua_w850_lag1d, ...) will have more
                                       missing values after the historical cutoff.
    era5_pressure_levels_12h_ist.csv   historical only (era5_pressure12h_hist).
                                       There is no live 4-point-grid equivalent here,
                                       so ua12_* features go to NaN beyond the
                                       historical period and may get dropped by
                                       build_features.py's DEAD_LIMIT if that pulls
                                       their missing share over 50%.
    cams_3h_ist.csv                    historical cams_3h_hist, extended with
                                       cams_hourly (observed hours only)
    live_fire_modis.csv,
    live_fire_viirs.csv                fire_detections_raw split by instrument and
                                       reshaped to exactly match
                                       fire_archive_M-C61_817146.csv /
                                       fire_nrt_J2V-C2_817148.csv's own columns. Run
                                       ingestion/rebuild_fire_daily.py next: it
                                       concatenates each with its historical file and
                                       calls fire_features.fire_daily() unchanged to
                                       regenerate fire_daily_features.csv.
    weather_forecast_future.csv        the forward-looking (is_forecast = true) rows
                                       from weather_surface_hourly, weather_pressure_hourly
                                       and cams_hourly, kept separate. Not read by
                                       build_features.py; exported for a future
                                       extension (notebook section 6.4) or manual use.

A small physics note on the derived era5_* columns: era5_inv_surface,
era5_inv_deep, era5_lapse_925_700 and era5_ventilation were pre-computed
upstream of the original era5_surface_hourly_ist.csv by a builder this project
did not have access to, so their exact formulas are not known here. The
formulas below are a reasonable physical approximation (documented inline) but
may not match the historical file's values exactly; expect a small, visible
step at the historical/live boundary in those four columns specifically. All
other era5_* and cams_* columns are a straight rename/unit-match of the source
field, so they should not show a boundary step.
"""
import numpy as np
import pandas as pd
from sqlalchemy import text

from config import PRESSURE_LEVELS
from db import get_engine

OUT = {
    "station": "final_delhi_dataset.csv",
    "era5_surface": "era5_surface_hourly_ist.csv",
    "era5_pressure6h": "era5_pressure_levels_6h_ist.csv",
    "era5_pressure12h": "era5_pressure_levels_12h_ist.csv",
    "cams": "cams_3h_ist.csv",
    "fire_modis": "live_fire_modis.csv",
    "fire_viirs": "live_fire_viirs.csv",
}


IST = "Asia/Kolkata"

# Explicit provider-priority order, lowest number = preferred. Several ingestion
# paths can write rows for the same (valid_time[, pressure_hpa]) key -- e.g. a
# live Open-Meteo pull and an archive backfill both land in weather_surface_hourly,
# and cds_backfill.py's ERA5 q/w rows share a timestamp+pressure with the regular
# Open-Meteo meteorological row in weather_pressure_hourly but carry different
# (non-overlapping) columns. Picking a record by sort order / drop_duplicates
# alone is undefined for ties and can silently keep the sparse row over the
# complete one, or trigger a duplicate-index pivot error. canonicalize() below
# resolves every one of these tables to exactly one row per key, preferring the
# highest-priority source for each column but back-filling any column that
# source left NULL from a lower-priority source at the same key -- so a sparse
# CDS q/w row never replaces a complete Open-Meteo meteorological row; it only
# fills the two columns the meteorological row doesn't have.
SOURCE_PRIORITY = {
    "weather_surface_hourly": {"open-meteo": 0, "open-meteo-archive": 1},
    "weather_pressure_hourly": {"open-meteo": 0, "open-meteo-archive": 1, "era5-cds-backfill": 2},
    "weather_pressure12h_hourly": {"open-meteo": 0, "open-meteo-archive": 1},
    "cams_hourly": {"cams-ads": 0, "cams-ads-backfill": 1, "open-meteo-aq": 2},
}


def canonicalize(df, key_cols, table):
    """Collapse df to one row per key_cols, per the SOURCE_PRIORITY rule above.
    Drops the 'source' column (it no longer identifies a single row once
    sources have been merged)."""
    if df.empty or "source" not in df.columns:
        return df
    priority = SOURCE_PRIORITY.get(table, {})
    df = df.copy()
    df["_prio"] = df["source"].map(priority).fillna(len(priority) + 1)
    df = df.sort_values(key_cols + ["_prio"])
    # Within each key, bfill propagates values from lower-priority (higher _prio)
    # rows backward into NULLs of the higher-priority row above them, then we keep
    # only the top-priority row per key (now filled in from the others).
    value_cols = [c for c in df.columns if c not in key_cols + ["_prio", "source"]]
    df[value_cols] = df.groupby(key_cols)[value_cols].bfill()
    df = df.groupby(key_cols, as_index=False).first()
    return df.drop(columns=["_prio", "source"], errors="ignore")


def q(engine, sql, **params):
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def to_naive_ist(series):
    """Convert a tz-aware (UTC, post-migration-fix) timestamp column back to
    naive IST wall-clock values, matching what the historical _ist.csv files
    and build_features.py already expect. Leaves an already-naive column
    alone (belt-and-braces for any table that predates the tz fix)."""
    if getattr(series.dt, "tz", None) is not None:
        return series.dt.tz_convert(IST).dt.tz_localize(None)
    return series


def floor_to_ist_hour(utc_series):
    """Floor a tz-aware UTC instant column to the IST hour, not the UTC hour.
    IST is UTC+5:30, so an IST-hour-aligned grid (00:00, 01:00, ... IST --
    what the historical CPCB/CAAQMS data and build_features.py use) is NOT
    aligned to UTC hour boundaries. Flooring in UTC first, as the code used
    to do, shifts every live CPCB reading by up to 30 minutes before it is
    ever compared against the IST-hour grid."""
    return utc_series.dt.tz_convert(IST).dt.floor("h").dt.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Station pollutants: historical wide table, extended with live long-format CPCB.
#
# Important gap: the data.gov.in CPCB resource gives pollutant concentrations
# ONLY (PM2.5, PM10, NO2, NH3, SO2, CO, OZONE, ...). It does not carry the
# on-site meteorological columns your historical final_delhi_dataset.csv has
# per station (temperature, humidity, wind_speed, wind_direction, pressure,
# solar_radiation, total_rainfall -- CPCB's own AT/RH/WS/WD/BP/SR/TOT-RF
# sensors). For live rows, those seven columns are backfilled from the single
# Delhi-wide Open-Meteo pull (weather_surface_hourly), applied identically to
# all five stations -- a city-wide proxy, not each station's own sensor.
# pressure is converted hPa -> mmHg to match the historical column's unit
# (BP (mmHg) in the original CPCB download). rainfall (the 15-minute RF column,
# distinct from the cumulative TOT-RF) and vertical_wind_speed have no live
# source and stay NaN; build_features.py already drops both columns outright
# (DROP_COLS), so this has no effect on the model.
# ---------------------------------------------------------------------------
WEATHER_BACKFILL = {
    "temperature": "temperature_2m", "humidity": "relative_humidity_2m",
    "wind_speed": "wind_speed_10m", "wind_direction": "wind_direction_10m",
    "solar_radiation": "shortwave_radiation", "total_rainfall": "precipitation",
}


def export_station(engine):
    hist = q(engine, 'SELECT * FROM station_hourly_hist ORDER BY "timestamp"')
    live_long = q(engine, "SELECT station, pollutant, value_avg, observed_at FROM cpcb_hourly ORDER BY observed_at")
    weather = q(engine, "SELECT valid_time, temperature_2m, relative_humidity_2m, wind_speed_10m, "
                        "wind_direction_10m, shortwave_radiation, precipitation, surface_pressure_hpa, source "
                        "FROM weather_surface_hourly WHERE is_forecast = false ORDER BY valid_time")
    weather = canonicalize(weather, ["valid_time"], "weather_surface_hourly")
    if live_long.empty:
        combined = hist
    else:
        live_long["observed_at"] = floor_to_ist_hour(live_long["observed_at"])
        live_wide = (live_long.groupby(["station", "observed_at", "pollutant"])["value_avg"].mean()
                              .unstack("pollutant").reset_index()
                              .rename(columns={"observed_at": "timestamp"}))
        if not weather.empty:
            w = weather.rename(columns={"valid_time": "timestamp"}).copy()
            w["pressure"] = w.pop("surface_pressure_hpa") * 0.750062    # hPa -> mmHg
            for dest, src in WEATHER_BACKFILL.items():
                w[dest] = w.pop(src)
            live_wide = live_wide.merge(w[["timestamp"] + list(WEATHER_BACKFILL) + ["pressure"]],
                                        on="timestamp", how="left")
        cutoff = hist["timestamp"].max() if not hist.empty else pd.Timestamp.min.tz_localize("UTC")
        live_wide = live_wide[live_wide["timestamp"] > cutoff]
        combined = pd.concat([hist, live_wide], ignore_index=True)
    combined = combined.sort_values(["station", "timestamp"]).drop_duplicates(["station", "timestamp"], keep="last")
    combined["timestamp"] = to_naive_ist(combined["timestamp"])
    combined.to_csv(OUT["station"], index=False)
    print(f"{OUT['station']}: {len(combined)} rows ({len(hist)} historical). "
          f"Live-row weather columns are a Delhi-wide proxy (see function docstring).")


# ---------------------------------------------------------------------------
# ERA5 surface: historical, extended with Open-Meteo observed hours
# ---------------------------------------------------------------------------
def derive_era5_surface(live):
    """Map Open-Meteo's field names onto the era5_* names era5_inputs.py expects,
    with the four inversion/ventilation proxies computed here (see module docstring)."""
    out = pd.DataFrame({"timestamp": live["valid_time"]})
    out["era5_pblh"] = live["boundary_layer_height"]
    out["era5_t2m"] = live["temperature_2m"]
    out["era5_rh"] = live["relative_humidity_2m"]
    out["era5_pressure"] = live["surface_pressure_hpa"] * 100.0   # hPa -> Pa, matching the historical file
    out["era5_solar"] = live["shortwave_radiation"]
    out["era5_rain"] = live["precipitation"]
    out["era5_wind10"] = live["wind_speed_10m"]
    rad = np.deg2rad(live["wind_direction_10m"])
    out["u10"] = -live["wind_speed_10m"] * np.sin(rad)
    out["v10"] = -live["wind_speed_10m"] * np.cos(rad)
    return out


def attach_derived_from_pressure(surface, pressure_live):
    """era5_inv_surface = T(925 hPa) - T(2 m): positive means warmer air sits above the
    surface (a capping inversion). era5_inv_deep = T(700 hPa) - T(925 hPa): the same idea
    over a deeper layer. era5_lapse_925_700 = T(925 hPa) - T(700 hPa) (the historical
    column's sign is unknown; this is the natural, usually-positive convention).
    era5_ventilation = boundary_layer_height x wind_speed_10m, the standard ventilation-
    coefficient approximation. u925/v925 from the 925 hPa wind speed and direction."""
    if pressure_live.empty:
        for c in ["era5_inv_surface", "era5_inv_deep", "era5_lapse_925_700", "u925", "v925"]:
            surface[c] = np.nan
        surface["era5_ventilation"] = surface["era5_pblh"] * surface["era5_wind10"]
        return surface
    p = pressure_live.pivot(index="valid_time", columns="pressure_hpa", values=["temperature", "wind_speed", "wind_direction"])
    t925 = p[("temperature", 925)] if ("temperature", 925) in p.columns else pd.Series(dtype=float)
    t700 = p[("temperature", 700)] if ("temperature", 700) in p.columns else pd.Series(dtype=float)
    merged = surface.merge(t925.rename("t925").reset_index().rename(columns={"valid_time": "timestamp"}),
                           on="timestamp", how="left")
    merged = merged.merge(t700.rename("t700").reset_index().rename(columns={"valid_time": "timestamp"}),
                          on="timestamp", how="left")
    merged["era5_inv_surface"] = merged["t925"] - merged["era5_t2m"]
    merged["era5_inv_deep"] = merged["t700"] - merged["t925"]
    merged["era5_lapse_925_700"] = merged["t925"] - merged["t700"]
    merged["era5_ventilation"] = merged["era5_pblh"] * merged["era5_wind10"]
    if ("wind_speed", 925) in p.columns:
        spd925 = p[("wind_speed", 925)].rename("spd925").reset_index().rename(columns={"valid_time": "timestamp"})
        dir925 = p[("wind_direction", 925)].rename("dir925").reset_index().rename(columns={"valid_time": "timestamp"})
        merged = merged.merge(spd925, on="timestamp", how="left").merge(dir925, on="timestamp", how="left")
        rad = np.deg2rad(merged["dir925"])
        merged["u925"] = -merged["spd925"] * np.sin(rad)
        merged["v925"] = -merged["spd925"] * np.cos(rad)
        merged = merged.drop(columns=["spd925", "dir925"])
    else:
        merged["u925"], merged["v925"] = np.nan, np.nan
    return merged.drop(columns=["t925", "t700"])


def export_era5_surface(engine):
    hist = q(engine, 'SELECT * FROM era5_surface_hist ORDER BY "timestamp"')
    live = q(engine, "SELECT * FROM weather_surface_hourly WHERE is_forecast = false ORDER BY valid_time")
    live = canonicalize(live, ["valid_time"], "weather_surface_hourly")
    pressure_live = q(engine, "SELECT * FROM weather_pressure_hourly WHERE is_forecast = false ORDER BY valid_time")
    pressure_live = canonicalize(pressure_live, ["valid_time", "pressure_hpa"], "weather_pressure_hourly")
    if live.empty:
        combined = hist
    else:
        derived = derive_era5_surface(live)
        derived = attach_derived_from_pressure(derived, pressure_live)
        cutoff = hist["timestamp"].max() if not hist.empty else pd.Timestamp.min.tz_localize("UTC")
        derived = derived[derived["timestamp"] > cutoff]
        cols = [c for c in hist.columns if c != "timestamp"]
        for c in cols:
            if c not in derived.columns:
                derived[c] = np.nan
        combined = pd.concat([hist, derived[["timestamp"] + cols]], ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    combined["timestamp"] = to_naive_ist(combined["timestamp"])
    combined.to_csv(OUT["era5_surface"], index=False)
    print(f"{OUT['era5_surface']}: {len(combined)} rows ({len(hist)} historical)")
    return combined["timestamp"].max() if not combined.empty else None


# ---------------------------------------------------------------------------
# ERA5 pressure levels, 6-hourly: historical, extended with Open-Meteo (u/v
# derived from speed/direction; q and w left NaN, see module docstring)
# ---------------------------------------------------------------------------
def export_era5_pressure6h(engine):
    hist = q(engine, 'SELECT * FROM era5_pressure6h_hist ORDER BY "timestamp", pressure_hpa')
    live = q(engine, "SELECT * FROM weather_pressure_hourly WHERE is_forecast = false "
                     "AND pressure_hpa = ANY(:levels) ORDER BY valid_time, pressure_hpa",
            levels=list(PRESSURE_LEVELS))
    live = canonicalize(live, ["valid_time", "pressure_hpa"], "weather_pressure_hourly")
    if live.empty:
        combined = hist
    else:
        out = pd.DataFrame({"timestamp": live["valid_time"], "pressure_hpa": live["pressure_hpa"],
                            "t": live["temperature"], "r": live["relative_humidity"],
                            # Real values where cds_backfill.py has landed them (source='era5-cds-backfill'
                            # rows); NaN everywhere else (Open-Meteo-only rows), same as before this column
                            # existed. Open-Meteo itself never provides either quantity at any endpoint.
                            "q": live["specific_humidity"], "w": live["vertical_velocity"]})
        rad = np.deg2rad(live["wind_direction"])
        out["u"] = -live["wind_speed"] * np.sin(rad)
        out["v"] = -live["wind_speed"] * np.cos(rad)
        cutoff = hist["timestamp"].max() if not hist.empty else pd.Timestamp.min.tz_localize("UTC")
        out = out[out["timestamp"] > cutoff]
        combined = pd.concat([hist, out[hist.columns if not hist.empty else out.columns]], ignore_index=True)
    combined = combined.sort_values(["timestamp", "pressure_hpa"]).drop_duplicates(["timestamp", "pressure_hpa"], keep="last")
    combined["timestamp"] = to_naive_ist(combined["timestamp"])
    combined.to_csv(OUT["era5_pressure6h"], index=False)
    print(f"{OUT['era5_pressure6h']}: {len(combined)} rows ({len(hist)} historical)")


def export_era5_pressure12h(engine):
    hist = q(engine, 'SELECT * FROM era5_pressure12h_hist ORDER BY "timestamp"')
    # weather_pressure12h_hourly: populated by backfill_openmeteo_12h_grid.py (t/u/v/r
    # only, same 4-point/700-800-900hPa box as the historical table -- no CDS needed here).
    live = q(engine, "SELECT * FROM weather_pressure12h_hourly WHERE is_forecast = false "
                     "ORDER BY valid_time, latitude, longitude, pressure_hpa")
    live = canonicalize(live, ["valid_time", "latitude", "longitude", "pressure_hpa"], "weather_pressure12h_hourly")
    if live.empty:
        combined = hist
    else:
        out = pd.DataFrame({"timestamp": live["valid_time"], "latitude": live["latitude"],
                            "longitude": live["longitude"], "pressure_hpa": live["pressure_hpa"],
                            "t": live["temperature"], "r": live["relative_humidity"]})
        rad = np.deg2rad(live["wind_direction"])
        out["u"] = -live["wind_speed"] * np.sin(rad)
        out["v"] = -live["wind_speed"] * np.cos(rad)
        cutoff = hist["timestamp"].max() if not hist.empty else pd.Timestamp.min.tz_localize("UTC")
        out = out[out["timestamp"] > cutoff]
        combined = pd.concat([hist, out[hist.columns if not hist.empty else out.columns]], ignore_index=True)
    combined = combined.sort_values(["timestamp", "latitude", "longitude", "pressure_hpa"]) \
                        .drop_duplicates(["timestamp", "latitude", "longitude", "pressure_hpa"], keep="last")
    combined["timestamp"] = to_naive_ist(combined["timestamp"])
    combined.to_csv(OUT["era5_pressure12h"], index=False)
    print(f"{OUT['era5_pressure12h']}: {len(combined)} rows ({len(hist)} historical)")


# ---------------------------------------------------------------------------
# CAMS: historical, extended with CAMS ADS observed hours
# ---------------------------------------------------------------------------
def export_cams(engine):
    hist = q(engine, 'SELECT * FROM cams_3h_hist ORDER BY "timestamp"')
    # cams_hourly rows are always forecast output (see fetch_cams.py) -- is_forecast is
    # never false here, so "observed hours" means is_best_estimate (the short-lead rows),
    # not is_forecast = false.
    live = q(engine, "SELECT valid_time AS timestamp, cams_aod550, cams_pm25, cams_pm10, cams_no2_col, "
                     "cams_o3_col_du, source FROM cams_hourly WHERE is_best_estimate = true ORDER BY valid_time")
    live = canonicalize(live, ["timestamp"], "cams_hourly")
    if live.empty:
        combined = hist
    else:
        cutoff = hist["timestamp"].max() if not hist.empty else pd.Timestamp.min.tz_localize("UTC")
        live = live[live["timestamp"] > cutoff]
        combined = pd.concat([hist, live], ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    combined["timestamp"] = to_naive_ist(combined["timestamp"])
    combined.to_csv(OUT["cams"], index=False)
    print(f"{OUT['cams']}: {len(combined)} rows ({len(hist)} historical)")


# ---------------------------------------------------------------------------
# Fire detections, reshaped to exactly match your two historical archive CSVs'
# columns, so ingestion/rebuild_fire_daily.py can concatenate them directly and
# hand the result to fire_features.fire_daily() unchanged.
#
# MODIS gap: FIRMS's near-real-time MODIS product does not carry the "type"
# field (0 = presumed vegetation fire, 1-3 = other sources) -- that field only
# exists in the standard/science-quality archive product your fire_archive_M-C61
# file already is. fire_features.load_modis() filters on type == 0, so live
# MODIS rows are written with type = 0 for all of them (assume vegetation fire,
# the majority class in the historical file) rather than being dropped
# outright. This is a disclosed approximation, not a verified fact about each
# live detection.
# ---------------------------------------------------------------------------
ARCHIVE_COLS = ["latitude", "longitude", "brightness", "scan", "track", "acq_date", "acq_time",
                "satellite", "instrument", "confidence", "version", "bright_t31", "frp", "daynight"]


def export_fire(engine):
    live = q(engine, f"SELECT {', '.join(ARCHIVE_COLS)} FROM fire_detections_raw ORDER BY acq_date, acq_time")
    modis = live[live["instrument"] == "MODIS"].copy()
    viirs = live[live["instrument"] == "VIIRS"].copy()
    modis["confidence"] = pd.to_numeric(modis["confidence"], errors="coerce")   # MODIS confidence is 0-100
    modis["type"] = 0   # see function docstring: NRT MODIS has no "type" field
    modis[ARCHIVE_COLS + ["type"]].to_csv(OUT["fire_modis"], index=False)
    viirs[ARCHIVE_COLS].to_csv(OUT["fire_viirs"], index=False)
    print(f"{OUT['fire_modis']}: {len(modis)} rows | {OUT['fire_viirs']}: {len(viirs)} rows. "
          f"Run ingestion/rebuild_fire_daily.py next to regenerate fire_daily_features.csv.")


# ---------------------------------------------------------------------------
# Forward-looking rows, kept separate (not read by build_features.py)
# ---------------------------------------------------------------------------
def export_future(engine):
    s = q(engine, "SELECT * FROM weather_surface_hourly WHERE is_forecast = true ORDER BY valid_time")
    p = q(engine, "SELECT * FROM weather_pressure_hourly WHERE is_forecast = true ORDER BY valid_time, pressure_hpa")
    # cams_hourly: is_forecast is always true (see fetch_cams.py), so "still forward-looking,
    # not yet today's best estimate" is is_best_estimate = false here, not is_forecast = true.
    c = q(engine, "SELECT * FROM cams_hourly WHERE is_best_estimate = false ORDER BY valid_time")
    s.to_csv("weather_forecast_future_surface.csv", index=False)
    p.to_csv("weather_forecast_future_pressure.csv", index=False)
    c.to_csv("weather_forecast_future_cams.csv", index=False)
    print(f"weather_forecast_future_*.csv: {len(s)} surface, {len(p)} pressure, {len(c)} CAMS rows "
          "(forward-looking; not consumed by build_features.py yet)")


def main():
    engine = get_engine()
    export_station(engine)
    export_era5_surface(engine)
    export_era5_pressure6h(engine)
    export_era5_pressure12h(engine)
    export_cams(engine)
    export_fire(engine)
    export_future(engine)


if __name__ == "__main__":
    main()
