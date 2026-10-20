"""
era5_inputs.py
Daily ERA5 meteorology for Delhi, built from three CSVs (all timestamps are IST,
labelled at :30, e.g. 05:30 = 00:00 UTC):

    era5_surface_hourly_ist.csv         hourly surface and derived boundary-layer fields
    era5_pressure_levels_6h_ist.csv     6-hourly, 13 levels (50-1000 hPa): t, u, v, r, q, w
    era5_pressure_levels_12h_ist.csv    12-hourly, 4 grid points around Delhi, 700/800/900 hPa

Everything returned here is a SAME-DAY value for each IST calendar day. build_features.py
lags every column before it is used, so nothing from day t reaches a forecast of day t
unless build_features.py says so.

Conventions
-----------
* Inversion variables are "temperature aloft minus temperature below", so a positive
  value means temperature rises with height (a stable, capped layer).
* Pressure (era5_pressure) is converted from Pa to hPa.
* 1000 hPa is usually below Delhi's surface (mean surface pressure is about 983 hPa), so
  no feature uses the 1000 hPa level.
* era5_inv_strength is identical to era5_inv_surface in the supplied file and is not used.
* era5_daily_features.csv is an exact daily aggregate of the hourly file; it is used only
  as a consistency check, not as a second source.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import read_csv_checked

SURFACE_CSV = Path("era5_surface_hourly_ist.csv")
LEVELS_6H_CSV = Path("era5_pressure_levels_6h_ist.csv")
LEVELS_12H_CSV = Path("era5_pressure_levels_12h_ist.csv")
DAILY_CHECK_CSV = Path("era5_daily_features.csv")

DELHI_LAT = 28.6
MIN_SURFACE_HOURS = 18      # a day needs this many hourly rows, else all its surface features are NaN
MIN_SAMPLES_6H = 3          # out of 4 samples per day
NIGHT_HOURS = list(range(0, 7))     # 00:30-06:30 IST, when surface inversions are strongest
DAY_HOURS = list(range(10, 18))     # 10:30-17:30 IST, the mixing-layer peak

SURFACE_FAMILY = "Met surface (ERA5)"
UPPER_FAMILY = "Met upper air (ERA5)"


def _read(path, required_cols=None):
    df = read_csv_checked(path, required_cols=required_cols, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def extraterrestrial_daily_mean(dates, lat=DELHI_LAT):
    """Daily-mean solar irradiance at the top of the atmosphere (W/m2) for a latitude.
    Used to turn ERA5 surface solar into a clearness index, which falls when aerosol
    and cloud dim the sun (the aerosol-radiation feedback in the problem statement)."""
    dates = pd.DatetimeIndex(dates)
    doy = dates.dayofyear.to_numpy()
    phi = np.deg2rad(lat)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0))
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    gsc = 1361.0
    return gsc * dr / np.pi * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))


# ---------------------------------------------------------------------------
# 1. Surface and boundary layer (hourly)
# ---------------------------------------------------------------------------
SURFACE_REQUIRED_COLS = ["timestamp", "u925", "v925", "era5_inv_surface", "era5_rain", "era5_solar",
                         "era5_pblh", "era5_inv_deep", "era5_lapse_925_700", "era5_ventilation",
                         "era5_t2m", "era5_rh", "era5_pressure", "era5_wind10", "u10", "v10"]
LEVELS6_REQUIRED_COLS = ["timestamp", "pressure_hpa", "t", "u", "v", "r", "q", "w"]
LEVELS12_REQUIRED_COLS = ["timestamp", "pressure_hpa", "latitude", "longitude", "t", "u", "v", "r"]


def surface_daily(path=SURFACE_CSV):
    s = _read(path, required_cols=SURFACE_REQUIRED_COLS)
    s["date"] = s["timestamp"].dt.floor("D")
    hr = s["timestamp"].dt.hour
    s["spd925"] = np.hypot(s["u925"], s["v925"])
    s["inv_pos"] = (s["era5_inv_surface"] > 0).astype(float)
    s["wet"] = (s["era5_rain"] > 0.1).astype(float)
    s["solar"] = s["era5_solar"].clip(lower=0)

    g = s.groupby("date")
    night = s[hr.isin(NIGHT_HOURS)].groupby("date")
    day = s[hr.isin(DAY_HOURS)].groupby("date")

    out = pd.DataFrame({
        "pblh_mean": g["era5_pblh"].mean(),
        "pblh_min": g["era5_pblh"].min(),
        "pblh_max": g["era5_pblh"].max(),
        "pblh_night_mean": night["era5_pblh"].mean(),
        "inv_surface_mean": g["era5_inv_surface"].mean(),
        "inv_surface_night": night["era5_inv_surface"].mean(),
        "inv_surface_max": g["era5_inv_surface"].max(),
        "inv_hours_frac": g["inv_pos"].mean(),
        "inv_deep_mean": g["era5_inv_deep"].mean(),
        "lapse_925_700_mean": g["era5_lapse_925_700"].mean(),
        "vent_mean": g["era5_ventilation"].mean(),
        "vent_min": g["era5_ventilation"].min(),
        "vent_day_mean": day["era5_ventilation"].mean(),
        "t2m_mean": g["era5_t2m"].mean(),
        "t2m_min": g["era5_t2m"].min(),
        "t2m_max": g["era5_t2m"].max(),
        "rh_mean": g["era5_rh"].mean(),
        "rh_min": g["era5_rh"].min(),
        "pressure_mean": g["era5_pressure"].mean() / 100.0,
        # sum / 24 so a missing night hour at the edge of the record does not inflate the mean
        "solar_mean": g["solar"].sum() / 24.0,
        "solar_max": g["solar"].max(),
        "rain_sum": g["era5_rain"].sum(),
        "rain_hours_frac": g["wet"].mean(),
        "wind10_mean": g["era5_wind10"].mean(),
        "u10_mean": g["u10"].mean(),
        "v10_mean": g["v10"].mean(),
        "u925_mean": g["u925"].mean(),
        "v925_mean": g["v925"].mean(),
        "spd925_mean": g["spd925"].mean(),
    })

    n = g["era5_pblh"].count()
    out.loc[n < MIN_SURFACE_HOURS, :] = np.nan

    out["dtr"] = out["t2m_max"] - out["t2m_min"]
    out["clearness_index"] = out["solar_mean"] / extraterrestrial_daily_mean(out.index)

    # Transport at 925 hPa (about 700-800 m): speed of the mean wind and the direction it blows FROM,
    # as a unit vector so the circular variable is usable by linear models.
    spd = np.hypot(out["u925_mean"], out["v925_mean"])
    out["transport_speed"] = spd
    safe = spd.where(spd > 1e-6)
    out["transport_from_sin"] = -out["u925_mean"] / safe
    out["transport_from_cos"] = -out["v925_mean"] / safe
    # vector-mean speed over scalar-mean speed: near 0 means the air is going round in circles
    out["recirculation_925"] = spd / out["spd925_mean"].where(out["spd925_mean"] > 1e-6)
    out["wind10_steadiness"] = np.hypot(out["u10_mean"], out["v10_mean"]) / out["wind10_mean"].where(out["wind10_mean"] > 1e-6)
    return out


# ---------------------------------------------------------------------------
# 2. Upper air, 6-hourly, regional mean, 13 levels
# ---------------------------------------------------------------------------
def levels6_daily(path=LEVELS_6H_CSV):
    p = _read(path, required_cols=LEVELS6_REQUIRED_COLS)
    p["spd"] = np.hypot(p["u"], p["v"])
    wide = p.pivot(index="timestamp", columns="pressure_hpa", values=["t", "u", "v", "r", "q", "w", "spd"])

    def L(var, lvl):
        return wide[(var, float(lvl))]

    f = pd.DataFrame(index=wide.index)
    for lvl in (925, 850, 700, 500):
        f[f"t{lvl}"] = L("t", lvl)
    for lvl in (925, 850, 700):
        f[f"r{lvl}"] = L("r", lvl)
    f["q925"] = L("q", 925)
    f["w850"] = L("w", 850)          # Pa/s, positive = sinking air (suppresses mixing)
    f["w700"] = L("w", 700)
    for lvl in (925, 850, 700, 500, 300):
        f[f"spd{lvl}"] = L("spd", lvl)
    f["u850"] = L("u", 850)
    f["v850"] = L("v", 850)
    f["dT_850_925"] = L("t", 850) - L("t", 925)    # > 0: warm air above the 925 hPa layer (capping inversion)
    f["dT_700_850"] = L("t", 700) - L("t", 850)
    f["cap_850_925"] = (f["dT_850_925"] > 0).astype(float)
    f["shear_925_850"] = np.hypot(L("u", 850) - L("u", 925), L("v", 850) - L("v", 925))

    date = f.index.floor("D")
    hr = f.index.hour
    daily = f.groupby(date).mean()
    n = f["t925"].groupby(date).count()
    daily.loc[n < MIN_SAMPLES_6H, :] = np.nan

    # 05:30 IST is the coldest part of the night, when inversions are at their strongest
    morning = f[hr == 5].copy()
    morning.index = morning.index.floor("D")
    daily["dT_850_925_0530"] = morning["dT_850_925"]
    daily["r925_0530"] = morning["r925"]
    daily.index.name = "date"
    return daily


# ---------------------------------------------------------------------------
# 3. Upper air, 12-hourly, 4 grid points around Delhi (700 / 800 / 900 hPa)
# ---------------------------------------------------------------------------
def levels12_daily(path=LEVELS_12H_CSV):
    p = _read(path, required_cols=LEVELS12_REQUIRED_COLS)
    p["spd"] = np.hypot(p["u"], p["v"])
    box = p.groupby(["timestamp", "pressure_hpa"])[["t", "u", "v", "r", "spd"]].mean()   # mean of the 4 points
    wide = box.unstack("pressure_hpa")

    f = pd.DataFrame(index=wide.index)
    f["t900"] = wide[("t", 900.0)]
    f["t800"] = wide[("t", 800.0)]
    f["u900"] = wide[("u", 900.0)]
    f["v900"] = wide[("v", 900.0)]
    f["spd900"] = wide[("spd", 900.0)]
    f["spd800"] = wide[("spd", 800.0)]
    f["r900"] = wide[("r", 900.0)]
    f["r800"] = wide[("r", 800.0)]
    f["dT_800_900"] = f["t800"] - f["t900"]
    f["cap_800_900"] = (f["dT_800_900"] > 0).astype(float)

    # horizontal temperature gradients at 900 hPa across the 0.25 degree box (north minus south, east minus west)
    p9 = p[p["pressure_hpa"] == 900]
    ns = p9.groupby(["timestamp", "latitude"])["t"].mean().unstack("latitude")
    ew = p9.groupby(["timestamp", "longitude"])["t"].mean().unstack("longitude")
    f["t900_ns_grad"] = ns.iloc[:, -1] - ns.iloc[:, 0]
    f["t900_ew_grad"] = ew.iloc[:, -1] - ew.iloc[:, 0]

    date = f.index.floor("D")
    hr = f.index.hour
    daily = f.groupby(date).mean()
    morning = f[hr == 5].copy()
    morning.index = morning.index.floor("D")
    daily["dT_800_900_0530"] = morning["dT_800_900"]
    daily.index.name = "date"
    return daily


# ---------------------------------------------------------------------------
# 4. Combine
# ---------------------------------------------------------------------------
def _check_against_daily_file(surface):
    """era5_daily_features.csv is an exact aggregate of the hourly file. A mismatch means
    the two files come from different downloads, so say so loudly."""
    if not DAILY_CHECK_CSV.exists():
        return
    try:
        ref = pd.read_csv(DAILY_CHECK_CSV, parse_dates=["date"]).set_index("date")
    except Exception as exc:
        # This file is only a consistency check, not a pipeline input -- a corrupt or
        # unreadable copy should not stop the feature build, just skip the check.
        print(f"WARNING: could not read {DAILY_CHECK_CSV} for the consistency check ({exc}); skipping it.")
        return
    pairs = {"pblh_mean": "pblh_mean", "pblh_max": "pblh_max", "t2m_mean": "t2m_mean",
             "t2m_min": "t2m_min", "t2m_max": "t2m_max", "rh_mean": "rh_mean",
             "wind10_mean": "wind10_mean", "vent_mean": "ventilation_mean", "rain_sum": "rain_sum_mm"}
    for mine, theirs in pairs.items():
        j = pd.concat([surface[mine], ref[theirs]], axis=1, keys=["a", "b"]).dropna()
        if len(j) and (j["a"] - j["b"]).abs().max() > 1e-6 * max(1.0, j["b"].abs().max()):
            print(f"WARNING: {mine} differs from {DAILY_CHECK_CSV.name}:{theirs}; the files may not match.")


def build_era5_daily():
    """Return (daily dataframe indexed by IST date, {column: feature family})."""
    try:
        a = surface_daily()
    except Exception as exc:
        raise RuntimeError(f"Failed to build ERA5 surface daily features from {SURFACE_CSV}: {exc}") from exc
    try:
        b = levels6_daily().add_prefix("ua_")
    except Exception as exc:
        raise RuntimeError(f"Failed to build ERA5 6-hourly pressure-level features from {LEVELS_6H_CSV}: {exc}") from exc
    try:
        c = levels12_daily().add_prefix("ua12_")
    except Exception as exc:
        raise RuntimeError(f"Failed to build ERA5 12-hourly pressure-level features from {LEVELS_12H_CSV}: {exc}") from exc
    _check_against_daily_file(a)
    daily = pd.concat([a, b, c], axis=1).sort_index()
    groups = {col: SURFACE_FAMILY for col in a.columns}
    groups.update({col: UPPER_FAMILY for col in list(b.columns) + list(c.columns)})
    return daily, groups
