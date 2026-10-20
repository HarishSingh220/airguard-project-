"""
build_features.py
Daily feature table for Delhi AQI forecasting, 1 to 10 days ahead, built only from CSV inputs.

Inputs (repository root)
------------------------
final_delhi_dataset.csv                 hourly station data (pollutants, local weather)
era5_surface_hourly_ist.csv             ERA5 surface and boundary layer, hourly, IST
era5_pressure_levels_6h_ist.csv         ERA5 13 pressure levels, 6-hourly, IST
era5_pressure_levels_12h_ist.csv        ERA5 700/800/900 hPa on a 4-point grid, 12-hourly, IST
cams_3h_ist.csv                         CAMS aerosol and composition, 3-hourly, IST
fire_daily_features.csv                 FIRMS MODIS (2022+) and VIIRS (2024-01-17+) daily features
(era5_daily_features.csv and cams_daily_ist.csv are exact daily aggregates of the files above
 and are used only as consistency checks.)

Outputs (folder features/)
--------------------------
daily_features.csv       one row per station and IST day: features, same-day actuals and targets
feature_list.txt         model features (no targets, no dead columns)
feature_groups.json      feature -> family; the notebook uses it for SHAP and ablation
targets.json             target name -> column prefix and the same-day actual column
met_daily_same_day.csv   same-day ERA5 / CAMS / fire values (not lagged), for plots and the dashboard

Rules
-----
* A feature on row t uses data up to the end of day t and nothing later.
  - Station history (AQI, pollutants, lags, rolling windows including day t) is up to day t.
  - ERA5, CAMS and fire inputs are lagged EXT_LAG_DAYS (1) day. Reanalysis reaches users days
    late, so the conservative choice is to treat day t's analysis as not yet available.
* Targets are AQI, PM2.5, ozone (daily mean and 8-hour maximum) and NOx on day t+h, h = 1..10.
* Each station sits on a complete daily calendar before lags are formed, so shift(k) is k days.
* A day needs at least 18 valid hours before its AQI, PM2.5, ozone or NOx value is used.
* The future-leakage test corrupts station AND external data after a cutoff day and checks
  that no feature dated on or before the cutoff changes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import cams_inputs as ci
import era5_inputs as ei
import fire_inputs as fi
from io_utils import DataFileError, read_csv_checked

DATA_CSV = Path("final_delhi_dataset.csv")
OUT_DIR = Path("features")

FORECAST_HORIZON = 10
MIN_HOURS_PER_DAY = 18
MAX_FFILL_HOURS = 3
DEAD_LIMIT = 0.5                      # drop a feature if more than this share is missing
KEEP_SPARSE_PREFIXES = ("viirs_",)    # VIIRS only exists from 2024-01-17; the models handle the gap
USE_VIIRS = True
EXT_LAG_DAYS = 1                      # lag applied to ERA5, CAMS and fire inputs
PERIOD_START = pd.Timestamp("2022-01-01")
PERIOD_END = pd.Timestamp("2025-12-31")

# Diwali (Lakshmi Puja) dates. Firecracker smoke is a known Delhi spike and the date is known
# in advance, so it is a legitimate calendar feature. Check these against your own calendar.
DIWALI_DATES = pd.to_datetime(["2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08"])

POLLUTANTS = ["PM25", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "O3", "Benzene", "Toluene"]
AQI_POLLUTANTS = ["PM25", "PM10", "NO2", "SO2", "NH3", "CO", "O3"]
LOCAL_WEATHER = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure",
                 "solar_radiation", "total_rainfall"]
DROP_COLS = ["O_Xylene", "Eth_Benzene", "MP_Xylene", "Xylene", "rainfall",
             "vertical_wind_speed", "source_file", "city"]

# target name -> (prefix of the target columns, same-day actual column)
TARGETS = {
    "AQI": ("target_h", "d_AQI"),
    "PM25": ("target_pm25_h", "d_PM25"),
    "O3": ("target_o3_h", "d_O3"),
    "O3_8H": ("target_o3max8_h", "d_O3_max8h"),
    "NOx": ("target_nox_h", "d_NOx"),
}

# External daily columns for which "change since yesterday" and a 3-day mean are added
DYNAMIC_EXT = ["pblh_mean", "inv_surface_night", "inv_hours_frac", "vent_mean", "pressure_mean",
               "t2m_mean", "clearness_index", "dtr", "cams_pm25_mean", "cams_aod550_mean",
               "ua_dT_850_925", "ua_w850"]

BREAKPOINTS = {
    "PM25": [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200), (90, 120, 200, 300),
             (120, 250, 300, 400), (250, 500, 400, 500)],
    "PM10": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200), (250, 350, 200, 300),
             (350, 430, 300, 400), (430, 600, 400, 500)],
    "NO2": [(0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 200), (180, 280, 200, 300),
            (280, 400, 300, 400), (400, 800, 400, 500)],
    "SO2": [(0, 40, 0, 50), (40, 80, 50, 100), (80, 380, 100, 200), (380, 800, 200, 300),
            (800, 1600, 300, 400), (1600, 2100, 400, 500)],
    "CO": [(0, 1.0, 0, 50), (1.0, 2.0, 50, 100), (2.0, 10, 100, 200), (10, 17, 200, 300),
           (17, 34, 300, 400), (34, 50, 400, 500)],
    "O3": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 168, 100, 200), (168, 208, 200, 300),
           (208, 748, 300, 400), (748, 1000, 400, 500)],
    "NH3": [(0, 200, 0, 50), (200, 400, 50, 100), (400, 800, 100, 200), (800, 1200, 200, 300),
            (1200, 1800, 300, 400), (1800, 2400, 400, 500)],
}


# ---------------------------------------------------------------------------
# Calendar helpers (shared with the notebook, which builds target-day calendar features)
# ---------------------------------------------------------------------------
def diwali_offset(dates):
    """Signed days from each date to the nearest Diwali in DIWALI_DATES (negative = before)."""
    arr = pd.DatetimeIndex(dates).to_numpy("datetime64[D]")
    dd = DIWALI_DATES.to_numpy("datetime64[D]")
    diff = (arr[:, None] - dd[None, :]).astype(int)
    idx = np.abs(diff).argmin(axis=1)
    return diff[np.arange(len(arr)), idx]


def diwali_window(dates, k=3):
    return (np.abs(diwali_offset(dates)) <= k).astype(int)


# ---------------------------------------------------------------------------
# 1. Station data: complete hourly grid, causal gap fill
# ---------------------------------------------------------------------------
def load_station_data(path=DATA_CSV):
    df = read_csv_checked(path, required_cols=["timestamp", "station"], parse_dates=["timestamp"])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    present = [c for c in POLLUTANTS + LOCAL_WEATHER if c in df.columns]
    if not present:
        raise DataFileError(
            f"{path} has none of the expected pollutant/weather columns {POLLUTANTS + LOCAL_WEATHER}. "
            f"Columns present: {list(df.columns)}"
        )
    df = df.sort_values(["station", "timestamp"])
    parts = []
    for st, g in df.groupby("station"):
        full = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="h")
        g = g.set_index("timestamp").reindex(full).rename_axis("timestamp").reset_index()
        g["station"] = st
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    fill = [c for c in POLLUTANTS + LOCAL_WEATHER if c in df.columns]
    df[fill] = df.groupby("station")[fill].transform(lambda s: s.ffill(limit=MAX_FFILL_HOURS))
    return df


# ---------------------------------------------------------------------------
# 2. CPCB AQI (causal trailing averages) and the 8-hour ozone mean
# ---------------------------------------------------------------------------
def sub_index(pollutant, s):
    bps = BREAKPOINTS[pollutant]
    s = s.clip(upper=bps[-1][1])
    out = pd.Series(np.nan, index=s.index)
    for lo, hi, ilo, ihi in bps:
        m = (s >= lo) & (s <= hi)
        out[m] = (ihi - ilo) / (hi - lo) * (s[m] - lo) + ilo
    return out


def add_aqi(df):
    parts = []
    for st, g in df.groupby("station", sort=False):
        g = g.copy()
        subs = []
        for p in AQI_POLLUTANTS:
            win, mp = (8, 6) if p in ("CO", "O3") else (24, 16)
            avg = g[p].rolling(win, min_periods=mp).mean()
            if p == "O3":
                g["O3_8h"] = avg          # trailing 8-hour mean, kept for the daily 8-hour maximum
            g[f"I_{p}"] = sub_index(p, avg)
            subs.append(f"I_{p}")
        n_ok = g[subs].notna().sum(axis=1)
        has_pm = g[["I_PM25", "I_PM10"]].notna().any(axis=1)
        g["AQI"] = np.where((n_ok >= 3) & has_pm, g[subs].max(axis=1), np.nan)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Daily table
# ---------------------------------------------------------------------------
def build_daily(raw, ext, ext_groups):
    """Return (daily dataframe, groups dict). `raw` has hourly station rows, `ext` has same-day
    external features indexed by date (it is lagged here)."""
    groups = {}

    h = add_aqi(raw)
    h["date"] = h["timestamp"].dt.floor("D")
    cols = [c for c in ["AQI"] + POLLUTANTS + LOCAL_WEATHER if c in h.columns]
    g = h.groupby(["station", "date"])
    d = g[cols].mean()
    d["O3_max8h"] = g["O3_8h"].max()
    for col in ["AQI", "PM25", "O3", "NOx"]:
        d.loc[g[col].count() < MIN_HOURS_PER_DAY, col] = np.nan
    d.loc[g["O3"].count() < MIN_HOURS_PER_DAY, "O3_max8h"] = np.nan
    d = d.rename(columns={c: f"d_{c}" for c in cols + ["O3_max8h"]})

    cal = pd.date_range(PERIOD_START, PERIOD_END, freq="D", name="date")
    stations = sorted(h["station"].unique())
    full_idx = pd.MultiIndex.from_product([stations, cal], names=["station", "date"])
    d = d.reindex(full_idx).reset_index()
    d = d.sort_values(["station", "date"]).reset_index(drop=True)

    def gcol(col):
        return d.groupby("station")[col]

    def add(name, values, group):
        d[name] = values
        groups[name] = group

    def roll(col, w, stat, min_periods):
        r = d[col].groupby(d["station"]).rolling(w, min_periods=min_periods)
        return getattr(r, stat)().reset_index(level=0, drop=True)

    # AQI history. Lags are days before day t; rolling windows include day t (known at its end).
    groups["d_AQI"] = "AQI history"
    for k in [1, 2, 3, 7, 14, 30]:
        add(f"AQI_lag_{k}d", gcol("d_AQI").shift(k), "AQI history")
    for w in [3, 7, 14, 30]:
        mp = max(2, w // 2)
        add(f"AQI_roll_mean_{w}d", roll("d_AQI", w, "mean", mp), "AQI history")
        add(f"AQI_roll_std_{w}d", roll("d_AQI", w, "std", mp), "AQI history")
    add("AQI_diff_1d", d["d_AQI"] - d["AQI_lag_1d"], "AQI history")
    add("AQI_diff_7d", d["d_AQI"] - d["AQI_lag_7d"], "AQI history")

    # Pollutant history
    for c in ["d_PM25", "d_PM10", "d_NO2", "d_O3", "d_NOx", "d_O3_max8h"]:
        if c in d.columns:
            for k in [1, 2, 3, 7]:
                add(f"{c}_lag_{k}d", gcol(c).shift(k), "Pollutant history")
    for c in ["d_PM25", "d_PM10"]:
        for w in [3, 7]:
            add(f"{c}_roll_mean_{w}d", roll(c, w, "mean", 2), "Pollutant history")
    for c in ["d_PM25", "d_PM10", "d_NO2", "d_O3", "d_NOx", "d_NO", "d_NH3", "d_SO2", "d_CO", "d_O3_max8h"]:
        if c in d.columns:
            groups[c] = "Pollutant current"
    add("d_PM25_over_PM10", d["d_PM25"] / d["d_PM10"].where(d["d_PM10"] > 0), "Pollutant current")
    for c in ["d_temperature", "d_humidity", "d_wind_speed", "d_pressure"]:
        if c in d.columns:
            groups[c] = "Local weather"

    # Delhi-wide state (mean of the stations), up to day t. Pollution episodes are regional.
    city = d.groupby("date")[["d_AQI", "d_PM25"]].mean()
    for c, nm in [("d_AQI", "AQI"), ("d_PM25", "PM25")]:
        add(f"city_{nm}_today", d["date"].map(city[c]), "City state")
        add(f"city_{nm}_lag_1d", d["date"].map(city[c].shift(1)), "City state")
        add(f"city_{nm}_roll_mean_7d", d["date"].map(city[c].rolling(7, min_periods=4).mean()), "City state")

    # Calendar
    dt = d["date"]
    add("dayofweek", dt.dt.dayofweek, "Calendar")
    add("month", dt.dt.month, "Calendar")
    add("is_weekend", (dt.dt.dayofweek >= 5).astype(int), "Calendar")
    add("doy_sin", np.sin(2 * np.pi * dt.dt.dayofyear / 365.25), "Calendar")
    add("doy_cos", np.cos(2 * np.pi * dt.dt.dayofyear / 365.25), "Calendar")
    add("is_stubble_season", dt.dt.month.isin([10, 11]).astype(int), "Calendar")
    add("diwali_window", diwali_window(dt), "Calendar")
    add("days_from_diwali", np.clip(diwali_offset(dt), -15, 15), "Calendar")

    # External daily features (ERA5 surface and upper air, CAMS, fire), lagged
    lag = EXT_LAG_DAYS
    ext = ext.reindex(cal)
    ext_lag = ext.shift(lag)
    for c in ext.columns:
        add(f"{c}_lag{lag}d", d["date"].map(ext_lag[c]), ext_groups[c])

    # Fire accumulation: older day and rolling sums, for the burning-intensity columns
    for c in ext.columns:
        if ext_groups[c] == fi.FAMILY and (c.endswith("_ti") or c.endswith("belt_count") or c.endswith("belt_frp")):
            s = ext[c].shift(lag)
            add(f"{c}_lag{lag + 1}d", d["date"].map(ext[c].shift(lag + 1)), fi.FAMILY)
            add(f"{c}_roll3d", d["date"].map(s.rolling(3, min_periods=1).sum()), fi.FAMILY)
            add(f"{c}_roll7d", d["date"].map(s.rolling(7, min_periods=1).sum()), fi.FAMILY)

    # Day-to-day change and 3-day mean of the main dynamic drivers
    for c in DYNAMIC_EXT:
        if c in ext.columns:
            s = ext[c].shift(lag)
            add(f"{c}_chg1d", d["date"].map(s - s.shift(1)), ext_groups[c])
            add(f"{c}_roll3d", d["date"].map(s.rolling(3, min_periods=2).mean()), ext_groups[c])

    # Met-chemistry coupling terms: smoke arriving while the boundary layer is shallow or capped.
    # Written as explicit products because linear models cannot find interactions by themselves.
    def L(c):
        return ext[c].shift(lag)

    coupling = "Met-chem coupling"
    if {"modis_ti", "inv_hours_frac"} <= set(ext.columns):
        add(f"fire_x_inversion_lag{lag}d",
            d["date"].map(np.log1p(L("modis_ti").clip(lower=0)) * L("inv_hours_frac")), coupling)
    if {"modis_ti", "vent_mean"} <= set(ext.columns):
        add(f"fire_over_vent_lag{lag}d",
            d["date"].map(np.log1p(L("modis_ti").clip(lower=0)) / np.log1p(L("vent_mean").clip(lower=0))), coupling)
    if {"cams_pm25_mean", "pblh_mean"} <= set(ext.columns):
        add(f"cams_pm25_per_pblh_lag{lag}d",
            d["date"].map(L("cams_pm25_mean") / (L("pblh_mean") + 50.0)), coupling)

    # Targets: value on day t+h
    for name, (prefix, actual) in TARGETS.items():
        for hh in range(1, FORECAST_HORIZON + 1):
            d[f"{prefix}{hh}"] = gcol(actual).shift(-hh)

    return d, groups


def finalise(d, groups):
    """Drop dead and constant features and return the feature list and group map."""
    feats = [c for c in groups if c in d.columns]
    assert not [c for c in feats if c.startswith("target")], "a target column is in the feature set"
    dead = [c for c in feats if d[c].isna().mean() > DEAD_LIMIT and not c.startswith(KEEP_SPARSE_PREFIXES)]
    const = [c for c in feats if c not in dead and d[c].nunique(dropna=True) <= 1]
    if dead:
        print("Dropped (more than", DEAD_LIMIT, "missing):", dead)
    if const:
        print("Dropped (constant):", const)
    feats = [c for c in feats if c not in dead and c not in const]
    # ratios can produce +-inf; boosted trees reject inf, so make it a plain missing value
    d[feats] = d[feats].replace([np.inf, -np.inf], np.nan)
    return feats, {c: groups[c] for c in feats}


# ---------------------------------------------------------------------------
# 4. Future-leakage test
# ---------------------------------------------------------------------------
def leakage_test(raw, ext, ext_groups, cutoff):
    """Corrupt station data and external data after the end of day `cutoff`. Every feature
    dated on or before `cutoff` must stay identical."""
    cutoff = pd.Timestamp(cutoff).normalize()
    base, groups = build_daily(raw, ext, ext_groups)

    corrupt = raw.copy()
    late = corrupt["timestamp"] >= cutoff + pd.Timedelta(days=1)
    num = [c for c in corrupt.select_dtypes("number").columns if c not in ("hour", "dayofweek", "month")]
    corrupt[num] = corrupt[num].astype(float)
    corrupt.loc[late, num] = corrupt.loc[late, num] + 9999.0

    ext_c = ext.copy().astype(float)
    ext_c.loc[ext_c.index > cutoff] = ext_c.loc[ext_c.index > cutoff] + 9999.0

    alt, _ = build_daily(corrupt, ext_c, ext_groups)

    feats = [c for c in groups if c in base.columns]
    past_base = base[base["date"] <= cutoff]
    past_alt = alt[alt["date"] <= cutoff]
    bad = []
    for c in feats:
        a = past_base[c].to_numpy(dtype=float)
        b = past_alt[c].to_numpy(dtype=float)
        if not np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-9):
            bad.append(c)
    status = "PASS" if not bad else f"FAIL ({len(bad)} features)"
    print("Leakage test:", status, bad[:10])
    return bad


def load_external():
    try:
        era5, g1 = ei.build_era5_daily()
    except Exception as exc:
        raise RuntimeError(f"Failed to build ERA5 daily features: {exc}") from exc
    try:
        cams, g2 = ci.build_cams_daily()
    except Exception as exc:
        raise RuntimeError(f"Failed to build CAMS daily features: {exc}") from exc
    try:
        fire, g3 = fi.load_fire_daily(use_viirs=USE_VIIRS)
    except Exception as exc:
        raise RuntimeError(f"Failed to load fire daily features: {exc}") from exc
    ext = pd.concat([era5, cams, fire], axis=1).sort_index()
    ext = ext.reindex(pd.date_range(PERIOD_START, PERIOD_END, freq="D", name="date"))
    groups = {**g1, **g2, **g3}
    dup = set(g1) & set(g2) | set(g1) & set(g3) | set(g2) & set(g3)
    assert not dup, f"external column names collide: {sorted(dup)}"
    return ext, groups


def main():
    global PERIOD_END
    try:
        OUT_DIR.mkdir(exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create output directory {OUT_DIR}: {exc}") from exc

    raw = load_station_data()
    # The latest *complete* day, not just the latest raw timestamp's calendar date: a 3-hourly
    # ingestion run can land mid-morning and leave the current day with only a few hours of
    # readings. Using that day's date directly would let it become PERIOD_END (and, downstream,
    # build_final_models.py's forecast origin) while most of its hours -- and therefore the
    # MIN_HOURS_PER_DAY-gated d_AQI/d_PM25/... "current day" values -- are still missing.
    # A day needs >= MIN_HOURS_PER_DAY hours of station data, for at least one station, before
    # its calendar date is trusted as complete.
    hours_per_station_day = raw.groupby(["station", raw["timestamp"].dt.normalize()])["timestamp"].count()
    best_station_hours = hours_per_station_day.groupby(level=1).max()  # best-covered station, per day
    complete_days = best_station_hours[best_station_hours >= MIN_HOURS_PER_DAY].index
    data_end = complete_days.max() if len(complete_days) else raw["timestamp"].max().normalize() - pd.Timedelta(days=1)
    if data_end > PERIOD_END:
        # Live ingestion extends final_delhi_dataset.csv past the original 2025-12-31 default;
        # follow it rather than silently truncating new days. Never moves PERIOD_END earlier.
        print(f"Extending PERIOD_END from {PERIOD_END.date()} to {data_end.date()} (latest COMPLETE day of station data)")
        PERIOD_END = data_end

    ext, ext_groups = load_external()
    print("External daily features:", ext.shape, "| families:",
          pd.Series(ext_groups).value_counts().to_dict())
    miss_ext = ext.isna().mean().sort_values(ascending=False)
    print("External columns with the most missing days:", miss_ext.head(5).round(3).to_dict())

    cutoff = PERIOD_END - pd.Timedelta(days=180)
    bad = leakage_test(raw, ext, ext_groups, cutoff)
    if bad:
        raise SystemExit(f"Leakage test failed for: {bad}")

    daily, groups = build_daily(raw, ext, ext_groups)
    feats, groups = finalise(daily, groups)

    actuals = [a for _, a in TARGETS.values()]
    target_cols = [c for c in daily.columns if c.startswith("target")]
    # Station-measured same-day weather, saved for the coupling analysis (not model features unless listed in groups)
    obs_extra = [c for c in ["d_solar_radiation", "d_temperature", "d_wind_speed", "d_humidity", "d_total_rainfall"]
                 if c in daily.columns]
    keep = list(dict.fromkeys(["station", "date"] + feats + actuals + obs_extra + target_cols))
    try:
        daily[keep].to_csv(OUT_DIR / "daily_features.csv", index=False)
        (OUT_DIR / "feature_list.txt").write_text("\n".join(feats))
        (OUT_DIR / "feature_groups.json").write_text(json.dumps(groups, indent=2))
        (OUT_DIR / "targets.json").write_text(json.dumps(
            {k: {"prefix": p, "actual": a} for k, (p, a) in TARGETS.items()}, indent=2))
        ext.to_csv(OUT_DIR / "met_daily_same_day.csv")
    except OSError as exc:
        raise OSError(f"Could not write outputs to {OUT_DIR}: {exc}") from exc

    miss = daily[feats].isna().mean()
    print("Daily table:", daily.shape, "| model features:", len(feats))
    print("Largest missing shares:", miss.sort_values(ascending=False).head(5).round(3).to_dict())
    print("Feature families:", pd.Series(groups).value_counts().to_dict())


if __name__ == "__main__":
    try:
        main()
    except DataFileError as exc:
        raise SystemExit(f"build_features.py: input data problem -- {exc}")