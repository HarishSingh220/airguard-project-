"""
fire_features.py
Daily stubble / vegetation-fire features from NASA FIRMS downloads.

Inputs (place in the project folder, or change the paths below)
---------------------------------------------------------------
MODIS_CSV  fire_archive_M-C61_817146.csv   MODIS Collection 6.1 archive, 2020-2025, India
VIIRS_CSV  fire_nrt_J2V-C2_817148.csv      VIIRS 375 m (NOAA-21) near-real-time, 2024-2025

Design decisions (from analysing the two files)
-----------------------------------------------
* MODIS is the PRIMARY series. It covers 2022-2025 with one sensor, so the
  series has no break. Keep only type == 0 (presumed vegetation fires) and
  confidence >= 50.
* VIIRS detects roughly 7x more fires than MODIS, so raw counts cannot be
  combined across the switch. VIIRS is kept as separate `viirs_*` columns,
  which are NaN before 2024-01-17.
* VIIRS confidence is categorical (n, l, h). Keep n and h. Drop l.
* The two files overlap for 2024-2025. Do not add them together. The daily
  counts correlate at r = 0.85 across that overlap, so they describe the same
  fires.
* Stubble burning peaks in October-November, and wheat residue burning peaks
  in April-May (VIIRS: 55% of stubble-belt detections fall in May). Both are
  kept. The month features let the model tell them apart.
"""
from pathlib import Path
import numpy as np
import pandas as pd

MODIS_CSV = Path("fire_archive_M-C61_817146.csv")
VIIRS_CSV = Path("fire_nrt_J2V-C2_817148.csv")

DELHI_LAT, DELHI_LON = 28.6139, 77.2090
BELT_LAT = (28.0, 32.5)      # Punjab + Haryana + western UP stubble belt
BELT_LON = (73.5, 77.5)
NEAR_KM = 300                # distance band for "nearby" fires
MODIS_MIN_CONF = 50
VIIRS_OK_CONF = ("n", "h")


def _dist_bearing(lat1, lon1, lat2, lon2):
    """Distance (km) and bearing from point 1 to point 2 (deg, clockwise from north)."""
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371 * np.arcsin(np.sqrt(a))
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return dist, np.rad2deg(np.arctan2(y, x)) % 360


def load_modis(path=MODIS_CSV):
    d = pd.read_csv(path, parse_dates=["acq_date"])
    d = d[(d["type"] == 0) & (d["confidence"] >= MODIS_MIN_CONF)].copy()
    d["date"] = d["acq_date"].dt.floor("D")
    d["source"] = "modis"
    return d[["date", "latitude", "longitude", "frp", "daynight", "source"]]


def load_viirs(path=VIIRS_CSV):
    d = pd.read_csv(path, parse_dates=["acq_date"])
    d = d[d["confidence"].astype(str).isin(VIIRS_OK_CONF)].copy()
    d["date"] = d["acq_date"].dt.floor("D")
    d["source"] = "viirs"
    return d[["date", "latitude", "longitude", "frp", "daynight", "source"]]


def _in_belt(d):
    return d["latitude"].between(*BELT_LAT) & d["longitude"].between(*BELT_LON)


def _aggregate(points, prefix):
    """Daily counts and FRP for all fires, belt fires, and nearby fires."""
    p = points.copy()
    p["_belt"] = _in_belt(p)
    dist, _ = _dist_bearing(DELHI_LAT, DELHI_LON, p["latitude"].to_numpy(), p["longitude"].to_numpy())
    p["_near"] = dist <= NEAR_KM
    g = p.groupby("date")
    out = pd.DataFrame({
        f"{prefix}_count": g.size(),
        f"{prefix}_frp": g["frp"].sum(),
        f"{prefix}_belt_count": g["_belt"].sum(),
        f"{prefix}_belt_frp": p[p["_belt"]].groupby("date")["frp"].sum(),
        f"{prefix}_near_count": g["_near"].sum(),
        f"{prefix}_night_count": p[p["daynight"] == "N"].groupby("date").size(),
    })
    return out


def _upwind_index(points, transport_from_by_date):
    """Transport-weighted fire index: fires upwind of Delhi on the day's wind,
    weighted by distance decay and FRP. Requires the day's transport direction
    (direction the wind comes FROM), which comes from ERA5 in build_features."""
    p = points[points["date"].isin(transport_from_by_date.index)].copy()
    p = p[p["latitude"].between(20, 40) & p["longitude"].between(65, 100)]  # cheap pre-filter
    dist, bearing = _dist_bearing(DELHI_LAT, DELHI_LON, p["latitude"].to_numpy(), p["longitude"].to_numpy())
    p["_dist"], p["_bearing"] = dist, bearing
    p = p[p["_dist"] <= 800]
    p["_from"] = p["date"].map(transport_from_by_date)
    align = np.clip(np.cos(np.deg2rad(p["_bearing"] - p["_from"])), 0, None)
    p["_w"] = align ** 2 * np.exp(-p["_dist"] / 300.0)
    return p.assign(_fw=p["frp"] * p["_w"]).groupby("date")["_fw"].sum()


def fire_daily(transport_from_by_date=None, start="2022-01-01", end="2025-12-31",
               modis_path=MODIS_CSV, viirs_path=VIIRS_CSV):
    """
    Build the daily fire table.

    transport_from_by_date : pd.Series indexed by daily timestamp, values in
        degrees (direction wind comes FROM). Pass None to skip the upwind index.

    Returns a DataFrame indexed by `date` with MODIS columns (all years in range)
    and VIIRS columns (NaN before 2024-01-17). No future data is used here;
    lags are added in build_features.py.
    """
    cal = pd.date_range(start, end, freq="D", name="date")

    modis = load_modis(modis_path)
    modis = modis[(modis["date"] >= cal[0]) & (modis["date"] <= cal[-1])]
    m_agg = _aggregate(modis, "modis").reindex(cal).fillna(0.0)

    out = m_agg.copy()
    if transport_from_by_date is not None:
        ti = _upwind_index(modis, transport_from_by_date.reindex(cal).dropna())
        out["modis_ti"] = ti.reindex(cal).fillna(0.0)

    if Path(viirs_path).exists():
        viirs = load_viirs(viirs_path)
        viirs = viirs[(viirs["date"] >= cal[0]) & (viirs["date"] <= cal[-1])]
        v_start = viirs["date"].min()
        v_agg = _aggregate(viirs, "viirs").reindex(cal).fillna(0.0)
        if transport_from_by_date is not None:
            ti_v = _upwind_index(viirs, transport_from_by_date.reindex(cal).dropna())
            v_agg["viirs_ti"] = ti_v.reindex(cal).fillna(0.0)
        # VIIRS does not exist before its first date: NaN, not zero, so models
        # can tell "no sensor" from "no fires".
        v_agg.loc[v_agg.index < v_start, :] = np.nan
        out = out.join(v_agg)
    else:
        print(f"VIIRS file not found ({viirs_path}); only MODIS features produced.")

    return out.fillna({c: 0.0 for c in out.columns if c.startswith("modis")})


def diagnostics(fd):
    """Quick sanity table. Use it to check the output after each run."""
    print("Daily fire table:", fd.shape, fd.index.min().date(), "to", fd.index.max().date())
    print("MODIS mean daily count by year:",
          fd["modis_count"].groupby(fd.index.year).mean().round(1).to_dict())
    if "viirs_count" in fd:
        print("VIIRS mean daily count (2024+):",
              fd.loc["2024":, "viirs_count"].mean().round(1))
    print("Share of MODIS belt fires in Oct-Nov:",
          round(fd.loc[fd.index.month.isin([10, 11]), "modis_belt_count"].sum()
                / fd["modis_belt_count"].sum() * 100, 1), "%")


if __name__ == "__main__":
    fd = fire_daily(transport_from_by_date=None)
    diagnostics(fd)
    fd.to_csv("daily_fire_features.csv")
    print("Saved daily_fire_features.csv")
