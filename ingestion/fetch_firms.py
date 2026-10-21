"""
fetch_firms.py
Pull near-real-time MODIS and VIIRS fire detections over the Punjab-Haryana-
Delhi NCR box from NASA FIRMS and upsert the raw detections into
fire_detections_raw. No aggregation happens here: ingestion/rebuild_fire_daily.py
combines this table with your two historical archive CSVs and calls
fire_features.fire_daily() unchanged, so its transport-weighting logic is
reused rather than re-implemented.

API: https://firms.modaps.eosdis.nasa.gov/api/area/
    /api/area/csv/{MAP_KEY}/{SOURCE}/{west},{south},{east},{north}/{DAY_RANGE}
DAY_RANGE=1 returns the last 24 hours, which is what a 1-3 hourly schedule needs;
FIRMS NRT detections typically land a few hours after the satellite overpass.

Schema note: your historical fire_nrt_J2V-C2_817148.csv (VIIRS) uses the column
name bright_t31 for its brightness-temperature field, the same name MODIS uses
-- not bright_ti4/bright_ti5, which is what FIRMS's own VIIRS documentation
normally calls it. The live API may return either name depending on product;
this script captures whichever appears and fetch_firms stores it under
bright_t31 so a later concatenation with your historical file lines up without
guessing at export time.
"""
import io
from datetime import datetime, timezone

import pandas as pd
import requests

from config import FIRE_BBOX, FIRMS_MAP_KEY
from db import get_engine, init_schema, log_run, upsert_dataframe
from http_utils import get_with_retry

SOURCES = ["MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]
DAY_RANGE = 1
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def fetch_source(source):
    west, south, east, north = FIRE_BBOX
    url = f"{BASE_URL}/{FIRMS_MAP_KEY}/{source}/{west},{south},{east},{north}/{DAY_RANGE}"
    resp = get_with_retry(url, timeout=60)
    resp.raise_for_status()
    text = resp.text
    if text.lower().startswith("invalid") or "error" in text[:200].lower():
        raise RuntimeError(f"FIRMS error for {source}: {text[:200]}")
    if not text.strip():
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.ParserError as exc:
        raise RuntimeError(f"FIRMS response for {source} could not be parsed as CSV: {exc}") from exc
    if df.empty:
        return df
    inst = "MODIS" if source.startswith("MODIS") else "VIIRS"
    df["satellite"] = df.get("satellite", source)
    df["instrument"] = inst
    df["source"] = "firms-nrt"
    # Normalise whichever brightness-temperature field the API used to bright_t31,
    # matching your historical VIIRS archive file (see module docstring).
    if "bright_t31" not in df.columns:
        for alt in ("bright_ti4", "bright_ti5"):
            if alt in df.columns:
                df["bright_t31"] = df[alt]
                break
    keep = ["satellite", "instrument", "acq_date", "acq_time", "latitude", "longitude", "frp", "confidence",
            "daynight", "brightness", "bright_t31", "scan", "track", "version", "source"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
    df["confidence"] = df["confidence"].astype(str)
    df["version"] = df["version"].astype(str)
    return df[keep]


def main():
    frames = []
    errors = []
    for source in SOURCES:
        try:
            frames.append(fetch_source(source))
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(subset=["satellite", "instrument", "acq_date", "acq_time", "latitude", "longitude"])

    # All sources failing is a real failure, not a partial one -- it means zero new fire
    # detections were ingested this run. Distinguish it from a genuine partial success
    # (some sources ok, some down) and raise so run_all.py's per-job try/except -- which
    # only catches exceptions -- actually marks this job failed instead of treating the
    # merely-logged "ok (partial)" status as completion.
    all_failed = len(errors) == len(SOURCES)

    engine = get_engine()
    init_schema(engine)
    started = datetime.now(timezone.utc)
    try:
        n = upsert_dataframe(engine, "fire_detections_raw", df,
                             pk_cols=["satellite", "instrument", "acq_date", "acq_time", "latitude", "longitude"])
        status = "ok" if not errors else ("error" if all_failed else "ok (partial)")
        log_run(engine, "fetch_firms", started, datetime.now(timezone.utc), n, status, "; ".join(errors))
        print(f"fire_detections_raw: upserted {n} rows from {len(frames)} of {len(SOURCES)} sources")
        if errors:
            print("Source errors:", *errors, sep="\n  ")
        if all_failed:
            raise RuntimeError(f"All {len(SOURCES)} FIRMS sources failed; 0 new fire detections "
                                f"ingested this run: {'; '.join(errors)}")
    except Exception as exc:
        if not all_failed:
            log_run(engine, "fetch_firms", started, datetime.now(timezone.utc), 0, "error", str(exc))
        raise


if __name__ == "__main__":
    main()
