"""
fetch_cpcb.py
Pull the current CPCB real-time reading for each Delhi station from the
official data.gov.in resource and upsert it into cpcb_hourly.

Resource: "Real time Air Quality Index from various locations"
    https://www.data.gov.in/resource/real-time-air-quality-index-various-locations
    https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69

This is CPCB's own live feed (refreshed roughly every 15-60 minutes), republished
through the government's open-data API, which is documented and stable, unlike
the CAAQM dashboard's internal endpoints. Each call returns the latest reading
per station and pollutant; it is not a time series and there is no history/
backfill endpoint, so a missed poll is a permanent gap for that window -- this
script runs on its own, separate, much more frequent schedule
(.github/workflows/ingest_cpcb.yml, every 15 minutes) rather than alongside
run_all.py's 3-hourly one, precisely to keep that gap small. See
ingest_cpcb.yml's header comment for the full reasoning.

If a station in config.STATIONS does not appear in the response, the exact name
CPCB uses may have changed; run this file directly to print the station names
seen and fix config.STATIONS.cpcb_name.

If config.CPCB_PROXY_URL is set, requests go through that relay instead of
straight to api.data.gov.in -- see cpcb_proxy_worker.js and config.py's docstring.
The proxy only helps with GitHub-Actions-IP-specific blocking; a 5xx/52x that
comes back identically through the proxy (the Worker just relays whatever
api.data.gov.in's own edge returns) means data.gov.in itself is down for
everyone right now, which no amount of retrying or proxying can route around
-- at that point the 15-minute schedule (not a longer retry budget within a
single run) is what limits the resulting gap.
"""
import math
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

from config import CPCB_API_KEY, CPCB_PROXY_URL, STATIONS
from db import get_engine, init_schema, log_run, upsert_dataframe
from http_utils import get_with_retry

RESOURCE = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
RESOURCE_PATH = f"/resource/{RESOURCE}"
DIRECT_URL = f"https://api.data.gov.in{RESOURCE_PATH}"
PAGE_SIZE = 500
EXPECTED_COLUMNS = [
    "station", "city", "pollutant", "unit", "value_avg", "value_min",
    "value_max", "observed_at", "source",
]
PRIMARY_KEY = ["station", "pollutant", "observed_at", "source"]


def request_url():
    """CPCB_PROXY_URL, if set, points at a relay (see cpcb_proxy_worker.js) that
    forwards this same /resource/<id> path to api.data.gov.in from a non-GitHub-
    Actions IP -- some of data.gov.in's outages/blocks have been specific to GH
    Actions runner ranges, which a relay sidesteps without changing anything else
    about the request (same params, same api-key, sent as a query param either way).
    Read at call time, not import time, so setting/clearing the env var takes
    effect without needing to reload this module.
    """
    return f"{CPCB_PROXY_URL}{RESOURCE_PATH}" if CPCB_PROXY_URL else DIRECT_URL

# CPCB's pollutant_id strings -> the names build_features.py/export_for_modeling.py use.
# Keys MUST be upper-case: build_dataframe() looks this dict up with
# (pollutant_id or "").strip().upper(), so a mixed-case key here (the old "NOx")
# never matches and that pollutant is silently dropped on every run.
POLLUTANT_MAP = {
    "PM2.5": "PM25", "PM10": "PM10", "NO2": "NO2", "NO": "NO", "NOX": "NOx",
    "NH3": "NH3", "SO2": "SO2", "CO": "CO", "OZONE": "O3", "O3": "O3",
    "BENZENE": "Benzene", "TOLUENE": "Toluene",
}
assert all(k == k.upper() for k in POLLUTANT_MAP), "POLLUTANT_MAP keys must be upper-case"


def fetch_all_records(state="Delhi"):
    url = request_url()
    print(f"fetching via proxy {CPCB_PROXY_URL}" if CPCB_PROXY_URL else "fetching api.data.gov.in directly")
    records, offset = [], 0
    while True:
        resp = get_with_retry(url, params={
            "api-key": CPCB_API_KEY, "format": "json", "limit": PAGE_SIZE, "offset": offset,
            "filters[state]": state,
        }, timeout=30, max_attempts=8, backoff=10.0)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"data.gov.in response at offset {offset} was not valid JSON ({exc}): "
                f"{resp.text[:200]!r}"
            ) from exc
        batch = payload.get("records", [])
        if not batch:
            break
        records.extend(batch)
        offset += PAGE_SIZE
        if len(batch) < PAGE_SIZE:
            break
    return records


def parse_timestamp(s):
    # CPCB format: "DD-MM-YYYY HH:MM:SS", given in IST (UTC+5:30) with no offset marker.
    from datetime import timedelta
    naive_ist = datetime.strptime(s, "%d-%m-%Y %H:%M:%S")
    return (naive_ist - timedelta(hours=5, minutes=30)).replace(tzinfo=timezone.utc)


def build_dataframe(records, station_lookup):
    rows = []
    for r in records:
        station_name = (r.get("station") or "").strip()
        if station_name not in station_lookup:
            continue
        pollutant = POLLUTANT_MAP.get((r.get("pollutant_id") or "").strip().upper())
        if pollutant is None:
            continue
        try:
            observed_at = parse_timestamp(r["last_update"])
        except (KeyError, ValueError):
            continue

        def num(*keys):
            # data.gov.in has exposed both pollutant_avg/min/max and
            # avg_value/min_value/max_value field names for this resource.
            # Try them in order so older responses and the current field names
            # both map into the same database columns.
            for key in keys:
                v = r.get(key)
                try:
                    value = float(v)
                    if math.isfinite(value):
                        return value
                except (TypeError, ValueError):
                    pass
            return None

        rows.append({
            "station": station_lookup[station_name], "city": "Delhi", "pollutant": pollutant,
            "unit": r.get("pollutant_unit") or r.get("unit"),
            "value_avg": num("pollutant_avg", "avg_value"),
            "value_min": num("pollutant_min", "min_value"),
            "value_max": num("pollutant_max", "max_value"),
            "observed_at": observed_at, "source": "data.gov.in",
        })
    # Keep a stable schema even when the API returns no matching records.
    # db.upsert_dataframe validates PK columns before checking df.empty.
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def _record_diagnostic(records, limit=3):
    """Return a small, credential-free sample of the response fields for logs."""
    relevant = ("station", "pollutant_id", "last_update", "pollutant_avg",
                "pollutant_min", "pollutant_max", "pollutant_unit", "avg_value",
                "min_value", "max_value", "unit")
    samples = []
    for record in records[:limit]:
        samples.append({key: record.get(key) for key in relevant if key in record})
    keys = sorted(records[0].keys()) if records else []
    return {"record_keys": keys, "sample_pollutant_fields": samples}


def main():
    if not CPCB_API_KEY:
        raise RuntimeError("CPCB_API_KEY is not set. Request your own key at data.gov.in "
                            "(My Account -> API key) and set it as an environment variable / "
                            "GitHub Actions secret -- see config.py.")
    station_lookup = {v["cpcb_name"]: k for k, v in STATIONS.items()}
    records = fetch_all_records()
    seen_stations = sorted({(r.get("station") or "").strip() for r in records})

    if "--list-stations" in sys.argv:
        print("\n".join(seen_stations))
        return

    engine = get_engine()
    init_schema(engine)
    started = datetime.now(timezone.utc)
    try:
        if not records:
            message = "data.gov.in returned no records for the Delhi filter; no rows inserted."
            log_run(engine, "fetch_cpcb", started, datetime.now(timezone.utc), 0, "ok", message)
            print(message)
            return

        df = build_dataframe(records, station_lookup)
        if df.empty:
            diagnostic = _record_diagnostic(records)
            raise RuntimeError(
                f"CPCB returned {len(records)} records, but none matched the configured stations, "
                f"pollutant IDs, and timestamp format. Configured station names: "
                f"{sorted(station_lookup)}. Response diagnostic: {diagnostic}"
            )

        # CPCB can return duplicate rows for the same database conflict key in a
        # single paginated response. PostgreSQL rejects one ON CONFLICT statement
        # that tries to update the same key twice, so retain the last source row.
        # export_for_modeling.py consumes value_avg specifically. Rows that only
        # have min/max but no average cannot contribute to the live feature table.
        value_mask = df["value_avg"].notna()
        no_value_rows = int((~value_mask).sum())
        if not value_mask.any():
            diagnostic = _record_diagnostic(records)
            raise RuntimeError(
                f"CPCB returned {len(records)} station/pollutant records, but all "
                f"{len(df)} mapped rows had NULL/non-numeric pollutant_avg values. "
                f"Refusing to persist rows that cannot contribute to the live feature export. "
                f"Response diagnostic: {diagnostic}"
            )
        df = df.loc[value_mask].copy()
        before_dedup = len(df)
        df = df.drop_duplicates(subset=PRIMARY_KEY, keep="last")
        duplicate_rows = before_dedup - len(df)

        matched = sorted(set(df["station"])) if not df.empty else []
        missing = [name for name in STATIONS if name not in matched]
        if missing:
            print(f"WARNING: no valid current reading for {missing}. "
                  "Check their cpcb_name in config.py against station names returned by CPCB:")
            print("\n".join(seen_stations))

        n = upsert_dataframe(engine, "cpcb_hourly", df, pk_cols=PRIMARY_KEY)
        notes = []
        if missing:
            notes.append(f"missing stations: {missing}")
        if no_value_rows:
            notes.append(f"dropped {no_value_rows} rows with no numeric pollutant values")
        if duplicate_rows:
            notes.append(f"collapsed {duplicate_rows} duplicate primary-key rows (kept last)")
        status = "ok" if not notes else "ok (partial)"
        message = "; ".join(notes)
        log_run(engine, "fetch_cpcb", started, datetime.now(timezone.utc), n, status, message)
        print(
            f"cpcb_hourly: upserted {n} valid, unique rows for {len(matched)} "
            f"of {len(STATIONS)} stations; dropped_no_value={no_value_rows}, "
            f"deduplicated={duplicate_rows}"
        )
    except Exception as exc:
        log_run(engine, "fetch_cpcb", started, datetime.now(timezone.utc), 0, "error", str(exc))
        raise


if __name__ == "__main__":
    main()
