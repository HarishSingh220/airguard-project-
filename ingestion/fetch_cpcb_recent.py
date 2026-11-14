"""Backfill recent hourly CPCB readings through CPCB's report export.

The data.gov.in resource used by fetch_cpcb.py is a latest-value snapshot and
cannot repair missed polling intervals. CPCB's report export exposes recent
station history without login. This job fetches a rolling 45-day window before
the daily forecast rebuild, covering the 14-day quality window and 30-day
pollutant lookbacks.
"""
from datetime import datetime, timedelta, timezone
from io import BytesIO
import math
import re
from zipfile import BadZipFile, ZipFile
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import CPCB_PROXY_URL, STATIONS
from db import get_engine, init_schema, log_run, upsert_dataframe

REPORT_URL = "https://airquality.cpcb.gov.in/cpcb-common-report/api/excel/export"
REPORT_PROXY_PATH = "/cpcb-common-report/api/excel/export"
SOURCE = "cpcb-report"
HISTORY_DAYS = 44  # date delta; includes 45 calendar days counting today
IST = ZoneInfo("Asia/Kolkata")
XML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

REPORT_PARAMETERS = [
    ("PM2.5", "PM25", "µg/m³"),
    ("PM10", "PM10", "µg/m³"),
    ("NO", "NO", "µg/m³"),
    ("NO2", "NO2", "µg/m³"),
    ("NOX", "NOx", "ppb"),
    ("NH3", "NH3", "µg/m³"),
    ("SO2", "SO2", "µg/m³"),
    ("CO", "CO", "mg/m³"),
    ("OZONE", "O3", "µg/m³"),
]
REPORT_COLUMNS = "PM25,PM10,NO,NO2,NOx,NH3,SO2,CO,Ozone"
EXPECTED_COLUMNS = [
    "station", "city", "pollutant", "unit", "value_avg", "value_min",
    "value_max", "observed_at", "source",
]
PRIMARY_KEY = ["station", "pollutant", "observed_at", "source"]


def _cell_text(cell, shared_strings):
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", XML_NS))
    value = cell.find("a:v", XML_NS)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text


def _column_index(cell_ref):
    letters = re.match(r"[A-Z]+", cell_ref or "")
    if not letters:
        raise ValueError(f"Invalid CPCB workbook cell reference: {cell_ref!r}")
    result = 0
    for char in letters.group(0):
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def parse_report_workbook(content, station, start_utc=None, end_utc=None):
    """Parse CPCB's XLSX report using only Python's standard library."""
    try:
        with ZipFile(BytesIO(content)) as workbook:
            names = set(workbook.namelist())
            shared_strings = []
            if "xl/sharedStrings.xml" in names:
                shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
                shared_strings = [
                    "".join(node.text or "" for node in item.findall(".//a:t", XML_NS))
                    for item in shared_root.findall("a:si", XML_NS)
                ]
            sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    except (BadZipFile, KeyError, ET.ParseError) as exc:
        raise ValueError(f"CPCB report was not a readable XLSX workbook: {exc}") from exc

    rows = []
    for row in sheet.findall(".//a:sheetData/a:row", XML_NS):
        cells = {_column_index(c.get("r")): _cell_text(c, shared_strings)
                 for c in row.findall("a:c", XML_NS)}
        rows.append(cells)

    header_index = next(
        (i for i, row in enumerate(rows)
         if any(value.strip().casefold() == "from date" for value in row.values())),
        None,
    )
    if header_index is None:
        raise ValueError("CPCB report has no 'From Date' data table")
    headers = {value.strip().upper(): index for index, value in rows[header_index].items()}
    time_col = headers.get("FROM DATE")
    pollutant_cols = [(name, pollutant, unit, headers.get(name))
                      for name, pollutant, unit in REPORT_PARAMETERS]
    if time_col is None or not any(index is not None for _, _, _, index in pollutant_cols):
        raise ValueError("CPCB report is missing its timestamp or pollutant columns")

    start_utc = pd.Timestamp(start_utc) if start_utc is not None else None
    end_utc = pd.Timestamp(end_utc) if end_utc is not None else None
    result = []
    for row in rows[header_index + 1:]:
        local_time = row.get(time_col, "").strip()
        try:
            observed_local = datetime.strptime(local_time, "%d-%m-%Y %H:%M").replace(tzinfo=IST)
        except ValueError:
            continue  # footer or blank row
        observed_at = observed_local.astimezone(timezone.utc)
        if start_utc is not None and observed_at < start_utc.to_pydatetime():
            continue
        if end_utc is not None and observed_at >= end_utc.to_pydatetime():
            continue

        for header, pollutant, unit, col_index in pollutant_cols:
            if col_index is None:
                continue
            raw_value = row.get(col_index, "").strip()
            try:
                value = float(raw_value)
                if not math.isfinite(value):
                    value = None
            except (TypeError, ValueError):
                value = None
            if value is None:
                continue
            result.append({
                "station": station,
                "city": "Delhi",
                "pollutant": pollutant,
                "unit": unit,
                "value_avg": value,
                "value_min": None,
                "value_max": None,
                "observed_at": observed_at,
                "source": SOURCE,
            })
    return pd.DataFrame(result, columns=EXPECTED_COLUMNS)


def _request_station_report(session, station, report_id, report_name, start, end, now):
    payload = {
        "reportDate": now.strftime("%d-%m-%Y"),
        "reportTime": now.strftime("%H:%M:%S"),
        "stationId": report_id,
        "state": "Delhi",
        "city": "Delhi",
        "station": report_name,
        "parameters": REPORT_COLUMNS,
        "columns": f"from_time,to_time,{REPORT_COLUMNS}",
        "avgPeriod": "1H",
        "fromDate": start.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": end.strftime("%Y-%m-%d %H:%M:%S"),
        "callid": "get-real-time-report-excel",
        "templateId": "get-real-time-report-excel",
        "callIdArguments": "stationId,fromDate,toDate,avgPeriod,parameters",
    }
    # GitHub-hosted runners cannot connect directly to CPCB's report host.
    # The project's Cloudflare relay can forward this allow-listed POST route
    # after deploying the updated cpcb_proxy_worker.js.
    url = f"{CPCB_PROXY_URL}{REPORT_PROXY_PATH}" if CPCB_PROXY_URL else REPORT_URL
    response = session.post(
        url,
        json=payload,
        headers={
            "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "Content-Type": "application/json",
        },
        timeout=90,
    )
    response.raise_for_status()
    if not response.content.startswith(b"PK"):
        detail = response.text[:250].replace("\n", " ")
        raise RuntimeError(f"CPCB returned no XLSX report for {station}: {detail}")
    frame = parse_report_workbook(
        response.content,
        station,
        start_utc=start.astimezone(timezone.utc),
        end_utc=end.astimezone(timezone.utc),
    )
    if frame.empty:
        raise RuntimeError(f"CPCB report for {station} contained no numeric pollutant readings")
    return frame


def main():
    started_at = datetime.now(timezone.utc)
    now = datetime.now(IST)
    start = datetime.combine(now.date() - timedelta(days=HISTORY_DAYS), datetime.min.time(), IST)
    end = now.replace(minute=0, second=0, microsecond=0)
    frames = []
    failures = []
    session = requests.Session()

    for station, config in STATIONS.items():
        try:
            frame = _request_station_report(
                session,
                station,
                config["cpcb_report_id"],
                config["cpcb_report_name"],
                start,
                end,
                now,
            )
            print(f"{station}: {len(frame)} valid pollutant-hour rows from CPCB report")
            frames.append(frame)
        except Exception as exc:
            failures.append(f"{station}: {type(exc).__name__}: {exc}")

    engine = get_engine()
    try:
        if frames:
            data = pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=PRIMARY_KEY, keep="last"
            )
            init_schema(engine)
            written = upsert_dataframe(engine, "cpcb_hourly", data, pk_cols=PRIMARY_KEY)
        else:
            written = 0
        finished_at = datetime.now(timezone.utc)
        status = "failed" if failures else "ok"
        message = "; ".join(failures) if failures else f"Fetched rolling {HISTORY_DAYS + 1}-day CPCB report window"
        log_run(engine, "cpcb_recent_report", started_at, finished_at, written, status, message)
    finally:
        engine.dispose()

    if failures:
        raise RuntimeError("CPCB recent-history backfill failed: " + "; ".join(failures))
    print(f"cpcb_hourly: upserted {written} rolling-window CPCB report rows")


if __name__ == "__main__":
    main()
