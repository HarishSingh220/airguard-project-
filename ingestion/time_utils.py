"""
time_utils.py
Shared helper for Open-Meteo's naive, IST-local timestamp grids.

fetch_openmeteo.py, backfill_openmeteo_12h_grid.py and
backfill_openmeteo_airquality.py each request Open-Meteo data with
timezone=Asia/Kolkata and previously carried their own identical copy of
this conversion. Centralised here so the three can't drift apart.
"""
import pandas as pd

IST = "Asia/Kolkata"


def parse_ist_grid(time_strings):
    """Open-Meteo's 'time' values are naive local-time labels when timezone
    is requested explicitly (no UTC offset in the string). Localize as IST,
    then convert to UTC so this matches every other table's instant convention.
    """
    naive = pd.to_datetime(time_strings)
    return naive.tz_localize(IST, ambiguous="infer", nonexistent="shift_forward").tz_convert("UTC")
