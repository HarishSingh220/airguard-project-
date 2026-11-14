from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query

from .. import config
from ..data_loader import df_to_records, read_csv

router = APIRouter(prefix="/forecast", tags=["forecast"])


def _validate_target(target: Optional[str]):
    if target is not None and target not in config.TARGET_INFO:
        raise HTTPException(status_code=404, detail=f"Unknown target {target!r}. "
                                                      f"Valid targets: {list(config.TARGET_INFO)}")


def _validate_station(station: str):
    if station not in config.STATION_COORDS:
        raise HTTPException(status_code=404, detail=f"Unknown station {station!r}. "
                                                      f"Valid stations: {list(config.STATION_COORDS)}")


@router.get("/city", response_model=None)
def city_forecast(target: Optional[str] = Query(None, description="AQI, PM25, O3, O3_8H or NOx; omit for all")):
    """10-day Delhi-wide forecast (mean across stations), one or all targets."""
    _validate_target(target)
    df = read_csv(config.CITY_FORECAST_CSV, "rebuild_forecast.yml (predict_forecast.py)",
                  parse_dates=["forecast_date"])
    if target:
        df = df[df["target"] == target]
    df = df.sort_values(["target", "horizon"])
    return df_to_records(df)


@router.get("/city/aqi-category", response_model=None)
def city_aqi_category():
    """City-wide AQI forecast with CPCB category (Good..Severe) and a display color."""
    df = read_csv(config.CITY_AQI_CATEGORY_CSV, "rebuild_forecast.yml (predict_forecast.py)",
                  parse_dates=["forecast_date"])
    df = df.sort_values("horizon")
    records = df_to_records(df)
    for r in records:
        r["category_color"] = config.CPCB_CATEGORY_COLORS.get(r.get("category"))
    return records


@router.get("/station/{station}", response_model=None)
def station_forecast(
    station: str = Path(..., description="Exact station name, e.g. 'ITO'"),
    target: Optional[str] = Query(None, description="AQI, PM25, O3, O3_8H or NOx; omit for all"),
):
    """10-day forecast for a single station, one or all targets."""
    _validate_station(station)
    _validate_target(target)
    df = read_csv(config.STATION_FORECAST_CSV, "rebuild_forecast.yml (predict_forecast.py)",
                  parse_dates=["forecast_date", "date"])
    df = df.rename(columns={"date": "origin_date"})
    df = df[df["station"] == station]
    if target:
        df = df[df["target"] == target]
    df = df.sort_values(["target", "horizon"])
    return df_to_records(df)


@router.get("/stations/latest", response_model=None)
def latest_per_station(target: Optional[str] = Query(None)):
    """Horizon-1 (tomorrow's) prediction for every station -- a one-call overview,
    handy for a map view or a summary table."""
    _validate_target(target)
    df = read_csv(config.STATION_FORECAST_CSV, "rebuild_forecast.yml (predict_forecast.py)",
                  parse_dates=["forecast_date", "date"])
    df = df[df["horizon"] == 1]
    if target:
        df = df[df["target"] == target]
    df = df.sort_values(["station", "target"])
    return df_to_records(df[["station", "target", "forecast_date", "prediction", "lower", "upper"]])


@router.get("/all", response_model=None)
def full_station_forecast():
    """The complete per-station, per-target, per-horizon forecast table, unfiltered.
    Large-ish (5 stations x 5 targets x 10 horizons); prefer the narrower
    endpoints above unless you genuinely need everything in one call."""
    df = read_csv(config.STATION_FORECAST_CSV, "rebuild_forecast.yml (predict_forecast.py)",
                  parse_dates=["forecast_date", "date"])
    return df_to_records(df.rename(columns={"date": "origin_date"}))
