from fastapi import APIRouter, HTTPException, Path, Query

from .. import config
from ..data_loader import df_to_records, read_csv

router = APIRouter(prefix="/history", tags=["history"])


def _validate_target(target: str):
    if target not in config.TARGET_INFO:
        raise HTTPException(status_code=404, detail=f"Unknown target {target!r}. "
                                                      f"Valid targets: {list(config.TARGET_INFO)}")


def _load_daily():
    return read_csv(config.DAILY_FEATURES_CSV, "rebuild_forecast.yml (build_features.py)",
                    parse_dates=["date"])


@router.get("/station/{station}", response_model=None)
def station_history(
    station: str = Path(..., description="Exact station name, e.g. 'ITO'"),
    target: str = Query("AQI", description="AQI, PM25, O3, O3_8H or NOx"),
    days: int = Query(30, ge=1, le=365, description="How many most-recent days to return"),
):
    """Actual (already-observed) daily values for one station -- for drawing the
    recent-history tail alongside the 10-day forecast."""
    if station not in config.STATION_COORDS:
        raise HTTPException(status_code=404, detail=f"Unknown station {station!r}. "
                                                      f"Valid stations: {list(config.STATION_COORDS)}")
    _validate_target(target)
    col = config.TARGET_INFO[target]["actual_col"]
    daily = _load_daily()
    if col not in daily.columns:
        raise HTTPException(status_code=500, detail=f"'{col}' is missing from daily_features.csv.")
    d = daily[daily["station"] == station][["date", col]].rename(columns={col: "value"})
    d = d.sort_values("date").tail(days)
    return df_to_records(d)


@router.get("/city", response_model=None)
def city_history(
    target: str = Query("AQI", description="AQI, PM25, O3, O3_8H or NOx"),
    days: int = Query(30, ge=1, le=365, description="How many most-recent days to return"),
):
    """Delhi-wide (mean across stations) actual daily values."""
    _validate_target(target)
    col = config.TARGET_INFO[target]["actual_col"]
    daily = _load_daily()
    if col not in daily.columns:
        raise HTTPException(status_code=500, detail=f"'{col}' is missing from daily_features.csv.")
    d = daily.groupby("date", as_index=False)[col].mean().rename(columns={col: "value"})
    d = d.sort_values("date").tail(days)
    return df_to_records(d)
