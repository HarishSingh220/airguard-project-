from datetime import date as Date
from typing import Optional

from pydantic import BaseModel


class Station(BaseModel):
    name: str
    lat: Optional[float] = None
    lon: Optional[float] = None


class Target(BaseModel):
    name: str
    label: str
    unit: str


class ForecastPoint(BaseModel):
    forecast_date: Date
    horizon: int
    prediction: float
    lower: float
    upper: float


class CityForecastPoint(ForecastPoint):
    target: str


class StationForecastPoint(ForecastPoint):
    target: str
    station: str
    origin_date: Date


class AqiCategoryPoint(ForecastPoint):
    category: Optional[str] = None
    category_color: Optional[str] = None


class LatestReading(BaseModel):
    station: str
    target: str
    forecast_date: Date
    prediction: float
    lower: float
    upper: float


class HistoryPoint(BaseModel):
    date: Date
    value: Optional[float] = None


class ModelWeights(BaseModel):
    target: str
    weights_by_horizon: dict


class ModelInfo(BaseModel):
    trained_at: Optional[str] = None
    targets: dict = {}


class IngestionLogRow(BaseModel):
    job: str
    started_at: str
    finished_at: Optional[str] = None
    rows: Optional[int] = None
    status: str
    message: Optional[str] = None


class HealthStatus(BaseModel):
    status: str  # "demo" | "unverified" | "degraded" | "ok"
    is_demo_data: bool
    outputs_present: bool
    features_present: bool
    forecast_generated_at: Optional[str] = None
    forecast_origin_date: Optional[str] = None
    forecast_origin_age_days: Optional[int] = None
    forecast_structurally_valid: bool
    forecast_validation_errors: list[str] = []
    feature_schema_ready: bool
    model_artifacts_ready: bool
    note: str
