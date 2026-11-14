"""
config.py
Paths and small fixed lookups the API reads from. Deliberately does NOT import
model_common.py or build_features.py: those modules eagerly load
features/daily_features.csv and require xgboost/lightgbm/sklearn just to
import, which would make the API fail to start on a machine that only has
outputs/ (the committed forecast CSVs) and not the full modeling environment.
The API only ever reads already-generated files -- it never fits or predicts
anything itself.
"""
import os
from pathlib import Path

# Load local repo-root .env before reading settings. Hosted platforms should
# inject environment variables directly; override=False keeps them authoritative.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(DEFAULT_REPO_ROOT / ".env", override=False)
except ImportError:
    pass

# backend/app/config.py -> repo root is two levels up
REPO_ROOT = Path(os.environ.get("AIRGUARD_REPO_ROOT", DEFAULT_REPO_ROOT)).resolve()

OUTPUTS_DIR = REPO_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FEATURES_DIR = REPO_ROOT / "features"
PARAMS_DIR = REPO_ROOT / "params"

STATION_FORECAST_CSV = OUTPUTS_DIR / "forecast_10d_station_final.csv"
CITY_FORECAST_CSV = OUTPUTS_DIR / "forecast_10d_city_final.csv"
CITY_AQI_CATEGORY_CSV = OUTPUTS_DIR / "forecast_10d_city_aqi_with_category.csv"
ENSEMBLE_WEIGHTS_USED_JSON = OUTPUTS_DIR / "final_ensemble_weights_used.json"
TRAINED_AT_JSON = MODELS_DIR / "trained_at.json"
DAILY_FEATURES_CSV = FEATURES_DIR / "daily_features.csv"

# Written by backend/dev/generate_sample_data.py whenever it (re)writes synthetic
# demo outputs, and deleted by predict_forecast.py only after a real forecast run
# has finished writing every output file. Its presence means "everything currently
# under outputs/ and features/daily_features.csv is fabricated demo data", not a
# real model run -- see /api/health in routers/meta.py.
DEMO_DATA_MARKER_JSON = OUTPUTS_DIR / "demo_data_marker.json"

# Artifacts a real `predict_forecast.py` run requires that this lightweight API
# never generates itself -- used by /api/health to report whether the real
# pipeline *could* run here, separately from whether the checked-in forecast
# CSVs look complete.
FEATURE_LIST_TXT = FEATURES_DIR / "feature_list.txt"
TARGETS_JSON = FEATURES_DIR / "targets.json"

# Mirrors ingestion/config.py's STATIONS -- kept here too so the API never has
# to import the ingestion package (and its db/cdsapi/requests dependencies)
# just to answer "what stations exist and where are they".
STATION_COORDS = {
    "Anand Vihar": {"lat": 28.6469, "lon": 77.3158},
    "Dwarka Sector 8": {"lat": 28.5710, "lon": 77.0710},
    "IMD Lodhi Road": {"lat": 28.5918, "lon": 77.2273},
    "ITO": {"lat": 28.6316, "lon": 77.2410},
    "Punjabi Bagh": {"lat": 28.6740, "lon": 77.1310},
}

# target name -> (same-day actual column in daily_features.csv, display unit)
TARGET_INFO = {
    "AQI": {"actual_col": "d_AQI", "unit": "index", "label": "Air Quality Index"},
    "PM25": {"actual_col": "d_PM25", "unit": "ug/m3", "label": "PM2.5"},
    "O3": {"actual_col": "d_O3", "unit": "ug/m3", "label": "Ozone (daily mean)"},
    "O3_8H": {"actual_col": "d_O3_max8h", "unit": "ug/m3", "label": "Ozone (8h max)"},
    "NOx": {"actual_col": "d_NOx", "unit": "ug/m3", "label": "NOx"},
}

CPCB_CATEGORY_COLORS = {
    "Good": "#4CAF7D",
    "Satisfactory": "#A8C84C",
    "Moderate": "#F2C94C",
    "Poor": "#F2994A",
    "Very poor": "#EB5757",
    "Severe": "#8B2E3C",
}

EXPECTED_STATIONS = list(STATION_COORDS.keys())
EXPECTED_TARGETS = list(TARGET_INFO.keys())
EXPECTED_HORIZONS = list(range(1, 11))

CORS_ALLOW_ORIGINS = [
    o.strip() for o in os.environ.get("AIRGUARD_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()
]
