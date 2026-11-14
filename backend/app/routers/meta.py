"""
AirGuard API health and static metadata endpoints.

Health validates the full published forecast grid, AQI category output, model
artifact manifest and feature schema. It does not claim that external providers
or PostgreSQL are reachable; use /api/ingestion/status for the DB-backed view.
"""
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import APIRouter

from .. import config
from ..data_loader import file_mtime_iso

router = APIRouter(tags=["meta"])
MAX_FORECAST_ORIGIN_AGE_DAYS = 2
IST = ZoneInfo("Asia/Kolkata")


def _read_csv_safe(path):
    if not path.exists():
        return None, f"{path.name} does not exist"
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return None, f"{path.name} could not be read: {exc}"
    if df.empty:
        return None, f"{path.name} is empty"
    return df, None


def _numeric_errors(df, filename):
    errors = []
    for column in ("prediction", "lower", "upper"):
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            errors.append(f"{filename} contains missing or non-finite {column} values")
    if {"lower", "upper"} <= set(df.columns):
        lower = pd.to_numeric(df["lower"], errors="coerce")
        upper = pd.to_numeric(df["upper"], errors="coerce")
        if (lower > upper).any():
            errors.append(f"{filename} has rows where lower > upper")
    return errors


def _parse_horizons(df, filename):
    values = pd.to_numeric(df["horizon"], errors="coerce")
    if values.isna().any() or not np.equal(values, np.floor(values)).all():
        return None, [f"{filename} contains invalid horizon values"]
    return set(values.astype(int).tolist()), []


def _origin_dates(values, filename):
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        return set(), [f"{filename} contains invalid forecast date values"]
    return set(parsed.dt.date.unique()), []


def _check_station_forecast(df):
    errors = []
    filename = config.STATION_FORECAST_CSV.name
    expected_cols = {
        "station", "date", "target", "horizon",
        "forecast_date", "prediction", "lower", "upper",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        return [f"{filename} is missing columns: {sorted(missing)}"], set()

    errors.extend(_numeric_errors(df, filename))
    horizons, horizon_errors = _parse_horizons(df, filename)
    errors.extend(horizon_errors)

    duplicates = int(df.duplicated(subset=["station", "target", "horizon"]).sum())
    if duplicates:
        errors.append(f"{filename} has {duplicates} duplicate (station, target, horizon) rows")

    actual = set()
    if horizons is not None:
        for row in df[["station", "target", "horizon"]].itertuples(index=False, name=None):
            try:
                actual.add((str(row[0]), str(row[1]), int(row[2])))
            except (TypeError, ValueError):
                continue
        expected = {
            (station, target, horizon)
            for station in config.EXPECTED_STATIONS
            for target in config.EXPECTED_TARGETS
            for horizon in config.EXPECTED_HORIZONS
        }
        missing_grid = expected - actual
        extra_grid = actual - expected
        if missing_grid:
            errors.append(f"{filename} is missing {len(missing_grid)} expected station/target/horizon combinations")
        if extra_grid:
            errors.append(f"{filename} contains {len(extra_grid)} unexpected station/target/horizon combinations")

    origins, date_errors = _origin_dates(df["date"], filename)
    errors.extend(date_errors)
    if len(origins) != 1:
        errors.append(f"{filename} must contain exactly one forecast origin date; found {sorted(origins)}")
    forecast_dates, forecast_date_errors = _origin_dates(df["forecast_date"], filename)
    errors.extend(forecast_date_errors)
    if len(forecast_dates) != len(config.EXPECTED_HORIZONS):
        errors.append(f"{filename} must contain 10 distinct forecast dates; found {len(forecast_dates)}")
    return errors, origins


def _check_city_forecast(df, station_origins):
    errors = []
    filename = config.CITY_FORECAST_CSV.name
    expected_cols = {"target", "forecast_date", "horizon", "prediction", "lower", "upper"}
    missing = expected_cols - set(df.columns)
    if missing:
        return [f"{filename} is missing columns: {sorted(missing)}"]

    errors.extend(_numeric_errors(df, filename))
    horizons, horizon_errors = _parse_horizons(df, filename)
    errors.extend(horizon_errors)
    if int(df.duplicated(subset=["target", "horizon"]).sum()):
        errors.append(f"{filename} has duplicate (target, horizon) rows")
    if horizons is not None:
        actual = set()
        for row in df[["target", "horizon"]].itertuples(index=False, name=None):
            try:
                actual.add((str(row[0]), int(row[1])))
            except (TypeError, ValueError):
                continue
        expected = {
            (target, horizon)
            for target in config.EXPECTED_TARGETS
            for horizon in config.EXPECTED_HORIZONS
        }
        if expected - actual:
            errors.append(f"{filename} is missing {len(expected - actual)} target/horizon combinations")
        if actual - expected:
            errors.append(f"{filename} contains {len(actual - expected)} unexpected target/horizon combinations")

    forecast_dates, date_errors = _origin_dates(df["forecast_date"], filename)
    errors.extend(date_errors)
    if len(forecast_dates) != len(config.EXPECTED_HORIZONS):
        errors.append(f"{filename} must contain 10 distinct forecast dates; found {len(forecast_dates)}")

    if station_origins and horizons is not None and not date_errors:
        city_origins = set(
            (pd.to_datetime(df["forecast_date"]) - pd.to_timedelta(
                pd.to_numeric(df["horizon"], errors="coerce"), unit="D"
            )).dt.date
        )
        if city_origins != station_origins:
            errors.append(
                f"{filename} origin date(s) {sorted(city_origins)} do not match "
                f"{config.STATION_FORECAST_CSV.name} origin date(s) {sorted(station_origins)}"
            )
    return errors


def _check_category_forecast(df, city_df):
    errors = []
    filename = config.CITY_AQI_CATEGORY_CSV.name
    expected_cols = {"target", "forecast_date", "horizon", "prediction", "lower", "upper", "category"}
    missing = expected_cols - set(df.columns)
    if missing:
        return [f"{filename} is missing columns: {sorted(missing)}"]
    errors.extend(_numeric_errors(df, filename))
    horizons, horizon_errors = _parse_horizons(df, filename)
    errors.extend(horizon_errors)
    if set(df["target"].astype(str)) != {"AQI"}:
        errors.append(f"{filename} must contain AQI rows only")
    if horizons is not None and horizons != set(config.EXPECTED_HORIZONS):
        errors.append(f"{filename} must contain horizons 1..10")
    if len(df) != len(config.EXPECTED_HORIZONS):
        errors.append(f"{filename} must contain exactly 10 rows")
    if int(df.duplicated(subset=["target", "horizon"]).sum()):
        errors.append(f"{filename} has duplicate target/horizon rows")
    if df["category"].isna().any() or df["category"].astype(str).str.strip().eq("").any():
        errors.append(f"{filename} has missing AQI category labels")

    # Category data is a decorated view of the AQI city forecast; detect a stale
    # or mismatched category file rather than silently showing conflicting values.
    if city_df is not None and {"target", "horizon"} <= set(city_df.columns):
        city_aqi = city_df[city_df["target"].astype(str) == "AQI"]
        if len(city_aqi) == len(config.EXPECTED_HORIZONS):
            try:
                left = city_aqi[["horizon", "forecast_date", "prediction", "lower", "upper"]].copy()
                right = df[["horizon", "forecast_date", "prediction", "lower", "upper"]].copy()
                left["horizon"] = pd.to_numeric(left["horizon"], errors="coerce")
                right["horizon"] = pd.to_numeric(right["horizon"], errors="coerce")
                merged = left.merge(right, on="horizon", suffixes=("_city", "_category"), how="outer", indicator=True)
                if not merged["_merge"].eq("both").all():
                    errors.append(f"{filename} horizon rows do not match the city AQI forecast")
                for col in ("forecast_date", "prediction", "lower", "upper"):
                    a = merged[f"{col}_city"]
                    b = merged[f"{col}_category"]
                    if col == "forecast_date":
                        same = a.astype(str).eq(b.astype(str))
                    else:
                        same = np.isclose(
                            pd.to_numeric(a, errors="coerce"),
                            pd.to_numeric(b, errors="coerce"),
                            rtol=1e-7, atol=1e-8, equal_nan=False,
                        )
                    if not bool(same.all()):
                        errors.append(f"{filename} {col} values differ from the city AQI forecast")
            except Exception as exc:
                errors.append(f"{filename} could not be compared with city AQI output: {exc}")
    return errors


def _check_feature_schema():
    errors = []
    if not config.DAILY_FEATURES_CSV.is_file():
        errors.append(f"{config.DAILY_FEATURES_CSV.relative_to(config.REPO_ROOT)} is missing")
    else:
        try:
            daily = pd.read_csv(config.DAILY_FEATURES_CSV, nrows=2)
            required = {"station", "date"} | {
                info["actual_col"] for info in config.TARGET_INFO.values()
            }
            missing = required - set(daily.columns)
            if missing:
                errors.append(f"daily_features.csv is missing required columns: {sorted(missing)}")
        except Exception as exc:
            errors.append(f"daily_features.csv could not be read: {exc}")
    if not config.FEATURE_LIST_TXT.is_file():
        errors.append(f"{config.FEATURE_LIST_TXT.name} is missing")
    else:
        features = [line.strip() for line in config.FEATURE_LIST_TXT.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        if not features:
            errors.append(f"{config.FEATURE_LIST_TXT.name} is empty")
    if not config.TARGETS_JSON.is_file():
        errors.append(f"{config.TARGETS_JSON.name} is missing")
    else:
        try:
            targets = json.loads(config.TARGETS_JSON.read_text(encoding="utf-8"))
            if not isinstance(targets, dict) or not targets:
                errors.append(f"{config.TARGETS_JSON.name} must contain a non-empty JSON object")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{config.TARGETS_JSON.name} is invalid: {exc}")
    return errors


def _check_model_artifacts():
    errors = []
    manifest_path = config.TRAINED_AT_JSON
    if not manifest_path.is_file():
        return False, [f"{manifest_path.name} is missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"{manifest_path.name} is invalid: {exc}"]

    targets = manifest.get("targets", {}) if isinstance(manifest, dict) else {}
    if not isinstance(targets, dict):
        return False, [f"{manifest_path.name} has an invalid 'targets' section"]

    for target in config.EXPECTED_TARGETS:
        metadata = targets.get(target)
        if not isinstance(metadata, dict):
            errors.append(f"{target}: target metadata is missing from {manifest_path.name}")
            continue
        members = metadata.get("members", [])
        if not isinstance(members, list) or not members:
            errors.append(f"{target}: trained member list is missing or empty")
            continue
        for member in members:
            for horizon in config.EXPECTED_HORIZONS:
                artifact = config.MODELS_DIR / target / f"{member}_h{horizon}.joblib"
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    errors.append(f"{target}: missing model artifact {artifact.relative_to(config.REPO_ROOT)}")
        band_path = config.MODELS_DIR / target / "oof_band.csv"
        band, err = _read_csv_safe(band_path)
        if err:
            errors.append(f"{target}: {err}")
        elif not {"q_lo", "q_hi"} <= set(band.columns):
            errors.append(f"{target}: {band_path.name} must contain q_lo and q_hi")
        else:
            values = band[["q_lo", "q_hi"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if values.shape[0] != len(config.EXPECTED_HORIZONS) or not np.isfinite(values).all():
                errors.append(f"{target}: {band_path.name} must contain 10 finite q_lo/q_hi rows")
    if not (config.MODELS_DIR / "final_ensemble_weights_used.json").is_file():
        errors.append("outputs/models/final_ensemble_weights_used.json is missing")
    else:
        try:
            weights = json.loads((config.MODELS_DIR / "final_ensemble_weights_used.json").read_text(encoding="utf-8"))
            if not isinstance(weights, dict) or not all(t in weights for t in config.EXPECTED_TARGETS):
                errors.append("final_ensemble_weights_used.json does not include all expected targets")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"final_ensemble_weights_used.json is invalid: {exc}")
    return not errors, errors


def _check_parameter_files():
    errors = []
    members = {"RandomForest", "LightGBM", "XGBoost", "Ridge"}
    horizons = {str(value) for value in config.EXPECTED_HORIZONS}
    for target in config.EXPECTED_TARGETS:
        best_path = config.PARAMS_DIR / f"best_params_{target}.json"
        weights_path = config.PARAMS_DIR / f"ensemble_weights_{target}.json"
        try:
            best = json.loads(best_path.read_text(encoding="utf-8"))
            if set(best) != members:
                errors.append(f"{best_path.name} does not contain the four expected model members")
            for member in members:
                if not isinstance(best.get(member), dict) or set(map(str, best[member])) != horizons:
                    errors.append(f"{best_path.name} has incomplete horizon parameters for {member}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{best_path.name} is missing or invalid: {exc}")

        try:
            weights = json.loads(weights_path.read_text(encoding="utf-8"))
            if set(map(str, weights)) != horizons:
                errors.append(f"{weights_path.name} must contain horizons 1..10")
            for horizon in config.EXPECTED_HORIZONS:
                values = weights.get(str(horizon), weights.get(horizon, {}))
                if not isinstance(values, dict) or set(values) != members:
                    errors.append(f"{weights_path.name} has incomplete member weights at horizon {horizon}")
                    continue
                numeric = pd.to_numeric(pd.Series(list(values.values())), errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(numeric).all() or (numeric < 0).any() or abs(float(numeric.sum()) - 1.0) > 1e-6:
                    errors.append(f"{weights_path.name} has invalid weights at horizon {horizon}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{weights_path.name} is missing or invalid: {exc}")
    return not errors, errors


def _forecast_checks():
    station_df, station_err = _read_csv_safe(config.STATION_FORECAST_CSV)
    city_df, city_err = _read_csv_safe(config.CITY_FORECAST_CSV)
    category_df, category_err = _read_csv_safe(config.CITY_AQI_CATEGORY_CSV)
    errors = [err for err in (station_err, city_err, category_err) if err]

    if station_df is None or city_df is None or category_df is None:
        return False, errors, None

    station_errors, origins = _check_station_forecast(station_df)
    errors.extend(station_errors)
    errors.extend(_check_city_forecast(city_df, origins))
    errors.extend(_check_category_forecast(category_df, city_df))
    origin_date = next(iter(origins)) if len(origins) == 1 else None
    return len(errors) == 0, errors, origin_date


@router.get("/health", response_model=None)
def health():
    is_demo = config.DEMO_DATA_MARKER_JSON.exists()
    structurally_ok, forecast_errors, origin_date = _forecast_checks()

    forecast_generated_at = file_mtime_iso(config.CITY_FORECAST_CSV)
    today = datetime.now(IST).date()
    origin_age_days = (today - origin_date).days if origin_date else None
    forecast_fresh = origin_age_days is not None and 0 <= origin_age_days <= MAX_FORECAST_ORIGIN_AGE_DAYS

    feature_errors = _check_feature_schema()
    feature_schema_ready = not feature_errors
    parameter_files_ready, parameter_errors = _check_parameter_files()
    model_artifacts_ready, model_artifact_errors = _check_model_artifacts()
    pipeline_ready = feature_schema_ready and parameter_files_ready and model_artifacts_ready

    if is_demo:
        status = "demo"
    elif not structurally_ok:
        status = "unverified"
    elif not forecast_fresh or not pipeline_ready:
        status = "degraded"
    else:
        status = "ok"

    if is_demo:
        note = "Bundled outputs are synthetic demo data, not production model forecasts."
    elif not structurally_ok:
        note = "Forecast outputs failed structural checks; see forecast_validation_errors."
    elif not forecast_fresh or not pipeline_ready:
        note = "Forecast is stale or the production feature/model artifacts are incomplete."
    else:
        note = "Forecast outputs and local production artifact manifest passed structural checks."

    return {
        "status": status,
        "is_demo_data": is_demo,
        "outputs_present": config.CITY_FORECAST_CSV.exists(),
        "features_present": config.DAILY_FEATURES_CSV.exists(),
        "forecast_generated_at": forecast_generated_at,
        "forecast_origin_date": origin_date.isoformat() if origin_date else None,
        "forecast_origin_age_days": origin_age_days,
        "forecast_structurally_valid": structurally_ok,
        "forecast_validation_errors": forecast_errors,
        "feature_schema_ready": feature_schema_ready,
        "feature_schema_errors": feature_errors,
        "parameter_files_ready": parameter_files_ready,
        "parameter_errors": parameter_errors,
        "model_artifacts_ready": model_artifacts_ready,
        "model_artifact_errors": model_artifact_errors,
        "pipeline_ready": pipeline_ready,
        "note": note,
    }


@router.get("/stations", response_model=None)
def list_stations():
    return [
        {"name": name, "lat": coords["lat"], "lon": coords["lon"]}
        for name, coords in config.STATION_COORDS.items()
    ]


@router.get("/targets", response_model=None)
def list_targets():
    return [
        {"name": name, "label": info["label"], "unit": info["unit"]}
        for name, info in config.TARGET_INFO.items()
    ]
