"""
predict_forecast_mlflow.py
Deployment-via-registry variant of predict_forecast.py.

predict_forecast.py loads the 40 joblib files per target straight off disk
(outputs/models/<target>/<member>_h<h>.joblib) and blends them by hand. This
script instead loads ONE MLflow model per target -- whatever version currently
holds the "production" alias in the Model Registry (set by
train_production_models.py's promotion step) -- and calls .predict() on it.

That's the point of wrapping the ensemble in mlflow_utils.AirGuardTargetModel:
any process that can reach the MLflow tracking/registry server (this script,
a notebook, a separate inference container, `mlflow models serve`, ...) can
ask for "models:/airguard-aqi@production" and get a working forecaster back,
with no knowledge of joblib paths, sklearn pipelines, or ensemble weights.

Run from the repository root, after build_features.py and at least one
successful train_production_models.py run (which is what actually registers
the "production" version in MLflow -- this script only reads from the
registry, same "no fitting here" spirit as predict_forecast.py):
    python predict_forecast_mlflow.py

Outputs (written to outputs/):
    forecast_10d_station_mlflow.csv
    forecast_10d_city_mlflow.csv

See MLFLOW_GUIDE.md for how to also serve a single target as a REST API with
`mlflow models serve`.
"""
import warnings

import mlflow
import pandas as pd

import model_common as mc
import mlflow_utils as mu

warnings.filterwarnings("ignore")


def forecast_target_mlflow(target_name):
    model_uri = f"models:/{mu.registered_model_name(target_name)}@{mu.ALIAS_PRODUCTION}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {model_uri} -- has train_production_models.py been run at least "
            f"once (so it could register and promote a 'production' version for {target_name!r})? "
            f"Original error: {exc}"
        ) from exc

    actual_col = mc.TARGETS[target_name]["actual"]
    lower, upper = mc.TARGET_BOUNDS[target_name]
    as_of = mc.latest_complete_date(actual_col)
    latest = mc.daily[mc.daily["date"] == as_of]
    if latest.empty:
        raise RuntimeError(f"No rows in daily_features.csv for the forecast origin date {as_of.date()}.")

    x_parts, meta_parts = [], []
    for h in mc.HORIZONS:
        x_h = mc.make_X(h).loc[latest.index][mc.FEATURE_COLS + ["station"]].copy()
        x_h["horizon"] = h
        x_parts.append(x_h)

        meta = latest[["station", "date"]].copy()
        meta["horizon"] = h
        meta["forecast_date"] = meta["date"] + pd.Timedelta(days=h)
        meta_parts.append(meta)

    X = pd.concat(x_parts, ignore_index=True)
    meta_df = pd.concat(meta_parts, ignore_index=True)

    result = model.predict(X)  # DataFrame: input columns + "prediction", same row order as X
    fc = meta_df.copy()
    fc["prediction"] = result["prediction"].to_numpy().clip(lower, upper)
    fc["target"] = target_name

    city = (fc.groupby(["target", "forecast_date", "horizon"], as_index=False)["prediction"]
              .mean().sort_values("forecast_date"))
    return fc, city


def main():
    mu.init_mlflow()
    all_station, all_city = [], []
    for target_name in mc.TARGETS:
        print(f"\n=== {target_name} (served from models:/{mu.registered_model_name(target_name)}@{mu.ALIAS_PRODUCTION}) ===")
        fc, city = forecast_target_mlflow(target_name)
        all_station.append(fc)
        all_city.append(city)
        print(f"  {len(fc)} predictions ({fc['station'].nunique()} stations x {fc['horizon'].nunique()} horizons)")

    station_df = pd.concat(all_station, ignore_index=True)
    city_df = pd.concat(all_city, ignore_index=True)

    station_path = mc.OUT_DIR / "forecast_10d_station_mlflow.csv"
    city_path = mc.OUT_DIR / "forecast_10d_city_mlflow.csv"
    station_df.to_csv(station_path, index=False)
    city_df.to_csv(city_path, index=False)

    print("\nDelhi-wide forecast (served from the MLflow Model Registry):")
    print(city_df.pivot(index="forecast_date", columns="target", values="prediction").round(1))
    print("\nWrote:")
    print(" ", station_path)
    print(" ", city_path)


if __name__ == "__main__":
    main()
