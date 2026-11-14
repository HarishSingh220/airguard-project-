"""
predict_forecast.py
The daily step of the walk-forward deployment (run by
.github/workflows/rebuild_forecast.yml). Loads the production models that
train_production_models.py already fit -- on its own ~30-day schedule -- and
scores them against TODAY's feature row from features/daily_features.csv.

No model fitting happens in this file. That's what makes it cheap enough to
run daily while the actual refit only happens every ~30 days: this script
just unpickles saved sklearn/xgboost/lightgbm pipelines and calls .predict().

Requires outputs/models/<target>/<model>_h<h>.joblib and
outputs/models/<target>/oof_band.csv to already exist. If they don't -- e.g.
this is the very first run on a fresh repo -- raises a clear error telling you
to run train_production_models.py (or trigger retrain_model.yml with
force=true) first.

Run from the repository root, after build_features.py:
    python predict_forecast.py

Outputs (written to outputs/):
    forecast_10d_station_final.csv    per station, per target, per horizon
    forecast_10d_city_final.csv       Delhi-wide average, per target, per horizon
    forecast_10d_city_aqi_with_category.csv
    final_ensemble_weights_used.json  copy of the weights the loaded models were trained with
"""
import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd

import model_common as mc

warnings.filterwarnings("ignore")

# Written by train_production_models.py at the end of the retrain that produced
# the .joblib pipelines this script loads -- the exact weights that version of
# each model was blended with. Reading params/ensemble_weights_*.json instead
# (as this script used to) risks blending saved pipelines with weights that
# have since been re-tuned for a *different*, not-yet-trained model version.
TRAINED_WEIGHTS_SNAPSHOT = mc.MODEL_DIR / "final_ensemble_weights_used.json"


def load_trained_weights(target_name):
    if not TRAINED_WEIGHTS_SNAPSHOT.exists():
        raise FileNotFoundError(
            f"{TRAINED_WEIGHTS_SNAPSHOT} not found -- no training-time ensemble-weight snapshot "
            f"exists yet. Run train_production_models.py (it writes this file alongside the "
            f".joblib pipelines) before predict_forecast.py."
        )
    try:
        all_weights = json.loads(TRAINED_WEIGHTS_SNAPSHOT.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{TRAINED_WEIGHTS_SNAPSHOT} is not valid JSON: {exc}") from exc
    if target_name not in all_weights:
        raise KeyError(f"{TRAINED_WEIGHTS_SNAPSHOT} has no entry for target {target_name!r}.")
    weights = all_weights[target_name]

    for h in mc.HORIZONS:
        by_h = weights.get(str(h))
        if by_h is None:
            raise ValueError(
                f"{TRAINED_WEIGHTS_SNAPSHOT}[{target_name!r}] is missing weights for horizon {h}."
            )
        for member, w in by_h.items():
            if not math.isfinite(w) or w < 0:
                raise ValueError(
                    f"{TRAINED_WEIGHTS_SNAPSHOT}[{target_name!r}][{h}][{member!r}] = {w!r} is not a "
                    f"finite, non-negative weight."
                )
        total = sum(by_h.values())
        if total <= 0:
            raise ValueError(
                f"{TRAINED_WEIGHTS_SNAPSHOT}[{target_name!r}][{h}] has no positive weight across "
                f"any member -- nothing to blend."
            )
        # Defensive normalization: the snapshot should already sum to ~1, but
        # re-normalize rather than trust that on faith.
        weights[str(h)] = {m: w / total for m, w in by_h.items()}

    return weights


def load_pipeline(target_name, member, h):
    path = mc.MODEL_DIR / target_name / f"{member}_h{h}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- no production model has been trained yet for "
            f"{target_name}/{member} at horizon {h}. Run train_production_models.py locally, "
            f"or trigger the 'Retrain production models' GitHub Actions workflow with "
            f"force=true, before running predict_forecast.py."
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model file {path} -- it may be corrupted, truncated, or saved "
            f"with an incompatible library version. Re-run train_production_models.py to "
            f"regenerate it. Original error: {exc}"
        ) from exc


def forecast_target(target_name):
    # params aren't needed here -- they're already baked into the saved pipelines.
    # weights are needed to combine the members' predictions the same way they were
    # combined when THIS version of the models was trained -- read from the
    # training-time snapshot, not from params/ensemble_weights_*.json, which may
    # have been re-tuned since these particular pipelines were fit.
    weights = load_trained_weights(target_name)
    actual_col = mc.TARGETS[target_name]["actual"]
    lower, upper = mc.TARGET_BOUNDS[target_name]
    members = sorted({m for h in weights for m, v in weights[h].items() if v > 0})
    if not members:
        raise RuntimeError(
            f"No ensemble members have nonzero weight for {target_name!r} -- check "
            f"ensemble_weights_{target_name}.json."
        )

    parts, base = {}, None
    for name in members:
        pipes = {h: load_pipeline(target_name, name, h) for h in mc.HORIZONS}
        fc_m = mc.predict_latest(pipes, actual_col=actual_col)
        base = fc_m[["station", "date", "horizon", "forecast_date"]] if base is None else base
        parts[name] = fc_m.set_index(["station", "horizon"])["prediction"]

    fc = base.copy()
    fc["prediction"] = [
        sum(weights[str(h)].get(m, 0.0) * parts[m].loc[(st, h)] for m in members)
        for st, h in zip(fc["station"], fc["horizon"])
    ]

    band_path = mc.MODEL_DIR / target_name / "oof_band.csv"
    if not band_path.exists():
        raise FileNotFoundError(
            f"{band_path} not found -- no uncertainty band has been trained for {target_name!r}. "
            f"A forecast without this would silently ship zero-width intervals instead of failing. "
            f"Run train_production_models.py (or trigger retrain_model.yml with force=true) first."
        )
    try:
        band = pd.read_csv(band_path, index_col="horizon")
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {band_path} ({exc}) -- it may be corrupted or truncated. "
            f"Re-run train_production_models.py to regenerate it."
        ) from exc

    missing_h = [h for h in mc.HORIZONS if h not in band.index]
    if missing_h:
        raise ValueError(
            f"{band_path} is missing horizon(s) {missing_h} for {target_name!r} -- "
            f"train_production_models.py should not have published an incomplete band."
        )
    if not {"q_lo", "q_hi"} <= set(band.columns):
        raise ValueError(f"{band_path} is missing the q_lo/q_hi columns.")
    band_h = band.loc[mc.HORIZONS, ["q_lo", "q_hi"]]
    if not np.isfinite(band_h.to_numpy(dtype=float)).all():
        raise ValueError(
            f"{band_path} has non-finite q_lo/q_hi value(s) for {target_name!r} at horizons "
            f"{mc.HORIZONS} -- a forecast would otherwise ship with an undefined or zero-width "
            f"uncertainty interval. Re-run train_production_models.py."
        )

    fc = fc.merge(band, left_on="horizon", right_index=True, how="left")
    fc["lower"] = (fc["prediction"] + fc["q_lo"]).clip(lower=lower, upper=upper)
    fc["upper"] = (fc["prediction"] + fc["q_hi"]).clip(lower=lower, upper=upper)
    fc["prediction"] = fc["prediction"].clip(lower=lower, upper=upper)
    fc["target"] = target_name
    fc = fc.drop(columns=["q_lo", "q_hi"])

    city = (fc.groupby(["target", "forecast_date", "horizon"], as_index=False)[["prediction", "lower", "upper"]]
              .mean().sort_values("forecast_date"))
    return fc, city, weights


def main():
    all_station, all_city, weights_used = [], [], {}
    for target_name in mc.TARGETS:
        print(f"\n=== {target_name} ===")
        fc, city, weights = forecast_target(target_name)
        all_station.append(fc)
        all_city.append(city)
        weights_used[target_name] = weights

    station_df = pd.concat(all_station, ignore_index=True)
    city_df = pd.concat(all_city, ignore_index=True)

    aqi_city = city_df[city_df["target"] == "AQI"].copy()
    aqi_city["category"] = pd.cut(aqi_city["prediction"], mc.CPCB_BINS, labels=mc.CPCB_LABELS, include_lowest=True)

    try:
        station_df.to_csv(mc.OUT_DIR / "forecast_10d_station_final.csv", index=False)
        city_df.to_csv(mc.OUT_DIR / "forecast_10d_city_final.csv", index=False)
        aqi_city.to_csv(mc.OUT_DIR / "forecast_10d_city_aqi_with_category.csv", index=False)
        (mc.OUT_DIR / "final_ensemble_weights_used.json").write_text(json.dumps(weights_used, indent=1))
    except OSError as exc:
        raise OSError(f"Could not write forecast outputs to {mc.OUT_DIR}: {exc}") from exc

    # Only now, after every real forecast output has been written successfully,
    # remove backend/dev/generate_sample_data.py's synthetic-data marker (if
    # present) -- /api/health treats its presence as "these are fabricated
    # numbers", so it must not be deleted a moment before the real files that
    # replace it are actually on disk.
    demo_marker = mc.OUT_DIR / "demo_data_marker.json"
    if demo_marker.exists():
        demo_marker.unlink()
        print(f"\nRemoved {demo_marker} -- outputs/ now holds a real forecast, not demo data.")

    print("\nDelhi-wide forecast:")
    print(city_df.pivot(index="forecast_date", columns="target", values="prediction").round(1))
    print("\nWrote:")
    for f in ["forecast_10d_station_final.csv", "forecast_10d_city_final.csv",
              "forecast_10d_city_aqi_with_category.csv", "final_ensemble_weights_used.json"]:
        print(" ", mc.OUT_DIR / f)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: predict_forecast.py failed: {exc}")
        raise
