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
import warnings

import joblib
import pandas as pd

import model_common as mc

warnings.filterwarnings("ignore")


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
    # combined when the ensemble was tuned.
    _, weights = mc.load_tuning(target_name)
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
    if band_path.exists():
        try:
            band = pd.read_csv(band_path, index_col="horizon")
        except Exception as exc:
            print(f"WARNING: could not read {band_path} ({exc}); forecast will ship with a zero-width band.")
            band = pd.DataFrame(columns=["q_lo", "q_hi"]).rename_axis("horizon")
    else:
        print(f"WARNING: no oof_band.csv for {target_name} -- forecast will ship with a zero-width band.")
        band = pd.DataFrame(columns=["q_lo", "q_hi"]).rename_axis("horizon")

    fc = fc.merge(band, left_on="horizon", right_index=True, how="left")
    fc["q_lo"] = fc["q_lo"].fillna(0.0)
    fc["q_hi"] = fc["q_hi"].fillna(0.0)
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
