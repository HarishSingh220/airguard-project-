"""
train_production_models.py
The ~30-day retrain step of the walk-forward deployment (run by
.github/workflows/retrain_model.yml, gated by ingestion/check_retrain_due.py).

For each of the five targets, refits the ensemble members that have nonzero
weight (LightGBM, XGBoost, RandomForest, Ridge) on every label known as of
the latest complete date in features/daily_features.csv, and refreshes the
out-of-fold residual band used for the forecast's uncertainty interval.

Fixed, not tuned: best_params_<target>.json and ensemble_weights_<target>.json
are loaded as-is and never written to here -- the SAME hyperparameters and the
SAME ensemble weights are used on every single retrain. Only the training data
window changes (it always extends to "today"). Re-tuning those files is a
separate, explicitly offline report-only operation -- see tune_parameters_offline.py --
this script only ever calls .fit(), never a hyperparameter search or parameter update.

Run from the repository root, after build_features.py:
    python train_production_models.py

Outputs (written to outputs/models/):
    <target>/<model>_h<h>.joblib      every fitted production pipeline
    <target>/oof_band.csv             the (q_lo, q_hi) residual band per horizon
    final_ensemble_weights_used.json  the (unchanged) weights applied this retrain, for the record
    trained_at.json                   when this retrain ran and which date it trained up to

predict_forecast.py (run daily) only ever reads what this script writes; it
never calls .fit() itself.

MLflow
------
This script also logs every retrain to MLflow (see mlflow_utils.py for the
full explanation): one parent run for the whole retrain, one nested child
run per target with its params/metrics/artifacts, and one new Model Registry
version per target, auto-promoted to the "production" alias when it's at
least as accurate (avg MAE, out-of-fold) as whatever currently holds that
alias. None of this changes the joblib files or outputs/ this script already
wrote -- predict_forecast.py keeps working exactly as before. To also serve
from the registry instead of the local joblib files, see
predict_forecast_mlflow.py.
"""
import json
import os
import sys
import warnings
from importlib.metadata import version as package_version
from datetime import datetime, timezone
from pathlib import Path

# Read developer .env before mlflow_utils resolves tracking/artifact settings.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
except ImportError:
    pass

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

import model_common as mc
import mlflow_utils as mu

warnings.filterwarnings("ignore")

# ingestion/db.py and config.py live in ingestion/, not at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "ingestion"))


def log_completion(started_at, status, message=""):
    """Record retraining in the ingestion_log table used by the 30-day gate.

    Local runs warn and continue if the optional database is unavailable. On
    GitHub Actions, a successful train must persist its completion marker; if
    that write fails, fail the workflow so the next daily due-check cannot
    silently launch another expensive retraining run.
    """
    try:
        import db  # noqa: E402  (import after sys.path edit, by necessity)
        engine = db.get_engine()
        if engine is not None:
            db.init_schema(engine)
            db.log_run(
                engine,
                "retrain_models",
                started_at,
                datetime.now(timezone.utc),
                None,
                status,
                message[:2000],
                raise_on_error=(
                    os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true"
                    and status == "ok"
                ),
            )
    except Exception as exc:
        print(f"WARNING: could not log retrain completion to the database ({exc}). "
              f"check_retrain_due.py will not see this run.")
        if os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true" and status == "ok":
            raise RuntimeError(
                "Models were trained, but the successful retrain marker could not be saved "
                "to ingestion_log. The workflow is stopped so the next due-check does not "
                "silently retrain again tomorrow."
            ) from exc


def main():
    started_at = datetime.now(timezone.utc)
    weights_used = {}
    metadata = {
        "trained_at": started_at.isoformat(),
        "model_runtime_versions": {
            package: package_version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "scipy", "joblib")
        },
        "targets": {},
    }

    # GitHub runners are ephemeral: production retraining must use a durable
    # metadata backend (PostgreSQL/remote tracking server) and S3-compatible
    # artifact storage. Local runs may still use SQLite + local artifacts.
    experiment_id = mu.init_mlflow()
    client = MlflowClient(tracking_uri=mu.TRACKING_URI, registry_uri=mu.REGISTRY_URI)

    try:
        with mlflow.start_run(run_name=f"retrain-{started_at:%Y-%m-%d}"):
            mlflow.set_tag("pipeline", "train_production_models.py")
            mlflow.set_tag("targets", ",".join(mc.TARGETS))
            mlflow.set_tag("parameter_policy", "fixed-json-no-auto-tuning")
            mlflow.set_tag("experiment_id", str(experiment_id))
            mlflow.log_dict(
                metadata["model_runtime_versions"],
                "metadata/model_runtime_versions.json",
            )

            for target_name in mc.TARGETS:
                print(f"\n=== Training {target_name} ===")
                params, weights = mc.load_tuning(target_name)
                prefix = mc.TARGETS[target_name]["prefix"]
                actual_col = mc.TARGETS[target_name]["actual"]
                members = sorted({m for h in weights for m, v in weights[h].items() if v > 0})
                if not members:
                    raise RuntimeError(
                        f"No ensemble members have nonzero weight for {target_name!r} -- check "
                        f"ensemble_weights_{target_name}.json."
                    )

                as_of = mc.latest_complete_date(actual_col)
                try:
                    (mc.MODEL_DIR / target_name).mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise OSError(f"Could not create model output directory {mc.MODEL_DIR / target_name}: {exc}") from exc

                # Nested run = one target of this retrain. Children of the parent run
                # created above, so the MLflow UI groups "this whole retrain" together
                # while still letting you compare AQI-vs-PM25, or this month's AQI run
                # vs last month's.
                with mlflow.start_run(run_name=target_name, nested=True):
                    mlflow.log_params({
                        "target": target_name,
                        "members": ",".join(members),
                        "n_horizons": len(mc.HORIZONS),
                        "trained_up_to": str(as_of.date()),
                        "actual_col": actual_col,
                    })
                    # Visibility for whether this retrain is happening shortly after a station-data
                    # outage (see build_features.py's station_data_missing_frac_30d) -- a high value
                    # here explains a worse-than-usual avg_mae without having to go dig through logs.
                    gap_col = "station_data_missing_frac_30d"
                    if gap_col in mc.daily.columns:
                        origin_rows = mc.daily.loc[mc.daily["date"] == as_of, gap_col]
                        if not origin_rows.empty:
                            mlflow.log_metric(f"{gap_col}_at_origin", float(origin_rows.max()))

                    pipelines_by_horizon = {}  # h -> {member_name: fitted Pipeline}, feeds the registry model below
                    for name in members:
                        pipes = mc.fit_production(mc.tuned_estimator(params, name), mc.MODEL_CONFIGS[name][1],
                                                  prefix, actual_col=actual_col)
                        for h, pipe in pipes.items():
                            out_path = mc.MODEL_DIR / target_name / f"{name}_h{h}.joblib"
                            try:
                                joblib.dump(pipe, out_path)
                            except OSError as exc:
                                raise OSError(f"Could not save fitted model to {out_path}: {exc}") from exc
                            pipelines_by_horizon.setdefault(h, {})[name] = pipe
                        print(f"  [{target_name}] fitted {name} ({len(pipes)} horizons)")

                    # oof_band.csv keeps its original two-column shape -- predict_forecast.py
                    # merges this straight into the forecast it ships, so its schema doesn't change.
                    #
                    # predict_forecast.py now requires a complete, finite band for every horizon
                    # (it raises rather than falling back to a zero-width interval), so refuse to
                    # publish this target's models at all if that requirement can't be met here --
                    # better to keep serving yesterday's forecast/models than to publish a model
                    # set predict_forecast.py can't actually use.
                    band = mc.oof_residuals(target_name, params, weights)
                    missing_h = [h for h in mc.HORIZONS if h not in band.index]
                    if missing_h or band.empty:
                        raise RuntimeError(
                            f"[{target_name}] oof_residuals() produced no usable band for horizon(s) "
                            f"{missing_h or mc.HORIZONS} -- refusing to publish this retrain's models. "
                            f"Check that enough history exists before WF_START for every validation fold."
                        )
                    band_vals = band.loc[mc.HORIZONS, ["q_lo", "q_hi"]].to_numpy(dtype=float)
                    if not np.isfinite(band_vals).all():
                        raise RuntimeError(
                            f"[{target_name}] oof_residuals() produced non-finite q_lo/q_hi value(s) -- "
                            f"refusing to publish this retrain's models."
                        )
                    band[["q_lo", "q_hi"]].to_csv(mc.MODEL_DIR / target_name / "oof_band.csv")

                    # --- Experiment tracking + evaluation -----------------------------------
                    # Per-horizon MAE/RMSE as step metrics -> the MLflow UI plots these as a
                    # curve across the 10-day horizon. avg_mae/avg_rmse let you compare targets,
                    # and the same target across retrains, as single numbers.
                    avg_mae = avg_rmse = None
                    if not band.empty:
                        for h, row in band.iterrows():
                            mlflow.log_metric("mae", row["mae"], step=int(h))
                            mlflow.log_metric("rmse", row["rmse"], step=int(h))
                        avg_mae = float(band["mae"].mean())
                        avg_rmse = float(band["rmse"].mean())
                        mlflow.log_metric("avg_mae", avg_mae)
                        mlflow.log_metric("avg_rmse", avg_rmse)
                    else:
                        print(f"  WARNING: [{target_name}] oof_residuals() returned no rows -- "
                              f"no evaluation metrics logged for this run.")

                    # --- Saving artifacts ----------------------------------------------------
                    # The exact tuning inputs and the full (mae/rmse-included) band, so a run
                    # fully explains itself later without cross-referencing params/*.json by hand.
                    mlflow.log_dict(params, "tuning/best_params.json")
                    mlflow.log_dict(weights, "tuning/ensemble_weights.json")
                    mlflow.log_dict({"target": target_name, "trained_up_to": str(as_of.date()),
                                     "feature_count": len(mc.FEATURE_COLS), "members": members,
                                     "horizons": mc.HORIZONS}, "metadata/training_manifest.json")
                    band_full_path = mc.MODEL_DIR / target_name / "oof_band_full.csv"
                    band.to_csv(band_full_path)
                    mlflow.log_artifact(str(band_full_path), artifact_path="evaluation")

                    # --- Saving the model + Model Registry / versioning -----------------------
                    # One pyfunc model per target, bundling every (member, horizon) pipeline --
                    # see mlflow_utils.AirGuardTargetModel. Logging it with registered_model_name
                    # creates a new version of "airguard-<target>" automatically.
                    sample_h = mc.HORIZONS[0]
                    input_example = mc.make_X(sample_h).iloc[:2][mc.FEATURE_COLS + ["station"]].copy()
                    input_example["horizon"] = sample_h
                    new_version = mu.log_and_register_target(
                        target_name, pipelines_by_horizon, weights, mc.TARGET_BOUNDS[target_name],
                        mc.FEATURE_COLS, members, input_example,
                    )

                    # --- Model comparison + promotion ----------------------------------------
                    # New version always gets the "staging" alias. It also gets "production" if
                    # it's at least as accurate (avg_mae) as whatever currently holds that alias,
                    # or if nothing holds it yet.
                    promoted, previous_mae = mu.promote_if_better(client, target_name, new_version, avg_mae)
                    mlflow.set_tag("registered_version", new_version)
                    mlflow.set_tag("promoted_to_production", str(promoted))
                    verdict = "PROMOTED to 'production'" if promoted else "kept in 'staging' only"
                    prev_str = f"{previous_mae:.2f}" if previous_mae is not None else "n/a (first version)"
                    mae_str = f"{avg_mae:.2f}" if avg_mae is not None else "n/a"
                    print(f"  [{target_name}] new version={new_version}, avg MAE={mae_str} "
                          f"(previous production={prev_str}) -> {verdict}")

                weights_used[target_name] = weights
                metadata["targets"][target_name] = {"trained_up_to": str(as_of.date()), "members": members}

        (mc.MODEL_DIR / "final_ensemble_weights_used.json").write_text(json.dumps(weights_used, indent=1))
        (mc.MODEL_DIR / "trained_at.json").write_text(json.dumps(metadata, indent=1))

        print("\nRetrain complete. Wrote:")
        print(" ", mc.MODEL_DIR, "(fitted pipelines + oof_band.csv per target)")
        print(" ", mc.MODEL_DIR / "trained_at.json")
        print(" ", f"MLflow experiment '{mu.EXPERIMENT_NAME}' at {mu.TRACKING_URI}")

        log_completion(started_at, "ok", f"targets={list(mc.TARGETS)}")

    except Exception as exc:
        log_completion(started_at, "error", str(exc))
        raise


if __name__ == "__main__":
    main()
