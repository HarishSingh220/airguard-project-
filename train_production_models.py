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
separate, deliberately much rarer job -- see research_retrain.yml -- this
script only ever calls .fit(), never a hyperparameter search.

Run from the repository root, after build_features.py:
    python train_production_models.py

Outputs (written to outputs/models/):
    <target>/<model>_h<h>.joblib      every fitted production pipeline
    <target>/oof_band.csv             the (q_lo, q_hi) residual band per horizon
    final_ensemble_weights_used.json  the (unchanged) weights applied this retrain, for the record
    trained_at.json                   when this retrain ran and which date it trained up to

predict_forecast.py (run daily) only ever reads what this script writes; it
never calls .fit() itself.
"""
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib

import model_common as mc

warnings.filterwarnings("ignore")

# ingestion/db.py and config.py live in ingestion/, not at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "ingestion"))


def log_completion(started_at, status, message=""):
    """Best-effort: records this retrain in the same ingestion_log table the
    live fetchers use, so check_retrain_due.py can find "the last successful
    retrain" without any extra bookkeeping file. If DATABASE_URL isn't set
    (e.g. a local run) this just warns instead of failing the retrain itself --
    the models are still trained and saved either way.
    """
    try:
        import db  # noqa: E402  (import after sys.path edit, by necessity)
        engine = db.get_engine()
        db.log_run(engine, "retrain_models", started_at, datetime.now(timezone.utc),
                   None, status, message[:2000])
    except Exception as exc:
        print(f"WARNING: could not log retrain completion to the database ({exc}). "
              f"check_retrain_due.py will not see this run.")


def main():
    started_at = datetime.now(timezone.utc)
    weights_used = {}
    metadata = {"trained_at": started_at.isoformat(), "targets": {}}

    try:
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

            for name in members:
                pipes = mc.fit_production(mc.tuned_estimator(params, name), mc.MODEL_CONFIGS[name][1],
                                          prefix, actual_col=actual_col)
                for h, pipe in pipes.items():
                    out_path = mc.MODEL_DIR / target_name / f"{name}_h{h}.joblib"
                    try:
                        joblib.dump(pipe, out_path)
                    except OSError as exc:
                        raise OSError(f"Could not save fitted model to {out_path}: {exc}") from exc
                print(f"  [{target_name}] fitted {name} ({len(pipes)} horizons)")

            band = mc.oof_residuals(target_name, params, weights)
            band.to_csv(mc.MODEL_DIR / target_name / "oof_band.csv")

            weights_used[target_name] = weights
            metadata["targets"][target_name] = {"trained_up_to": str(as_of.date()), "members": members}

        (mc.MODEL_DIR / "final_ensemble_weights_used.json").write_text(json.dumps(weights_used, indent=1))
        (mc.MODEL_DIR / "trained_at.json").write_text(json.dumps(metadata, indent=1))

        print("\nRetrain complete. Wrote:")
        print(" ", mc.MODEL_DIR, "(fitted pipelines + oof_band.csv per target)")
        print(" ", mc.MODEL_DIR / "trained_at.json")

        log_completion(started_at, "ok", f"targets={list(mc.TARGETS)}")

    except Exception as exc:
        log_completion(started_at, "error", str(exc))
        raise


if __name__ == "__main__":
    main()
