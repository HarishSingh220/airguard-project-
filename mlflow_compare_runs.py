"""
mlflow_compare_runs.py
Command-line version of the MLflow UI's "Compare runs" view: pulls every
per-target run logged by train_production_models.py and prints, per target,
the most recent few retrains side by side with their avg MAE/RMSE and which
one (if any) is the current "production" alias.

Usage:
    python mlflow_compare_runs.py              # last 3 runs per target
    python mlflow_compare_runs.py --n 10        # last 10 runs per target
"""
import argparse

import mlflow
from mlflow.tracking import MlflowClient

import mlflow_utils as mu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="runs per target to show")
    args = parser.parse_args()

    mu.init_mlflow()
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(mu.EXPERIMENT_NAME)
    if exp is None:
        print(f"No experiment named {mu.EXPERIMENT_NAME!r} yet -- run train_production_models.py first.")
        return

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
    if "params.target" not in runs.columns:
        print("No per-target runs logged yet -- run train_production_models.py first.")
        return
    runs = runs[runs["params.target"].notna()]

    for target_name in sorted(runs["params.target"].unique()):
        print(f"\n=== {target_name} ===")
        name = mu.registered_model_name(target_name)
        try:
            prod_version = client.get_model_version_by_alias(name, mu.ALIAS_PRODUCTION).version
        except Exception:
            prod_version = None

        sub = runs[runs["params.target"] == target_name].head(args.n)
        cols = {
            "start_time": "when",
            "tags.registered_version": "version",
            "metrics.avg_mae": "avg_mae",
            "metrics.avg_rmse": "avg_rmse",
            "tags.promoted_to_production": "promoted?",
        }
        present = [c for c in cols if c in sub.columns]
        view = sub[present].rename(columns=cols)
        if "version" in view.columns:
            view["is_current_production"] = view["version"] == prod_version
        print(view.to_string(index=False))


if __name__ == "__main__":
    main()
