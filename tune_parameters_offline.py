"""Run the expensive AirGuard hyperparameter + ensemble-weight search locally.

This file is intentionally NOT invoked by GitHub Actions. It reads the latest
features, evaluates candidates with chronological folds strictly before
model_common.WF_START, and writes candidate JSON files to
outputs/parameter_tuning_preview/ by default.

The production params/*.json files are intentionally read-only. This script
never overwrites or promotes its results; review the preview files manually.
It does not call MLflow, upload data, or contact external services.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

mc = None  # loaded after argument parsing so `--help` works before features exist


def load_model_common():
    """Import the production feature schema only after CLI help/args are handled."""
    global mc
    try:
        import model_common as model_common_module
    except Exception as exc:  # model_common validates generated feature files on import
        raise SystemExit(
            "Could not load the production feature schema. Run `python build_features.py` "
            "from the repository root first, and verify features/daily_features.csv, "
            f"features/targets.json and features/feature_list.txt exist. Details: {exc}"
        ) from exc
    mc = model_common_module
    return mc


TUNE_HORIZONS = (1, 5, 10)
PARAM_SPACES = {
    "LightGBM": {
        "n_estimators": [300, 500, 800, 1000],
        "learning_rate": loguniform(0.01, 0.1),
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [10, 20, 40, 80],
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.4, 0.6),
        "reg_lambda": loguniform(0.1, 30),
        "max_depth": [-1, 6, 10],
    },
    "XGBoost": {
        "n_estimators": [300, 500, 800, 1000],
        "learning_rate": loguniform(0.01, 0.1),
        "max_depth": [3, 4, 6, 8],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.4, 0.6),
        "reg_lambda": loguniform(0.5, 30),
        "gamma": [0, 0.5, 2],
    },
    "RandomForest": {
        "n_estimators": [150, 250, 300, 500],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": [0.2, 0.33, 0.5, 1.0],
    },
    "Ridge": {"alpha": loguniform(1, 1000)},
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Refusing to write non-finite tuning values to JSON.")
    return value


def default_params(member: str) -> dict:
    base = mc.MODEL_CONFIGS[member][0].get_params()
    return {key: base[key] for key in PARAM_SPACES[member]}


def fold_windows(horizon: int, folds: int, fold_days: int):
    cutoff = pd.Timestamp(mc.WF_START)
    dates = mc.daily["date"]
    windows = []
    for index in range(folds):
        start = cutoff - pd.Timedelta(days=fold_days * (folds - index))
        end = min(
            start + pd.Timedelta(days=fold_days - 1),
            cutoff - pd.Timedelta(days=1 + horizon),
        )
        if end >= start and (dates >= start).any():
            windows.append((start, end))
    if not windows:
        raise RuntimeError(
            f"No valid chronological validation folds for horizon {horizon}. "
            f"Check WF_START={cutoff.date()} and the date range in features/daily_features.csv."
        )
    return windows


def _make_pipeline(member: str, params: dict):
    estimator, prep_kind = mc.MODEL_CONFIGS[member]
    safe_params = {key: value for key, value in params.items() if value is not None}
    return Pipeline([
        ("prep", mc.make_prep(prep_kind, mc.FEATURE_COLS)),
        ("model", clone(estimator).set_params(**safe_params)),
    ])


def cross_validate_target(
    target_name: str,
    member: str,
    horizon: int,
    params: dict,
    folds: int,
    fold_days: int,
    min_train_rows: int,
    collect_oof: bool = False,
):
    prefix = mc.TARGETS[target_name]["prefix"]
    target_col = f"{prefix}{horizon}"
    if target_col not in mc.daily.columns:
        raise KeyError(f"Missing {target_col!r} in features/daily_features.csv.")

    dates = mc.daily["date"]
    X = mc.make_X(horizon)
    y = mc.daily[target_col]
    squared_error = 0.0
    count = 0
    predictions = []

    for valid_start, valid_end in fold_windows(horizon, folds, fold_days):
        train_mask = (dates <= valid_start - pd.Timedelta(days=horizon)) & y.notna()
        valid_mask = (dates >= valid_start) & (dates <= valid_end) & y.notna()
        train_count = int(train_mask.sum())
        valid_count = int(valid_mask.sum())
        if train_count < min_train_rows or valid_count == 0:
            continue

        pipeline = _make_pipeline(member, params)
        pipeline.fit(X.loc[train_mask], y.loc[train_mask])
        predicted = np.asarray(pipeline.predict(X.loc[valid_mask]), dtype=float)
        actual = y.loc[valid_mask].to_numpy(dtype=float)
        finite = np.isfinite(predicted) & np.isfinite(actual)
        if not finite.any():
            continue
        predicted = predicted[finite]
        actual = actual[finite]
        rows = mc.daily.index[valid_mask].to_numpy()[finite]
        squared_error += float(np.square(predicted - actual).sum())
        count += len(actual)
        if collect_oof:
            predictions.append(pd.DataFrame({"row": rows, "actual": actual, "prediction": predicted}))

    score = float(np.sqrt(squared_error / count)) if count else float("inf")
    if collect_oof:
        frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(
            columns=["row", "actual", "prediction"]
        )
        return score, frame
    return score


def nearest_tuned_horizon(horizon: int) -> int:
    return min(TUNE_HORIZONS, key=lambda candidate: (abs(candidate - horizon), candidate))


def tune_target(target_name: str, trials: int, folds: int, fold_days: int,
                min_train_rows: int, random_state: int):
    tuned: dict[tuple[str, int], dict] = {}
    report = []
    print(f"\n=== Tuning {target_name} ===")

    for member_index, member in enumerate(mc.MEMBERS_ALL):
        baseline = default_params(member)
        sampled = list(ParameterSampler(
            PARAM_SPACES[member],
            n_iter=trials,
            random_state=random_state + member_index * 101 + list(mc.TARGETS).index(target_name) * 1009,
        ))
        candidates = [baseline] + [json_safe(candidate) for candidate in sampled]

        for horizon in TUNE_HORIZONS:
            scored = []
            for candidate_index, candidate in enumerate(candidates):
                value = cross_validate_target(
                    target_name, member, horizon, candidate,
                    folds, fold_days, min_train_rows,
                )
                scored.append(value)
                print(
                    f"  {member:13s} h={horizon:2d} candidate={candidate_index + 1:02d}/"
                    f"{len(candidates):02d} RMSE={value:.3f}"
                )
            best_index = int(np.argmin(scored))
            if not np.isfinite(scored[best_index]):
                raise RuntimeError(
                    f"All tuning candidates failed for target={target_name}, member={member}, "
                    f"horizon={horizon}. Check data coverage and validation-fold sizes."
                )
            tuned[(member, horizon)] = candidates[best_index]
            baseline_score = scored[0]
            best_score = scored[best_index]
            report.append({
                "target": target_name,
                "member": member,
                "tuned_horizon": horizon,
                "baseline_cv_rmse": baseline_score,
                "best_cv_rmse": best_score,
                "improvement_pct": (100 * (baseline_score - best_score) / baseline_score)
                    if np.isfinite(baseline_score) and baseline_score else 0.0,
                "selected_candidate": best_index + 1,
                "candidate_count": len(candidates),
            })
            print(f"  -> selected {member} h={horizon}; CV RMSE {baseline_score:.3f} -> {best_score:.3f}")

    # Materialize a complete horizon map. Production expects all 10 horizons,
    # while the expensive search is run at horizons 1, 5 and 10 (as in the
    # original research notebook); intervening horizons inherit the nearest.
    params_by_member = {member: {} for member in mc.MEMBERS_ALL}
    for member in mc.MEMBERS_ALL:
        for horizon in mc.HORIZONS:
            source_horizon = nearest_tuned_horizon(horizon)
            params_by_member[member][str(horizon)] = tuned[(member, source_horizon)]

    # Fit non-negative ensemble weights on identical, chronological OOF rows
    # for each production horizon, with no use of data at/after WF_START.
    weights_by_horizon = {}
    for horizon in mc.HORIZONS:
        member_oof = {}
        for member in mc.MEMBERS_ALL:
            params = params_by_member[member][str(horizon)]
            _, oof = cross_validate_target(
                target_name, member, horizon, params,
                folds, fold_days, min_train_rows, collect_oof=True,
            )
            if not oof.empty:
                member_oof[member] = oof.set_index("row")[["actual", "prediction"]]

        if not member_oof:
            raise RuntimeError(f"No out-of-fold predictions available for {target_name} horizon {horizon}.")
        common_rows = sorted(set.intersection(*(set(frame.index) for frame in member_oof.values())))
        if not common_rows:
            raise RuntimeError(
                f"No common OOF rows across ensemble members for {target_name} horizon {horizon}."
            )
        available_members = list(member_oof)
        actual = member_oof[available_members[0]].loc[common_rows, "actual"]
        prediction_matrix = np.column_stack([
            member_oof[member].loc[common_rows, "prediction"].to_numpy(dtype=float)
            for member in available_members
        ])
        # All members should share the same labels; verify instead of silently
        # mixing different evaluation rows.
        for member, frame in member_oof.items():
            if not np.allclose(frame.loc[common_rows, "actual"], actual, equal_nan=True):
                raise RuntimeError(f"OOF labels disagree across members for {target_name} h={horizon}.")

        weight_values, _ = nnls(prediction_matrix, actual.to_numpy(dtype=float))
        total = float(weight_values.sum())
        if total <= 0:
            weight_values = np.full(len(available_members), 1.0 / len(available_members))
        else:
            weight_values = weight_values / total
        horizon_weight_map = {member: 0.0 for member in mc.MEMBERS_ALL}
        horizon_weight_map.update({
            member: float(weight_values[index]) for index, member in enumerate(available_members)
        })
        weights_by_horizon[str(horizon)] = horizon_weight_map
        print(
            f"  weights h={horizon}: " + ", ".join(
                f"{member}={weights_by_horizon[str(horizon)][member]:.2f}"
                for member in mc.MEMBERS_ALL
            )
        )

    return params_by_member, weights_by_horizon, report


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Run the full AirGuard parameter search offline.")
    parser.add_argument("--targets", nargs="+", help="Targets to tune (default: all production targets).")
    parser.add_argument("--trials", type=int, default=10, help="Random candidates per member/horizon (default: 10).")
    parser.add_argument("--folds", type=int, default=4, help="Chronological CV folds (default: 4).")
    parser.add_argument("--fold-days", type=int, default=90, help="Days in each validation fold (default: 90).")
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    import os
    os.chdir(ROOT)
    load_model_common()
    args.targets = args.targets or list(mc.TARGETS)
    unknown_targets = sorted(set(args.targets) - set(mc.TARGETS))
    if unknown_targets:
        parser.error(f"Unknown target(s): {unknown_targets}. Valid targets: {list(mc.TARGETS)}")

    if args.trials < 1 or args.folds < 1 or args.fold_days < 1 or args.min_train_rows < 1:
        parser.error("--trials, --folds, --fold-days and --min-train-rows must all be positive.")

    preview_root = ROOT / "outputs" / "parameter_tuning_preview"
    preview_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for target in args.targets:
        params, weights, report = tune_target(
            target, args.trials, args.folds, args.fold_days,
            args.min_train_rows, args.random_state,
        )
        summary_rows.extend(report)
        _write_json(preview_root / f"best_params_{target}.json", params)
        _write_json(preview_root / f"ensemble_weights_{target}.json", weights)

    summary = pd.DataFrame(summary_rows)
    summary_path = preview_root / "tuning_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "wf_start": str(pd.Timestamp(mc.WF_START).date()),
        "horizons_tuned": list(TUNE_HORIZONS),
        "all_production_horizons": list(mc.HORIZONS),
        "trials_per_member_horizon": args.trials,
        "folds": args.folds,
        "fold_days": args.fold_days,
        "targets": args.targets,
        "applied_to_production": False,
        "production_params_modified": False,
        "method": "expanding chronological validation before WF_START; NNLS ensemble weights",
    }
    _write_json(preview_root / "manifest.json", manifest)

    print(f"\nCandidate files written to: {preview_root}")
    print(f"Tuning summary: {summary_path}")

    print("Production params were NOT changed. This offline utility only creates reviewable candidates.")



if __name__ == "__main__":
    main()
