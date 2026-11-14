"""
Check that outputs/models/ contains a complete production model set.

Exit status stays zero so daily ingestion/features can still complete on a
fresh clone; `ready=true|false` is written to GITHUB_OUTPUT when running in
Actions, allowing the forecast step to be skipped cleanly if model binaries
have not yet been restored/committed.
"""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "models"
TARGETS = ("AQI", "PM25", "O3", "O3_8H", "NOx")
HORIZONS = tuple(range(1, 11))
ALLOWED_MEMBERS = {"RandomForest", "LightGBM", "XGBoost", "Ridge"}


def check_artifacts():
    errors = []
    manifest_path = MODEL_DIR / "trained_at.json"
    if not manifest_path.is_file():
        return False, [f"Missing {manifest_path.relative_to(ROOT)}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"Cannot read trained_at.json: {exc}"]

    trained_targets = manifest.get("targets") if isinstance(manifest, dict) else None
    if not isinstance(trained_targets, dict):
        return False, ["trained_at.json has no valid 'targets' object"]

    for target in TARGETS:
        target_info = trained_targets.get(target)
        if not isinstance(target_info, dict):
            errors.append(f"{target}: target entry missing from trained_at.json")
            continue
        members = target_info.get("members")
        if not isinstance(members, list) or not members:
            errors.append(f"{target}: no trained model members listed")
            continue
        for member in members:
            if not isinstance(member, str) or member not in ALLOWED_MEMBERS:
                errors.append(f"{target}: unsupported model member {member!r} in trained_at.json")
                continue
            for horizon in HORIZONS:
                path = MODEL_DIR / target / f"{member}_h{horizon}.joblib"
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"Missing {path.relative_to(ROOT)}")

        band_path = MODEL_DIR / target / "oof_band.csv"
        if not band_path.is_file():
            errors.append(f"Missing {band_path.relative_to(ROOT)}")
        else:
            try:
                with band_path.open("r", newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                if not rows or not {"q_lo", "q_hi"}.issubset(reader.fieldnames or []):
                    errors.append(f"{band_path.relative_to(ROOT)} must contain q_lo/q_hi columns")
                else:
                    if len(rows) != len(HORIZONS):
                        errors.append(f"{band_path.relative_to(ROOT)} must contain 10 horizon rows")
                    for row in rows:
                        for column in ("q_lo", "q_hi"):
                            try:
                                value = float(row[column])
                                if not (value == value and abs(value) != float("inf")):
                                    raise ValueError("not finite")
                            except (TypeError, ValueError):
                                errors.append(
                                    f"{band_path.relative_to(ROOT)} contains an invalid {column} value"
                                )
                                break
            except (OSError, csv.Error) as exc:
                errors.append(f"Cannot read {band_path.relative_to(ROOT)}: {exc}")

    weights_path = MODEL_DIR / "final_ensemble_weights_used.json"
    if not weights_path.is_file():
        errors.append(f"Missing {weights_path.relative_to(ROOT)}")
    else:
        try:
            weights = json.loads(weights_path.read_text(encoding="utf-8"))
            if not isinstance(weights, dict) or not all(target in weights for target in TARGETS):
                errors.append("final_ensemble_weights_used.json does not include all five targets")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Cannot read final_ensemble_weights_used.json: {exc}")

    return not errors, errors


def main():
    ready, errors = check_artifacts()
    if ready:
        print("Production model artifact check passed.")
    else:
        print("Production forecast is not ready; model artifact check failed:")
        for error in errors:
            print(f"  - {error}")
        print("Daily ingestion and feature generation may continue, but forecast generation will be skipped.")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"ready={'true' if ready else 'false'}\n")


if __name__ == "__main__":
    main()
