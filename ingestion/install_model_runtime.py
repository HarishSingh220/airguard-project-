"""
Install the same numerical/model libraries used when the current joblib models
were trained. New train_production_models.py runs persist exact package versions
in outputs/models/trained_at.json. Older manifests fall back to the bounded
requirements-model-runtime.txt file with an explicit warning.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "models" / "trained_at.json"
FALLBACK = ROOT / "requirements-model-runtime.txt"
PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "scipy", "joblib")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_-]*$")


def main():
    versions = {}
    if MANIFEST.is_file():
        try:
            metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
            candidate = metadata.get("model_runtime_versions", {})
            if isinstance(candidate, dict):
                versions = candidate
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: cannot read model runtime manifest ({exc}); using bounded fallback dependencies.")

    if all(
        isinstance(versions.get(package), str) and VERSION_RE.fullmatch(versions[package])
        for package in PACKAGES
    ):
        requirements = [f"{package}=={versions[package]}" for package in PACKAGES]
        print("Installing exact model runtime versions recorded with the trained joblib artifacts:")
        for requirement in requirements:
            print(f"  {requirement}")
        subprocess.run([sys.executable, "-m", "pip", "install", *requirements], check=True)
    else:
        print(
            "WARNING: trained_at.json has no complete model_runtime_versions section. "
            "Installing bounded fallback versions. After the next production retrain, "
            "the exact runtime will be recorded for subsequent forecast jobs."
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(FALLBACK)],
            check=True,
        )


if __name__ == "__main__":
    main()
