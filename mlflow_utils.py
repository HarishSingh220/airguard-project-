"""AirGuard's persistent MLflow tracking, artifact storage and model registry helpers.

Production configuration
------------------------
Metadata / registry: PostgreSQL via DATABASE_URL or MLFLOW_TRACKING_URI.
Artifacts:           S3 or S3-compatible object storage via MLFLOW_ARTIFACT_URI.

For local development, SQLite + local artifacts remain available when no remote
settings are supplied. GitHub Actions deliberately requires remote metadata and
remote artifact storage so registered models never point at an ephemeral runner.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from urllib.parse import urlparse

# Load a root-level .env for local developer runs. GitHub Actions and hosted
# deployments provide environment variables directly, so the file is optional.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
except ImportError:
    pass

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

DEFAULT_EXPERIMENT_NAME = "AirGuard-Production-Retrain"
ALIAS_STAGING = "staging"
ALIAS_PRODUCTION = "production"


def _normalize_database_url(value: str) -> str:
    """Normalize common provider-style Postgres URLs for SQLAlchemy/MLflow."""
    if value.startswith("postgres://"):
        return "postgresql://" + value[len("postgres://"):]
    for prefix in ("postgresql+psycopg2://", "postgresql+psycopg://"):
        if value.startswith(prefix):
            return "postgresql://" + value[len(prefix):]
    return value


def resolve_tracking_uri() -> str:
    explicit = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if explicit:
        return _normalize_database_url(explicit)
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return _normalize_database_url(database_url)
    return "sqlite:///mlflow.db"


TRACKING_URI = resolve_tracking_uri()
REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", "").strip() or TRACKING_URI
ARTIFACT_URI = os.getenv("MLFLOW_ARTIFACT_URI", "").strip().rstrip("/")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME).strip()


def _is_remote_tracking_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme in {"postgresql", "mysql", "mssql", "http", "https"}


def _is_s3_artifact_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme == "s3" and bool(parsed.netloc)


def _artifact_location_matches(experiment_location: str, requested_root: str) -> bool:
    """True when an existing experiment already writes under the requested root."""
    root = requested_root.rstrip("/")
    existing = (experiment_location or "").rstrip("/")
    return existing == root or existing.startswith(root + "/")


def _get_or_create_remote_experiment(client: MlflowClient, name: str, artifact_root: str):
    experiment = client.get_experiment_by_name(name)
    if experiment and experiment.lifecycle_stage == "active":
        if _artifact_location_matches(experiment.artifact_location, artifact_root):
            return experiment
        return None

    if experiment and experiment.lifecycle_stage != "active":
        client.restore_experiment(experiment.experiment_id)
        experiment = client.get_experiment(experiment.experiment_id)
        if _artifact_location_matches(experiment.artifact_location, artifact_root):
            return experiment
        return None

    try:
        experiment_id = client.create_experiment(
            name,
            artifact_location=artifact_root,
            tags={
                "project": "AirGuard",
                "artifact_store": "s3-compatible",
                "managed_by": "mlflow_utils.py",
            },
        )
    except MlflowException:
        # Another workflow may have created the same experiment concurrently.
        experiment = client.get_experiment_by_name(name)
        if experiment and _artifact_location_matches(experiment.artifact_location, artifact_root):
            return experiment
        raise
    return client.get_experiment(experiment_id)


def init_mlflow(require_remote_artifacts: bool | None = None) -> str:
    """Configure MLflow and return the active experiment ID.

    Existing local-artifact experiments are NOT silently reused when an S3
    artifact root is configured. A new ``<name>-remote-artifacts`` experiment is
    created instead, since MLflow experiment artifact locations are immutable.
    """
    global TRACKING_URI, REGISTRY_URI, ARTIFACT_URI, EXPERIMENT_NAME

    TRACKING_URI = resolve_tracking_uri()
    REGISTRY_URI = _normalize_database_url(
        os.getenv("MLFLOW_REGISTRY_URI", "").strip() or TRACKING_URI
    )
    ARTIFACT_URI = os.getenv("MLFLOW_ARTIFACT_URI", "").strip().rstrip("/")
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME).strip()

    if not EXPERIMENT_NAME:
        raise RuntimeError("MLFLOW_EXPERIMENT_NAME cannot be empty.")

    if require_remote_artifacts is None:
        require_remote_artifacts = os.getenv("GITHUB_ACTIONS", "").lower() == "true"

    if require_remote_artifacts:
        if not _is_remote_tracking_uri(TRACKING_URI):
            raise RuntimeError(
                "Production MLflow requires a persistent tracking backend. Set DATABASE_URL "
                "to PostgreSQL or MLFLOW_TRACKING_URI to a remote MLflow server; SQLite is "
                "not persistent on GitHub Actions runners."
            )
        if not _is_s3_artifact_uri(ARTIFACT_URI):
            raise RuntimeError(
                "Production MLflow requires MLFLOW_ARTIFACT_URI such as "
                "s3://<bucket>/airguard/mlflow-artifacts. Configure an S3-compatible bucket "
                "before running the retraining workflow."
            )
        if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
            raise RuntimeError(
                "GitHub Actions needs AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (or an "
                "equivalent workload-identity setup) to write MLflow artifacts."
            )
        if not os.getenv("AWS_DEFAULT_REGION"):
            raise RuntimeError(
                "Set AWS_DEFAULT_REGION for the MLflow S3-compatible artifact store "
                "(use 'auto' for Cloudflare R2)."
            )
        endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "").strip()
        if not endpoint:
            raise RuntimeError(
                "Set MLFLOW_S3_ENDPOINT_URL for the configured S3-compatible artifact store. "
                "For Cloudflare R2 use https://<account-id>.r2.cloudflarestorage.com."
            )
        parsed_endpoint = urlparse(endpoint)
        if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
            raise RuntimeError(
                "MLFLOW_S3_ENDPOINT_URL must be an absolute http(s) endpoint URL, "
                "for example https://<account-id>.r2.cloudflarestorage.com."
            )

    if ARTIFACT_URI and not _is_s3_artifact_uri(ARTIFACT_URI):
        raise ValueError(
            "MLFLOW_ARTIFACT_URI must be an S3-compatible URI, for example "
            "s3://bucket-name/airguard/mlflow-artifacts."
        )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(REGISTRY_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI, registry_uri=REGISTRY_URI)

    if ARTIFACT_URI:
        # A previous experiment may have been created with file:/// artifacts.
        # Artifact locations cannot be changed in-place, so use an explicitly
        # named remote-backed experiment rather than registering broken versions.
        experiment = _get_or_create_remote_experiment(client, EXPERIMENT_NAME, ARTIFACT_URI)
        if experiment is None:
            remote_name = f"{EXPERIMENT_NAME}-remote-artifacts"
            experiment = _get_or_create_remote_experiment(client, remote_name, ARTIFACT_URI)
            if experiment is None:
                raise RuntimeError(
                    f"Experiment {remote_name!r} already exists with a different artifact location. "
                    "Set a new MLFLOW_EXPERIMENT_NAME or use a fresh artifact prefix."
                )
            EXPERIMENT_NAME = remote_name
        mlflow.set_experiment(experiment_id=experiment.experiment_id)
        print(f"MLflow tracking: {TRACKING_URI}")
        print(f"MLflow artifacts: {experiment.artifact_location}")
        print(f"MLflow experiment: {EXPERIMENT_NAME} ({experiment.experiment_id})")
    else:
        if require_remote_artifacts:
            # Defensive duplicate check; normally the validation above catches it.
            raise RuntimeError("Remote MLflow artifact storage is required in production.")
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            mlflow.set_experiment(EXPERIMENT_NAME)
            experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        else:
            mlflow.set_experiment(experiment_id=experiment.experiment_id)
        print(
            "WARNING: MLflow is using local/default artifact storage. This is suitable "
            "for local development only; configure MLFLOW_ARTIFACT_URI for durable runs."
        )
    return experiment.experiment_id


def registered_model_name(target_name: str) -> str:
    return f"airguard-{target_name.lower()}"


class AirGuardTargetModel(mlflow.pyfunc.PythonModel):
    """One target's complete ensemble, across all members and forecast horizons."""

    def __init__(self, pipelines_by_horizon, weights, bounds, feature_cols, members):
        self.pipelines_by_horizon = pipelines_by_horizon
        self.weights = weights
        self.bounds = tuple(bounds)
        self.feature_cols = list(feature_cols)
        self.members = list(members)

    def predict(self, context, model_input, params=None):
        import numpy as np
        import pandas as pd

        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("AirGuardTargetModel expects a pandas DataFrame input.")
        required = self.feature_cols + ["station", "horizon"]
        missing = [column for column in required if column not in model_input.columns]
        if missing:
            raise ValueError(f"AirGuardTargetModel input is missing columns: {missing}")
        if model_input.empty:
            out = model_input.copy()
            out["prediction"] = pd.Series(dtype=float)
            return out

        if model_input["horizon"].isna().any():
            raise ValueError("The horizon column cannot contain null values.")
        if model_input["station"].isna().any():
            raise ValueError("The station column cannot contain null values.")

        numeric_horizons = pd.to_numeric(model_input["horizon"], errors="coerce")
        if numeric_horizons.isna().any() or not np.allclose(numeric_horizons, numeric_horizons.astype(int)):
            raise ValueError("Horizon values must be integers from 1 through 10.")
        if not numeric_horizons.between(1, 10).all():
            raise ValueError("Horizon values must be integers from 1 through 10.")

        preds = np.full(len(model_input), np.nan, dtype=float)
        for horizon_value in sorted(numeric_horizons.unique()):
            horizon = int(horizon_value)
            mask = numeric_horizons.eq(horizon).to_numpy()
            pipelines = self.pipelines_by_horizon.get(horizon)
            if not pipelines:
                raise ValueError(f"No fitted pipelines for horizon {horizon} in this model.")
            horizon_weights = self.weights.get(str(horizon), self.weights.get(horizon, {}))
            blend = np.zeros(int(mask.sum()), dtype=float)
            weight_sum = 0.0
            for member in self.members:
                weight = float(horizon_weights.get(member, 0.0))
                if weight <= 0:
                    continue
                if member not in pipelines:
                    raise ValueError(f"Missing pipeline for member {member!r}, horizon {horizon}.")
                x = model_input.loc[mask, self.feature_cols + ["station"]]
                blend += weight * pipelines[member].predict(x)
                weight_sum += weight
            if weight_sum <= 0:
                raise ValueError(f"No positive ensemble weights for horizon {horizon}.")
            preds[mask] = blend / weight_sum

        if not np.isfinite(preds).all():
            raise ValueError("Input contains missing or non-finite forecast horizons.")
        lower, upper = self.bounds
        out = model_input.copy()
        out["prediction"] = np.clip(preds, lower, upper)
        return out


def log_and_register_target(
    target_name,
    pipelines_by_horizon,
    weights,
    bounds,
    feature_cols,
    members,
    input_example,
):
    """Log a target's ensemble to remote artifact storage and register a version."""
    model = AirGuardTargetModel(pipelines_by_horizon, weights, bounds, feature_cols, members)
    prediction_example = model.predict(None, input_example)
    signature = mlflow.models.infer_signature(input_example, prediction_example)
    # Pin the exact training runtime so a later registry download does not
    # silently resolve a different sklearn/XGBoost/LightGBM ABI or behavior.
    from importlib.metadata import version as package_version, PackageNotFoundError

    distributions = [
        "mlflow", "numpy", "pandas", "scikit-learn",
        "xgboost", "lightgbm", "scipy", "joblib",
    ]
    try:
        pip_requirements = [
            f"{distribution}=={package_version(distribution)}"
            for distribution in distributions
        ]
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Cannot package the MLflow model because runtime dependency "
            f"{exc.name!r} is not installed in the training environment."
        ) from exc

    # AirGuardTargetModel is defined in this module rather than in the training
    # entrypoint. Include this source file so a clean MLflow serving environment
    # can import the custom PyFunc class when it downloads a registered model.
    model_info = mlflow.pyfunc.log_model(
        artifact_path="forecast_model",
        python_model=model,
        code_paths=[str(Path(__file__).resolve())],
        registered_model_name=registered_model_name(target_name),
        input_example=input_example,
        signature=signature,
        pip_requirements=pip_requirements,
    )
    version = model_info.registered_model_version
    if version is None:
        raise RuntimeError(
            f"MLflow logged the {target_name} model but did not return a registered model version. "
            "Check registry permissions and the tracking/registry backend configuration."
        )
    return str(version)


def promote_if_better(
    client: MlflowClient,
    target_name: str,
    new_version: str,
    new_avg_mae: float | None,
    metric_key: str = "avg_mae",
):
    """Set staging for every version; promote only a finite MAE that is no worse."""
    if new_avg_mae is None or not math.isfinite(float(new_avg_mae)):
        raise ValueError(f"Cannot promote {target_name}: {metric_key} is missing or non-finite.")

    name = registered_model_name(target_name)
    client.set_registered_model_alias(name, ALIAS_STAGING, str(new_version))

    previous_mae = None
    current = None
    try:
        current = client.get_model_version_by_alias(name, ALIAS_PRODUCTION)
        current_run = client.get_run(current.run_id)
        value = current_run.data.metrics.get(metric_key)
        previous_mae = float(value) if value is not None else None
    except MlflowException as exc:
        # Only a genuinely missing production alias means this is the first
        # promotion. Authentication, network, permission and registry errors
        # must fail closed rather than accidentally promote a new version.
        error_code = str(getattr(exc, "error_code", "") or "").upper()
        message = str(exc).upper()
        missing_alias = (
            error_code in {"RESOURCE_DOES_NOT_EXIST", "RESOURCE_NOT_FOUND"}
            or "RESOURCE_DOES_NOT_EXIST" in message
            or "RESOURCE_NOT_FOUND" in message
            or ("ALIAS" in message and ("NOT FOUND" in message or "DOES NOT EXIST" in message))
        )
        if not missing_alias:
            raise
        current = None

    promote = current is None or previous_mae is None or float(new_avg_mae) <= previous_mae
    if promote:
        client.set_registered_model_alias(name, ALIAS_PRODUCTION, str(new_version))
    return promote, previous_mae
