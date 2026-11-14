# AirGuard MLflow: persistent tracking, artifacts and registry

AirGuard now uses MLflow for durable experiment tracking and a model registry,
without changing the daily joblib-based production forecast path.

## Architecture

| Concern | Production location | Why |
|---|---|---|
| Run metadata, params, metrics and registry aliases | Persistent PostgreSQL | Survives short-lived GitHub Actions runners |
| Pyfunc model packages, OOF bands and run artifacts | S3-compatible object storage | The model files remain fetchable from later runs and other machines |
| Operational inference models | `outputs/models/**/*.joblib` | Existing `predict_forecast.py` keeps serving without MLflow |
| Fixed hyperparameters and ensemble weights | `params/*.json` | Production retraining reads these files only |

`mlruns/` and `mlflow.db` are local development stores and are ignored by Git.
They are not the production MLflow persistence layer.

Each call to `train_production_models.py` logs one parent retrain run and one
nested run per target. Child runs include the target, training cutoff, model
members, fixed parameter files, per-horizon MAE/RMSE, average MAE/RMSE, OOF
residual bands, and a manifest. A single registered model per target packages
the full target ensemble across all ten horizons. The new version receives the
`staging` alias; it receives `production` only if its finite OOF average MAE is
no worse than the current production alias (or no production alias exists).

## One-time infrastructure setup

You need two persistent services:

1. A PostgreSQL database reachable from GitHub Actions and your deployment
   environment. MLflow creates its own tables in this database; it does not
   overwrite AirGuard's ingestion tables.
2. An S3-compatible bucket for MLflow artifacts. AWS S3, Cloudflare R2 and
   MinIO are supported through the S3 artifact plugin interface exposed by
   MLflow/boto3. Create a dedicated bucket or prefix, for example:
   `s3://your-bucket/airguard/mlflow-artifacts`.

Add the following in GitHub repository **Settings → Secrets and variables →
Actions → New repository secret**:

| Secret | Required | Example / purpose |
|---|---:|---|
| `DATABASE_URL` | Yes | `postgresql://user:password@host:5432/db?sslmode=require` |
| `MLFLOW_ARTIFACT_URI` | Yes | `s3://your-bucket/airguard/mlflow-artifacts` |
| `AWS_ACCESS_KEY_ID` | Yes | S3-compatible object-store access key |
| `AWS_SECRET_ACCESS_KEY` | Yes | S3-compatible object-store secret |
| `AWS_DEFAULT_REGION` | Yes | `auto` for Cloudflare R2 |
| `MLFLOW_S3_ENDPOINT_URL` | Yes for this R2 setup | `https://<account-id>.r2.cloudflarestorage.com` |
| `MLFLOW_TRACKING_URI` | Optional | Overrides `DATABASE_URL` when using a separate tracking server |
| `MLFLOW_REGISTRY_URI` | Optional | Overrides the tracking URI for the registry |

For Cloudflare R2, the endpoint format is typically
`https://<account-id>.r2.cloudflarestorage.com`; use the R2 access key/secret,
not your Cloudflare account password. The bucket must exist before running the
workflow. Grant the key permission to list/read/write objects in that bucket.

Do not commit `.env`. For local work, copy `.env.example` to `.env`, fill the
same values, and install `python-dotenv` (already included in
`requirements-production.txt`). The root `.env` is loaded automatically by
`mlflow_utils.py` and `train_production_models.py`.

## Run and inspect locally

```powershell
python -m pip install -r requirements-production.txt
python build_features.py
python train_production_models.py
python mlflow_compare_runs.py --n 5
```

When `DATABASE_URL` and `MLFLOW_ARTIFACT_URI` are set, local runs write to the
same durable stores used by Actions. Without remote settings, local-only
experiments use `sqlite:///mlflow.db` and the local artifact store; those runs
are not visible to another machine and should not be treated as production
registry artifacts.

To start the MLflow browser UI against the PostgreSQL tracking store:

```powershell
mlflow ui --backend-store-uri "$env:DATABASE_URL" --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. If your artifacts are in S3/R2, the process that
runs the UI also needs the object-store credentials and optional
`MLFLOW_S3_ENDPOINT_URL` so artifact previews can be fetched.

## Model serving options

The default daily GitHub Action continues to use `predict_forecast.py`, which
loads the committed `.joblib` files. This is intentional: it keeps the
forecasting path independent of MLflow uptime and registry availability.

For registry-backed inference, run after at least one successful production
retrain has registered and promoted each target:

```powershell
python predict_forecast_mlflow.py
```

This loads `models:/airguard-<target>@production` for each target and writes
separate `*_mlflow.csv` outputs. The registry-backed service needs access to
PostgreSQL and the S3-compatible artifact store. It does not fit or tune a
model.

## Parameter policy: fixed in Actions, tuning offline only

The scheduled 30-day workflow only refits models using the committed
`params/best_params_*.json` and `params/ensemble_weights_*.json`. It never
changes these files. The old scheduled `research_retrain.yml` workflow should
be deleted from the GitHub repository; it is deliberately not part of this
project package.

The standalone `tune_parameters_offline.py` is the only supported full tuning
entry point. It does not call MLflow or any external service. It writes preview
JSONs and a score report to `outputs/parameter_tuning_preview/` by default:

```powershell
python build_features.py
python tune_parameters_offline.py --trials 10
```

The tuner is intentionally report-only: it has no option to replace
`params/*.json`. Keep those committed files fixed. Do not add this script to a
GitHub Actions workflow or schedule. The old
walk-forward notebook remains as research material with `RUN_TUNING=False` and
`RUN_SHAP=False` by default; it is not run by production Actions.

## Troubleshooting

- **`MLFLOW_ARTIFACT_URI` missing:** GitHub Actions fails before training. Set
  the bucket URI as a repository secret.
- **`NoCredentialsError` / S3 access denied:** verify key/secret, region,
  endpoint and bucket policy.
- **Old experiment points at `file:///...`:** MLflow experiment artifact
  locations are immutable. `mlflow_utils.py` creates a separate
  `AirGuard-Production-Retrain-remote-artifacts` experiment rather than
  continuing to register models whose files live on an ephemeral runner.
- **Registry model version cannot load from another machine:** confirm that
  the registered version's `artifact_uri` starts with the configured S3
  bucket/prefix and that the consuming process has object-store access.
- **Daily forecast still works while MLflow is unavailable:** expected. The
  main production forecast is joblib-based and deliberately independent of
  the registry.
