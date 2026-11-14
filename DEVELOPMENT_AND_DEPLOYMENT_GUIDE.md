# AirGuard — development, setup and production deployment guide

**Updated:** 12 November 2026  
**Repository layout:** one repository containing the Python forecasting/ingestion pipeline, FastAPI backend, React/Vite frontend and four GitHub Actions workflows.

> **Current deployment path:** because your Neon database is already populated and the ingestion/training Actions have already run, follow [`DEPLOYMENT_RUNBOOK.md`](DEPLOYMENT_RUNBOOK.md) first. It uses AWS S3 for new MLflow artifacts and explicitly skips repeated historical migration/retraining. This older reference retains the alternative Cloudflare R2 + Render + Vercel setup notes.

This guide is written for the project files in this release. Follow the stages in order: local setup → PostgreSQL → historical data migration → live ingestion → features → model artifacts → persistent MLflow → backend → frontend → production verification.

> **Release-specific data note:** The ZIP contains the source code, fixed parameter JSONs, input historical CSVs and generated feature files. It does **not** contain your previously trained `.joblib` model binaries. Copy your existing `outputs/models/` directory from your current local AirGuard project into this project before expecting a real joblib forecast. The ZIP also contains a `demo_data_marker.json`; the bundled forecast CSVs are demo output, not live forecasts. Details are in `PROJECT_DATA_STATUS.md`.

---

## 1. Architecture and responsibilities

| Component | Responsibility | Runtime |
|---|---|---|
| React + Vite frontend | Overview, Forecast, Stations, Model Health and Methodology pages | Vercel static hosting or local Vite dev server |
| FastAPI backend | Read-only JSON API over committed forecast/feature files; optional ingestion status from PostgreSQL | Render or local Uvicorn |
| PostgreSQL | Historical/live data tables, ingestion log, MLflow runs and Model Registry metadata | Neon (or another managed PostgreSQL provider) |
| Object storage | MLflow model packages, evaluation artifacts and logged files | Private Cloudflare R2 S3-compatible bucket |
| GitHub Actions | CPCB/FIRMS/Open-Meteo/CAMS ingestion, daily forecast rebuild and 30-day due-check/retrain | GitHub-hosted runners |
| `outputs/models/*.joblib` | Joblib production model artifacts used by the normal forecast path | Tracked in Git if each file stays within GitHub's limits |
| `params/*.json` | Fixed production hyperparameters and ensemble weights | Tracked in Git; not changed automatically |
| MLflow | Experiment history, metrics, model packages, model versions and aliases | PostgreSQL + R2 |
| `mlruns/`, `mlflow.db` | Local-only MLflow state | Developer machine only; ignored by Git |
| `tune_parameters_offline.py` | Offline parameter search producing review-only candidate files | Developer machine only; never scheduled in GitHub Actions |

### Data flow

```text
Historical CSVs ── one-time migration ──┐
                                       ├── Neon PostgreSQL
CPCB / FIRMS / Open-Meteo / CAMS ──────┘         │
                                                 ├── MLflow metadata / registry
                                                 │
                                                 └── export_for_modeling.py
                                                          │
                                                          ▼
                                                    CSV source files
                                                          │
                                                    build_features.py
                                                          │
                                              data_quality_check.py
                                                          │
                       ┌──────────────────────────────────┴────────────────────┐
                       ▼                                                       ▼
         predict_forecast.py (daily)                        train_production_models.py (~30 days)
         committed .joblib only                             fixed params/*.json only
                       │                                                       │
                       ▼                                                       ├── overwrite outputs/models/*.joblib
             outputs/forecast_*.csv                                             ├── log MLflow runs to PostgreSQL
                       │                                                       └── log artifacts to R2
                       └──────────────────────┬────────────────────────────────┘
                                              ▼
                              FastAPI reads outputs/ and features/
                                              │
                                      React / Vite frontend
```

**Normal daily forecasting intentionally does not depend on MLflow.** It loads the `.joblib` artifacts in `outputs/models/`. MLflow is used by scheduled retraining and optional registry-backed inference. This keeps the regular forecast path working even if the MLflow registry is temporarily unavailable.

---

## 2. The four GitHub Actions workflows

Keep these four workflow files. No parameter-tuning workflow should exist.

| File | UTC schedule | India time (IST) | Work performed |
|---|---:|---:|---|
| `.github/workflows/ingest.yml` | `15 */3 * * *` | Every 3 hours at :45 | Runs `ingestion/run_all.py`: FIRMS, Open-Meteo and CAMS. CPCB is deliberately separate. |
| `.github/workflows/ingest_cpcb.yml` | `*/15 * * * *` | Every 15 minutes | Runs `ingestion/fetch_cpcb.py` to capture current CPCB readings more frequently. |
| `.github/workflows/rebuild_forecast.yml` | `0 1 * * *` | 06:30 daily | Validates PostgreSQL, exports source data, rebuilds fire/features, validates feature quality, checks model files and runs the joblib forecast when the full model set is ready. |
| `.github/workflows/retrain_model.yml` | `30 1 * * *` | 07:00 daily due-check | Checks whether 30 days have passed since the last successful retrain. If due (or manually forced), refits with the existing fixed parameters and logs to MLflow. |

GitHub Actions cron timing is approximate and can be delayed by GitHub. It is not a real-time scheduler.

### Fixed-parameter policy

- `retrain_model.yml` never runs hyperparameter search.
- `params/best_params_*.json` and `params/ensemble_weights_*.json` are read as inputs and not rewritten by production training.
- `workflow_dispatch` input `force=true` means **refit now using the same fixed parameters**; it does not tune them.
- `tune_parameters_offline.py` is the only full parameter-search entry point. It writes candidate JSONs and score reports to `outputs/parameter_tuning_preview/`, not to `params/`.
- Do not add `tune_parameters_offline.py` to any workflow. Remove any old `research_retrain.yml` if it still exists in the GitHub repository.

### Workflow safety behavior

- The daily forecast and retrain workflows share a concurrency group so their Git auto-commits do not race each other.
- The forecast workflow checks the production model artifact set before inference. If `.joblib` files/OOF bands are missing, it skips prediction and records the reason instead of writing a misleading forecast.
- The forecast workflow can still rebuild and commit feature CSVs while models are absent.
- The production retrain logs a successful completion row to `ingestion_log`; the due-check uses this row to calculate the next 30-day interval.

---

## 3. Prerequisites

Install or create the following before starting:

- Git and GitHub access to the AirGuard repository.
- Python **3.11** for local development (the four Actions workflows use Python 3.11).
- Node.js 20 LTS or another supported current LTS and npm for Vite.
- PostgreSQL database (Neon is the documented deployment path).
- Private S3-compatible object bucket (Cloudflare R2 is the documented path for MLflow artifacts).
- API credentials: CPCB/data.gov.in, NASA FIRMS, Copernicus Atmosphere Data Store (ADS).
- Optional Copernicus Climate Data Store (CDS) credentials only if you run the offline ERA5 pressure-level backfill.
- Render account for the API and Vercel account for the frontend, if deploying those services there.

The frontend uses React 18 and Vite 5. The backend uses FastAPI/Uvicorn. The API does not import the ML libraries for normal serving.

---

## 4. Prepare the repository and preserve your existing trained models

1. Download/extract this project into a new folder or merge the source changes into your current Git working tree.
2. Preserve your existing populated `outputs/models/` directory. **Do not replace it with the ZIP's `outputs/models/` folder**, because the uploaded project archive did not contain the `.joblib` binaries.
3. From PowerShell, copy your old local model directory into the final project. Replace the example paths with your actual locations:

   ```powershell
   $old = "C:\Users\haris\Downloads\airguard-project-fixed\outputs\models"
   $new = "C:\Users\haris\Downloads\AirGuard-Final-Project\outputs\models"
   robocopy $old $new /E
   ```

   `robocopy` exit codes 0–7 generally indicate copy success or success with differences; 8 or higher indicates a copy failure. Verify the files afterwards:

   ```powershell
   Get-ChildItem .\outputs\models -Recurse -Filter "*.joblib" |
       Select-Object FullName, Length
   python ingestion/check_model_artifacts.py
   ```

   The model check should print `Production model artifact check passed.` The archive's `trained_at.json` by itself is not a substitute for the actual joblib files, OOF bands and model-weight snapshot.

4. Ensure all five fixed parameter files and all five fixed weight files remain in `params/`:

   ```text
   params/best_params_AQI.json
   params/best_params_PM25.json
   params/best_params_O3.json
   params/best_params_O3_8H.json
   params/best_params_NOx.json
   params/ensemble_weights_AQI.json
   params/ensemble_weights_PM25.json
   params/ensemble_weights_O3.json
   params/ensemble_weights_O3_8H.json
   params/ensemble_weights_NOx.json
   ```

5. Check `.gitignore` is present. It ignores `.env`, local virtual environments, `mlruns/`, `mlflow.db`, frontend build output and local tuning previews. It does **not** ignore `outputs/models/*.joblib`.
6. `final_delhi_dataset.csv` is a generated, gitignored file. This release includes it for local migration and the updated `data_preprosessing.ipynb` now has a final export cell to recreate it from `aqi_data_delhi/*.csv`.
7. Confirm there is no old parameter tuning workflow:

   ```powershell
   Get-ChildItem .github\workflows
   ```

   The list should contain `ingest.yml`, `ingest_cpcb.yml`, `rebuild_forecast.yml` and `retrain_model.yml` only.

### Git history warning

Your repository previously had oversized MLflow `.pkl` artifacts in Git history and a rejected push. The current `.gitignore` prevents future `mlruns/` files from being added, but `.gitignore` does not remove already committed files from old commits. Do not force-push a rewritten branch until you have a verified backup and have compared the remote history. Store MLflow artifacts in R2, not in Git.

---

## 5. Local Windows development setup

Open PowerShell at the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked in the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the dependency groups needed for full development:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r ingestion/requirements.txt
python -m pip install -r backend/requirements.txt
```

`requirements.txt` includes the broader notebook/research environment. For a smaller production-training environment, use `requirements-production.txt`; for a forecast runner that only needs to load committed models, `ingestion/install_model_runtime.py` installs the recorded model runtime versions (or bounded fallback versions).

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` locally. Never commit it or paste secret values into chat. Hosted GitHub Actions and Render environment variables override local `.env` values because the project uses `load_dotenv(..., override=False)`.

### Environment variables

| Variable | Required where | Purpose |
|---|---|---|
| `DATABASE_URL` | Migration, ingestion, forecast/retrain Actions, API ingestion-status endpoint | PostgreSQL connection URL. Use the provider-generated SSL-enabled URL. |
| `FIRMS_MAP_KEY` | FIRMS ingestion | NASA FIRMS MAP_KEY. |
| `CPCB_API_KEY` | CPCB ingestion | data.gov.in API key. |
| `CPCB_PROXY_URL` | Optional | URL of your proxy/Cloudflare Worker if GitHub runners cannot reach data.gov.in reliably. |
| `CDSAPI_URL` | CAMS ingestion | Copernicus **Atmosphere Data Store** API root: `https://ads.atmosphere.copernicus.eu/api`. |
| `CDSAPI_KEY` | CAMS ingestion | Copernicus ADS token. This is written to `~/.cdsapirc` by `ingest.yml`. |
| `CDS_CLIMATE_API_URL` | Optional offline ERA5 CDS backfill | Separate Climate Data Store endpoint, not used by scheduled CAMS ingestion. |
| `CDS_CLIMATE_API_KEY` | Optional offline ERA5 CDS backfill | Separate Climate Data Store credential. |
| `MLFLOW_EXPERIMENT_NAME` | Local MLflow and retrain Action | Defaults to `AirGuard-Production-Retrain`. |
| `MLFLOW_TRACKING_URI` | Optional override | Persistent MLflow SQL backend; defaults to `DATABASE_URL`. |
| `MLFLOW_REGISTRY_URI` | Optional override | Model Registry backend; defaults to tracking URI. |
| `MLFLOW_ARTIFACT_URI` | Persistent MLflow training | S3 URI such as `s3://airguard-mlflow/airguard/mlflow-artifacts`. |
| `AWS_ACCESS_KEY_ID` | R2/S3 artifact upload/download | R2 S3-compatible access key ID. |
| `AWS_SECRET_ACCESS_KEY` | R2/S3 artifact upload/download | R2 S3-compatible secret. |
| `AWS_DEFAULT_REGION` | R2/S3 artifact upload/download | Use `auto` for Cloudflare R2. |
| `MLFLOW_S3_ENDPOINT_URL` | R2/MinIO | Absolute S3-compatible endpoint URL. For R2: `https://<account-id>.r2.cloudflarestorage.com`. |
| `AIRGUARD_CORS_ORIGINS` | FastAPI backend | Comma-separated exact frontend origins, with scheme and no trailing slash. |
| `VITE_API_BASE_URL` | Frontend build | Public FastAPI origin, injected at Vite build time. |

`CDSAPI_URL`/`CDSAPI_KEY` and `CDS_CLIMATE_API_URL`/`CDS_CLIMATE_API_KEY` are **different credentials for different Copernicus services**. Do not interchange them.

### Run the backend locally

In terminal 1, from the repository root:

```powershell
python -m pip install -r backend/requirements.txt
Set-Location backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: `http://127.0.0.1:8000/docs`  
Health: `http://127.0.0.1:8000/api/health`

### Run the frontend locally

In terminal 2:

```powershell
Set-Location frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:5173`. In development, the frontend defaults to `http://localhost:8000` if `VITE_API_BASE_URL` is not set. In production it deliberately does **not** fall back to localhost; the Vercel build must set the deployed API URL.

### Local demo mode

If you only want to view the interface before the real data pipeline is ready, run:

```powershell
# From the repository root
python backend/dev/generate_sample_data.py
```

This writes synthetic demo output and a `outputs/demo_data_marker.json` marker. It can overwrite files under `outputs/` and `features/`; do not run it over your real production outputs. The API should identify demo data as `status: demo`.

---

## 6. Create the Neon PostgreSQL database

1. Create a Neon project and database, for example `airguard-prod` / `airguard`.
2. Create a dedicated database role and strong generated password.
3. In Neon **Connect**, copy the SSL-enabled PostgreSQL connection string. Keep it private.
4. Use the **direct/non-pooled** URL for the historical migration and MLflow schema upgrades. A pooled URL is useful for high-concurrency app requests, but direct connections are preferable for schema migrations and database administration.
5. Store the connection URL as `DATABASE_URL` in local `.env`, GitHub Actions secrets and Render environment (Render only needs it for the ingestion-status endpoint).

A URL typically resembles:

```text
postgresql://USER:PASSWORD@HOST/airguard?sslmode=require
```

Use the exact string generated by Neon; do not copy the example literally. URL-encode special characters in a password if needed. The project normalizes common `postgres://` and `postgresql+psycopg2://` schemes before connecting.

### Initialize the AirGuard schema

Once local `.env` contains `DATABASE_URL`:

```powershell
python ingestion/validate_database.py
```

This creates the schema if needed, then intentionally fails until the five historical tables have rows. It is a preflight, not the migration itself.

### Generate the historical station CSV

The project includes 20 historical station CSVs in `aqi_data_delhi/`. The notebook now exports the canonical file required by migration:

```powershell
jupyter nbconvert --to notebook --execute --inplace data_preprosessing.ipynb
```

After execution, confirm:

```powershell
Test-Path .\final_delhi_dataset.csv
Import-Csv .\final_delhi_dataset.csv | Select-Object -First 3
```

The export should contain the five known station names and columns used by the feature builder, including `timestamp`, `station`, `PM25`, `PM10`, `NO2`, `NOx`, `O3`, `temperature`, `humidity`, wind and pressure fields.

### Migrate the five historical sources once

The repository contains the corresponding historical ERA5/CAMS files. Check all five inputs:

```powershell
$required = @(
  "final_delhi_dataset.csv",
  "era5_surface_hourly_ist.csv",
  "era5_pressure_levels_6h_ist.csv",
  "era5_pressure_levels_12h_ist.csv",
  "cams_3h_ist.csv"
)
$required | ForEach-Object { "$_ : $(Test-Path $_)" }
```

All must be `True`. Then migrate:

```powershell
python ingestion/migrate_historical.py --yes
python ingestion/validate_database.py
```

The migration uses upserts and is safe to rerun after correcting an input CSV. It localizes historical naive timestamps to IST and converts them to UTC before inserting into `TIMESTAMPTZ` columns.

Do not run scheduled forecast/retrain workflows until the five required historical tables contain data. The workflow preflight checks this automatically.

---

## 7. Create the Cloudflare R2 bucket for MLflow artifacts

PostgreSQL holds MLflow tracking metadata and Model Registry records. The actual model packages and logged artifact files must be placed in persistent object storage because GitHub-hosted runners are temporary.

1. In Cloudflare dashboard, open **R2 Object Storage**.
2. Create a private bucket, for example `airguard-mlflow`.
3. Create an R2 API token scoped to that bucket with Object Read & Write permissions.
4. Save its Access Key ID and Secret Access Key. Do not use your Cloudflare account password or a global account API token.
5. Copy the S3 endpoint for your Cloudflare account.
6. Configure:

   ```text
   MLFLOW_ARTIFACT_URI=s3://airguard-mlflow/airguard/mlflow-artifacts
   AWS_ACCESS_KEY_ID=<R2 access key id>
   AWS_SECRET_ACCESS_KEY=<R2 secret access key>
   AWS_DEFAULT_REGION=auto
   MLFLOW_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
   ```

The bucket must exist before the first training job. Keep it private. The URI uses `s3://bucket-name/prefix`; the endpoint is the account-level S3 API endpoint.

If using AWS S3 instead, select the actual bucket region and use the AWS S3 endpoint or omit `MLFLOW_S3_ENDPOINT_URL` only if the runtime is configured for native AWS S3. This repository's production validation currently expects an explicit endpoint for the configured S3-compatible deployment, so for AWS adjust that validation/configuration intentionally rather than leaving it blank.

---

## 8. Install persistent MLflow metadata tables and test storage

Install the production requirements locally:

```powershell
python -m pip install -r requirements-production.txt
```

Set the local remote MLflow variables in `.env`, then in PowerShell load the current shell values (the Python scripts load `.env` themselves, but these commands use environment variables directly):

```powershell
$env:DATABASE_URL = "<your Neon URL>"
$env:MLFLOW_TRACKING_URI = $env:DATABASE_URL
$env:MLFLOW_REGISTRY_URI = $env:DATABASE_URL
$env:MLFLOW_ARTIFACT_URI = "s3://airguard-mlflow/airguard/mlflow-artifacts"
$env:AWS_ACCESS_KEY_ID = "<R2 key id>"
$env:AWS_SECRET_ACCESS_KEY = "<R2 secret>"
$env:AWS_DEFAULT_REGION = "auto"
$env:MLFLOW_S3_ENDPOINT_URL = "https://<account-id>.r2.cloudflarestorage.com"
$env:AWS_EC2_METADATA_DISABLED = "true"
```

Run the MLflow backend-store upgrade using the same database URL:

```powershell
mlflow db upgrade $env:DATABASE_URL
```

Run a remote tracking/artifact smoke test. This creates a small test experiment/run and uploads one text artifact to R2:

```powershell
@'
import mlflow
import mlflow_utils as mu
mu.init_mlflow(require_remote_artifacts=True)
with mlflow.start_run(run_name="airguard-storage-smoke-test"):
    mlflow.log_param("smoke_test", "true")
    mlflow.log_metric("connection_ok", 1.0)
    mlflow.log_text("AirGuard R2 artifact upload succeeded.", "smoke-test/connection.txt")
print("MLflow PostgreSQL tracking and R2 artifact upload passed.")
'@ | python -
```

Only proceed when this prints success and the test artifact is visible in the private bucket. If the test fails, do not trigger production retraining yet.

### MLflow UI (local developer only)

```powershell
mlflow ui --backend-store-uri $env:DATABASE_URL --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. The process needs the R2 credentials and `MLFLOW_S3_ENDPOINT_URL` to browse/download artifacts. Do not expose an unauthenticated MLflow UI publicly.

### What is stored in MLflow

For each production retrain, the script creates a parent retrain run and nested target runs. Runs capture fixed parameter values, training cutoff, member list, MAE/RMSE by horizon, average metrics, OOF residual bands and a training manifest. A registered model per target packages the selected member pipelines across all ten horizons. New versions receive the `staging` alias; the `production` alias is moved only when the new finite average OOF MAE is no worse than the current production version (or no production alias exists).

Normal dashboard inference remains joblib-based. Registry-backed inference is optional and uses `predict_forecast_mlflow.py` after production aliases and R2 artifacts exist.

---

## 9. Configure GitHub Actions repository secrets

Open GitHub repository → **Settings → Secrets and variables → Actions → New repository secret**.

Add these exact secret names:

| Secret | Required | Used by |
|---|---:|---|
| `DATABASE_URL` | Yes | All ingestion, migration-related workflow steps, due-check, feature export, retraining, optional API DB endpoint |
| `FIRMS_MAP_KEY` | Yes for FIRMS | `ingest.yml` |
| `CPCB_API_KEY` | Yes for CPCB | `ingest.yml` (environment only) and `ingest_cpcb.yml` |
| `CDSAPI_URL` | Yes for CAMS | `ingest.yml` writes ADS `~/.cdsapirc` |
| `CDSAPI_KEY` | Yes for CAMS | `ingest.yml` writes ADS `~/.cdsapirc` |
| `CPCB_PROXY_URL` | Optional | Ingestion workflows; use only if you deploy the CPCB relay |
| `MLFLOW_ARTIFACT_URI` | Yes for production retrain | R2 bucket URI |
| `AWS_ACCESS_KEY_ID` | Yes for R2 | R2 access key |
| `AWS_SECRET_ACCESS_KEY` | Yes for R2 | R2 secret |
| `AWS_DEFAULT_REGION` | Set to `auto` for R2 | R2 region setting; current workflow defaults to `auto` if omitted |
| `MLFLOW_S3_ENDPOINT_URL` | Yes for R2 | R2 S3 endpoint |
| `MLFLOW_TRACKING_URI` | Optional | Separate tracking backend; leave unset to fall back to `DATABASE_URL` |
| `MLFLOW_REGISTRY_URI` | Optional | Separate registry backend; leave unset to fall back to tracking URI |

Do not create secrets with different names such as `NASA_FIRMS_MAP_KEY`; the workflow expects the exact spellings above.

### ADS vs CDS credential reminder

The scheduled CAMS workflow uses Copernicus **Atmosphere Data Store (ADS)** credentials in `CDSAPI_URL` and `CDSAPI_KEY`. The separate `CDS_CLIMATE_API_URL` and `CDS_CLIMATE_API_KEY` are for optional manual Climate Data Store ERA5 pressure-level backfill, not these four scheduled workflows.

---

## 10. Run the ingestion workflows in a safe order

After the database schema and historical migration pass:

1. In GitHub → Actions, choose **Ingest live data** → Run workflow. This runs FIRMS, Open-Meteo and CAMS. CPCB is not run by `run_all.py`.
2. Choose **Ingest CPCB (frequent)** → Run workflow. This polls the current CPCB readings.
3. Inspect each job log and the `ingestion_log` table. A partial source outage should be visible as a failed/partial source, not silently treated as a complete success.
4. Check the latest rows in the live tables. Key tables include `cpcb_hourly`, `weather_surface_hourly`, `weather_pressure_hourly`, `cams_hourly` and `fire_detections_raw`.

The local equivalents, from the repository root after setting `.env`, are:

```powershell
python ingestion/run_all.py
python ingestion/fetch_cpcb.py
```

`run_all.py` runs FIRMS, Open-Meteo and CAMS sequentially and attempts the other sources even if one fails. It deliberately excludes CPCB because the dedicated 15-minute workflow is responsible for that feed.

---

## 11. Build and validate the feature dataset

After historical migration and at least one ingestion cycle:

```powershell
python ingestion/validate_database.py
python ingestion/export_for_modeling.py
python ingestion/rebuild_fire_daily.py
python build_features.py
python ingestion/data_quality_check.py
```

What each command does:

- `validate_database.py`: checks that the required historical tables are present and non-empty.
- `export_for_modeling.py`: exports/reconciles the historical and live PostgreSQL tables into the CSV files expected by the existing feature pipeline.
- `rebuild_fire_daily.py`: creates the daily FIRMS feature table from stored detections.
- `build_features.py`: generates `features/daily_features.csv`, `feature_list.txt`, `feature_groups.json`, `targets.json` and other feature outputs; includes a future-leakage check.
- `data_quality_check.py`: checks the model schema and freshness of the live feature tail. It will intentionally fail when the dataset is only historical/stale, until live data extends the tail.

A clean feature build from the historical CSVs bundled in this release was verified: 7,305 station-day rows, 216 model features, and the feature builder's leakage test passed. The historical files end on 31 December 2025, so `data_quality_check.py` correctly reports stale data until the database/live ingestion extends the timeline. Do not disable this freshness check just to make the workflow green.

---

## 12. Restore/verify production `.joblib` model artifacts

The ZIP provided for this final source review does not contain `.joblib` files. The existing local project on your PC is the source for those files. Restore the full directory using the `robocopy` example in Section 4.

Run:

```powershell
python ingestion/check_model_artifacts.py
```

Expected output:

```text
Production model artifact check passed.
```

The checker validates:

- the `trained_at.json` manifest;
- each manifest-listed member model for horizons 1–10 for all five targets;
- each target's `oof_band.csv` with ten finite `q_lo`/`q_hi` rows;
- `outputs/models/final_ensemble_weights_used.json` with all five targets.

If a model file is missing, the checker writes `ready=false` to GitHub Actions and the daily forecast job skips prediction. This is safer than publishing an incomplete forecast.

The daily job installs the exact numerical/model dependency versions in `trained_at.json` when they are present. Older manifests without `model_runtime_versions` use bounded fallback versions from `requirements-model-runtime.txt`.

### GitHub size limits for joblib files

GitHub warns for files over 50 MB and rejects normal Git files over 100 MB. Your previously reported RandomForest joblib files were around 53–59 MB, so they are warnings but below the hard limit. If any joblib later exceeds 100 MB, use Git LFS or persistent object storage plus a download step in the forecast workflow; do not put MLflow `.pkl` artifacts back in normal Git history.

---

## 13. Run the first production retrain with fixed parameters

Only do this after:

- Neon is reachable and `validate_database.py` passes;
- historical migration and live ingestion are complete enough to build current features;
- `build_features.py` and `data_quality_check.py` pass;
- R2 bucket and MLflow smoke test pass;
- GitHub Actions secrets are configured.

From GitHub Actions, select **Retrain production models (fixed parameters, 30-day cycle)** → Run workflow. For an intentional immediate first run, set `force=true`. When there has never been a successful retrain marker, the daily gate also treats the first run as due, so the force input is not strictly required.

The training script:

1. reads `params/best_params_*.json` and `params/ensemble_weights_*.json`;
2. fits the enabled members at each target/horizon;
3. refreshes `outputs/models/<target>/*.joblib` and `oof_band.csv`;
4. saves `outputs/models/trained_at.json` including runtime package versions;
5. logs parent/target runs, metrics, model artifacts and registry versions to MLflow;
6. writes `retrain_models` success status to `ingestion_log` so the due-check can schedule the next interval.

The workflow commits the `outputs/models/**` files. It does not commit `mlruns/`, does not change `params/`, and does not run `tune_parameters_offline.py`.

Check GitHub Actions summary/logs, the R2 prefix, MLflow runs and registered models. Only after this succeeds should the normal daily workflow be expected to forecast using the new models.

---

## 14. Deploy the FastAPI backend to Render

The API reads files from repository-root `outputs/`, `features/` and `params/`. Keep the Render service's **Root Directory empty / repository root**. Do not set its root to `backend`, or it will not find sibling project folders.

Create a Render Web Service connected to the GitHub repo and `main` branch, with:

| Setting | Value |
|---|---|
| Root Directory | Leave blank (repo root) |
| Environment | Python |
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT` |

Set environment variables:

```text
DATABASE_URL=<Neon PostgreSQL URL>
AIRGUARD_CORS_ORIGINS=https://<your-vercel-domain>
```

Add your custom frontend domain to `AIRGUARD_CORS_ORIGINS` as a comma-separated exact origin if applicable. The API only needs the database URL for `/api/ingestion/status/*`; most dashboard routes read the checked-out files and do not use MLflow or numerical training libraries.

Enable automatic deployment from `main`. The daily forecast/retrain GitHub Actions commit the refreshed data/model files; Render must deploy the commit containing updated outputs. If your Git integration does not deploy commits made by Actions, configure a Render Deploy Hook or another deployment trigger.

### Test the backend

Replace the URL with the Render service URL:

```powershell
$API = "https://<your-render-service>.onrender.com"
Invoke-RestMethod "$API/"
Invoke-RestMethod "$API/api/health"
Invoke-RestMethod "$API/api/stations"
Invoke-RestMethod "$API/api/targets"
Invoke-RestMethod "$API/api/forecast/city?target=AQI"
Invoke-RestMethod "$API/api/forecast/station/Dwarka%20Sector%208?target=AQI"
Invoke-RestMethod "$API/api/history/city?target=AQI&days=30"
Invoke-RestMethod "$API/api/model/info"
Invoke-RestMethod "$API/api/ingestion/status/latest-per-job"
```

`/api/ingestion/status/latest-per-job` requires `DATABASE_URL`. Forecast endpoints can return HTTP 503 when their required forecast/features file has not yet been produced. `/api/health` should report `demo` while `outputs/demo_data_marker.json` exists; it should not be interpreted as production healthy/live status.

---

## 15. Deploy the React frontend to Vercel

Import the same GitHub repository in Vercel:

| Setting | Value |
|---|---|
| Framework | Vite |
| Root Directory | `frontend` |
| Install Command | `npm install` |
| Build Command | `npm run build` |
| Output Directory | `dist` |

Set this environment variable for Production (and Preview if desired):

```text
VITE_API_BASE_URL=https://<your-render-service>.onrender.com
```

Do not include a trailing slash. Vite injects this value at **build time**. After changing it, redeploy the frontend.

After Vercel deploys, return to Render and set `AIRGUARD_CORS_ORIGINS` to the exact Vercel URL (and custom domain, if any), save and redeploy the API.

Open the site and check the browser developer tools → Network. All `/api/...` requests should point to the Render API, not `localhost:8000`. The frontend contains Overview, Forecast, Stations, Model Health and Methodology pages.

---

## 16. Full system acceptance checklist

Do these in order; do not begin with the frontend if the data pipeline is not ready.

| # | Check | Expected |
|---:|---|---|
| 1 | `DATABASE_URL` connection | `SELECT 1` / `validate_database.py` reaches Neon |
| 2 | Historical station export | Notebook writes `final_delhi_dataset.csv` with all five stations |
| 3 | Historical migration | Five `_hist` tables have non-zero rows |
| 4 | R2 smoke test | MLflow can write and read an artifact in the private bucket |
| 5 | `ingest.yml` manual dispatch | FIRMS, Open-Meteo and CAMS complete or clearly report partial source failures |
| 6 | `ingest_cpcb.yml` manual dispatch | CPCB readings are upserted and job logged |
| 7 | `export_for_modeling.py` | CSV inputs are refreshed from the database |
| 8 | `build_features.py` | `daily_features.csv`, feature list and target metadata are generated; leakage test passes |
| 9 | `data_quality_check.py` | Current live tail is fresh; no unexpected schema/missingness failures |
| 10 | Model artifact checker | `ready=true` for all `.joblib` artifacts and residual bands |
| 11 | `rebuild_forecast.yml` | Real forecast CSVs are generated; demo marker removed only after successful real prediction |
| 12 | FastAPI | Health/stations/forecast endpoints return expected JSON; ingestion endpoint works with DB URL |
| 13 | Vercel | All five pages load and frontend API requests reach Render |
| 14 | `retrain_model.yml` | Fixed-parameter retrain logs MLflow metrics/models, promotes aliases correctly and writes successful DB marker |
| 15 | Registry-backed inference (optional) | `predict_forecast_mlflow.py` downloads production alias artifacts from R2 |

---

## 17. Offline parameter tuning (developer only)

Use this only when you intentionally want to research candidate hyperparameters. It can be computationally expensive.

```powershell
python build_features.py
python tune_parameters_offline.py --trials 10
```

Optional controls are available for targets/folds/trials; run `python tune_parameters_offline.py --help` to inspect them. Candidate files and score reports are written under `outputs/parameter_tuning_preview/`. The program does not have an apply option and does not write to `params/`. Review candidate metrics manually. If you later choose new fixed parameters, make a separate reviewed manual edit to the relevant `params/*.json` file and commit it intentionally.

Do not schedule this script in GitHub Actions.

---

## 18. Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `DATABASE_URL is not set` | Local `.env` missing or Action secret missing | Add `DATABASE_URL` to the root `.env` locally and GitHub Actions secrets. |
| `relation ingestion_log does not exist` | Schema not initialized | Run `python ingestion/validate_database.py`; it creates schema, then checks historical row counts. |
| Historical migration stops before writing | One or more of the five required input CSVs are missing | Generate `final_delhi_dataset.csv`; verify the four ERA5/CAMS files. |
| Empty historical table | CSV columns/rows not compatible or migration not executed | Inspect migration logs and table counts; correct input then rerun safe upsert migration. |
| `FIRMS_MAP_KEY is not set` | Wrong secret name | Add secret exactly `FIRMS_MAP_KEY`. |
| CAMS unauthorized or missing `.cdsapirc` | ADS URL/token wrong or ADS dataset terms not accepted | Use the ADS endpoint/token, not Climate Data Store credentials; verify dataset terms/access. |
| CPCB API returns 403/429 or no rows | data.gov.in key/quota or runner network restriction | Check API key, resource ID and station names; optionally configure `CPCB_PROXY_URL`. |
| `data_quality_check.py` says file stale | Live ingestion has not extended the historical feature timeline | Run ingestion and export steps; do not bypass the staleness check. |
| `ready=false` from model check | `.joblib`, OOF band, manifest or weights snapshot missing | Restore the complete `outputs/models/` directory before forecasting. |
| `MLFLOW_ARTIFACT_URI` missing | R2 bucket URI is not configured | Set `s3://bucket/prefix` as a GitHub Actions secret. |
| R2 `NoCredentialsError` / AccessDenied | Key/secret, region, endpoint or bucket permissions wrong | Verify bucket-scoped read/write token, `AWS_DEFAULT_REGION=auto`, and `MLFLOW_S3_ENDPOINT_URL`. |
| Model Registry alias missing | No successful model registered/promoted yet | Run a successful retrain and inspect each target's registered versions/aliases. |
| Forecast job skips prediction | Model checker reports incomplete model set | Expected safe behavior; restore or produce every required model artifact. |
| Backend returns 503 for ingestion status | Backend has no working `DATABASE_URL` or `ingestion_log` schema | Add Neon URL to Render and verify DB/schema availability. |
| Backend cannot find `outputs/` or `features/` | Render Root Directory was set to `backend` | Clear Render Root Directory; run backend command from `cd backend` while keeping repo root checkout. |
| Frontend reports API URL not configured | `VITE_API_BASE_URL` not set when Vite built | Add it in Vercel and redeploy. |
| Browser CORS error | Render CORS origin differs from Vercel origin | Set exact scheme/host in `AIRGUARD_CORS_ORIGINS`; redeploy Render. |
| GitHub rejects a file >100 MB | Oversized file in Git history | Keep MLflow `.pkl` files out of Git; use object storage. A `.gitignore` change alone does not purge old commits. |

---

## 19. Security and operational maintenance

- Never commit `.env`, `.cdsapirc`, database URLs, ADS tokens, CPCB/FIRMS keys or R2 credentials.
- Use GitHub Actions Secrets and host-provider environment settings for secrets.
- Keep the R2 bucket private; use a bucket-scoped read/write token.
- Do not expose the MLflow UI publicly without authentication and network controls; bind local UI to `127.0.0.1`.
- Keep `mlruns/` and `mlflow.db` out of Git. These local stores are not the production persistence layer.
- Back up Neon and ensure R2 objects for registered model versions remain available.
- Monitor all four workflows after enabling schedules.
- Verify model artifact presence after every retrain. Keep the `.joblib` files, OOF bands, `trained_at.json` and model weight snapshot together.
- Keep production parameter JSONs stable unless a reviewed offline research result is explicitly adopted.
- Check the `/api/health` response. `demo` is not production healthy, and a historical-only feature table is stale.

---

## 20. Useful official documentation

- Neon connection strings and pooling: <https://neon.com/docs/connect/connection-pooling>
- Cloudflare R2 S3 API: <https://developers.cloudflare.com/r2/get-started/s3/>
- Copernicus ADS API setup: <https://ads.atmosphere.copernicus.eu/how-to-api>
- NASA FIRMS MAP_KEY: <https://firms.modaps.eosdis.nasa.gov/api/map_key/>
- Render FastAPI deployment: <https://render.com/docs/deploy-fastapi>
- Render monorepo support: <https://render.com/docs/monorepo-support>
- Vercel Vite deployment: <https://vercel.com/docs/frameworks/frontend/vite>
- GitHub Actions workflow syntax: <https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax>
- MLflow tracking and artifact stores: <https://mlflow.org/docs/latest/tracking/server/>
- MLflow SQL backend store: <https://mlflow.org/docs/latest/tracking/backend-stores/>
