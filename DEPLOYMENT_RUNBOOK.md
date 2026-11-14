# AirGuard production deployment runbook

**Scope:** deploy the uploaded AirGuard project using the existing Neon database and previously executed GitHub Actions workflows. This runbook is deliberately written to avoid duplicating work that has already been completed.

**Recommended deployment layout**

| Component | Service | Purpose |
|---|---|---|
| Source control and automation | GitHub + GitHub Actions | Source of truth, scheduled ingestion, feature rebuilds, forecast generation and retraining gate |
| Operational database | Existing Neon PostgreSQL project `airguard-aqi`, branch `main` | Historical/live air-quality data, ingestion log and MLflow metadata/registry |
| MLflow artifact store | Private AWS S3 bucket | Durable model packages and evaluation artifacts |
| API | Render Web Service | Read-only FastAPI API over the latest committed forecast/feature files, plus optional database-backed ingestion status |
| Frontend | Vercel | React/Vite static dashboard |

There is **no separate MLflow server required** for the current design. Training code writes MLflow metadata directly to PostgreSQL and large artifacts to S3. You can run the MLflow UI locally to inspect the shared tracking database. MLflow separates run/registry metadata in its backend store from model and other binary artifacts in its artifact store. See the [MLflow backend-store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/) and [artifact-store](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/) documentation.

---

## 0. Read this first: what is already done

The November 12, 2026 Neon audit found that:

- Neon is reachable and the five required historical tables already contain data through December 31, 2025.
- Live data ingestion workflows have run repeatedly. Do not repeat a historical backfill or bulk migration as part of deployment.
- `cpcb_hourly` had 2,723 placeholder rows with NULL pollutant values. The merged patch prevents new unusable averages from being stored and filters existing NULL averages from feature export. It does not delete old records.
- Older CAMS logs include HTTP 400 failures. The uploaded `ingestion/fetch_cams.py` already uses `data_format=netcdf` and `download_format=zip`, without the old `grid` argument.
- MLflow's existing metadata is in PostgreSQL, but the previously registered artifacts reference `file:///C:/.../mlruns/...`. Those local Windows paths cannot be fetched by GitHub-hosted runners or Render.
- The uploaded project ZIP does not include the production `.joblib` files, the per-target `oof_band.csv` files, or the model weight snapshot required by the normal forecast pipeline.

**Therefore, do not run these commands now:**

```powershell
python ingestion/migrate_historical.py --yes
python train_production_models.py
python tune_parameters_offline.py
```

The first is a one-time historical migration and your Neon tables are already populated. The second is a full retraining job and your training workflow has already been run multiple times. The third is offline research-only tuning. Revisit retraining only if the checks in Stage 6 prove the production joblib artifacts are unavailable or you deliberately need new S3-backed MLflow model versions.

Your uploaded ZIP is a source package, not proof of the current contents of the live GitHub branch. Check `origin/main` before restoring any model files or overwriting your existing checkout.

## 1. Prepare your Windows machine

Install/check these tools:

- Git for Windows
- Python 3.11 (same minor version used by GitHub Actions)
- Node.js 20 LTS or newer supported LTS; this project's Vite 5 supports modern Node
- AWS CLI v2
- GitHub CLI (`gh`) optional

Check versions in PowerShell:

```powershell
git --version
py -3.11 --version
node --version
npm --version
aws --version
gh --version
```

After you enter the repository clone and copy the merged source files (Stage 2), create the local Python environment once:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-production.txt -r ingestion/requirements.txt -r backend/requirements.txt
```

For notebook/research work, install the additional root development requirements too:

```powershell
python -m pip install -r requirements.txt
```

The project includes `.python-version` set to `3.11`. This is important for Render: new Render Python services currently default to Python 3.14, while this repository's GitHub workflows and pinned backend dependencies are aligned to Python 3.11. Render supports pinning the version with `.python-version` or `PYTHON_VERSION`; see [Render Python versions](https://render.com/docs/python-version).

Extract the updated project ZIP to a temporary package folder. The ZIP contains one top-level `AirGuard-Final-Project` directory:

```powershell
$ExtractRoot = "$HOME\Downloads\AirGuard-Merged-Package"
Expand-Archive "$HOME\Downloads\AirGuard-Final-Project-Merged.zip" `
  -DestinationPath $ExtractRoot -Force
$PackageRoot = Join-Path $ExtractRoot "AirGuard-Final-Project"
Set-Location $PackageRoot
```

If you already have a working clone of the GitHub repository, use that clone rather than creating a second Git history. If you do not have a local clone yet:

```powershell
gh auth login
git clone https://github.com/<OWNER>/<REPOSITORY>.git "$HOME\Downloads\AirGuard-repository"
Set-Location "$HOME\Downloads\AirGuard-repository"
```

Do not copy the ZIP's `outputs/models/` directory on top of a populated directory without first checking it.

## 2. Update GitHub safely

### 2.1 Inspect the existing repository

From the existing clone:

```powershell
Set-Location "C:\path\to\your\existing\AirGuard-repository"
git status --short
git remote -v
git branch --show-current
git fetch origin
git branch -vv
git log --oneline --decorate -8 origin/main
```

If your default branch is not `main`, substitute the actual branch consistently below. Never fix a divergent-history issue by force-pushing a new extracted ZIP over the remote repository.

### 2.2 Create a patch branch

```powershell
git switch main
git pull --ff-only origin main
git switch -c fix/cpcb-live-data-export
```

Copy the merged package changes into the clone. Run this from the repository clone after setting `$PackageRoot` to the extracted package path. This deliberately copies only changed source/docs files; it does not overwrite forecast outputs, features or trained models:

```powershell
$PackageRoot = "$HOME\Downloads\AirGuard-Merged-Package\AirGuard-Final-Project"
Copy-Item "$PackageRoot\ingestion\fetch_cpcb.py" .\ingestion\ -Force
Copy-Item "$PackageRoot\ingestion\export_for_modeling.py" .\ingestion\ -Force
Copy-Item "$PackageRoot\.python-version" . -Force
Copy-Item "$PackageRoot\.env.example" . -Force
Copy-Item "$PackageRoot\README.md" . -Force
Copy-Item "$PackageRoot\DEVELOPMENT_AND_DEPLOYMENT_GUIDE.md" . -Force
Copy-Item "$PackageRoot\DEPLOYMENT_RUNBOOK.md" . -Force
Copy-Item "$PackageRoot\PATCH_MERGE_REPORT.md" . -Force
Copy-Item "$PackageRoot\tests\test_fetch_cpcb.py" .\tests\ -Force
```

The changed files to carry into the clone are:

```text
ingestion/fetch_cpcb.py
ingestion/export_for_modeling.py
.python-version
.env.example
README.md
DEVELOPMENT_AND_DEPLOYMENT_GUIDE.md
DEPLOYMENT_RUNBOOK.md
PATCH_MERGE_REPORT.md
tests/test_fetch_cpcb.py
```

Do not overwrite `outputs/models/` from the ZIP: this uploaded archive does not contain the trained production binaries. Restore those from your successful training output only after checking which files already exist on `origin/main`.

The existing `.github/workflows/retrain_model.yml` has intentionally been preserved. The supplied patch variant removed `validate_database.py` from the retraining path and changed the concurrency group away from the shared repository-write group. The checked-in project already uses `AWS_DEFAULT_REGION: auto` as its fallback, so replacing that workflow was unnecessary and less safe.

### 2.3 Validate before committing

Run from repository root:

```powershell
python -m compileall -q ingestion backend
python -m unittest discover -s tests -v
python ingestion/check_model_artifacts.py
git diff --check
git status --short
```

The model-artifact check is expected to fail if you are testing the source ZIP alone, because the archive omitted model binaries. It must pass against the actual branch you intend to deploy before the real forecast workflow can generate a forecast.

Stage only the intended files:

```powershell
git add ingestion/fetch_cpcb.py ingestion/export_for_modeling.py `
  .python-version .env.example README.md DEVELOPMENT_AND_DEPLOYMENT_GUIDE.md `
  DEPLOYMENT_RUNBOOK.md PATCH_MERGE_REPORT.md tests/test_fetch_cpcb.py
git diff --cached --stat
git diff --cached
git commit -m "Fix CPCB ingestion and document deployment"
git push -u origin fix/cpcb-live-data-export
```

Open a pull request into `main`, review it, and merge it. Then update your local main:

```powershell
git switch main
git pull --ff-only origin main
```

If starting from a brand-new empty GitHub repository, clone it first and copy the project contents into that clone. Do not run `git init` inside an unrelated existing project or use `git push --force` to overwrite remote history. Keep `.env`, `.venv`, `mlruns/`, `mlflow.db`, and credentials out of Git. The current `.gitignore` already excludes these.

## 3. Verify the existing Neon database (do not re-migrate)

### 3.1 Confirm the connection string

In Neon Console, open the existing project `airguard-aqi`, branch `main`, and choose **Connect**. Use the exact SSL-enabled connection string. Do not edit the hostname or remove `sslmode=require`. Use the direct (non-pooler) URL for database migrations/MLflow metadata operations; a pooled URL may be used for small app request traffic. Neon documents how to select pooled vs direct URLs in its [connection-string guide](https://github.com/neondatabase/website/blob/main/content/faqs/find-database-connection-string-url.md).

Create the local environment file once:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `DATABASE_URL` to the Neon connection string. Do not paste the value into Git, a commit, a screenshot or a public issue.

### 3.2 Test the connection and historical table counts

```powershell
python ingestion/validate_database.py
```

Expected result: database preflight passes and the five required historical tables are non-empty. The script runs idempotent `CREATE TABLE IF NOT EXISTS` schema setup and count checks; it does not call the historical migration. If you want a direct connection smoke test:

```powershell
@'
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
load_dotenv()
url = os.environ["DATABASE_URL"]
engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})
with engine.connect() as conn:
    print(conn.execute(text("SELECT current_database(), current_schema(), 1")).one())
engine.dispose()
'@ | python -
```

Do not run `migrate_historical.py --yes` again: the audit confirmed data in `station_hourly_hist`, `cams_3h_hist`, `era5_surface_hist`, `era5_pressure6h_hist` and `era5_pressure12h_hist`. Only run that migration against a genuinely new/empty database after verifying the source CSVs and taking a database backup.

## 4. Create AWS S3 storage for MLflow artifacts

Use a private, dedicated S3 bucket. This bucket stores MLflow model packages/evaluation artifacts; it is not the AirGuard operational PostgreSQL database and is not the dashboard's static-file store.

### 4.1 Create the bucket

In AWS Console:

1. Select the region closest to your workloads, for example `Asia Pacific (Mumbai) – ap-south-1`.
2. Create a globally unique bucket name such as `airguard-mlflow-prod-<unique-suffix>`.
3. Keep **Block all public access** enabled.
4. Enable default server-side encryption (SSE-S3/AES-256, or your approved KMS key).
5. Do not enable public website hosting or public ACLs.

The S3 service recommends keeping Block Public Access enabled. See [S3 Block Public Access](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-block-public-access.html).

CLI alternative (after `aws configure` or your approved AWS SSO flow):

```powershell
aws sts get-caller-identity
$Bucket = "airguard-mlflow-prod-<unique-suffix>"
$Region = "ap-south-1"
aws s3api create-bucket --bucket $Bucket --region $Region `
  --create-bucket-configuration "LocationConstraint=$Region"
aws s3api put-public-access-block --bucket $Bucket `
  --public-access-block-configuration `
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
$Encryption = @{ Rules = @(@{ ApplyServerSideEncryptionByDefault = @{ SSEAlgorithm = "AES256" } }) } `
  | ConvertTo-Json -Depth 5 -Compress
aws s3api put-bucket-encryption --bucket $Bucket `
  --server-side-encryption-configuration $Encryption
```

If the bucket already exists, do not create another one; verify its region, privacy and encryption instead.

### 4.2 Create a restricted IAM identity

Create an IAM user/access key for GitHub Actions (or use a supported federated/OIDC role if you later revise the workflow). Do not use AWS root credentials. Restrict its S3 permissions to this bucket and the AirGuard prefix. The minimum policy should allow listing the bucket and reading/writing/deleting objects under `airguard/mlflow-artifacts/*` (plus `GetBucketLocation`); avoid broad `s3:*` on `*`.

Keep the access key ID and secret access key private. Create two GitHub repository secrets:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

For local tests, use your local AWS profile or temporary PowerShell environment variables. The current `mlflow_utils.py` explicitly checks for the two environment variables when running in GitHub Actions.

### 4.3 Decide the artifact URI and endpoint

For the example Mumbai bucket:

```text
MLFLOW_ARTIFACT_URI=s3://airguard-mlflow-prod-<unique-suffix>/airguard/mlflow-artifacts
AWS_DEFAULT_REGION=ap-south-1
MLFLOW_S3_ENDPOINT_URL=https://s3.ap-south-1.amazonaws.com
```

The source code currently requires an explicit `MLFLOW_S3_ENDPOINT_URL`, including for standard AWS S3. Use the regional S3 endpoint that matches the bucket's region. Do not use the R2 value `auto` for an AWS S3 bucket. If you choose Cloudflare R2 instead, use its account endpoint and `AWS_DEFAULT_REGION=auto`; do not mix the two providers' credentials/endpoints.

## 5. Configure MLflow against the existing Neon metadata store

### 5.1 Understand what is already configured

The audit found MLflow tables and registry records already installed in Neon. Do not create a second database just for MLflow. In this project:

- Metadata/registry: PostgreSQL via `MLFLOW_TRACKING_URI` (defaults to `DATABASE_URL`) and `MLFLOW_REGISTRY_URI` (defaults to tracking URI).
- Model artifacts: S3 via `MLFLOW_ARTIFACT_URI` and `MLFLOW_S3_ENDPOINT_URL`.
- Standard dashboard forecasts: local committed joblib files under `outputs/models/`; this path does not call MLflow.

The old experiment is tied to a Windows `file:///C:/...` artifact location. MLflow experiment artifact locations are immutable. With S3 configured, `mlflow_utils.py` deliberately creates/selects `AirGuard-Production-Retrain-remote-artifacts` rather than trying to reuse that old local-artifact experiment. Keep the old experiment/model versions as historical records; do not delete them.

### 5.2 Add local MLflow variables

Add these to `.env` (do not commit the actual values):

```dotenv
MLFLOW_EXPERIMENT_NAME=AirGuard-Production-Retrain
MLFLOW_TRACKING_URI=postgresql://<role>:<encoded-password>@<direct-neon-host>/<database>?sslmode=require
MLFLOW_REGISTRY_URI=postgresql://<role>:<encoded-password>@<direct-neon-host>/<database>?sslmode=require
MLFLOW_ARTIFACT_URI=s3://airguard-mlflow-prod-<unique-suffix>/airguard/mlflow-artifacts
AWS_ACCESS_KEY_ID=<IAM-access-key-id>
AWS_SECRET_ACCESS_KEY=<IAM-secret-access-key>
AWS_DEFAULT_REGION=ap-south-1
MLFLOW_S3_ENDPOINT_URL=https://s3.ap-south-1.amazonaws.com
AWS_EC2_METADATA_DISABLED=true
```

You may omit `MLFLOW_TRACKING_URI` and `MLFLOW_REGISTRY_URI` if both should use `DATABASE_URL`. If explicitly set, ensure they point to the same existing Neon project/branch/database unless you intentionally maintain separate stores.

Install only the production training/MLflow dependency set for this test:

```powershell
python -m pip install -r requirements-production.txt
```

### 5.3 Test S3 without retraining

```powershell
@'
import os
import boto3
from dotenv import load_dotenv
load_dotenv()
s3 = boto3.client(
    "s3",
    region_name=os.environ["AWS_DEFAULT_REGION"],
    endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
)
bucket = os.environ["MLFLOW_ARTIFACT_URI"].split("/", 3)[2]
key = "airguard/mlflow-artifacts/deployment-smoke-test.txt"
s3.put_object(Bucket=bucket, Key=key, Body=b"AirGuard S3 connection OK")
body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
assert body == b"AirGuard S3 connection OK"
s3.delete_object(Bucket=bucket, Key=key)
print("S3 write/read/delete passed")
'@ | python -
```

This test writes and removes one tiny object. It does not alter MLflow records or models.

### 5.4 Test MLflow metadata and artifact connectivity without training

```powershell
@'
import mlflow
import mlflow_utils as mu
mu.init_mlflow(require_remote_artifacts=True)
with mlflow.start_run(run_name="airguard-deployment-smoke-test"):
    mlflow.log_param("deployment_smoke_test", "true")
    mlflow.log_metric("connection_ok", 1.0)
    mlflow.log_text("AirGuard MLflow artifact upload passed.", "smoke-test/connection.txt")
print("MLflow PostgreSQL metadata + S3 artifact test passed")
'@ | python -
```

This creates a small test run and object, not a model fit. It also verifies the remote-artifact experiment can be created/selected. To inspect MLflow in a local browser:

```powershell
mlflow ui --backend-store-uri "$env:DATABASE_URL" --host 127.0.0.1 --port 5000
```

If PowerShell has not populated `$env:DATABASE_URL`, either set it for this terminal or let Python load `.env` (the CLI process itself does not automatically load the project `.env`). Do not expose an unauthenticated MLflow UI directly to the public internet.

**Important for this project:** a remote MLflow experiment alone does not prove the old registered aliases are portable. The new run/model versions must have S3-backed artifact locations. The code always sets `staging` to the new registered version, but only moves `production` when its OOF MAE is no worse than the current production alias. Your current `production` alias may therefore continue pointing at an old local artifact until a candidate is promoted. The normal dashboard does not use these aliases; it serves the joblib forecast CSVs produced by `predict_forecast.py`.

## 6. Configure GitHub Actions secrets and permissions

In GitHub, open **Repository → Settings → Secrets and variables → Actions → New repository secret**.

Create these secrets with the exact names:

| Secret | Used by | Notes |
|---|---|---|
| `DATABASE_URL` | Ingestion, database preflight/export, retrain due-check, training | Existing Neon direct SSL URL recommended for jobs/MLflow |
| `CPCB_API_KEY` | `ingest_cpcb.yml` | Personal data.gov.in API key |
| `CPCB_PROXY_URL` | `ingest_cpcb.yml` | Optional; only when using the CPCB relay |
| `FIRMS_MAP_KEY` | `ingest.yml` | NASA FIRMS key |
| `CDSAPI_URL` | `ingest.yml` | ADS URL: `https://ads.atmosphere.copernicus.eu/api` |
| `CDSAPI_KEY` | `ingest.yml` | Copernicus Atmosphere Data Store token, not the climate CDS token |
| `MLFLOW_ARTIFACT_URI` | `retrain_model.yml` | Private S3 bucket + prefix |
| `AWS_ACCESS_KEY_ID` | `retrain_model.yml` | Restricted IAM access key |
| `AWS_SECRET_ACCESS_KEY` | `retrain_model.yml` | Restricted IAM secret |
| `AWS_DEFAULT_REGION` | `retrain_model.yml` | For the example S3 bucket, `ap-south-1` |
| `MLFLOW_S3_ENDPOINT_URL` | `retrain_model.yml` | For the example S3 bucket, `https://s3.ap-south-1.amazonaws.com` |
| `MLFLOW_TRACKING_URI` | Optional | Leave unset to use `DATABASE_URL` |
| `MLFLOW_REGISTRY_URI` | Optional | Leave unset to use tracking URI |

Do not add the contents of `.env` as one secret. Add each key/value separately through the GitHub UI. If you use GitHub CLI, `gh secret set NAME` prompts for the value without putting it directly in the command history; repeat it for each required secret. For example:

```powershell
gh auth status
gh secret set DATABASE_URL
gh secret set CPCB_API_KEY
gh secret set FIRMS_MAP_KEY
gh secret set CDSAPI_URL
gh secret set CDSAPI_KEY
gh secret set MLFLOW_ARTIFACT_URI
gh secret set AWS_ACCESS_KEY_ID
gh secret set AWS_SECRET_ACCESS_KEY
gh secret set AWS_DEFAULT_REGION
gh secret set MLFLOW_S3_ENDPOINT_URL
```

Set `CPCB_PROXY_URL` only when you use that proxy. Set `MLFLOW_TRACKING_URI` and `MLFLOW_REGISTRY_URI` only when you want to override the workflow's fallback to `DATABASE_URL`.

GitHub Actions secrets are intended for credentials and are consumed in workflow expressions as `secrets.NAME`; see [GitHub Actions secrets](https://docs.github.com/en/code-security/reference/secret-security/secret-types).

Under **Settings → Actions → General → Workflow permissions**, ensure the repository allows read/write permissions for `GITHUB_TOKEN`, because the forecast/retrain workflows commit generated outputs. The workflow YAML already declares `contents: write` for those jobs. If `main` is protected, configure an allowed bot/PR flow rather than disabling branch protections blindly.

Do not add `CDS_CLIMATE_API_URL` or `CDS_CLIMATE_API_KEY` for ordinary live ingestion. Those are separate optional Climate Data Store credentials for an offline backfill, whereas the scheduled CAMS workflow uses the Atmosphere Data Store credentials.

## 7. Validate and run GitHub Actions in a controlled way

The workflows already run repeatedly. Do not manually launch all of them several times. Use one targeted validation run after the patch has been merged and secrets are saved. From a terminal, you can see workflow names and trigger a single run with GitHub CLI:

```powershell
gh workflow list
gh workflow run ingest_cpcb.yml --ref main
gh run list --workflow ingest_cpcb.yml --limit 5
# Then follow the run using its ID:
gh run watch <RUN_ID>
```

For the combined ingest workflow, use `gh workflow run ingest.yml --ref main` once and inspect it the same way. Use the GitHub UI instead if you prefer visual logs.

### 7.1 CPCB: one manual run

GitHub → **Actions → Ingest CPCB (frequent) → Run workflow**. Run it once on `main`.

Then query Neon (Neon SQL Editor or `psql`):

```sql
SELECT job, started_at, finished_at, rows_written, status, message
FROM ingestion_log
WHERE job = 'fetch_cpcb'
ORDER BY started_at DESC
LIMIT 5;

SELECT COUNT(*) AS total_rows,
       COUNT(value_avg) AS rows_with_numeric_average,
       MAX(observed_at) AS latest_observation
FROM cpcb_hourly;
```

The key acceptance condition is that new valid CPCB reads have non-NULL `value_avg`. Existing placeholder rows are intentionally retained and will still be counted in `total_rows`. If the new job fails with a diagnostic saying the mapped `pollutant_avg` values are all unusable, inspect the recorded response keys/sample fields and correct the upstream mapping. Do not delete the old rows to make the count look cleaner.

### 7.2 CAMS/Open-Meteo/FIRMS: one combined validation run

GitHub → **Actions → Ingest live data → Run workflow**. This runs the existing combined job once. The newer `fetch_cams.py` in the upload already uses the request format without the `grid` parameter; do not re-add `grid`.

Check:

```sql
SELECT job, started_at, finished_at, rows_written, status, message
FROM ingestion_log
WHERE job IN ('fetch_cams', 'fetch_openmeteo', 'fetch_firms')
ORDER BY started_at DESC
LIMIT 15;

SELECT run_time, COUNT(*) AS rows, MAX(valid_time) AS latest_valid_time
FROM cams_hourly
GROUP BY run_time
ORDER BY run_time DESC
LIMIT 5;
```

A zero-row FIRMS poll can be normal when no new fire detections are available. A fresh CAMS `run_time` and successful ingestion log row are the useful checks. Old HTTP 400 entries do not mean the current code is still sending the old request; inspect the newest run.

### 7.3 Avoid another backfill

The audit found repeated live writes to `cams_hourly`, `weather_surface_hourly`, `weather_pressure_hourly`, and `fire_detections_raw`, as well as already populated historical tables. Do not run the backfill scripts (`cds_backfill.py`, `backfill_openmeteo*.py`, `migrate_historical.py`) just to deploy the dashboard.

## 8. Refresh features and forecast without retraining

The next intended workflow is **Actions → Rebuild daily AirGuard forecast → Run workflow**. It does the DB preflight, exports historical/live data, rebuilds the daily fire features, builds the feature table, checks data quality and verifies the model artifacts before generating forecasts.

Before running it, verify your actual production models on the GitHub `main` branch:

```powershell
git ls-tree -r --name-only origin/main outputs/models | `
  Select-String '\.joblib$|oof_band\.csv|final_ensemble_weights_used\.json'
```

Then locally, after pulling the branch:

```powershell
python ingestion/check_model_artifacts.py
```

Expected: `Production model artifact check passed.`

### If the model check passes

Run the daily rebuild once. It uses your existing joblib models; it does not call `.fit()` or retrain. Verify that forecast CSVs update and `outputs/demo_data_marker.json` is removed only after a real prediction run completes. Check the latest `forecast_date` in `outputs/forecast_10d_city_final.csv` and the API health response after deployment.

### If the model check fails

The provided ZIP is missing the joblib files and uncertainty bands. Check your prior trained local working directory and the GitHub branch from the successful training runs. If the complete artifacts are in your old local project, copy only that model directory into the current clone:

```powershell
$OldModels = "C:\path\to\previous-airguard-project\outputs\models"
robocopy $OldModels ".\outputs\models" /E
python ingestion/check_model_artifacts.py
```

`robocopy` exit codes 0–7 generally mean success or success with differences; 8+ indicates a copy failure. Verify that all target folders have member/horizon `.joblib` files and `oof_band.csv`, plus `outputs/models/final_ensemble_weights_used.json`.

Before committing, check file sizes and ensure no model file exceeds GitHub's 100 MB single-file limit. If your artifacts are too large, the current workflow needs an explicit S3 download step before `predict_forecast.py`; do not just upload an oversized model or force-push. A Git LFS migration also requires changing the workflow so LFS objects are fetched on the runner.

### When to retrain

You said the training Action has already run multiple times. Do not select `force=true` during ordinary deployment. Only run one deliberate retrain if all of these are true:

1. the current joblib artifacts cannot be recovered from your existing local/GitHub training outputs, or you specifically need a fresh remote-artifact MLflow model version;
2. database and feature preflights pass;
3. CPCB/CAMS source validations are complete enough for the training window;
4. S3 write/read smoke test passes; and
5. the GitHub S3/Neon secrets are present.

Why this matters: the audit's registered versions use local `file:///C:/...` artifact paths. A successful retrain with S3 configured should create/use the `-remote-artifacts` experiment and publish new S3-backed model versions. It is computationally expensive and is not required to serve normal dashboard forecasts if the complete joblib models are already present. The existing fixed-parameter policy remains in force; do not run offline tuning in Actions.

## 9. Deploy the FastAPI backend to Render

The backend is intentionally lightweight and read-only: it serves the latest CSV/JSON outputs in the repository. Only `/api/ingestion/*` needs `DATABASE_URL`. The API does not need XGBoost/LightGBM/MLflow to start.

Create a Render **Web Service** connected to the GitHub repository and the `main` branch. Use these settings:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Root Directory | Leave blank (repository root) |
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/api/health` |
| Auto Deploy | On Commit (or After CI Checks Pass, if CI is configured) |

Set the environment variables:

```text
DATABASE_URL=<same Neon SSL URL>
AIRGUARD_CORS_ORIGINS=https://<your-vercel-project>.vercel.app
```

Do not set `PYTHON_VERSION=3.11` as an environment variable: Render requires a fully qualified patch version in that variable. The repository-root `.python-version` file already pins the service to Python 3.11 and accepts a minor-only version. Only add `PYTHON_VERSION` if you deliberately choose a specific full version such as `3.11.x` after checking Render's currently supported patch releases. Add any custom frontend domain as another comma-separated origin. Do not put a trailing slash in the origin. Do not set Render's Root Directory to `backend`: `backend/app/config.py` needs its parent repository folders `outputs/`, `features/` and `params/`. Render's root-directory and build/start command behavior is described in its [deployment docs](https://render.com/docs/your-first-deploy); the FastAPI start pattern is also in [Render's FastAPI guide](https://render.com/docs/deploy-fastapi).

Enable automatic deploys from `main`. This is important because GitHub Actions commits refreshed forecast/features/model files to the repository; the Render checkout must receive those commits for the file-backed API to return new outputs. Render documents automatic on-commit deploy behavior [here](https://render.com/docs/deploys).

After deployment, test from PowerShell:

```powershell
$API = "https://<your-render-service>.onrender.com"
Invoke-RestMethod "$API/"
Invoke-RestMethod "$API/api/health"
Invoke-RestMethod "$API/api/stations"
Invoke-RestMethod "$API/api/targets"
Invoke-RestMethod "$API/api/forecast/city?target=AQI"
Invoke-RestMethod "$API/api/forecast/station/ITO?target=PM25"
Invoke-RestMethod "$API/api/history/city?target=AQI&days=30"
Invoke-RestMethod "$API/api/model/info"
Invoke-RestMethod "$API/api/ingestion/status/latest-per-job"
```

The ingestion-status endpoint returns 503 when the DB URL is missing/unreachable. Forecast endpoints return 503 if the needed forecast files are not present yet. Health `status: demo` means the demo marker remains; `status: degraded` means the forecast is stale or required artifacts/schema are incomplete. Do not call those production-ready.

## 10. Deploy the React/Vite frontend to Vercel

### Dashboard deployment

Import the same GitHub repository into Vercel. Use:

| Setting | Value |
|---|---|
| Framework preset | Vite |
| Root Directory | `frontend` |
| Install Command | `npm install` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Production branch | `main` |

Set this environment variable for Production (and Preview if desired):

```text
VITE_API_BASE_URL=https://<your-render-service>.onrender.com
```

This is the API origin only. Do not append `/api` or a trailing slash. Vite embeds `VITE_*` variables at build time, so redeploy after changing the API URL. Vercel documents the Vite framework and `VITE_` variables [here](https://vercel.com/docs/frameworks/frontend/vite).

Optional CLI path:

```powershell
npm install
npm run build
npm install -g vercel
vercel login
vercel link
vercel deploy
vercel deploy --prod
```

Run the CLI from `frontend/` so it links the correct Vercel project root. You can also import the repo via the Vercel dashboard, which is usually simpler for the initial setup. See [Vercel CLI deployment](https://vercel.com/docs/projects/deploy-from-cli).

### Finish CORS after you get the Vercel URL

Return to Render → Environment and update:

```text
AIRGUARD_CORS_ORIGINS=https://<exact-vercel-production-origin>
```

If you use both a Vercel domain and a custom domain, separate them with commas. Save and redeploy. In the browser, open Developer Tools → Network and confirm requests go to your Render API, not `localhost:8000`. The frontend source intentionally has no localhost fallback in production builds; a missing `VITE_API_BASE_URL` causes a clear API configuration error rather than silently calling the visitor's own computer.

## 11. End-to-end acceptance checks

Run these in sequence:

1. **GitHub:** merged patch is on `main`; no `.env`, credential, `mlruns/` or `mlflow.db` files are committed.
2. **Neon:** `python ingestion/validate_database.py` passes; historical counts remain non-zero.
3. **S3:** put/get/delete smoke test passes; bucket is private.
4. **MLflow:** metadata + text artifact smoke test passes; experiment artifact location begins with `s3://`.
5. **CPCB:** a fresh `fetch_cpcb` run logs valid non-NULL `value_avg` rows or reports a clear upstream mapping/response problem without inserting placeholders.
6. **CAMS:** latest `fetch_cams` log is successful and `cams_hourly` has a recent run.
7. **Model check:** `python ingestion/check_model_artifacts.py` passes against `main` before a real forecast run.
8. **Feature build:** `data_quality_check.py` passes freshness/schema checks during the daily rebuild.
9. **Forecast:** the rebuild writes fresh forecast CSVs; demo marker is removed only by a successful real prediction run.
10. **Backend:** `/api/health` returns `status: ok`; forecast and ingestion status routes return expected data.
11. **Frontend:** Vercel app loads, browser Network calls hit the Render origin, and there are no CORS/failed requests.
12. **Retraining:** no forced retrain has been triggered unnecessarily. When a new S3-backed retrain is intentionally run, check MLflow model run artifact URIs and aliases separately.

## 12. Normal operating schedule

The repository workflows already schedule recurring work:

| Workflow | Schedule | Function |
|---|---|---|
| `ingest_cpcb.yml` | Every 15 minutes | Poll CPCB current readings |
| `ingest.yml` | Every 3 hours | FIRMS, Open-Meteo and CAMS ingestion |
| `rebuild_forecast.yml` | Daily, 01:00 UTC (06:30 IST) | DB export, fire/features rebuild, forecast if joblib artifacts are complete |
| `retrain_model.yml` | Daily, 01:30 UTC (07:00 IST) | Checks the database retrain marker; fits fixed parameters only when due (30 days) or explicitly forced |

You do not need to manually run these workflows on every deployment. Scheduled GitHub Actions run only from the repository's default branch. Check the Actions tab when DB logs appear stale; the DB ingestion log alone cannot prove a scheduled workflow attempted after the audit snapshot.

## 13. Troubleshooting / connection map

| Symptom | Where it breaks | Check |
|---|---|---|
| `DATABASE_URL is not set` | Local/Actions/Render config | Exact variable name; root `.env` for local code; GitHub secret for Actions; Render variable for API |
| Neon connect timeout / SSL error | PostgreSQL network | Copy fresh Neon connection string; retain `sslmode=require`; use direct host for migrations/MLflow; check Neon compute/branch state |
| CPCB job green but still no values | API payload mapping | Inspect newest `fetch_cpcb` log message; patch fails visibly rather than inserting another all-NULL batch |
| PostgreSQL `CardinalityViolation` on CPCB upsert | Duplicate rows within one batch | The patch drops duplicate `(station, pollutant, observed_at, source)` keys before upsert |
| CPCB API 403/429/520–524 | data.gov.in/proxy/network | Check key/quota, current station names, and optional `CPCB_PROXY_URL`; retry schedule is every 15 minutes |
| CAMS 400 invalid request | ADS request/config | Verify deployed `fetch_cams.py` has no `grid`; confirm `CDSAPI_URL` points to ADS and token has dataset access |
| MLflow `file:///C:/...` artifact path | Old registered model version | Do not use old version on a hosted runner; configure S3 and write a new remote-backed run/model |
| `NoCredentialsError` / S3 AccessDenied | AWS IAM/S3 | Key pair, bucket region, endpoint, prefix and bucket-scoped IAM permissions |
| Model check `ready=false` | Missing joblib/OOF artifacts | Recover complete `outputs/models/` from the prior training workspace/repository; never create empty placeholders |
| Forecast rebuild skips predictions | Model artifact check | Expected safeguard if model files are incomplete; it may refresh features but will not generate new real forecasts |
| Backend has missing `outputs/` files | Render monorepo setup | Root Directory must be blank; start command changes into `backend/` while repo root remains checked out |
| Ingestion status endpoint gives 503 | Backend → Neon | Check Render `DATABASE_URL`, SSL and `ingestion_log` table; API intentionally masks raw DB errors from public responses |
| Browser says CORS blocked | Vercel → Render | `AIRGUARD_CORS_ORIGINS` must exactly match scheme + hostname of the live frontend |
| Frontend says API URL not configured | Vercel build-time env | Add `VITE_API_BASE_URL` and redeploy the frontend |
| `npm install` hangs/fails | Build environment / registry | Retry locally with working npm registry access; check proxy/firewall and Vercel build logs; don't commit `node_modules/` |

## 14. Security and rollback

- Never commit `.env`, `.cdsapirc`, AWS keys, Neon URLs, CPCB keys, FIRMS keys or ADS tokens.
- Keep the S3 artifact bucket private with public access blocked.
- Use a restricted IAM identity, not AWS root keys.
- Do not expose MLflow UI without authentication and network controls.
- Back up Neon and retain MLflow model-version objects in S3; do not delete the historical local-artifact versions from registry metadata during initial cutover.
- Keep `params/*.json` fixed unless a reviewed research change is approved.
- Do not force-push to `main` or rewrite existing Git history during deployment.
- For rollback, redeploy the last known-good Render/Vercel release and revert the specific Git commit through a normal revert PR. Do not delete live database rows as a rollback mechanism.
