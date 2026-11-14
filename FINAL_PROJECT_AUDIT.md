# AirGuard — final project audit

**Audit date:** 12 November 2026  
**Audited input:** `AirGuard-updated-project-with-actions.zip`  
**Scope:** four GitHub Actions workflows, Python ingestion/training/forecasting, FastAPI backend, React/Vite frontend, configuration, persistence and model artifacts.

## Executive result

The project source has been checked and several concrete code and deployment defects were fixed. All four supplied workflows are retained. Routine production retraining continues to use the fixed files in `params/`; no parameter-tuning workflow was added. Full parameter search remains a local-only operation through `tune_parameters_offline.py`.

**This is a source-code audit, not a live production acceptance test.** No production PostgreSQL connection, Cloudflare R2 bucket, CPCB/FIRMS/Copernicus credentials, Render deployment or Vercel deployment was supplied. Those external connections cannot be certified from a ZIP archive.

## Changes applied

1. **Local environment loading:** root `.env` is loaded by ingestion and backend configuration with `override=False`, so hosted environment variables remain authoritative.
2. **Copernicus endpoint split:** scheduled CAMS uses the Atmosphere Data Store (ADS) endpoint and credentials. The separate Climate Data Store credentials remain separate for the offline ERA5 pressure-level backfill.
3. **Windows CAMS compatibility:** CAMS downloads use the platform temporary directory rather than a hard-coded `/tmp` path.
4. **PostgreSQL URL compatibility:** common provider URL schemes (`postgres://`, `postgresql+psycopg2://`, and `postgresql+psycopg://`) are normalized before connecting.
5. **First-run retraining gate:** the 30-day due-check initializes the database schema before reading `ingestion_log`, so a new database does not fail on a missing table.
6. **Reliable retrain marker:** the training script now imports `os`; successful retraining on GitHub Actions requires the `retrain_models` completion row to be written. The general ingestion logger remains best-effort for ordinary fetch jobs, but supports strict failure for this marker.
7. **Persistent MLflow:** PostgreSQL is used for run/registry metadata and an S3-compatible bucket for model artifacts. The R2 S3 endpoint is now validated for GitHub Actions, and the R2 region defaults to `auto`.
8. **Portable registered models:** the `AirGuardTargetModel` source module is packaged with MLflow `code_paths`; the logged model environment pins the exact installed versions of the relevant numerical/model dependencies.
9. **Daily model compatibility:** the forecast workflow installs model runtime versions recorded in `outputs/models/trained_at.json`; older manifests use a bounded fallback with a warning.
10. **Database/data preflight:** forecast and retraining workflows validate the five required historical tables before exporting/model building. The historical migration script validates input files before writing to the database.
11. **Model-artifact preflight:** daily forecasting checks the complete expected `.joblib`, OOF band and ensemble-weight artifacts. If model binaries are missing, the daily job reports a clear skip rather than running prediction against an incomplete model set.
12. **Forecast/demo state:** successful real forecasts commit deletion of `outputs/demo_data_marker.json`, so the API can stop identifying the data as demo after a complete real forecast is produced.
13. **Workflow coordination:** daily forecast and 30-day retraining share a concurrency group to prevent competing auto-commits to the same production repository files.
14. **Frontend connection handling:** production builds no longer fall back to a visitor's own `localhost:8000`. Production requires `VITE_API_BASE_URL`; local Vite development retains the localhost default.
15. **API and configuration checks:** backend health checks validate forecast layout, date/horizon coverage, finite bounds, parameter JSONs, feature schema and model artifact presence. Ingestion-status DB failures return a clear 503 response rather than an unhandled error.
16. **Git hygiene:** local MLflow state and secrets remain ignored; the production `.joblib` model files are not ignored. Full OOF diagnostic CSVs are excluded from Git because they are already logged as MLflow artifacts.

## GitHub Actions retained

| Workflow | Schedule (UTC) | Function |
|---|---:|---|
| `ingest.yml` | `15 */3 * * *` | FIRMS, Open-Meteo and CAMS ingestion |
| `ingest_cpcb.yml` | `*/15 * * * *` | CPCB polling every 15 minutes |
| `rebuild_forecast.yml` | `0 1 * * *` | Daily database export, features, quality checks and joblib forecast |
| `retrain_model.yml` | `30 1 * * *` | Daily due-check; full refit only when 30 days are elapsed or manually forced |

The daily retraining workflow reads the committed `params/best_params_*.json` and `params/ensemble_weights_*.json` files. It does not tune or rewrite those files. Manual `force` means “refit now with the same fixed parameters”, not “tune parameters”. The only full-tuning entry point is `tune_parameters_offline.py` on a developer machine; it writes preview outputs and cannot overwrite production parameters.

## Verification performed

| Check | Result | Detail |
|---|---|---|
| Python syntax | PASS | 48 Python files parsed; zero syntax errors. |
| JSON syntax | PASS | All project JSON files parsed. |
| Workflow YAML | PASS | All four workflow files parsed and each contains one job. |
| Workflow command paths | PASS | Every `python ...py` command referenced by the four workflows resolves to a file using the workflow working directory. |
| Parameter JSON coverage | PASS | Five targets; fixed parameter and weight files parsed; horizons 1–10 and ensemble weight consistency validated in the audit. |
| DB logger strictness | PASS | A synthetic database-write failure is re-raised when `raise_on_error=True`; ordinary ingestion logging remains best-effort. |
| Backend API smoke tests | PASS | Root, health, stations, targets, forecast, history and model routes returned HTTP 200 against bundled sample/demo output. |
| Ingestion status route | EXPECTED 503 | No `DATABASE_URL` was supplied in the audit runtime; the route reports the DB dependency as unavailable. |
| Frontend JS/JSX syntax | PASS | TypeScript transpiler parsed all four frontend JS/JSX files; zero syntax errors. |
| Frontend production build | NOT VERIFIED | `npm install` timed out in this environment; dependencies could not be installed, so `npm run build` was not run. No package lock was generated. |
| Live PostgreSQL / R2 / MLflow | NOT VERIFIED | No private service credentials or live service access were provided. MLflow is not installed in the audit runtime. |
| Full train/forecast run | NOT RUN | The source archive does not contain the actual trained model files or all historical input/schema outputs. |

## Remaining project data/deployment blockers

The uploaded ZIP is a source snapshot and **does not contain the production `.joblib` binaries**. In this archive, `outputs/models/` contains only `trained_at.json`; it contains zero `.joblib` models, no per-target `oof_band.csv`, and no `outputs/models/final_ensemble_weights_used.json`. The artifact checker therefore correctly reports that the forecast is not ready and the daily forecast job will skip model prediction.

The archive also does not contain:

- `final_delhi_dataset.csv` (intentionally gitignored local export from historical station data);
- `features/feature_list.txt` and `features/targets.json` (generated by `build_features.py`);
- the per-target model files needed by `predict_forecast.py`.

The upload does contain `outputs/demo_data_marker.json`. It marks the currently bundled forecast CSVs as synthetic demo output; the forecast CSVs end on **4 November 2026**. Do not present these forecast values as live production forecasts. The marker is removed only after `predict_forecast.py` completes a successful real forecast, and the workflow now commits that deletion.

**Do not replace your populated local project with the complete ZIP from this audit.** Apply the audit patch over your existing project so your local `.joblib` files and generated training data are not overwritten or lost. Then verify that the production model files are committed or are downloaded by the workflows from persistent storage. If any model file is over GitHub's 100 MB hard limit, put it in Git LFS or object storage instead of ordinary Git.

## Required secrets and service configuration

### GitHub Actions repository secrets

- `DATABASE_URL`: PostgreSQL connection URL. Use the SSL-enabled URL provided by your database provider.
- `FIRMS_MAP_KEY`: NASA FIRMS map key.
- `CDSAPI_URL`: `https://ads.atmosphere.copernicus.eu/api` for scheduled CAMS ingestion.
- `CDSAPI_KEY`: Copernicus ADS API token.
- `CPCB_API_KEY`: data.gov.in key.
- `CPCB_PROXY_URL`: optional deployed proxy URL if CPCB blocks GitHub-hosted runner requests.
- `MLFLOW_ARTIFACT_URI`: an existing bucket/prefix, e.g. `s3://<bucket>/airguard/mlflow-artifacts`.
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: R2 S3 credentials.
- `AWS_DEFAULT_REGION`: `auto` for Cloudflare R2.
- `MLFLOW_S3_ENDPOINT_URL`: `https://<account-id>.r2.cloudflarestorage.com`.
- `MLFLOW_TRACKING_URI` and `MLFLOW_REGISTRY_URI`: optional only if using a separate persistent tracking/registry backend; otherwise both fall back to `DATABASE_URL`.

Use `CDS_CLIMATE_API_URL` / `CDS_CLIMATE_API_KEY` only for the offline Climate Data Store ERA5 pressure-level backfill. Do not reuse those credentials for the scheduled CAMS ADS request.

### Hosting environment variables

- FastAPI host: `DATABASE_URL`, `AIRGUARD_CORS_ORIGINS` (the exact deployed frontend origin, including `https://`, no trailing slash), and any provider credentials used by services launched on that host.
- Vercel frontend: `VITE_API_BASE_URL` set to the public FastAPI origin **at build time**.
- MLflow registry/artifact clients: PostgreSQL tracking URI and the R2 bucket credentials/endpoint must be available to the client that loads registered models.

## Final acceptance sequence

1. Apply the patch over the existing local repo; keep `.joblib`, fixed parameters and historical source data.
2. Ensure the old `research_retrain.yml` tuning workflow is absent from GitHub. Do not delete any of the four supplied workflows.
3. Configure Neon/PostgreSQL and R2 secrets listed above.
4. Run one ingestion workflow manually; verify new rows appear in the five historical/live tables and `ingestion_log`.
5. Migrate historical station/ERA5/CAMS CSV data once if the five required historical tables are not yet populated.
6. Run `rebuild_forecast.yml`; verify features build, model artifact check is `ready=true`, and the demo marker is deleted only after real predictions are written.
7. Trigger `retrain_model.yml` manually with `force=true` only after DB, feature data and R2 are ready. Verify MLflow parent/target runs, registered versions, artifact URIs and `ingestion_log` success marker.
8. Deploy FastAPI; verify `/api/health` and `/api/ingestion/status/latest-per-job` from the hosted backend.
9. Deploy Vite frontend with `VITE_API_BASE_URL`; verify browser network requests reach the backend and CORS permits the deployed origin.
10. Wait for/trigger daily forecast after the newly trained models are committed or restored; verify forecast origin/freshness and health status `ok`.

