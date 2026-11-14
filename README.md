# AirGuard — Delhi NCR air-quality forecasting

AirGuard builds 10-day daily forecasts for AQI, PM2.5, ozone, ozone 8-hour maximum and NOx across five Delhi monitoring stations. It combines station observations with ERA5/CAMS atmospheric context and FIRMS fire indicators.

This repository contains the Python ingestion and modeling pipeline, a read-only FastAPI backend, a React/Vite dashboard, persistent MLflow tracking/registry integration, and four GitHub Actions workflows.

## Start here

- **[Production deployment runbook (existing Neon data + prior training/ingestion)](DEPLOYMENT_RUNBOOK.md)** — recommended order for this deployment; avoids repeating historical migrations and retraining.
- **[Patch merge and deployment audit](PATCH_MERGE_REPORT.md)** — merged-file decisions, test results and remaining blockers.
- **[Development and deployment reference](DEVELOPMENT_AND_DEPLOYMENT_GUIDE.md)** — full Windows setup, PostgreSQL/Neon, R2, historical migration, GitHub Actions, MLflow, Render and Vercel deployment.
- **[MLflow guide](MLFLOW_GUIDE.md)** — PostgreSQL tracking/registry, S3-compatible artifact storage and model aliases.
- **[Project data status](PROJECT_DATA_STATUS.md)** — important note about model binaries, demo outputs and the historical data bundle.
- **[Final verification](FINAL_VERIFICATION.md)** — checks that passed, blockers and what still needs a credentialed deployment test.
- **[Final project audit](FINAL_PROJECT_AUDIT.md)** — source-level fixes and verification boundaries.

## Architecture

| Area | Location |
|---|---|
| Forecast feature generation | `build_features.py`, `features/` |
| Live and historical ingestion | `ingestion/` |
| Fixed production hyperparameters/weights | `params/` |
| Joblib production retraining | `train_production_models.py` |
| Daily joblib inference | `predict_forecast.py` |
| Optional MLflow registry inference | `predict_forecast_mlflow.py` |
| MLflow integration | `mlflow_utils.py`, `MLFLOW_GUIDE.md` |
| FastAPI API | `backend/app/` |
| React/Vite frontend | `frontend/` |
| GitHub Actions | `.github/workflows/` |

## Local development (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r ingestion/requirements.txt
python -m pip install -r backend/requirements.txt
Copy-Item .env.example .env
```

Run the API in one terminal:

```powershell
Set-Location backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Run the frontend in another terminal:

```powershell
Set-Location frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:5173`; FastAPI docs are at `http://localhost:8000/docs`.

## Production workflows

1. `ingest.yml` — FIRMS, Open-Meteo and CAMS every three hours.
2. `ingest_cpcb.yml` — CPCB polling every 15 minutes.
3. `rebuild_forecast.yml` — daily feature/forecast refresh using existing `.joblib` models.
4. `retrain_model.yml` — daily 30-day due-check; when due, refit with the fixed committed parameter JSONs and log to MLflow.

There is no automated hyperparameter-tuning workflow. `tune_parameters_offline.py` is local-only and writes preview candidates under `outputs/parameter_tuning_preview/`; it never overwrites `params/`.

## Important data/model note

The final source bundle intentionally excludes private credentials and local MLflow state. The source upload used to assemble this package also omitted production `.joblib` model binaries; restore your existing `outputs/models/` directory from your local copy before running real forecasts. The bundled forecast files have a demo marker, and the bundled historical data ends on 2025-12-31. Follow the guide to migrate history, configure live ingestion and generate fresh production forecasts before using the dashboard operationally.
