# AirGuard API

A thin, **read-only** FastAPI layer over the files the pipeline already produces
(`outputs/`, `features/`, `params/`). It never calls `.fit()` or `.predict()` itself --
see `app/data_loader.py`'s docstring for why it deliberately avoids importing
`model_common.py`.

## Run it

```bash
cd backend
pip install -r requirements.txt

# Optional: generate fake-but-shaped sample data so you have something to look
# at before the real pipeline / GitHub Actions have run even once. Run from
# the REPO ROOT, not from backend/:
cd ..
python backend/dev/generate_sample_data.py
cd backend

uvicorn app.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

## Endpoints

| Method | Path | What it returns |
|---|---|---|
| GET | `/api/health` | Whether `outputs/`/`features/` exist yet, and when the forecast was last generated |
| GET | `/api/stations` | The 5 stations with lat/lon |
| GET | `/api/targets` | AQI, PM25, O3, O3_8H, NOx -- label + unit |
| GET | `/api/forecast/city?target=AQI` | 10-day Delhi-wide forecast (mean across stations); omit `target` for all 5 |
| GET | `/api/forecast/city/aqi-category` | City AQI forecast with CPCB category + color |
| GET | `/api/forecast/station/{station}?target=AQI` | 10-day forecast for one station |
| GET | `/api/forecast/stations/latest?target=AQI` | Tomorrow's (horizon=1) prediction for every station in one call |
| GET | `/api/forecast/all` | The full station x target x horizon forecast table, unfiltered |
| GET | `/api/history/station/{station}?target=AQI&days=30` | Recent actual (observed) daily values for one station |
| GET | `/api/history/city?target=AQI&days=30` | Recent actual Delhi-wide daily values |
| GET | `/api/model/weights?target=AQI` | Ensemble member weights actually used for the current forecast |
| GET | `/api/model/info` | When models were last retrained and up to what date |
| GET | `/api/ingestion/status?job=fetch_cpcb` | Recent `ingestion_log` rows (requires `DATABASE_URL`; 503 otherwise) |
| GET | `/api/ingestion/status/latest-per-job` | Most recent run of every ingestion job at a glance |

Every endpoint above (except `/api/ingestion/*`) reads a file already committed by
`rebuild_forecast.yml` or `retrain_model.yml` -- the API needs no database and no
ML libraries (xgboost/lightgbm/sklearn) to run. If a file doesn't exist yet, the
endpoint returns `503` with a message naming the workflow that produces it, not a
crash.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `AIRGUARD_REPO_ROOT` | two directories up from this file | Where to find `outputs/`, `features/`, `params/` |
| `AIRGUARD_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated list of allowed frontend origins |
| `DATABASE_URL` | unset | Same Postgres URL the ingestion scripts use; only needed for `/api/ingestion/*` |

## Deploying

This is a stateless process reading local files -- deploy it anywhere that can run
`uvicorn`/`gunicorn` and has the repo (with its `outputs/`/`features/`) checked out,
e.g. a small VM or container that pulls the repo on a schedule, or runs as a sidecar
in the same CI artifact as `rebuild_forecast.yml`. There's no `research_retrain.yml`
job anymore (removed), so don't expect `params/`/`ensemble_weights_*.json` to update
automatically -- re-tune manually (see the repo README) when you want fresh ones.
