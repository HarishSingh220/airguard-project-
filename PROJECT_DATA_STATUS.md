# AirGuard package data and artifact status

This package was assembled from `AirGuard-updated-project-with-actions.zip` and overlaid with the final audit patch. It also includes a repaired historical preprocessing notebook, a generated historical station CSV, freshly built feature metadata, and the updated development/deployment guide.

## Included

- Python ingestion, feature engineering, model training, forecasting and MLflow source.
- FastAPI backend and React/Vite frontend.
- The four supplied GitHub Actions workflows: `ingest.yml`, `ingest_cpcb.yml`, `rebuild_forecast.yml`, `retrain_model.yml`.
- All five fixed production target parameter JSONs and five fixed ensemble weight JSONs.
- Historical input CSVs under `aqi_data_delhi/`, ERA5/CAMS/FIRMS files, generated `final_delhi_dataset.csv`, and generated `features/` metadata.
- `.env.example`, `.gitignore`, MLflow guide and final verification report.

## Not included / must be restored from your local machine or hosted services

- **Production `.joblib` binaries**: the upload used to assemble this package did not contain any `*.joblib` files. Restore your existing complete `outputs/models/` directory before expecting the normal forecast workflow to run. Run `python ingestion/check_model_artifacts.py` and require the `Production model artifact check passed.` message.
- **Local `.env` and all API/database credentials**: intentionally excluded for security. Configure your own `.env` and GitHub/provider secrets.
- **Local `mlruns/` and `mlflow.db`**: intentionally excluded. Production MLflow persistence is PostgreSQL for tracking/registry metadata plus private S3-compatible object storage for artifacts.

## Demo and freshness status

The bundle includes `outputs/demo_data_marker.json`. Until a complete real forecast has run, the API reports demo data. Do not present the bundled forecast CSVs as live forecasts.

The historical station dataset ends on 2025-12-31. Building features from the bundled historical inputs succeeds and produces 7,305 station-day rows and 216 model features, but `ingestion/data_quality_check.py` correctly reports stale data until the live PostgreSQL ingestion/export pipeline extends the feature table to the current period.

## Recreate local data

The final cell in `data_preprosessing.ipynb` now validates and exports `final_delhi_dataset.csv` from the 20 CSVs under `aqi_data_delhi/`:

```powershell
jupyter nbconvert --to notebook --execute --inplace data_preprosessing.ipynb
python build_features.py
```

`final_delhi_dataset.csv` is gitignored on purpose. It is included in this ZIP as a local migration input but should not be committed to Git unless you intentionally change the repository data policy.
