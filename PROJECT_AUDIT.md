# AirGuard project audit — 12 November 2026

## Changes in this delivery

- Added persistent MLflow tracking and registry configuration: PostgreSQL for run/registry metadata and S3-compatible storage for model artifacts.
- Kept the normal daily forecast path on committed `outputs/models/**/*.joblib` files. It does not depend on MLflow availability.
- Kept 30-day production retraining fixed-parameter-only. It reads the committed `params/*.json` files and does not write to them.
- Removed the scheduled full-tuning workflow from this delivery. The historical GitHub repository's `.github/workflows/research_retrain.yml` must also be removed from the user's checkout if it still exists.
- Added `tune_parameters_offline.py` as a local-only research utility. It writes candidate files and a score report under `outputs/parameter_tuning_preview/`; there is no apply/overwrite option.
- Set `RUN_TUNING=False` and `RUN_SHAP=False` in the research notebook defaults.
- Reworked the React frontend into Overview, Forecast, Stations, Model health and Methodology views, with responsive navigation and updated visual design.
- Added production requirements, environment examples, Git ignore rules and setup documentation.

## Checks completed

| Check | Result | Notes |
|---|---|---|
| Python syntax compilation | PASS | `compileall` completed for project Python files. |
| Parameter JSON parsing | PASS | All five `best_params_*.json` and five `ensemble_weights_*.json` parse. |
| Production parameter coverage | PASS | Each model has horizons 1–10; estimator `set_params` accepted each parameter dictionary. |
| Ensemble weight validation | PASS | All five targets include horizons 1–10; every horizon's non-negative weights sum to 1 within tolerance. |
| Workflow YAML parsing | PASS | Both delivered workflows parse as YAML mappings with jobs. |
| Notebook JSON parsing | PASS | 51 cells; tuning and SHAP defaults are false. |
| Offline tuner CLI | PASS | `python tune_parameters_offline.py --help` runs without importing unavailable feature schema files. |
| Backend API smoke tests | PASS | Health, stations, targets, city forecast, latest station forecast, city history and model info endpoints returned HTTP 200 against bundled demo outputs. |
| Frontend JSX/JS parse | PASS | TypeScript transpiler parsed `App.jsx`, `api.js`, `aqi.js` and `main.jsx`. |
| Frontend production build | NOT VERIFIED | npm dependency installation timed out in this environment; `npm run build` could not be run. |
| MLflow end-to-end tracking/registry | NOT VERIFIED | MLflow is not installed in the audit runtime, and no PostgreSQL or S3/R2 credentials were supplied. Configuration and code are present, but the first real remote run must validate secrets, DB connectivity and bucket permissions. |
| Full production train / offline tune | NOT RUN | The uploaded archive contains no production `.joblib` model files, no `final_delhi_dataset.csv`, and no generated `features/targets.json` or `features/feature_list.txt`. Generate features from the real source data before running training or tuning. |

## Important deployment setup still required

Configure GitHub Actions repository secrets before the 30-day retraining job runs:

- `DATABASE_URL` — persistent PostgreSQL URL.
- `MLFLOW_ARTIFACT_URI` — S3 bucket prefix, e.g. `s3://bucket/airguard/mlflow-artifacts`.
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`.
- `MLFLOW_S3_ENDPOINT_URL` when using R2 or MinIO.
- Existing ingestion provider secrets such as CPCB, FIRMS and CDS credentials.

The GitHub-hosted runner is ephemeral. PostgreSQL stores run metadata and registry entries; S3-compatible storage retains the pyfunc model bundles and other MLflow artifacts. A local `mlruns/` folder is not the production artifact store.

## Protect existing local production artifacts

The uploaded source ZIP contains zero `.joblib` files under `outputs/models/`. The updated package therefore does not replace or recreate the models that are present in the user's local checkout. Preserve/copy the existing local `outputs/models/**/*.joblib` files and their current `trained_at.json` when applying this update. Do not delete the local `mlruns/` folder as a side effect of applying the patch; this delivery only configures Git to ignore local MLflow stores.

## Applying to an existing Git checkout

1. Back up the working folder and current `outputs/models/` first.
2. Prefer `AirGuard-safe-update-patch.zip`; it contains only changed/new source, frontend, requirements, documentation and workflow files (no model/data/output files).
3. If `.github/workflows/research_retrain.yml` exists in the checkout, remove it with `git rm .github/workflows/research_retrain.yml`.
4. Keep any unrelated ingestion workflow such as `ingest.yml` unchanged.
5. Configure the MLflow secrets above before triggering a production retrain.
6. Run local checks and inspect `git status` before committing or pushing. Do not force-push rewritten history without reconciling the remote branch.
