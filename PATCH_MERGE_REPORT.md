# AirGuard patch merge and deployment audit

**Merged package date:** 13 November 2026  
**Input sources:** `AirGuard-Final-Project.zip`, `AirGuard-Neon-Audit-Patch.zip`, and `AirGuard-Neon-Database-Audit.md`.

## Patch merge decisions

| File | Decision | Result |
|---|---|---|
| `ingestion/fetch_cpcb.py` | Merged | Stable empty schema, finite numeric values only, safe response diagnostic, no-record handling, rejects all-invalid mapped payloads, drops missing averages, deduplicates the composite upsert key, and logs partial coverage |
| `ingestion/export_for_modeling.py` | Merged | Live CPCB query now filters out `value_avg IS NULL` |
| `.github/workflows/retrain_model.yml` | Kept the existing project version | The patch ZIP's version removed database preflight and changed the shared `airguard-production-repository-writes` concurrency group. Those are not safe changes for this project's paired daily rebuild/retrain jobs. The checked-in version already has the `auto` fallback for R2 |
| `ingestion/fetch_cams.py` | Kept existing project version | It already omits `grid` and separates `data_format=netcdf` from `download_format=zip` |

The patch is non-destructive: it does not delete existing CPCB NULL rows, alter Neon schema/data, change historical data, modify fixed parameters, or launch training.

## Additional deployment improvements

- Added `.python-version` (`3.11`) so Render does not silently select a newer Python runtime than the one used by the project workflows.
- Added `DEPLOYMENT_RUNBOOK.md`, tailored to the already-populated Neon database and already-run ingestion/training workflows.
- Added five unit tests for CPCB parsing safeguards.
- Updated README links and `.env.example` AWS S3 endpoint guidance.

## Checks run in this package workspace

| Check | Result |
|---|---|
| Python compileall for ingestion/backend/tests | PASS |
| CPCB unit tests | PASS (5 tests; no network/database needed) |
| JSON parsing | PASS (16 files) |
| GitHub Actions YAML parsing | PASS (4 workflows) |
| JavaScript syntax (`api.js`, `aqi.js`, `vite.config.js`) | PASS |
| FastAPI TestClient smoke | PASS: 12 file-backed routes returned 200 |
| `/api/ingestion/status/latest-per-job` without a configured DB URL | Expected 503 |
| `ingestion/check_model_artifacts.py` | NOT READY: 206 required model/band/weight file checks are missing in the uploaded ZIP |
| `/api/health` | `status=demo`; marker exists; forecast origin is 2026-10-25; model artifacts are not ready |
| Frontend `npm install` / production build | NOT VERIFIED: package installation timed out in this environment; no frontend source changes were made |
| Live Neon connection from this workspace | NOT re-tested: no `DATABASE_URL` secret was provided. The attached Nov 12 audit did establish successful read-only Neon connectivity |
| Live AWS S3 / MLflow artifact test | NOT run: AWS credentials were not provided |
| Live CPCB/CAMS API requests | NOT run: API credentials were not provided |

## Deployment blockers to resolve

1. Check the actual GitHub `main` branch for the trained `.joblib` models. The uploaded ZIP does not contain them. Restore the complete existing model directory before running the daily forecast workflow. Do not force retraining merely to compensate for a source ZIP packaging omission.
2. Configure the private AWS S3 bucket and the `MLFLOW_*`/AWS secrets before expecting new MLflow model artifacts to be portable. Existing registered versions still point to the Windows-local artifact store.
3. Run one CPCB workflow after merging the patch and verify fresh numeric `value_avg` values. Existing placeholder NULL rows remain in Neon by design.
4. Run one combined live ingestion workflow and inspect the newest CAMS log row to confirm the updated no-`grid` request succeeds.
5. Only after model artifacts are verified should the real forecast rebuild be treated as ready. The currently packaged frontend can render the dashboard, but the forecast outputs are marked demo/stale.
