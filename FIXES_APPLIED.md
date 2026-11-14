# Fixes applied (per the review doc)

All 9 issues in the review were verified against the actual code before fixing.
Nothing below is a guess -- each bug was located and reproduced in the code as shipped.

## P0 · Fix first
**1. `ingestion/db.py` — unquoted identifiers + NULL handling**
- `upsert_dataframe()` now quotes every table/column name it emits (`_quote_ident`),
  so mixed-case columns like `"PM25"`/`"NOx"` in `station_hourly_hist` are no longer
  silently folded to the wrong case.
- `df.where(pd.notnull(df), None)` replaced with `df.astype(object).where(...)` --
  the old version cast straight back to `float64` and reverted `None` to `NaN` before
  `to_sql()` ever saw it, so missing values were never actually written as SQL `NULL`.

## P0 · Silent data loss
**2. `ingestion/fetch_cpcb.py` — NOx mapping**
- `POLLUTANT_MAP` key fixed from `"NOx"` to `"NOX"` (the lookup always `.upper()`s
  first, so the old key could never match). Added an assertion that every key is
  upper-case so this can't regress silently again.
- Missing-station coverage is now logged as `"ok (partial)"` with the missing station
  list in `ingestion_log.message`, instead of always logging `"ok"`.

## P0 · Feature inconsistency
**3. Open-Meteo wind units**
- Added `wind_speed_unit=ms` to every Open-Meteo request (`fetch_openmeteo.py`,
  `backfill_openmeteo.py`, `backfill_openmeteo_12h_grid.py`).
- Each now also checks the returned `hourly_units` and raises if a wind-speed unit
  other than m/s ever comes back, instead of assuming.

## P0 · Export failure risk / P0 · Backfill interaction
**4 & 5. Duplicate timestamps / sparse pressure-level rows in `export_for_modeling.py`**
- Added `SOURCE_PRIORITY` (explicit provider-preference order per table) and
  `canonicalize()`, which collapses every weather/CAMS query to exactly one row per
  key, preferring the highest-priority source per *column* (back-filling only the
  columns a higher-priority source left NULL, e.g. CDS's `specific_humidity`/
  `vertical_velocity` filled into the Open-Meteo meteorological row) rather than
  picking one row per key at random via `sort + drop_duplicates(keep="last")`.
- Applied before every pivot/merge that used to be vulnerable to the
  `ValueError: Index contains duplicate entries` failure mode, and before the
  final historical/live concat in every `export_*` function.

## P1 · Quality gate
**6. `ingestion/data_quality_check.py` — stale file treated as live**
- Added a staleness check: if the feature file's own last date is more than
  `MAX_STALENESS_DAYS` behind today, the check now fails clearly with
  *"this file has no real live tail"* instead of silently measuring the stale
  tail's known historical missingness and calling it a new failure.
- Per-feature failures now require live-tail missingness to exceed **both** the
  absolute ceiling **and** the feature's own historical missingness by
  `BASELINE_MARGIN` -- so a feature whose live-tail gap simply matches its
  lifelong historical gap (confirmed: `d_NH3`, `d_SO2`, `d_temperature`, etc. in
  the included dataset) no longer fails the build.

## P1 · Credentials and failure reporting
**7.**
- `config.py`: removed the hardcoded shared CPCB sample-key fallback;
  `fetch_cpcb.py` now raises a clear `RuntimeError` at startup if `CPCB_API_KEY`
  isn't set, instead of silently using the shared key.
- `cds_backfill.py`: `--url`/`--key` CLI args are now actually read and passed
  into `cdsapi.Client(...)` when given (previously accepted nowhere, despite the
  docstring/config.py telling users to pass them).
- `backfill_openmeteo.py`, `backfill_openmeteo_12h_grid.py`,
  `backfill_openmeteo_airquality.py`: now track failed chunks and `sys.exit(1)`
  if any chunk failed, instead of always exiting 0.
- `fetch_firms.py`: when *every* FIRMS source fails (not just some), it now
  raises instead of only logging `"ok (partial)"` -- so `run_all.py`'s
  per-job `try/except` (which only reacts to raised exceptions) correctly marks
  the step failed instead of treating a zero-detections run as complete.

## Two further integration issues
**8. CAMS forecast mislabeled as "observed"**
- `cams_hourly` gets two new columns: `is_best_estimate` and `lead_time_hours`
  (migration-safe `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` added to `schema.sql`
  for already-existing databases).
- `is_forecast` is now **always `true`** for every CAMS-sourced row (it's a
  forecast product line, full stop) in `fetch_cams.py`, `cds_backfill.py`'s
  `load_cams_no2_o3()`, and `backfill_openmeteo_airquality.py` -- no longer
  derived from "has valid_time now passed".
- `is_best_estimate` (short-lead, closest to "now" at fetch time) replaces the
  old `is_forecast = false` check in `export_for_modeling.py`'s
  `export_cams()`/`export_future()`, so a past-valid-time forecast row is
  exported as "CAMS's best current estimate", never relabeled as a measurement.

**9. Research work mixed into the daily production workflow**
- `rebuild_forecast.yml` now only does ingest → export → features → quality gate →
  `build_final_models.py` → commit. No notebook execution, no tuning, no SHAP, no
  notebook committed daily.
- Added `research_retrain.yml`: a separate, weekly-by-default (or manual)
  workflow that runs the walk-forward notebook with tuning/SHAP at its own
  schedule and commits only the notebook + tuned params/weights/SHAP outputs --
  it no longer fights the daily workflow over `features/*`/`outputs/forecast_10d_*.csv`.

---
Not run/tested in this environment per request — review the diffs above before merging,
particularly the Postgres-touching changes (`db.py`, `schema.sql`, `export_for_modeling.py`)
against a real database.

## 2026-10-26 — fixes from AIRGUARD_AUDIT_REPORT.md

The audit report's "Change made" notes described fixes that were **not actually present**
in the archive it shipped alongside — verified against the code (exact-match RNG diff
against `generate_sample_data.py`, reading each file named in the report) and then applied
for real:

**Critical — bundled forecasts were synthetic and unmarked as such**
- `backend/dev/generate_sample_data.py` now writes `outputs/demo_data_marker.json`
  whenever it (re)writes demo data.
- `outputs/demo_data_marker.json` has been added now, retroactively, for the forecast CSVs
  already in this archive — confirmed by regenerating the sample data and diffing
  station/target/horizon/prediction/lower/upper against the shipped CSV: 250/250 rows match.
- `predict_forecast.py` deletes that marker, but only after every real forecast output has
  been written successfully, so there's never a window where a half-written real forecast
  and a deleted marker coexist.

**High — health check could label synthetic/stale output as healthy**
- `backend/app/routers/meta.py` rewritten: `/api/health` now reports
  `demo` / `unverified` / `degraded` / `ok`, backed by real checks — demo-marker presence,
  forecast schema/grid/bounds validation, city↔station forecast-origin agreement, forecast
  file and origin age, and feature-schema/model-artifact presence (checked separately from
  whether the checked-in forecast CSVs look complete).
- `backend/app/config.py` gained `DEMO_DATA_MARKER_JSON`, `FEATURE_LIST_TXT`,
  `TARGETS_JSON`, `EXPECTED_STATIONS`/`EXPECTED_TARGETS`/`EXPECTED_HORIZONS`.
- `backend/app/schemas.py`'s `HealthStatus` extended to match the new response shape.
- `frontend/src/components/Header.jsx` rewritten to show the four states distinctly
  (including an "unreachable" fallback if `/api/health` itself fails), with a short
  explanatory banner under `demo`/`unverified`/`degraded`.

**High — missing uncertainty bands could silently ship as zero-width**
- `predict_forecast.py` now raises if `outputs/models/<target>/oof_band.csv` is missing,
  unreadable, missing a horizon, or has a non-finite `q_lo`/`q_hi`, instead of filling with 0.
- `train_production_models.py` now refuses to publish a target's `.joblib` pipelines if
  `oof_residuals()` can't produce a complete, finite band for every horizon.

**High — ensemble weights could drift from the trained model version**
- `predict_forecast.py` now reads and validates `outputs/models/final_ensemble_weights_used.json`
  (the training-time snapshot `train_production_models.py` writes) instead of
  `params/ensemble_weights_*.json`, checking every weight is finite/non-negative/non-zero-sum
  per horizon and re-normalizing defensively. Confirmed the existing root-level
  `outputs/final_ensemble_weights_used.json` / `backend/app/config.py` /
  `backend/app/routers/model.py` "for inspection" copy is a deliberate second file, not a
  path mismatch — left as-is.

**High — feature-quality gate accepted a partial demo schema**
- `ingestion/data_quality_check.py` now fails closed (exit 1) if `features/feature_list.txt`
  or `features/targets.json` are missing/empty/unparseable, if `(station, date)` keys are
  missing or duplicated, if any listed feature column is absent, or if any target's actual
  column or horizon-target columns (1–10) are incomplete. Confirmed this correctly fails on
  the bundled 7-column demo table, naming the two missing files.

**Medium — daily feature-schema drift could break an older saved model**
- `model_common.py`'s `predict_latest()` now aligns each prediction frame to the saved
  pipeline's own `feature_names_in_`: a feature added since training is dropped, a feature
  the pipeline expects but the current build no longer has is passed as NaN for the fitted
  step's own imputer to handle, instead of crashing on a column mismatch.

**Still open**
- `README.md` has not yet been reconciled with the above (the audit report's "Files changed"
  list already named it; this pass didn't touch it).
- This archive still lacks the model/feature artifacts the audit flagged as structurally
  absent (`*.joblib`, `oof_band.csv`, `final_ensemble_weights_used.json` under
  `outputs/models/`, `features/feature_list.txt`, `features/targets.json`,
  `final_delhi_dataset.csv`) — none of the fixes above fabricate those; a real
  `train_production_models.py` → `predict_forecast.py` run on real source data is still
  required before `/api/health` can report `status=ok`.


## 2026-11-12 — operations and product pass

- Production retraining reads and validates committed `params/*.json` files; it never tunes or writes them.
- The scheduled full-tuning workflow is removed from the delivered workflow set. The local-only `tune_parameters_offline.py` produces preview candidates under `outputs/parameter_tuning_preview/` and cannot overwrite production params. The walk-forward notebook defaults `RUN_TUNING` and `RUN_SHAP` to false.
- MLflow uses PostgreSQL for tracking/registry metadata and an S3-compatible bucket for durable run and model artifacts. GitHub Actions refuses to run production MLflow logging without those stores and credentials.
- The frontend now has Overview, Forecast, Stations, Model health and Methodology views, with responsive navigation and API-backed data.
- Verification performed on this packaged snapshot is documented in `PROJECT_AUDIT.md`.

## 2026-11-12 update

- Supersedes the earlier historical note about `research_retrain.yml`: the
  scheduled full-tuning workflow is removed from this delivery and must be
  deleted from the existing GitHub checkout if it is still present.
- Production retraining uses committed fixed parameter/weight JSON files only.
  Full tuning remains a local report-only script and cannot overwrite them.
- MLflow tracking and model artifacts are configured for PostgreSQL plus
  S3-compatible object storage; GitHub Actions requires the corresponding
  repository secrets before a retrain can run.
- Frontend refreshed into Overview, Forecast, Stations, Model health, and
  Methodology pages.
