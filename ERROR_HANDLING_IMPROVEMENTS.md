# Error & exception handling improvements

Scope: every script that reads a CSV input, calls an external API, writes to
the database, or trains/loads a model. Goal in each case: turn a generic
exception (or a crash with no exception at all) into one that names the file,
column, target, or API call involved and says what to do about it.

Verified in this environment: build_features.py runs end-to-end against the
supplied CSVs (leakage test still passes), and every modified file passes a
syntax check. train_production_models.py / predict_forecast.py could not be
exercised end-to-end here (xgboost/lightgbm aren't installed and this sandbox
has no network access to install them, matching the "not run/tested" note
already in FIXES_APPLIED.md) -- review the model_common.py / predict_forecast.py
/ train_production_models.py diffs against your own environment before
merging, the same way FIXES_APPLIED.md asked for the database-touching changes.

## New shared helpers

**`io_utils.py`** (repo root) -- `read_csv_checked()` and `DataFileError`.
Every CSV read in the feature pipeline goes through this now. Replaces
whatever pandas/the OS happened to raise (`FileNotFoundError`,
`EmptyDataError`, `ParserError`, a bare `ValueError` from a bad
`parse_dates` column, a `KeyError` three calls later the first time a
missing column is used) with one exception type whose message names the
file, where it looked, and -- for a missing-column failure -- which columns
were required vs. actually present.

**`ingestion/http_utils.py`** -- `get_with_retry()`. None of the ingestion
scripts retried a dropped connection, timeout, or transient 5xx/429 before;
a single blip failed the whole scheduled run. This retries up to 4 times
with backoff and only gives up (raising, same as before) if every attempt
fails, so every script's existing try/except/log_run handling is unchanged,
it just sees fewer spurious failures.

## Feature-building pipeline

- **`era5_inputs.py`, `cams_inputs.py`, `fire_inputs.py`**: every
  `pd.read_csv()` replaced with `read_csv_checked()` plus an explicit
  required-columns list per file (surface/levels6/levels12/CAMS/fire). The
  optional `*_daily_ist.csv`/`era5_daily_features.csv` consistency checks
  are wrapped so a corrupt or unreadable check-file prints a warning and is
  skipped instead of crashing the whole build over a file that was only
  ever a sanity check. `era5_inputs.build_era5_daily()` now wraps each of
  its three sub-builders so a failure says which of surface/6h-levels/
  12h-levels broke.
- **`build_features.py`**: `load_station_data()` validates `timestamp` /
  `station` are present and that at least one pollutant/weather column
  exists, instead of failing deep inside the per-station reindex loop.
  `load_external()` wraps each of ERA5/CAMS/fire with context on which one
  failed. `main()` wraps `OUT_DIR.mkdir()` and the final CSV/JSON writes so
  a full disk or a permissions problem is reported clearly instead of a
  raw `OSError`/`PermissionError` traceback; the `__main__` guard catches
  `DataFileError` specifically and exits with a short, actionable message.

## Model training / prediction

- **`model_common.py`**: the module-level load of `daily_features.csv`,
  `targets.json`, and `feature_list.txt` (previously three unguarded calls
  at import time -- the first thing that broke if `build_features.py`
  hadn't been run yet) now raises a clear `FileNotFoundError` naming the
  missing file and telling you to run `build_features.py`, and separately
  validates the files aren't empty, that `d_AQI` exists, and that
  `feature_list.txt`'s entries actually match columns in the loaded table
  (catches the two files coming from different, out-of-sync builds).
  `load_tuning()` now raises clearly for an unknown target name, a missing
  params/weights file, or invalid JSON in either. `fit_production()`
  validates the target column exists and that at least one training row
  survives the horizon cutoff before calling `.fit()` (previously: a
  cryptic sklearn error on an empty `X`), and wraps the fit itself with
  horizon/row-count context. `predict_latest()` rejects an empty `pipes`
  dict and an empty `latest` slice instead of producing a confusing
  downstream `.predict()` failure or an empty forecast with no warning.
- **`predict_forecast.py`**: `load_pipeline()` now also catches a
  corrupted/incompatible `.joblib` file (not just "file missing").
  `forecast_target()` raises clearly if an ensemble has zero nonzero-weight
  members (instead of dividing by a zero weight sum later), and the
  `oof_band.csv` read is wrapped so a corrupt band file degrades to the
  existing "zero-width band" fallback instead of crashing. The final CSV/
  JSON writes are wrapped for disk/permission errors, and `__main__` prints
  a one-line "predict_forecast.py failed: ..." before re-raising so the
  GitHub Actions log shows the cause immediately.
- **`train_production_models.py`**: same zero-nonzero-weight-members guard
  as `predict_forecast.py`, plus the per-target model directory creation
  and each `joblib.dump()` are now wrapped for clear `OSError` messages.
  (Its outer try/except that logs failures to `ingestion_log` and
  re-raises was already in place and is unchanged.)

## Ingestion (`ingestion/`)

- **`db.py`**: `get_engine()` reports a malformed `DATABASE_URL` clearly
  instead of letting SQLAlchemy's engine-creation error surface as-is.
  `init_schema()` and `upsert_dataframe()` catch `OperationalError`
  (connection problems) and other `SQLAlchemyError`s separately, re-raising
  with the table name and row count so a schema mismatch or a dead
  connection is distinguishable at a glance. `upsert_dataframe()` also
  validates `pk_cols` is non-empty and that every pk column is actually
  present in the DataFrame before building any SQL. `log_run()` -- used
  inside other scripts' `except` blocks to record a failure -- no longer
  raises if the log insert itself fails; it prints a warning instead, so a
  database hiccup while *logging* an error can't suppress the real error
  or crash an otherwise-successful run.
- **`fetch_openmeteo.py`, `fetch_cpcb.py`, `fetch_firms.py`,
  `backfill_openmeteo.py`, `backfill_openmeteo_12h_grid.py`,
  `backfill_openmeteo_airquality.py`**: every `requests.get()` call now
  goes through `get_with_retry()`. Every `resp.json()` call is wrapped so a
  non-JSON response (an HTML error page, a truncated body) raises a
  message that includes a snippet of what was actually returned, instead
  of a bare `JSONDecodeError`. `fetch_openmeteo.py`'s `main()` was
  reordered so the fetch step is inside the same try/except that already
  logged DB-upsert failures to `ingestion_log`, so a network failure is
  now logged the same way a database failure is. The three
  `backfill_*.py` CLI scripts now validate the `start`/`end` date
  arguments up front (`date_chunks()` raising on a bad date previously
  surfaced as a pandas `ValueError` mid-run); a bad date or an inverted
  range now exits immediately with a one-line message instead of after
  already connecting to the database.

## Deliberately left alone

- `ingestion/fetch_firms.py`'s and `backfill_openmeteo.py`'s existing
  per-source / per-chunk partial-failure tracking (`errors`/
  `failed_chunks` lists, "ok (partial)" status, `sys.exit(1)` on any
  failure) was already solid per `FIXES_APPLIED.md` and wasn't touched
  beyond the retry wrapper and JSON-parse guard.
- `ingestion/fetch_cpcb.py`'s `main()` structure (fetch happens before the
  database engine is created) was left as-is: the early `--list-stations`
  exit path genuinely needs the fetch to happen first, so restructuring it
  to log fetch failures to `ingestion_log` the way `fetch_openmeteo.py` now
  does would have meant creating a throwaway DB engine on a pure CLI
  utility path.
- `ingestion/cds_backfill.py`, `ingestion/rebuild_fire_daily.py`,
  `ingestion/migrate_historical.py`, `ingestion/data_quality_check.py`,
  `ingestion/export_for_modeling.py`, `ingestion/check_retrain_due.py`
  were reviewed but not changed -- each already has a top-level
  try/except around its main work, and none of them had the specific gaps
  (unguarded CSV reads with no column validation, un-retried HTTP calls,
  un-validated empty-data edge cases) this pass targeted. Worth a follow-up
  pass if you want the same `read_csv_checked()` treatment applied there.
