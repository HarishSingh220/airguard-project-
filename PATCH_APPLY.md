# Applying the AirGuard safe update patch

This patch is designed to be extracted **over your existing AirGuard project folder**.
It intentionally does not contain `outputs/`, input datasets, local MLflow artifacts,
`.joblib` model binaries, virtual environments or Git metadata.

## Before extracting

1. Back up the project folder, especially `outputs/models/` and any local data.
2. Extract this ZIP into the root of your existing `airguard-project-fixed` folder.
3. Allow files in the patch to replace matching source/config/frontend files.
4. Do not replace your existing model binaries or local data directories.

## Remove the old tuning action

The earlier GitHub repository had a workflow named `research_retrain.yml` which ran
full parameter tuning. It is not included in this patch. From the repository root,
run this only if the file still exists:

```powershell
if (Test-Path .github/workflows/research_retrain.yml) {
    git rm .github/workflows/research_retrain.yml
}
```

Do not remove unrelated ingestion workflows.

## Frontend cleanup

The refreshed `App.jsx` no longer imports the old standalone view components. If these
files are still present in your checkout and have no imports elsewhere, remove them:

- `frontend/src/ForecastChart.jsx`
- `frontend/src/ForecastTable.jsx`
- `frontend/src/Header.jsx`
- `frontend/src/HeroReading.jsx`
- `frontend/src/ModelFooter.jsx`
- `frontend/src/StationTabs.jsx`
- `frontend/src/TargetTabs.jsx`

(The list is advisory: the uploaded ZIP already did not include these old components.)

## Fixed parameter policy

Routine Actions only read `params/best_params_*.json` and
`params/ensemble_weights_*.json`. Full tuning is local only:

```powershell
python build_features.py
python tune_parameters_offline.py --trials 10
```

The tuner writes candidate files under `outputs/parameter_tuning_preview/` and never
updates production parameter files. Do not add it to a workflow or schedule.

## MLflow prerequisites

Set up PostgreSQL and an S3-compatible bucket, then add the required GitHub Actions
secrets documented in `MLFLOW_GUIDE.md`. The first real retrain is the integration
acceptance test; the ZIP audit could not connect to your private services.

## Review and commit

```powershell
git status
git diff -- .gitignore params train_production_models.py model_common.py
git add .
git commit -m "Add persistent MLflow and refresh AirGuard dashboard"
```

Keep your existing `outputs/models/**/*.joblib` files. Do not use `git push --force`.
