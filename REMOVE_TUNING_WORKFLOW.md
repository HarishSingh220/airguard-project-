# Remove automatic full-parameter tuning

The weekly full-tuning GitHub Actions workflow must not exist in the production
repository. The uploaded ZIP did not include `.github/workflows/research_retrain.yml`,
but the live GitHub repository did. Remove it from your checkout before pushing
this project update:

```powershell
git rm .github/workflows/research_retrain.yml
```

If Git reports that the file does not exist, it is already absent. Do not
recreate it. Keep `baseline_walkforward_CLEAN_v9.ipynb` as research material;
its `RUN_TUNING` and `RUN_SHAP` switches are off by default.

Full tuning is available only as a local command:

```powershell
python build_features.py
python tune_parameters_offline.py --trials 10
```

It writes candidate parameter/weight JSONs and a tuning report to
`outputs/parameter_tuning_preview/`. It has no apply flag and never modifies
`params/*.json`. Production retraining only reads those committed fixed files.

Stage the project changes and commit them:

```powershell
git add .
git commit -m "Add persistent MLflow and improve AirGuard dashboard"
```
