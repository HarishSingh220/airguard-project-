"""
build_final_models.py
Manual one-shot convenience wrapper: trains the production models AND scores
today's forecast in a single local run (train_production_models.py, then
predict_forecast.py). The automated deployment runs these as two separate
scripts on two separate schedules instead:

    train_production_models.py   every ~30 days (retrain_model.yml)
    predict_forecast.py          daily            (rebuild_forecast.yml)

See model_common.py for the data/feature/model logic shared by both, and
README.md's "Scheduled retraining (30-day walk-forward)" section for the
overall design. Use *this* script locally -- e.g. right after first setting
everything up, when nothing has been trained yet and you want one command
that does both steps.

Run from the repository root, after build_features.py:
    python build_final_models.py
"""
import predict_forecast
import train_production_models


def main():
    train_production_models.main()
    predict_forecast.main()


if __name__ == "__main__":
    main()
