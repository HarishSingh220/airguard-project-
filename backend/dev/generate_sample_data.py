"""
generate_sample_data.py
DEV/DEMO ONLY. Writes synthetic (NOT real) versions of the files
rebuild_forecast.yml / retrain_model.yml normally produce, so the API and
the React frontend can be exercised end-to-end on a fresh clone -- before
Postgres, GitHub Actions secrets, or a real model training run exist.

Every value here is fabricated with numpy's RNG. Delete outputs/ and
features/ (or just re-run the real pipeline) to replace this with real data.

Run from the repository root:
    python backend/dev/generate_sample_data.py
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs"
MODEL_DIR = OUT_DIR / "models"
FEATURE_DIR = REPO_ROOT / "features"

STATIONS = ["Anand Vihar", "Dwarka Sector 8", "IMD Lodhi Road", "ITO", "Punjabi Bagh"]
TARGETS = {
    "AQI": (0, 500, 180),
    "PM25": (0, 1000, 90),
    "O3": (0, 500, 40),
    "O3_8H": (0, 500, 45),
    "NOx": (0, 1000, 60),
}
CPCB_BINS = [0, 50, 100, 200, 300, 400, 500]
CPCB_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]
HORIZONS = list(range(1, 11))
RNG = np.random.default_rng(42)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

    # ---- outputs/forecast_10d_station_final.csv ----
    rows = []
    for station in STATIONS:
        station_bias = RNG.normal(0, 15)
        for target, (lo, hi, base) in TARGETS.items():
            for h in HORIZONS:
                pred = max(lo, base + station_bias + RNG.normal(0, 20) - h * 1.5)
                pred = min(hi, pred)
                band = 10 + h * 2.5
                rows.append({
                    "station": station, "date": today.date(), "target": target, "horizon": h,
                    "forecast_date": (today + pd.Timedelta(days=h)).date(),
                    "prediction": round(pred, 1),
                    "lower": round(max(lo, pred - band), 1),
                    "upper": round(min(hi, pred + band), 1),
                })
    station_df = pd.DataFrame(rows)
    station_df.to_csv(OUT_DIR / "forecast_10d_station_final.csv", index=False)

    # ---- outputs/forecast_10d_city_final.csv (mean across stations) ----
    city_df = (station_df.groupby(["target", "forecast_date", "horizon"], as_index=False)
              [["prediction", "lower", "upper"]].mean().round(1))
    city_df.to_csv(OUT_DIR / "forecast_10d_city_final.csv", index=False)

    # ---- outputs/forecast_10d_city_aqi_with_category.csv ----
    aqi_city = city_df[city_df["target"] == "AQI"].copy()
    aqi_city["category"] = pd.cut(aqi_city["prediction"], CPCB_BINS, labels=CPCB_LABELS, include_lowest=True)
    aqi_city.to_csv(OUT_DIR / "forecast_10d_city_aqi_with_category.csv", index=False)

    # ---- outputs/final_ensemble_weights_used.json ----
    members = ["RandomForest", "LightGBM", "XGBoost", "Ridge"]
    weights = {}
    for target in TARGETS:
        by_h = {}
        for h in HORIZONS:
            w = RNG.dirichlet(np.ones(len(members)))
            by_h[str(h)] = {m: round(float(x), 4) for m, x in zip(members, w)}
        weights[target] = by_h
    (OUT_DIR / "final_ensemble_weights_used.json").write_text(json.dumps(weights, indent=1))

    # ---- outputs/models/trained_at.json ----
    trained_at = {
        "trained_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
        "targets": {t: {"trained_up_to": str((today - pd.Timedelta(days=1)).date()),
                        "members": members} for t in TARGETS},
    }
    (MODEL_DIR / "trained_at.json").write_text(json.dumps(trained_at, indent=1))

    # ---- features/daily_features.csv (last 60 days, actuals only) ----
    hist_rows = []
    for station in STATIONS:
        station_bias = RNG.normal(0, 15)
        level = 180 + station_bias
        for days_ago in range(60, 0, -1):
            level += RNG.normal(0, 12)
            level = float(np.clip(level, 20, 480))
            d = (today - pd.Timedelta(days=days_ago)).date()
            hist_rows.append({
                "station": station, "date": d,
                "d_AQI": round(level, 1),
                "d_PM25": round(max(0, level * 0.55 + RNG.normal(0, 8)), 1),
                "d_O3": round(max(0, 40 + RNG.normal(0, 10)), 1),
                "d_O3_max8h": round(max(0, 55 + RNG.normal(0, 12)), 1),
                "d_NOx": round(max(0, 60 + RNG.normal(0, 15)), 1),
            })
    pd.DataFrame(hist_rows).to_csv(FEATURE_DIR / "daily_features.csv", index=False)

    # ---- outputs/demo_data_marker.json ----
    # /api/health (backend/app/routers/meta.py) reads this and reports
    # status="demo" whenever it exists -- it must never be able to confuse this
    # fabricated bundle with a real forecast. predict_forecast.py deletes this
    # file, but only after a real run has finished writing every forecast
    # output, so there's never a window where a half-written real forecast and
    # a deleted marker coexist.
    (OUT_DIR / "demo_data_marker.json").write_text(json.dumps({
        "is_demo_data": True,
        "generated_by": "backend/dev/generate_sample_data.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Every value in outputs/ and features/daily_features.csv from this run is "
                "fabricated with numpy's RNG (seed 42). Not real model output. Run the real "
                "pipeline (build_features.py, train_production_models.py, predict_forecast.py, "
                "or the GitHub Actions workflows) to replace it.",
    }, indent=1))

    print(f"Wrote synthetic demo data under {OUT_DIR} and {FEATURE_DIR}.")
    print("This is FAKE data for trying the API/frontend end-to-end -- run the real")
    print("pipeline (build_features.py, train_production_models.py, predict_forecast.py,")
    print("or the GitHub Actions workflows) to replace it.")
    print(f"Wrote {OUT_DIR / 'demo_data_marker.json'} -- /api/health will report status=\"demo\" "
          f"until a real predict_forecast.py run removes it.")


if __name__ == "__main__":
    main()
