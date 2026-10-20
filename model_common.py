"""
model_common.py
Shared data loading, feature construction and model definitions for the
production AirGuard forecast. Split out of what used to be a single
build_final_models.py so that training and prediction can run on separate
schedules:

    train_production_models.py   refits the four ensemble members on the
                                  latest data, using the tuned hyperparameters
                                  and ensemble weights AS-IS (no tuning here --
                                  see "Fixed, not tuned" below). Runs every
                                  ~30 days (.github/workflows/retrain_model.yml).

    predict_forecast.py          loads the models train_production_models.py
                                  already fit and saved, and scores today's
                                  feature row. No fitting. Runs daily
                                  (.github/workflows/rebuild_forecast.yml).

Everything in this file is read-only with respect to best_params_*.json and
ensemble_weights_*.json -- nothing here ever re-tunes a hyperparameter or
re-fits an ensemble weight. Those files are produced once by the walk-forward
notebook (research_retrain.yml, on its own separate schedule) and are simply
loaded and reused by every retrain.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError("lightgbm is not installed. Install it with: pip install lightgbm") from exc

from build_features import diwali_offset, diwali_window

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURE_DIR = Path("features")
OUT_DIR = Path("outputs")
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)

FORECAST_HORIZON = 10
HORIZONS = list(range(1, FORECAST_HORIZON + 1))
RANDOM_STATE = 42

# Where the tuned params / weights live (edit if you move them). "Fixed, not
# tuned": train_production_models.py reads these every retrain and never
# writes to them -- only research_retrain.yml's notebook run does that.
PARAM_DIR = Path("params")  # falls back to repo root if params/ doesn't exist
BEST_PARAMS_FILES = {
    "AQI": "best_params_AQI.json",
    "NOx": "best_params_NOx.json",
    "O3": "best_params_O3.json",
    "O3_8H": "best_params_O3_8H.json",
    "PM25": "best_params_PM25.json",
}
ENSEMBLE_WEIGHTS_FILES = {
    "AQI": "ensemble_weights_AQI.json",
    "NOx": "ensemble_weights_NOx.json",
    "O3": "ensemble_weights_O3.json",
    "O3_8H": "ensemble_weights_O3_8H.json",
    "PM25": "ensemble_weights_PM25.json",
}

MEMBERS_ALL = ["RandomForest", "LightGBM", "XGBoost", "Ridge"]

# Cross-validation folds used only to size the residual/uncertainty band (cheap: no
# retraining across walk-forward origins, just a handful of time-blocked validation
# folds before WF_START). Recomputed once per retrain, alongside the production fit.
WF_START = pd.Timestamp("2025-10-01")
BAND_FOLDS = 4
BAND_FOLD_DAYS = 90

# Output clipping bounds per target (physical / regulatory ranges)
TARGET_BOUNDS = {
    "AQI": (0, 500),
    "PM25": (0, 1000),
    "O3": (0, 500),
    "O3_8H": (0, 500),
    "NOx": (0, 1000),
}

CPCB_BINS = [0, 50, 100, 200, 300, 400, 500]
CPCB_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]


def _resolve(fname):
    p = PARAM_DIR / fname
    return p if p.exists() else Path(fname)


# ---------------------------------------------------------------------------
# Data (same loading/feature logic as the notebook)
# ---------------------------------------------------------------------------
def _require(relpath):
    p = FEATURE_DIR / relpath
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run build_features.py from the repository root before "
            f"training or predicting (it writes everything under {FEATURE_DIR}/)."
        )
    return p


try:
    daily = (pd.read_csv(_require("daily_features.csv"), parse_dates=["date"])
               .sort_values(["station", "date"]).reset_index(drop=True))
except pd.errors.EmptyDataError as exc:
    raise RuntimeError(f"{FEATURE_DIR / 'daily_features.csv'} is empty.") from exc
if daily.empty:
    raise RuntimeError(f"{FEATURE_DIR / 'daily_features.csv'} parsed but has zero rows.")
if "d_AQI" not in daily.columns:
    raise RuntimeError(
        f"{FEATURE_DIR / 'daily_features.csv'} has no 'd_AQI' column -- re-run build_features.py."
    )

try:
    TARGETS = json.loads(_require("targets.json").read_text())
except json.JSONDecodeError as exc:
    raise RuntimeError(f"{FEATURE_DIR / 'targets.json'} is not valid JSON: {exc}") from exc

BASE_FEATURES = [c.strip() for c in _require("feature_list.txt").read_text().splitlines() if c.strip()]
if not BASE_FEATURES:
    raise RuntimeError(f"{FEATURE_DIR / 'feature_list.txt'} is empty -- re-run build_features.py.")
BASE_FEATURES = [c for c in BASE_FEATURES if c in daily.columns]
if not BASE_FEATURES:
    raise RuntimeError(
        "None of the columns listed in feature_list.txt are present in daily_features.csv -- "
        "the two files look like they came from different build_features.py runs."
    )
daily["AQI"] = daily["d_AQI"]

TGT_CAL_COLS = ["tgt_dayofweek", "tgt_is_weekend", "tgt_month", "tgt_doy_sin", "tgt_doy_cos",
                "tgt_diwali_window", "tgt_days_from_diwali"]
FEATURE_COLS = BASE_FEATURES + TGT_CAL_COLS

_X_cache = {}


def target_calendar(h):
    td = pd.DatetimeIndex(daily["date"]) + pd.Timedelta(days=h)
    return pd.DataFrame({
        "tgt_dayofweek": td.dayofweek,
        "tgt_is_weekend": (td.dayofweek >= 5).astype(int),
        "tgt_month": td.month,
        "tgt_doy_sin": np.sin(2 * np.pi * td.dayofyear / 365.25),
        "tgt_doy_cos": np.cos(2 * np.pi * td.dayofyear / 365.25),
        "tgt_diwali_window": diwali_window(td),
        "tgt_days_from_diwali": np.clip(diwali_offset(td), -15, 15),
    }, index=daily.index)


def make_X(h):
    if h not in _X_cache:
        _X_cache[h] = pd.concat([daily[BASE_FEATURES + ["station"]], target_calendar(h)], axis=1)
    return _X_cache[h]


def make_prep(kind, num_cols):
    cat = ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["station"])
    if kind == "passthrough":
        num = ("num", "passthrough", list(num_cols))
    elif kind == "impute":
        num = ("num", SimpleImputer(strategy="median", keep_empty_features=True), list(num_cols))
    elif kind == "impute_scale":
        num = ("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                                ("sc", StandardScaler())]), list(num_cols))
    else:
        raise ValueError(kind)
    return ColumnTransformer([num, cat], remainder="drop")


XGB = XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85,
                   objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1)
MODEL_CONFIGS = {
    "XGBoost": (XGB, "passthrough"),
    "LightGBM": (LGBMRegressor(n_estimators=700, num_leaves=63, learning_rate=0.03, subsample=0.85, subsample_freq=1,
                               colsample_bytree=0.85, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1), "passthrough"),
    "RandomForest": (RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=2,
                                           random_state=RANDOM_STATE, n_jobs=-1), "impute"),
    "Ridge": (Ridge(alpha=30.0), "impute_scale"),
}


# ---------------------------------------------------------------------------
# Tuned params + weights, per target -- READ ONLY. Nothing in this file (or in
# train_production_models.py, which is the only thing that calls fit_production)
# ever re-tunes these; they are loaded fresh every retrain and plugged straight
# into clones of the base estimators above via tuned_estimator().
# ---------------------------------------------------------------------------
def load_tuning(target):
    if target not in BEST_PARAMS_FILES:
        raise KeyError(f"Unknown target {target!r}. Expected one of {list(BEST_PARAMS_FILES)}.")
    params_path = _resolve(BEST_PARAMS_FILES[target])
    weights_path = _resolve(ENSEMBLE_WEIGHTS_FILES[target])
    try:
        params = json.loads(params_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{params_path} not found -- tuned hyperparameters for {target!r} are missing. "
            f"Run the walk-forward notebook (research_retrain.yml) first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{params_path} is not valid JSON: {exc}") from exc
    try:
        weights = json.loads(weights_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{weights_path} not found -- ensemble weights for {target!r} are missing. "
            f"Run the walk-forward notebook (research_retrain.yml) first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{weights_path} is not valid JSON: {exc}") from exc
    return params, weights


def member_params(params, name, h):
    return params[name][str(h)]


def tuned_estimator(params, name):
    """h -> cloned estimator carrying the tuned hyperparameters for that exact horizon."""
    from sklearn.base import clone
    base = MODEL_CONFIGS[name][0]

    def build(h):
        p = member_params(params, name, h)
        p = {k: v for k, v in p.items() if v is not None}  # RandomForest max_depth can be JSON null
        return clone(base).set_params(**p)
    return build


# ---------------------------------------------------------------------------
# Production fit + forecast helpers
# ---------------------------------------------------------------------------
def latest_complete_date(actual_col):
    """The forecast origin for a target: the latest date with a real (non-null)
    current-day value in actual_col, for at least one station -- not just
    daily['date'].max().
    """
    valid = daily.loc[daily[actual_col].notna(), "date"]
    if valid.empty:
        raise ValueError(f"No non-null '{actual_col}' values anywhere in daily -- cannot pick a forecast origin.")
    as_of = valid.max()
    if as_of < daily["date"].max():
        print(f"NOTE: forecast origin backed off to {as_of.date()} "
              f"(daily table's max date {daily['date'].max().date()} has no '{actual_col}' value yet)")
    return as_of


def fit_production(estimator_fn, prep_kind, target_prefix, horizons=None, actual_col="d_AQI"):
    """Refit on every label known as of the latest COMPLETE date for this target.
    This is the only function in the codebase that calls .fit() on a production
    member model -- only train_production_models.py calls it, on its ~30-day
    schedule. predict_forecast.py never calls this; it only loads what this
    already saved.
    """
    from sklearn.base import clone
    horizons = HORIZONS if horizons is None else list(horizons)
    as_of = latest_complete_date(actual_col)
    pipes = {}
    for h in horizons:
        target = f"{target_prefix}{h}"
        if target not in daily.columns:
            raise KeyError(f"'{target}' is not a column in daily_features.csv (horizon {h}).")
        m = (daily["date"] <= as_of - pd.Timedelta(days=h)) & daily[target].notna()
        n_train = int(m.sum())
        if n_train == 0:
            raise RuntimeError(
                f"No training rows available for target '{target}' at horizon {h} "
                f"(forecast origin {as_of.date()}). daily_features.csv may not have enough "
                f"history yet for this horizon."
            )
        est_h = estimator_fn(h)
        pipe = Pipeline([("prep", make_prep(prep_kind, FEATURE_COLS)), ("model", clone(est_h))])
        try:
            pipe.fit(make_X(h).loc[m], daily.loc[m, target])
        except Exception as exc:
            raise RuntimeError(
                f"Fitting failed for target '{target}' at horizon {h} ({n_train} training rows): {exc}"
            ) from exc
        pipes[h] = pipe
    return pipes


def predict_latest(pipes, actual_col="d_AQI"):
    if not pipes:
        raise ValueError("predict_latest() got an empty `pipes` dict -- nothing to predict with.")
    as_of = latest_complete_date(actual_col)
    latest = daily[daily["date"] == as_of]
    if latest.empty:
        raise RuntimeError(f"No rows in daily_features.csv for the forecast origin date {as_of.date()}.")
    out = []
    for h, pipe in pipes.items():
        t = latest[["station", "date"]].copy()
        t["horizon"] = h
        t["forecast_date"] = t["date"] + pd.Timedelta(days=h)
        try:
            t["prediction"] = pipe.predict(make_X(h).loc[latest.index])
        except Exception as exc:
            raise RuntimeError(f"Prediction failed at horizon {h}: {exc}") from exc
        out.append(t)
    return pd.concat(out, ignore_index=True)


def tune_folds():
    cutoff = WF_START
    folds = []
    for k in range(BAND_FOLDS):
        v_start = cutoff - pd.Timedelta(days=BAND_FOLD_DAYS * (BAND_FOLDS - k))
        v_end = v_start + pd.Timedelta(days=BAND_FOLD_DAYS - 1)
        folds.append((v_start, v_end))
    return folds


def oof_residuals(target_name, params, weights):
    """Out-of-fold residuals of the WEIGHTED ensemble (tuned params + supplied weights), per
    horizon, from a handful of time-blocked validation folds strictly before WF_START. Cheap
    stand-in for a full walk-forward re-run, used only to size the forecast's uncertainty band.
    Also fits models internally (on each fold) -- it's "training" in the same sense as
    fit_production, just for band-sizing rather than production inference, so it only runs
    from train_production_models.py as well.
    """
    from sklearn.base import clone
    prefix = TARGETS[target_name]["prefix"]
    dates = daily["date"]
    folds = tune_folds()
    band_rows = []
    for h in HORIZONS:
        target = f"{prefix}{h}"
        X_h = make_X(h)
        member_oof = {}
        y_oof = None
        for name in MEMBERS_ALL:
            w = weights.get(str(h), weights.get(h, {})).get(name, 0.0)
            if w <= 0:
                continue
            est, kind = MODEL_CONFIGS[name]
            p = {k: v for k, v in member_params(params, name, h).items() if v is not None}
            preds, ys, rows = [], [], []
            for v_start, v_end in folds:
                tr = (dates <= v_start - pd.Timedelta(days=h)) & daily[target].notna()
                va = (dates >= v_start) & (dates <= v_end) & daily[target].notna()
                if tr.sum() < 500 or va.sum() == 0:
                    continue
                pipe = Pipeline([("prep", make_prep(kind, FEATURE_COLS)), ("model", clone(est).set_params(**p))])
                pipe.fit(X_h.loc[tr], daily.loc[tr, target])
                pred = pipe.predict(X_h.loc[va])
                preds.append(pred)
                ys.append(daily.loc[va, target].to_numpy())
                rows.append(daily.index[va])
            if not preds:
                continue
            member_oof[name] = (np.concatenate(rows), np.concatenate(preds))
            if y_oof is None:
                y_oof = (np.concatenate(rows), np.concatenate(ys))
        if not member_oof:
            continue
        # align all members on the same rows
        common_idx = set(y_oof[0])
        for rows, _ in member_oof.values():
            common_idx &= set(rows)
        common_idx = np.array(sorted(common_idx))
        if len(common_idx) == 0:
            continue
        blend = np.zeros(len(common_idx))
        wsum = 0.0
        for name, (rows, preds) in member_oof.items():
            w = weights.get(str(h), weights.get(h, {})).get(name, 0.0)
            lookup = dict(zip(rows, preds))
            blend += w * np.array([lookup[i] for i in common_idx])
            wsum += w
        if wsum > 0:
            blend /= wsum
        y_lookup = dict(zip(*y_oof))
        y_true = np.array([y_lookup[i] for i in common_idx])
        err = y_true - blend
        band_rows.append({"horizon": h, "q_lo": np.quantile(err, 0.10), "q_hi": np.quantile(err, 0.90)})
    return pd.DataFrame(band_rows).set_index("horizon") if band_rows else pd.DataFrame(
        columns=["q_lo", "q_hi"]).rename_axis("horizon")
