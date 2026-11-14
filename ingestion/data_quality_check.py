"""
data_quality_check.py
Quantifies Finding 5 instead of leaving it as prose in the README: how much of
features/daily_features.csv's live tail (the period past the historical cutoff)
is actually missing, per feature, and whether that's within the gaps the
project already knows about and has documented, or something new.

Run after build_features.py, before the notebook / build_final_models.py:
    python ingestion/data_quality_check.py

Exit code is non-zero (fails the workflow step) if any feature NOT in
KNOWN_GAP_COLUMNS has live-tail missingness above MAX_UNEXPECTED_MISSING, or
if a KNOWN_GAP_COLUMNS feature's missingness jumps far past what's already
expected -- i.e. it draws the line between "the documented, accepted
approximations" and "something broke or drifted further than that".

This does not fix the underlying source gaps (q/w on pressure levels, the
four derived ERA5 columns, FIRMS NRT's missing `type` field, etc.) -- those
need either a compatible live data source or a retrain without the affected
features, which is a product decision, not something a script can decide for
you. What this gives you is the actual numbers to make that decision with,
every run, instead of discovering it from a bad forecast.
"""
import json
import sys
from pathlib import Path

import pandas as pd

FEATURES_PATH = Path("features/daily_features.csv")
FEATURE_LIST_PATH = Path("features/feature_list.txt")
TARGETS_JSON_PATH = Path("features/targets.json")
FORECAST_HORIZONS = list(range(1, 11))

# Columns the README already documents as having a different, weaker live
# source than their historical counterpart. These are *expected* to show
# materially higher missingness after the historical cutoff -- that is not,
# by itself, a regression. Tune the allowed ceiling per column as you learn
# the real live-tail rates; these starting values are deliberately generous.
KNOWN_GAP_COLUMNS = {
    # vertical velocity (w) on pressure levels: not available from Open-Meteo's
    # general forecast model, NaN in the live tail (verified names, feature_list.txt)
    "ua_w850_lag1d": 0.95, "ua_w700_lag1d": 0.95,
    "ua_w850_chg1d": 0.95, "ua_w850_roll3d": 0.95,
    # the 12h / 4-point pressure grid has no live source at all -- historical-only
    "ua12_t900_lag1d": 1.0, "ua12_t800_lag1d": 1.0, "ua12_u900_lag1d": 1.0, "ua12_v900_lag1d": 1.0,
    "ua12_spd900_lag1d": 1.0, "ua12_spd800_lag1d": 1.0, "ua12_r900_lag1d": 1.0, "ua12_r800_lag1d": 1.0,
    "ua12_dT_800_900_lag1d": 1.0, "ua12_cap_800_900_lag1d": 1.0,
    "ua12_t900_ns_grad_lag1d": 1.0, "ua12_t900_ew_grad_lag1d": 1.0, "ua12_dT_800_900_0530_lag1d": 1.0,
    # the four derived ERA5 columns computed with export_for_modeling.py's own approximate
    # formulas live (inversion/ventilation proxies) -- a visible step, not full missingness
    "inv_surface_mean_lag1d": 0.10, "inv_surface_night_lag1d": 0.10, "inv_surface_max_lag1d": 0.10,
    "inv_hours_frac_lag1d": 0.10, "inv_deep_mean_lag1d": 0.10, "lapse_925_700_mean_lag1d": 0.10,
    "vent_mean_lag1d": 0.10, "vent_min_lag1d": 0.10, "vent_day_mean_lag1d": 0.10,
}

# Any feature actually used by the model (i.e. present in daily_features.csv)
# that is NOT in KNOWN_GAP_COLUMNS should stay close to its historical
# missingness. This is the ceiling for an *unexpected* gap.
MAX_UNEXPECTED_MISSING = 0.15

# Documented, known-dated station-data outages (not a per-column gap like
# KNOWN_GAP_COLUMNS above -- a whole-pipeline event, like CPCB's live feed having no
# history/backfill endpoint, see fetch_cpcb.py's docstring). Right after such a gap
# closes, every AQI/pollutant lag and rolling feature (the "families" listed) is
# legitimately ~100% missing in the live tail for up to LAG_GRACE_DAYS -- that's the
# gap re-appearing through the features that look back over it, not a new failure.
# Fill in real dates once you know them (see README's "gap" discussion / MLFLOW_GUIDE.md);
# leave the list empty if there's no known gap to account for.
#
# Example once dates are known:
#   {"start": "2025-12-31", "end": "2026-10-11",
#    "families": {"AQI history", "Pollutant history", "Pollutant current", "City state"},
#    "note": "CPCB live feed has no backfill endpoint -- permanent gap for this window"}
KNOWN_GAP_WINDOWS = []

# Longest lookback window any "Pollutant history"/"AQI history" feature uses
# (AQI_roll_mean_30d, station_data_missing_frac_30d) -- a known gap keeps inflating
# those features' missingness for this many days after live data resumes, purely
# because the lookback window still partially overlaps the gap. Matches
# build_features.py's longest rolling/lag window (30 days).
LAG_GRACE_DAYS = 30

FEATURE_GROUPS_PATH = Path("features/feature_groups.json")

# How many of the most recent days count as "the live tail" for this check.
# Generous on purpose -- this is a coarse early-warning check, not a precise
# historical/live boundary detector (export_for_modeling.py knows the real
# boundary per source; this script intentionally doesn't need to).
LIVE_TAIL_DAYS = 14

# If the file's own last date is more than this many days behind today, it has
# no real live tail at all -- the "last 14 days" of a stale file is still
# historical data, not freshly-ingested data, and comparing it against the
# live-tail ceilings would just be re-measuring a known historical gap and
# calling it a new failure. Catch that case explicitly instead.
MAX_STALENESS_DAYS = 2

# A known-gap or unexpected-gap feature only fails if its live-tail missingness
# is *worse than its own historical baseline* by more than this margin, on top
# of the absolute ceiling. This is what actually distinguishes "a new ingestion
# failure" from "a feature that has always had this much missingness" -- an
# absolute ceiling alone treats the two identically.
BASELINE_MARGIN = 0.05


def _check_model_schema_present(df):
    """A missingness-only check on daily_features.csv is not enough to prove the
    columns the forecasting estimators actually use exist: that file can have
    the right shape and still be a demo/partial table (see
    backend/dev/generate_sample_data.py's 7-column sample). Fails closed --
    returns a list of fatal problems -- if the full generated feature schema
    (feature_list.txt, targets.json), the station/date key, or every horizon
    target column for every declared target isn't actually present.
    """
    problems = []

    if not FEATURE_LIST_PATH.exists():
        problems.append(f"{FEATURE_LIST_PATH} not found -- run build_features.py first.")
    if not TARGETS_JSON_PATH.exists():
        problems.append(f"{TARGETS_JSON_PATH} not found -- run build_features.py first.")
    if problems:
        return problems  # nothing else below can be checked without these

    try:
        feature_list = [c.strip() for c in FEATURE_LIST_PATH.read_text().splitlines() if c.strip()]
    except OSError as exc:
        return [f"{FEATURE_LIST_PATH} could not be read: {exc}"]
    if not feature_list:
        problems.append(f"{FEATURE_LIST_PATH} is empty.")

    try:
        targets = json.loads(TARGETS_JSON_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return problems + [f"{TARGETS_JSON_PATH} could not be read/parsed: {exc}"]
    if not targets:
        problems.append(f"{TARGETS_JSON_PATH} is empty.")

    if "station" not in df.columns or "date" not in df.columns:
        problems.append("daily_features.csv is missing the station and/or date key column(s).")
    else:
        dup = df.duplicated(subset=["station", "date"]).sum()
        if dup:
            problems.append(f"daily_features.csv has {int(dup)} duplicate (station, date) rows.")

    missing_features = [c for c in feature_list if c not in df.columns]
    if missing_features:
        problems.append(
            f"{len(missing_features)} feature(s) listed in {FEATURE_LIST_PATH} are absent from "
            f"daily_features.csv, e.g. {missing_features[:5]}."
        )

    for target_name, meta in (targets.items() if isinstance(targets, dict) else []):
        prefix = meta.get("prefix") if isinstance(meta, dict) else None
        actual_col = meta.get("actual") if isinstance(meta, dict) else None
        if actual_col and actual_col not in df.columns:
            problems.append(f"target {target_name!r}'s actual column {actual_col!r} is missing.")
        if prefix:
            missing_h = [h for h in FORECAST_HORIZONS if f"{prefix}{h}" not in df.columns]
            if missing_h:
                problems.append(
                    f"target {target_name!r} is missing horizon target column(s) for {missing_h} "
                    f"(expected {prefix}1 .. {prefix}{FORECAST_HORIZONS[-1]})."
                )

    return problems


def main():
    if not FEATURES_PATH.exists():
        print(f"SKIP: {FEATURES_PATH} not found (run build_features.py first)")
        return 0

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    if df.empty:
        print("SKIP: daily_features.csv is empty")
        return 0

    schema_problems = _check_model_schema_present(df)
    if schema_problems:
        print(f"FAIL: the full model feature/target schema is not present -- "
              f"a missingness-only check on daily_features.csv cannot substitute for this:")
        for p in schema_problems:
            print(f"  - {p}")
        return 1

    # Required-input staleness is a separate, simpler check: if the file itself
    # hasn't moved in MAX_STALENESS_DAYS, ingestion has stalled (or never ran),
    # and the "last 14 days" below isn't actually live data -- it's just the
    # tail of the historical file. Fail on that directly rather than letting it
    # masquerade as a live-tail missingness problem.
    today = pd.Timestamp.now().normalize()
    staleness_days = (today - df["date"].max()).days
    if staleness_days > MAX_STALENESS_DAYS:
        print(f"FAIL: {FEATURES_PATH} is stale -- its last date is {df['date'].max().date()}, "
              f"{staleness_days} days behind today ({today.date()}). This file has no real live "
              f"tail; treating its final {LIVE_TAIL_DAYS} days as 'live' would just re-measure "
              f"known historical missingness. Check that the ingestion workflow is actually running.")
        return 1

    cutoff = df["date"].max() - pd.Timedelta(days=LIVE_TAIL_DAYS)
    tail = df[df["date"] > cutoff]
    hist = df[df["date"] <= cutoff]
    if tail.empty:
        print("SKIP: no rows in the live-tail window yet")
        return 0

    feature_cols = [c for c in df.columns if c not in ("date", "station") and not c.startswith("target")]
    tail_missing = tail[feature_cols].isna().mean()
    hist_missing = hist[feature_cols].isna().mean() if not hist.empty else pd.Series(0.0, index=feature_cols)

    print(f"Live-tail window: {tail['date'].min().date()} .. {tail['date'].max().date()} "
          f"({tail['date'].nunique()} days)")

    # Which columns get a temporarily relaxed ceiling because the live tail still falls
    # within LAG_GRACE_DAYS of a documented gap window closing -- see KNOWN_GAP_WINDOWS.
    # This expires automatically: once today is more than LAG_GRACE_DAYS past a window's
    # end, that window no longer relaxes anything, and persistently-high missingness
    # past that point correctly fails again as a real, new problem.
    grace_relaxed = {}
    if KNOWN_GAP_WINDOWS:
        try:
            feature_groups = json.loads(FEATURE_GROUPS_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            feature_groups = {}
        for win in KNOWN_GAP_WINDOWS:
            gap_end = pd.Timestamp(win["end"])
            grace_end = gap_end + pd.Timedelta(days=LAG_GRACE_DAYS)
            if tail["date"].max() <= grace_end:
                for col in feature_cols:
                    if feature_groups.get(col) in win["families"]:
                        grace_relaxed[col] = win["note"]

    failures = []
    report_rows = []
    for col in feature_cols:
        tm, hm = tail_missing[col], hist_missing.get(col, 0.0)
        report_rows.append((col, hm, tm))
        if col in grace_relaxed:
            continue  # within a documented gap's grace period -- expected, not a new failure
        ceiling = KNOWN_GAP_COLUMNS.get(col, MAX_UNEXPECTED_MISSING)
        # Fail only if the live tail both exceeds the absolute ceiling AND is
        # meaningfully worse than this feature's own historical rate -- a tail
        # that merely matches its historical missingness (e.g. a sensor gap
        # that has always been there) is not a new ingestion failure.
        if tm > ceiling and tm > hm + BASELINE_MARGIN:
            failures.append((col, hm, tm, ceiling, col in KNOWN_GAP_COLUMNS))

    if grace_relaxed:
        print(f"\n{len(grace_relaxed)} feature(s) exempted this run -- within a documented gap's "
              f"grace period:")
        for col, note in list(grace_relaxed.items())[:10]:
            print(f"  - {col}: {note}")
        if len(grace_relaxed) > 10:
            print(f"  ... and {len(grace_relaxed) - 10} more")

    report = pd.DataFrame(report_rows, columns=["feature", "historical_missing", "live_tail_missing"])
    report = report.sort_values("live_tail_missing", ascending=False)
    print("\nTop 15 features by live-tail missingness:")
    print(report.head(15).to_string(index=False, float_format=lambda x: f"{x:.1%}"))

    out_path = Path("outputs")
    out_path.mkdir(exist_ok=True)
    report.to_csv(out_path / "data_quality_report.csv", index=False)
    print(f"\nFull report written to {out_path / 'data_quality_report.csv'}")

    if failures:
        print(f"\nFAIL: {len(failures)} feature(s) exceed their missingness ceiling in the live tail:")
        for col, hm, tm, ceiling, known in failures:
            tag = "documented gap, but worse than expected" if known else "UNEXPECTED gap"
            print(f"  - {col}: historical={hm:.1%} live_tail={tm:.1%} ceiling={ceiling:.1%} ({tag})")
        print("\nThis does not mean the pipeline is unusable -- it means a feature's live coverage has "
              "drifted past what's accounted for. Decide per Finding 5: get a compatible source, drop "
              "the feature and retrain, or raise its ceiling here if the new rate is actually expected.")
        return 1

    print("\nOK: all features within their expected live-tail missingness ceiling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
