"""
Preflight for the forecasting/retraining workflows.

Run from the repository root:
    python ingestion/validate_database.py

Creates the application schema if needed, then requires the five migrated
historical source tables to contain rows. This prevents a fresh/partially
migrated Postgres database from producing empty CSV exports that fail much
later inside feature engineering.
"""
from sqlalchemy import text

from db import get_engine, init_schema

REQUIRED_HISTORICAL_TABLES = (
    "station_hourly_hist",
    "era5_surface_hist",
    "era5_pressure6h_hist",
    "era5_pressure12h_hist",
    "cams_3h_hist",
)


def main():
    engine = get_engine()
    init_schema(engine)

    counts = {}
    with engine.connect() as conn:
        for table in REQUIRED_HISTORICAL_TABLES:
            # Table names are fixed constants above, never user-provided input.
            counts[table] = int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())

    print("Historical PostgreSQL row counts:")
    for table, count in counts.items():
        print(f"  {table}: {count:,}")

    empty = [table for table, count in counts.items() if count <= 0]
    if empty:
        raise SystemExit(
            "Database preflight failed: required historical tables are empty: "
            + ", ".join(empty)
            + ". Generate final_delhi_dataset.csv from data_preprosessing.ipynb, "
              "ensure the ERA5/CAMS historical CSVs are present, then run "
              "`python ingestion/migrate_historical.py --yes` once."
        )

    print("Database preflight passed: schema exists and all historical tables contain data.")


if __name__ == "__main__":
    main()
