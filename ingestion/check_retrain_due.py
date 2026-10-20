"""
check_retrain_due.py
Gate for the 30-day retrain workflow (.github/workflows/retrain_model.yml).

Looks at ingestion_log for the most recent row with job='retrain_models' and
status='ok' -- written by train_production_models.py at the end of a
successful retrain -- and compares its finished_at to now(). If at least
RETRAIN_INTERVAL_DAYS have passed, or no successful retrain has ever been
logged, this is "due": writes due=true to $GITHUB_OUTPUT, and the workflow's
remaining steps (which all have `if: steps.gate.outputs.due == 'true'`) run.
Otherwise due=false and they're skipped.

This checks the database rather than relying on a fixed day-of-month cron so
a missed or failed run doesn't permanently shift the schedule: if a run is
skipped (outage, quota, whatever), the next scheduled check simply finds the
gap is now >30 days and retrains then, rather than waiting for a calendar
date it already missed.

Run from ingestion/ (same directory as db.py/config.py):
    python check_retrain_due.py

FORCE_RETRAIN=true (set from the workflow_dispatch "force" input) always
answers due=true, skipping the database check entirely -- useful for the very
first run, when nothing has been trained yet and a query would find no rows
anyway, and for forcing an out-of-cycle retrain on demand.
"""
import os
from datetime import datetime, timezone

from sqlalchemy import text

from db import get_engine

RETRAIN_INTERVAL_DAYS = 30


def main():
    force = os.environ.get("FORCE_RETRAIN", "").strip().lower() in ("1", "true", "yes")
    github_output = os.environ.get("GITHUB_OUTPUT")

    if force:
        due, reason = True, "forced via workflow_dispatch input"
    else:
        engine = get_engine()
        with engine.connect() as conn:
            last = conn.execute(text(
                "SELECT finished_at FROM ingestion_log "
                "WHERE job = 'retrain_models' AND status = 'ok' "
                "ORDER BY finished_at DESC LIMIT 1"
            )).scalar()
        if last is None:
            due, reason = True, "no successful retrain has been logged yet"
        else:
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - last).total_seconds() / 86400
            due = age_days >= RETRAIN_INTERVAL_DAYS
            reason = (f"last successful retrain was {age_days:.1f} days ago "
                      f"(threshold {RETRAIN_INTERVAL_DAYS})")

    print(f"retrain due: {due} ({reason})")
    if github_output:
        with open(github_output, "a") as fh:
            fh.write(f"due={'true' if due else 'false'}\n")


if __name__ == "__main__":
    main()
