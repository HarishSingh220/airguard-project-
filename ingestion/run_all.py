"""
run_all.py
Runs three fetchers in sequence: firms, openmeteo, cams. One source failing
(a FIRMS or ADS outage) does not stop the others; failures are collected and
raised together at the end so the GitHub Actions run is still marked failed
and visible, but partial data for the run is not lost.

CPCB is deliberately NOT run here -- it lives solely in the separate, much
more frequent ingest_cpcb.yml workflow (every 15 minutes, vs this file's
3-hourly schedule). Unlike the other three, CPCB's API has no history/backfill
endpoint -- it only ever returns each station's CURRENT reading -- so a missed
or infrequent poll is a permanent, unrecoverable gap for that window, which
the other three self-heal from on their own rolling windows. Running it here
too would just double-spend CPCB_API_KEY's quota for no benefit the frequent
workflow doesn't already cover better, and would reintroduce the risk of its
retry budget eating into this run's share of ingest.yml's 20-minute timeout.

This only catches what each job's main() raises -- a job that logs "ok" (or
"ok (partial)") and returns normally is treated as a success here, so each
fetch_*.py script is responsible for actually raising when it has nothing
usable to report (see fetch_firms.py's all-sources-failed case) rather than
swallowing a total failure into a status string nothing else reads.
"""
import os
import sys
import time
import traceback

import fetch_cams
import fetch_firms
import fetch_openmeteo

JOBS = [("firms", fetch_firms.main), ("openmeteo", fetch_openmeteo.main),
       ("cams", fetch_cams.main)]


def _write_step_summary(results):
    """results: [(name, ok, duration_seconds), ...]. Writes a per-job markdown
    table to $GITHUB_STEP_SUMMARY, which GitHub Actions renders directly on the
    run's summary page -- so e.g. "firms/openmeteo actually landed rows even
    though this run shows red overall" is visible at a glance instead of only
    in the raw log. GITHUB_STEP_SUMMARY is unset outside Actions (local runs,
    tests), in which case this is a silent no-op; it never affects run_all.py's
    own exit code, which should keep reflecting a real failure as a real
    failure."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = ["## Ingestion results", "", "| Job | Status | Duration |", "| --- | --- | --- |"]
    for name, ok, duration in results:
        status = "✅ ok" if ok else "❌ FAILED"
        lines.append(f"| {name} | {status} | {duration:.0f}s |")
    with open(summary_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def main():
    failures = []
    results = []
    for name, fn in JOBS:
        print(f"\n=== {name} ===")
        started = time.monotonic()
        try:
            fn()
            results.append((name, True, time.monotonic() - started))
        except Exception:
            print(f"FAILED: {name}")
            traceback.print_exc()
            failures.append(name)
            results.append((name, False, time.monotonic() - started))
    _write_step_summary(results)
    if failures:
        print(f"\n{len(failures)} of {len(JOBS)} jobs failed: {failures}")
        sys.exit(1)
    print("\nAll ingestion jobs completed.")


if __name__ == "__main__":
    main()