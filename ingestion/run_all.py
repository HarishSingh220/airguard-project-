"""
run_all.py
Runs the four fetchers in sequence. One source failing (a changed CPCB station
name, a FIRMS or ADS outage) does not stop the others; failures are collected
and raised together at the end so the GitHub Actions run is still marked
failed and visible, but partial data for the run is not lost.

This only catches what each job's main() raises -- a job that logs "ok" (or
"ok (partial)") and returns normally is treated as a success here, so each
fetch_*.py script is responsible for actually raising when it has nothing
usable to report (see fetch_firms.py's all-sources-failed case) rather than
swallowing a total failure into a status string nothing else reads.
"""
import sys
import traceback

import fetch_cams
import fetch_cpcb
import fetch_firms
import fetch_openmeteo

JOBS = [("cpcb", fetch_cpcb.main), ("firms", fetch_firms.main),
       ("openmeteo", fetch_openmeteo.main), ("cams", fetch_cams.main)]


def main():
    failures = []
    for name, fn in JOBS:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception:
            print(f"FAILED: {name}")
            traceback.print_exc()
            failures.append(name)
    if failures:
        print(f"\n{len(failures)} of {len(JOBS)} jobs failed: {failures}")
        sys.exit(1)
    print("\nAll ingestion jobs completed.")


if __name__ == "__main__":
    main()
