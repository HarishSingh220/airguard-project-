"""
data_loader.py
Reads the CSV/JSON files the pipeline already produces (outputs/, features/,
params/) and caches each one in memory, keyed on the file's mtime -- so a
fresh commit from rebuild_forecast.yml or retrain_model.yml (new file on
disk / new deploy) is picked up on the next request with no restart needed,
while repeated requests in between don't re-parse the CSV every time.

Every loader raises a FileNotFoundDataError (not a bare FileNotFoundError)
for a missing file, which routers turn into a clean 503 with a message
pointing at which GitHub Actions workflow produces that file -- the API
should never 500 just because the pipeline hasn't run yet on a fresh clone.
"""
import json
import threading
from pathlib import Path

import pandas as pd

_lock = threading.Lock()
_cache: dict[Path, tuple[float, object]] = {}


class FileNotFoundDataError(FileNotFoundError):
    """A pipeline output file is missing. Carries the workflow that produces it."""

    def __init__(self, path: Path, produced_by: str):
        self.path = path
        self.produced_by = produced_by
        super().__init__(
            f"{path} does not exist yet. It is produced by {produced_by}. "
            f"Run that workflow (or the equivalent script locally) first."
        )


def _cached(path: Path, loader, produced_by: str):
    if not path.exists():
        raise FileNotFoundDataError(path, produced_by)
    mtime = path.stat().st_mtime
    with _lock:
        hit = _cache.get(path)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    value = loader(path)
    with _lock:
        _cache[path] = (mtime, value)
    return value


def read_csv(path: Path, produced_by: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    def _load(p):
        return pd.read_csv(p, parse_dates=parse_dates)
    # DataFrames are mutated by callers (filtering, renaming) in some routers, so
    # hand back a copy of the cached frame rather than the cached object itself.
    return _cached(path, _load, produced_by).copy(deep=True)


def read_json(path: Path, produced_by: str):
    def _load(p):
        return json.loads(p.read_text())
    return _cached(path, _load, produced_by)


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts: date/datetime columns become ISO
    date strings, and NaN (which plain json.dumps emits as the non-standard
    literal `NaN`, breaking strict JSON parsers like the browser's) becomes
    None/null.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.date.astype(str)
    df = df.astype(object).where(pd.notnull(df), None)
    return df.to_dict(orient="records")


def file_mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    import datetime
    return datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc).isoformat()
