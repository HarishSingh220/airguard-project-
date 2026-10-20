"""
io_utils.py
Shared helper for reading the CSV inputs used across the feature-building
scripts (build_features.py, era5_inputs.py, cams_inputs.py, fire_inputs.py).

Every CSV here is a required pipeline input. Previously a missing, empty, or
malformed file surfaced as a generic pandas/OS exception (or, worse, a
KeyError several function calls later the first time a missing column was
looked up inside a groupby). read_csv_checked() turns all of those into one
exception type -- DataFileError -- with a message that names the file and
says exactly what is wrong with it, so a broken pipeline run points straight
at the input that needs fixing.
"""
from pathlib import Path

import pandas as pd


class DataFileError(RuntimeError):
    """A required CSV input is missing, empty, unreadable, or missing a
    column the pipeline depends on."""


def read_csv_checked(path, required_cols=None, parse_dates=None, **kwargs):
    """pandas.read_csv with error messages a person can act on.

    Raises DataFileError -- instead of FileNotFoundError / EmptyDataError /
    ParserError / a bare ValueError -- if the file does not exist, is empty,
    fails to parse, parses to zero rows, or is missing any column named in
    required_cols.
    """
    path = Path(path)
    if not path.exists():
        raise DataFileError(
            f"Required input file not found: {path} (looked under {path.resolve().parent}). "
            f"Check that you're running this from the repository root and that the file "
            f"has already been fetched or generated."
        )
    if path.stat().st_size == 0:
        raise DataFileError(f"{path} exists but is empty (0 bytes).")
    try:
        df = pd.read_csv(path, parse_dates=parse_dates, **kwargs)
    except pd.errors.EmptyDataError as exc:
        raise DataFileError(f"{path} has no columns / no data to parse.") from exc
    except pd.errors.ParserError as exc:
        raise DataFileError(f"{path} could not be parsed as CSV: {exc}") from exc
    except ValueError as exc:
        # e.g. a column named in parse_dates isn't actually present in the file
        raise DataFileError(f"{path} could not be read: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DataFileError(f"{path} is not readable as text (bad encoding): {exc}") from exc
    if df.empty:
        raise DataFileError(f"{path} parsed but has zero rows.")
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise DataFileError(
                f"{path} is missing required column(s) {missing}. "
                f"Columns present: {list(df.columns)}"
            )
    return df
