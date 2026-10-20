"""
db.py
Postgres engine and a generic "upsert a DataFrame" helper used by every
fetch_*.py script, so each script only has to build a DataFrame and name the
primary-key columns.
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from config import DATABASE_URL

SCHEMA_SQL = Path(__file__).with_name("schema.sql")


def get_engine():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    url = DATABASE_URL
    # Some managed Postgres providers (older Heroku-style URLs, and a few others)
    # still hand out "postgres://" rather than "postgresql://". SQLAlchemy 2.0's
    # postgres dialect plugin was removed, so create_engine() raises
    # NoSuchModuleError on that scheme before ever attempting to connect --
    # normalize it here so the one place every script gets its engine from
    # doesn't depend on which exact scheme the provider happened to give you.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    try:
        return create_engine(url)
    except Exception as exc:
        # create_engine() only validates the URL string (it does not connect yet), so a
        # malformed DATABASE_URL -- e.g. a typo'd scheme or missing "://" -- fails here
        # with a clear message instead of surfacing later as an unrelated-looking error
        # from whatever the first query happens to be.
        raise RuntimeError(f"Could not create a database engine from DATABASE_URL: {exc}") from exc


def _split_sql_statements(sql):
    """Split a .sql file into individual statements on ';', ignoring any ';'
    that appears inside a '--' line comment or a '...' string literal.

    A plain `.split(";")` breaks the moment a comment contains a semicolon
    (as ours did) -- the fragment after it is no longer valid SQL, which
    fails inside init_schema()'s single transaction and silently rolls back
    every table, not just the ones after the bad split point. This scans
    char-by-char instead, so it is safe regardless of what punctuation ends
    up in a future comment or string.
    """
    statements = []
    buf = []
    in_line_comment = False
    in_string = False
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_string:
            buf.append(ch)
            if ch == "'" and nxt == "'":   # escaped '' inside a string literal
                buf.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_string = False
            i += 1
            continue
        if ch == "-" and nxt == "-":
            in_line_comment = True
            buf.append(ch)
            i += 1
            continue
        if ch == "'":
            in_string = True
            buf.append(ch)
            i += 1
            continue
        if ch == ";":
            statements.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return [s.strip() for s in statements if s.strip()]


def init_schema(engine):
    """Create every table if it does not exist yet. Safe to call every run."""
    try:
        schema_sql = SCHEMA_SQL.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"schema.sql not found at {SCHEMA_SQL} -- expected next to db.py.") from exc
    statements = _split_sql_statements(schema_sql)
    try:
        with engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
    except OperationalError as exc:
        raise ConnectionError(
            f"Could not connect to/execute against the database while creating the schema. "
            f"Check DATABASE_URL and that the database is reachable. Original error: {exc}"
        ) from exc
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Failed to initialise the database schema: {exc}") from exc


def _quote_ident(name):
    """Double-quote a Postgres identifier, escaping embedded quotes. Every table
    and column name this module emits goes through this -- station_hourly_hist
    has mixed-case columns ("PM25", "NOx", ...) which Postgres folds to lowercase
    if left unquoted, so an unquoted INSERT/UPDATE silently targets the wrong
    (nonexistent, or wrong-case) column instead of raising."""
    return '"' + str(name).replace('"', '""') + '"'


def upsert_dataframe(engine, table, df, pk_cols, batch_size=5000):
    """Insert df into `table`, updating any row whose primary key already exists
    (an "upsert"). Uses a temp table plus INSERT ... ON CONFLICT, the standard
    pattern for pandas + Postgres since pandas.to_sql has no native upsert.

    df's columns must be a subset of the table's columns and include pk_cols.
    Returns the number of rows sent (not the number actually changed).
    """
    if df is None:
        raise TypeError(f"upsert_dataframe({table!r}, ...) got df=None -- expected a DataFrame (possibly empty).")
    if not pk_cols:
        raise ValueError(f"upsert_dataframe({table!r}, ...) needs at least one primary-key column in pk_cols.")
    missing_pk = [c for c in pk_cols if c not in df.columns]
    if missing_pk:
        raise ValueError(f"upsert_dataframe({table!r}, ...): pk_cols {missing_pk} not present in the "
                          f"DataFrame's columns {list(df.columns)}.")
    if df.empty:
        return 0
    # Cast to object first: df.where(pd.notnull(df), None) on a float64 column
    # casts straight back to float64 and silently turns the Nones back into
    # NaN, which to_sql then writes as NaN (not SQL NULL) into float columns.
    df = df.astype(object).where(pd.notnull(df), None)
    cols = list(df.columns)
    update_cols = [c for c in cols if c not in pk_cols]
    q_cols = [_quote_ident(c) for c in cols]
    q_pk = [_quote_ident(c) for c in pk_cols]
    set_clause = ", ".join(f'{_quote_ident(c)} = EXCLUDED.{_quote_ident(c)}' for c in update_cols) or None
    q_table = _quote_ident(table)
    tmp = f"_tmp_{table}"
    q_tmp = _quote_ident(tmp)

    try:
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS {q_tmp}'))
            conn.execute(text(f'CREATE TEMP TABLE {q_tmp} (LIKE {q_table} INCLUDING DEFAULTS)'))
            df.to_sql(tmp, conn, if_exists="append", index=False, chunksize=batch_size, method="multi")
            col_list = ", ".join(q_cols)
            pk_list = ", ".join(q_pk)
            upsert_sql = f'INSERT INTO {q_table} ({col_list}) SELECT {col_list} FROM {q_tmp} ' \
                         f'ON CONFLICT ({pk_list}) DO '
            upsert_sql += f'UPDATE SET {set_clause}' if set_clause else 'NOTHING'
            conn.execute(text(upsert_sql))
            conn.execute(text(f'DROP TABLE IF EXISTS {q_tmp}'))
    except OperationalError as exc:
        raise ConnectionError(
            f"Could not reach the database while upserting {len(df)} row(s) into {table!r}. "
            f"Check DATABASE_URL and that the database is reachable. Original error: {exc}"
        ) from exc
    except SQLAlchemyError as exc:
        raise RuntimeError(
            f"Upsert into {table!r} failed ({len(df)} row(s), pk_cols={pk_cols}). This usually means the "
            f"DataFrame's columns don't match the table's schema (wrong name, wrong type, or a NOT NULL "
            f"column left out). Original error: {exc}"
        ) from exc
    return len(df)


def log_run(engine, job, started_at, finished_at, rows_written, status, message=""):
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO ingestion_log (job, started_at, finished_at, rows_written, status, message) "
                     "VALUES (:job, :started_at, :finished_at, :rows_written, :status, :message)"),
                {"job": job, "started_at": started_at, "finished_at": finished_at,
                 "rows_written": rows_written, "status": status, "message": (message or "")[:2000]},
            )
    except SQLAlchemyError as exc:
        # Logging the run is important for monitoring (check_retrain_due.py reads this
        # table) but it must never be what fails an otherwise-successful ingestion run,
        # so this is deliberately a warning, not a re-raise.
        print(f"WARNING: could not write to ingestion_log for job={job!r} status={status!r}: {exc}")
