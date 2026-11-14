from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .. import db

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


def _read_ingestion_rows(sql: str, params: dict):
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL is not configured on this API deployment, so ingestion status is unavailable.",
        )
    try:
        with engine.connect() as conn:
            return [dict(row) for row in conn.execute(text(sql), params).mappings().all()]
    except SQLAlchemyError as exc:
        # Do not return connection strings or provider exception details to
        # public API clients. The action logs retain the full diagnostic.
        raise HTTPException(
            status_code=503,
            detail=(
                "The AirGuard database could not serve ingestion status. "
                "Check DATABASE_URL, database reachability, and that ingestion/schema.sql "
                "has been initialized by an ingestion job or deployment preflight."
            ),
        ) from exc


@router.get("/status", response_model=None)
def ingestion_status(
    job: str | None = Query(None, description="Filter to one job, e.g. 'fetch_cpcb', 'retrain_models'"),
    limit: int = Query(20, ge=1, le=200),
):
    """Most recent rows from ingestion_log."""
    sql = (
        "SELECT job, started_at, finished_at, rows_written, status, message "
        "FROM ingestion_log "
        + ("WHERE job = :job " if job else "")
        + "ORDER BY started_at DESC LIMIT :limit"
    )
    params = {"limit": limit, **({"job": job} if job else {})}
    return _read_ingestion_rows(sql, params)


@router.get("/status/latest-per-job", response_model=None)
def latest_per_job():
    """The latest record for each distinct ingestion/training job."""
    sql = (
        "SELECT DISTINCT ON (job) job, started_at, finished_at, rows_written, status, message "
        "FROM ingestion_log ORDER BY job, started_at DESC"
    )
    return _read_ingestion_rows(sql, {})
