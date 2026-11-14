"""
Optional connection to the same PostgreSQL database used by ingestion.
Only /api/ingestion/status requires it; file-backed forecast endpoints work
without DATABASE_URL.
"""
import os
from functools import lru_cache
from pathlib import Path

# Load local repo-root environment for `uvicorn` started from backend/.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass


def _database_url() -> str:
    value = os.environ.get("DATABASE_URL", "").strip()
    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://"):]
    if value.startswith("postgresql+psycopg2://") or value.startswith("postgresql+psycopg://"):
        value = "postgresql://" + value.split("://", 1)[1]
    return value


@lru_cache(maxsize=1)
def get_engine():
    url = _database_url()
    if not url:
        return None
    from sqlalchemy import create_engine
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"connect_timeout": 10},
    )
