"""
main.py
AirGuard API -- a thin, read-only HTTP layer over the files
rebuild_forecast.yml / retrain_model.yml already commit to the repo
(outputs/, features/, params/). It never fits or predicts anything itself;
see ..data_loader for why it deliberately avoids importing model_common.py.

Run locally (from backend/):
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000

Interactive docs then at http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .data_loader import FileNotFoundDataError
from .routers import forecast, history, ingestion, meta, model

app = FastAPI(
    title="AirGuard API",
    description="Delhi NCR 10-day AQI / PM2.5 / ozone / NOx forecast -- read-only over the "
                "pipeline's committed outputs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(FileNotFoundDataError)
def missing_file_handler(request, exc: FileNotFoundDataError):
    # A pipeline output not existing yet (fresh clone, workflow hasn't run) is an
    # expected, recoverable state -- 503 (service not ready) with a clear pointer
    # to what produces the file, not a 500.
    return JSONResponse(status_code=503, content={"detail": str(exc)})


app.include_router(meta.router, prefix="/api")
app.include_router(forecast.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(model.router, prefix="/api")
app.include_router(ingestion.router, prefix="/api")


@app.get("/")
def root():
    return {"name": "AirGuard API", "docs": "/docs", "health": "/api/health"}
