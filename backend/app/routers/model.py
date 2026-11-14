from fastapi import APIRouter, HTTPException

from .. import config
from ..data_loader import read_json

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/weights", response_model=None)
def ensemble_weights(target: str | None = None):
    """The ensemble weights (per member, per horizon) actually used to produce
    the forecast currently being served -- written by predict_forecast.py,
    copied straight from the file it loaded, so this always matches reality
    even if ensemble_weights_<target>.json is re-tuned later."""
    data = read_json(config.ENSEMBLE_WEIGHTS_USED_JSON, "rebuild_forecast.yml (predict_forecast.py)")
    if target is not None:
        if target not in data:
            raise HTTPException(status_code=404, detail=f"Unknown target {target!r}. "
                                                          f"Valid targets: {list(data)}")
        return {"target": target, "weights_by_horizon": data[target]}
    return [{"target": t, "weights_by_horizon": w} for t, w in data.items()]


@router.get("/info", response_model=None)
def model_info():
    """When the currently-deployed models were last retrained and up to what
    date, per target -- written by train_production_models.py on its
    ~30-day schedule."""
    return read_json(config.TRAINED_AT_JSON, "retrain_model.yml (train_production_models.py)")
