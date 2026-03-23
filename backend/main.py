"""FastAPI application for the Investment XGBoost prediction service."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import FUND_TICKERS, COMPARISON_LABELS, HORIZON_LABELS
from data_fetcher import clear_cache
from model import train_all_models, load_models, get_predictions, get_historical_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

model_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_store
    logger.info("Starting up — loading or training models...")
    store = load_models()
    if store is None:
        logger.info("No saved models found, training from scratch")
        store = train_all_models()
    model_store = store
    logger.info(f"Ready with {len(model_store)} models")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Investment XGBoost API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/predictions")
async def predictions_endpoint():
    """Return predictions for all comparisons and horizons."""
    if not model_store:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    results = get_predictions(model_store)
    return [
        {
            "comparison": r.comparison,
            "comparisonLabel": COMPARISON_LABELS.get(r.comparison, r.comparison),
            "horizon": r.horizon,
            "horizonLabel": HORIZON_LABELS.get(r.horizon, r.horizon),
            "prediction": r.prediction,
            "confidence": r.confidence,
            "recommendation": r.recommendation,
            "direction": r.direction,
            "magnitude": r.magnitude,
        }
        for r in results
    ]


@app.get("/api/historical")
async def historical_endpoint(comparison: str = Query(default="value_growth")):
    """Return historical performance data for a comparison pair."""
    if comparison not in FUND_TICKERS:
        raise HTTPException(status_code=400, detail=f"Unknown comparison: {comparison}")
    return get_historical_data(comparison)


@app.get("/api/feature-importance")
async def feature_importance_endpoint(
    comparison: str = Query(default="value_growth"),
    horizon: str = Query(default="6m"),
):
    """Return feature importance for a specific model."""
    key = (comparison, horizon)
    if key not in model_store:
        raise HTTPException(status_code=404, detail=f"Model not found: {comparison}/{horizon}")

    importance = model_store[key]["importance"]
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    return [{"feature": name, "importance": round(val, 4)} for name, val in sorted_features[:15]]


@app.get("/api/model-metrics")
async def model_metrics_endpoint():
    """Return backtest metrics for all models."""
    metrics = []
    for (comp, horizon), entry in model_store.items():
        m = entry["metrics"]
        metrics.append({
            "comparison": comp,
            "comparisonLabel": COMPARISON_LABELS.get(comp, comp),
            "horizon": horizon,
            "horizonLabel": HORIZON_LABELS.get(horizon, horizon),
            **m,
        })
    return metrics


@app.post("/api/retrain")
async def retrain_endpoint():
    """Force data refresh and model retraining."""
    global model_store
    clear_cache()
    model_store = train_all_models(force_refresh=True)
    return {"status": "ok", "models_trained": len(model_store)}
