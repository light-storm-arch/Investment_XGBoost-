"""Model training, walk-forward validation, and prediction logic."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from config import (
    FUND_TICKERS,
    COMPARISON_LABELS,
    HORIZON_LABELS,
    MAGNITUDE_THRESHOLDS,
    XGBOOST_PARAMS,
    WF_N_SPLITS,
    WF_MIN_TRAIN_DAYS,
)
from data_fetcher import get_combined_dataset, clear_cache
from features import build_features, get_feature_columns, get_target_columns

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "trained_models.pkl"
HORIZON_KEYS = ["1m", "3m", "6m", "12m"]


@dataclass
class ModelResult:
    comparison: str
    horizon: str
    prediction: float
    confidence: str
    recommendation: str
    direction: str
    magnitude: str
    feature_importance: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


def _walk_forward_validate(X: pd.DataFrame, y: pd.Series, n_splits: int, min_train: int) -> dict:
    """Walk-forward validation returning aggregate metrics."""
    n = len(X)
    if n < min_train + n_splits:
        # Not enough data for proper walk-forward, fall back to simple split
        min_train = max(int(n * 0.6), 30)

    test_size = n - min_train
    step = max(test_size // n_splits, 1)

    all_preds = []
    all_actuals = []

    for i in range(n_splits):
        split_point = min_train + i * step
        if split_point >= n:
            break

        end_point = min(split_point + step, n)

        X_train = X.iloc[:split_point]
        y_train = y.iloc[:split_point]
        X_test = X.iloc[split_point:end_point]
        y_test = y.iloc[split_point:end_point]

        if len(X_train) < 20 or len(X_test) == 0:
            continue

        # Train with early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

        model = XGBRegressor(**XGBOOST_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_test)
        all_preds.extend(preds)
        all_actuals.extend(y_test.values)

    if not all_preds:
        return {"mae": np.nan, "rmse": np.nan, "directional_accuracy": 0.5, "ic": 0.0}

    preds_arr = np.array(all_preds)
    actuals_arr = np.array(all_actuals)

    mae = np.mean(np.abs(preds_arr - actuals_arr))
    rmse = np.sqrt(np.mean((preds_arr - actuals_arr) ** 2))

    # Directional accuracy: did we predict the correct sign?
    correct_direction = np.sign(preds_arr) == np.sign(actuals_arr)
    directional_accuracy = np.mean(correct_direction)

    # Information coefficient (rank correlation)
    if len(preds_arr) > 2:
        ic, _ = spearmanr(preds_arr, actuals_arr)
        ic = 0.0 if np.isnan(ic) else ic
    else:
        ic = 0.0

    return {
        "mae": round(float(mae), 6),
        "rmse": round(float(rmse), 6),
        "directional_accuracy": round(float(directional_accuracy), 4),
        "ic": round(float(ic), 4),
    }


def _determine_confidence(directional_accuracy: float, prediction: float, hist_std: float) -> str:
    abs_z = abs(prediction / hist_std) if hist_std > 0 else 0
    if directional_accuracy >= 0.60 and abs_z > 1.0:
        return "high"
    if directional_accuracy >= 0.55 or abs_z > 0.5:
        return "medium"
    return "low"


def _prediction_to_recommendation(prediction: float, comparison_key: str, hist_std: float) -> tuple[str, str, str]:
    """Convert a raw prediction to (recommendation, direction, magnitude)."""
    pair = FUND_TICKERS[comparison_key]
    direction = pair["a_label"].split("(")[0].strip() if prediction > 0 else pair["b_label"].split("(")[0].strip()

    abs_pred = abs(prediction)
    magnitude = "strong"
    rec_detail = ""
    for threshold, mag_name, detail in MAGNITUDE_THRESHOLDS:
        if abs_pred < threshold:
            magnitude = mag_name
            rec_detail = detail
            break

    if magnitude == "negligible":
        recommendation = f"Neutral — {rec_detail}"
    else:
        recommendation = f"{magnitude.title()} {direction} tilt — {rec_detail}"

    return recommendation, direction, magnitude


def train_all_models(force_refresh: bool = False) -> dict:
    """Train all 8 models (2 comparisons x 4 horizons). Returns model store dict."""
    combined = get_combined_dataset(force_refresh=force_refresh)
    comparisons = list(FUND_TICKERS.keys())
    store = {}

    for comp_key in comparisons:
        logger.info(f"Building features for {comp_key}")
        feat_df = build_features(combined, comp_key)
        feature_cols = get_feature_columns(feat_df)

        for horizon_key in HORIZON_KEYS:
            target_col = f"fwd_spread_{horizon_key}"
            if target_col not in feat_df.columns:
                continue

            valid = feat_df[target_col].notna()
            X = feat_df.loc[valid, feature_cols].copy()
            y = feat_df.loc[valid, target_col].copy()

            if len(X) < 30:
                logger.warning(f"Skipping {comp_key}/{horizon_key}: only {len(X)} samples")
                continue

            # Walk-forward validation
            metrics = _walk_forward_validate(X, y, WF_N_SPLITS, WF_MIN_TRAIN_DAYS)
            logger.info(f"{comp_key}/{horizon_key} — DA: {metrics['directional_accuracy']:.1%}, IC: {metrics['ic']:.3f}")

            # Train final model on all data
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)
            final_model = XGBRegressor(**XGBOOST_PARAMS)
            final_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            importance = dict(zip(X.columns, final_model.feature_importances_))
            hist_std = float(y.std())

            store[(comp_key, horizon_key)] = {
                "model": final_model,
                "metrics": metrics,
                "importance": importance,
                "hist_std": hist_std,
                "feature_cols": feature_cols,
            }

    # Save to disk
    joblib.dump(store, MODEL_PATH)
    logger.info(f"Saved {len(store)} models to {MODEL_PATH}")
    return store


def _walk_forward_detail(X: pd.DataFrame, y: pd.Series, n_splits: int,
                         min_train: int, xgb_params: dict | None = None) -> pd.DataFrame:
    """Walk-forward validation returning per-row predictions with dates."""
    params = xgb_params or XGBOOST_PARAMS
    n = len(X)
    if n < min_train + n_splits:
        min_train = max(int(n * 0.6), 30)

    test_size = n - min_train
    step = max(test_size // n_splits, 1)

    records = []
    for i in range(n_splits):
        split_point = min_train + i * step
        if split_point >= n:
            break
        end_point = min(split_point + step, n)

        X_train = X.iloc[:split_point]
        y_train = y.iloc[:split_point]
        X_test = X.iloc[split_point:end_point]
        y_test = y.iloc[split_point:end_point]

        if len(X_train) < 20 or len(X_test) == 0:
            continue

        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_test)
        for date, pred, actual in zip(X_test.index, preds, y_test.values):
            records.append({"date": date, "predicted": float(pred), "actual": float(actual)})

    return pd.DataFrame(records)


def train_custom_split(comparison_key: str, horizon_key: str, cutoff_date: str,
                       xgb_params: dict | None = None) -> dict:
    """Train on data before cutoff_date, test on data after. Returns detail dict."""
    params = xgb_params or XGBOOST_PARAMS
    combined = get_combined_dataset()
    feat_df = build_features(combined, comparison_key)
    feature_cols = get_feature_columns(feat_df)
    target_col = f"fwd_spread_{horizon_key}"

    if target_col not in feat_df.columns:
        return {"error": "Target column not found"}

    valid = feat_df[target_col].notna()
    X = feat_df.loc[valid, feature_cols].copy()
    y = feat_df.loc[valid, target_col].copy()

    cutoff = pd.Timestamp(cutoff_date)
    train_mask = X.index <= cutoff
    test_mask = X.index > cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) < 30:
        return {"error": f"Only {len(X_train)} training samples before cutoff"}
    if len(X_test) < 1:
        return {"error": "No test samples after cutoff"}

    # Train with early stopping using last 15% of training data as validation
    split_idx = max(int(len(X_train) * 0.85), 20)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_test)
    preds_arr = np.array(preds)
    actuals_arr = y_test.values

    mae = float(np.mean(np.abs(preds_arr - actuals_arr)))
    rmse = float(np.sqrt(np.mean((preds_arr - actuals_arr) ** 2)))
    da = float(np.mean(np.sign(preds_arr) == np.sign(actuals_arr)))
    ic = 0.0
    if len(preds_arr) > 2:
        ic_val, _ = spearmanr(preds_arr, actuals_arr)
        ic = 0.0 if np.isnan(ic_val) else float(ic_val)

    # Cumulative signal P&L: go long spread when predicted > 0, short when < 0
    signal = np.sign(preds_arr)
    strategy_returns = signal * actuals_arr
    cum_strategy = np.cumsum(strategy_returns)
    cum_buyhold = np.cumsum(actuals_arr)

    detail_df = pd.DataFrame({
        "date": X_test.index,
        "predicted": preds_arr,
        "actual": actuals_arr,
        "signal": signal,
        "strategy_return": strategy_returns,
        "cum_strategy": cum_strategy,
        "cum_buyhold": cum_buyhold,
    })

    importance = dict(zip(X_test.columns, model.feature_importances_))

    return {
        "detail": detail_df,
        "metrics": {"mae": round(mae, 6), "rmse": round(rmse, 6),
                    "directional_accuracy": round(da, 4), "ic": round(ic, 4)},
        "importance": importance,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def get_walkforward_detail(comparison_key: str, horizon_key: str,
                           xgb_params: dict | None = None) -> pd.DataFrame:
    """Return walk-forward prediction details for visualization."""
    combined = get_combined_dataset()
    feat_df = build_features(combined, comparison_key)
    feature_cols = get_feature_columns(feat_df)
    target_col = f"fwd_spread_{horizon_key}"

    if target_col not in feat_df.columns:
        return pd.DataFrame()

    valid = feat_df[target_col].notna()
    X = feat_df.loc[valid, feature_cols].copy()
    y = feat_df.loc[valid, target_col].copy()

    if len(X) < 30:
        return pd.DataFrame()

    return _walk_forward_detail(X, y, WF_N_SPLITS, WF_MIN_TRAIN_DAYS, xgb_params)


def load_models() -> dict | None:
    """Load previously trained models from disk."""
    if MODEL_PATH.exists():
        logger.info("Loading models from disk")
        return joblib.load(MODEL_PATH)
    return None


def get_predictions(store: dict) -> list[ModelResult]:
    """Generate predictions for all models using the latest available features."""
    combined = get_combined_dataset()
    results = []

    for (comp_key, horizon_key), entry in store.items():
        feat_df = build_features(combined, comp_key)
        feature_cols = entry["feature_cols"]

        # Use the most recent row with complete features
        latest = feat_df[feature_cols].dropna().iloc[[-1]]
        model = entry["model"]
        pred = float(model.predict(latest)[0])

        hist_std = entry["hist_std"]
        confidence = _determine_confidence(entry["metrics"]["directional_accuracy"], pred, hist_std)
        recommendation, direction, magnitude = _prediction_to_recommendation(pred, comp_key, hist_std)

        results.append(ModelResult(
            comparison=comp_key,
            horizon=horizon_key,
            prediction=round(pred, 6),
            confidence=confidence,
            recommendation=recommendation,
            direction=direction,
            magnitude=magnitude,
            feature_importance=entry["importance"],
            metrics=entry["metrics"],
        ))

    return results


def get_historical_data(comparison_key: str) -> dict:
    """Return historical performance data for the frontend chart."""
    combined = get_combined_dataset()
    pair = FUND_TICKERS[comparison_key]
    long_ticker = pair["a"]
    short_ticker = pair["b"]

    long_prices = combined[long_ticker].dropna()
    short_prices = combined[short_ticker].dropna()

    # Cumulative returns (base 100)
    long_cum = (long_prices / long_prices.iloc[0]) * 100
    short_cum = (short_prices / short_prices.iloc[0]) * 100

    # Monthly spread return
    long_ret = long_prices.pct_change()
    short_ret = short_prices.pct_change()
    spread = long_ret - short_ret

    dates = [d.strftime("%Y-%m-%d") for d in long_cum.index]

    return {
        "dates": dates,
        "cumulative_long": [round(v, 2) for v in long_cum.values],
        "cumulative_short": [round(v, 2) for v in short_cum.values],
        "spread": [round(v, 6) if not np.isnan(v) else 0 for v in spread.values],
        "long_label": pair["a_label"],
        "short_label": pair["b_label"],
    }
