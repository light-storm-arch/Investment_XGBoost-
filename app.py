"""Investment Comparison App — Streamlit UI for XGBoost fund pair predictions."""

import sys
from pathlib import Path

# Add backend to path so imports work
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import logging
import traceback

import streamlit as st
import pandas as pd
import numpy as np

from config import FUND_TICKERS, COMPARISON_LABELS, HORIZON_LABELS, XGBOOST_PARAMS
from data_fetcher import clear_cache
from model import (
    train_all_models, load_models, get_predictions, get_historical_data,
    get_walkforward_detail, train_custom_split,
)

st.set_page_config(
    page_title="Investment XGBoost Predictions",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Model loading (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models...")
def _load_or_train_models():
    store = load_models()
    if store is None:
        store = train_all_models()
    return store


def get_model_store():
    """Get model store, respecting retrain requests."""
    if st.session_state.get("force_retrain"):
        st.session_state["force_retrain"] = False
        clear_cache()
        _load_or_train_models.clear()
    return _load_or_train_models()


# ---------------------------------------------------------------------------
# Sidebar — Hyperparameter Tuning
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Model Settings")
    st.caption("Adjust XGBoost hyperparameters. Changes apply to the Walk-Forward and Custom Split sections below (not the default predictions).")

    use_custom_params = st.toggle("Use custom hyperparameters", value=False)

    n_estimators = st.slider("n_estimators", 50, 1000, XGBOOST_PARAMS["n_estimators"], step=50)
    st.caption(
        "Early stopping is active (30 rounds). Training halts once validation loss "
        "stops improving — effective tree count may be lower than n_estimators."
    )
    max_depth = st.slider("max_depth", 2, 10, XGBOOST_PARAMS["max_depth"])
    st.caption(
        "Controls tree complexity. Higher = more complex patterns but higher "
        "overfitting risk. Lower = simpler, more generalizable model."
    )
    learning_rate = st.select_slider(
        "learning_rate",
        options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        value=XGBOOST_PARAMS["learning_rate"],
    )
    st.caption(
        "Step size per boosting round. Lower = more rounds needed but often "
        "better generalization. Higher = faster training but may overshoot."
    )
    subsample = st.slider("subsample", 0.5, 1.0, XGBOOST_PARAMS["subsample"], step=0.05)
    st.caption(
        "Fraction of training rows used per tree. Lower adds randomness that "
        "can reduce overfitting. 1.0 uses all rows."
    )
    colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, XGBOOST_PARAMS["colsample_bytree"], step=0.05)
    st.caption(
        "Fraction of features considered per tree. Lower forces the model to "
        "learn from different feature subsets, reducing overfitting."
    )
    reg_alpha = st.select_slider(
        "reg_alpha (L1)",
        options=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        value=XGBOOST_PARAMS["reg_alpha"],
    )
    st.caption(
        "L1 regularization — pushes unimportant feature weights to zero. "
        "Higher = sparser model, acts as feature selection."
    )
    reg_lambda = st.select_slider(
        "reg_lambda (L2)",
        options=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        value=XGBOOST_PARAMS["reg_lambda"],
    )
    st.caption(
        "L2 regularization — penalizes large weights. Higher = smoother "
        "predictions and less overfitting."
    )
    min_child_weight = st.slider("min_child_weight", 1, 50, XGBOOST_PARAMS["min_child_weight"])
    st.caption(
        "Minimum data weight in a leaf node. Higher = more conservative splits, "
        "prevents the model from learning noise in small samples."
    )

    if use_custom_params:
        custom_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
            "objective": "reg:squarederror",
            "random_state": 42,
            "early_stopping_rounds": 30,
            "eval_metric": "rmse",
        }
    else:
        custom_params = None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_btn = st.columns([5, 1])
with col_title:
    st.title("Investment Comparison")
    st.caption("XGBoost-powered fund pair predictions")
with col_btn:
    st.write("")  # spacer
    if st.button("Retrain Models", type="secondary"):
        st.session_state["force_retrain"] = True
        st.rerun()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

try:
    store = get_model_store()
    results = get_predictions(store)
except Exception as e:
    logging.error(traceback.format_exc())
    error_msg = str(e)
    if "FRED_API_KEY" in error_msg:
        st.error(f"FRED API key issue: {error_msg}")
        st.info("Set FRED_API_KEY in a `.env` file or as an environment variable. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
    elif "Yahoo Finance" in error_msg or "yfinance" in error_msg.lower():
        st.error(f"Data download issue: {error_msg}")
        st.info("Yahoo Finance may be temporarily unavailable. Try again in a few minutes.")
    elif "FRED API error" in error_msg:
        st.error(f"FRED data issue: {error_msg}")
        st.info("The FRED API may be temporarily unavailable, or your API key may be invalid.")
    else:
        st.error(f"Failed to load models or generate predictions: {type(e).__name__}: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Prediction cards
# ---------------------------------------------------------------------------

CONFIDENCE_COLORS = {"high": "green", "medium": "orange", "low": "gray"}


def _render_comparison(comp_key: str):
    label = COMPARISON_LABELS[comp_key]
    pair = FUND_TICKERS[comp_key]
    preds = [r for r in results if r.comparison == comp_key]
    preds.sort(key=lambda r: ["1m", "3m", "6m", "12m"].index(r.horizon))

    st.subheader(f"{label} ({pair['a']} vs {pair['b']})")

    if not preds:
        st.warning(f"No predictions available for {label}. Model training may have failed — check logs.")
        return

    cols = st.columns(len(preds))
    for col, p in zip(cols, preds):
        with col:
            pct = p.prediction * 100
            sign = "+" if pct > 0 else ""
            color = "green" if pct > 0 else "red"

            st.markdown(
                f"**{HORIZON_LABELS[p.horizon]}** &nbsp; "
                f":{CONFIDENCE_COLORS[p.confidence]}[{p.confidence}]"
            )
            st.markdown(f"### :{color}[{sign}{pct:.2f}%]")
            st.caption(p.recommendation)

            # Magnitude bar
            bar_pct = {"negligible": 10, "modest": 35, "moderate": 60, "strong": 90}.get(p.magnitude, 50)
            st.progress(bar_pct / 100)

    # Historical chart
    hist = get_historical_data(comp_key)
    chart_df = pd.DataFrame({
        "Date": pd.to_datetime(hist["dates"]),
        hist["long_label"]: hist["cumulative_long"],
        hist["short_label"]: hist["cumulative_short"],
    }).set_index("Date")

    st.line_chart(chart_df, height=350)


for comp_key in FUND_TICKERS:
    _render_comparison(comp_key)
    st.divider()

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

st.subheader("Feature Importance")

fi_col1, fi_col2 = st.columns(2)
with fi_col1:
    fi_comp = st.selectbox(
        "Comparison",
        list(FUND_TICKERS.keys()),
        format_func=lambda k: COMPARISON_LABELS[k],
        key="fi_comp",
    )
with fi_col2:
    fi_horizon = st.selectbox("Horizon", ["1m", "3m", "6m", "12m"], index=2, key="fi_horizon")

fi_key = (fi_comp, fi_horizon)
if fi_key in store:
    importance = store[fi_key]["importance"]
    fi_df = (
        pd.DataFrame({"feature": importance.keys(), "importance": importance.values()})
        .sort_values("importance", ascending=True)
        .tail(15)
    )
    st.bar_chart(fi_df.set_index("feature"), horizontal=True, height=400)
else:
    st.info("No model available for this combination.")

# ---------------------------------------------------------------------------
# Model metrics table
# ---------------------------------------------------------------------------

st.subheader("Model Backtest Metrics")

metrics_rows = []
for (comp, horizon), entry in store.items():
    m = entry["metrics"]
    metrics_rows.append({
        "Comparison": COMPARISON_LABELS.get(comp, comp),
        "Horizon": HORIZON_LABELS.get(horizon, horizon),
        "MAE": m.get("mae"),
        "RMSE": m.get("rmse"),
        "Dir. Accuracy": m.get("directional_accuracy"),
        "IC": m.get("ic"),
    })

if metrics_rows:
    metrics_df = pd.DataFrame(metrics_rows)

    # Format for display
    styled = metrics_df.style.format({
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "Dir. Accuracy": "{:.1%}",
        "IC": "{:.3f}",
    }).map(
        lambda v: "color: green; font-weight: bold" if isinstance(v, (int, float)) and v >= 0.55
        else ("color: red; font-weight: bold" if isinstance(v, (int, float)) and v < 0.50 else ""),
        subset=["Dir. Accuracy"],
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Walk-Forward Visualization
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Walk-Forward Out-of-Sample Results")
st.caption("Predicted vs actual spread returns from walk-forward validation. "
           "Toggle custom hyperparameters in the sidebar to see how they change results.")

wf_col1, wf_col2 = st.columns(2)
with wf_col1:
    wf_comp = st.selectbox(
        "Comparison",
        list(FUND_TICKERS.keys()),
        format_func=lambda k: COMPARISON_LABELS[k],
        key="wf_comp",
    )
with wf_col2:
    wf_horizon = st.selectbox("Horizon", ["1m", "3m", "6m", "12m"], index=2, key="wf_horizon")

if st.button("Run Walk-Forward", type="primary", key="run_wf"):
    with st.spinner("Running walk-forward validation..."):
        st.session_state["wf_detail"] = get_walkforward_detail(wf_comp, wf_horizon, custom_params)
        st.session_state["wf_detail_key"] = (wf_comp, wf_horizon)

wf_detail = st.session_state.get("wf_detail", pd.DataFrame())

if not wf_detail.empty:
    wf_chart = wf_detail.set_index("date")[["predicted", "actual"]]
    st.line_chart(wf_chart, height=350)

    # Show walk-forward metrics
    wf_preds = wf_detail["predicted"].values
    wf_actuals = wf_detail["actual"].values
    wf_da = float(np.mean(np.sign(wf_preds) == np.sign(wf_actuals)))
    wf_mae = float(np.mean(np.abs(wf_preds - wf_actuals)))

    wf_m1, wf_m2, wf_m3 = st.columns(3)
    wf_m1.metric("Samples", len(wf_detail))
    wf_m2.metric("Dir. Accuracy", f"{wf_da:.1%}")
    wf_m3.metric("MAE", f"{wf_mae:.4f}")
else:
    st.info("Click 'Run Walk-Forward' to run out-of-sample validation.")

# ---------------------------------------------------------------------------
# Custom Train/Test Split
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Custom Train/Test Split")
st.caption("Choose a cutoff date: the model trains on data before it and tests on data after. "
           "See out-of-sample predictions, strategy P&L, and metrics.")

cs_col1, cs_col2, cs_col3 = st.columns(3)
with cs_col1:
    cs_comp = st.selectbox(
        "Comparison",
        list(FUND_TICKERS.keys()),
        format_func=lambda k: COMPARISON_LABELS[k],
        key="cs_comp",
    )
with cs_col2:
    cs_horizon = st.selectbox("Horizon", ["1m", "3m", "6m", "12m"], index=2, key="cs_horizon")
with cs_col3:
    cs_cutoff = st.date_input(
        "Training cutoff date",
        value=pd.Timestamp("2020-01-01"),
        min_value=pd.Timestamp("2010-01-01"),
        max_value=pd.Timestamp("2025-06-01"),
        key="cs_cutoff",
    )

if st.button("Run Custom Split", type="primary", key="run_custom_split"):
    with st.spinner("Training model with custom split..."):
        cs_result = train_custom_split(cs_comp, cs_horizon, str(cs_cutoff), custom_params)

    if "error" in cs_result:
        st.error(cs_result["error"])
    else:
        detail = cs_result["detail"]
        metrics = cs_result["metrics"]

        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Train Size", cs_result["train_size"])
        m2.metric("Test Size", cs_result["test_size"])
        m3.metric("Dir. Accuracy", f"{metrics['directional_accuracy']:.1%}")
        m4.metric("MAE", f"{metrics['mae']:.4f}")
        m5.metric("IC", f"{metrics['ic']:.3f}")

        # Predicted vs Actual chart
        st.markdown("**Predicted vs Actual Spread Returns**")
        pred_chart = detail.set_index("date")[["predicted", "actual"]]
        st.line_chart(pred_chart, height=300)

        # Cumulative Strategy P&L
        st.markdown("**Cumulative Returns: Model Signal vs Buy & Hold Spread**")
        pnl_chart = detail.set_index("date")[["cum_strategy", "cum_buyhold"]].rename(
            columns={"cum_strategy": "Model Signal", "cum_buyhold": "Buy & Hold Spread"}
        )
        st.line_chart(pnl_chart, height=300)

        # Strategy summary
        total_strat = float(detail["cum_strategy"].iloc[-1])
        total_bh = float(detail["cum_buyhold"].iloc[-1])
        s1, s2, s3 = st.columns(3)
        s1.metric("Model Signal Total Return", f"{total_strat:.2%}")
        s2.metric("Buy & Hold Spread Total Return", f"{total_bh:.2%}")
        s3.metric("Excess Return", f"{total_strat - total_bh:.2%}",
                  delta=f"{total_strat - total_bh:+.2%}")

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Methodology")

with st.expander("Price-Based Features"):
    st.markdown("""
These features capture the relative momentum, volatility, and drawdown characteristics
of each fund pair. All are computed from monthly adjusted close prices.

| Feature | Description |
|---|---|
| **spread_ret_1m / 3m / 6m / 12m** | Cumulative return spread (long minus short) over 1, 3, 6, and 12 months. Captures trend at multiple time scales. |
| **spread_momentum_3v12** | Short-term vs long-term momentum (3-month spread minus 12-month spread). Positive values suggest recent acceleration in the long fund's favor. |
| **long_ret_1m / short_ret_1m** | Individual fund 1-month returns. Provides the model with absolute performance context beyond the spread. |
| **long_vol_6m / short_vol_6m** | 6-month rolling standard deviation of monthly returns for each fund. Measures recent realized volatility. |
| **relative_vol** | Ratio of long fund volatility to short fund volatility. Values above 1.0 indicate the long fund is more volatile. |
| **long_drawdown / short_drawdown** | Current price relative to the 12-month rolling high, minus 1. Measures how far each fund has fallen from its recent peak. |
""")

with st.expander("Macroeconomic Features"):
    st.markdown("""
Macro features are sourced from the Federal Reserve Economic Data (FRED) API.
Both the level and 3-month change are included so the model can distinguish
between "rates are high" and "rates are rising."

| Feature | FRED Series | Description |
|---|---|---|
| **treasury_10y / _chg_3m** | GS10 | 10-Year Treasury yield level and its 3-month change. Reflects long-term interest rate environment. |
| **yield_spread / _chg_3m** | T10Y2Y | 2-to-10-year yield curve slope. Inversion (negative values) has historically preceded recessions. |
| **cpi_yoy / _chg_3m** | CPIAUCSL | Year-over-year CPI inflation rate and its 3-month change. Captures the inflation regime. |
| **unemployment / _chg_3m** | UNRATE | Unemployment rate and 3-month change. Rising unemployment signals economic weakness. |
| **fed_funds / _chg_3m** | FEDFUNDS | Federal funds rate and 3-month change. Reflects monetary policy stance and direction. |
| **credit_spread / _chg_3m** | BAA minus AAA | Corporate credit spread (BAA yield minus AAA yield). Widens during stress, narrows in risk-on environments. |
| **gdp_yoy** | GDP | Year-over-year real GDP growth rate. Quarterly data forward-filled to monthly. |
""")

with st.expander("Model & Validation"):
    st.markdown("""
**Algorithm** — XGBoost gradient-boosted regression (`reg:squarederror`) predicts the
forward cumulative spread return (long fund minus short fund) over 1, 3, 6, and
12-month horizons. A separate model is trained for each comparison pair and horizon
(8 models total).

**Walk-Forward Validation** — Models are evaluated using an expanding-window walk-forward
approach with 5 splits and a minimum training window of ~10 years (2,520 trading days).
At each split the model trains on all data up to that point and predicts the next block.
This avoids look-ahead bias and simulates real-time usage.

**Early Stopping** — Each fit uses early stopping with a patience of 30 rounds, monitored
on a held-out 20% validation set (most recent portion of the training window). Training
halts once RMSE stops improving, so the effective number of trees is often lower than
`n_estimators`.

**Confidence Levels** — Predictions are assigned high, medium, or low confidence based on
two factors: the model's historical directional accuracy (from walk-forward) and the
z-score of the current prediction relative to the historical standard deviation of the
target spread.

**Magnitude Thresholds** — The absolute prediction value determines the recommended
portfolio tilt:
- < 1%: Negligible — equal weight / no tilt
- 1–3%: Modest — slight overweight (~55/45)
- 3–6%: Moderate — meaningful overweight (~65/35)
- > 6%: Strong — significant overweight (~75/25)
""")

with st.expander("Data Sources"):
    st.markdown("""
| Source | Data | Tickers / Series |
|---|---|---|
| **Yahoo Finance** (via yfinance) | Monthly adjusted close prices | VTV (Value), VUG (Growth), VTI (US Total Market), VXUS (International) |
| **FRED API** | Macroeconomic indicators | GS10, T10Y2Y, CPIAUCSL, UNRATE, GDP, FEDFUNDS, BAA, AAA |

All data is resampled to month-end frequency and history begins from **2004-01-01**
(~20 years). Fund prices and macro data are joined on the common date index. A FRED API
key is required — get a free one at
[fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html).
""")
