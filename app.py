"""Investment Comparison App — Streamlit UI for XGBoost fund pair predictions."""

import sys
from pathlib import Path

# Add backend to path so imports work
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import streamlit as st
import pandas as pd
import numpy as np

from config import FUND_TICKERS, COMPARISON_LABELS, HORIZON_LABELS
from data_fetcher import clear_cache
from model import train_all_models, load_models, get_predictions, get_historical_data

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

store = get_model_store()
results = get_predictions(store)

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
