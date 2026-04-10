"""Configuration constants for the Investment XGBoost app."""

from datetime import timedelta
from pathlib import Path

FUND_TICKERS = {
    "value_growth": {"a": "VTV", "b": "VUG", "a_label": "Value (VTV)", "b_label": "Growth (VUG)"},
    "us_intl": {"a": "VTI", "b": "VXUS", "a_label": "US (VTI)", "b_label": "International (VXUS)"},
}

ALL_TICKERS = ["VTV", "VUG", "VTI", "VXUS"]

# Auxiliary market tickers (VIX, US Dollar index, Gold)
AUX_TICKERS = {
    "^VIX": "vix",
    "DX-Y.NYB": "usd_index",
    "GLD": "gld",
}

COMPARISON_LABELS = {
    "value_growth": "Value vs Growth",
    "us_intl": "US vs International",
}

# FRED economic data series
FRED_SERIES = {
    "treasury_10y": "GS10",
    "yield_spread_2_10": "T10Y2Y",
    "yield_spread_3m_10y": "T10Y3M",       # 3m-to-10Y spread (more sensitive inversion signal)
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "gdp": "GDP",
    "fed_funds": "FEDFUNDS",
    "baa_yield": "BAA",
    "aaa_yield": "AAA",
    "consumer_sentiment": "UMCSENT",        # Univ. of Michigan Consumer Sentiment
    "indpro": "INDPRO",                     # Industrial Production index
    "retail_sales": "RSAFS",               # Advance Retail Sales (seasonally adjusted)
}

# Prediction horizons in trading days
HORIZONS = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "12m": 252,
}

HORIZON_LABELS = {
    "1m": "1 Month",
    "3m": "3 Months",
    "6m": "6 Months",
    "12m": "12 Months",
}

# Data start date (~20 years of history)
DATA_START_DATE = "2004-01-01"

# Snapshot data directory (bundled historical data for fast startup)
SNAPSHOT_DIR = Path(__file__).parent / "data"

# Cache TTL — prices change daily, macro data updates weekly/monthly
CACHE_MAX_AGE_PRICES = timedelta(days=1)
CACHE_MAX_AGE_MACRO = timedelta(days=7)

# Horizons loaded on startup vs on demand
DEFAULT_HORIZONS = ["3m", "6m"]
EXTRA_HORIZONS = ["1m", "12m"]

# XGBoost default hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 10,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 30,
    "eval_metric": "rmse",
}

# Walk-forward validation settings
WF_N_SPLITS = 5
WF_MIN_TRAIN_DAYS = 2520  # ~10 years

# Magnitude thresholds (annualized spread)
MAGNITUDE_THRESHOLDS = [
    (0.01, "negligible", "Equal weight / no tilt"),
    (0.03, "modest", "Slight overweight (~55/45)"),
    (0.06, "moderate", "Meaningful overweight (~65/35)"),
    (float("inf"), "strong", "Significant overweight (~75/25)"),
]
