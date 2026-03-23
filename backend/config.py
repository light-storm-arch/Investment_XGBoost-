"""Configuration constants for the Investment XGBoost app."""

FUND_TICKERS = {
    "value_growth": {"a": "VVIAX", "b": "VIGAX", "a_label": "Value (VVIAX)", "b_label": "Growth (VIGAX)"},
    "us_intl": {"a": "VTSAX", "b": "VTIAX", "a_label": "US (VTSAX)", "b_label": "International (VTIAX)"},
}

ALL_TICKERS = ["VVIAX", "VIGAX", "VTSAX", "VTIAX"]

COMPARISON_LABELS = {
    "value_growth": "Value vs Growth",
    "us_intl": "US vs International",
}

# FRED economic data series
FRED_SERIES = {
    "treasury_10y": "GS10",
    "yield_spread_2_10": "T10Y2Y",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "gdp": "GDP",
    "fed_funds": "FEDFUNDS",
    "baa_yield": "BAA",
    "aaa_yield": "AAA",
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
