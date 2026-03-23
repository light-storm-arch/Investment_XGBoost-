"""Feature engineering: transforms raw data into model-ready feature matrices."""

import numpy as np
import pandas as pd

from config import FUND_TICKERS


def build_features(df: pd.DataFrame, comparison_key: str) -> pd.DataFrame:
    """Build feature matrix and forward-looking targets for a given comparison pair.

    Args:
        df: Combined dataset from data_fetcher (month-end indexed).
        comparison_key: One of 'value_growth' or 'us_intl'.

    Returns:
        DataFrame with feature columns and target columns (fwd_spread_1m, etc.).
    """
    pair = FUND_TICKERS[comparison_key]
    long_ticker = pair["a"]
    short_ticker = pair["b"]

    result = pd.DataFrame(index=df.index)

    # Monthly returns
    long_ret = df[long_ticker].pct_change()
    short_ret = df[short_ticker].pct_change()
    spread_ret = long_ret - short_ret

    # --- Price-based features ---
    result["spread_ret_1m"] = spread_ret
    result["spread_ret_3m"] = spread_ret.rolling(3).sum()
    result["spread_ret_6m"] = spread_ret.rolling(6).sum()
    result["spread_ret_12m"] = spread_ret.rolling(12).sum()
    result["spread_momentum_3v12"] = result["spread_ret_3m"] - result["spread_ret_12m"]

    result["long_ret_1m"] = long_ret
    result["short_ret_1m"] = short_ret

    result["long_vol_6m"] = long_ret.rolling(6).std()
    result["short_vol_6m"] = short_ret.rolling(6).std()
    result["relative_vol"] = result["long_vol_6m"] / result["short_vol_6m"].replace(0, np.nan)

    # Drawdown from 12-month rolling max
    long_price = df[long_ticker]
    short_price = df[short_ticker]
    result["long_drawdown"] = long_price / long_price.rolling(12).max() - 1
    result["short_drawdown"] = short_price / short_price.rolling(12).max() - 1

    # --- Macro features ---
    if "treasury_10y" in df.columns:
        result["treasury_10y"] = df["treasury_10y"]
        result["treasury_10y_chg_3m"] = df["treasury_10y"].diff(3)

    if "yield_spread_2_10" in df.columns:
        result["yield_spread"] = df["yield_spread_2_10"]
        result["yield_spread_chg_3m"] = df["yield_spread_2_10"].diff(3)

    if "cpi_yoy" in df.columns:
        result["cpi_yoy"] = df["cpi_yoy"]
        result["cpi_yoy_chg_3m"] = df["cpi_yoy"].diff(3)

    if "unemployment" in df.columns:
        result["unemployment"] = df["unemployment"]
        result["unemployment_chg_3m"] = df["unemployment"].diff(3)

    if "fed_funds" in df.columns:
        result["fed_funds"] = df["fed_funds"]
        result["fed_funds_chg_3m"] = df["fed_funds"].diff(3)

    if "credit_spread" in df.columns:
        result["credit_spread"] = df["credit_spread"]
        result["credit_spread_chg_3m"] = df["credit_spread"].diff(3)

    if "gdp_yoy" in df.columns:
        result["gdp_yoy"] = df["gdp_yoy"]

    # --- Target variables: forward cumulative spread returns ---
    for horizon_months, label in [(1, "1m"), (3, "3m"), (6, "6m"), (12, "12m")]:
        result[f"fwd_spread_{label}"] = spread_ret.rolling(horizon_months).sum().shift(-horizon_months)

    # Drop rows with NaN from rolling windows
    result.dropna(subset=get_feature_columns(result), inplace=True)

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return only feature columns (exclude target columns)."""
    return [c for c in df.columns if not c.startswith("fwd_spread_")]


def get_target_columns() -> list[str]:
    """Return target column names."""
    return ["fwd_spread_1m", "fwd_spread_3m", "fwd_spread_6m", "fwd_spread_12m"]
