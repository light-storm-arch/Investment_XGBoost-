"""Data fetching module: downloads fund prices from yfinance and macro data from FRED."""

import os
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

from config import ALL_TICKERS, FRED_SERIES, DATA_START_DATE

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_MAX_AGE = timedelta(days=1)

_combined_cache: pd.DataFrame | None = None


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < CACHE_MAX_AGE


def _download_single_ticker(ticker: str, max_retries: int = 4) -> pd.Series:
    """Download a single ticker with exponential backoff retries."""
    for attempt in range(max_retries):
        try:
            raw = yf.download(ticker, start=DATA_START_DATE, auto_adjust=True, progress=False)
            if raw.empty:
                raise ValueError(f"Empty data for {ticker}")
            if isinstance(raw.columns, pd.MultiIndex):
                return raw["Close"][ticker]
            return raw["Close"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {ticker} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to download {ticker} after {max_retries} attempts: {e}")


def fetch_fund_prices() -> pd.DataFrame:
    """Download adjusted close prices for all fund tickers, resampled to month-end."""
    cache = _cache_path("fund_prices")
    if _is_fresh(cache):
        logger.info("Loading fund prices from cache")
        return pd.read_parquet(cache)

    logger.info("Downloading fund prices from yfinance")
    series = {}
    for ticker in ALL_TICKERS:
        logger.info(f"Downloading {ticker}")
        try:
            series[ticker] = _download_single_ticker(ticker)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch price data for {ticker} from Yahoo Finance: {e}"
            ) from e
        time.sleep(1)  # brief pause between tickers to avoid rate limits

    prices = pd.DataFrame(series)

    # Resample to month-end using last valid price
    monthly = prices.resample("ME").last()
    monthly.dropna(how="all", inplace=True)

    monthly.to_parquet(cache)
    return monthly


def fetch_fred_data() -> pd.DataFrame:
    """Download macro indicators from FRED, aligned to month-end frequency."""
    cache = _cache_path("fred_data")
    if _is_fresh(cache):
        logger.info("Loading FRED data from cache")
        return pd.read_parquet(cache)

    api_key = os.getenv("FRED_API_KEY")
    # Also check Streamlit secrets if available
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("FRED_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not set. Set it in .env, environment, or Streamlit secrets. "
                           "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
    if api_key == "your_fred_api_key_here":
        raise RuntimeError("FRED_API_KEY is set to the placeholder value from .env.example. "
                           "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")

    fred = Fred(api_key=api_key)
    series_dict: dict[str, pd.Series] = {}

    for name, series_id in FRED_SERIES.items():
        logger.info(f"Fetching FRED series: {series_id}")
        last_err = None
        for attempt in range(3):
            try:
                s = fred.get_series(series_id, observation_start=DATA_START_DATE)
                series_dict[name] = s
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Retry {attempt + 1}/3 for FRED {series_id} after {wait}s: {e}")
                    time.sleep(wait)
        else:
            raise RuntimeError(
                f"FRED API error fetching '{series_id}' ({name}) after 3 attempts: {last_err}. "
                f"Check that your FRED_API_KEY is valid."
            ) from last_err

    df = pd.DataFrame(series_dict)
    df.index = pd.to_datetime(df.index)

    # GDP is quarterly — forward-fill to monthly
    df["gdp"] = df["gdp"].ffill()

    # Resample everything to month-end
    df = df.resample("ME").last()
    df = df.ffill()

    # Derived features
    df["cpi_yoy"] = df["cpi"].pct_change(12) * 100
    df["credit_spread"] = df["baa_yield"] - df["aaa_yield"]
    df["gdp_yoy"] = df["gdp"].pct_change(4) * 100  # quarterly pct_change before ffill gives YoY

    df.to_parquet(cache)
    return df


def get_combined_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """Merge fund prices and FRED data on a common month-end index."""
    global _combined_cache
    if _combined_cache is not None and not force_refresh:
        return _combined_cache

    prices = fetch_fund_prices()
    macro = fetch_fred_data()

    combined = prices.join(macro, how="inner")
    combined.dropna(subset=ALL_TICKERS, inplace=True)

    _combined_cache = combined
    logger.info(f"Combined dataset: {len(combined)} monthly observations, {combined.shape[1]} columns")
    return combined


def clear_cache():
    """Remove cached data files to force re-download."""
    global _combined_cache
    _combined_cache = None
    for f in CACHE_DIR.glob("*.parquet"):
        f.unlink()
    logger.info("Cache cleared")
