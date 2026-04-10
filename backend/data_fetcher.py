"""Data fetching module: downloads fund prices from yfinance and macro data from FRED.

Uses snapshot data files (backend/data/) to avoid re-downloading 20 years of history.
Only incremental data from the snapshot cutoff date to today is fetched from APIs.
"""

import json
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

from config import (
    ALL_TICKERS, AUX_TICKERS, FRED_SERIES, DATA_START_DATE,
    SNAPSHOT_DIR, CACHE_MAX_AGE_PRICES, CACHE_MAX_AGE_MACRO,
)

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

_combined_cache: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def _is_fresh(path: Path, max_age: timedelta = CACHE_MAX_AGE_PRICES) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < max_age


def _get_snapshot_cutoff() -> str | None:
    """Read the snapshot cutoff date from snapshot_meta.json."""
    meta_path = SNAPSHOT_DIR / "snapshot_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)["cutoff_date"]
    except Exception:
        return None


def _load_snapshot(filename: str) -> pd.DataFrame | None:
    """Load a snapshot parquet file if it exists."""
    path = SNAPSHOT_DIR / filename
    if path.exists():
        return pd.read_parquet(path)
    return None


def _incremental_start(cutoff: str) -> str:
    """Return a start date a few days before cutoff to ensure overlap for dedup."""
    return (pd.Timestamp(cutoff) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")


def _download_single_ticker(ticker: str, max_retries: int = 4,
                            start: str | None = None) -> pd.Series:
    """Download a single ticker with exponential backoff retries."""
    start = start or DATA_START_DATE
    for attempt in range(max_retries):
        try:
            raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
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


def _merge_snapshot_and_incremental(snapshot: pd.DataFrame, incremental: pd.DataFrame) -> pd.DataFrame:
    """Combine snapshot with incremental data, dedup on index, resample to month-end."""
    combined = pd.concat([snapshot, incremental])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    monthly = combined.resample("ME").last()
    monthly.dropna(how="all", inplace=True)
    return monthly


# ---------------------------------------------------------------------------
# Fund prices
# ---------------------------------------------------------------------------

def fetch_fund_prices() -> pd.DataFrame:
    """Download adjusted close prices for all fund tickers, resampled to month-end."""
    cache = _cache_path("fund_prices")
    if _is_fresh(cache, CACHE_MAX_AGE_PRICES):
        logger.info("Loading fund prices from cache")
        return pd.read_parquet(cache)

    snapshot = _load_snapshot("fund_prices_snapshot.parquet")
    cutoff = _get_snapshot_cutoff()

    if snapshot is not None and cutoff:
        # Incremental fetch: only download data from cutoff to present
        start = _incremental_start(cutoff)
        logger.info(f"Loading fund prices: snapshot + incremental from {start}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_download_single_ticker, ticker, start=start): ticker
                for ticker in ALL_TICKERS
            }
            series = {}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    series[ticker] = future.result()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to fetch price data for {ticker} from Yahoo Finance: {e}"
                    ) from e

        incr = pd.DataFrame(series).resample("ME").last()
        monthly = _merge_snapshot_and_incremental(snapshot, incr)
    else:
        # Fallback: full download (no snapshot available)
        logger.info("Downloading fund prices from yfinance (full history)")
        series = {}
        for ticker in ALL_TICKERS:
            logger.info(f"Downloading {ticker}")
            try:
                series[ticker] = _download_single_ticker(ticker)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to fetch price data for {ticker} from Yahoo Finance: {e}"
                ) from e
            time.sleep(1)

        prices = pd.DataFrame(series)
        monthly = prices.resample("ME").last()
        monthly.dropna(how="all", inplace=True)

    monthly.to_parquet(cache)
    return monthly


# ---------------------------------------------------------------------------
# Auxiliary prices
# ---------------------------------------------------------------------------

def fetch_aux_prices() -> pd.DataFrame:
    """Download auxiliary market data (VIX, USD index, Gold), resampled to month-end."""
    cache = _cache_path("aux_prices")
    if _is_fresh(cache, CACHE_MAX_AGE_PRICES):
        logger.info("Loading auxiliary prices from cache")
        return pd.read_parquet(cache)

    snapshot = _load_snapshot("aux_prices_snapshot.parquet")
    cutoff = _get_snapshot_cutoff()

    if snapshot is not None and cutoff:
        start = _incremental_start(cutoff)
        logger.info(f"Loading aux prices: snapshot + incremental from {start}")

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_download_single_ticker, ticker, start=start): col_name
                for ticker, col_name in AUX_TICKERS.items()
            }
            series = {}
            for future in as_completed(futures):
                col_name = futures[future]
                try:
                    series[col_name] = future.result()
                except Exception as e:
                    logger.warning(f"Could not fetch {col_name}: {e} — skipping")

        if not series:
            return snapshot

        incr = pd.DataFrame(series).resample("ME").last()
        monthly = _merge_snapshot_and_incremental(snapshot, incr)
    else:
        logger.info("Downloading auxiliary market data from yfinance (full history)")
        series = {}
        for ticker, col_name in AUX_TICKERS.items():
            logger.info(f"Downloading {ticker}")
            try:
                series[col_name] = _download_single_ticker(ticker)
            except Exception as e:
                logger.warning(f"Could not fetch {ticker} ({col_name}): {e} — skipping")
            time.sleep(1)

        if not series:
            return pd.DataFrame()

        prices = pd.DataFrame(series)
        monthly = prices.resample("ME").last()
        monthly.dropna(how="all", inplace=True)

    monthly.to_parquet(cache)
    return monthly


# ---------------------------------------------------------------------------
# FRED macro data
# ---------------------------------------------------------------------------

def _compute_fred_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived FRED features (shared by fetch and snapshot generation)."""
    df["cpi_yoy"] = df["cpi"].pct_change(12) * 100
    df["credit_spread"] = df["baa_yield"] - df["aaa_yield"]
    df["gdp_yoy"] = df["gdp"].pct_change(4) * 100
    if "indpro" in df.columns:
        df["indpro_yoy"] = df["indpro"].pct_change(12) * 100
    if "retail_sales" in df.columns:
        df["retail_sales_yoy"] = df["retail_sales"].pct_change(12) * 100
    return df


def fetch_fred_data() -> pd.DataFrame:
    """Download macro indicators from FRED, aligned to month-end frequency."""
    cache = _cache_path("fred_data")
    if _is_fresh(cache, CACHE_MAX_AGE_MACRO):
        logger.info("Loading FRED data from cache")
        return pd.read_parquet(cache)

    api_key = os.getenv("FRED_API_KEY")
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

    snapshot = _load_snapshot("fred_data_snapshot.parquet")
    cutoff = _get_snapshot_cutoff()

    fred = Fred(api_key=api_key)
    obs_start = _incremental_start(cutoff) if (snapshot is not None and cutoff) else DATA_START_DATE

    if snapshot is not None and cutoff:
        logger.info(f"Loading FRED data: snapshot + incremental from {obs_start}")
    else:
        logger.info("Downloading FRED data (full history)")

    series_dict: dict[str, pd.Series] = {}
    for name, series_id in FRED_SERIES.items():
        logger.info(f"Fetching FRED series: {series_id}")
        last_err = None
        for attempt in range(3):
            try:
                s = fred.get_series(series_id, observation_start=obs_start)
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

    incr_df = pd.DataFrame(series_dict)
    incr_df.index = pd.to_datetime(incr_df.index)

    if snapshot is not None and cutoff:
        # Merge snapshot (has derived columns) with incremental raw data
        # We need to recompute derived columns on the full series
        # First, get raw columns from snapshot (exclude derived)
        derived_cols = {"cpi_yoy", "credit_spread", "gdp_yoy", "indpro_yoy", "retail_sales_yoy"}
        raw_snapshot = snapshot.drop(columns=[c for c in derived_cols if c in snapshot.columns])

        # GDP forward-fill for incremental
        if "gdp" in incr_df.columns:
            incr_df["gdp"] = incr_df["gdp"].ffill()
        incr_df = incr_df.resample("ME").last()
        incr_df = incr_df.ffill()

        # Merge raw snapshot + incremental
        combined = pd.concat([raw_snapshot, incr_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined = combined.resample("ME").last()
        combined = combined.ffill()

        df = _compute_fred_derived(combined)
    else:
        df = incr_df
        df["gdp"] = df["gdp"].ffill()
        df = df.resample("ME").last()
        df = df.ffill()
        df = _compute_fred_derived(df)

    df.to_parquet(cache)
    return df


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------

def get_combined_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """Merge fund prices and FRED data on a common month-end index.

    Fetches all three data sources concurrently for speed.
    """
    global _combined_cache
    if _combined_cache is not None and not force_refresh:
        return _combined_cache

    # Fetch all data sources in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_prices = executor.submit(fetch_fund_prices)
        future_macro = executor.submit(fetch_fred_data)
        future_aux = executor.submit(fetch_aux_prices)

        prices = future_prices.result()
        macro = future_macro.result()
        aux = future_aux.result()

    combined = prices.join(macro, how="inner")
    if not aux.empty:
        combined = combined.join(aux, how="left")
    combined.dropna(subset=ALL_TICKERS, inplace=True)

    _combined_cache = combined
    logger.info(f"Combined dataset: {len(combined)} monthly observations, {combined.shape[1]} columns")
    return combined


def clear_cache():
    """Remove cached data files to force re-download (does not touch snapshot files)."""
    global _combined_cache
    _combined_cache = None
    for f in CACHE_DIR.glob("*.parquet"):
        f.unlink()
    logger.info("Cache cleared")
