"""Macro regime detection using the latest FRED data.

Classifies the current environment across four dimensions:
  - Rate environment (Fed Funds 3-month change)
  - Yield curve (2-to-10 spread level)
  - Credit conditions (BAA-AAA spread)
  - Inflation regime (CPI YoY)

Each dimension includes a relevance note for Value/Growth and US/International tilts.
"""

import pandas as pd


# ── Thresholds ────────────────────────────────────────────────────────────────

_RATE_RISING_THRESH = 0.25      # 25bps 3-month Fed Funds change
_RATE_FALLING_THRESH = -0.25
_YC_INVERTED_THRESH = 0.0
_YC_FLAT_THRESH = 0.5
_CREDIT_STRESSED_THRESH = 1.5   # BAA-AAA spread in percentage points
_CPI_HIGH_THRESH = 4.0
_CPI_MODERATE_THRESH = 2.0

# ── Regime notes ──────────────────────────────────────────────────────────────

_RATE_NOTES = {
    "Rising": {
        "value_growth": "Historically favors Value — higher rates compress long-duration growth valuations",
        "us_intl": "Tends to strengthen the USD, creating a headwind for unhedged International returns",
    },
    "Falling": {
        "value_growth": "Historically favors Growth — lower discount rates lift long-duration cash flows",
        "us_intl": "USD typically weakens, which provides a tailwind for International returns",
    },
    "Neutral": {
        "value_growth": "Stable rates are broadly neutral; other factors likely dominate",
        "us_intl": "Stable rates reduce a major USD catalyst; watch relative growth differentials",
    },
}

_YC_NOTES = {
    "Inverted": {
        "value_growth": "Inversion signals recession risk — defensive/Value tilt gains historical support",
        "us_intl": "Recession risk tends to benefit the US (safe-haven demand) over International",
    },
    "Flat": {
        "value_growth": "Flat curve signals late-cycle — mixed signal for Value vs Growth",
        "us_intl": "Late-cycle conditions slightly favor the US over emerging-market-heavy International",
    },
    "Normal": {
        "value_growth": "Normal curve signals expansion — cyclicals and Growth can both perform",
        "us_intl": "Expansion environments can favor International markets with higher beta",
    },
}

_CREDIT_NOTES = {
    "Stressed": {
        "value_growth": "Wide credit spreads signal risk-off — defensive Value sectors historically hold up better",
        "us_intl": "Risk-off environments typically favor the US over International (especially EM)",
    },
    "Benign": {
        "value_growth": "Tight spreads signal risk appetite — Growth can outperform in risk-on regimes",
        "us_intl": "Benign credit conditions can support International, particularly EM allocations",
    },
}

_CPI_NOTES = {
    "High": {
        "value_growth": "High inflation historically favors Value — energy, materials, and financials benefit",
        "us_intl": "High US inflation complicates the Fed's path; watch USD and commodity exporters",
    },
    "Moderate": {
        "value_growth": "Moderate inflation is the goldilocks zone — broadly supportive for equities overall",
        "us_intl": "Moderate inflation is neutral across the US/International split",
    },
    "Low": {
        "value_growth": "Low inflation historically favors Growth — secular growers command premium multiples",
        "us_intl": "Low inflation may allow easier monetary policy globally, benefiting International",
    },
}


def get_current_regime(combined: pd.DataFrame) -> dict:
    """Classify the current macro regime from the latest row of the combined dataset.

    Args:
        combined: Combined dataset from data_fetcher.get_combined_dataset().
                  Must contain FRED columns (fed_funds, yield_spread_2_10,
                  credit_spread, cpi_yoy).

    Returns:
        Dict with keys: 'rates', 'yield_curve', 'credit', 'inflation'.
        Each value is a dict with 'label', 'detail', 'color',
        'value_growth_note', and 'us_intl_note'.
    """
    if combined.empty:
        return {}

    latest = combined.iloc[-1]
    result: dict = {}

    # ── Rate environment ──────────────────────────────────────────────────────
    if "fed_funds" in combined.columns and len(combined) >= 4:
        fed_now = float(latest["fed_funds"])
        fed_3m_ago = float(combined["fed_funds"].iloc[-4])
        fed_chg = fed_now - fed_3m_ago

        if fed_chg > _RATE_RISING_THRESH:
            label = "Rising"
            color = "orange"
            detail = f"Fed Funds: {fed_now:.2f}% (+{fed_chg:.2f}% over 3m)"
        elif fed_chg < _RATE_FALLING_THRESH:
            label = "Falling"
            color = "blue"
            detail = f"Fed Funds: {fed_now:.2f}% ({fed_chg:.2f}% over 3m)"
        else:
            label = "Neutral"
            color = "gray"
            detail = f"Fed Funds: {fed_now:.2f}% ({fed_chg:+.2f}% over 3m)"

        result["rates"] = {
            "label": label,
            "detail": detail,
            "color": color,
            **_RATE_NOTES[label],
        }

    # ── Yield curve ───────────────────────────────────────────────────────────
    if "yield_spread_2_10" in combined.columns:
        yc = float(latest["yield_spread_2_10"])

        if yc < _YC_INVERTED_THRESH:
            label = "Inverted"
            color = "red"
        elif yc < _YC_FLAT_THRESH:
            label = "Flat"
            color = "orange"
        else:
            label = "Normal"
            color = "green"

        result["yield_curve"] = {
            "label": label,
            "detail": f"2s10s spread: {yc:+.2f}%",
            "color": color,
            **_YC_NOTES[label],
        }

    # ── Credit conditions ─────────────────────────────────────────────────────
    if "credit_spread" in combined.columns:
        cs = float(latest["credit_spread"])

        if cs > _CREDIT_STRESSED_THRESH:
            label = "Stressed"
            color = "red"
        else:
            label = "Benign"
            color = "green"

        result["credit"] = {
            "label": label,
            "detail": f"BAA-AAA spread: {cs:.2f}%",
            "color": color,
            **_CREDIT_NOTES[label],
        }

    # ── Inflation ─────────────────────────────────────────────────────────────
    if "cpi_yoy" in combined.columns:
        cpi = float(latest["cpi_yoy"])

        if cpi > _CPI_HIGH_THRESH:
            label = "High"
            color = "red"
        elif cpi > _CPI_MODERATE_THRESH:
            label = "Moderate"
            color = "green"
        else:
            label = "Low"
            color = "blue"

        result["inflation"] = {
            "label": label,
            "detail": f"CPI YoY: {cpi:.1f}%",
            "color": color,
            **_CPI_NOTES[label],
        }

    return result
