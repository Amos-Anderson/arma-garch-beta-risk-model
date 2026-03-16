"""
data_utils.py -- Beta Risk Model
Download, clean, and compute log returns for a list of tickers.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_RAW = Path(__file__).parents[1] / "data" / "raw"
DATA_PROC = Path(__file__).parents[1] / "data" / "processed"


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list of ticker strings, e.g. ['^GSPC', 'AAPL']
    start   : 'YYYY-MM-DD'
    end     : 'YYYY-MM-DD'

    Returns
    -------
    DataFrame with DatetimeIndex, one column per ticker (adjusted close).
    """
    raise NotImplementedError


def check_corporate_actions(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Flag suspicious single-day price jumps (>40%) that may indicate
    unadjusted splits or data errors.

    Returns DataFrame of flagged dates and tickers.
    """
    raise NotImplementedError


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns: r_t = log(S_t / S_{t-1}).
    Drops the first NaN row.
    """
    raise NotImplementedError


def save_processed(returns: pd.DataFrame, filename: str) -> None:
    """Save returns DataFrame to data/processed/ as Parquet."""
    raise NotImplementedError


def load_processed(filename: str) -> pd.DataFrame:
    """Load a Parquet file from data/processed/."""
    raise NotImplementedError