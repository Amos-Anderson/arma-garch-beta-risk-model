"""
data_utils.py - Beta Risk Model
Download, clean, and compute log returns for a list of tickers.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_RAW  = Path(__file__).parents[1] / "data" / "raw"
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
    Saves raw CSV to data/raw/.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].copy()

    # If only one ticker, yfinance returns a Series - normalise to DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)

    # Save raw
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    prices.to_csv(DATA_RAW / "prices_raw.csv")
    print(f"Downloaded {prices.shape[1]} tickers × {prices.shape[0]} days")
    return prices


def check_corporate_actions(prices: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame:
    """
    Flag single-day price moves larger than `threshold` (default 40%).
    These may indicate unadjusted splits or data errors.

    Returns DataFrame of flagged dates and tickers. Empty = data is clean.
    """
    pct_change = prices.pct_change().abs()
    flags = []
    for col in pct_change.columns:
        bad_dates = pct_change.index[pct_change[col] > threshold]
        for date in bad_dates:
            flags.append({
                "ticker": col,
                "date": date,
                "pct_move": pct_change.loc[date, col]
            })
    result = pd.DataFrame(flags)
    if result.empty:
        print("Corporate action check PASSED - no suspicious jumps detected")
    else:
        print(f"WARNING: {len(result)} suspicious price moves found:")
        print(result.to_string(index=False))
    return result


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns: r_t = log(S_t / S_{t-1}).
    Drops the first NaN row produced by the shift.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"Log returns shape: {log_returns.shape}")
    return log_returns


def summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualised summary statistics per ticker.
    Kurtosis > 3 confirms heavy tails relative to Gaussian.
    """
    stats = pd.DataFrame({
        "mean (daily)"  : returns.mean(),
        "std (daily)"   : returns.std(),
        "mean (annual)" : returns.mean() * 252,
        "vol (annual)"  : returns.std() * np.sqrt(252),
        "skewness"      : returns.skew(),
        "excess kurtosis": returns.kurtosis(),  # pandas returns excess kurtosis (Gaussian = 0)
        "min"           : returns.min(),
        "max"           : returns.max(),
        "obs"           : returns.count(),
    }).T
    return stats


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to data/processed/ as Parquet."""
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    path = DATA_PROC / filename
    df.to_parquet(path)
    print(f"Saved to {path}")


def load_processed(filename: str) -> pd.DataFrame:
    """Load a Parquet file from data/processed/."""
    path = DATA_PROC / filename
    df = pd.read_parquet(path)
    print(f"Loaded from {path}  shape={df.shape}")
    return df