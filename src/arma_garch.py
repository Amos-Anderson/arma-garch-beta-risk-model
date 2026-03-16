"""
arma_garch.py
ARMA(1,1)-GARCH(1,1) fitting and rolling-window innovation extraction.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from typing import NamedTuple


class GARCHResult(NamedTuple):
    innovations: np.ndarray   # standardised residuals epsilon_t / sigma_t
    sigma_t: np.ndarray       # conditional volatility forecasts
    params: dict              # fitted ARMA-GARCH parameters
    converged: bool           # did the optimizer converge?


def fit_arma_garch(returns: np.ndarray, p: int = 1, q: int = 1) -> GARCHResult:
    """
    Fit ARMA(p,q)-GARCH(1,1) to a return series via MLE.

    Parameters
    ----------
    returns : 1-D array of log returns (a single estimation window)
    p, q    : ARMA orders (default 1,1 per course requirement)

    Returns
    -------
    GARCHResult namedtuple.

    Raises
    ------
    RuntimeError if convergence check fails (alpha+beta >= 1 or omega <= 0).
    """
    raise NotImplementedError


def rolling_window_innovations(
    returns: pd.Series,
    estimation_window: int = 250,
    assessment_window: int = 1000,
) -> pd.DataFrame:
    """
    Slide an estimation window across the return series and collect
    one-step-ahead sigma forecasts and standardised innovations.

    Parameters
    ----------
    returns            : full return series (needs >= estimation + assessment periods)
    estimation_window  : number of periods used to fit ARMA-GARCH (default 250)
    assessment_window  : number of one-step predictions to collect (default 1000)

    Returns
    -------
    DataFrame indexed by prediction date with columns:
        ['return', 'sigma_forecast', 'innovation']
    """
    raise NotImplementedError


def ljung_box_test(innovations: np.ndarray, lags: int = 10) -> dict:
    """
    Run Ljung-Box test on innovations and innovations^2.
    Returns dict with p-values. p > 0.05 means no remaining autocorrelation.
    """
    raise NotImplementedError