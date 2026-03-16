"""
assessment.py
Statistical assessment tools for backtesting the risk model.
"""

import numpy as np
import pandas as pd
from scipy import stats


def count_exceedances(actual_returns: np.ndarray, var_series: np.ndarray) -> int:
    """Count how many actual returns fell below the VaR threshold."""
    return int(np.sum(actual_returns < var_series))


def binomial_pvalue(n: int, exceedances: int, confidence_level: float) -> float:
    """
    Two-sided binomial p-value for the number of VaR exceedances.

    Parameters
    ----------
    n                 : total number of predictions (typically 1000)
    exceedances       : observed number of exceedances
    confidence_level  : VaR confidence level (0.99 or 0.95)

    Returns
    -------
    p-value. < 0.05 → reject null hypothesis that model is correctly calibrated.
    """
    p = 1.0 - confidence_level
    return float(stats.binom_test(exceedances, n, p, alternative='two-sided'))


def ks_statistic(innovations: np.ndarray, cdf_func) -> dict:
    """
    Kolmogorov-Smirnov test: max|F_empirical(x) - F_model(x)|.

    Parameters
    ----------
    innovations : standardised residuals
    cdf_func    : callable, takes array x, returns model CDF values

    Returns
    -------
    dict with keys 'statistic' and 'pvalue'.
    """
    raise NotImplementedError


def anderson_darling(innovations: np.ndarray, cdf_func) -> float:
    """
    Anderson-Darling statistic with emphasis on tail accuracy.
    AD^2 = n * integral[(F_empirical - F_model)^2 / (F*(1-F)) dF]

    Returns the AD^2 statistic. Smaller = better fit.
    """
    raise NotImplementedError


def christoffersen_test(hit_sequence: np.ndarray) -> dict:
    """
    Christoffersen (1998) test for independence of VaR exceedances.

    Tests the null hypothesis that exceedances are i.i.d. Bernoulli —
    i.e. that ARMA-GARCH successfully removed clustering.

    Parameters
    ----------
    hit_sequence : binary array, 1 if exceedance on that day else 0

    Returns
    -------
    dict with keys 'statistic', 'pvalue', 'independent' (bool).
    """
    raise NotImplementedError


def pvalue_color(pvalue: float, exceedances: int, expected: float) -> str:
    """
    Return 'green', 'red', or 'blue' per professor's color-coding convention.

    green : p >= 0.05 (model passes)
    red   : p < 0.05 AND exceedances > expected (risk understated — failure)
    blue  : p < 0.05 AND exceedances < expected (risk overstated — usable)
    """
    if pvalue >= 0.05:
        return 'green'
    return 'red' if exceedances > expected else 'blue'