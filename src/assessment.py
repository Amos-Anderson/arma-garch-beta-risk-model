"""
assessment.py — Beta Risk Model
Statistical assessment tools for backtesting the risk model.
VaR exceedances, binomial p-values, KS, Anderson-Darling, Christoffersen.
"""

import numpy as np
import pandas as pd
from scipy import stats


def count_exceedances(actual_returns: np.ndarray,
                      var_series: np.ndarray) -> int:
    """Count how many actual returns fell below the VaR threshold."""
    return int(np.sum(actual_returns < var_series))


def binomial_pvalue(n: int, exceedances: int,
                    confidence_level: float) -> float:
    """
    Two-sided binomial p-value for the number of VaR exceedances.

    Parameters
    ----------
    n                 : total predictions (typically 1000)
    exceedances       : observed number of exceedances
    confidence_level  : VaR confidence level (0.99 or 0.95)

    Returns
    -------
    p-value. < 0.05 → reject null hypothesis that model is correct.
    """
    p = 1.0 - confidence_level
    return float(stats.binomtest(exceedances, n, p,
                                 alternative="two-sided").pvalue)


def pvalue_color(pvalue: float, exceedances: int,
                 expected: float) -> str:
    """
    Color coding per professor's convention:
      green : p >= 0.05  (model passes)
      red   : p < 0.05 AND exceedances > expected  (risk understated)
      blue  : p < 0.05 AND exceedances < expected  (risk overstated)
    """
    if pvalue >= 0.05:
        return "green"
    return "red" if exceedances > expected else "blue"


def christoffersen_test(hit_sequence: np.ndarray) -> dict:
    """
    Christoffersen (1998) independence test for VaR exceedances.

    Tests whether exceedances cluster (serial dependence) or are
    independent — i.e. whether ARMA-GARCH removed volatility clustering.

    Parameters
    ----------
    hit_sequence : binary array, 1 if exceedance on day t else 0

    Returns
    -------
    dict with keys: statistic, pvalue, independent (bool)
    """
    hits = np.asarray(hit_sequence, dtype=int)
    n    = len(hits)

    # Transition counts
    n00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))

    # Guard against division by zero
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return {"statistic": np.nan, "pvalue": np.nan, "independent": None}

    # Transition probabilities
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi   = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Guard against log(0)
    eps = 1e-10
    pi01  = np.clip(pi01, eps, 1 - eps)
    pi11  = np.clip(pi11, eps, 1 - eps)
    pi    = np.clip(pi,   eps, 1 - eps)

    # Log-likelihood ratio statistic
    ll_indep = (
          (n00 + n10) * np.log(1 - pi)
        + (n01 + n11) * np.log(pi)
    )
    ll_dep = (
          n00 * np.log(1 - pi01)
        + n01 * np.log(pi01)
        + n10 * np.log(1 - pi11)
        + n11 * np.log(pi11)
    )

    stat   = -2 * (ll_indep - ll_dep)
    pvalue = float(stats.chi2.sf(stat, df=1))

    return {
        "statistic":   round(float(stat), 4),
        "pvalue":      round(pvalue, 4),
        "independent": pvalue > 0.05,
    }


def anderson_darling(innovations: np.ndarray, cdf_func) -> float:
    """
    Anderson-Darling statistic with extra weight on tail accuracy.
    AD² = n ∫ (F_empirical - F_model)² / (F(1-F)) dF

    Parameters
    ----------
    innovations : standardised residuals (1-D array)
    cdf_func    : callable, takes array x, returns model CDF values

    Returns
    -------
    AD² statistic. Smaller = better tail fit.
    """
    x   = np.sort(innovations)
    n   = len(x)
    cdf = np.clip(cdf_func(x), 1e-10, 1 - 1e-10)
    i   = np.arange(1, n + 1)

    ad = -n - np.mean(
        (2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1]))
    )
    return float(ad)