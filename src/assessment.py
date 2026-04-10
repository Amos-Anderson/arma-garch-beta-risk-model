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


def anderson_darling_pit(pit_values: np.ndarray) -> float:
    """
    Anderson-Darling statistic on PIT values against Uniform(0,1).

    Since the PIT values u_t = F_t(r_t) should be U[0,1] if the model
    is correctly specified, the theoretical CDF is F(x) = x.
    This simplifies the AD formula.

    AD² = -n - (1/n) Σ (2i-1) [ln(u_i) + ln(1 - u_{n+1-i})]

    Parameters
    ----------
    pit_values : 1-D array of PIT values in [0, 1]

    Returns
    -------
    AD² statistic. Smaller = better fit to Uniform(0,1).
    """
    u = np.sort(np.asarray(pit_values, dtype=float))
    n = len(u)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    i = np.arange(1, n + 1)

    # For Uniform(0,1): CDF(u) = u, so log(CDF) = log(u)
    ad = -n - np.mean(
        (2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))
    )
    return float(ad)


def pit_qq(pit_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    QQ plot data for PIT values against Uniform(0,1).

    Following Kaufman (AMS 603, slide 37):
        model_quantiles = sorted(F_t(r_t))
        empirical_quantiles = (i+1) / (n+1)  for i = 0..n-1

    If model is correct, points lie on the 45-degree line.

    Parameters
    ----------
    pit_values : 1-D array of PIT values u_t = F_t(r_t)

    Returns
    -------
    (empirical_quantiles, model_quantiles) — both 1-D arrays, sorted.
    Plot empirical on x-axis, model on y-axis.
    """
    model_quantiles = np.sort(np.asarray(pit_values, dtype=float))
    n = len(model_quantiles)
    empirical_quantiles = np.array(
        [(i + 1) / (n + 1) for i in range(n)]
    )
    return empirical_quantiles, model_quantiles


def pit_ks_test(pit_values: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov test on PIT values against Uniform(0,1).

    Following Kaufman (AMS 603, slide 42):
        ks_stat, ks_p = scipy.stats.kstest(pit_values, 'uniform')

    Parameters
    ----------
    pit_values : 1-D array of PIT values u_t = F_t(r_t)

    Returns
    -------
    dict with keys: statistic, pvalue, pass (bool, p > 0.05)
    """
    pit = np.asarray(pit_values, dtype=float)
    ks_stat, ks_p = stats.kstest(pit, 'uniform')
    return {
        "statistic": round(float(ks_stat), 4),
        "pvalue":    round(float(ks_p), 4),
        "pass":      ks_p > 0.05,
    }


# --- Legacy wrapper (backward compatibility) ---
def anderson_darling(innovations: np.ndarray, cdf_func) -> float:
    """
    Anderson-Darling statistic (legacy interface).
    For the corrected per-window model, use anderson_darling_pit() instead.
    """
    x   = np.sort(innovations)
    n   = len(x)
    cdf = np.clip(cdf_func(x), 1e-10, 1 - 1e-10)
    i   = np.arange(1, n + 1)

    ad = -n - np.mean(
        (2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1]))
    )
    return float(ad)