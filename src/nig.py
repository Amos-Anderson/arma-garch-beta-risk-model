"""
nig.py
Normal Inverse Gaussian (NIG) distribution: fitting, PDF, CDF, VaR, CVaR.
"""

import numpy as np
import scipy.stats as stats
from scipy.special import kv, kve   # modified Bessel function of second kind
from dataclasses import dataclass


@dataclass
class NIGParams:
    alpha: float   # tail heaviness (> 0)
    beta: float    # asymmetry (-alpha < beta < alpha)
    mu: float      # location
    delta: float   # scale (> 0)


def nig_pdf(x: np.ndarray, params: NIGParams) -> np.ndarray:
    """
    Compute the NIG probability density function.

    Numerical note: compute in log-space to avoid overflow from the
    exponential factor when alpha*delta is large.
    """
    raise NotImplementedError


def nig_cdf(x: np.ndarray, params: NIGParams, n_points: int = 2000) -> np.ndarray:
    """
    Numerical CDF by integrating nig_pdf via Simpson's rule.
    """
    raise NotImplementedError


def fit_nig_mle(innovations: np.ndarray) -> NIGParams:
    """
    Fit NIG parameters to standardised innovations via maximum log-likelihood.
    Uses NLOPT (COBYLA) for robust multivariate optimisation.

    Starting point: method-of-moments estimates.
    Constraints: alpha > |beta|, delta > 0.

    Returns NIGParams. Raises RuntimeError if optimiser does not converge.
    """
    raise NotImplementedError


def compute_var(sigma_forecast: float, params: NIGParams, level: float = 0.99) -> float:
    """
    Compute next-day Value at Risk at given confidence level.

    VaR_alpha = sigma_forecast * quantile(NIG, 1 - alpha)

    Parameters
    ----------
    sigma_forecast : GARCH one-step-ahead volatility
    params         : fitted NIG parameters on standardised innovations
    level          : confidence level, e.g. 0.99 for VaR99

    Returns
    -------
    VaR as a negative number (a loss threshold).
    """
    raise NotImplementedError


def compute_cvar(sigma_forecast: float, params: NIGParams, level: float = 0.99) -> float:
    """
    Compute CVaR (Expected Shortfall): E[r | r < VaR_alpha].
    Required by Basel III.
    """
    raise NotImplementedError