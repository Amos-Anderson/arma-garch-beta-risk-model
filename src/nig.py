"""
nig.py — Beta Risk Model
Normal Inverse Gaussian (NIG) distribution:
fitting, PDF, CDF, VaR, CVaR.
"""

import numpy as np
import pandas as pd
from scipy.special import kv
from scipy import stats, integrate, optimize
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")


@dataclass
class NIGParams:
    alpha: float  # tail heaviness — larger = lighter tails (> 0)
    beta:  float  # asymmetry — negative = left skew (-alpha < beta < alpha)
    mu:    float  # location
    delta: float  # scale (> 0)

    def __repr__(self):
        return (f"NIGParams(alpha={self.alpha:.4f}, beta={self.beta:.4f}, "
                f"mu={self.mu:.4f}, delta={self.delta:.4f})")


def nig_pdf(x: np.ndarray, p: NIGParams) -> np.ndarray:
    """
    NIG probability density function computed in log-space
    to avoid overflow from the exponential factor.
    """
    x   = np.asarray(x, dtype=float)
    gam = np.sqrt(p.alpha**2 - p.beta**2)
    q   = np.sqrt(p.delta**2 + (x - p.mu)**2)

    # Log-space computation
    log_pdf = (
        np.log(p.alpha * p.delta / np.pi)
        + p.delta * gam
        + p.beta * (x - p.mu)
        - np.log(q)
        + np.log(kv(1, p.alpha * q))
    )
    return np.exp(log_pdf)


def nig_cdf(x: np.ndarray, p: NIGParams,
            lower: float = -20.0, n_points: int = 2000) -> np.ndarray:
    """
    Numerical CDF by integrating nig_pdf using the trapezoid rule.
    """
    x = np.asarray(x, dtype=float)
    results = np.zeros_like(x)

    for i, xi in enumerate(x):
        grid = np.linspace(lower, xi, n_points)
        pdf_vals = nig_pdf(grid, p)
        results[i] = np.trapezoid(pdf_vals, grid)

    return np.clip(results, 0.0, 1.0)


def _nig_log_likelihood(params: np.ndarray,
                        innovations: np.ndarray) -> float:
    """
    Negative log-likelihood for NIG — minimised by the optimizer.
    Returns large penalty value on invalid parameter combinations.
    """
    alpha, beta, mu, delta = params

    # Hard constraints
    if delta <= 0 or alpha <= 0 or abs(beta) >= alpha:
        return 1e10

    pdf_vals = nig_pdf(innovations, NIGParams(alpha, beta, mu, delta))

    # Guard against zeros or negatives before log
    pdf_vals = np.where(pdf_vals > 1e-300, pdf_vals, 1e-300)
    return -np.sum(np.log(pdf_vals))


def fit_nig_mle(innovations: np.ndarray) -> NIGParams:
    """
    Fit NIG parameters to standardised innovations via MLE.
    Uses method-of-moments for starting values then refines with
    Nelder-Mead (robust to non-smooth likelihood surfaces).

    Returns (NIGParams, converged: bool).
    If MLE fails all attempts, returns method-of-moments estimate
    with converged=False instead of raising — this prevents the
    rolling loop from crashing on difficult windows.
    """
    innovations = np.asarray(innovations, dtype=float)
    innovations = innovations[np.isfinite(innovations)]

    # Method-of-moments starting values
    m1   = np.mean(innovations)
    m2   = np.var(innovations)
    skew = stats.skew(innovations)
    kurt = stats.kurtosis(innovations)  # excess kurtosis

    # Guard against degenerate moments
    kurt  = max(kurt, 0.1)
    skew2 = skew**2

    # MoM estimates (from NIG moment formulas)
    try:
        beta_init  = np.sign(skew) * min(abs(skew) / (2 * np.sqrt(kurt / 3 + 1e-6)), 0.9)
        alpha_init = max(np.sqrt(3 * kurt - 4 * skew2) / m2, 1.0)
        delta_init = max(m2 / np.sqrt(alpha_init**2 - beta_init**2 + 1e-6), 0.1)
        mu_init    = m1 - beta_init * delta_init / np.sqrt(alpha_init**2 - beta_init**2 + 1e-6)
    except Exception:
        alpha_init, beta_init, mu_init, delta_init = 2.0, 0.0, 0.0, 1.0

    mom_fallback = NIGParams(alpha=alpha_init, beta=beta_init,
                             mu=mu_init, delta=delta_init)

    x0 = np.array([alpha_init, beta_init, mu_init, delta_init])

    # Multiple starting points for robustness
    starting_points = [
        x0,
        np.array([2.0,  0.0, 0.0, 1.0]),
        np.array([5.0,  0.0, 0.0, 0.5]),
        np.array([1.5, -0.1, 0.0, 1.0]),
    ]

    best_result = None
    best_val    = np.inf

    for x_start in starting_points:
        try:
            res = optimize.minimize(
                _nig_log_likelihood,
                x_start,
                args=(innovations,),
                method="Nelder-Mead",
                options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6},
            )
            if res.success and res.fun < best_val:
                alpha, beta, mu, delta = res.x
                if delta > 0 and alpha > 0 and abs(beta) < alpha:
                    best_val    = res.fun
                    best_result = res
        except Exception:
            continue

    if best_result is None:
        # Return MoM fallback — don't crash the rolling loop
        return mom_fallback, False

    alpha, beta, mu, delta = best_result.x
    return NIGParams(alpha=alpha, beta=beta, mu=mu, delta=delta), True


def nig_quantile(p_level: float, params: NIGParams,
                 lower: float = -15.0, upper: float = 15.0,
                 tol: float = 1e-6) -> float:
    """
    Compute the quantile of NIG at probability level p_level
    via bisection on the CDF. Used for VaR computation.
    """
    def objective(x):
        return nig_cdf(np.array([x]), params)[0] - p_level

    # Check bounds contain the root
    f_low = objective(lower)
    f_up  = objective(upper)

    if f_low > 0:
        return lower
    if f_up < 0:
        return upper

    result = optimize.brentq(objective, lower, upper, xtol=tol)
    return result


def compute_var(mu_forecast: float, sigma_forecast: float,
                params: NIGParams, level: float = 0.99) -> float:
    """
    Next-day Value at Risk at confidence level `level`.

    Following Kaufman (AMS 603, slide 33) affine transform:
        r_{t+1} ~ NIG(params, mu_{t+1} + mu_NIG, sigma_{t+1} * delta_NIG)

    VaR_α = μ_forecast + σ_forecast × Q_NIG(1 - α)

    where Q_NIG already accounts for the NIG location parameter μ_NIG.

    Returns a negative number representing the loss threshold.
    E.g. VaR99 = -0.025 means there is a 1% chance of losing
    more than 2.5% tomorrow.
    """
    quantile = nig_quantile(1.0 - level, params)
    return float(mu_forecast + sigma_forecast * quantile)


def compute_cvar(mu_forecast: float, sigma_forecast: float,
                 params: NIGParams,
                 level: float = 0.99, n_points: int = 500) -> float:
    """
    Conditional VaR (Expected Shortfall) at confidence level `level`.

    CVaR = E[R | R < VaR_α]  where R = μ + σ·Z,  Z ~ NIG(params)

    Computed by numerical integration over the affine-transformed
    return distribution. Required by Basel III.
    """
    var = compute_var(mu_forecast, sigma_forecast, params, level)

    # Integration bounds in return space
    # Lower bound: map the NIG deep left tail to return space
    lower = mu_forecast + sigma_forecast * nig_quantile(1e-6, params)
    grid  = np.linspace(lower, var, n_points)

    # PDF of R = μ + σ·Z is  f_Z((r - μ)/σ) / σ
    z_vals     = (grid - mu_forecast) / sigma_forecast
    pdf_vals   = nig_pdf(z_vals, params) / sigma_forecast
    numerator  = np.trapezoid(grid * pdf_vals, grid)
    denominator = np.trapezoid(pdf_vals, grid)

    if abs(denominator) < 1e-12:
        return var  # fallback

    return float(numerator / denominator)