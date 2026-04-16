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


def _nig_neg_loglik_nlopt(params, grad, innovations):
    """
    NIG negative log-likelihood in nlopt signature: f(params, grad).
    nlopt requires grad argument even for derivative-free methods.
    """
    alpha, beta, mu, delta = params

    # nlopt enforces bounds, but guard edge cases numerically
    if delta <= 1e-10 or alpha <= 1e-10 or abs(beta) >= alpha - 1e-10:
        return 1e10

    pdf_vals = nig_pdf(innovations, NIGParams(alpha, beta, mu, delta))
    pdf_vals = np.where(pdf_vals > 1e-300, pdf_vals, 1e-300)
    return float(-np.sum(np.log(pdf_vals)))


def fit_nig_mle(innovations: np.ndarray):
    """
    Fit NIG parameters to standardised innovations via MLE.

    Following Kaufman (AMS 603, slide 32): uses nlopt with bounded
    derivative-free optimisation (LN_BOBYQA, with LN_COBYLA fallback).
    Constraints α > 0, δ > 0, |β| < α are enforced as explicit bounds.

    Uses method-of-moments for starting values.

    Returns (NIGParams, converged: bool).
    If MLE fails, returns MoM estimate with converged=False.
    """
    import nlopt

    innovations = np.asarray(innovations, dtype=float)
    innovations = innovations[np.isfinite(innovations)]

    # --- Method-of-moments starting values ---
    m1   = np.mean(innovations)
    m2   = np.var(innovations)
    skew = stats.skew(innovations)
    kurt = stats.kurtosis(innovations)  # excess kurtosis

    # Guard against degenerate moments
    kurt  = max(kurt, 0.1)
    skew2 = skew**2

    try:
        beta_init  = np.sign(skew) * min(abs(skew) / (2 * np.sqrt(kurt / 3 + 1e-6)), 0.9)
        alpha_init = max(np.sqrt(max(3 * kurt - 4 * skew2, 0.01)) / max(m2, 1e-6), 0.5)
        delta_init = max(m2 / np.sqrt(alpha_init**2 - beta_init**2 + 1e-6), 0.1)
        mu_init    = m1 - beta_init * delta_init / np.sqrt(alpha_init**2 - beta_init**2 + 1e-6)
    except Exception:
        alpha_init, beta_init, mu_init, delta_init = 2.0, 0.0, 0.0, 1.0

    mom_fallback = NIGParams(alpha=alpha_init, beta=beta_init,
                             mu=mu_init, delta=delta_init)

    # --- nlopt bounds (tightened for standardised innovations) ---
    # The innovations have mean ≈ 0 and variance ≈ 1 by construction.
    # NIG variance = α²·δ / γ³ where γ = √(α² − β²).
    # For var ≈ 1, δ and α cannot both be large — they are identifiability-coupled.
    # Restrict to economically plausible ranges to prevent degenerate solutions.
    #
    # params = [alpha, beta, mu, delta]
    #   alpha ∈ [0.2, 20]   — controls tail heaviness; 20 is essentially Gaussian
    #   beta  ∈ [-19.9, 19.9] — |β| < α is enforced by the likelihood penalty
    #   mu    ∈ [-2, 2]     — location for standardised data is small
    #   delta ∈ [0.1, 10]   — scale for standardised data is O(1)
    num_variables = 4
    lower_bounds = [0.2,  -19.9, -2.0, 0.1]
    upper_bounds = [20.0,  19.9,  2.0, 10.0]

    def clip(x, lo, hi):
        return max(min(x, hi), lo)

    initial_values = [
        clip(alpha_init, 0.5, 19.0),
        clip(beta_init, -18.0, 18.0),
        clip(mu_init, -1.5, 1.5),
        clip(delta_init, 0.2, 9.0),
    ]

    # Objective: captures innovations via closure
    def objective(params, grad):
        return _nig_neg_loglik_nlopt(params, grad, innovations)

    # --- Try LN_BOBYQA first, then LN_COBYLA as fallback ---
    best_params = None
    best_val    = np.inf

    for algo in [nlopt.LN_BOBYQA, nlopt.LN_COBYLA]:
        try:
            opt = nlopt.opt(algo, num_variables)
            opt.set_lower_bounds(lower_bounds)
            opt.set_upper_bounds(upper_bounds)
            opt.set_min_objective(objective)
            opt.set_xtol_abs(1e-6)
            opt.set_ftol_abs(1e-6)
            opt.set_maxeval(10000)

            x = opt.optimize(initial_values)
            val = opt.last_optimum_value()

            alpha, beta, mu, delta = x
            if val < best_val and delta > 0 and alpha > 0 and abs(beta) < alpha:
                best_val = val
                best_params = x
                break  # LN_BOBYQA succeeded, no need for fallback

        except Exception:
            continue

    # --- Also try from a default starting point ---
    if best_params is None or best_val > 1e9:
        default_start = [2.0, 0.0, 0.0, 1.0]
        for algo in [nlopt.LN_BOBYQA, nlopt.LN_COBYLA]:
            try:
                opt = nlopt.opt(algo, num_variables)
                opt.set_lower_bounds(lower_bounds)
                opt.set_upper_bounds(upper_bounds)
                opt.set_min_objective(objective)
                opt.set_xtol_abs(1e-6)
                opt.set_ftol_abs(1e-6)
                opt.set_maxeval(10000)

                x = opt.optimize(default_start)
                val = opt.last_optimum_value()

                alpha, beta, mu, delta = x
                if val < best_val and delta > 0 and alpha > 0 and abs(beta) < alpha:
                    best_val = val
                    best_params = x
                    break

            except Exception:
                continue

    if best_params is None:
        return mom_fallback, False

    alpha, beta, mu, delta = best_params
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