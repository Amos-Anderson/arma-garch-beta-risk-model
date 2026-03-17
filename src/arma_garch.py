"""
arma_garch.py — Beta Risk Model
ARMA(1,1)-GARCH(1,1) fitting and rolling-window innovation extraction.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from typing import NamedTuple
import warnings
warnings.filterwarnings("ignore")


class GARCHResult(NamedTuple):
    innovations: np.ndarray  # standardised residuals: epsilon_t / sigma_t
    sigma_t:     np.ndarray  # conditional volatility (annualised if scaled)
    params:      dict        # fitted parameter dictionary
    converged:   bool        # did the optimiser converge?


def fit_arma_garch(returns: np.ndarray) -> GARCHResult:
    """
    Fit ARMA(1,1)-GARCH(1,1) to a return series via MLE.

    Parameters
    ----------
    returns : 1-D array of log returns (single estimation window, ~250 obs)

    Returns
    -------
    GARCHResult namedtuple with innovations, sigma_t, params, converged flag.

    Raises
    ------
    RuntimeError if GARCH variance parameters violate stationarity:
        alpha + beta must be < 1, omega must be > 0.
    """
    model = arch_model(
        returns * 100,    # scale to % returns for numerical stability
        mean="ARX",
        lags=1,
        vol="GARCH",
        p=1, q=1,
        dist="normal",
    )

    result = model.fit(
        disp="off",
        options={"maxiter": 500},
    )

    # --- Convergence check -------------------------------------------
    params = result.params.to_dict()
    omega = params.get("omega", 0)
    alpha = params.get("alpha[1]", 1)
    beta  = params.get("beta[1]", 1)

    converged = (
        result.optimization_result.success
        and omega > 0
        and alpha >= 0
        and beta  >= 0
        and (alpha + beta) < 1.0
    )

    if not converged:
        raise RuntimeError(
            f"GARCH did not converge. "
            f"omega={omega:.4f}, alpha={alpha:.4f}, beta={beta:.4f}, "
            f"alpha+beta={alpha+beta:.4f}"
        )

    # Standardised innovations: residuals / conditional std
    # Drop NaN padding that arch inserts at the start of the arrays
    resid       = pd.Series(result.resid).dropna().values / 100
    sigma       = pd.Series(result.conditional_volatility).dropna().values / 100

    # Align lengths - arch occasionally produces arrays of slightly different length
    min_len     = min(len(resid), len(sigma))
    resid       = resid[-min_len:]
    sigma       = sigma[-min_len:]
    innovations = resid / sigma

    

    return GARCHResult(
        innovations = innovations,
        sigma_t     = sigma,
        params      = params,
        converged   = converged,
    )


def rolling_window_innovations(
    returns:            pd.Series,
    estimation_window:  int = 250,
    assessment_window:  int = 1000,
) -> pd.DataFrame:
    """
    Slide an estimation window across the return series, collecting
    one-step-ahead sigma forecasts and standardised innovations.

    Parameters
    ----------
    returns           : full return series (needs >= est + assess periods)
    estimation_window : periods used to fit ARMA-GARCH at each step
    assessment_window : number of consecutive one-step predictions

    Returns
    -------
    DataFrame indexed by prediction date with columns:
        return          — actual log return on that date
        sigma_forecast  — GARCH one-step-ahead conditional volatility
        innovation      — standardised residual (return / sigma_forecast)
    """
    n     = len(returns)
    start = n - assessment_window   # index of first prediction date

    records = []
    n_failed = 0

    for i in range(assessment_window):
        est_start = start - estimation_window + i
        est_end   = start + i                       # exclusive
        pred_idx  = start + i                       # prediction date index

        window = returns.iloc[est_start:est_end].values
        actual = returns.iloc[pred_idx]
        date   = returns.index[pred_idx]

        try:
            res = fit_arma_garch(window)
            # One-step-ahead sigma: last value of fitted conditional vol
            sigma_hat = res.sigma_t[-1]
            innovation = actual / sigma_hat

            records.append({
                "date":           date,
                "return":         actual,
                "sigma_forecast": sigma_hat,
                "innovation":     innovation,
            })

        except RuntimeError as e:
            n_failed += 1
            # On convergence failure, carry forward last known sigma
            if records:
                sigma_hat  = records[-1]["sigma_forecast"]
                innovation = actual / sigma_hat
                records.append({
                    "date":           date,
                    "return":         actual,
                    "sigma_forecast": sigma_hat,
                    "innovation":     innovation,
                })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{assessment_window} windows complete "
                  f"({n_failed} convergence failures so far)")

    df = pd.DataFrame(records).set_index("date")
    print(f"\nDone. {len(df)} predictions. "
          f"Convergence failures: {n_failed} "
          f"({100*n_failed/assessment_window:.1f}%)")
    return df


def ljung_box_test(innovations: np.ndarray, lags: int = 10) -> dict:
    """
    Ljung-Box test on innovations and innovations² (squared).
    p > 0.05 on both implies no remaining autocorrelation or ARCH effects.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_innov = acorr_ljungbox(innovations,        lags=[lags], return_df=True)
    lb_sq    = acorr_ljungbox(innovations ** 2,   lags=[lags], return_df=True)

    return {
        "lb_stat_innov":  float(lb_innov["lb_stat"].values[0]),
        "lb_pval_innov":  float(lb_innov["lb_pvalue"].values[0]),
        "lb_stat_sq":     float(lb_sq["lb_stat"].values[0]),
        "lb_pval_sq":     float(lb_sq["lb_pvalue"].values[0]),
    }