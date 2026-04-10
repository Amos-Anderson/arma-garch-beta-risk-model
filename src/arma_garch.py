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
    innovations:    np.ndarray  # standardised residuals ν_t = ε_t / σ_t (in-sample)
    sigma_t:        np.ndarray  # conditional volatility series (in-sample)
    mu_forecast:    float       # one-step-ahead ARMA conditional mean  μ_{t+1}
    sigma_forecast: float       # one-step-ahead GARCH conditional vol  σ_{t+1}
    params:         dict        # fitted parameter dictionary
    converged:      bool        # did the optimiser converge?


def fit_arma_garch(returns: np.ndarray) -> GARCHResult:
    """
    Fit ARMA(1,1)-GARCH(1,1) to a return series via Gaussian MLE.

    Following Kaufman (AMS 603, slide 18): fit ARMA-GARCH with Gaussian
    log-likelihood on innovations, then fit the heavy-tailed distribution
    (NIG) separately on the standardised innovations.

    Parameters
    ----------
    returns : 1-D array of log returns (single estimation window, ~250 obs)

    Returns
    -------
    GARCHResult namedtuple with:
        innovations    — in-sample standardised residuals ν_t = ε_t / σ_t
        sigma_t        — in-sample conditional volatility series
        mu_forecast    — one-step-ahead conditional mean μ_{t+1}
        sigma_forecast — one-step-ahead conditional volatility σ_{t+1}
        params         — fitted ARMA-GARCH parameter dictionary
        converged      — bool

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

    # --- One-step-ahead forecasts (Kaufman slide 33) -------------------
    # result.forecast() gives the ARMA mean and GARCH variance for t+1.
    # The arch library works in %-return space, so we unscale by /100.
    fcast = result.forecast(horizon=1)

    # Mean forecast: μ_{t+1} = c + φ·r_t + θ·ε_t  (ARMA conditional mean)
    mu_forecast = float(fcast.mean.iloc[-1, 0]) / 100.0

    # Variance forecast: σ²_{t+1} = ω + α·ε²_t + β·σ²_t
    sigma_forecast = float(np.sqrt(fcast.variance.iloc[-1, 0])) / 100.0

    # --- In-sample standardised innovations (for NIG fitting) --------
    # Drop NaN padding that arch inserts at the start of the arrays
    resid       = pd.Series(result.resid).dropna().values / 100
    sigma       = pd.Series(result.conditional_volatility).dropna().values / 100

    # Align lengths — arch occasionally produces arrays of slightly different length
    min_len     = min(len(resid), len(sigma))
    resid       = resid[-min_len:]
    sigma       = sigma[-min_len:]
    innovations = resid / sigma

    return GARCHResult(
        innovations    = innovations,
        sigma_t        = sigma,
        mu_forecast    = mu_forecast,
        sigma_forecast = sigma_forecast,
        params         = params,
        converged      = converged,
    )


def rolling_window_innovations(
    returns:            pd.Series,
    estimation_window:  int = 250,
    assessment_window:  int = 1000,
) -> pd.DataFrame:
    """
    Slide an estimation window across the return series.
    At each step: fit ARMA-GARCH, fit NIG to in-sample innovations,
    compute one-step-ahead VaR/CVaR, and record the PIT value.

    This is the CORRECTED version: NIG is fitted per-window (inside
    the loop), not once globally after the loop.

    Following Kaufman (AMS 603):
      - Slide 18: fit ARMA-GARCH with Gaussian MLE
      - Slide 33: affine transform  r_{t+1} ~ NIG(params, μ + μ_NIG, σ·δ_NIG)
      - Slide 36: PIT values  u_t = F_t(r_t) should be U[0,1]

    Parameters
    ----------
    returns           : full return series (needs >= est + assess periods)
    estimation_window : periods used to fit ARMA-GARCH at each step
    assessment_window : number of consecutive one-step predictions

    Returns
    -------
    DataFrame indexed by prediction date with columns:
        return          — actual log return on that date
        mu_forecast     — ARMA conditional mean forecast μ_{t+1}
        sigma_forecast  — GARCH conditional volatility forecast σ_{t+1}
        innovation      — out-of-sample standardised residual (r - μ) / σ
        nig_alpha       — NIG α (tail heaviness) for this window
        nig_beta        — NIG β (asymmetry) for this window
        nig_mu          — NIG μ (location) for this window
        nig_delta       — NIG δ (scale) for this window
        nig_converged   — whether NIG MLE converged for this window
        t_dof           — Student-T degrees of freedom for this window
        var_95          — VaR at 95% confidence
        var_99          — VaR at 99% confidence
        cvar_95         — CVaR at 95% confidence
        cvar_99         — CVaR at 99% confidence
        pit_nig         — PIT value from per-window NIG CDF
        pit_t           — PIT value from per-window Student-T CDF
        pit_gauss       — PIT value from Gaussian Φ(z) CDF
    """
    try:
        from src.nig import fit_nig_mle, compute_var, compute_cvar, nig_cdf
    except ImportError:
        from nig import fit_nig_mle, compute_var, compute_cvar, nig_cdf

    from scipy import stats

    n     = len(returns)
    start = n - assessment_window   # index of first prediction date

    records     = []
    n_garch_fail = 0
    n_nig_fail   = 0

    # Carry-forward state for convergence failures
    last_mu    = 0.0
    last_sigma = np.std(returns.values)  # unconditional vol as initial fallback
    last_nig   = None
    last_t_dof = None

    for i in range(assessment_window):
        est_start = start - estimation_window + i
        est_end   = start + i                       # exclusive
        pred_idx  = start + i                       # prediction date index

        window = returns.iloc[est_start:est_end].values
        actual = returns.iloc[pred_idx]
        date   = returns.index[pred_idx]

        # --- Step 1: ARMA-GARCH fitting ---
        garch_ok = True
        try:
            garch_res  = fit_arma_garch(window)
            mu_hat     = garch_res.mu_forecast
            sigma_hat  = garch_res.sigma_forecast
            innovations = garch_res.innovations
            last_mu    = mu_hat
            last_sigma = sigma_hat
        except RuntimeError:
            n_garch_fail += 1
            garch_ok    = False
            mu_hat      = last_mu
            sigma_hat   = last_sigma
            innovations = None

        # --- Step 2: NIG fitting on in-sample innovations ---
        nig_ok = False
        if garch_ok and innovations is not None:
            nig_params, nig_ok = fit_nig_mle(innovations)
            if nig_ok:
                last_nig = nig_params
            elif last_nig is not None:
                nig_params = last_nig  # carry forward
            # else: nig_params is the MoM fallback from fit_nig_mle
        else:
            if last_nig is not None:
                nig_params = last_nig
                nig_ok = False
            else:
                # Very first window failed — skip this prediction
                continue

        if not nig_ok:
            n_nig_fail += 1

        # --- Step 2b: Student-T fitting on in-sample innovations ---
        # Following Kaufman (slide 29): fit T(dof, loc=0, scale=1)
        # scipy.stats.t.fit with floc=0, fscale=1 fixes location and scale,
        # fitting only the degrees of freedom — exactly Kaufman's approach.
        if garch_ok and innovations is not None:
            try:
                t_dof, t_loc, t_scale = stats.t.fit(innovations, floc=0, fscale=1)
                last_t_dof = t_dof
            except Exception:
                t_dof = last_t_dof if last_t_dof is not None else 5.0
        else:
            t_dof = last_t_dof if last_t_dof is not None else 5.0

        # --- Step 3: VaR / CVaR via affine transform (Kaufman slide 33) ---
        var_95  = compute_var(mu_hat, sigma_hat, nig_params, level=0.95)
        var_99  = compute_var(mu_hat, sigma_hat, nig_params, level=0.99)
        cvar_95 = compute_cvar(mu_hat, sigma_hat, nig_params, level=0.95)
        cvar_99 = compute_cvar(mu_hat, sigma_hat, nig_params, level=0.99)

        # --- Step 4: Out-of-sample standardised innovation & PIT ---
        # z_{t+1} = (r_{t+1} - μ_{t+1}) / σ_{t+1}
        z_oos = (actual - mu_hat) / sigma_hat

        # PIT values — all three distributions evaluated on the same z_oos
        # NIG PIT: u_t = F_NIG(z_oos)           (per-window NIG)
        # T PIT:   u_t = F_T(z_oos; dof)        (per-window Student-T, Kaufman slide 36)
        # Gauss PIT: u_t = Φ(z_oos)             (N(0,1) — no free params)
        pit_nig   = float(nig_cdf(np.array([z_oos]), nig_params)[0])
        pit_t     = float(stats.t.cdf(z_oos, t_dof, loc=0, scale=1))
        pit_gauss = float(stats.norm.cdf(z_oos))

        records.append({
            "date":           date,
            "return":         actual,
            "mu_forecast":    mu_hat,
            "sigma_forecast": sigma_hat,
            "innovation":     z_oos,
            "nig_alpha":      nig_params.alpha,
            "nig_beta":       nig_params.beta,
            "nig_mu":         nig_params.mu,
            "nig_delta":      nig_params.delta,
            "nig_converged":  nig_ok,
            "t_dof":          t_dof,
            "var_95":         var_95,
            "var_99":         var_99,
            "cvar_95":        cvar_95,
            "cvar_99":        cvar_99,
            "pit_nig":        pit_nig,
            "pit_t":          pit_t,
            "pit_gauss":      pit_gauss,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{assessment_window} windows complete "
                  f"(GARCH fails: {n_garch_fail}, NIG fails: {n_nig_fail})")

    df = pd.DataFrame(records).set_index("date")
    print(f"\nDone. {len(df)} predictions.")
    print(f"  GARCH convergence failures: {n_garch_fail} "
          f"({100*n_garch_fail/assessment_window:.1f}%)")
    print(f"  NIG convergence failures:   {n_nig_fail} "
          f"({100*n_nig_fail/assessment_window:.1f}%)")
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