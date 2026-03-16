# ARMA-GARCH Beta Risk Model

A professional-grade next-day equity risk model built on ARMA-GARCH(1,1)
with Normal Inverse Gaussian (NIG) heavy-tailed innovations.

Produces daily Value at Risk (VaR) and Conditional Value at Risk (CVaR)
forecasts via a 250-period rolling estimation window. Model performance is
assessed across multiple asset classes using binomial backtesting,
Kolmogorov-Smirnov, Anderson-Darling, and Christoffersen statistics.

## Repository structure
```
src/             Core model modules — importable, independently testable
notebooks/       Analysis notebooks 01–05, designed to run in sequence
data/            Raw and processed data (excluded from version control)
outputs/         Generated figures and tables (excluded from version control)
report/          Final PDF report (excluded from version control)
```

## Reproducing the analysis
```bash
conda create -n beta-risk-env python=3.11 -y
conda activate beta-risk-env
pip install -r requirements.txt
jupyter lab
```

Open and run notebooks in order: `01_data` → `02_arma_garch` →
`03_nig_fitting` → `04_risk_measures` → `05_assessment`.

## Asset classes

- S&P 500 (^GSPC)
- [additional assets listed here after completion]

## Key results

[populate after completion]

## Dependencies

See `requirements.txt`. Core libraries: `arch`, `scipy`, `statsmodels`, `nlopt`.
