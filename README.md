# Predicting the Closing Cross: Mean Reversion in NYSE Closing Auction Returns

Columbia University MSFE — Empirical Finance, Spring 2026  
Max Chen, Leon Kwok, Amy Shi, Gloria Gang

## Scripts

| File | Purpose |
|------|---------|
| `sp500_analysis.py` | Main analysis — OLS, Fama-MacBeth, long-short portfolio, Markov regime model (S&P 500-487 and DJIA-30) |
| `robustness_analysis.py` | Robustness panels — event-day filter, sub-period (pre/post COVID), alternative signal specifications |
| `beta_adjustment_test.py` | Validates rolling 126-day market-model beta adjustment vs simple mean subtraction |
| `generate_paper_figures.py` | Generates all publication figures using verified regression outputs from the paper |
| `output_treatment_test.py` | Exploratory comparison of output variable specifications |

## Data

Place `taq_sp500.csv`, `crsp_sp500.csv`, `taq_intraday.csv`, and `crsp_daily.csv` one directory above the scripts (`DATA = "../"`).

## Key Results

- **Signal**: DriftxAuc = ClosingDrift × AuctionShare
- **OLS**: t = −2.70 (two-way clustered SE, Petersen 2009)
- **Fama-MacBeth**: t = −10.64 (Newey-West, 5 lags, 1,885 cross-sections)
- **Long-short Sharpe**: 2.50 (net of 2 bps/day cost, 8/8 years positive)
- **Markov amplification**: 2.16× stronger reversal in volatile regimes vs calm