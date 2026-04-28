"""
beta_adjustment_test.py
=======================
Tests whether market-model beta adjustment produces a cleaner output variable
than simple cross-sectional mean subtraction.

Current approach (simple):
    overnight_adj = overnight_return - mean(all stocks that day)
    → assumes beta=1 for every stock

Proposed approach (market model):
    rolling beta_i  = cov(R_i, R_mkt) / var(R_mkt)  [126-day rolling]
    rolling alpha_i = mean(R_i) - beta_i * mean(R_mkt)
    overnight_adj_beta = overnight_return - (alpha_i + beta_i * R_mkt)
    → each stock has its own market sensitivity

Compares:
    - Main OLS t-stat: overnight_adj ~ drift_x_auc
    - Fama-MacBeth t-stat
    - Long-short Sharpe
    - Win rate
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams.update({'figure.figsize': (14, 6), 'font.size': 11,
                     'axes.titlesize': 13, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'savefig.dpi': 150})

DATA = "../"
OUT  = "./"
ROLL = 126   # rolling window for beta estimation


# ============================================================
# HELPERS
# ============================================================
def star(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'n.s.'

def two_way_codes(df):
    sym  = pd.Categorical(df['symbol']).codes
    date = pd.Categorical(df['DATE']).codes
    return np.column_stack([sym, date])

def run_ols(df, y, x):
    X = sm.add_constant(df[x])
    r = sm.OLS(df[y], X).fit(cov_type='cluster',
                              cov_kwds={'groups': two_way_codes(df)})
    return r.params[x], r.tvalues[x], r.pvalues[x]

def run_fm(df, y, x):
    slopes, dates = [], []
    for d, sub in df.groupby('DATE'):
        if len(sub) < 5:
            continue
        try:
            r = sm.OLS(sub[y].values, sm.add_constant(sub[x].values)).fit()
            slopes.append(r.params[1])
            dates.append(d)
        except Exception:
            continue
    slopes_s = pd.Series(slopes, index=dates)
    nw = sm.OLS(slopes_s.values, np.ones(len(slopes_s))).fit(
        cov_type='HAC', cov_kwds={'maxlags': 5})
    return nw.params[0], nw.tvalues[0], nw.pvalues[0], len(slopes_s)

def run_longshort(df, signal, y):
    ls_data = df[[signal, y, 'DATE', 'symbol']].copy()
    ls_data = ls_data[ls_data[signal] != 0]

    def daily_ls(grp):
        if len(grp) < 5:
            return np.nan
        try:
            q = pd.qcut(grp[signal], 5, labels=['Q1','Q2','Q3','Q4','Q5'],
                        duplicates='drop')
        except ValueError:
            return np.nan
        means = grp.assign(q=q).groupby('q', observed=True)[y].mean()
        if 'Q1' not in means.index or 'Q5' not in means.index:
            return np.nan
        return means['Q1'] - means['Q5']

    ls = ls_data.groupby('DATE').apply(daily_ls).dropna()
    sharpe   = ls.mean() / ls.std() * np.sqrt(252)
    nw       = sm.OLS(ls.values, np.ones(len(ls))).fit(
        cov_type='HAC', cov_kwds={'maxlags': 5})
    win_rate = (ls > 0).mean()
    yearly   = ls.groupby(ls.index.year).agg(['mean','std','count'])
    yearly['t'] = yearly['mean'] / (yearly['std'] / np.sqrt(yearly['count']))
    return ls, sharpe, nw.tvalues[0], nw.pvalues[0], win_rate, yearly


# ============================================================
# LOAD & STANDARD PIPELINE
# ============================================================
print("Loading data...")
taq_raw  = pd.read_csv(DATA + "taq_sp500.csv",   parse_dates=['DATE'])
crsp_raw = pd.read_csv(DATA + "crsp_sp500.csv",  parse_dates=['date'], low_memory=False)

taq = taq_raw[taq_raw['symbol'] == taq_raw['SYM_ROOT']].copy()
taq = taq.sort_values(['symbol', 'DATE']).reset_index(drop=True)

taq['closing_drift'] = (taq['CPrc'] - taq['ptime_4pm']) / taq['ptime_4pm']
taq['auction_share'] = taq['CSize'] / taq['total_vol_m'].replace(0, np.nan)
taq['next_open']     = taq.groupby('symbol')['OPrc'].shift(-1)
taq['overnight_return'] = taq['next_open'] / taq['CPrc'] - 1
taq.loc[taq['overnight_return'].abs() > 0.15, 'overnight_return'] = np.nan

crsp = crsp_raw[['date','TICKER','RET']].copy()
crsp.columns = ['DATE','SYM_ROOT','crsp_ret']
crsp['crsp_ret'] = pd.to_numeric(crsp['crsp_ret'], errors='coerce')
taq = taq.merge(crsp, on=['DATE','SYM_ROOT'], how='left')

taq['prev_cprc']  = taq.groupby('symbol')['CPrc'].shift(1)
taq['taq_ret']    = taq['CPrc'] / taq['prev_cprc'] - 1
taq['ret_disc']   = (taq['taq_ret'] - taq['crsp_ret']).abs()
taq['split_flag'] = (taq['ret_disc'] > 0.02).fillna(False).astype(int)
flagged  = np.flatnonzero(taq['split_flag'].values == 1)
prev_arr = flagged - 1; sym_arr = taq['symbol'].values
same_sym = (prev_arr >= 0) & (sym_arr[prev_arr.clip(min=0)] == sym_arr[flagged])
affected = np.unique(np.concatenate([flagged, prev_arr[same_sym]]))
for c in ['closing_drift','overnight_return','next_open']:
    taq.loc[affected, c] = np.nan
taq = taq.drop(columns=['prev_cprc','taq_ret','ret_disc'])

taq['event_day']   = (taq['crsp_ret'].abs() > 0.03).fillna(False)
taq['drift_x_auc'] = taq['closing_drift'] * taq['auction_share']

print(f"Raw sample: {len(taq):,} rows, {taq['SYM_ROOT'].nunique()} tickers")


# ============================================================
# METHOD 1: SIMPLE MEAN SUBTRACTION (current)
# ============================================================
print("\nMethod 1: Simple mean subtraction...")
dm = taq.groupby('DATE')['overnight_return'].transform('mean')
taq['overnight_simple'] = taq['overnight_return'] - dm


# ============================================================
# METHOD 2: ROLLING MARKET-MODEL BETA ADJUSTMENT
# ============================================================
print("Method 2: Rolling market-model beta adjustment (126-day window)...")

# Step 1: equal-weighted market return each day
R_mkt = taq.groupby('DATE')['overnight_return'].mean().rename('R_mkt')
taq = taq.merge(R_mkt.reset_index(), on='DATE', how='left')

# Step 2: pivot to wide matrix for vectorized rolling
ret_wide = taq.pivot_table(index='DATE', columns='symbol',
                            values='overnight_return')
mkt_wide = ret_wide.mean(axis=1)   # equal-weighted market

# Step 3: rolling beta = cov(R_i, R_mkt) / var(R_mkt)  [126-day window]
print("  Computing rolling betas (vectorized)...")

def rolling_beta_alpha(ret_wide, mkt_series, window=126):
    """Vectorized rolling beta and alpha for all stocks at once."""
    mkt = mkt_series.values
    betas  = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns, dtype=float)
    alphas = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns, dtype=float)

    for i in range(window, len(ret_wide)):
        r_window   = ret_wide.iloc[i-window:i].values      # (window, n_stocks)
        m_window   = mkt[i-window:i]                        # (window,)
        valid_mask = ~np.isnan(m_window)
        m_w        = m_window[valid_mask]

        if len(m_w) < 60:
            continue

        m_mean = m_w.mean()
        m_var  = np.nanvar(m_w, ddof=1)
        if m_var < 1e-12:
            continue

        for j, col in enumerate(ret_wide.columns):
            r_j = r_window[:, j][valid_mask]
            both_valid = ~np.isnan(r_j)
            if both_valid.sum() < 60:
                continue
            r_jv = r_j[both_valid]
            m_jv = m_w[both_valid]
            cov_val = np.cov(r_jv, m_jv, ddof=1)[0, 1]
            var_val = np.var(m_jv, ddof=1)
            if var_val < 1e-12:
                continue
            b = cov_val / var_val
            a = r_jv.mean() - b * m_jv.mean()
            betas.iloc[i][col]  = b
            alphas.iloc[i][col] = a

    return betas, alphas

# Vectorized per-stock rolling (efficient)
print("  Running rolling estimation (may take ~1 min for 487 stocks)...")

# More efficient: use pandas rolling with apply per stock
def rolling_beta_per_stock(stock_rets, mkt_rets, window=126, min_obs=60):
    # deduplicate — keep last if multiple rows per date
    stock_rets = stock_rets.groupby(level=0).last()
    mkt_rets   = mkt_rets.groupby(level=0).last()

    df = pd.DataFrame({'r': stock_rets, 'm': mkt_rets}).dropna()
    if len(df) < min_obs:
        return pd.Series(np.nan, index=stock_rets.index), \
               pd.Series(np.nan, index=stock_rets.index)

    betas  = pd.Series(np.nan, index=df.index)
    alphas = pd.Series(np.nan, index=df.index)

    for i in range(window, len(df)):
        w = df.iloc[i-window:i]
        if len(w) < min_obs:
            continue
        var_val = w['m'].var(ddof=1)
        if var_val < 1e-12:
            continue
        cov_val = np.cov(w['r'], w['m'], ddof=1)[0,1]
        b = cov_val / var_val
        betas.iloc[i]  = b
        alphas.iloc[i] = w['r'].mean() - b * w['m'].mean()

    return betas, alphas

# Run per stock — build a long-format result
beta_records  = []
alpha_records = []

taq_sorted = taq.sort_values(['symbol','DATE'])
mkt_series = taq_sorted.groupby('DATE')['overnight_return'].mean()

for sym, grp in taq_sorted.groupby('symbol'):
    grp_indexed = grp.set_index('DATE')['overnight_return']
    mkt_aligned = mkt_series.reindex(grp_indexed.groupby(level=0).last().index)
    b, a = rolling_beta_per_stock(grp_indexed, mkt_aligned, ROLL, min_obs=60)
    b = b.reset_index(); b.columns = ['DATE','beta']; b['symbol'] = sym
    a = a.reset_index(); a.columns = ['DATE','alpha']; a['symbol'] = sym
    beta_records.append(b)
    alpha_records.append(a)

betas_long  = pd.concat(beta_records,  ignore_index=True)
alphas_long = pd.concat(alpha_records, ignore_index=True)

params = betas_long.merge(alphas_long, on=['DATE','symbol'])
taq = taq.merge(params, on=['DATE','symbol'], how='left')

# Step 4: compute beta-adjusted overnight return
# predicted = alpha_i + beta_i * R_mkt
# residual  = overnight_return - predicted
taq['predicted']         = taq['alpha'] + taq['beta'] * taq['R_mkt']
taq['overnight_betaadj'] = taq['overnight_return'] - taq['predicted']

n_beta = taq['beta'].notna().sum()
n_adj  = taq['overnight_betaadj'].notna().sum()
print(f"  Beta estimated for {n_beta:,} / {len(taq):,} stock-days")
print(f"  Beta-adjusted overnight available: {n_adj:,} stock-days")
print(f"  Beta stats: mean={taq['beta'].mean():.3f}  std={taq['beta'].std():.3f}  "
      f"min={taq['beta'].min():.3f}  max={taq['beta'].max():.3f}")


# ============================================================
# COMPARE: run same tests on both output variables
# ============================================================
print("\n" + "="*65)
print("COMPARISON: Simple Mean vs Beta-Adjusted Output")
print("="*65)

clean_simple = taq[~taq['event_day']].dropna(
    subset=['overnight_simple','drift_x_auc']).copy()
clean_beta   = taq[~taq['event_day']].dropna(
    subset=['overnight_betaadj','drift_x_auc','beta']).copy()

print(f"\nSample sizes:")
print(f"  Simple:       {len(clean_simple):,} stock-days")
print(f"  Beta-adjusted: {len(clean_beta):,} stock-days")

# OLS
print(f"\n{'Test':<45}  {'Simple':^30}  {'Beta-Adjusted':^30}")
print("-"*105)

c1, t1, p1 = run_ols(clean_simple, 'overnight_simple',   'drift_x_auc')
c2, t2, p2 = run_ols(clean_beta,   'overnight_betaadj',  'drift_x_auc')
print(f"  {'OLS: overnight ~ drift_x_auc':<43}  "
      f"coef={c1:+.4f}  t={t1:+.2f} {star(p1):<5}   "
      f"coef={c2:+.4f}  t={t2:+.2f} {star(p2)}")

# Fama-MacBeth
fm1 = run_fm(clean_simple, 'overnight_simple',  'drift_x_auc')
fm2 = run_fm(clean_beta,   'overnight_betaadj', 'drift_x_auc')
print(f"  {'Fama-MacBeth':<43}  "
      f"avg={fm1[0]:+.4f}  t={fm1[1]:+.2f} {star(fm1[2]):<5}   "
      f"avg={fm2[0]:+.4f}  t={fm2[1]:+.2f} {star(fm2[2])}")

# Long-short
ls1, sh1, lst1, lsp1, wr1, yr1 = run_longshort(clean_simple, 'drift_x_auc', 'overnight_simple')
ls2, sh2, lst2, lsp2, wr2, yr2 = run_longshort(clean_beta,   'drift_x_auc', 'overnight_betaadj')
print(f"  {'Long-Short Sharpe':<43}  {sh1:.2f}{'':28}  {sh2:.2f}")
print(f"  {'Long-Short HAC t':<43}  t={lst1:+.2f} {star(lsp1):<24}  t={lst2:+.2f} {star(lsp2)}")
print(f"  {'Long-Short Win Rate':<43}  {wr1*100:.1f}%{'':27}  {wr2*100:.1f}%")

print(f"\nYear-by-year long-short:")
print(f"{'Year':<6}  {'Simple (bps)':>14}  {'t':>6}  {'BetaAdj (bps)':>14}  {'t':>6}")
print("-"*50)
for yr in sorted(set(yr1.index) & set(yr2.index)):
    r1 = yr1.loc[yr]; r2 = yr2.loc[yr]
    mk1 = '▲' if r1['mean'] > 0 else '▼'
    mk2 = '▲' if r2['mean'] > 0 else '▼'
    print(f"  {yr}  {mk1} {r1['mean']*10000:+.1f}  t={r1['t']:+.2f}    "
          f"{mk2} {r2['mean']*10000:+.1f}  t={r2['t']:+.2f}")


# ============================================================
# FIGURES
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Beta distribution
ax = axes[0]
betas_valid = taq['beta'].dropna()
ax.hist(betas_valid, bins=80, color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.3)
ax.axvline(1.0, color='red', linestyle='--', linewidth=1.2, label='beta=1 (assumed by simple)')
ax.axvline(betas_valid.mean(), color='black', linestyle='--', linewidth=1.2,
           label=f'mean beta={betas_valid.mean():.2f}')
ax.set_xlabel('Rolling beta (stock vs equal-weighted market)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Rolling Betas\n(S&P 500 stocks, 126-day window)', fontweight='bold')
ax.legend(fontsize=9)

# Panel 2: OLS t-stat comparison
ax = axes[1]
labels  = ['Simple\nMean Sub', 'Beta-Model\nAdjusted']
t_stats = [abs(t1), abs(t2)]
colors  = ['#4393c3', '#c0392b']
bars = ax.bar(labels, t_stats, color=colors, alpha=0.85, width=0.4)
ax.axhline(1.96, color='black', linestyle='--', linewidth=0.8, label='t=1.96 (5% sig)')
ax.axhline(2.58, color='gray',  linestyle='--', linewidth=0.8, label='t=2.58 (1% sig)')
for bar, t, p in zip(bars, [t1, t2], [p1, p2]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f't={t:.2f}{star(p)}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('|t-statistic|')
ax.set_title('OLS Signal Strength\novernight ~ drift_x_auc', fontweight='bold')
ax.legend(fontsize=9)

# Panel 3: Long-short cumulative return
ax = axes[2]
cum1 = ls1.cumsum() * 10000
cum2 = ls2.cumsum() * 10000
ax.plot(cum1.index, cum1.values, color='#4393c3', linewidth=1.2,
        label=f'Simple (Sharpe={sh1:.2f})')
ax.plot(cum2.index, cum2.values, color='#c0392b', linewidth=1.2,
        label=f'Beta-Adjusted (Sharpe={sh2:.2f})')
ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
ax.set_ylabel('Cumulative Q1-Q5 return (bps)')
ax.set_title('Long-Short Portfolio\nSimple vs Beta-Adjusted Output', fontweight='bold')
ax.legend(fontsize=9)

fig.suptitle('Market-Model Beta Adjustment vs Simple Mean Subtraction\n'
             '(S&P 500-487 stocks, drift_x_auc signal, 2016–2023)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'fig_beta_adjustment.png')
plt.close()
print("\n  Saved fig_beta_adjustment.png")
print("\n=== beta_adjustment_test.py done ===")
