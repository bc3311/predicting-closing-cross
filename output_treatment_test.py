"""
output_treatment_test.py
========================
Tests 7 different treatments of the overnight return (output variable)
to find the cleanest idiosyncratic signal for drift_x_auc to predict.

The core idea: overnight_return contains noise from many sources.
Stripping these out leaves a cleaner target for our auction signal.

Treatments tested:
    T1  Simple mean subtraction     — subtract daily cross-sectional mean (baseline)
    T2  Beta-adjusted               — rolling alpha + beta vs equal-weighted market
    T3  Sector-adjusted             — subtract GICS sector × day mean
    T4  Beta + Sector               — beta for market, then sector residual
    T5  PCA(1)                      — project out 1st principal component (market factor)
    T6  PCA(3)                      — project out 3 PCs (market + 2 sector factors)
    T7  Vol-standardised            — beta-adjusted, then divide by rolling 60-day vol
                                      (normalises heteroskedasticity across stocks)

For each treatment, report:
    - OLS t-stat   (overnight_treated ~ drift_x_auc, 2-way clustered SE)
    - Fama-MacBeth t-stat
    - Long-short Sharpe  (sort by drift_x_auc, Q1-Q5)
    - Markov discrete split: volatile t, calm t, ratio
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams.update({'figure.figsize': (16, 6), 'font.size': 11,
                     'axes.titlesize': 12, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'savefig.dpi': 150})

DATA = "../"
OUT  = "./"
ROLL = 126


# ============================================================
# HELPERS
# ============================================================
def star(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'n.s.'

def two_way_codes(df):
    sym  = pd.Categorical(df['symbol']).codes
    date = pd.Categorical(df['DATE']).codes
    return np.column_stack([sym, date])

def run_ols(df, y):
    X = sm.add_constant(df['drift_x_auc'])
    r = sm.OLS(df[y], X).fit(cov_type='cluster',
                              cov_kwds={'groups': two_way_codes(df)})
    return r.params['drift_x_auc'], r.tvalues['drift_x_auc'], r.pvalues['drift_x_auc']

def run_fm(df, y):
    slopes, dates = [], []
    for d, sub in df.groupby('DATE'):
        if len(sub) < 5: continue
        try:
            r = sm.OLS(sub[y].values,
                       sm.add_constant(sub['drift_x_auc'].values)).fit()
            slopes.append(r.params[1]); dates.append(d)
        except: continue
    s = pd.Series(slopes, index=dates)
    nw = sm.OLS(s.values, np.ones(len(s))).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    return nw.params[0], nw.tvalues[0], nw.pvalues[0]

def run_ls(df, y):
    ls_data = df[['DATE','symbol','drift_x_auc', y]].copy()
    ls_data = ls_data[ls_data['drift_x_auc'] != 0]
    def daily_ls(grp):
        if len(grp) < 5: return np.nan
        try:
            q = pd.qcut(grp['drift_x_auc'], 5,
                        labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
        except: return np.nan
        means = grp.assign(q=q).groupby('q', observed=True)[y].mean()
        if 'Q1' not in means.index or 'Q5' not in means.index: return np.nan
        return means['Q1'] - means['Q5']
    ls = ls_data.groupby('DATE').apply(daily_ls).dropna()
    sharpe = ls.mean() / ls.std() * np.sqrt(252)
    nw = sm.OLS(ls.values, np.ones(len(ls))).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    return sharpe, nw.tvalues[0], nw.pvalues[0], (ls > 0).mean(), ls

def run_markov_split(df, y, P_vol):
    p_df = P_vol.reset_index(); p_df.columns = ['DATE','P_volatile']
    merged = df.merge(p_df, on='DATE', how='left').dropna(
        subset=[y,'drift_x_auc','P_volatile'])
    sub_v = merged[merged['P_volatile'] >= 0.5]
    sub_c = merged[merged['P_volatile'] <  0.5]
    def ols_sub(sub):
        if len(sub) < 50: return np.nan, np.nan, np.nan
        X = sm.add_constant(sub['drift_x_auc'])
        r = sm.OLS(sub[y], X).fit(cov_type='HC1')
        return r.params['drift_x_auc'], r.tvalues['drift_x_auc'], r.pvalues['drift_x_auc']
    cv, tv, pv = ols_sub(sub_v)
    cc, tc, pc = ols_sub(sub_c)
    ratio = abs(cv)/abs(cc) if abs(cc) > 1e-10 else np.nan
    return cv, tv, pv, cc, tc, pc, ratio


# ============================================================
# LOAD & BASE PIPELINE
# ============================================================
print("Loading data...")
taq_raw  = pd.read_csv(DATA + "taq_sp500.csv",  parse_dates=['DATE'])
crsp_raw = pd.read_csv(DATA + "crsp_sp500.csv", parse_dates=['date'], low_memory=False)
sectors  = pd.read_csv('/tmp/sp500_sectors.csv')
sectors.columns = ['SYM_ROOT','sector']

taq = taq_raw[taq_raw['symbol'] == taq_raw['SYM_ROOT']].copy()
taq = taq.sort_values(['symbol','DATE']).reset_index(drop=True)
taq['closing_drift']    = (taq['CPrc'] - taq['ptime_4pm']) / taq['ptime_4pm']
taq['auction_share']    = taq['CSize'] / taq['total_vol_m'].replace(0, np.nan)
taq['next_open']        = taq.groupby('symbol')['OPrc'].shift(-1)
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
taq = taq.merge(sectors, on='SYM_ROOT', how='left')
print(f"Base sample: {len(taq):,} rows, {taq['SYM_ROOT'].nunique()} tickers")


# ============================================================
# TREATMENT 1: SIMPLE MEAN SUBTRACTION
# ============================================================
print("\nT1: Simple mean subtraction...")
taq['T1'] = taq['overnight_return'] - taq.groupby('DATE')['overnight_return'].transform('mean')


# ============================================================
# TREATMENT 2: BETA-ADJUSTED (rolling market model)
# ============================================================
print("T2: Rolling beta adjustment...")

def rolling_beta_per_stock(stock_rets, mkt_rets, window=126, min_obs=60):
    stock_rets = stock_rets.groupby(level=0).last()
    mkt_rets   = mkt_rets.groupby(level=0).last()
    df = pd.DataFrame({'r': stock_rets, 'm': mkt_rets}).dropna()
    if len(df) < min_obs:
        return pd.Series(np.nan, index=stock_rets.index), \
               pd.Series(np.nan, index=stock_rets.index)
    betas = pd.Series(np.nan, index=df.index)
    alphas= pd.Series(np.nan, index=df.index)
    for i in range(window, len(df)):
        w = df.iloc[i-window:i]
        if len(w) < min_obs: continue
        var_val = w['m'].var(ddof=1)
        if var_val < 1e-12: continue
        b = np.cov(w['r'], w['m'], ddof=1)[0,1] / var_val
        betas.iloc[i]  = b
        alphas.iloc[i] = w['r'].mean() - b * w['m'].mean()
    return betas, alphas

mkt_series = taq.groupby('DATE')['overnight_return'].mean()
beta_records, alpha_records = [], []
for sym, grp in taq.sort_values(['symbol','DATE']).groupby('symbol'):
    grp_idx = grp.set_index('DATE')['overnight_return']
    mkt_al  = mkt_series.reindex(grp_idx.groupby(level=0).last().index)
    b, a = rolling_beta_per_stock(grp_idx, mkt_al, ROLL)
    b = b.reset_index(); b.columns=['DATE','beta'];  b['symbol']=sym
    a = a.reset_index(); a.columns=['DATE','alpha']; a['symbol']=sym
    beta_records.append(b); alpha_records.append(a)

params = pd.concat(beta_records, ignore_index=True).merge(
         pd.concat(alpha_records, ignore_index=True), on=['DATE','symbol'])
taq = taq.merge(params, on=['DATE','symbol'], how='left')
taq['R_mkt'] = taq.groupby('DATE')['overnight_return'].transform('mean')
taq['T2']    = taq['overnight_return'] - (taq['alpha'] + taq['beta'] * taq['R_mkt'])
print(f"  Beta coverage: {taq['T2'].notna().sum():,} / {len(taq):,} stock-days")


# ============================================================
# TREATMENT 3: SECTOR-ADJUSTED
# Subtract sector × day mean — removes sector-wide news
# e.g., all tech stocks gapping up on AI announcement
# ============================================================
print("T3: Sector-adjusted...")
sector_mean = taq.groupby(['DATE','sector'])['overnight_return'].transform('mean')
taq['T3']   = taq['overnight_return'] - sector_mean


# ============================================================
# TREATMENT 4: BETA + SECTOR ADJUSTED
# First remove market beta, then remove sector co-movement
# ============================================================
print("T4: Beta + Sector adjusted...")
# Start from beta residual, then subtract sector mean of that residual
taq['T4_pre']    = taq['T2']
sec_mean_of_resid= taq.groupby(['DATE','sector'])['T4_pre'].transform('mean')
taq['T4']        = taq['T4_pre'] - sec_mean_of_resid


# ============================================================
# TREATMENT 5: PCA(1) — project out 1st principal component
# PC1 ≈ market factor, data-driven (no beta estimation needed)
# Full-sample PCA: in-sample, fine for thesis signal demonstration
# ============================================================
print("T5: PCA(1) — projecting out 1 factor...")
ret_wide = taq.pivot_table(index='DATE', columns='symbol', values='overnight_return')
ret_filled = ret_wide.fillna(0)  # fill missing with 0 (conservative)

pca1 = PCA(n_components=1)
scores1   = pca1.fit_transform(ret_filled.values)          # T × 1
loadings1 = pca1.components_                               # 1 × N
explained1= pca1.explained_variance_ratio_[0]
print(f"  PC1 explains {explained1*100:.1f}% of overnight return variance")

reconstruction1 = scores1 @ loadings1                      # T × N
resid1 = ret_filled.values - reconstruction1
resid1_df = pd.DataFrame(resid1, index=ret_wide.index, columns=ret_wide.columns)
resid1_df[ret_wide.isna()] = np.nan                         # restore NaNs where original was missing
resid1_long = resid1_df.stack(future_stack=True).reset_index()
resid1_long.columns = ['DATE','symbol','T5']
taq = taq.merge(resid1_long, on=['DATE','symbol'], how='left')


# ============================================================
# TREATMENT 6: PCA(3) — project out 3 principal components
# PC1=market, PC2+PC3≈sector factors
# ============================================================
print("T6: PCA(3) — projecting out 3 factors...")
pca3 = PCA(n_components=3)
scores3    = pca3.fit_transform(ret_filled.values)
loadings3  = pca3.components_
explained3 = pca3.explained_variance_ratio_.sum()
print(f"  PC1-3 explain {explained3*100:.1f}% of overnight return variance")

reconstruction3 = scores3 @ loadings3
resid3 = ret_filled.values - reconstruction3
resid3_df = pd.DataFrame(resid3, index=ret_wide.index, columns=ret_wide.columns)
resid3_df[ret_wide.isna()] = np.nan
resid3_long = resid3_df.stack(future_stack=True).reset_index()
resid3_long.columns = ['DATE','symbol','T6']
taq = taq.merge(resid3_long, on=['DATE','symbol'], how='left')


# ============================================================
# TREATMENT 7: VOL-STANDARDISED
# Beta-adjusted return ÷ rolling 60-day volatility
# Removes heteroskedasticity — high-vol stocks don't dominate
# Intuition: a 1% move for a low-vol stock is more meaningful
#             than a 1% move for a high-vol stock
# ============================================================
print("T7: Vol-standardised (beta-adjusted ÷ rolling vol)...")
taq_sorted = taq.sort_values(['symbol','DATE'])
rolling_std = (taq_sorted.groupby('symbol')['overnight_return']
               .transform(lambda x: x.shift(1).rolling(60, min_periods=20).std()))
taq['rolling_vol'] = rolling_std.values
taq['T7'] = taq['T2'] / taq['rolling_vol'].replace(0, np.nan)
# Winsorize T7 at ±10 (standardised units)
taq.loc[taq['T7'].abs() > 10, 'T7'] = np.nan
print(f"  T7 coverage: {taq['T7'].notna().sum():,} stock-days")


# ============================================================
# FIT MARKOV ONCE (shared across treatments)
# ============================================================
print("\nFitting Markov HMM (M2: mean_abs_ret)...")
daily_agg = taq.groupby('DATE').agg(
    mean_abs_ret=('overnight_return', lambda x: x.abs().mean())
).dropna().reset_index().sort_values('DATE')
daily_ms = daily_agg.set_index('DATE')

model  = MarkovRegression(daily_ms['mean_abs_ret'], k_regimes=2,
                          trend='c', switching_variance=True)
res_m  = model.fit(maxiter=2000, em_iter=300, disp=False)
s0, s1 = res_m.params['sigma2[0]'], res_m.params['sigma2[1]']
vol    = 1 if s1 > s0 else 0
sv, sc = np.sqrt(max(s0,s1)), np.sqrt(min(s0,s1))
P_vol  = pd.Series(res_m.smoothed_marginal_probabilities[vol],
                   index=daily_ms.index, name='P_volatile')
n_vol  = int((P_vol > 0.5).sum())
print(f"  sigma_vol={sv*10000:.1f}bps  sigma_calm={sc*10000:.1f}bps  vol_days={n_vol}")


# ============================================================
# RUN ALL TREATMENTS
# ============================================================
treatments = {
    'T1  Simple mean sub':      'T1',
    'T2  Beta-adjusted':        'T2',
    'T3  Sector-adjusted':      'T3',
    'T4  Beta+Sector':          'T4',
    'T5  PCA(1)':               'T5',
    'T6  PCA(3)':               'T6',
    'T7  Vol-standardised':     'T7',
}

results = {}
print("\n" + "="*90)
print(f"{'Treatment':<26}  {'OLS t':>8}  {'FM t':>8}  {'Sharpe':>8}  "
      f"{'LS t':>8}  {'WinRate':>8}  {'Vol_t':>8}  {'Calm_t':>8}  {'Ratio':>7}")
print("-"*90)

for label, col in treatments.items():
    clean = taq[~taq['event_day']].dropna(subset=[col, 'drift_x_auc']).copy()
    if len(clean) < 1000:
        print(f"  {label:<24}  insufficient data")
        continue

    c_ols, t_ols, p_ols = run_ols(clean, col)
    fm_avg, fm_t, fm_p  = run_fm(clean, col)
    sharpe, ls_t, ls_p, wr, ls_ser = run_ls(clean, col)
    cv, tv, pv, cc, tc, pc, ratio  = run_markov_split(clean, col, P_vol)

    results[label] = {
        'col': col, 'n': len(clean),
        'ols': (c_ols, t_ols, p_ols),
        'fm':  (fm_avg, fm_t, fm_p),
        'ls':  (sharpe, ls_t, ls_p, wr),
        'ls_series': ls_ser,
        'markov': (cv, tv, pv, cc, tc, pc, ratio),
    }

    print(f"  {label:<24}  {t_ols:>+7.2f}{star(p_ols):<3}  {fm_t:>+7.2f}{star(fm_p):<3}  "
          f"{sharpe:>7.2f}  {ls_t:>+7.2f}{star(ls_p):<3}  {wr*100:>7.1f}%  "
          f"{tv:>+7.2f}{star(pv):<3}  {tc:>+7.2f}{star(pc):<3}  {ratio:>6.2f}x")


# ============================================================
# BEST TREATMENT SUMMARY
# ============================================================
print("\n" + "="*90)
print("BEST TREATMENT BY METRIC:")

ols_winner  = max(results, key=lambda k: abs(results[k]['ols'][1]))
fm_winner   = max(results, key=lambda k: abs(results[k]['fm'][1]))
sh_winner   = max(results, key=lambda k: results[k]['ls'][0])
rat_winner  = max(results, key=lambda k: results[k]['markov'][6]
                  if not np.isnan(results[k]['markov'][6]) else 0)

print(f"  Highest OLS t-stat:       {ols_winner:<26} t={results[ols_winner]['ols'][1]:+.2f}")
print(f"  Highest FM  t-stat:       {fm_winner:<26} t={results[fm_winner]['fm'][1]:+.2f}")
print(f"  Highest Sharpe:           {sh_winner:<26} Sharpe={results[sh_winner]['ls'][0]:.2f}")
print(f"  Highest Markov ratio:     {rat_winner:<26} ratio={results[rat_winner]['markov'][6]:.2f}x")


# ============================================================
# FIGURES
# ============================================================
labels_short = ['T1\nSimple', 'T2\nBeta', 'T3\nSector',
                'T4\nBeta+Sec', 'T5\nPCA(1)', 'T6\nPCA(3)', 'T7\nVol-Std']
ols_ts   = [results[k]['ols'][1]    for k in results]
ols_ps   = [results[k]['ols'][2]    for k in results]
fm_ts    = [results[k]['fm'][1]     for k in results]
fm_ps    = [results[k]['fm'][2]     for k in results]
sharpes  = [results[k]['ls'][0]     for k in results]
ratios   = [results[k]['markov'][6] for k in results]
vol_ts   = [results[k]['markov'][1] for k in results]
calm_ts  = [results[k]['markov'][4] for k in results]

fig, axes = plt.subplots(2, 2, figsize=(18, 10))

def bar_chart(ax, vals, title, ylabel, threshold=None, threshold2=None):
    colors = ['#2166ac' if abs(v) == max(abs(x) for x in vals) else '#6baed6' for v in vals]
    bars = ax.bar(labels_short, vals, color=colors, alpha=0.85, edgecolor='gray', linewidth=0.4)
    if threshold:
        ax.axhline(threshold,  color='black', linestyle='--', linewidth=0.8,
                   label=f'{threshold} threshold')
    if threshold2:
        ax.axhline(threshold2, color='gray',  linestyle='--', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                v + (0.05 if v >= 0 else -0.15),
                f'{v:.2f}', ha='center', fontsize=8)
    return ax

bar_chart(axes[0,0], ols_ts,  'OLS t-stat (overnight ~ drift_x_auc)',
          't-statistic', threshold=-1.96, threshold2=-2.58)
bar_chart(axes[0,1], fm_ts,   'Fama-MacBeth t-stat',
          't-statistic', threshold=-1.96, threshold2=-2.58)
bar_chart(axes[1,0], sharpes, 'Long-Short Sharpe Ratio', 'Sharpe')
bar_chart(axes[1,1], ratios,  'Markov Volatile/Calm Ratio', 'ratio', threshold=1.0)

for ax in axes.flat:
    ax.legend(fontsize=8)

fig.suptitle('Output Variable Treatment Comparison — S&P 500-491\n'
             'Which noise removal gives the cleanest overnight return for drift_x_auc to predict?',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'fig_output_treatments.png')
plt.close()
print("\n  Saved fig_output_treatments.png")

# Long-short cumulative return for each treatment
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes_flat = axes.flat
for i, (label, res) in enumerate(results.items()):
    ax = axes_flat[i]
    cum = res['ls_series'].cumsum() * 10000
    ax.plot(cum.index, cum.values, linewidth=0.9, color='steelblue')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.fill_between(cum.index, cum.values, 0,
                    where=cum.values > 0, alpha=0.2, color='green')
    ax.fill_between(cum.index, cum.values, 0,
                    where=cum.values < 0, alpha=0.2, color='red')
    sh = res['ls'][0]
    ax.set_title(f"{label}\nSharpe={sh:.2f}", fontsize=10, fontweight='bold')
    ax.set_ylabel('Cumul. bps')
axes_flat[-1].set_visible(False)
fig.suptitle('Long-Short Cumulative Return by Treatment\n(sort by drift_x_auc, Q1-Q5)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'fig_output_treatments_ls.png')
plt.close()
print("  Saved fig_output_treatments_ls.png")

print("\n=== output_treatment_test.py done ===")
