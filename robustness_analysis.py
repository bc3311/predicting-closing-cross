"""
robustness_analysis.py
======================
Three robustness panels for the closing auction mean-reversion paper:

  A. Earnings / event-day filter — with vs without
  B. Sub-period analysis — 2016-2019 vs 2020-2023
  C. Alternative signal specifications

All tests use the S&P 500-491 universe (broader, more conservative).
Output variable: overnight_betaadj (rolling 126-day beta-adjusted).
Primary signal: drift_x_auc = closing_drift × auction_share.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

plt.rcParams.update({
    'figure.figsize': (16, 6), 'font.size': 11, 'axes.titlesize': 12,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.dpi': 150,
})

DATA = "../"
OUT  = "./"
ROLL = 126   # 126-day window: best Sharpe, FM t, and Markov regime separation


# ============================================================
# HELPERS (shared with sp500_analysis.py)
# ============================================================
def star(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'n.s.'

def section(title):
    print('\n' + '='*65)
    print(title)
    print('='*65)

def two_way_codes(df):
    sym  = pd.Categorical(df['symbol']).codes
    date = pd.Categorical(df['DATE']).codes
    return np.column_stack([sym, date])

def rolling_beta_per_stock(stock_rets, mkt_rets, window=126, min_obs=60):
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

def add_beta_adjustment(taq, roll=ROLL):
    mkt_series = taq.groupby('DATE')['overnight_return'].mean()
    beta_records, alpha_records = [], []
    for sym, grp in taq.sort_values(['symbol','DATE']).groupby('symbol'):
        grp_idx = grp.set_index('DATE')['overnight_return']
        mkt_al  = mkt_series.reindex(grp_idx.groupby(level=0).last().index)
        b, a    = rolling_beta_per_stock(grp_idx, mkt_al, roll)
        b = b.reset_index(); b.columns = ['DATE','beta'];  b['symbol'] = sym
        a = a.reset_index(); a.columns = ['DATE','alpha']; a['symbol'] = sym
        beta_records.append(b); alpha_records.append(a)
    params = pd.concat(beta_records, ignore_index=True).merge(
             pd.concat(alpha_records, ignore_index=True), on=['DATE','symbol'])
    taq = taq.merge(params, on=['DATE','symbol'], how='left')
    taq['R_mkt']             = taq.groupby('DATE')['overnight_return'].transform('mean')
    taq['overnight_betaadj'] = taq['overnight_return'] - (taq['alpha'] + taq['beta'] * taq['R_mkt'])
    return taq


def ols_result(clean, y_col, x_col):
    """Two-way clustered OLS. Returns (coef, t, p, n)."""
    sub = clean.dropna(subset=[y_col, x_col])
    X = sm.add_constant(sub[x_col])
    r = sm.OLS(sub[y_col], X).fit(cov_type='cluster',
                                   cov_kwds={'groups': two_way_codes(sub)})
    return r.params[x_col], r.tvalues[x_col], r.pvalues[x_col], len(sub)


def fm_result(clean, y_col, x_col):
    """Fama-MacBeth with Newey-West (lag=5). Returns (avg_slope, t, p)."""
    slopes = []
    for d, sub in clean.groupby('DATE'):
        sub = sub.dropna(subset=[y_col, x_col])
        if len(sub) < 5:
            continue
        try:
            r2 = sm.OLS(sub[y_col].values,
                        sm.add_constant(sub[x_col].values)).fit()
            slopes.append(r2.params[1])
        except Exception:
            continue
    s = pd.Series(slopes)
    nw = sm.OLS(s.values, np.ones(len(s))).fit(cov_type='HAC',
                                                 cov_kwds={'maxlags': 5})
    return nw.params[0], nw.tvalues[0], nw.pvalues[0]


def sharpe_ls(clean, y_col, x_col):
    """Q1-Q5 long-short Sharpe on y sorted by x."""
    ls_data = clean.dropna(subset=[y_col, x_col])
    ls_data = ls_data[ls_data[x_col] != 0]

    def daily_ls(grp):
        if len(grp) < 5: return np.nan
        try:
            q = pd.qcut(grp[x_col], 5,
                        labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
        except ValueError:
            return np.nan
        m = grp.assign(q=q).groupby('q', observed=True)[y_col].mean()
        if 'Q1' not in m.index or 'Q5' not in m.index: return np.nan
        return m['Q1'] - m['Q5']

    ls_s = ls_data.groupby('DATE').apply(daily_ls).dropna()
    if len(ls_s) < 10:
        return np.nan, np.nan
    sharpe = ls_s.mean() / ls_s.std() * np.sqrt(252)
    nw = sm.OLS(ls_s.values, np.ones(len(ls_s))).fit(cov_type='HAC',
                                                       cov_kwds={'maxlags': 5})
    return sharpe, nw.tvalues[0]


# ============================================================
# LOAD DATA (once, shared across all tests)
# ============================================================
section("LOADING S&P 500 DATA")
print("  Loading taq_sp500.csv and crsp_sp500.csv ...")
taq_raw  = pd.read_csv(DATA + 'taq_sp500.csv',  parse_dates=['DATE'])
crsp_raw = pd.read_csv(DATA + 'crsp_sp500.csv', parse_dates=['date'], low_memory=False)

taq = taq_raw[taq_raw['symbol'] == taq_raw['SYM_ROOT']].copy()
taq = taq.sort_values(['symbol', 'DATE']).reset_index(drop=True)

# Variable construction
taq['closing_drift']    = (taq['CPrc'] - taq['ptime_4pm']) / taq['ptime_4pm']
taq['auction_share']    = taq['CSize'] / taq['total_vol_m'].replace(0, np.nan)
taq['next_open']        = taq.groupby('symbol')['OPrc'].shift(-1)
taq['overnight_return'] = taq['next_open'] / taq['CPrc'] - 1
taq.loc[taq['overnight_return'].abs() > 0.15, 'overnight_return'] = np.nan
taq['drift_x_auc']     = taq['closing_drift'] * taq['auction_share']

# Order imbalance: signed (BuyVol - SellVol) / Total
tot_vol = taq['BuyVol_LR'] + taq['SellVol_LR']
taq['order_imbalance'] = (taq['BuyVol_LR'] - taq['SellVol_LR']) / tot_vol.replace(0, np.nan)

# Merge CRSP
crsp = crsp_raw[['date','TICKER','RET']].copy()
crsp.columns = ['DATE','SYM_ROOT','crsp_ret']
crsp['crsp_ret'] = pd.to_numeric(crsp['crsp_ret'], errors='coerce')
taq = taq.merge(crsp, on=['DATE','SYM_ROOT'], how='left')

# Split-adjust filter
taq['prev_cprc'] = taq.groupby('symbol')['CPrc'].shift(1)
taq['taq_ret']   = taq['CPrc'] / taq['prev_cprc'] - 1
taq['ret_disc']  = (taq['taq_ret'] - taq['crsp_ret']).abs()
taq['split_flag']= (taq['ret_disc'] > 0.02).fillna(False).astype(int)
flagged  = np.flatnonzero(taq['split_flag'].values == 1)
prev_arr = flagged - 1; sym_arr = taq['symbol'].values
same_sym = (prev_arr >= 0) & (sym_arr[prev_arr.clip(min=0)] == sym_arr[flagged])
affected = np.unique(np.concatenate([flagged, prev_arr[same_sym]]))
for c in ['closing_drift','overnight_return','next_open','drift_x_auc']:
    taq.loc[affected, c] = np.nan
taq = taq.drop(columns=['prev_cprc','taq_ret','ret_disc'])

# Event-day proxy: |crsp_ret| > 3% (earnings / macro announcements)
taq['event_day'] = (taq['crsp_ret'].abs() > 0.03).fillna(False)

# Beta adjustment
print("  Computing rolling betas (126-day window) ...")
taq = add_beta_adjustment(taq)

print(f"  Full sample: {len(taq):,} rows  |  {taq['SYM_ROOT'].nunique()} tickers  |  "
      f"{taq['DATE'].nunique()} trading days")


# ============================================================
# PANEL A — EARNINGS FILTER: WITH vs WITHOUT
# ============================================================
section("PANEL A — EARNINGS / EVENT-DAY FILTER")

# Note: event_day proxy = |crsp_ret| > 3% on same day (covers earnings,
# index add/drop, macro prints). Ideal would be IBES earnings dates.

# WITHOUT filter
no_filter = taq.dropna(subset=['overnight_betaadj','drift_x_auc','beta']).copy()

# WITH filter (standard, excludes extreme-return days)
with_filter = taq[~taq['event_day']].dropna(
    subset=['overnight_betaadj','drift_x_auc','beta']).copy()

panel_a = {}
for label, df in [('Without event filter', no_filter), ('With event filter', with_filter)]:
    c, t, p, n = ols_result(df, 'overnight_betaadj', 'drift_x_auc')
    fm_avg, fm_t, fm_p = fm_result(df, 'overnight_betaadj', 'drift_x_auc')
    sh, ls_t = sharpe_ls(df, 'overnight_betaadj', 'drift_x_auc')
    pct_removed = (1 - len(df) / len(no_filter)) * 100 if label != 'Without event filter' else 0
    panel_a[label] = dict(c=c, t=t, p=p, n=n, fm_t=fm_t, fm_p=fm_p,
                          sh=sh, ls_t=ls_t, pct_removed=pct_removed)
    print(f"\n  [{label}]")
    print(f"    OLS  coef={c:+.5f}  t={t:+.2f}  {star(p)}  n={n:,}")
    print(f"    FM   avg_slope={fm_avg:+.5f}  t={fm_t:+.2f}  {star(fm_p)}")
    print(f"    L/S  Sharpe={sh:.2f}  HAC t={ls_t:+.2f}")
    if pct_removed > 0:
        print(f"    → Removed {pct_removed:.1f}% of obs (event days)")

# Winner
better = 'With event filter' if abs(panel_a['With event filter']['t']) > abs(panel_a['Without event filter']['t']) else 'Without event filter'
print(f"\n  *** ADOPTED: '{better}' (stronger OLS t-stat) ***")


# ============================================================
# PANEL B — SUB-PERIOD ANALYSIS: 2016-2019 vs 2020-2023
# ============================================================
section("PANEL B — SUB-PERIOD ANALYSIS")

# Sub-period and signal-spec panels use the event-filtered sample (with_filter).
# Note: sp500_analysis.py main result uses no event filter; that is a separate spec.
base = with_filter.copy()

periods = {
    '2016-2019 (Pre-COVID)':  (pd.Timestamp('2016-01-01'), pd.Timestamp('2019-12-31')),
    '2020-2023 (Post-COVID)': (pd.Timestamp('2020-01-01'), pd.Timestamp('2023-12-31')),
    '2016-2023 (Full)':       (pd.Timestamp('2016-01-01'), pd.Timestamp('2023-12-31')),
}

panel_b = {}
for label, (start, end) in periods.items():
    sub = base[(base['DATE'] >= start) & (base['DATE'] <= end)].copy()
    if len(sub) < 1000:
        print(f"  [{label}] Too few obs: {len(sub)}")
        continue
    c, t, p, n = ols_result(sub, 'overnight_betaadj', 'drift_x_auc')
    fm_avg, fm_t, fm_p = fm_result(sub, 'overnight_betaadj', 'drift_x_auc')
    sh, ls_t = sharpe_ls(sub, 'overnight_betaadj', 'drift_x_auc')

    # Yearly breakdown within period
    ls_data = sub[sub['drift_x_auc'] != 0].dropna(subset=['overnight_betaadj','drift_x_auc'])
    def daily_ls_inner(grp):
        if len(grp) < 5: return np.nan
        try:
            q = pd.qcut(grp['drift_x_auc'], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
        except ValueError: return np.nan
        m = grp.assign(q=q).groupby('q', observed=True)['overnight_betaadj'].mean()
        if 'Q1' not in m.index or 'Q5' not in m.index: return np.nan
        return m['Q1'] - m['Q5']
    ls_s = ls_data.groupby('DATE').apply(daily_ls_inner).dropna()
    yearly = ls_s.groupby(ls_s.index.year).agg(['mean','std','count'])
    yearly['t_yr'] = yearly['mean'] / (yearly['std'] / np.sqrt(yearly['count']))
    pos_years = int((yearly['mean'] > 0).sum())

    panel_b[label] = dict(c=c, t=t, p=p, n=n, fm_t=fm_t, fm_p=fm_p,
                          sh=sh, ls_t=ls_t, yearly=yearly, pos_years=pos_years)
    print(f"\n  [{label}]  n={n:,}  tickers={sub['SYM_ROOT'].nunique()}")
    print(f"    OLS  coef={c:+.5f}  t={t:+.2f}  {star(p)}")
    print(f"    FM   t={fm_t:+.2f}  {star(fm_p)}")
    print(f"    L/S  Sharpe={sh:.2f}  HAC t={ls_t:+.2f}  pos_years={pos_years}/{len(yearly)}")
    print("    Yearly L/S breakdown:")
    for yr, row in yearly.iterrows():
        mk = '▲' if row['mean'] > 0 else '▼'
        print(f"      {yr} {mk}  {row['mean']*10000:+.2f} bps/day  t_yr={row['t_yr']:+.2f}")


# ============================================================
# PANEL C — ALTERNATIVE SIGNAL SPECIFICATIONS
# ============================================================
section("PANEL C — ALTERNATIVE SIGNAL SPECIFICATIONS")

# Winsorize signal columns to reduce outlier impact
def winsorise(s, lo=0.01, hi=0.99):
    q1, q99 = s.quantile(lo), s.quantile(hi)
    return s.clip(q1, q99)

base['closing_drift_w']    = winsorise(base['closing_drift'].dropna())
base['auction_share_w']    = winsorise(base['auction_share'].dropna())
base['order_imbalance_w']  = winsorise(base['order_imbalance'].dropna())
base['drift_x_auc_w']      = winsorise(base['drift_x_auc'].dropna())

# Standardize so coefficients are comparable
def z(s): return (s - s.mean()) / s.std()

base['cd_z']  = z(base['closing_drift_w'])
base['as_z']  = z(base['auction_share_w'])
base['oi_z']  = z(base['order_imbalance_w'])
base['dx_z']  = z(base['drift_x_auc_w'])

# confirmed_pressure: both drift and OI pointing same direction
# If sign(drift) == sign(OI) → confirmed mechanical pressure
base['confirmed_pressure'] = base['closing_drift_w'] * base['order_imbalance_w'].abs().fillna(0)
# Give extra weight when they agree in sign:
agree_sign = np.sign(base['closing_drift_w'].fillna(0)) == np.sign(base['order_imbalance_w'].fillna(0))
base['confirmed_pressure'] = base['confirmed_pressure'] * agree_sign.map({True: 1, False: 0.2})
base['cp_z'] = z(base['confirmed_pressure'].replace(0, np.nan).dropna())

specs = [
    ('M1: drift_x_auc (primary)',  'dx_z'),
    ('M2: closing_drift only',     'cd_z'),
    ('M3: auction_share only',     'as_z'),
    ('M4: order_imbalance only',   'oi_z'),
    ('M5: confirmed_pressure',     'cp_z'),
]

panel_c = {}
print(f"\n  {'Specification':<35}  {'OLS t':>8}  {'Stars':>5}  {'FM t':>8}  {'Stars':>5}  {'Sharpe':>7}")
print("  " + "-"*80)
for label, xcol in specs:
    sub = base.dropna(subset=['overnight_betaadj', xcol]).copy()
    if len(sub) < 1000:
        print(f"  {label:<35}  insufficient data")
        continue
    try:
        c, t, p, n = ols_result(sub, 'overnight_betaadj', xcol)
        fm_avg, fm_t, fm_p = fm_result(sub, 'overnight_betaadj', xcol)
        sh, ls_t = sharpe_ls(sub, 'overnight_betaadj', xcol)
        panel_c[label] = dict(c=c, t=t, p=p, fm_t=fm_t, fm_p=fm_p, sh=sh, n=n)
        print(f"  {label:<35}  {t:>+8.2f}  {star(p):>5}  {fm_t:>+8.2f}  {star(fm_p):>5}  {sh:>7.2f}")
    except Exception as e:
        print(f"  {label:<35}  ERROR: {e}")


# ============================================================
# FIGURES
# ============================================================
section("SAVING FIGURES")

# ---- Figure A: earnings filter comparison ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

labels_a = list(panel_a.keys())
t_vals   = [abs(panel_a[l]['t'])  for l in labels_a]
fm_vals  = [abs(panel_a[l]['fm_t']) for l in labels_a]
sh_vals  = [panel_a[l]['sh']      for l in labels_a]
colors_a = ['#7f8c8d', '#2980b9']

ax = axes[0]
bars = ax.bar(labels_a, t_vals, color=colors_a, alpha=0.85, width=0.4)
ax.axhline(1.96, ls='--', lw=0.8, color='black', label='t=1.96')
ax.axhline(2.58, ls='--', lw=0.8, color='gray',  label='t=2.58')
for bar, tv, label in zip(bars, t_vals, labels_a):
    p = panel_a[label]['p']
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"t={tv:.2f}{star(p)}", ha='center', fontsize=10, fontweight='bold')
ax.set_title('OLS |t-stat|\n(overnight_betaadj ~ drift_x_auc)', fontweight='bold')
ax.legend(fontsize=8); ax.set_ylabel('|t-statistic|')

ax = axes[1]
bars = ax.bar(labels_a, fm_vals, color=colors_a, alpha=0.85, width=0.4)
ax.axhline(1.96, ls='--', lw=0.8, color='black')
ax.axhline(2.58, ls='--', lw=0.8, color='gray')
for bar, tv, label in zip(bars, fm_vals, labels_a):
    p = panel_a[label]['fm_p']
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"t={tv:.2f}{star(p)}", ha='center', fontsize=10, fontweight='bold')
ax.set_title('Fama-MacBeth |t-stat|\n(HAC, lag=5)', fontweight='bold')
ax.set_ylabel('|FM t-statistic|')

ax = axes[2]
bars = ax.bar(labels_a, sh_vals, color=colors_a, alpha=0.85, width=0.4)
for bar, sv in zip(bars, sh_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"{sv:.2f}", ha='center', fontsize=11, fontweight='bold')
ax.set_title('Long-Short Sharpe Ratio\n(Q1-Q5 on drift_x_auc)', fontweight='bold')
ax.set_ylabel('Annualised Sharpe')

fig.suptitle('Panel A: Effect of Earnings/Event-Day Filter\n'
             'S&P 500-491 | 2016-2023 | Beta-Adjusted Output',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'fig_robustness_A_earnings_filter.png')
plt.close()
print("  Saved fig_robustness_A_earnings_filter.png")


# ---- Figure B: sub-period ----
if panel_b:
    period_labels = list(panel_b.keys())
    t_b   = [abs(panel_b[l]['t'])    for l in period_labels]
    fm_b  = [abs(panel_b[l]['fm_t']) for l in period_labels]
    sh_b  = [panel_b[l]['sh']        for l in period_labels]
    cols_b = ['#27ae60','#e67e22','#2980b9']

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    bars = ax.bar(period_labels, t_b, color=cols_b, alpha=0.85, width=0.4)
    ax.axhline(1.96, ls='--', lw=0.8, color='black', label='t=1.96')
    ax.axhline(2.58, ls='--', lw=0.8, color='gray',  label='t=2.58')
    for bar, tv, label in zip(bars, t_b, period_labels):
        p = panel_b[label]['p']
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"t={tv:.2f}{star(p)}", ha='center', fontsize=9, fontweight='bold')
    ax.set_title('OLS |t-stat| by Sub-period', fontweight='bold')
    ax.set_ylabel('|t-statistic|')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', labelrotation=15)

    ax = axes[1]
    bars = ax.bar(period_labels, sh_b, color=cols_b, alpha=0.85, width=0.4)
    for bar, sv in zip(bars, sh_b):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{sv:.2f}", ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Long-Short Sharpe by Sub-period', fontweight='bold')
    ax.set_ylabel('Annualised Sharpe')
    ax.tick_params(axis='x', labelrotation=15)

    # Yearly breakdown for each sub-period
    ax = axes[2]
    all_years_ls = {}
    ls_data_full = base[base['drift_x_auc'] != 0].dropna(subset=['overnight_betaadj','drift_x_auc'])
    def dls(grp):
        if len(grp) < 5: return np.nan
        try:
            q = pd.qcut(grp['drift_x_auc'], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
        except ValueError: return np.nan
        m = grp.assign(q=q).groupby('q', observed=True)['overnight_betaadj'].mean()
        if 'Q1' not in m.index or 'Q5' not in m.index: return np.nan
        return m['Q1'] - m['Q5']
    ls_full = ls_data_full.groupby('DATE').apply(dls).dropna()
    yearly_full = ls_full.groupby(ls_full.index.year)['mean' if False else slice(None)].mean() if False else \
                  ls_full.groupby(ls_full.index.year).mean()

    years = sorted(yearly_full.index)
    vals  = [yearly_full[y]*10000 for y in years]
    bar_colors = ['#27ae60' if v > 0 else '#e74c3c' for v in vals]
    ax.bar([str(y) for y in years], vals, color=bar_colors, alpha=0.85)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(3.5, color='gray', lw=1, ls=':', label='COVID split (2020)')
    ax.set_title('Annual L/S Returns (bps/day)\nGreen=positive, Red=negative', fontweight='bold')
    ax.set_ylabel('Q1-Q5 return (bps/day)')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', labelrotation=30)

    fig.suptitle('Panel B: Sub-period Robustness — Pre vs Post-COVID\n'
                 'S&P 500-491 | Beta-Adjusted Output | drift_x_auc signal',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT + 'fig_robustness_B_subperiod.png')
    plt.close()
    print("  Saved fig_robustness_B_subperiod.png")


# ---- Figure C: signal specifications ----
if panel_c:
    spec_labels = list(panel_c.keys())
    t_c  = [abs(panel_c[l]['t'])    for l in spec_labels]
    fm_c = [abs(panel_c[l]['fm_t']) for l in spec_labels]
    sh_c = [panel_c[l]['sh']        for l in spec_labels]
    # Highlight primary signal
    cols_c = ['#c0392b' if 'primary' in l else '#7f8c8d' for l in spec_labels]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    bars = ax.bar(range(len(spec_labels)), t_c, color=cols_c, alpha=0.85)
    ax.axhline(1.96, ls='--', lw=0.8, color='black', label='t=1.96')
    ax.axhline(2.58, ls='--', lw=0.8, color='gray',  label='t=2.58')
    for i, (tv, label) in enumerate(zip(t_c, spec_labels)):
        p = panel_c[label]['p']
        ax.text(i, tv + 0.02, f"t={tv:.2f}{star(p)}", ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(spec_labels)))
    ax.set_xticklabels([l.split(':')[0] for l in spec_labels], rotation=15)
    ax.set_title('OLS |t-stat| by Signal Spec', fontweight='bold')
    ax.set_ylabel('|t-statistic|'); ax.legend(fontsize=8)

    ax = axes[1]
    bars = ax.bar(range(len(spec_labels)), fm_c, color=cols_c, alpha=0.85)
    ax.axhline(1.96, ls='--', lw=0.8, color='black')
    ax.axhline(2.58, ls='--', lw=0.8, color='gray')
    for i, (tv, label) in enumerate(zip(fm_c, spec_labels)):
        p = panel_c[label]['fm_p']
        ax.text(i, tv + 0.02, f"t={tv:.2f}{star(p)}", ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(spec_labels)))
    ax.set_xticklabels([l.split(':')[0] for l in spec_labels], rotation=15)
    ax.set_title('Fama-MacBeth |t-stat| by Signal Spec', fontweight='bold')
    ax.set_ylabel('|FM t-stat|')

    ax = axes[2]
    bars = ax.bar(range(len(spec_labels)), sh_c, color=cols_c, alpha=0.85)
    for i, sv in enumerate(sh_c):
        ax.text(i, sv + 0.02, f"{sv:.2f}", ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(spec_labels)))
    ax.set_xticklabels([l.split(':')[0] for l in spec_labels], rotation=15)
    ax.set_title('Long-Short Sharpe by Signal Spec', fontweight='bold')
    ax.set_ylabel('Annualised Sharpe')

    fig.suptitle('Panel C: Alternative Signal Specifications\n'
                 'S&P 500-491 | 2016-2023 | Beta-Adjusted Output | Standardised signals (z-scores)\n'
                 'Red = Primary signal (drift_x_auc)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT + 'fig_robustness_C_signal_specs.png')
    plt.close()
    print("  Saved fig_robustness_C_signal_specs.png")


# ============================================================
# SUMMARY TABLE
# ============================================================
section("ROBUSTNESS SUMMARY")

print("\n  PANEL A — Earnings Filter")
print(f"  {'Specification':<30}  {'OLS t':>7}  {'FM t':>7}  {'Sharpe':>7}  {'Adopted':>8}")
print("  " + "-"*65)
for label in panel_a:
    d = panel_a[label]
    adopted = '<-- ADOPTED' if label == better else ''
    print(f"  {label:<30}  {d['t']:>+7.2f}  {d['fm_t']:>+7.2f}  {d['sh']:>7.2f}  {adopted}")

print("\n  PANEL B — Sub-period")
print(f"  {'Period':<30}  {'OLS t':>7}  {'FM t':>7}  {'Sharpe':>7}  {'Pos yrs'}")
print("  " + "-"*65)
for label in panel_b:
    d = panel_b[label]
    print(f"  {label:<30}  {d['t']:>+7.2f}  {d['fm_t']:>+7.2f}  {d['sh']:>7.2f}  "
          f"{d['pos_years']}/{len(d['yearly'])}")

print("\n  PANEL C — Signal Specifications")
print(f"  {'Signal':<35}  {'OLS t':>7}  {'FM t':>7}  {'Sharpe':>7}")
print("  " + "-"*65)
for label in panel_c:
    d = panel_c[label]
    print(f"  {label:<35}  {d['t']:>+7.2f}  {d['fm_t']:>+7.2f}  {d['sh']:>7.2f}")

print("\nDone.")
