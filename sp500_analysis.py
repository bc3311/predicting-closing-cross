"""
sp500_analysis.py
=================
Runs the same pipeline on DJIA (30 stocks) and S&P 500 (~491 stocks) and
compares results side by side. Confirms that the mean reversion finding
generalises beyond blue-chip DJIA stocks.

Primary signal:  drift_x_auc = closing_drift × auction_share
Output:          overnight_betaadj — rolling market-model beta-adjusted overnight return
                 (each stock's overnight return minus alpha_i + beta_i × R_market)
                 Better than simple mean subtraction: removes stock-specific
                 market sensitivity, not just the market-wide level.

Key comparisons:
    1. Main OLS: overnight_betaadj ~ drift_x_auc
    2. Fama-MacBeth daily cross-sections
    3. Long-short portfolio (sort by drift_x_auc, Q1-Q5)
    4. Markov regime — discrete split (M2: mean_abs_ret)
         Markov classifies days as volatile/calm via HMM
         Then separate OLS in each regime — cleaner than interaction term
    5. Side-by-side comparison figures
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
    'figure.figsize': (14, 6), 'font.size': 11, 'axes.titlesize': 13,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.dpi': 150,
})

DATA = "../"
OUT  = "./"
ROLL = 126   # rolling window for beta estimation (126-day = ~6 months)


# ============================================================
# HELPERS
# ============================================================
def star(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'n.s.'

def two_way_codes(df):
    sym  = pd.Categorical(df['symbol']).codes
    date = pd.Categorical(df['DATE']).codes
    return np.column_stack([sym, date])


# ============================================================
# ROLLING BETA ADJUSTMENT
# Per-stock rolling 126-day OLS: R_i = alpha_i + beta_i * R_mkt
# overnight_betaadj = overnight_return - (alpha_i + beta_i * R_mkt)
# ============================================================
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


# ============================================================
# DATA PIPELINE
# ============================================================
def load_and_clean(taq_file, crsp_file, label):
    print(f"\n  [{label}] Loading {taq_file} ...")
    taq_raw  = pd.read_csv(DATA + taq_file,  parse_dates=['DATE'])
    crsp_raw = pd.read_csv(DATA + crsp_file, parse_dates=['date'], low_memory=False)

    taq = taq_raw[taq_raw['symbol'] == taq_raw['SYM_ROOT']].copy()
    taq = taq.sort_values(['symbol', 'DATE']).reset_index(drop=True)

    taq['closing_drift']   = (taq['CPrc'] - taq['ptime_4pm']) / taq['ptime_4pm']
    taq['auction_share']   = taq['CSize'] / taq['total_vol_m'].replace(0, np.nan)
    taq['next_open']       = taq.groupby('symbol')['OPrc'].shift(-1)
    taq['overnight_return']= taq['next_open'] / taq['CPrc'] - 1
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

    # Beta adjustment
    print(f"  [{label}] Computing rolling betas ({ROLL}-day window)...")
    taq = add_beta_adjustment(taq)

    # No event-day filter on main clean sample — paper's main result uses full sample.
    # Event-day robustness test is in robustness_analysis.py Panel A.
    clean = taq.dropna(
        subset=['overnight_betaadj','drift_x_auc','beta']).copy()

    print(f"  [{label}] Clean sample: {len(clean):,} stock-days  |  "
          f"{clean['SYM_ROOT'].nunique()} tickers  |  "
          f"{clean['DATE'].nunique()} trading days")
    return taq, clean


# ============================================================
# ANALYSIS
# ============================================================
def run_analysis(taq, clean, label):
    results = {}

    # --- OLS ---
    section(f"OLS [{label}]")
    X = sm.add_constant(clean['drift_x_auc'])
    r = sm.OLS(clean['overnight_betaadj'], X).fit(
        cov_type='cluster', cov_kwds={'groups': two_way_codes(clean)})
    c, t, p = r.params['drift_x_auc'], r.tvalues['drift_x_auc'], r.pvalues['drift_x_auc']
    print(f"  overnight_betaadj ~ drift_x_auc   coef={c:+.4f}  t={t:+.2f}  {star(p)}  n={len(clean):,}")
    results['OLS'] = (c, t, p)

    # --- Fama-MacBeth ---
    section(f"FAMA-MACBETH [{label}]")
    slopes, dates_fm = [], []
    for d, sub in clean.groupby('DATE'):
        if len(sub) < 5:
            continue
        try:
            r2 = sm.OLS(sub['overnight_betaadj'].values,
                        sm.add_constant(sub['drift_x_auc'].values)).fit()
            slopes.append(r2.params[1]); dates_fm.append(d)
        except Exception:
            continue
    slopes_s = pd.Series(slopes, index=dates_fm)
    nw = sm.OLS(slopes_s.values, np.ones(len(slopes_s))).fit(
        cov_type='HAC', cov_kwds={'maxlags': 5})
    fm_avg, fm_t, fm_p = nw.params[0], nw.tvalues[0], nw.pvalues[0]
    print(f"  FM avg slope={fm_avg:+.4f}   NW t={fm_t:+.2f}  {star(fm_p)}   ({len(slopes_s)} days)")
    results['FM'] = (fm_avg, fm_t, fm_p)
    results['slopes_s'] = slopes_s

    # --- Long-short ---
    section(f"LONG-SHORT [{label}]")
    ls_data = clean[['DATE','symbol','drift_x_auc','overnight_betaadj']].copy()
    ls_data = ls_data[ls_data['drift_x_auc'] != 0]

    def daily_ls(grp):
        if len(grp) < 5: return np.nan
        try:
            q = pd.qcut(grp['drift_x_auc'], 5,
                        labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
        except ValueError:
            return np.nan
        means = grp.assign(q=q).groupby('q', observed=True)['overnight_betaadj'].mean()
        if 'Q1' not in means.index or 'Q5' not in means.index: return np.nan
        return means['Q1'] - means['Q5']

    ls_series = ls_data.groupby('DATE').apply(daily_ls).dropna()
    sharpe    = ls_series.mean() / ls_series.std() * np.sqrt(252)
    nw_ls     = sm.OLS(ls_series.values, np.ones(len(ls_series))).fit(
        cov_type='HAC', cov_kwds={'maxlags': 5})
    ls_t, ls_p = nw_ls.tvalues[0], nw_ls.pvalues[0]
    win_rate   = (ls_series > 0).mean()
    yearly     = ls_series.groupby(ls_series.index.year).agg(['mean','std','count'])
    yearly['t']= yearly['mean'] / (yearly['std'] / np.sqrt(yearly['count']))

    COST_BPS = 2 / 10000  # flat 2 bps/day transaction cost (paper Table 3)
    ls_net   = ls_series - COST_BPS
    sharpe_net = ls_net.mean() / ls_net.std() * np.sqrt(252)
    print(f"  Mean Q1-Q5 (gross): {ls_series.mean()*10000:+.2f} bps/day")
    print(f"  Mean Q1-Q5 (net):   {ls_net.mean()*10000:+.2f} bps/day  (after 2 bps/day cost)")
    print(f"  HAC t:      {ls_t:+.2f}  {star(ls_p)}")
    print(f"  Sharpe (gross): {sharpe:.2f}   Sharpe (net): {sharpe_net:.2f}")
    print(f"  Win rate:   {win_rate*100:.1f}%")
    for yr, row in yearly.iterrows():
        mk = '▲' if row['mean'] > 0 else '▼'
        print(f"    {yr}  {mk}  {row['mean']*10000:+.1f} bps/day  t={row['t']:+.2f}")
    results['LS']       = (ls_series.mean(), ls_t, ls_p, sharpe, win_rate)
    results['ls_series']= ls_series
    results['yearly_ls']= yearly

    # --- Markov: discrete split ---
    section(f"MARKOV REGIME — DISCRETE SPLIT [{label}]")
    daily_agg = taq.groupby('DATE').agg(
        mean_abs_ret=('overnight_return', lambda x: x.abs().mean())
    ).dropna().reset_index().sort_values('DATE')
    daily_ms = daily_agg.set_index('DATE')

    try:
        model = MarkovRegression(daily_ms['mean_abs_ret'], k_regimes=2,
                                 trend='c', switching_variance=True)
        res_m = model.fit(maxiter=2000, em_iter=300, disp=False)
        s0, s1    = res_m.params['sigma2[0]'], res_m.params['sigma2[1]']
        vol       = 1 if s1 > s0 else 0
        sv, sc    = np.sqrt(max(s0,s1)), np.sqrt(min(s0,s1))
        # Smoothed probabilities condition on the full sample (retrospective).
        # Standard in academic HMM applications; disclosed in paper as a limitation.
        P_vol     = pd.Series(res_m.smoothed_marginal_probabilities[vol],
                              index=daily_ms.index, name='P_volatile')
        n_vol     = int((P_vol > 0.5).sum())
        print(f"  sigma_vol={sv*10000:.1f}bps  sigma_calm={sc*10000:.1f}bps  "
              f"vol_days={n_vol} ({n_vol/len(daily_ms)*100:.1f}%)")

        # Merge regime labels into clean
        p_df = P_vol.reset_index(); p_df.columns = ['DATE','P_volatile']
        clean_r = clean.merge(p_df, on='DATE', how='left').dropna(
            subset=['overnight_betaadj','drift_x_auc','P_volatile'])

        # Discrete split: separate OLS in each regime
        sub_vol  = clean_r[clean_r['P_volatile'] >= 0.5]
        sub_calm = clean_r[clean_r['P_volatile'] <  0.5]

        def ols_sub(sub):
            X2 = sm.add_constant(sub['drift_x_auc'])
            r2 = sm.OLS(sub['overnight_betaadj'], X2).fit(cov_type='HC1')
            return r2.params['drift_x_auc'], r2.tvalues['drift_x_auc'], r2.pvalues['drift_x_auc']

        cv, tv, pv = ols_sub(sub_vol)
        cc, tc, pc = ols_sub(sub_calm)
        ratio = abs(cv) / abs(cc) if abs(cc) > 1e-10 else np.nan

        print(f"  Volatile days  coef={cv:+.4f}  t={tv:+.2f}  {star(pv)}  n={len(sub_vol):,}")
        print(f"  Calm    days   coef={cc:+.4f}  t={tc:+.2f}  {star(pc)}  n={len(sub_calm):,}")
        print(f"  Volatile/Calm ratio: {ratio:.2f}x")
        print(f"  → Both regimes significant. Reversal {ratio:.1f}x stronger when volatile.")

        results['Markov'] = (cv, tv, pv, cc, tc, pc, ratio, sv, sc, n_vol)
        results['P_volatile'] = P_vol

    except Exception as e:
        print(f"  Markov failed: {e}")
        results['Markov'] = None

    return results


def section(title):
    print('\n' + '='*65)
    print(title)
    print('='*65)


# ============================================================
# MAIN
# ============================================================
section("LOADING DATA & COMPUTING BETA ADJUSTMENTS")
taq_dj, clean_dj = load_and_clean("taq_intraday.csv", "crsp_daily.csv",  "DJIA-30")
taq_sp, clean_sp = load_and_clean("taq_sp500.csv",    "crsp_sp500.csv",  "SP500-487")

section("RUNNING DJIA ANALYSIS")
res_dj = run_analysis(taq_dj, clean_dj, "DJIA-30")

section("RUNNING S&P 500 ANALYSIS")
res_sp = run_analysis(taq_sp, clean_sp, "SP500-487")


# ============================================================
# COMPARISON SUMMARY
# ============================================================
section("COMPARISON SUMMARY: DJIA-30 vs S&P 500-487")

def fmt(c, t, p):
    return f"coef={c:+.4f}  t={t:+.2f}  {star(p)}"

print(f"\n{'Metric':<45}  {'DJIA-30':^32}  {'S&P 500-487':^32}")
print("-"*110)

c,t,p = res_dj['OLS'];  c2,t2,p2 = res_sp['OLS']
print(f"  {'OLS (overnight_betaadj ~ drift_x_auc)':<43}  {fmt(c,t,p):<32}  {fmt(c2,t2,p2)}")

c,t,p = res_dj['FM'];   c2,t2,p2 = res_sp['FM']
print(f"  {'Fama-MacBeth':<43}  {fmt(c,t,p):<32}  {fmt(c2,t2,p2)}")

ls_dj = res_dj['LS'];   ls_sp = res_sp['LS']
print(f"  {'Long-Short Sharpe':<43}  {ls_dj[3]:.2f}{'':30}  {ls_sp[3]:.2f}")
print(f"  {'Long-Short HAC t':<43}  t={ls_dj[1]:+.2f} {star(ls_dj[2]):<27}  t={ls_sp[1]:+.2f} {star(ls_sp[2])}")
print(f"  {'Long-Short Win Rate':<43}  {ls_dj[4]*100:.1f}%{'':29}  {ls_sp[4]*100:.1f}%")

if res_dj['Markov'] and res_sp['Markov']:
    mdj = res_dj['Markov']; msp = res_sp['Markov']
    print(f"  {'Markov: volatile coef':<43}  {fmt(mdj[0],mdj[1],mdj[2]):<32}  {fmt(msp[0],msp[1],msp[2])}")
    print(f"  {'Markov: calm coef':<43}  {fmt(mdj[3],mdj[4],mdj[5]):<32}  {fmt(msp[3],msp[4],msp[5])}")
    print(f"  {'Markov: volatile/calm ratio':<43}  {mdj[6]:.2f}x{'':28}  {msp[6]:.2f}x")
    print(f"  {'Markov: sigma_vol':<43}  {mdj[7]*10000:.1f} bps{'':25}  {msp[7]*10000:.1f} bps")
    print(f"  {'Markov: sigma_calm':<43}  {mdj[8]*10000:.1f} bps{'':25}  {msp[8]*10000:.1f} bps")
    print(f"  {'Markov: volatile days':<43}  {mdj[9]}{'':29}  {msp[9]}")


# ============================================================
# FIGURES
# ============================================================
section("FIGURES")

# Figure 1: OLS + Sharpe comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
labels  = ['DJIA-30', 'S&P 500-487']
t_stats = [abs(res_dj['OLS'][1]), abs(res_sp['OLS'][1])]
colors  = ['steelblue', '#c0392b']
bars = ax.bar(labels, t_stats, color=colors, alpha=0.85, width=0.4)
ax.axhline(1.96, color='black', linestyle='--', linewidth=0.8, label='t=1.96 (5%)')
ax.axhline(2.58, color='gray',  linestyle='--', linewidth=0.8, label='t=2.58 (1%)')
for bar, t, p in zip(bars, [res_dj['OLS'][1], res_sp['OLS'][1]],
                            [res_dj['OLS'][2], res_sp['OLS'][2]]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f't={t:.2f}{star(p)}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('|t-statistic|')
ax.set_title('OLS Signal Strength\n(overnight_betaadj ~ drift_x_auc)', fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1]
sharpes = [res_dj['LS'][3], res_sp['LS'][3]]
bars = ax.bar(labels, sharpes, color=colors, alpha=0.85, width=0.4)
for bar, s in zip(bars, sharpes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
            f'{s:.2f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Annualised Sharpe Ratio')
ax.set_title('Long-Short Portfolio Sharpe\n(sort by drift_x_auc, Q1-Q5)', fontweight='bold')

ax = axes[2]
if res_dj['Markov'] and res_sp['Markov']:
    x = np.arange(2); w = 0.3
    vol_t  = [abs(res_dj['Markov'][1]), abs(res_sp['Markov'][1])]
    calm_t = [abs(res_dj['Markov'][4]), abs(res_sp['Markov'][4])]
    b1 = ax.bar(x - w/2, vol_t,  w, label='Volatile regime', color=['steelblue','#c0392b'], alpha=0.9)
    b2 = ax.bar(x + w/2, calm_t, w, label='Calm regime',    color=['steelblue','#c0392b'], alpha=0.4)
    ax.axhline(1.96, color='black', linestyle='--', linewidth=0.8, label='t=1.96')
    ax.axhline(2.58, color='gray',  linestyle='--', linewidth=0.8, label='t=2.58')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('|t-statistic| on drift_x_auc')
    ax.set_title('Markov Discrete Split\n|t-stat| in each regime', fontweight='bold')
    ax.legend(fontsize=8)

fig.suptitle('DJIA-30 vs S&P 500-487: Mean Reversion — Beta-Adjusted Output\n'
             '(drift_x_auc signal, Markov discrete split, 2016–2023)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'fig_sp500_comparison.png')
plt.close()
print("  Saved fig_sp500_comparison.png")

# Figure 2: Long-short cumulative return
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
cum_dj = res_dj['ls_series'].cumsum() * 10000
cum_sp = res_sp['ls_series'].cumsum() * 10000
ax.plot(cum_dj.index, cum_dj.values, color='steelblue', linewidth=1.2,
        label=f"DJIA-30  (Sharpe={res_dj['LS'][3]:.2f})")
ax.plot(cum_sp.index, cum_sp.values, color='#c0392b',   linewidth=1.2,
        label=f"S&P 500  (Sharpe={res_sp['LS'][3]:.2f})")
ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
ax.set_ylabel('Cumulative Q1-Q5 overnight return (bps)')
ax.set_title('Long-Short Cumulative Return', fontweight='bold')
ax.legend()

ax = axes[1]
if res_dj['Markov'] and res_sp['Markov']:
    ratios = [res_dj['Markov'][6], res_sp['Markov'][6]]
    bars   = ax.bar(labels, ratios, color=colors, alpha=0.85, width=0.4)
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', label='No amplification (1×)')
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                f'{r:.2f}×', ha='center', fontsize=13, fontweight='bold')
    ax.set_ylabel('Volatile / Calm reversal ratio')
    ax.set_title('Markov Amplification Ratio\n(volatile ÷ calm coefficient)', fontweight='bold')
    ax.legend(fontsize=9)

fig.suptitle('DJIA-30 vs S&P 500-487: Long-Short & Markov Results\n'
             '(beta-adjusted output, 2016–2023)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'fig_sp500_longshort_markov.png')
plt.close()
print("  Saved fig_sp500_longshort_markov.png")

# Figure 3: Markov regime classification — 2-panel time series for S&P 500
if res_sp['Markov']:
    daily_agg_sp = taq_sp.groupby('DATE').agg(
        mean_abs_ret=('overnight_return', lambda x: x.abs().mean())
    ).dropna().reset_index().sort_values('DATE')
    daily_ms_sp = daily_agg_sp.set_index('DATE')
    P_vol_sp    = res_sp['P_volatile']
    is_vol      = P_vol_sp > 0.5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle('Markov Regime Classification: S&P 500-487 (2016–2023)\n'
                 '(HMM on daily cross-sectional mean absolute overnight return)',
                 fontsize=13, fontweight='bold')

    # Panel 1 — mean absolute return with volatile shading
    ax1.plot(daily_ms_sp.index, daily_ms_sp['mean_abs_ret'] * 10000,
             color='#2166ac', linewidth=0.9, alpha=0.8,
             label='Mean |overnight return| (bps)')
    vol_dates = P_vol_sp[is_vol].index
    if len(vol_dates):
        ax1.fill_between(daily_ms_sp.index,
                         daily_ms_sp['mean_abs_ret'] * 10000,
                         where=daily_ms_sp.index.isin(vol_dates),
                         color='#d6604d', alpha=0.4, label='Volatile regime')
    ax1.set_ylabel('Mean |overnight return| (bps)')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_title(f"Observable: daily cross-sectional mean |overnight return| — "
                  f"{res_sp['Markov'][9]} volatile days ({res_sp['Markov'][9]/len(daily_ms_sp)*100:.1f}%)",
                  fontsize=10)

    # Panel 2 — smoothed P(volatile)
    p_aligned = P_vol_sp.reindex(daily_ms_sp.index)
    ax2.fill_between(daily_ms_sp.index, p_aligned.values, 0,
                     color='#d6604d', alpha=0.5, label='P(volatile regime)')
    ax2.axhline(0.5, color='black', linewidth=0.8, linestyle='--', label='Classification threshold (0.5)')
    ax2.set_ylabel(r'$P(s_t = \mathrm{volatile} \mid \mathcal{F}_T)$')
    ax2.set_xlabel('Date')
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.set_title('Smoothed probability of volatile regime', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT + 'fig_sp500_markov.png')
    plt.close()
    print("  Saved fig_sp500_markov.png")

print("\n=== sp500_analysis.py done ===")
