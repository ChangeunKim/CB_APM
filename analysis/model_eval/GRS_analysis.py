"""
Refactored from analysis/GRS.ipynb (kept as the source notebook; this module
is the maintained, runnable version). Gibbons-Ross-Shanken (GRS) test for
pricing errors (alphas) of a linear factor model, run three ways:

1. `compare_model_vs_ff_factors` -- CB-APM's 9 analyst-concept long-short
   decile factors (VW, built from LIVE inference through
   `checkpoints/<horizon>_<lambda>/`, i.e. forward passes through already
   -trained models -- not retraining; see `analysis.util.get_Xy_cbapm`)
   against three sets of standard test assets (25 Portfolios 5x5, 25
   Portfolios ME x Prior, 30 Industry Portfolios), compared to CAPM / FF3 /
   Carhart4 / FF5 / FF6.
2. `compare_portfolio_vs_ff_factors` -- CB-APM's own predicted-return decile
   portfolios (from the frozen `final_results/<h>_<lambda>.pickle`, no live
   inference) across the lambda grid, as test assets against the same
   standard factor models.
3. `compare_concept_portfolio_vs_ff_factors` -- decile portfolios sorted on
   each of CB-APM's 9 forecasted concept variables individually, same
   standard factor models.

NOTE on VW weighting in part 1: the source notebook value-weights each
decile leg by RAW `Size` (log market equity from data/input_<h>.csv), not
`exp(Size)` (true market equity) -- see `analysis.util.portfolio_weights`'s
own docstring on this distinction. That is reproduced here unchanged (not
"fixed") to match the notebook's cached GRS statistics exactly; see
analysis/outputs/DEVIATIONS.md for the numeric verification.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist

from analysis.util import REPO_ROOT, get_Xy_cbapm, load_final_result_pickle, CONSENSUS_VARS
from analysis.model_eval.hjd import load_ff_csv
from config import get_config

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'GRS_analysis'

BENCHMARK_MODELS = ['CAPM', 'FF3', 'Carhart4', 'FF5', 'FF6']

# Readable labels for the 9 CONSENSUS_VARS, matching the notebook's manual
# `cons.columns` rename in `compare_concept_portfolio_vs_ff_factors`.
CONCEPT_LABELS = {
    'AnalystRevision': 'EPS forecast revision',
    'ChangeInRecommendation': 'Change in recommendation',
    'ChForecastAccrual': 'Change in Forecast and Accrual',
    'EarningsForecastDisparity': 'Long-vs-short EPS forecasts',
    'FEPS': 'Analyst earnings per share',
    'ForecastDispersion': 'EPS Forecast Dispersion',
    'REV6': 'Earnings forecast revisions',
    'AnalystValue': 'Analyst Value',
    'AOP': 'Analyst Optimism',
}


def grs_test(returns: pd.DataFrame, factors: pd.DataFrame, rf=None, asset_names=None) -> dict:
    """
    GRS test for pricing errors (alphas) of a linear factor model.

    returns : T x N asset (excess-of-rf) return panel.
    factors : T x K factor return panel.
    rf : risk-free rate (Series, scalar, or None -> 0), aligned to `returns`.
    """
    R = returns.copy()
    F = factors.copy()
    if rf is None:
        rf_vec = np.zeros(len(R))
    elif np.isscalar(rf):
        rf_vec = np.full(len(R), float(rf))
    else:
        rf_vec = rf.loc[R.index].values

    F = F.loc[R.index]

    df_all = pd.concat([R, F, pd.Series(rf_vec, index=R.index, name='_rf_')], axis=1).dropna()
    T = len(df_all)
    if T == 0:
        raise ValueError('No overlapping, non-missing observations after alignment.')

    R = df_all[R.columns]
    F = df_all[F.columns]
    rf_vec = df_all['_rf_'].values

    Rex = R.values - rf_vec[:, None]
    X = np.column_stack([np.ones(T), F.values])
    N = Rex.shape[1]
    K = F.shape[1]

    if T <= (K + 1):
        raise ValueError(f'Insufficient T relative to K: need T > K+1, got T={T}, K={K}')
    if T <= (N + K):
        raise ValueError(f'Insufficient T relative to N and K: need T > N+K, got T={T}, N={N}, K={K}')

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    beta_hat = XtX_inv @ (X.T @ Rex)
    alpha = beta_hat[0, :]
    resid = Rex - X @ beta_hat

    S = (resid.T @ resid) / (T - (K + 1))
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    mu_f = F.mean().values.reshape(-1, 1)
    Sigma_f = np.atleast_2d(np.cov(F.values, rowvar=False, ddof=1))
    try:
        Sigma_f_inv = np.linalg.inv(Sigma_f)
    except np.linalg.LinAlgError:
        Sigma_f_inv = np.linalg.pinv(Sigma_f)

    alpha_vec = alpha.reshape(-1, 1)
    term1 = (alpha_vec.T @ S_inv @ alpha_vec).item()
    term2 = (1 + (mu_f.T @ Sigma_f_inv @ mu_f)).item()
    df_num = N
    df_den = T - N - K
    if df_den <= 0:
        raise ValueError(f'Nonpositive denominator dof: T-N-K = {df_den}. Reduce N or K, or extend T.')
    F_stat = (df_den / df_num) * (term1 / term2)
    pval = 1 - f_dist.cdf(F_stat, df_num, df_den)

    alpha_mean_abs = float(np.mean(np.abs(alpha)))
    alpha_rms = float(np.sqrt(np.mean(alpha ** 2)))
    alpha_annual = (1 + alpha) ** 12 - 1
    alpha_mean_abs_annual = float(np.mean(np.abs(alpha_annual)))
    alpha_rms_annual = float(np.sqrt(np.mean(alpha_annual ** 2)))

    if asset_names is None:
        asset_names = list(R.columns)
    alpha_df = pd.DataFrame({'alpha_monthly': alpha, 'alpha_annual': alpha_annual}, index=asset_names)

    return {
        'F': F_stat, 'pval': pval,
        'alpha_monthly': alpha, 'alpha_annual': alpha_annual,
        'alpha_mean_abs_monthly': alpha_mean_abs, 'alpha_mean_abs_annual': alpha_mean_abs_annual,
        'alpha_rms_monthly': alpha_rms, 'alpha_rms_annual': alpha_rms_annual,
        'alpha_df': alpha_df, 'S': S, 'mu_f': mu_f.flatten(), 'Sigma_f': Sigma_f,
        'T': T, 'N': N, 'K': K,
    }


def restrict_period(df, start_dt, end_dt):
    df = df.sort_index()
    return df.loc[(df.index >= start_dt) & (df.index <= end_dt)]


def load_standard_factors(data_dir=None):
    """
    CAPM/FF3/Carhart4/FF5/FF6 factor + rf panels, matching the notebook's
    `_load_standard_factors`: CAPM/FF3/Carhart4 use the FF3 file's own RF;
    FF5/FF6 use the FF5 file's own RF (falls back to FF3's RF if absent).
    """
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'

    ff3 = load_ff_csv(data_dir / 'F-F_Research_Data_Factors.csv')
    rf3 = ff3['RF']
    ff5 = load_ff_csv(data_dir / 'F-F_Research_Data_5_Factors_2x3.csv')
    rf5 = ff5['RF'] if 'RF' in ff5.columns else rf3.reindex(ff5.index)
    mom = load_ff_csv(data_dir / 'F-F_Momentum_Factor.csv').rename(columns={'Mom': 'UMD'})

    out = {
        'CAPM': {'factors': ff3[['Mkt-RF']].copy(), 'rf': rf3.copy()},
        'FF3': {'factors': ff3[['Mkt-RF', 'SMB', 'HML']].copy(), 'rf': rf3.copy()},
        'FF5': {'factors': ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].copy(), 'rf': rf5.copy()},
    }

    common_c4 = out['FF3']['factors'].index.intersection(mom.index)
    out['Carhart4'] = {
        'factors': pd.concat([out['FF3']['factors'].loc[common_c4], mom[['UMD']].loc[common_c4]], axis=1),
        'rf': rf3.loc[common_c4],
    }
    common_ff6 = out['FF5']['factors'].index.intersection(mom.index)
    out['FF6'] = {
        'factors': pd.concat([out['FF5']['factors'].loc[common_ff6], mom[['UMD']].loc[common_ff6]], axis=1),
        'rf': rf5.loc[common_ff6],
    }
    return out


def _pretty_row(name, res):
    return {
        'Factor Model': name,
        'GRS F-statistic': res['F'],
        'p-value': res['pval'],
        'Mean Abs Alpha (Monthly)': res['alpha_mean_abs_monthly'],
        'Mean Abs Alpha (Annual)': res['alpha_mean_abs_annual'],
        'RMS Alpha (Monthly)': res['alpha_rms_monthly'],
        'RMS Alpha (Annual)': res['alpha_rms_annual'],
        'Number of Factors (K)': res['K'],
        'Sample Size (T)': res['T'],
        'Number of Assets (N)': res['N'],
    }


def build_model_factors(horizon, weight_lambda, train_date='2020-01-01',
                         embedding_method='autoencoder', checkpoints_dir=None,
                         data_dir=None, device=None):
    """
    CB-APM factor-mimicking portfolios (VW long-short deciles, one per
    analyst concept) for the GRS test, via live inference through
    `get_Xy_cbapm`. Size (market equity proxy) from input_<horizon>.csv is
    used for VW weights; realized 1-month returns from target_1month.csv.
    """
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_df = pd.read_csv(data_dir / f'input_{horizon}.csv')
    target_df = pd.read_csv(data_dir / 'target_1month.csv')
    input_df['date'] = pd.to_datetime(input_df['date'])
    target_df['date'] = pd.to_datetime(target_df['date'])
    size_df = input_df[['date', 'permno', 'Size']].copy()

    signal_info = pd.read_csv(data_dir / 'info' / 'SignalDoc.csv')
    info = signal_info[signal_info['Acronym'].isin(input_df.columns)]
    config = get_config(weight_lambda)

    X_cbapm_with_index, _ = get_Xy_cbapm(
        train_date=train_date, input=input_df, target=target_df, info=info,
        config=config, device=device, horizon=horizon, weight_lambda=weight_lambda,
        embedding_method=embedding_method, checkpoints_dir=checkpoints_dir,
    )

    df = (
        X_cbapm_with_index
        .merge(target_df[['date', 'permno', 'Return']], on=['date', 'permno'], how='inner')
        .merge(size_df, on=['date', 'permno'], how='inner')
    )
    df = df[df['date'] < pd.to_datetime(train_date)].copy()
    df = df.sort_values(['date', 'permno'])

    concept_cols = [c for c in df.columns if c not in ('date', 'permno', 'Return', 'Size')]

    factor_series_list = []
    for concept in concept_cols:
        df['rank'] = df.groupby('date')[concept].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))
        long_leg = df[df['rank'] == 9]
        short_leg = df[df['rank'] == 0]
        long_ret = long_leg.groupby('date').apply(lambda x: (x['Size'] * x['Return']).sum() / x['Size'].sum())
        short_ret = short_leg.groupby('date').apply(lambda x: (x['Size'] * x['Return']).sum() / x['Size'].sum())
        factor_series_list.append((long_ret - short_ret).rename(concept))

    factors_df = pd.concat(factor_series_list, axis=1).sort_index()
    print(f'[INFO] CB-APM long-short factors constructed. shape={factors_df.shape}, K={len(concept_cols)}')
    return factors_df


def compare_model_vs_ff_factors(horizon, weight_lambdas, portfolio_returns,
                                 embedding_method='autoencoder', train_date='2020-01-01',
                                 start_date='1994-01-01', end_date='2020-01-01',
                                 checkpoints_dir=None, data_dir=None, device=None):
    """Model factors (one or many lambdas) vs standard factor models, same test assets."""
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    R_all = restrict_period(portfolio_returns, start_dt, end_dt)

    rows = []
    details = {}
    for lam in weight_lambdas:
        label = f'Model Factor (λ={lam:g})'
        print(f'  [Model] Running GRS for {label}')
        factors_df = build_model_factors(
            horizon, lam, train_date=train_date, embedding_method=embedding_method,
            checkpoints_dir=checkpoints_dir, data_dir=data_dir, device=device,
        ).sort_index()

        common_start = max(factors_df.index.min(), R_all.index.min(), start_dt)
        common_end = min(factors_df.index.max(), R_all.index.max(), end_dt)
        F = factors_df.loc[(factors_df.index >= common_start) & (factors_df.index <= common_end)]
        R = R_all.loc[(R_all.index >= common_start) & (R_all.index <= common_end)]

        res = grs_test(R, F, rf=None)
        rows.append(_pretty_row(label, res))
        details[label] = res

    std = load_standard_factors(data_dir=data_dir)
    for name in BENCHMARK_MODELS:
        if name not in std:
            continue
        fac = restrict_period(std[name]['factors'], start_dt, end_dt)
        rf = restrict_period(std[name]['rf'], start_dt, end_dt)
        common_idx = R_all.index.intersection(fac.index).intersection(rf.index)
        if len(common_idx) < 24:
            print(f'  - [Skip] {name}: too few overlapping observations (T={len(common_idx)}).')
            continue
        res = grs_test(R_all.loc[common_idx], fac.loc[common_idx], rf.loc[common_idx])
        rows.append(_pretty_row(name, res))
        details[name] = res

    return pd.DataFrame(rows), details


def decile_portfolio_from_scores(df, score_col, return_col='actual', date_col='date',
                                  n_bins=10, min_names=10):
    """Cross-sectional decile sort each date on `score_col`; equal-weighted mean of
    `return_col` per bin. Shared by both `compare_portfolio_vs_ff_factors` (sorts
    on CB-APM's predicted return) and `compare_concept_portfolio_vs_ff_factors`
    (sorts on each forecasted concept variable) -- the notebook duplicated this
    logic almost verbatim across both cells."""
    rows = []
    for date, g in df.groupby(date_col):
        if len(g) < min_names:
            continue
        g = g.copy()
        try:
            g['decile'] = pd.qcut(g[score_col], n_bins, labels=False, duplicates='drop') + 1
        except ValueError:
            continue
        means = g.groupby('decile')[return_col].mean()
        rows.append(pd.DataFrame({date_col: date, **means.to_dict()}, index=[0]))

    if not rows:
        return pd.DataFrame()
    port = pd.concat(rows, ignore_index=True).sort_values(date_col).set_index(date_col)
    port.columns = [f'Decile{int(c)}' for c in port.columns]
    port.index = pd.to_datetime(port.index)
    return port


def load_realized_1month_returns(data_dir=None):
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
    target = pd.read_csv(data_dir / 'target_1month.csv')
    target['date'] = pd.to_datetime(target['date'])
    return target[['date', 'permno', 'Return']].rename(columns={'Return': 'actual'})


def compare_portfolio_vs_ff_factors(horizon='12month', weight_lambdas=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                     data_dir=None, results_dir=None,
                                     start_date='1994-01-01', end_date='2020-01-01'):
    """GRS tests for CB-APM predicted-return decile portfolios (from the frozen
    final_results pickles) vs standard factor models, across the lambda grid."""
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
    results_dir = Path(results_dir) if results_dir else REPO_ROOT / 'final_results'
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)

    std = load_standard_factors(data_dir=data_dir)
    target = load_realized_1month_returns(data_dir=data_dir)

    summary_rows = []
    details = {}
    for lam in weight_lambdas:
        pickle_path = results_dir / f'{horizon}_{lam}.pickle'
        if not pickle_path.exists():
            print(f'[Skip] {pickle_path} not found.')
            continue

        raw = load_final_result_pickle(horizon, lam, results_dir=results_dir)
        forecast = raw['forecast_target'].rename(columns={'Return': 'forecast'})
        merged = forecast.merge(target, on=['date', 'permno'], how='inner').dropna(subset=['forecast', 'actual'])

        port = decile_portfolio_from_scores(merged, score_col='forecast', return_col='actual')
        port = restrict_period(port, start_dt, end_dt)
        if port.empty:
            continue

        for name in BENCHMARK_MODELS:
            if name not in std:
                continue
            fac = restrict_period(std[name]['factors'], start_dt, end_dt)
            rf = restrict_period(std[name]['rf'], start_dt, end_dt)
            common_idx = port.index.intersection(fac.index).intersection(rf.index)
            if len(common_idx) < 24:
                print(f'  - [Skip] {name} (λ={lam}): T={len(common_idx)} too short.')
                continue
            res = grs_test(port.loc[common_idx], fac.loc[common_idx], rf.loc[common_idx])
            label = f'{name} (λ={lam})'
            summary_rows.append(_pretty_row(label, res))
            details[label] = res

    return pd.DataFrame(summary_rows), details


def compare_concept_portfolio_vs_ff_factors(horizon='12month', weight_lambdas=(1.0,), data_dir=None,
                                             results_dir=None, start_date='1994-01-01', end_date='2020-01-01'):
    """GRS tests for decile portfolios formed on each of CB-APM's 9 forecasted
    concept variables individually (from the frozen final_results pickles) vs
    standard factor models."""
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
    results_dir = Path(results_dir) if results_dir else REPO_ROOT / 'final_results'
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)

    std = load_standard_factors(data_dir=data_dir)
    target = load_realized_1month_returns(data_dir=data_dir)

    summary_rows = []
    details = {}
    for lam in weight_lambdas:
        pickle_path = results_dir / f'{horizon}_{lam}.pickle'
        if not pickle_path.exists():
            print(f'[Skip] {pickle_path} not found.')
            continue

        raw = load_final_result_pickle(horizon, lam, results_dir=results_dir)
        cons = raw['forecast_concept']

        for concept_var in CONSENSUS_VARS:
            cons_sub = cons[['date', 'permno', concept_var]].rename(columns={concept_var: 'concept'})
            merged = cons_sub.merge(target, on=['date', 'permno'], how='inner').dropna(subset=['concept', 'actual'])
            if merged['concept'].nunique() < 10:
                continue

            port = decile_portfolio_from_scores(merged, score_col='concept', return_col='actual')
            port = restrict_period(port, start_dt, end_dt)
            if port.empty:
                continue

            for name in BENCHMARK_MODELS:
                if name not in std:
                    continue
                fac = restrict_period(std[name]['factors'], start_dt, end_dt)
                rf = restrict_period(std[name]['rf'], start_dt, end_dt)
                common_idx = port.index.intersection(fac.index).intersection(rf.index)
                if len(common_idx) < 24:
                    continue
                res = grs_test(port.loc[common_idx], fac.loc[common_idx], rf.loc[common_idx])
                row = _pretty_row(name, res)
                row['λ'] = lam
                row['Concept'] = CONCEPT_LABELS.get(concept_var, concept_var)
                summary_rows.append(row)
                details[f'{name} (λ={lam}, {concept_var})'] = res

    df = pd.DataFrame(summary_rows)
    if not df.empty:
        ordered_cols = ['Concept', 'λ'] + [c for c in df.columns if c not in ('Concept', 'λ')]
        df = df[ordered_cols]
    return df, details


def _plot_grs_by_lambda(port_df, out_dir, horizon):
    df = port_df.copy()
    df['_model'] = df['Factor Model'].str.extract(r'^(\w+) \(')
    df['_lambda'] = df['Factor Model'].str.extract(r'λ=([\d.]+)\)').astype(float)

    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12,
                          'axes.linewidth': 1.2})
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#937860']
    markers = ['o', 's', '^', 'D', 'v']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(BENCHMARK_MODELS):
        sub = df[df['_model'] == name].sort_values('_lambda')
        if sub.empty:
            continue
        ax.plot(sub['_lambda'], sub['GRS F-statistic'], marker=markers[i % len(markers)],
                markersize=6, label=name, color=colors[i % len(colors)], linewidth=2, alpha=0.9)

    ax.set_xlabel('Hyperparameter (lambda)', fontsize=11)
    ax.set_ylabel('GRS F-statistic', fontsize=11)
    ax.grid(True, axis='both', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, facecolor='white', framealpha=0.8, loc='best', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f'{horizon}_grs_fstat_by_lambda.png', dpi=150)
    plt.close(fig)


def run(horizon='12month', train_date='2020-01-01', embedding_method='autoencoder',
        checkpoints_dir=None, data_dir=None, results_dir=None, out_dir=None,
        start_date='1994-01-01', end_date='2020-01-01', device=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
    results_dir = Path(results_dir) if results_dir else REPO_ROOT / 'final_results'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    portfolio_files = {
        '25_Portfolios_5x5': data_dir / '25_Portfolios_5x5.csv',
        '25_Portfolios_ME_Prior_12_2': data_dir / '25_Portfolios_ME_Prior_12_2.csv',
        '30_Industry_Portfolios': data_dir / '30_Industry_Portfolios.csv',
    }
    model_lambdas = [0.1, 0.5, 1.0]

    print('=' * 70)
    print('Part 1: CB-APM model factors (live inference) vs standard factor models')
    print('=' * 70)
    model_vs_ff_results = {}
    for port_name, path in portfolio_files.items():
        print(f'\nTest assets: {port_name}')
        R = load_ff_csv(path)
        comp_df, _ = compare_model_vs_ff_factors(
            horizon=horizon, weight_lambdas=model_lambdas, portfolio_returns=R,
            embedding_method=embedding_method, train_date=train_date,
            start_date=start_date, end_date=end_date,
            checkpoints_dir=checkpoints_dir, data_dir=data_dir, device=device,
        )
        print(comp_df.to_string(index=False, float_format='%.6f'))
        comp_df.to_csv(out_dir / f'{horizon}_model_vs_ff_{port_name}.csv', index=False)
        model_vs_ff_results[port_name] = comp_df

    print('\n' + '=' * 70)
    print('Part 2: CB-APM predicted decile portfolios vs standard factor models')
    print('=' * 70)
    port_lambdas = [round(x * 0.2, 1) for x in range(6)]
    port_df, _ = compare_portfolio_vs_ff_factors(
        horizon=horizon, weight_lambdas=port_lambdas, data_dir=data_dir,
        results_dir=results_dir, start_date=start_date, end_date=end_date,
    )
    print(port_df.to_string(index=False, float_format='%.6f'))
    port_df.to_csv(out_dir / f'{horizon}_portfolio_vs_ff.csv', index=False)

    print('\n' + '=' * 70)
    print('Part 3: CB-APM concept-sorted decile portfolios vs standard factor models')
    print('=' * 70)
    concept_df, _ = compare_concept_portfolio_vs_ff_factors(
        horizon=horizon, weight_lambdas=[1.0], data_dir=data_dir,
        results_dir=results_dir, start_date=start_date, end_date=end_date,
    )
    print(concept_df.to_string(index=False, float_format='%.6f'))
    concept_df.to_csv(out_dir / f'{horizon}_concept_portfolio_vs_ff.csv', index=False)

    if not port_df.empty:
        _plot_grs_by_lambda(port_df, out_dir, horizon)

    print('\nAll analyses complete.')
    return {
        'model_vs_ff': model_vs_ff_results,
        'portfolio_vs_ff': port_df,
        'concept_portfolio_vs_ff': concept_df,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    parser.add_argument('--train_date', default='2020-01-01')
    args = parser.parse_args()
    run(horizon=args.horizon, train_date=args.train_date)
