"""
Refactored from analysis/portfolio.ipynb (kept as the source notebook; this
module is the maintained, runnable version). Portfolio-performance analysis
of the CB framework's predicted returns across the lambda grid:

  - summarize_sorted_portfolios: realized-return summary tables for single-
    sort deciles and double-sort (consensus quintile x predicted-return
    quintile) portfolios, with H-L spreads (feeds the LaTeX table generators
    in analysis/util.py).
  - plot_portfolio_performance: cumulative-return plot + per-lambda portfolio
    construction (single_sort / double_sort / screening strategies), with
    optional proportional transaction-cost adjustment.
  - calculate_portfolio_metrics: mean/std/Sharpe/max-1M-loss/max-DD/turnover
    summary table from a plot_portfolio_performance() result.
"""
import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from analysis.util import (
    REPO_ROOT, MAX_DD, turnover, generate_single_sort_latex_table, generate_double_sort_latex_table,
)

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'portfolio_analysis'

CONSENSUS_COLUMN_NAMES = [
    'date', 'permno', 'EPS forecast revision', 'Change in recommendation',
    'Change in Forecast and Accrual', 'Long-vs-short EPS forecasts',
    'Analyst earnings per share', 'EPS Forecast Dispersion',
    'Earnings forecast revisions', 'Analyst Value', 'Analyst Optimism',
]


def summarize_sorted_portfolios(horizon='12month', strategy='single_sort', strategy_params=None,
                                 stock_info=None, results_dir=None):
    """
    Summarize realized returns of sorted portfolios with H-L spreads (%).
      single_sort: deciles by predicted return -> realized mean by decile + H-L (10th-1st).
      double_sort: 5x5 (consensus quintile x predicted-return quintile) -> realized
                   mean by cell + row-wise and column-wise H-L.
    """
    results_dir = Path(results_dir) if results_dir else REPO_ROOT / 'results'
    summary_dict = {}
    target_lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if strategy == 'single_sort':
        target_lambdas.append(0.0)

    for file in os.listdir(results_dir):
        if not file.startswith(horizon) or not file.endswith('.pickle'):
            continue
        lam = float(file.split(f'{horizon}_')[1].split('.pickle')[0])
        if lam not in target_lambdas:
            continue

        with open(results_dir / file, 'rb') as f:
            output = pickle.load(f)

        forecast = output['forecast_target']
        forecast.columns = ['date', 'permno', 'forecast']
        forecast['date'] = pd.to_datetime(forecast['date'])

        if horizon != '1month':
            one_path = results_dir / f'1month_{lam}.pickle'
            if one_path.exists():
                with open(one_path, 'rb') as f1:
                    actual = pickle.load(f1)['actual_target']
            else:
                actual = output['actual_target']
        else:
            actual = output['actual_target']

        actual.columns = ['date', 'permno', 'actual']
        actual['date'] = pd.to_datetime(actual['date'])

        merged = pd.merge(actual, forecast, on=['date', 'permno']).dropna()
        if stock_info is not None:
            merged = pd.merge(merged, stock_info, on=['permno'], how='left')

        if strategy == 'double_sort':
            cons = output['forecast_concept']
            cons.columns = CONSENSUS_COLUMN_NAMES
            cons_var = strategy_params.get('cons_var', 'Analyst earnings per share')
            merged = pd.merge(merged, cons[['date', 'permno', cons_var]], on=['date', 'permno'], how='left')

        real_means = []
        for _, group in merged.groupby('date'):
            if strategy == 'single_sort':
                try:
                    group = group.copy()
                    group['pred_rank'] = pd.qcut(group['forecast'], 10, labels=False, duplicates='drop') + 1
                except ValueError:
                    continue
                real_means.append(group.groupby('pred_rank')['actual'].mean())

            elif strategy == 'double_sort':
                cons_var = strategy_params.get('cons_var', 'Analyst earnings per share')
                group = group.copy()
                try:
                    group['cons_rank'] = pd.qcut(group[cons_var], 5, labels=False, duplicates='drop') + 1
                except ValueError:
                    continue

                group['pred_rank'] = np.nan
                for c in group['cons_rank'].dropna().unique():
                    sub_idx = group[group['cons_rank'] == c].index
                    sub = group.loc[sub_idx]
                    try:
                        group.loc[sub_idx, 'pred_rank'] = pd.qcut(sub['forecast'], 5, labels=False, duplicates='drop') + 1
                    except ValueError:
                        continue

                temp = group.dropna(subset=['cons_rank', 'pred_rank'])
                real_means.append(temp.groupby(['cons_rank', 'pred_rank'])['actual'].mean().unstack())

        if strategy == 'single_sort':
            real_df = pd.concat(real_means, axis=1).mean(axis=1) * 100
            real_df.loc['H-L'] = real_df.loc[real_df.index.max()] - real_df.loc[real_df.index.min()]
            real_df.name = f'{lam:.1f}'
            summary_dict[lam] = real_df
        else:
            full_idx = pd.Index(range(1, 6), name='cons_rank')
            full_cols = pd.Index(range(1, 6), name='pred_rank')
            matrices = [r.reindex(index=full_idx, columns=full_cols) for r in real_means]
            real_matrix = pd.concat(matrices).groupby(level=0).mean() * 100

            real_matrix['H-L'] = real_matrix[5] - real_matrix[1]
            hl_row = real_matrix.loc[5] - real_matrix.loc[1]
            hl_row.name = 'H-L'
            real_matrix = pd.concat([real_matrix, pd.DataFrame([hl_row])])
            real_matrix.index.name = 'Consensus Quintile'
            summary_dict[lam] = real_matrix

    if strategy == 'single_sort':
        result = pd.concat(summary_dict.values(), axis=1)
        result.columns = [f'{lam:.1f}' for lam in sorted(summary_dict.keys())]
        result.index.name = 'Decile'
        return result
    return summary_dict


def generate_portfolio(curr_returns, strategy='single_sort', params=None):
    """Long/short permno positions for one date snapshot, per strategy."""
    params = params or {}

    if strategy == 'single_sort':
        upper_quantile = params.get('upper_quantile', 0.9)
        lower_quantile = params.get('lower_quantile', 0.1)
        long_cut = curr_returns.quantile(upper_quantile)['forecast']
        short_cut = curr_returns.quantile(lower_quantile)['forecast']
        long_mask = curr_returns['forecast'] >= long_cut
        short_mask = curr_returns['forecast'] <= short_cut

    elif strategy == 'double_sort':
        upper_quantile_cons = params.get('upper_quantile_cons', 0.75)
        lower_quantile_cons = params.get('lower_quantile_cons', 0.25)
        upper_quantile_ret = params.get('upper_quantile_ret', 0.75)
        lower_quantile_ret = params.get('lower_quantile_ret', 0.25)
        sorting_sign = params.get('sorting_sign', 'positive')
        cons_var = params.get('cons_var', 'Analyst earnings per share')

        if sorting_sign == 'positive':
            long_con = curr_returns.quantile(upper_quantile_cons)[cons_var]
            short_con = curr_returns.quantile(lower_quantile_cons)[cons_var]
            long_mask_con = curr_returns[cons_var] >= long_con
            short_mask_con = curr_returns[cons_var] <= short_con
        elif sorting_sign == 'negative':
            long_con = curr_returns.quantile(lower_quantile_cons)[cons_var]
            short_con = curr_returns.quantile(upper_quantile_cons)[cons_var]
            long_mask_con = curr_returns[cons_var] <= long_con
            short_mask_con = curr_returns[cons_var] >= short_con
        else:
            raise ValueError(f'Unknown sorting_sign: {sorting_sign}')

        long_posit_con = curr_returns[long_mask_con]['permno']
        short_posit_con = curr_returns[short_mask_con]['permno']

        long_fore = curr_returns[curr_returns['permno'].isin(long_posit_con)].quantile(upper_quantile_ret)['forecast']
        short_fore = curr_returns[curr_returns['permno'].isin(short_posit_con)].quantile(lower_quantile_ret)['forecast']
        long_mask = curr_returns['permno'].isin(long_posit_con) & (curr_returns['forecast'] >= long_fore)
        short_mask = curr_returns['permno'].isin(short_posit_con) & (curr_returns['forecast'] <= short_fore)

    elif strategy == 'screening':
        cons_var = params.get('cons_var', 'Analyst earnings per share')
        cons_ref = params.get('cons_ref', 0.5)
        sorting_sign = params.get('sorting_sign', 'positive')
        upper_quantile = params.get('upper_quantile', 0.9)
        lower_quantile = params.get('lower_quantile', 0.1)

        cons_threshold = curr_returns[cons_var].quantile(cons_ref)
        if sorting_sign == 'positive':
            long_mask_con = curr_returns[cons_var] >= cons_threshold
            short_mask_con = curr_returns[cons_var] <= cons_threshold
        elif sorting_sign == 'negative':
            long_mask_con = curr_returns[cons_var] <= cons_threshold
            short_mask_con = curr_returns[cons_var] >= cons_threshold
        else:
            raise ValueError(f'Unknown sorting_sign: {sorting_sign}')

        long_screened = curr_returns[long_mask_con]
        short_screened = curr_returns[short_mask_con]
        long_mask = pd.Series(False, index=curr_returns.index)
        short_mask = pd.Series(False, index=curr_returns.index)

        if len(long_screened) > 0:
            long_fore = long_screened.quantile(upper_quantile)['forecast']
            long_screened_mask = long_screened['forecast'] >= long_fore
            long_mask[long_screened_mask.index] = long_screened_mask.values
        if len(short_screened) > 0:
            short_fore = short_screened.quantile(lower_quantile)['forecast']
            short_screened_mask = short_screened['forecast'] <= short_fore
            short_mask[short_screened_mask.index] = short_screened_mask.values
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    return curr_returns[long_mask]['permno'], curr_returns[short_mask]['permno']


def calculate_portfolio_weights(curr_returns, long_posit, short_posit, weight_type='value'):
    """Portfolio weights (value- or equal-weighted) and the resulting period return."""
    if weight_type == 'value':
        long_total = curr_returns[curr_returns['permno'].isin(long_posit)]['Size'].sum()
        short_total = curr_returns[curr_returns['permno'].isin(short_posit)]['Size'].sum()
        long_weight = curr_returns[curr_returns['permno'].isin(long_posit)]['Size'] / long_total
        short_weight = -curr_returns[curr_returns['permno'].isin(short_posit)]['Size'] / short_total
    elif weight_type == 'equal':
        long_weight = pd.Series(1 / len(long_posit), index=long_posit.index)
        short_weight = pd.Series(-1 / len(short_posit), index=short_posit.index)
    else:
        raise ValueError(f'Unknown weight type: {weight_type}')

    portfolio_weight = pd.concat([long_weight, short_weight])
    portfolio_weight.name = 'weight'
    portfolio_weight = curr_returns.join(portfolio_weight, how='left')[['date', 'permno', 'weight']].fillna(0).set_index('date')

    long_return = curr_returns[curr_returns['permno'].isin(long_posit)]['actual'].values * long_weight
    short_return = curr_returns[curr_returns['permno'].isin(short_posit)]['actual'].values * short_weight
    portfolio_return = pd.concat([long_return, short_return]).sum()

    return portfolio_weight, portfolio_return


def load_other_data(data_dir=None):
    """Risk-free rate, S&P 500 benchmark log returns, and firm Size/Price for VW portfolios."""
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'

    welch_goyal = pd.read_csv(data_dir / 'raw' / 'welch_goyal_raw.csv')
    welch_goyal['yyyymm'] = pd.to_datetime(welch_goyal['yyyymm'], format='%Y%m')
    welch_goyal = welch_goyal.rename(columns={'yyyymm': 'date'})
    risk_free = welch_goyal[['date', 'Rfree']]

    snp_500 = welch_goyal[['date', 'Index']].copy()
    snp_500.loc[:, 'return'] = np.log(snp_500['Index']) - np.log(snp_500['Index'].shift(1))
    snp_500 = snp_500.drop('Index', axis=1)

    start_date = pd.to_datetime('2013-01-01')
    end_date = pd.to_datetime('2023-01-01')
    snp_500 = snp_500[(snp_500['date'] >= start_date) & (snp_500['date'] <= end_date)]

    czd = pd.read_csv(data_dir / 'raw' / 'open_source_asset_pricing.csv')
    stock_info = czd[['date', 'permno', 'Size', 'Price']]
    stock_info = pd.DataFrame({'date': pd.to_datetime(stock_info['date']), **stock_info.drop('date', axis=1)})

    return risk_free, snp_500, stock_info


def plot_portfolio_performance(horizon='12month', strategy='single_sort', weight_type='value',
                                strategy_params=None, risk_free=None, stock_info=None, snp_500=None,
                                transaction_cost=0.0, results_dir=None, out_dir=None, tag=None):
    """
    Cumulative-return plot + per-lambda portfolio construction, with optional
    proportional transaction-cost adjustment: r_t(net) = r_t(gross) -
    transaction_cost * turnover_t.
    """
    results_dir = Path(results_dir) if results_dir else REPO_ROOT / 'results'
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f'{strategy}_{weight_type}_tc{transaction_cost}'

    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 11,
        'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'lines.linewidth': 1, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    })

    portfolio_summary = {'return': pd.DataFrame(), 'weight': {}, 'returns': {}}
    final_returns = {}
    fig, ax = plt.subplots(figsize=(20, 8))

    target_lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if strategy == 'single_sort':
        target_lambdas.append(0.0)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(target_lambdas)))
    color_idx = 0
    returns = None
    return_series_by_lambda = {}  # date-indexed pd.Series per lambda -- see note below

    for file in os.listdir(results_dir):
        if not (file.startswith(horizon) and file.endswith('.pickle')):
            continue
        weight_lambda = float(file.split(f'{horizon}_')[1].split('.pickle')[0])
        if weight_lambda not in target_lambdas:
            continue

        with open(results_dir / file, 'rb') as f:
            output = pickle.load(f)

        forecast = output['forecast_target']
        forecast.columns = ['date', 'permno', 'forecast']
        forecast['date'] = pd.to_datetime(forecast['date'])

        if horizon != '1month':
            one_path = results_dir / f'1month_{weight_lambda}.pickle'
            actual = pickle.load(open(one_path, 'rb'))['actual_target'] if one_path.exists() else output['actual_target']
        else:
            actual = output['actual_target']
        actual.columns = ['date', 'permno', 'actual']
        actual['date'] = pd.to_datetime(actual['date'])

        returns = pd.merge(actual, forecast, on=['date', 'permno']).dropna()

        if strategy in ('double_sort', 'screening'):
            consensuses = output['forecast_concept']
            consensuses.columns = CONSENSUS_COLUMN_NAMES
            cons_var = strategy_params.get('cons_var', 'Analyst earnings per share')
            returns = pd.merge(returns, consensuses[['date', 'permno', cons_var]],
                                on=['date', 'permno'], how='left', validate='many_to_one')

        returns = pd.merge(returns, stock_info, how='left')

        portfolio_returns = []
        individual_returns = pd.DataFrame()
        portfolio_weights = pd.DataFrame()

        for date, curr in returns.groupby('date'):
            individual_returns = pd.concat([individual_returns, curr[['date', 'permno', 'actual']].set_index('date')])
            long_pos, short_pos = generate_portfolio(curr, strategy=strategy, params=strategy_params)
            w, r = calculate_portfolio_weights(curr, long_pos, short_pos, weight_type=weight_type)
            portfolio_weights = pd.concat([portfolio_weights, w])
            portfolio_returns.append(r)

        if transaction_cost > 0.0:
            returns_tc = individual_returns.copy()
            returns_tc['actual'] = np.exp(returns_tc['actual']) - 1
            turnover_list = turnover(returns_tc, portfolio_weights)
            tc_array = np.array([0.0] + turnover_list)
            T = len(portfolio_returns)
            if len(tc_array) < T:
                tc_array = np.concatenate([tc_array, np.zeros(T - len(tc_array))])
            elif len(tc_array) > T:
                tc_array = tc_array[:T]
            gross = np.array(portfolio_returns)
            portfolio_returns = list(gross - transaction_cost * tc_array)

        # groupby('date') iterates in sorted date order, so portfolio_returns
        # lines up with the sorted unique dates for THIS lambda's own pickle.
        lambda_dates = pd.to_datetime(sorted(returns['date'].unique()))
        label = f'lambda={weight_lambda}'
        portfolio = pd.DataFrame(index=lambda_dates)
        portfolio['return'] = portfolio_returns
        portfolio[label] = portfolio['return'].cumsum()
        final_returns[label] = portfolio[label].iloc[-1]

        ax.plot(portfolio.index, portfolio[label], color=colors[color_idx % len(colors)], linewidth=1)
        color_idx += 1

        # BUGFIX vs. the source notebook: different lambda pickles can cover a
        # different number of test dates (e.g. one lambda's 1month_<lambda>.pickle
        # missing, falling back to a differently-sized actual_target) -- the
        # notebook's original `portfolio_summary['return'][weight_lambda] =
        # portfolio_returns` assumed identical lengths across ALL lambdas and
        # would ValueError otherwise. Storing a date-indexed Series per lambda
        # and assembling the final DataFrame via pd.DataFrame(...) afterwards
        # auto-aligns on the union of dates (NaN where a lambda lacks a date)
        # instead of crashing.
        return_series_by_lambda[weight_lambda] = pd.Series(portfolio_returns, index=lambda_dates)
        portfolio_summary['weight'][weight_lambda] = portfolio_weights
        portfolio_summary['returns'][weight_lambda] = individual_returns

    if returns is None:
        print(f"No result files found for horizon '{horizon}' in {results_dir}")
        plt.close(fig)
        return portfolio_summary

    portfolio_summary['return'] = pd.DataFrame(return_series_by_lambda)
    dates = portfolio_summary['return'].index

    snp_aligned = snp_500.copy()
    snp_aligned['date'] = pd.to_datetime(snp_aligned['date'])
    snp_aligned = snp_aligned.set_index('date').reindex(dates)
    portfolio_summary['return']['S&P 500'] = snp_aligned['return'].values
    portfolio_summary['return'] = portfolio_summary['return'].reset_index().rename(columns={'index': 'date'})

    snp_plot = snp_aligned['return'].fillna(0).cumsum()
    final_returns['S&P 500'] = snp_plot.iloc[-1]
    ax.plot(snp_plot.index, snp_plot.values, linestyle='--', color='black', linewidth=1)

    ax.legend(list(final_returns.keys()), title='Portfolios', title_fontsize=25,
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=25)
    ax.set_ylabel('Cumulative Log Returns', fontsize=30)
    ax.set_xlabel('Year', fontsize=30)
    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=23)
    ax.grid()

    fig.tight_layout()
    fig.savefig(out_dir / f'{horizon}_{tag}_cumulative_returns.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    portfolio_summary['return'].to_csv(out_dir / f'{horizon}_{tag}_returns.csv', index=False)
    return portfolio_summary


def calculate_portfolio_metrics(portfolio_summary):
    """Mean/std/cumulative/Sharpe/max-1M-loss/max-DD/turnover summary table."""
    portfolio_analysis = pd.DataFrame()
    portfolio_return = portfolio_summary['return'].drop('date', axis=1)

    portfolio_analysis['mean return'] = portfolio_return.mean()
    portfolio_analysis['std'] = portfolio_return.std()

    cumsum = portfolio_return.cumsum()
    portfolio_analysis['cumulative return'] = cumsum.iloc[-1]

    annualized_return = np.exp(portfolio_return.mean() * 12) - 1
    annualized_volatility = portfolio_return.std() * np.sqrt(12)
    portfolio_analysis['Sharpe'] = annualized_return / annualized_volatility

    portfolio_analysis['max 1M loss'] = (1 - np.exp(portfolio_return.min())) * 100

    wealth = np.exp(portfolio_return.cumsum())
    portfolio_analysis['max DD'] = wealth.apply(MAX_DD) * 100

    portfolio_turnover = pd.Series(dtype=float)
    for weight_key in portfolio_summary['weight']:
        weight_returns = portfolio_summary['returns'][weight_key].copy()
        weights = portfolio_summary['weight'][weight_key].copy()
        weight_returns['actual'] = np.exp(weight_returns['actual']) - 1
        portfolio_turnover[weight_key] = np.mean(turnover(weight_returns, weights))
    portfolio_analysis['turnover'] = portfolio_turnover * 100

    return portfolio_analysis


def run(horizon='12month', out_dir=None, results_dir=None, data_dir=None,
        scenarios=None, run_latex_tables=True):
    """
    Runs a representative battery of the notebook's driver cells: single-sort
    and double-sort realized-return summaries (+ LaTeX tables), then
    cumulative-return plots and metrics for a configurable list of
    (strategy, weight_type, transaction_cost, strategy_params) scenarios.
    Default scenarios match the notebook's own worked examples.
    """
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {}

    single_sort_results = summarize_sorted_portfolios(horizon=horizon, strategy='single_sort', results_dir=results_dir)
    single_sort_results.to_csv(out_dir / f'{horizon}_single_sort_summary.csv')
    result['single_sort_summary'] = single_sort_results
    if run_latex_tables:
        (out_dir / f'{horizon}_single_sort.tex').write_text(
            generate_single_sort_latex_table(single_sort_results), encoding='utf-8')

    double_sort_results = summarize_sorted_portfolios(
        horizon=horizon, strategy='double_sort', results_dir=results_dir,
        strategy_params={'cons_var': 'Analyst earnings per share'})
    for lam, mat in double_sort_results.items():
        mat.to_csv(out_dir / f'{horizon}_double_sort_lambda{lam}.csv')
    result['double_sort_summary'] = double_sort_results
    if run_latex_tables:
        (out_dir / f'{horizon}_double_sort.tex').write_text(
            generate_double_sort_latex_table(double_sort_results, 'FEPS'), encoding='utf-8')

    risk_free, snp_500, stock_info = load_other_data(data_dir=data_dir)

    if scenarios is None:
        scenarios = [
            {'tag': 'single_value', 'strategy': 'single_sort', 'weight_type': 'value',
             'strategy_params': {'upper_quantile': 0.9, 'lower_quantile': 0.1}, 'transaction_cost': 0.0},
            {'tag': 'double_value', 'strategy': 'double_sort', 'weight_type': 'value',
             'strategy_params': {'upper_quantile_cons': 0.8, 'lower_quantile_cons': 0.2,
                                  'upper_quantile_ret': 0.8, 'lower_quantile_ret': 0.2,
                                  'sorting_sign': 'positive', 'cons_var': 'Analyst earnings per share'},
             'transaction_cost': 0.0},
            {'tag': 'single_value_tc50bps', 'strategy': 'single_sort', 'weight_type': 'value',
             'strategy_params': {'upper_quantile': 0.9, 'lower_quantile': 0.1}, 'transaction_cost': 0.005},
            {'tag': 'single_value_tc25bps', 'strategy': 'single_sort', 'weight_type': 'value',
             'strategy_params': {'upper_quantile': 0.9, 'lower_quantile': 0.1}, 'transaction_cost': 0.0025},
            {'tag': 'single_value_tc75bps', 'strategy': 'single_sort', 'weight_type': 'value',
             'strategy_params': {'upper_quantile': 0.9, 'lower_quantile': 0.1}, 'transaction_cost': 0.0075},
            {'tag': 'screening_feps_neg', 'strategy': 'screening', 'weight_type': 'value',
             'strategy_params': {'cons_var': 'Analyst earnings per share', 'cons_ref': 0.1,
                                  'sorting_sign': 'negative', 'upper_quantile': 0.9, 'lower_quantile': 0.1},
             'transaction_cost': 0.0},
            {'tag': 'screening_dispersion_neg', 'strategy': 'screening', 'weight_type': 'value',
             'strategy_params': {'cons_var': 'EPS Forecast Dispersion', 'cons_ref': 0.9,
                                  'sorting_sign': 'negative', 'upper_quantile': 0.9, 'lower_quantile': 0.1},
             'transaction_cost': 0.0},
        ]

    result['scenarios'] = {}
    for sc in scenarios:
        print(f"Running portfolio scenario: {sc['tag']}")
        summary = plot_portfolio_performance(
            horizon=horizon, strategy=sc['strategy'], weight_type=sc['weight_type'],
            strategy_params=sc['strategy_params'], risk_free=risk_free, snp_500=snp_500,
            stock_info=stock_info, transaction_cost=sc['transaction_cost'],
            results_dir=results_dir, out_dir=out_dir, tag=sc['tag'])
        metrics = calculate_portfolio_metrics(summary)
        metrics.to_csv(out_dir / f"{horizon}_{sc['tag']}_metrics.csv")
        result['scenarios'][sc['tag']] = {'summary': summary, 'metrics': metrics}

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    result = run(horizon=args.horizon)
    for tag, sc in result['scenarios'].items():
        print(f'\n{tag}:')
        print(sc['metrics'])
