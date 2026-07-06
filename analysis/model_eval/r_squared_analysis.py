"""
Refactored from analysis/r_squared.ipynb (kept as the source notebook; this
module is the maintained, runnable version). Out-of-sample R2 across the
lambda grid (read from results/<h>_<lambda>.csv, written by run.py), plus a
per-period naive (lambda=0) vs best-lambda comparison with an Excel summary
(tables/<h>_r2_analysis_summary.xlsx).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from analysis.util import REPO_ROOT

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'r_squared_analysis'

# results/ (written by the current run.py) is the primary source and gives
# the full, fine-grained lambda grid. But a handful of results/ files do NOT
# match the frozen, paper-verified final_results/ artifacts even though the
# filename/lambda is the same -- confirmed: results/12month_0.0.csv (Return
# whole-period R2 = -1.42%) and results/12month_0.3.csv (0.51%) diverge from
# final_results/12month_0.0.csv (7.63%) and final_results/12month_0.3.csv
# (10.46%), which are the values analysis/outputs/DEVIATIONS.md's E0 check
# and the source notebook (analysis/r_squared.ipynb, cell 4's cached
# `display(whole_periods)` output) both confirm as correct. Every other
# horizon/lambda file that exists in both directories is byte-identical
# (verified: 264/266 common files match, only the two 12month files above
# differ) -- this is isolated stale/bad data in results/ for those two
# files, not a lambda=0-specific formula difference. See
# _resolve_result_files, which lets final_results/ override results/ file by
# file so the fine grid is preserved everywhere else.
DEFAULT_FINAL_RESULTS_DIR = REPO_ROOT / 'final_results'


def _resolve_result_files(horizon, results_dir, final_results_dir=None):
    """
    Build {weight_lambda: path} for a horizon's whole-period result CSVs,
    preferring final_results_dir's copy of a file when the same
    '<horizon>_<lambda>.csv' name exists in both directories (see module
    docstring above for why).
    """
    results_dir = Path(results_dir)
    final_results_dir = Path(final_results_dir) if final_results_dir else DEFAULT_FINAL_RESULTS_DIR

    files = {}
    if results_dir.exists():
        for file in os.listdir(results_dir):
            if file.startswith(horizon) and file.endswith('.csv') and not file.endswith('mse.csv'):
                weight_lambda = float(file.split(f'{horizon}_')[1].split('.csv')[0])
                files[weight_lambda] = results_dir / file
    if final_results_dir.exists():
        for file in os.listdir(final_results_dir):
            if file.startswith(horizon) and file.endswith('.csv') and not file.endswith('mse.csv'):
                weight_lambda = float(file.split(f'{horizon}_')[1].split('.csv')[0])
                files[weight_lambda] = final_results_dir / file
    return files

DEFAULT_TEST_COLUMNS = ['2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01',
                         '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']
DEFAULT_INDEX_NAMES = ['2013-01-01~2013-12-31', '2014-01-01~2014-12-31', '2015-01-01~2015-12-31',
                        '2016-01-01~2016-12-31', '2017-01-01~2017-12-31', '2018-01-01~2018-12-31',
                        '2019-01-01~2019-12-31', '2020-01-01~2020-12-31', '2021-01-01~2021-12-31',
                        '2022-01-01~2022-12-31']


def _plot_style():
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 15,
        'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'lines.linewidth': 2, 'lines.markersize': 6,
        'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    })


def plot_r_squared(horizon, result_path=None, save_path=None, out_dir=None, final_results_dir=None):
    """Out-of-sample R2 across the lambda grid (whole-period, asset returns + consensus average)."""
    result_path = Path(result_path) if result_path else REPO_ROOT / 'results'
    save_path = Path(save_path) if save_path else REPO_ROOT / 'tables'
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_style()

    whole_periods = pd.DataFrame()
    result = None
    files = _resolve_result_files(horizon, result_path, final_results_dir)
    for weight_lambda, path in files.items():
        result = pd.read_csv(path)
        whole_periods[weight_lambda] = result['Whole periods']

    if result is None:
        print(f"No R2 files found for horizon '{horizon}' in {result_path}")
        return None

    whole_periods.index = result['Unnamed: 0']
    whole_periods.loc['Consensus average', 0.000] = np.nan
    whole_periods = whole_periods.reindex(sorted(whole_periods.columns, key=float), axis=1)

    whole_periods.T.to_csv(save_path / f'{horizon}_r_squared.csv')
    whole_periods.T.to_csv(out_dir / f'{horizon}_r_squared.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#2E86AB', '#A23B72']

    ax1.plot(whole_periods.columns, whole_periods.iloc[-1], '--', color=colors[0], linewidth=2,
              markersize=6, markerfacecolor=colors[0], markeredgecolor='black', markeredgewidth=0.5)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('Hyperparameter (lambda)')
    ax1.set_ylabel('Out-of-Sample R2 (%)')
    ax1.set_title('Asset Returns', pad=15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(whole_periods.columns, whole_periods.iloc[-2], '--', color=colors[1], linewidth=2,
              markersize=6, markerfacecolor=colors[1], markeredgecolor='black', markeredgewidth=0.5)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('Hyperparameter (lambda)')
    ax2.set_ylabel('Out-of-Sample R2 (%)')
    ax2.set_title('Consensus Variables', pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(out_dir / f'{horizon}_r_squared.png', dpi=150)
    plt.close(fig)
    return whole_periods


def load_r2_data_and_best_lambda(horizon, columns, results_dir=None, final_results_dir=None):
    results_dir = Path(results_dir) if results_dir else REPO_ROOT / 'results'
    data_dict = {}
    r_squared_by_lambda = {}
    files = _resolve_result_files(horizon, results_dir, final_results_dir)
    for weight_lambda, path in files.items():
        result = pd.read_csv(path)
        data_dict[weight_lambda] = result
        r_squared_by_lambda[weight_lambda] = result['Whole periods'].iloc[-1]
    non_zero_lambdas = {k: v for k, v in r_squared_by_lambda.items() if k != 0.0}
    best_lambda = max(non_zero_lambdas, key=non_zero_lambdas.get)
    return data_dict, r_squared_by_lambda, best_lambda


def calculate_r2_summaries(columns, index_name, data_dict, r_squared_by_lambda, best_lambda):
    summary_df = pd.DataFrame({
        'lambda': [0.0, best_lambda, 'improvement'],
        'R_squared': [
            r_squared_by_lambda[0.0], r_squared_by_lambda[best_lambda],
            r_squared_by_lambda[best_lambda] - r_squared_by_lambda[0.0],
        ],
    })
    naive_returns = data_dict[0.0].iloc[-1][columns]
    best_returns = data_dict[best_lambda].iloc[-1][columns]
    period_comparison_df = pd.DataFrame({
        'Period': [p.split('~')[0][:4] for p in index_name],
        'Naive_R2': [naive_returns[c] for c in columns],
        'Best_R2': [best_returns[c] for c in columns],
        'Diff': [best_returns[c] - naive_returns[c] for c in columns],
    })

    improved_periods, declined_periods = [], []
    for col, period_name in zip(columns, index_name):
        improvement = best_returns[col] - naive_returns[col]
        if improvement > 0:
            improved_periods.append((period_name.split('~')[0][:4], improvement))
        elif improvement < 0:
            declined_periods.append((period_name.split('~')[0][:4], improvement))
    improved_periods_df = pd.DataFrame(improved_periods, columns=['Period', 'Improvement'])
    declined_periods_df = pd.DataFrame(declined_periods, columns=['Period', 'Decline'])

    summary_stats_df = pd.DataFrame({
        'Average Improvement': [sum(v for _, v in improved_periods) / len(improved_periods) if improved_periods else 0],
        'Average Decline': [sum(v for _, v in declined_periods) / len(declined_periods) if declined_periods else 0],
        'Total Periods': [len(columns)],
        'Improvement Rate (%)': [len(improved_periods) / len(columns) * 100 if columns else 0],
    })
    return summary_df, period_comparison_df, improved_periods_df, declined_periods_df, summary_stats_df, best_lambda


def save_r2_results_to_excel(period_comparison_df, improved_periods_df, declined_periods_df,
                              summary_stats_df, summary_df, filename):
    filename = Path(filename)
    sheets = {
        'Period_Comparison': period_comparison_df, 'Improved_Periods': improved_periods_df,
        'Declined_Periods': declined_periods_df, 'Summary_Stats': summary_stats_df,
        'R2_Summary': summary_df,
    }
    try:
        with pd.ExcelWriter(filename) as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    except ImportError:
        # openpyxl not installed in this env -- fall back to one csv per sheet
        # (see analysis/outputs/DEVIATIONS.md)
        sheet_dir = filename.with_suffix('')
        sheet_dir.mkdir(parents=True, exist_ok=True)
        for sheet_name, df in sheets.items():
            df.to_csv(sheet_dir / f'{sheet_name}.csv', index=False)


def plot_r2_comparison(columns, index_name, data_dict, best_lambda, out_dir, horizon):
    _plot_style()
    selected_lambdas = [0.0, best_lambda]
    fig_widths = [5, 8]
    fig, axes = plt.subplots(1, len(selected_lambdas), figsize=(sum(fig_widths), 6),
                              gridspec_kw={'width_ratios': fig_widths})
    colors = ['#2E86AB', '#A23B72']

    for idx, weight_lambda in enumerate(selected_lambdas):
        ax = axes[idx]
        result = data_dict[weight_lambda]
        index = np.arange(len(columns))
        if weight_lambda == 0.0:
            ax.bar(index, result.iloc[-1][columns], 0.5, label='Asset Returns',
                   color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)
        else:
            bar_width = 0.35
            ax.bar(index - bar_width / 2, result.iloc[-1][columns], bar_width, label='Asset Returns',
                   color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5, hatch='////')
            ax.bar(index + bar_width / 2, result.iloc[-2][columns], bar_width, label='Consensus Variables',
                   color=colors[1], alpha=0.8, edgecolor='black', linewidth=0.5, hatch='....')

        ax.axhline(y=0.0, color='black', linestyle='-', linewidth=2, alpha=0.8)
        ax.set_xticks(index)
        ax.set_xticklabels([name.split('~')[0][:4] for name in index_name], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_title('Naive Neural Network (lambda=0)' if weight_lambda == 0.0 else f'Best Model (lambda={best_lambda:.3f})', pad=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=True, fancybox=False, shadow=False, edgecolor='black',
                   framealpha=0.9, fontsize=15, loc='upper left')

        all_bar_values = []
        for wl2 in selected_lambdas:
            r2 = data_dict[wl2]
            all_bar_values.extend(list(r2.iloc[-1][columns]))
            if wl2 != 0.0:
                all_bar_values.extend(list(r2.iloc[-2][columns]))
        y_min, y_max = min(all_bar_values), max(all_bar_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        ax.tick_params(axis='x', pad=6)
        ax.set_xlabel('Testing Period Start Year', fontsize=18)
        ax.set_ylabel('Out-of-Sample R2 (%)', fontsize=18)

    fig.tight_layout()
    fig.savefig(out_dir / f'{horizon}_r2_comparison.png', dpi=150)
    plt.close(fig)


def plot_r_squared_by_period(columns, index_name, horizon, results_dir=None, tables_dir=None, out_dir=None,
                              final_results_dir=None):
    """Per-period naive (lambda=0) vs best-lambda R2 comparison + Excel summary."""
    results_dir = Path(results_dir) if results_dir else REPO_ROOT / 'results'
    tables_dir = Path(tables_dir) if tables_dir else REPO_ROOT / 'tables'
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_style()

    data_dict, r_squared_by_lambda, best_lambda = load_r2_data_and_best_lambda(
        horizon, columns, results_dir=results_dir, final_results_dir=final_results_dir)
    summary_df, period_comparison_df, improved_periods_df, declined_periods_df, summary_stats_df, best_lambda = \
        calculate_r2_summaries(columns, index_name, data_dict, r_squared_by_lambda, best_lambda)

    filename = tables_dir / f'{horizon}_r2_analysis_summary.xlsx'
    save_r2_results_to_excel(period_comparison_df, improved_periods_df, declined_periods_df,
                              summary_stats_df, summary_df, filename)
    save_r2_results_to_excel(period_comparison_df, improved_periods_df, declined_periods_df,
                              summary_stats_df, summary_df, out_dir / f'{horizon}_r2_analysis_summary.xlsx')

    plot_r2_comparison(columns, index_name, data_dict, best_lambda, out_dir, horizon)
    return summary_df, period_comparison_df, best_lambda


def _derive_test_columns(horizon, results_dir, final_results_dir=None):
    """
    DEFAULT_TEST_COLUMNS/DEFAULT_INDEX_NAMES (from the source notebook) are
    hardcoded to a 10-window, 2014-2023 test-date scheme -- that matched an
    older results/ snapshot but NOT the current results/ (which uses a
    9-window, 2015-2023 scheme, matching the current run.py). Rather than
    hand a stale hardcoded list to callers, derive the columns actually
    present for this horizon/results_dir at call time.
    """
    files = _resolve_result_files(horizon, results_dir, final_results_dir)
    path = files.get(0.0)
    if path is None or not Path(path).exists():
        return DEFAULT_TEST_COLUMNS, DEFAULT_INDEX_NAMES
    cols = [c for c in pd.read_csv(path, index_col=0).columns if c != 'Whole periods']
    index_names = [f'{pd.Timestamp(c).year - 1}-01-01~{pd.Timestamp(c).year - 1}-12-31' for c in cols]
    return cols, index_names


def run(horizons=('1month', '3month', '6month', '12month'), columns=None, index_name=None,
        results_dir=None, tables_dir=None, out_dir=None, final_results_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir_resolved = Path(results_dir) if results_dir else REPO_ROOT / 'results'

    whole_period_results, by_period_results = {}, {}
    for horizon in horizons:
        print(f'Plotting R2 for {horizon} prediction horizon')
        wp = plot_r_squared(horizon, result_path=results_dir, save_path=tables_dir, out_dir=out_dir,
                             final_results_dir=final_results_dir)
        if wp is not None:
            whole_period_results[horizon] = wp

        cols = columns or _derive_test_columns(horizon, results_dir_resolved, final_results_dir)[0]
        idx_names = index_name or _derive_test_columns(horizon, results_dir_resolved, final_results_dir)[1]

        print(f'Plotting R2-by-period for {horizon} prediction horizon')
        try:
            summary_df, period_comparison_df, best_lambda = plot_r_squared_by_period(
                cols, idx_names, horizon, results_dir=results_dir, tables_dir=tables_dir, out_dir=out_dir,
                final_results_dir=final_results_dir)
            by_period_results[horizon] = {'summary': summary_df, 'by_period': period_comparison_df, 'best_lambda': best_lambda}
        except Exception as e:
            print(f'R2-by-period failed for {horizon}: {e}')

    return {'whole_period': whole_period_results, 'by_period': by_period_results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizons', nargs='+', default=['1month', '3month', '6month', '12month'])
    args = parser.parse_args()
    result = run(horizons=args.horizons)
    for horizon, info in result['by_period'].items():
        print(f"{horizon}: best_lambda={info['best_lambda']}")
