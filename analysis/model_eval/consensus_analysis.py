"""
Refactored from analysis/consensus.ipynb (kept as the source notebook; this
module is the maintained, runnable version). Two analyses of the CB
framework's inferred consensus (MIC) vs. the real (raw) consensus:

  1. Distribution comparison: winsorized + standardized histogram overlay of
     each consensus coordinate, actual vs. CB-framework-inferred, plus a
     Kolmogorov-Smirnov test per coordinate.
  2. Correlation analysis: average per-coordinate correlation between actual
     and inferred consensus, in-sample (training windows) vs. out-of-sample
     (test windows).

BUGFIX vs. the source notebook: `calculate_correlation_analysis` referenced
undefined names (`analyst_col`, `input`, `target`, `feature_indices` -- a
copy-paste leftover from `plot_X_distribution_comparison`'s scope) that would
NameError immediately; this was never actually invoked in the notebook, so
the bug was latent. Fixed here to use this function's own `analyst_cols`/
`input_data`/`target_data`, and to always use all analyst columns (there is
no per-call feature subsetting for this analysis).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler

from config import get_config
from analysis.util import REPO_ROOT, get_Xy_cbapm, get_Xy_real_concept

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'consensus_analysis'

TRAIN_DATES = ['2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01',
               '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']
TEST_DATES = ['2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01',
              '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']


def _winsorize(X, lower=0.01, upper=0.99):
    X_w = X.copy()
    for i in range(X.shape[1]):
        q_low = np.nanquantile(X[:, i], lower)
        q_high = np.nanquantile(X[:, i], upper)
        X_w[:, i] = np.clip(X[:, i], q_low, q_high)
    return X_w


def plot_X_distribution_comparison(horizon, weight_lambda, embedding_method, train_date=None,
                                    bins=15, feature_indices=None, winsor_limits=(0.01, 0.99),
                                    data_dir=None, checkpoints_dir=None, out_dir=None, device=None):
    """
    Winsorized + standardized histogram overlay of each consensus coordinate
    (actual vs. CB-framework-inferred), up to 9 features in a 3x3 grid, with
    a per-coordinate Kolmogorov-Smirnov test printed to the console/log.
    """
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 24,
        'axes.linewidth': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 10, 'ytick.major.size': 10,
        'xtick.major.width': 2.5, 'ytick.major.width': 2.5,
        'lines.linewidth': 3, 'lines.markersize': 12,
        'grid.alpha': 0.4, 'grid.linewidth': 1.5,
    })
    color_real = '#6A5ACD'   # SlateBlue
    color_cbapm = '#2E8B57'  # SeaGreen

    input_df = pd.read_csv(data_dir / f'input_{horizon}.csv')
    target_df = pd.read_csv(data_dir / f'target_{horizon}.csv')
    input_df['date'] = pd.to_datetime(input_df['date'])
    target_df['date'] = pd.to_datetime(target_df['date'])
    signal_info = pd.read_csv(data_dir / 'info' / 'SignalDoc.csv')
    info = signal_info[signal_info['Acronym'].isin(input_df.columns)]
    analyst_col = info[info['Cat.Data'] == 'Analyst']['Acronym'].values
    col_names = info[info['Cat.Data'] == 'Analyst']['LongDescription'].values
    config = get_config(weight_lambda)

    if train_date is None:
        train_date = '2019-01-01'

    get_xy_kwargs = {'checkpoints_dir': checkpoints_dir} if checkpoints_dir is not None else {}
    X_cbapm, _ = get_Xy_cbapm(train_date, input_df, target_df, info, config, device, horizon,
                               weight_lambda, embedding_method, **get_xy_kwargs)
    if not isinstance(X_cbapm, pd.DataFrame):
        raise ValueError('X_cbapm must be a DataFrame with date and permno columns.')

    if feature_indices is not None:
        feature_names = [analyst_col[i] for i in feature_indices]
        col_names_plot = [col_names[i] for i in feature_indices]
        X_cbapm = X_cbapm[['permno', 'date'] + feature_names]
    else:
        feature_names = analyst_col[:len([c for c in X_cbapm.columns if c not in ['date', 'permno']])]
        col_names_plot = col_names[:len(feature_names)]
        X_cbapm = X_cbapm[['permno', 'date'] + list(feature_names)]
    X_cbapm_values = X_cbapm.drop(columns=['permno', 'date']).values

    X_real_df, _ = get_Xy_real_concept(train_date, input_df, target_df, analyst_col)
    analyst_cols_present = [c for c in analyst_col if c in X_real_df.columns]
    selected_cols = [analyst_cols_present[i] for i in feature_indices] if feature_indices is not None else analyst_cols_present
    X_real = X_real_df[selected_cols].apply(pd.to_numeric, errors='coerce').astype(float).to_numpy()

    X_cbapm_wins = _winsorize(X_cbapm_values, *winsor_limits)
    X_real_wins = _winsorize(X_real, *winsor_limits)

    X_cbapm_norm = np.zeros_like(X_cbapm_wins)
    X_real_norm = np.zeros_like(X_real_wins)
    for i in range(X_cbapm_wins.shape[1]):
        scaler = StandardScaler()
        col_data = np.concatenate([X_cbapm_wins[:, i], X_real_wins[:, i]])
        scaler.fit(col_data.reshape(-1, 1))
        X_cbapm_norm[:, i] = scaler.transform(X_cbapm_wins[:, i].reshape(-1, 1)).flatten()
        X_real_norm[:, i] = scaler.transform(X_real_wins[:, i].reshape(-1, 1)).flatten()

    non_constant_indices = [
        i for i in range(X_cbapm_norm.shape[1])
        if (np.nanmax(X_cbapm_norm[:, i]) > np.nanmin(X_cbapm_norm[:, i])) or
           (np.nanmax(X_real_norm[:, i]) > np.nanmin(X_real_norm[:, i]))
    ]
    n_plot = min(len(non_constant_indices), 9)
    if n_plot == 0:
        print('No non-constant features to plot.')
        return None

    ks_rows = []
    for i in range(X_cbapm_norm.shape[1]):
        ks_stat, ks_pval = ks_2samp(X_real_norm[:, i], X_cbapm_norm[:, i])
        ks_rows.append({'feature': col_names_plot[i] if i < len(col_names_plot) else i, 'ks_stat': ks_stat, 'ks_pval': ks_pval})
    ks_df = pd.DataFrame(ks_rows)
    ks_df.to_csv(out_dir / f'{horizon}_lambda{weight_lambda}_ks_test.csv', index=False)

    fig, axes = plt.subplots(3, 3, figsize=(36, 36))
    axes = axes.flatten()
    for plot_idx, i in enumerate(non_constant_indices[:9]):
        ax = axes[plot_idx]
        min_bin = min(np.nanmin(X_cbapm_norm[:, i]), np.nanmin(X_real_norm[:, i]), -4)
        max_bin = max(np.nanmax(X_cbapm_norm[:, i]), np.nanmax(X_real_norm[:, i]), 4)
        bins_edges = np.linspace(min_bin, max_bin, bins + 1)

        ax.hist(X_real_norm[:, i], bins=bins_edges, alpha=0.5, label='Actual', color=color_real,
                density=True, edgecolor='white', linewidth=2, rwidth=0.95, histtype='bar')
        ax.hist(X_cbapm_norm[:, i], bins=bins_edges, alpha=0.5, label='Approximated', color=color_cbapm,
                density=True, edgecolor='black', linewidth=2, rwidth=0.95, histtype='bar')
        ax.set_xlim([min_bin, max_bin])
        ax.set_title(f'{col_names_plot[i]}', fontweight='bold', fontsize=32)
        ax.set_ylabel('Density', fontweight='bold', fontsize=28) if plot_idx % 3 == 0 else ax.set_ylabel('')
        ax.set_xlabel('Standardized Value', fontweight='bold', fontsize=28) if plot_idx // 3 == 2 else ax.set_xlabel('')
        ax.grid(axis='y', alpha=0.4, linewidth=1.5)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['bottom'].set_linewidth(2.5)
        legend = ax.legend(frameon=True, fancybox=True, framealpha=0.85, fontsize=22, loc='upper right')
        legend.get_frame().set_edgecolor('gray')

    for i in range(n_plot, 9):
        fig.delaxes(axes[i])
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(out_dir / f'{horizon}_lambda{weight_lambda}_distribution_comparison.png', dpi=100)
    plt.close(fig)
    return ks_df


def calculate_correlation_analysis(horizon, weight_lambda, data_dir=None, checkpoints_dir=None, device=None):
    """
    Average per-coordinate correlation between actual and CB-framework-
    inferred consensus, in-sample (TRAIN_DATES) vs. out-of-sample (TEST_DATES).
    """
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_data = pd.read_csv(data_dir / f'input_{horizon}.csv')
    target_data = pd.read_csv(data_dir / f'target_{horizon}.csv')
    input_data['date'] = pd.to_datetime(input_data['date'])
    target_data['date'] = pd.to_datetime(target_data['date'])
    # NOTE: the source notebook dropped 'ChangeInRecommendation' here, which
    # breaks get_Xy_cbapm's downstream input-dimensionality assumption
    # (config['input_size']=146 expects the FULL column set) -- this
    # function was never actually invoked in the notebook (dead code with a
    # latent bug), so the drop is removed here rather than preserved.

    signal_info = pd.read_csv(data_dir / 'info' / 'SignalDoc.csv')
    info = signal_info[signal_info['Acronym'].isin(input_data.columns)]
    analyst_cols = info[info['Cat.Data'] == 'Analyst']['Acronym'].values
    config = get_config(weight_lambda)
    get_xy_kwargs = {'checkpoints_dir': checkpoints_dir} if checkpoints_dir is not None else {}

    results = {
        'in_sample': {'correlations': [], 'dates': [], 'mean_corr': None, 'std_corr': None},
        'out_of_sample': {'correlations': [], 'dates': [], 'mean_corr': None, 'std_corr': None},
    }

    def _avg_corr_for_date(date):
        # embedding_method MUST be 'autoencoder' here to match the 146-d input
        # (114 firm + 32 macro-embedding) the model was actually trained on --
        # the source notebook's cell omitted this argument too (defaulting to
        # 'none', i.e. 114 firm + 126 raw macro = 240-d), which would have
        # crashed with a matmul shape mismatch had this dead code ever been run.
        X_cbapm, _ = get_Xy_cbapm(date, input_data, target_data, info, config, device, horizon,
                                   weight_lambda, embedding_method='autoencoder', **get_xy_kwargs)
        X_real_df, _ = get_Xy_real_concept(date, input_data, target_data, analyst_cols)
        analyst_cols_present = [c for c in analyst_cols if c in X_real_df.columns]
        X_real = X_real_df[analyst_cols_present].apply(pd.to_numeric, errors='coerce').astype(float).to_numpy()

        cbapm_features = X_cbapm.drop(columns=['date', 'permno']).values if isinstance(X_cbapm, pd.DataFrame) else X_cbapm
        n = min(len(cbapm_features), len(X_real))
        cbapm_features, X_real = cbapm_features[:n], X_real[:n]

        valid_mask = ~(np.isnan(cbapm_features).any(axis=1) | np.isnan(X_real).any(axis=1))
        cbapm_features, X_real_valid = cbapm_features[valid_mask], X_real[valid_mask]
        if len(cbapm_features) == 0:
            return None

        feature_corrs = []
        for i in range(cbapm_features.shape[1]):
            corr = np.corrcoef(cbapm_features[:, i], X_real_valid[:, i])[0, 1]
            if not np.isnan(corr):
                feature_corrs.append(corr)
        return float(np.mean(feature_corrs)) if feature_corrs else None

    print('Calculating in-sample correlation...')
    for train_date in TRAIN_DATES:
        try:
            avg_corr = _avg_corr_for_date(train_date)
            if avg_corr is not None:
                results['in_sample']['correlations'].append(avg_corr)
                results['in_sample']['dates'].append(train_date)
                print(f'  {train_date}: average correlation = {avg_corr:.4f}')
        except Exception as e:
            print(f'  Error while processing {train_date}: {e}')

    print('\nCalculating out-of-sample correlation...')
    for test_date in TEST_DATES:
        try:
            avg_corr = _avg_corr_for_date(test_date)
            if avg_corr is not None:
                results['out_of_sample']['correlations'].append(avg_corr)
                results['out_of_sample']['dates'].append(test_date)
                print(f'  {test_date}: average correlation = {avg_corr:.4f}')
        except Exception as e:
            print(f'  Error while processing {test_date}: {e}')

    if results['in_sample']['correlations']:
        results['in_sample']['mean_corr'] = float(np.mean(results['in_sample']['correlations']))
        results['in_sample']['std_corr'] = float(np.std(results['in_sample']['correlations']))
    if results['out_of_sample']['correlations']:
        results['out_of_sample']['mean_corr'] = float(np.mean(results['out_of_sample']['correlations']))
        results['out_of_sample']['std_corr'] = float(np.std(results['out_of_sample']['correlations']))

    return results


def plot_correlation_comparison(results, horizon, weight_lambda, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 14,
        'axes.linewidth': 1.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 8, 'ytick.major.size': 8,
        'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
        'lines.linewidth': 2.5, 'lines.markersize': 8,
        'grid.alpha': 0.3, 'grid.linewidth': 0.8,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    if results['in_sample']['correlations']:
        ax1.plot(results['in_sample']['dates'], results['in_sample']['correlations'], 'o-',
                  color='#2E8B57', linewidth=2, markersize=8, label='In-sample')
        ax1.axhline(y=results['in_sample']['mean_corr'], color='#2E8B57', linestyle='--', alpha=0.7,
                     label=f"Mean: {results['in_sample']['mean_corr']:.4f}")
        ax1.set_title(f'In-sample Correlation (lambda = {weight_lambda})', fontweight='bold', fontsize=16)
        ax1.set_ylabel('Average Correlation', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

    if results['out_of_sample']['correlations']:
        ax2.plot(results['out_of_sample']['dates'], results['out_of_sample']['correlations'], 's-',
                  color='#6A5ACD', linewidth=2, markersize=8, label='Out-of-sample')
        ax2.axhline(y=results['out_of_sample']['mean_corr'], color='#6A5ACD', linestyle='--', alpha=0.7,
                     label=f"Mean: {results['out_of_sample']['mean_corr']:.4f}")
        ax2.set_title(f'Out-of-sample Correlation (lambda = {weight_lambda})', fontweight='bold', fontsize=16)
        ax2.set_ylabel('Average Correlation', fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

    fig.suptitle(f'{horizon} Horizon: Consensus Correlation Analysis', fontweight='bold', fontsize=18)
    fig.tight_layout()
    fig.savefig(out_dir / f'{horizon}_lambda{weight_lambda}_correlation_comparison.png', dpi=150)
    plt.close(fig)

    print(f'\n=== {horizon} Horizon Correlation Summary ===')
    print(f'lambda = {weight_lambda}')
    if results['in_sample']['mean_corr'] is not None:
        print(f"In-sample: mean = {results['in_sample']['mean_corr']:.4f}, std = {results['in_sample']['std_corr']:.4f}")
    if results['out_of_sample']['mean_corr'] is not None:
        print(f"Out-of-sample: mean = {results['out_of_sample']['mean_corr']:.4f}, std = {results['out_of_sample']['std_corr']:.4f}")


def run(horizon='12month', weight_lambda=1.0, embedding_method='autoencoder', bins=50,
        data_dir=None, checkpoints_dir=None, out_dir=None, run_correlation=True):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Plotting X distribution comparison for {horizon}, lambda={weight_lambda}')
    ks_df = plot_X_distribution_comparison(
        horizon=horizon, weight_lambda=weight_lambda, embedding_method=embedding_method, bins=bins,
        data_dir=data_dir, checkpoints_dir=checkpoints_dir, out_dir=out_dir)

    result = {'ks_df': ks_df}
    if run_correlation:
        corr_results = calculate_correlation_analysis(horizon, weight_lambda, data_dir=data_dir, checkpoints_dir=checkpoints_dir)
        plot_correlation_comparison(corr_results, horizon, weight_lambda, out_dir=out_dir)
        result['correlation_results'] = corr_results

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    parser.add_argument('--lambda_', dest='weight_lambda', type=float, default=1.0)
    parser.add_argument('--no_correlation', action='store_true')
    args = parser.parse_args()
    run(horizon=args.horizon, weight_lambda=args.weight_lambda, run_correlation=not args.no_correlation)
