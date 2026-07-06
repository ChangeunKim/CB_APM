"""
Refactored from analysis/HJD.ipynb (kept as the source notebook; this module
is the maintained, runnable version). Hansen-Jagannathan Distance (HJD)
analysis: for each lambda in the grid, extracts the CB-framework's inferred
consensus (MIC) as a set of candidate pricing factors, and measures how much
adding them to standard benchmark factor models (CAPM/FF3/FF5/FF6) reduces
the HJD against several test-asset portfolios (25 Portfolios 5x5, 25
Portfolios ME x Prior, 30 Industry Portfolios).

NOTE: uses analysis/util.py::get_Xy_cbapm, which loads trained models from
`checkpoints/<horizon>_<lambda>/` (the CURRENT pipeline's checkpoints, not
final_checkpoints/) via `utils.data_utils.create_dataloaders` -- i.e. this
performs live model INFERENCE (forward pass through an already-trained
model), not training. Requires `checkpoints/<horizon>_<lambda>/{train_date}model_{i}.pt`
and the matching autoencoder checkpoint to exist for every lambda in the grid.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from analysis.util import REPO_ROOT, get_Xy_cbapm
from config import get_config

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'hjd'

BENCHMARKS = {
    'CAPM': ['Mkt-RF'],
    'FF3': ['Mkt-RF', 'SMB', 'HML'],
    'FF5': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
    'FF6': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD'],
}

PORTFOLIO_FILES = {
    '25_Portfolios_5x5': 'data/25_Portfolios_5x5.csv',
    '25_Portfolios_ME_Prior_12_2': 'data/25_Portfolios_ME_Prior_12_2.csv',
    '30_Industry_Portfolios': 'data/30_Industry_Portfolios.csv',
}


def load_ff_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    col_orig = df.columns[0]
    df[col_orig] = df[col_orig].astype(str).str.strip()
    df = df[df[col_orig].str.match(r'^\d{6}$')].copy()
    df['date'] = pd.to_datetime(df[col_orig], format='%Y%m')
    df = df.set_index('date')
    if col_orig != 'date':
        df = df.drop(columns=[col_orig], errors='ignore')
    return df.astype(float) / 100.0


def load_factors_and_assets(data_dir=None):
    if data_dir is None:
        data_dir = REPO_ROOT / 'data'
    data_dir = Path(data_dir)

    ff5_df = load_ff_csv(data_dir / 'F-F_Research_Data_5_Factors_2x3.csv')
    mom_df = load_ff_csv(data_dir / 'F-F_Momentum_Factor.csv').rename(columns={'Mom': 'UMD'})
    rf = ff5_df['RF']
    all_bench_factors = pd.concat([ff5_df.drop(columns=['RF']), mom_df], axis=1).dropna()

    portfolios_ex = {}
    for name, rel_path in PORTFOLIO_FILES.items():
        df = load_ff_csv(REPO_ROOT / rel_path)
        df_ex = df.sub(rf, axis=0).drop(columns=['RF'], errors='ignore').dropna()
        portfolios_ex[name] = df_ex

    return all_bench_factors, portfolios_ex


def get_implied_sdf(factors, returns_ex):
    """Implied SDF m_t = 1 - (f_t - f_bar)'b, b chosen to price the test assets."""
    f = factors.values if isinstance(factors, pd.DataFrame) else factors
    r = returns_ex.values if isinstance(returns_ex, pd.DataFrame) else returns_ex

    f_bar = f.mean(axis=0)
    f_adj = f - f_bar
    sigma_f = np.cov(f, rowvar=False)
    if sigma_f.ndim == 0:
        sigma_f = sigma_f.reshape(1, 1)

    sigma_f_inv = np.linalg.pinv(sigma_f)
    pricing_errors_f = (f.T @ r / len(f)).mean(axis=1)
    b = sigma_f_inv @ pricing_errors_f

    return 1 - f_adj @ b


def calculate_hj_distance(sdf, returns_ex):
    r = returns_ex.values if isinstance(returns_ex, pd.DataFrame) else returns_ex
    pricing_errors = (sdf[:, None] * r).mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(r, rowvar=False))
    return np.sqrt(pricing_errors.T @ cov_inv @ pricing_errors)


def run(horizon='12month', train_date='2020-01-01', lambdas=None, checkpoints_dir=None,
        data_dir=None, out_dir=None, device=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if lambdas is None:
        lambdas = [round(x * 0.1, 1) for x in range(11)]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading factors and test assets...')
    all_bench_factors, portfolios_ex = load_factors_and_assets(data_dir=data_dir)

    print('Loading CB-framework base data...')
    input_df = pd.read_csv(REPO_ROOT / f'data/input_{horizon}.csv')
    target_df = pd.read_csv(REPO_ROOT / f'data/target_{horizon}.csv')
    input_df['date'] = pd.to_datetime(input_df['date'])
    target_df['date'] = pd.to_datetime(target_df['date'])
    signal_info = pd.read_csv(REPO_ROOT / 'data/info/SignalDoc.csv')
    info = signal_info[signal_info['Acronym'].isin(input_df.columns)]

    portfolio_results = {}
    for port_name, ta_ex in portfolios_ex.items():
        print(f'\nAnalysing portfolio: {port_name}')
        rows = []
        for lam in tqdm(lambdas, desc=f'lambdas for {port_name}'):
            config = get_config(lam)
            X_cbapm, _ = get_Xy_cbapm(
                train_date=train_date, input=input_df, target=target_df, info=info,
                config=config, device=device, horizon=horizon, weight_lambda=lam,
                embedding_method='autoencoder', checkpoints_dir=checkpoints_dir)
            X_cbapm['date'] = pd.to_datetime(X_cbapm['date'])
            concepts_ts = X_cbapm.groupby('date').mean().drop(columns=['permno'], errors='ignore')

            common_idx = concepts_ts.index.intersection(ta_ex.index).intersection(all_bench_factors.index)
            c_ts = concepts_ts.loc[common_idx]
            p_ex = ta_ex.loc[common_idx]
            bench_f = all_bench_factors.loc[common_idx]

            row = {'lambda': lam}
            for name, cols in BENCHMARKS.items():
                b_data = bench_f[cols]
                sdf_b = get_implied_sdf(b_data, p_ex)
                hjd_b = calculate_hj_distance(sdf_b, p_ex)

                aug_data = pd.concat([b_data, c_ts], axis=1)
                sdf_aug = get_implied_sdf(aug_data, p_ex)
                hjd_aug = calculate_hj_distance(sdf_aug, p_ex)

                row[f'{name}_hjd'] = hjd_b
                row[f'{name}_aug_hjd'] = hjd_aug
                row[f'{name}_reduction'] = (hjd_b - hjd_aug) / hjd_b

            rows.append(row)

        df_res = pd.DataFrame(rows)
        df_res.to_csv(out_dir / f'{horizon}_{port_name}_hjd.csv', index=False)
        portfolio_results[port_name] = df_res

    print('\nAll analyses complete.')
    _plot_results(portfolio_results, out_dir, horizon)
    return portfolio_results


def _plot_results(portfolio_results, out_dir, horizon):
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 12,
        'axes.linewidth': 1.2, 'figure.figsize': (12, 7),
    })
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3']
    markers = ['o', 's', '^', 'D', 'v', 'x', '*']

    for port_name, df_res in portfolio_results.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, name in enumerate(BENCHMARKS.keys()):
            ax.plot(df_res['lambda'], df_res[f'{name}_reduction'] * 100,
                    marker=markers[i % len(markers)], markersize=6, label=f'vs {name}',
                    color=colors[i % len(colors)], linewidth=2, alpha=0.9)

        ax.set_xlabel('Hyperparameter (lambda)', fontsize=11)
        ax.set_ylabel('HJD Reduction (%)', fontsize=11)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.grid(True, axis='both', alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=True, facecolor='white', framealpha=0.8, loc='best', fontsize=10)

        y_vals = df_res[[f'{n}_reduction' for n in BENCHMARKS.keys()]].values * 100
        ax.set_ylim(y_vals.min() - 0.5, y_vals.max() + 0.5)

        fig.tight_layout()
        fig.savefig(out_dir / f'{horizon}_{port_name}_hjd_reduction.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    parser.add_argument('--train_date', default='2020-01-01')
    args = parser.parse_args()
    result = run(horizon=args.horizon, train_date=args.train_date)
    for name, df in result.items():
        print(f'\n{name}:')
        print(df)
