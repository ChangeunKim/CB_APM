"""
Refactored from analysis/mse.ipynb (kept as the source notebook; this module
is the maintained, runnable version). Out-of-sample MSE across the lambda
grid (read straight from results/<h>_<lambda>_mse.csv, written by run.py),
and optional in-sample MSE (re-runs INFERENCE ONLY -- forward passes through
already-trained models in checkpoints/<h>_<lambda>/ over each window's own
TRAINING data -- not retraining) for a given lambda list.
"""
import argparse
import os
import traceback
from pathlib import Path

import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm

from analysis.util import REPO_ROOT, get_lambda_list_for_horizon
from models.networks import ConceptBottleneckModel
from models.metrics import evaluate_mse
from models.test import test
from utils.data_utils import create_dataloaders
from config import get_config

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'mse_analysis'

TRAIN_DATES = ['2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01',
               '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']


def calc_in_sample_mse(horizon, weight_lambda, checkpoints_dir=None, data_dir=None, device=None):
    """
    In-sample MSE (mean over the TRAIN_DATES grid) via INFERENCE ONLY through
    already-trained checkpoints (no training) -- averages each window's own
    training-period MSE, matching the notebook's original semantics.
    """
    if checkpoints_dir is None:
        checkpoints_dir = REPO_ROOT / 'checkpoints'
    checkpoints_dir = Path(checkpoints_dir)
    if data_dir is None:
        data_dir = REPO_ROOT / 'data'
    data_dir = Path(data_dir)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_df = pd.read_csv(data_dir / f'input_{horizon}.csv')
    target_df = pd.read_csv(data_dir / f'target_{horizon}.csv')
    input_df['date'] = pd.to_datetime(input_df['date'])
    target_df['date'] = pd.to_datetime(target_df['date'])
    signal_info = pd.read_csv(data_dir / 'info' / 'SignalDoc.csv')
    info = signal_info[signal_info['Acronym'].isin(input_df.columns)]

    mse_list = []
    for train_date in TRAIN_DATES:
        model_dir = checkpoints_dir / f'{horizon}_{weight_lambda}'
        autoencoder_path = str(model_dir / f'{train_date}_autoencoder_model0.pt')
        config = get_config(weight_lambda)
        try:
            train_loader, _, _, _, _ = create_dataloaders(
                input_df, target_df, info, train_date, train_date, train_date,
                batch_size=config['batch_size'], embedding_method='autoencoder',
                model_path=autoencoder_path)
            models = []
            for i in range(config['ensemble']):
                model = ConceptBottleneckModel(
                    config['input_size'], config['concept_hidden_sizes'], config['concept_output_size'],
                    config['final_hidden_sizes'], config['final_output_size']).to(device)
                model_path = model_dir / f'{train_date}model_{i}.pt'
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                models.append(model)
            actual_concept, actual_target, forecast_concept, forecast_target = test(
                models, train_loader, config['ensemble'], device)
            mse = evaluate_mse(actual_concept, actual_target, forecast_concept, forecast_target, info)
            mse_list.append(mse)
        except Exception as e:
            print(f'{train_date} calculation failed: {e}')
            continue

    if len(mse_list) == 0:
        raise RuntimeError('In-sample MSE calculation failed for all train_dates.')
    mse_df = pd.concat(mse_list, axis=1)
    mse_df.columns = TRAIN_DATES[:len(mse_list)]
    mse_df['Whole periods'] = mse_df.mean(axis=1)
    return mse_df


def load_oos_mse(horizon, results_dir=None):
    if results_dir is None:
        results_dir = REPO_ROOT / 'results'
    results_dir = Path(results_dir)

    parsed = {}
    for file in os.listdir(results_dir):
        # exclude '..._is_mse.csv' (in-sample) files -- 'mse.csv'.endswith matches
        # both, and the original notebook's naive parsing just skipped these with
        # a warning; excluding them upfront avoids the noise without changing results.
        if file.startswith(horizon) and file.endswith('mse.csv') and not file.endswith('_is_mse.csv'):
            try:
                lambda_part = file.replace(f'{horizon}_', '').replace('_mse.csv', '')
                weight_lambda = float(lambda_part)
            except (ValueError, IndexError) as e:
                print(f"Warning: could not parse lambda from '{file}': {e}")
                continue
            result = pd.read_csv(results_dir / file)
            n_dates = len([c for c in result.columns if c not in ('Unnamed: 0', 'Whole periods')])
            parsed[weight_lambda] = (result, n_dates)

    whole_periods = pd.DataFrame()
    if not parsed:
        return whole_periods

    # A handful of results/<horizon>_<lambda>_mse.csv files are missing test
    # dates relative to the rest of the lambda grid (a stale/partial re-run
    # artifact -- e.g. 12month lambda=0.0/0.3 are missing the earliest window,
    # 2014-01-01; see analysis/outputs/DEVIATIONS.md). Averaging their 'Whole
    # periods' column over a different, smaller set of test dates than every
    # other lambda produces a spurious spike in the MSE-vs-lambda curve that
    # is not present in the notebook. Excluding incomplete files (rather than
    # trusting each file's own possibly-inconsistent mean) reproduces the
    # notebook's actual curve.
    canonical_n = pd.Series([n for _, n in parsed.values()]).mode().iloc[0]

    result = None
    for weight_lambda, (res, n_dates) in sorted(parsed.items()):
        if n_dates != canonical_n:
            print(f"Warning: '{horizon}' lambda={weight_lambda} has {n_dates} test dates "
                  f"(expected {canonical_n}); excluding from OOS MSE curve (see DEVIATIONS.md)")
            continue
        result = res
        whole_periods[weight_lambda] = result['Whole periods']

    if result is not None:
        whole_periods.index = result['Unnamed: 0']
        whole_periods = whole_periods.drop('Consensus average MSE', errors='ignore')
        whole_periods = whole_periods.reindex(sorted(whole_periods.columns), axis=1)
    return whole_periods


def plot_mse(whole_periods, horizon, mode, out_dir):
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 15,
        'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'lines.linewidth': 2, 'lines.markersize': 6,
        'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#2E86AB', '#A23B72']
    ylabel = 'Out-of-Sample MSE' if mode == 'oos' else 'In-Sample MSE'

    ax1.plot(whole_periods.columns, whole_periods.iloc[-1], '--', color=colors[0], linewidth=2,
              markersize=6, markerfacecolor=colors[0], markeredgecolor='black', markeredgewidth=0.5)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('Hyperparameter (lambda)')
    ax1.set_ylabel(ylabel)
    ax1.set_title('Asset Returns', pad=15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(whole_periods.columns, whole_periods.iloc[-2], '--', color=colors[1], linewidth=2,
              markersize=6, markerfacecolor=colors[1], markeredgecolor='black', markeredgewidth=0.5)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('Hyperparameter (lambda)')
    ax2.set_ylabel(ylabel)
    ax2.set_title('Consensus Variables', pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(out_dir / f'{horizon}_{mode}_mse.png', dpi=150)
    plt.close(fig)


def run(horizons=('1month', '3month', '6month', '12month'), run_in_sample_for=('1month', '12month'),
        results_dir=None, checkpoints_dir=None, data_dir=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    oos_results = {}
    for horizon in horizons:
        print(f'Plotting OOS MSE for {horizon} prediction horizon')
        whole_periods = load_oos_mse(horizon, results_dir=results_dir)
        if whole_periods.empty:
            print(f"No valid MSE files found for horizon '{horizon}'")
            continue
        whole_periods.to_csv(out_dir / f'{horizon}_oos_mse.csv')
        plot_mse(whole_periods, horizon, 'oos', out_dir)
        oos_results[horizon] = whole_periods

    is_results = {}
    for horizon in run_in_sample_for:
        lambda_list = get_lambda_list_for_horizon(horizon, results_dir=str(results_dir or REPO_ROOT / 'results'))
        whole_periods = pd.DataFrame()
        for weight_lambda in tqdm(lambda_list, desc=f'In-sample MSE by lambda ({horizon})'):
            try:
                mse_df = calc_in_sample_mse(horizon, weight_lambda, checkpoints_dir=checkpoints_dir, data_dir=data_dir)
                mse_df.to_csv((Path(results_dir) if results_dir else REPO_ROOT / 'results') / f'{horizon}_{weight_lambda}_is_mse.csv')
                whole_periods[weight_lambda] = mse_df['Whole periods']
            except Exception as e:
                print(f'lambda={weight_lambda} calculation failed: {e}')
                traceback.print_exc()
        if not whole_periods.empty:
            whole_periods = whole_periods.reindex(sorted(whole_periods.columns), axis=1)
            whole_periods.to_csv(out_dir / f'{horizon}_is_mse.csv')
            plot_mse(whole_periods, horizon, 'is', out_dir)
            is_results[horizon] = whole_periods

    return {'oos': oos_results, 'is': is_results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizons', nargs='+', default=['1month', '3month', '6month', '12month'])
    parser.add_argument('--in_sample_horizons', nargs='+', default=['1month', '12month'])
    args = parser.parse_args()
    run(horizons=args.horizons, run_in_sample_for=args.in_sample_horizons)
