"""
Refactored from analysis/coefficient.ipynb (kept as the source notebook; this
module is the maintained, runnable version). Two analyses:

  1. Linear-head coefficient dot plot across expanding windows at a chosen
     lambda, colored by that window's OOS R2 (coefficient stability/sign
     visualization -- same purpose as analysis/interpretability.py's I1, but
     using the literal averaged network weights rather than a projection;
     see analysis/outputs/DEVIATIONS.md for why those two can disagree).
  2. Pooled-OLS panel regressions (Driscoll-Kraay/kernel-HAC SEs) of returns
     on (a) the CB-framework's inferred consensus (MIC) and (b) the raw
     consensus, for a LaTeX comparison table.

DEPENDENCIES NOT in requirements.txt (added there by this refactor):
`linearmodels` (PanelOLS) for run_ols_regression, needed only by
rolling_regression/rolling_concept_regression_with_cbapm/
rolling_real_concept_regression. The coefficient dot-plot functions
(extract_all_last_layer_coefficients, plot_coeff_ranges, coefficients_analysis)
do NOT need it. The original notebook also imported `shap` purely for a
diverging colormap (red_blue) -- replaced here with matplotlib's built-in
'RdBu_r' to drop that dependency entirely (same diverging-colormap purpose).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import get_config
from models.metrics import r2_score
from analysis.util import (
    REPO_ROOT, get_best_lambda_from_summary, get_lambda_list_for_horizon,
    load_and_average_ensemble_models, get_Xy_cbapm, get_Xy_real_concept,
    generate_ols_latex_table,
)

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'coefficient_analysis'

TRAIN_DATES = ['2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01',
               '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']


def extract_all_last_layer_coefficients(horizon, results_dir=None, checkpoints_dir=None, device=None):
    """
    Iterates over all weight_lambda and train_date for a given horizon, loads
    all ensemble members, averages the last layer weights and bias, and
    returns a list of dicts: {lambda, train_date, weights, bias}.
    """
    results_dir = Path(results_dir) if results_dir else REPO_ROOT / 'results'
    checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else REPO_ROOT / 'checkpoints'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_coeffs = []
    lambda_list = set(get_lambda_list_for_horizon(horizon, results_dir=str(results_dir)))

    for weight_lambda in lambda_list:
        config = get_config(weight_lambda)
        for train_date in TRAIN_DATES:
            model_dir = str(checkpoints_dir / f'{horizon}_{weight_lambda}')
            avg_weights, avg_bias = load_and_average_ensemble_models(model_dir, train_date, config, device)
            if avg_weights is not None:
                all_coeffs.append({'lambda': weight_lambda, 'train_date': train_date,
                                    'weights': avg_weights, 'bias': avg_bias})
    return all_coeffs


def plot_coeff_ranges(all_coeffs, variable_names, best_lambda, r2_by_period, out_dir, horizon):
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 15,
        'axes.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 6, 'ytick.major.size': 6,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'lines.linewidth': 2, 'lines.markersize': 7,
        'grid.alpha': 0.3, 'grid.linewidth': 0.7,
    })

    cmap = plt.cm.RdBu_r  # diverging colormap; replaces the original shap.plots.colors red_blue
    weights = np.array([d['weights'] for d in all_coeffs if d['lambda'] == best_lambda])
    if weights.ndim == 3:
        weights = weights[:, -1, :]  # last output (asset return)
    weights_T = weights.T  # (n_vars, n_dates)

    if weights_T.shape[1] != len(r2_by_period):
        # checkpoints/ can have more expanding-window checkpoints than results/
        # has result columns (a pre-existing pipeline-drift issue, not
        # introduced by this refactor -- see analysis/outputs/DEVIATIONS.md).
        # Align on the END (most recent windows), which is where the two
        # series actually correspond; the extra checkpoint(s), if any, are
        # from the earliest window(s).
        n = min(weights_T.shape[1], len(r2_by_period))
        print(f'WARNING: {weights_T.shape[1]} coefficient windows vs {len(r2_by_period)} R2-by-period '
              f'values -- truncating both to the last {n} (see analysis/outputs/DEVIATIONS.md).')
        weights_T = weights_T[:, -n:]
        r2_by_period = r2_by_period[-n:]

    fig, ax = plt.subplots(figsize=(11, 6))
    norm = plt.Normalize(vmin=np.min(r2_by_period), vmax=np.max(r2_by_period))
    for i, var in enumerate(variable_names):
        colors = cmap(norm(r2_by_period))
        ax.scatter(weights_T[i], np.full_like(weights_T[i], i + 1), color=colors, s=60, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Coefficient Value', fontsize=14)
    ax.set_ylabel('Consensus Variables', fontsize=14)
    ax.set_yticks(np.arange(1, len(variable_names) + 1))
    ax.set_yticklabels(variable_names, fontsize=12)
    ax.grid(axis='x', alpha=0.3, linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(f'Coefficient Dot Plot (lambda = {best_lambda})')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Out-of-Sample R2', fontsize=12)

    fig.tight_layout()
    fig.savefig(out_dir / f'{horizon}_coeff_dotplot_lambda{best_lambda}.png', dpi=150)
    plt.close(fig)


def coefficients_analysis(horizon, best_lambda=None, results_dir=None, checkpoints_dir=None,
                           tables_dir=None, data_dir=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
    tables_dir = Path(tables_dir) if tables_dir else REPO_ROOT / 'tables'

    input_df = pd.read_csv(data_dir / f'input_{horizon}.csv')
    input_df['date'] = pd.to_datetime(input_df['date'])
    signal_info = pd.read_csv(data_dir / 'info' / 'SignalDoc.csv')
    info = signal_info[signal_info['Acronym'].isin(input_df.columns)]
    variable_names = info[info['Cat.Data'] == 'Analyst']['LongDescription'].values

    all_coeffs = extract_all_last_layer_coefficients(horizon, results_dir=results_dir, checkpoints_dir=checkpoints_dir)

    if best_lambda is None:
        best_lambda = get_best_lambda_from_summary(horizon, tables_dir=tables_dir)

    r2_path = tables_dir / f'{horizon}_r2_analysis_summary.xlsx'
    try:
        r2_df = pd.read_excel(r2_path, sheet_name='Period_Comparison')
    except ImportError:
        # openpyxl not installed -- fall back to the per-sheet csv folder
        # analysis/r_squared_analysis.py::save_r2_results_to_excel writes in
        # this same situation (see analysis/outputs/DEVIATIONS.md).
        csv_fallback = r2_path.with_suffix('') / 'Period_Comparison.csv'
        if not csv_fallback.exists():
            raise
        r2_df = pd.read_csv(csv_fallback)
    r2_by_period = r2_df['Best_R2'].values

    plot_coeff_ranges(all_coeffs, variable_names, best_lambda, r2_by_period, out_dir, horizon)
    return all_coeffs, best_lambda


# ---------------------------------------------------------------------------
# Panel-regression comparison (real vs. CB-framework-inferred consensus).
# Requires `linearmodels` (not in requirements.txt before this refactor --
# added). Import is deferred to first use so the coefficient dot-plot
# functions above work even without it installed.
# ---------------------------------------------------------------------------

def run_ols_regression(X, y, cov_type='kernel', bandwidth=11, kernel='bartlett'):
    """
    Pooled OLS on overlapping 12-month returns with overlap-robust inference:
    no firm/time fixed effects; Driscoll-Kraay / kernel-HAC SEs (Bartlett
    kernel, bandwidth=11, matching an MA(11) process from 12m overlapping
    returns).
    """
    from linearmodels.panel import PanelOLS
    from sklearn.metrics import r2_score as sk_r2_score, mean_squared_error

    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame with 'date' and 'permno' columns.")
    if 'date' not in X or 'permno' not in X:
        raise ValueError("X must contain 'date' and 'permno' columns.")

    if isinstance(y, np.ndarray):
        y = pd.Series(y, index=X.index, name='y')
    elif isinstance(y, pd.Series):
        y = y.rename('y')
    else:
        raise ValueError('y must be a numpy array or pandas Series.')

    df = X.copy()
    df['y'] = y
    id_cols = ['permno', 'date']
    reg_cols = [c for c in df.columns if c not in (id_cols + ['y'])]
    df = df.dropna(subset=['y'] + reg_cols)
    if df.empty:
        return {
            'coef': np.nan, 'bias': np.nan, 'r2': np.nan, 'mse': np.nan,
            'pvalues': np.nan, 'bias_pvalue': np.nan, 'tvalues': np.nan,
            'bias_tvalue': np.nan, 'stderr': np.nan, 'bias_stderr': np.nan, 'n_samples': 0,
        }

    df['permno'] = df['permno'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['permno', 'date']).sort_index()
    df.index = df.index.set_names(['permno', 'date'])
    df[reg_cols + ['y']] = df[reg_cols + ['y']].astype(np.float32)

    y_dep = df['y'].copy()
    X_exog = df[reg_cols].copy()
    if 'const' not in X_exog.columns:
        X_exog = X_exog.assign(const=np.float32(1.0))
    else:
        X_exog['const'] = X_exog['const'].astype(np.float32)

    y_dep.index = y_dep.index.set_names(['permno', 'date'])
    X_exog.index = X_exog.index.set_names(['permno', 'date'])
    common_idx = y_dep.index.intersection(X_exog.index)
    y_dep = y_dep.loc[common_idx]
    X_exog = X_exog.loc[common_idx]

    model = PanelOLS(y_dep, X_exog, entity_effects=False, time_effects=False)
    if cov_type == 'kernel':
        results = model.fit(cov_type='kernel', kernel=kernel, bandwidth=bandwidth, debiased=True)
    elif cov_type == 'clustered':
        results = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    else:
        raise ValueError("cov_type must be 'kernel' or 'clustered'.")

    y_pred = results.fitted_values
    try:
        y_pred.index = y_pred.index.set_names(['permno', 'date'])
    except Exception:
        pass
    y_pred = y_pred.reindex(y_dep.index)
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.iloc[:, 0] if y_pred.shape[1] == 1 else y_pred.get('fitted', y_pred.iloc[:, 0])

    y_true_np = y_dep.to_numpy(dtype=np.float32)
    y_pred_np = y_pred.to_numpy(dtype=np.float32)
    r2 = np.float32(sk_r2_score(y_true_np, y_pred_np))
    mse = np.float32(mean_squared_error(y_true_np, y_pred_np))

    params = results.params.astype(np.float32)
    stderr = results.std_errors.reindex(params.index).astype(np.float32)
    tstats = results.tstats.reindex(params.index).astype(np.float32)
    pvals = results.pvalues.reindex(params.index).astype(np.float32)

    bias = float(params.get('const', np.nan))
    bias_stderr = float(stderr.get('const', np.nan))
    bias_tvalue = float(tstats.get('const', np.nan))
    bias_pvalue = float(pvals.get('const', np.nan))

    coef = params.drop(labels=['const'], errors='ignore')
    se = stderr.drop(labels=['const'], errors='ignore')
    tvals = tstats.drop(labels=['const'], errors='ignore')
    pvalues = pvals.drop(labels=['const'], errors='ignore')

    return {
        'coef': coef, 'bias': bias, 'r2': r2, 'mse': mse,
        'pvalues': pvalues, 'bias_pvalue': bias_pvalue,
        'tvalues': tvals, 'bias_tvalue': bias_tvalue,
        'stderr': se, 'bias_stderr': bias_stderr, 'n_samples': int(len(common_idx)),
    }


def rolling_regression(horizon, get_Xy_for_window, get_Xy_args=None, train_date='2020-01-01'):
    get_Xy_args = get_Xy_args or {}
    X, y = get_Xy_for_window(train_date, **get_Xy_args)
    reg_result = run_ols_regression(X, y)
    reg_result['train_date'] = train_date
    return pd.DataFrame(reg_result)


def rolling_concept_regression_with_cbapm(horizon, weight_lambda, embedding_method,
                                           data_dir=None, checkpoints_dir=None, device=None):
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_df = pd.read_csv(data_dir / f'input_{horizon}.csv')
    target_df = pd.read_csv(data_dir / f'target_{horizon}.csv')
    input_df['date'] = pd.to_datetime(input_df['date'])
    target_df['date'] = pd.to_datetime(target_df['date'])
    signal_info = pd.read_csv(data_dir / 'info' / 'SignalDoc.csv')
    info = signal_info[signal_info['Acronym'].isin(input_df.columns)]
    config = get_config(weight_lambda)
    get_Xy_args = dict(input=input_df, target=target_df, info=info, config=config, device=device,
                        horizon=horizon, weight_lambda=weight_lambda, embedding_method=embedding_method)
    if checkpoints_dir is not None:
        get_Xy_args['checkpoints_dir'] = checkpoints_dir
    return rolling_regression(horizon, get_Xy_cbapm, get_Xy_args)


def rolling_real_concept_regression(horizon, data_dir=None):
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
    input_df = pd.read_csv(data_dir / f'input_{horizon}.csv')
    target_df = pd.read_csv(data_dir / f'target_{horizon}.csv')
    input_df['date'] = pd.to_datetime(input_df['date'])
    target_df['date'] = pd.to_datetime(target_df['date'])
    signal_info = pd.read_csv(data_dir / 'info' / 'SignalDoc.csv')
    info = signal_info[signal_info['Acronym'].isin(input_df.columns)]
    analyst_col = info[info['Cat.Data'] == 'Analyst']['Acronym'].values
    get_Xy_args = dict(input=input_df, target=target_df, analyst_col=analyst_col)
    return rolling_regression(horizon, get_Xy_real_concept, get_Xy_args)


def run(horizon='12month', weight_lambda=1.0, out_dir=None, data_dir=None,
        results_dir=None, checkpoints_dir=None, tables_dir=None, run_panel_regression=True):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Plotting coefficients for {horizon} prediction horizon')
    all_coeffs, best_lambda = coefficients_analysis(
        horizon, best_lambda=weight_lambda, results_dir=results_dir,
        checkpoints_dir=checkpoints_dir, tables_dir=tables_dir, data_dir=data_dir, out_dir=out_dir)

    result = {'all_coeffs': all_coeffs, 'best_lambda': best_lambda}

    if run_panel_regression:
        try:
            approx_summary = rolling_concept_regression_with_cbapm(
                horizon, weight_lambda, 'autoencoder', data_dir=data_dir, checkpoints_dir=checkpoints_dir)
            real_summary = rolling_real_concept_regression(horizon, data_dir=data_dir)

            data_dir_r = Path(data_dir) if data_dir else REPO_ROOT / 'data'
            input_df = pd.read_csv(data_dir_r / f'input_{horizon}.csv')
            signal_info = pd.read_csv(data_dir_r / 'info' / 'SignalDoc.csv')
            info = signal_info[signal_info['Acronym'].isin(input_df.columns)]
            variable_names = info[info['Cat.Data'] == 'Analyst']['LongDescription'].values

            latex_table = generate_ols_latex_table(real_summary, approx_summary, variable_names)
            (out_dir / f'{horizon}_ols_comparison.tex').write_text(latex_table, encoding='utf-8')
            approx_summary.to_csv(out_dir / f'{horizon}_approx_concept_regression.csv')
            real_summary.to_csv(out_dir / f'{horizon}_real_concept_regression.csv')
            result.update({'approx_summary': approx_summary, 'real_summary': real_summary, 'latex_table': latex_table})
        except ImportError as e:
            print(f'Panel regression skipped -- missing dependency: {e}')

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    parser.add_argument('--lambda_', dest='weight_lambda', type=float, default=1.0)
    parser.add_argument('--no_panel_regression', action='store_true')
    args = parser.parse_args()
    run(horizon=args.horizon, weight_lambda=args.weight_lambda,
        run_panel_regression=not args.no_panel_regression)
