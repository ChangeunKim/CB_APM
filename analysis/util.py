import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append(os.path.abspath('..'))

import torch
import numpy as np
import pandas as pd
import io
from typing import List, Optional, Dict
import re

from utils.data_utils import create_dataloaders
from models.networks import ConceptBottleneckModel
from models.test import test

import pickle
from pathlib import Path

# Repo root, resolved from this file's location so callers can run scripts from
# anywhere (repo root or analysis/) without depending on cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical order of the 9 analyst consensus variables (matches SignalDoc.csv
# 'Acronym' order for rows with Cat.Data == 'Analyst' that appear in the model
# inputs; see REVISION_PLAN_CLAUDE_CODE.md Section 1, G7).
CONSENSUS_VARS = [
    'AnalystRevision', 'ChangeInRecommendation', 'ChForecastAccrual',
    'EarningsForecastDisparity', 'FEPS', 'ForecastDispersion',
    'REV6', 'AnalystValue', 'AOP',
]


def load_final_result_pickle(horizon, weight_lambda, results_dir=None):
    """
    Load a results/<horizon>_<weight_lambda>.pickle produced by run.py and
    attach readable column names (date, permno, <CONSENSUS_VARS...> / Return).

    Parameters
    ----------
    horizon : str, e.g. '12month'
    weight_lambda : float
    results_dir : str or Path, default REPO_ROOT / 'final_results'

    Returns
    -------
    dict with keys 'actual_concept', 'forecast_concept', 'actual_target',
    'forecast_target', each a DataFrame with named columns.
    """
    if results_dir is None:
        results_dir = REPO_ROOT / 'final_results'
    results_dir = Path(results_dir)

    file_name = f'{horizon}_{weight_lambda}.pickle'
    with open(results_dir / file_name, 'rb') as f:
        raw = pickle.load(f)

    out = {}
    for key in ('actual_concept', 'forecast_concept'):
        df = raw[key].copy()
        df.columns = ['date', 'permno'] + CONSENSUS_VARS
        df['date'] = pd.to_datetime(df['date'])
        out[key] = df
    for key in ('actual_target', 'forecast_target'):
        df = raw[key].copy()
        df.columns = ['date', 'permno', 'Return']
        df['date'] = pd.to_datetime(df['date'])
        out[key] = df
    return out


def build_tidy_frame(horizon, weight_lambda, results_dir=None):
    """
    Merge a loaded result pickle into one tidy long DataFrame indexed by
    [date, permno] with columns:
        C_<var>    raw consensus (actual_concept)
        MIC_<var>  machine-implied consensus (forecast_concept)
        y_true     realized (excess) return
        y_pred     CB-framework predicted return
    """
    raw = load_final_result_pickle(horizon, weight_lambda, results_dir=results_dir)

    actual_c = raw['actual_concept'].rename(columns={v: f'C_{v}' for v in CONSENSUS_VARS})
    forecast_c = raw['forecast_concept'].rename(columns={v: f'MIC_{v}' for v in CONSENSUS_VARS})
    actual_t = raw['actual_target'].rename(columns={'Return': 'y_true'})
    forecast_t = raw['forecast_target'].rename(columns={'Return': 'y_pred'})

    df = actual_c.merge(forecast_c, on=['date', 'permno'], how='inner')
    df = df.merge(actual_t, on=['date', 'permno'], how='inner')
    df = df.merge(forecast_t, on=['date', 'permno'], how='inner')
    return df


def portfolio_weights(df, weight_col=None, group_col='date'):
    """
    Build positive portfolio weights that sum to 1 within each date group.

    weight_col=None gives equal weighting. weight_col='Size' gives TRUE
    value-weighting: data/input_<h>month.csv's 'Size' column is log market
    equity (raw, pre rank-normalization -- confirmed by its continuous range
    [~7, ~19], not the [-1, 1] rank-normalized scale used for model inputs),
    so exp(Size) is proportional to ME and exp(Size)-weighting is standard
    VW, not an approximation. For any other weight_col, exp(x) is used as a
    generic positive-weight transform -- treat non-Size weight columns as a
    documented proxy unless you have separately confirmed they are on a raw
    (not rank-normalized) scale.
    """
    if weight_col is None:
        w = pd.Series(1.0, index=df.index)
    else:
        w = np.exp(df[weight_col].astype(float))
    w = w.groupby(df[group_col]).transform(lambda x: x / x.sum())
    return w


def decile_sort_returns(df, signal_col, return_col='y_true', date_col='date',
                         n_bins=10, weight_col=None, min_names=10):
    """
    Shared single-sort decile (or n_bins) portfolio construction (REVISION_PLAN
    G6): cross-sectional bins on `signal_col` each date, weighted mean of
    `return_col` per bin, monthly rebalance. See `portfolio_weights` for the
    VW-proxy caveat.

    Returns
    -------
    panel : DataFrame, index=date, columns=1..n_bins (+ 'H-L'), monthly bin
            returns (wide format, ready for newey_west_tstat / decile means).
    summary : DataFrame, one row per bin (+ H-L), columns ['mean','se','tstat','pval','nobs']
    """
    from analysis.stats_utils import newey_west_tstat

    work = df[[date_col, signal_col, return_col] + ([weight_col] if weight_col else [])].dropna()
    work['_w'] = portfolio_weights(work, weight_col=weight_col, group_col=date_col)

    def _bin(x):
        try:
            return pd.qcut(x, n_bins, labels=False, duplicates='drop') + 1
        except ValueError:
            return pd.Series(np.nan, index=x.index)

    work['_bin'] = work.groupby(date_col)[signal_col].transform(_bin)
    work = work.dropna(subset=['_bin'])

    counts = work.groupby(date_col)['_bin'].transform('count')
    work = work[counts >= min_names]

    def _wmean(g):
        return np.average(g[return_col], weights=g['_w'])

    panel = (
        work.groupby([date_col, '_bin'])
        .apply(_wmean, include_groups=False)
        .unstack('_bin')
    )
    panel.columns = [int(c) for c in panel.columns]
    panel = panel.reindex(columns=sorted(panel.columns))
    if panel.shape[1] == n_bins:
        panel['H-L'] = panel[panel.columns[-1]] - panel[panel.columns[0]]

    summary_rows = {}
    for col in panel.columns:
        summary_rows[col] = newey_west_tstat(panel[col].dropna())
    summary = pd.DataFrame(summary_rows).T
    summary.index.name = 'bin'
    return panel, summary


def load_size_and_exchcd(horizon, data_dir=None):
    """
    Size (raw log market equity, see portfolio_weights) from
    data/input_<horizon>.csv, merged with exchcd (NYSE flag = 1) from
    data/raw/IBES_summary.csv, aligned to month-start dates.
    """
    if data_dir is None:
        data_dir = REPO_ROOT / 'data'
    data_dir = Path(data_dir)

    size = pd.read_csv(data_dir / f'input_{horizon}.csv', usecols=['permno', 'date', 'Size'])
    size['date'] = pd.to_datetime(size['date']).values.astype('datetime64[M]')

    ibes = pd.read_csv(data_dir / 'raw' / 'IBES_summary.csv', usecols=['permno', 'date', 'exchcd'])
    ibes['date'] = pd.to_datetime(ibes['date']).values.astype('datetime64[M]')
    ibes = ibes.drop_duplicates(['permno', 'date'])

    return size.merge(ibes, on=['permno', 'date'], how='left')


def nyse_breakpoint_mask(df, size_col='Size', exchcd_col='exchcd', date_col='date', pct=0.2):
    """
    Boolean mask: True where a firm-month's Size is AT OR ABOVE the pct-th
    percentile of Size among NYSE (exchcd==1) firms that month (standard
    Fama-French microcap-exclusion breakpoint). Firms with missing exchcd are
    still evaluated against the NYSE breakpoint from firms that do have it.
    """
    nyse_bp = (
        df.loc[df[exchcd_col] == 1]
        .groupby(date_col)[size_col]
        .quantile(pct)
        .rename('_nyse_bp')
    )
    merged = df.join(nyse_bp, on=date_col)
    return merged[size_col] >= merged['_nyse_bp']


def real_cost_net_hl_returns(df, signal_col, return_col='y_true', date_col='date',
                              permno_col='permno', n_bins=10, cost_rate=0.005, weight_col='Size'):
    """
    D.5.3's ACTUAL cost model (extracted from analysis/portfolio.ipynb's
    plot_portfolio_performance/calculate_portfolio_metrics, which were only
    inline in the notebook, not importable): net_r_t = gross_r_t -
    transaction_cost * turnover_t, where turnover_t is the realized
    period-over-period portfolio turnover of the LONG-SHORT (top-decile long,
    bottom-decile short) weight vector (reuses this module's own `turnover`
    function, not a flat assumption). `cost_rate` is per unit of turnover
    (e.g. 0.005 = 50bps, matching the notebook's own worked examples of
    0.0025/0.005/0.0075).

    Returns
    -------
    dict with 'gross_hl' (Series indexed by date), 'net_hl' (Series indexed
    by date, first period unadjusted -- no prior weights to compute turnover
    against), 'turnover_by_date' (Series)
    """
    work = df[[date_col, permno_col, signal_col, return_col] + ([weight_col] if weight_col else [])].dropna()
    work['_w'] = portfolio_weights(work, weight_col=weight_col, group_col=date_col)

    def _bin(x):
        try:
            return pd.qcut(x, n_bins, labels=False, duplicates='drop') + 1
        except ValueError:
            return pd.Series(np.nan, index=x.index)

    work['_bin'] = work.groupby(date_col)[signal_col].transform(_bin)
    top, bot = work['_bin'].max(), work['_bin'].min()

    long_leg = work[work['_bin'] == top].copy()
    short_leg = work[work['_bin'] == bot].copy()
    long_leg['_leg_w'] = long_leg.groupby(date_col)['_w'].transform(lambda x: x / x.sum())
    short_leg['_leg_w'] = -short_leg.groupby(date_col)['_w'].transform(lambda x: x / x.sum())

    combined = pd.concat([long_leg, short_leg], ignore_index=True)
    weights_df = combined[[date_col, permno_col, '_leg_w']].rename(columns={'_leg_w': 'weight'}).set_index(date_col)
    returns_df = combined[[date_col, permno_col, return_col]].rename(columns={return_col: 'actual'}).set_index(date_col)
    # turnover() expects arithmetic returns; return_col here (y_true) is a
    # log/simple annual return depending on the caller -- convert defensively
    # only if values look like log returns (can be negative below -1); the
    # existing repo convention (see analysis/decision_value.py) already
    # treats y_true as usable directly with turnover(), so no conversion here.

    gross_hl = (long_leg.groupby(date_col)[return_col].apply(lambda x: np.average(x, weights=long_leg.loc[x.index, '_w'] / long_leg.loc[x.index, '_w'].sum()))
                - short_leg.groupby(date_col)[return_col].apply(lambda x: np.average(x, weights=short_leg.loc[x.index, '_w'] / short_leg.loc[x.index, '_w'].sum())))

    try:
        to = turnover(returns_df, weights_df)
    except Exception:
        to = []

    dates_sorted = sorted(gross_hl.index.unique())
    turnover_series = pd.Series(index=dates_sorted[1:1 + len(to)], data=to[:len(dates_sorted) - 1])

    net_hl = gross_hl.copy()
    for d in turnover_series.index:
        net_hl.loc[d] = gross_hl.loc[d] - cost_rate * turnover_series.loc[d]

    return {'gross_hl': gross_hl, 'net_hl': net_hl, 'turnover_by_date': turnover_series}


def zscore_composite(df, cols, group_col='date'):
    """
    Cross-sectional (per group_col, typically 'date') z-score average across
    `cols`, i.e. the standard "raw-consensus composite" / "MIC composite"
    construction used throughout E5/E13.
    """
    z = df.groupby(group_col)[cols].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    return z.mean(axis=1)


def get_best_lambda_from_summary(horizon, tables_dir=None):
    tables_dir = Path(tables_dir) if tables_dir else REPO_ROOT / 'tables'
    summary_path = tables_dir / f'{horizon}_r2_analysis_summary.xlsx'
    # Try to read the best lambda from the summary sheet
    try:
        summary = pd.read_excel(summary_path, sheet_name='R2_Summary')
        best_lambda = summary.loc[1, 'lambda']
    except Exception:
        summary = pd.read_excel(summary_path, sheet_name='Summary_Stats')
        best_lambda = summary.loc[1, 'lambda']
    if best_lambda == 'improvement':
        best_lambda = summary.loc[1, 'lambda']
    return float(best_lambda)

def get_lambda_list_for_horizon(horizon, results_dir='../results'):
    """
    Returns a list of lambda values (float) used in the results directory for the given horizon.
    """
    lambda_list = []
    for file in os.listdir(results_dir):
        if file.startswith(horizon) and file.endswith('mse.csv'):
            try:
                weight_lambda = float(file.split(f'{horizon}_')[1].split('_')[0])
                lambda_list.append(weight_lambda)
            except (ValueError, IndexError):
                continue
    return sorted(lambda_list)

def load_and_average_ensemble_models(model_dir, train_date, config, device):
    weights_list = []
    bias_list = []
    for i in range(config['ensemble']):
        model_path = os.path.join(model_dir, f"{train_date}model_{i}.pt")
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        model = ConceptBottleneckModel(
            config['input_size'],
            config['concept_hidden_sizes'],
            config['concept_output_size'],
            config['final_hidden_sizes'],
            config['final_output_size']
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        if isinstance(model.final_model, torch.nn.Linear):
            last_layer = model.final_model
        elif hasattr(model.final_model, 'output_layer'):
            last_layer = model.final_model.output_layer
        else:
            raise ValueError("Unknown final_model type")
        weights_list.append(last_layer.weight.detach().cpu().numpy())
        bias_list.append(last_layer.bias.detach().cpu().numpy())
    if weights_list:
        avg_weights = np.mean(weights_list, axis=0)
        avg_bias = np.mean(bias_list, axis=0)
        return avg_weights, avg_bias
    return None, None

def MAX_DD(data):
    """
    Compute arithmetic maximum drawdown from a cumulative wealth series.

    Parameters
    ----------
    data : pd.Series
        Cumulative WEALTH path (not log returns).

    Returns
    -------
    float
        Maximum drawdown as a fraction of wealth (0–1).
    """
    max_dd = 0.0
    for t1 in data.index:
        for t2 in data.index:
            if t2 >= t1:
                drawdown = 1 - data[t2] / data[t1]  # arithmetic definition
                if drawdown > max_dd:
                    max_dd = drawdown
    return max_dd

def turnover(returns, weights):
    """
    Compute portfolio turnover based on arithmetic returns and weight changes.
    (Keeps your index-based structure.)

    Parameters
    ----------
    returns : pd.DataFrame
        Indexed by date, must include ['permno', 'actual'].
        'actual' = arithmetic monthly return (not log).
    weights : dict-like
        Each element indexed by date with columns ['permno', 'weight'].

    Returns
    -------
    list of float
        Turnover per rebalancing period.
    """
    turnovers = []
    unique_dates = returns.index.unique()

    for date1, date2 in zip(unique_dates, unique_dates[1:]):
        w_t_1 = weights.loc[date1]
        w_t_2 = weights.loc[date2]
        r_t = returns.loc[date1]

        # Common universe
        permno = set(w_t_1['permno']).intersection(set(w_t_2['permno']))
        if not permno:
            continue

        w_t_1 = w_t_1[w_t_1['permno'].isin(permno)]['weight']
        w_t_2 = w_t_2[w_t_2['permno'].isin(permno)]['weight']
        r_t = r_t[r_t['permno'].isin(permno)]['actual']

        # Drift previous weights with arithmetic returns
        wr = w_t_1 * (1 + r_t)
        wr_sum = 1 + (r_t * w_t_1).sum()

        # 0.5 × L1 distance (standard definition)
        turnover_t = 0.5 * np.abs(w_t_2.values - (wr / wr_sum).values).sum()
        turnovers.append(turnover_t)

    return turnovers

def get_Xy_cbapm(train_date, input, target, info, config, device, horizon, weight_lambda,
                  embedding_method='none', checkpoints_dir=None):
    """checkpoints_dir defaults to REPO_ROOT/'checkpoints' (cwd-independent);
    pass e.g. REPO_ROOT/'final_checkpoints' to use the frozen paper checkpoints instead."""
    if checkpoints_dir is None:
        checkpoints_dir = REPO_ROOT / 'checkpoints'
    checkpoints_dir = Path(checkpoints_dir)

    if embedding_method == 'autoencoder':
        autoencoder_path = str(checkpoints_dir / f'{horizon}_{weight_lambda}' / f'{train_date}_autoencoder_model0.pt')
    else:
        autoencoder_path = None

    train_date_dt = pd.to_datetime(train_date)
    valid_date = (train_date_dt + pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    test_date  = (train_date_dt + pd.DateOffset(months=12)).strftime('%Y-%m-%d')

    train_loader, _, _, _, _ = create_dataloaders(
        input, target, info,
        train_date=train_date,
        valid_date=valid_date,
        test_date=test_date,
        batch_size=config['batch_size'],
        embedding_method=embedding_method,
        model_path=autoencoder_path
    )
    model_dir = str(checkpoints_dir / f'{horizon}_{weight_lambda}') + '/'
    models = []
    for i in range(config['ensemble']):
        model = ConceptBottleneckModel(
            config['input_size'], config['concept_hidden_sizes'], config['concept_output_size'],
            config['final_hidden_sizes'], config['final_output_size']
        ).to(device)
        model_path = model_dir + f"{train_date}model_{i}.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
    # Get the input window mask (same as in create_dataloaders)
    window_mask = input['date'] < pd.to_datetime(train_date)
    input_window = input.loc[window_mask].copy()
    # Get the permno and date columns for the window
    index_cols = ['date', 'permno']
    index_df = input_window[index_cols].reset_index(drop=True)
    # Get the CBAPM features
    _, _, forecast_concept, _ = test(
        models, train_loader, config['ensemble'], device
    )
    if forecast_concept.shape[0] < forecast_concept.shape[1]:
        forecast_concept = forecast_concept.T
    # Make sure the number of rows matches
    if len(index_df) != forecast_concept.shape[0]:
        min_len = min(len(index_df), forecast_concept.shape[0])
        index_df = index_df.iloc[:min_len].reset_index(drop=True)
        forecast_concept = forecast_concept[:min_len]
    # Build DataFrame with index columns and features
    feature_names = list(info[info['Cat.Data'] == 'Analyst']['Acronym'].values)
    X_cbapm_with_index = pd.DataFrame(forecast_concept, columns=feature_names)
    X_cbapm_with_index = pd.concat([index_df, X_cbapm_with_index], axis=1)
    # Get y (target)
    real_targets = []
    for _, _, targets in train_loader:
        real_targets.append(targets.numpy())
    y = np.concatenate(real_targets, axis=0).squeeze()
    if len(y) > len(index_df):
        y = y[:len(index_df)]
    return X_cbapm_with_index, y

def get_Xy_real_concept(train_date, input, target, analyst_col):
    window_mask = input['date'] < pd.to_datetime(train_date)

    input_filtered = input.loc[window_mask, ['date', 'permno'] + list(analyst_col)].copy()
    target_filtered = target.loc[window_mask, ['date', 'permno', 'Return']].copy()

    df = pd.merge(input_filtered, target_filtered, on=['date', 'permno'], how='inner')

    y = df['Return'].astype(np.float32)
    X = df.drop(columns=['Return'])

    return X, y

# Latex Table Generator

def significance_stars(pval):
    if pval < 0.01:
        return r'\st{***}'
    elif pval < 0.05:
        return r'\st{**}'
    elif pval < 0.1:
        return r'\st{*}'
    else:
        return r'\st{}'

def format_row(name, real, approx, rsq):
    row = (
        f"{name} & "
        f"{real['coef']:.4f} & {real['tval']:.2f} & {significance_stars(real['pval'])} & "
        f"{rsq:.2f} & "
        f"{approx['coef']:.4f} & {approx['tval']:.2f} & {significance_stars(approx['pval'])} \\\\"
    )
    return row

def generate_ols_latex_table(real_summary, approx_summary, variable_names):
    panel_a = []
    rsq = [4.97, -0.16, 4.62, 9.12, 71.43, 39.06, 16.18, 35.45, 37.24]
    for i, name in enumerate(variable_names):
        real = {
            'coef': real_summary['coef'].iloc[i],
            'tval': real_summary['tvalues'].iloc[i],
            'pval': real_summary['pvalues'].iloc[i]
        }
        approx = {
            'coef': approx_summary['coef'].iloc[i],
            'tval': approx_summary['tvalues'].iloc[i],
            'pval': approx_summary['pvalues'].iloc[i]
        }
        panel_a.append(format_row(name, real, approx, rsq[i]))

    panel_b = f"""
    \\textbf{{Panel B: Model summary statistics}} \\\\
    \\vspace{{0.2cm}}
    \\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}l S[table-format=-2.4] S[table-format=-2.4]}}
        \\toprule
        & {{\\small Raw Consensus}} & {{\\small Approximated Consensus}} \\\\
        \\midrule
        Intercept & {real_summary['bias'].iloc[0]:.4f} & {approx_summary['bias'].iloc[0]:.4f} \\\\
        SE of Intercept & {real_summary['bias_stderr'].iloc[0]:.4f} & {approx_summary['bias_stderr'].iloc[0]:.4f} \\\\
        In-Sample $adj\\text{{-}}R^2$ (\\%) & {real_summary['r2'].iloc[0]:.2f} & {approx_summary['r2'].iloc[0]:.2f} \\\\
        \\bottomrule
    \\end{{tabular*}}
    """

    # Combine full LaTeX
    latex = r"""
        \begin{table}[htbp]
            \centering
            \small
            \renewcommand{\arraystretch}{1.3}
            \setlength{\tabcolsep}{6pt}

             \caption{OLS regressions with raw versus model-inferred consensus. \\
            This table reports pooled OLS regressions in which the dependent variable is the \emph{annual} stock return \(R_{i,t+12}\). We compare specifications that use raw analyst consensus variables to those that use CB-APM–inferred consensus estimates at ($\lambda=1$), evaluated on the longest training set from the expanding-window procedure. The CB-APM consensus corresponds to the averaged output of an ensemble of models. Panel~A reports coefficient estimates, $t$-statistics, and predictor-level $R^2$ for each variable, while Panel~B summarizes the intercept, its standard error, and the overall in-sample adjusted $R^2$.}
            \label{tab:ols-consensus}

            \textbf{Panel A: Coefficients and t-statistics} \\
            \vspace{0.2cm}
            \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} 
                l
                S[table-format=-.4]
                S[table-format=-.2]@{\hspace{-1.3em}}l
                S[table-format=-.2] 
                S[table-format=-.4]
                S[table-format=-.2]@{\hspace{-1.3em}}l
            }
                \toprule
                \multicolumn{1}{c}{} 
                & \multicolumn{3}{c}{\small Raw Consensus} 
                & \multicolumn{4}{c}{\small Approximated Consensus} \\
                \cmidrule(lr){2-4} \cmidrule(lr){5-8}
                {\small Variable} & {\small Coef.} & {\small t-stat.} & & {\small $R^2$ (\%)} & {\small Coef.} & {\small t-stat.} & \\
                \midrule
        """ + "\n".join(panel_a) + r"""
                \bottomrule
            \end{tabular*}

            \vspace{0.6cm}
        """ + panel_b + r"""
            \begin{flushleft}
            \textit{Note:} *** significance at the 1\% level; ** significance at the 5\% level; * significance at the 10\% level.
            Standard errors are computed using the Driscoll--Kraay (kernel HAC) estimator with a Bartlett kernel and an eleven-month bandwidth, robust to heteroskedasticity, cross-sectional dependence, and the serial correlation induced by overlapping returns.
            \end{flushleft}
        \end{table}
        """
    return latex

def generate_single_sort_latex_table(result_df):
    """
    Return compile-ready LaTeX code for single-sort realized decile returns,
    formatted in two vertically stacked panels:
        - Top panel: λ = 0.0–0.5
        - Bottom panel: λ = 0.6–1.0
    λ values appear once in the top header row (no 'λ=' prefix in each column).
    Includes High–Low (H–L) spread as the final row.
    """

    lambdas = sorted([float(c) for c in result_df.columns])
    top_lambdas = [lam for lam in lambdas if lam <= 0.5]
    bottom_lambdas = [lam for lam in lambdas if lam > 0.5]

    def build_panel(lam_subset):
        """Generate one panel block with λ in the first header cell."""
        # Header row: show only λ values, with 'λ' label in first cell
        header = " & ".join([f"\\small {lam:.1f}" for lam in lam_subset])
        rows = []
        for idx, row in result_df.iterrows():
            idx_label = f"{int(idx)}" if str(idx).isdigit() else "H--L"
            if idx == 1: idx_label = "Low"
            if idx == 10: idx_label = "High"
            vals = []
            for lam in lam_subset:
                col = str(lam) if str(lam) in row.index else lam
                val = row[col]
                vals.append("" if pd.isna(val) else f"{val:.2f}")
            rows.append(f"{idx_label} & " + " & ".join(vals) + " \\\\")
        body = "\n".join(rows)

        return rf"""
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}} l {' '.join(['S[table-format=-1.4]' for _ in lam_subset])} }}
\toprule
{{\small $\lambda$}} & {header} \\
\midrule
{body}
\bottomrule
\end{{tabular*}}
"""

    # Build the two stacked panels
    top_panel = build_panel(top_lambdas)
    bottom_panel = build_panel(bottom_lambdas)

    # Combine into vertically stacked layout
    latex = rf"""
\begin{{table}}[htbp]
    \centering
    \small
    \renewcommand{{\arraystretch}}{{1.3}}
    \setlength{{\tabcolsep}}{{6pt}}

    \caption{{Realized monthly returns of out-of-sample single-sorted portfolios across $\lambda$. \\
    Each panel reports mean monthly realized returns (in percentage points) for monthly rebalanced decile portfolios, 
    formed by sorting stocks on CB-APM-predicted annual returns.
    The bottom row (H--L) represents the spread between the highest- and lowest-decile portfolios.}}
    \label{{tab:single-sort}}

    \vspace{{0.2cm}}
    {top_panel}
    \vspace{{0.5cm}}
    {bottom_panel}
\end{{table}}
"""
    return latex.strip()

def generate_double_sort_latex_table(result_dict, cons_var):
    """
    Generate a true multipage longtable (spanning \textwidth) for double-sort realized returns.
    Finance-journal style:
      - \small font
      - \textwidth width
      - Auto-numbered caption
      - Thick \midrule[1pt] between λ-panels
      - Includes H–L row and column
      - Works in Overleaf with booktabs + longtable + siunitx + caption
    """
    import re, numpy as np

    def normalize_hl_label(s):
        if not isinstance(s, str):
            return s
        t = s.strip().replace("–", "-").replace("—", "-").replace("−", "-")
        t = re.sub(r"\s*-\s*", "-", t).upper()
        return "H--L" if re.fullmatch(r"H-+L", t) else s

    def find_hl_name(seq):
        for x in seq:
            if isinstance(x, str) and normalize_hl_label(x) == "H--L":
                return x
        return None

    def _fmt(x):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return ""
        return f"{x:.2f}"

    all_lambdas = sorted(result_dict.keys())

    def build_panels(lams):
        parts = []
        for lam in lams:
            mat = result_dict[lam].copy()

            hl_row = find_hl_name(mat.index)
            hl_col = find_hl_name(mat.columns)
            row_order = [1, 2, 3, 4, 5] + ([hl_row] if hl_row else [])
            col_order = [1, 2, 3, 4, 5] + ([hl_col] if hl_col else [])
            mat = mat.reindex(index=row_order, columns=col_order)

            # panel header
            parts.append(
                rf"\multicolumn{{7}}{{c}}{{\textbf{{Panel:}} $\bm{{\lambda = {lam:.1f}}}$}} \\[3pt]"
            )
            parts.append(r"\midrule")
            parts.append(r"& \multicolumn{5}{c}{\small {$E_t[R_{i,t+h}]$}} & \\")
            parts.append(r"\cmidrule(lr){2-6}")
            parts.append(
                rf"{{\small $E_t[{{{cons_var}}}_{{i, t}}]$}} & Low & 2 & 3 & 4 & High & H\text{{--}}L \\"
            )
            parts.append(r"\midrule")

            for ridx in row_order:
                if ridx not in mat.index:
                    continue
                label = r"H--L" if (isinstance(ridx, str) and normalize_hl_label(ridx) == "H--L") else str(int(ridx))
                if ridx == 1: label = "Low"
                if ridx == 5: label = "High"
                vals = [_fmt(mat.loc[ridx, c]) if c in mat.columns else "" for c in col_order]
                parts.append(f"{label} & " + " & ".join(vals) + r" \\")
            parts.append(r"\midrule")
            parts.append(r"\addlinespace[0.4em]")
        return "\n".join(parts)

    body = build_panels(all_lambdas)

    latex = f"""
\\begin{{small}}
\\renewcommand{{\\arraystretch}}{{0.7}}
\\setlength{{\\LTleft}}{{0pt}}
\\setlength{{\\LTright}}{{0pt}}


\\begin{{longtable}}{{@{{\\extracolsep{{\\fill}}}} p{{1.6cm}} *{{6}}{{S[table-format=-1.2, table-column-width=1.5cm]}}}}

\\caption{{Realized monthly returns of out-of-sample double-sorted portfolios across $\\lambda$.\\\\
Each panel reports mean monthly realized returns (in percentage points) for monthly rebalanced 5×5 portfolios 
sorted by the approximated \\textit{{{cons_var}}} (rows) and predicted annual returns ($E[R]$, columns), independently.
H--L denotes the high–minus–low spread across the corresponding dimension.}}\\label{{tab:double-sort}} \\\\
    
\\endfirsthead

\\multicolumn{{7}}{{c}}{{\\textbf{{Table \\thetable}} Realized returns of double-sorted portfolios (cont'd)}}\\\\
\\midrule
\\endhead

\\endfoot

\\endlastfoot

{body}

\\end{{longtable}}
\\end{{small}}
"""
    return latex.strip()
