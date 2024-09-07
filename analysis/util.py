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

def get_best_lambda_from_summary(horizon):
    summary_path = f'../tables/{horizon}_r2_analysis_summary.xlsx'
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

def get_Xy_cbapm(train_date, input, target, info, config, device, horizon, weight_lambda, embedding_method='none'):
    if embedding_method == 'autoencoder':
        autoencoder_path = f'../checkpoints/{horizon}_{weight_lambda}/{train_date}_autoencoder_model0.pt'
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
    model_dir = f'../checkpoints/{horizon}_{weight_lambda}/'
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
            'coef': real_summary['coef'][i],
            'tval': real_summary['tvalues'][i],
            'pval': real_summary['pvalues'][i]
        }
        approx = {
            'coef': approx_summary['coef'][i],
            'tval': approx_summary['tvalues'][i],
            'pval': approx_summary['pvalues'][i]
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
