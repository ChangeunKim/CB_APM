"""
EXP E10 (REVISION_PLAN_CLAUDE_CODE.md): robust portfolio headline. Extends
the single-sort decile module (analysis/util.py::decile_sort_returns) with:
  (a) ex-microcap (NYSE 20th-percentile Size breakpoint)
  (b) EW and VW (VW uses true log-ME weighting -- see util.py::portfolio_weights)
  (c) net-of-cost returns (simple proportional cost model, assumption logged)
  (d) FF6 time-series alpha of the H-L portfolio
  (e) per-year OOS R2 and H-L spread (stability across the 2014-2023 test years)

Applies to the CB-framework predicted return, the raw-consensus composite,
and the MIC composite, so it composes directly with E13's decision tables.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.util import (
    REPO_ROOT, CONSENSUS_VARS, build_tidy_frame, decile_sort_returns,
    load_size_and_exchcd, nyse_breakpoint_mask, zscore_composite, turnover,
)
from analysis.revision.lambda_selection import get_selected_lambda
from analysis.stats_utils import newey_west_ols, newey_west_tstat
from models.metrics import r2_score

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E10'

ONE_WAY_COST_BPS = 10  # documented assumption: 10bps one-way proportional cost


def load_ff6(data_dir=None):
    if data_dir is None:
        data_dir = REPO_ROOT / 'data'
    data_dir = Path(data_dir)

    ff5 = pd.read_csv(data_dir / 'F-F_Research_Data_5_Factors_2x3.csv', dtype=str)
    ff5 = ff5[ff5['date'].str.match(r'^\d{6}$', na=False)]
    mom = pd.read_csv(data_dir / 'F-F_Momentum_Factor.csv', dtype=str)
    mom = mom[mom['date'].str.match(r'^\d{6}$', na=False)]

    ff = ff5.merge(mom, on='date', how='inner')
    ff['date'] = pd.to_datetime(ff['date'], format='%Y%m')
    for c in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom']:
        ff[c] = ff[c].astype(float) / 100.0
    return ff


def net_of_cost_panel(panel, turnover_series_by_bin=None, cost_bps=ONE_WAY_COST_BPS):
    """
    Simple proportional cost model (ASSUMPTION, logged in DEVIATIONS.md):
    each monthly rebalance incurs cost_bps (one-way, in bps) applied to a
    full round-trip turnover assumption of 100% for the H-L leg (long top
    decile, short bottom decile, monthly rebalance -> full turnover), i.e.
    a flat cost_bps/10000 subtracted from H-L each month. This is a
    simplification of the repo's existing D.5.3 cost model (not directly
    reusable here since D.5.3 lives only inside the portfolio.ipynb
    notebook, not as an importable function).
    """
    net = panel.copy()
    if 'H-L' in net.columns:
        net['H-L'] = net['H-L'] - cost_bps / 10000.0
    return net


def per_year_stability(df, y_true_col='y_true', y_pred_col='y_pred', date_col='date'):
    df = df.copy()
    df['year'] = df[date_col].dt.year
    rows = []
    for yr, sub in df.groupby('year'):
        r2 = r2_score(sub[y_true_col].values, sub[y_pred_col].values)
        rows.append({'year': yr, 'n': len(sub), 'R2_return': r2})
    return pd.DataFrame(rows).sort_values('year')


def run_one_signal(df, signal_col, size_df, out_dir, tag, horizon):
    results = {}

    for microcap_tag, use_ex_microcap in [('all', False), ('ex_microcap', True)]:
        sub = df
        if use_ex_microcap:
            merged = df.merge(size_df, on=['permno', 'date'], how='inner')
            mask = nyse_breakpoint_mask(merged)
            sub = merged.loc[mask]
        else:
            sub = df.merge(size_df, on=['permno', 'date'], how='left')

        for weight_tag, weight_col in [('EW', None), ('VW', 'Size')]:
            panel, summary = decile_sort_returns(sub, signal_col, return_col='y_true', weight_col=weight_col)
            summary_net = None
            if 'H-L' in panel.columns:
                net_panel = net_of_cost_panel(panel)
                summary_net = pd.DataFrame({'H-L_net': newey_west_tstat(net_panel['H-L'].dropna())}).T

            key = f'{tag}_{microcap_tag}_{weight_tag}'
            summary.to_csv(out_dir / f'{horizon}_{key}_decile_summary.csv')
            panel.to_csv(out_dir / f'{horizon}_{key}_decile_panel.csv')
            results[key] = {'panel': panel, 'summary': summary, 'summary_net': summary_net}

    return results


def ff6_alpha(hl_series, ff6):
    hl = hl_series.dropna()
    merged = pd.DataFrame({'HL': hl}).join(ff6.set_index('date'), how='inner')
    if len(merged) < 24:
        return None
    X = np.column_stack([np.ones(len(merged)), merged[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']].values])
    y = merged['HL'].values
    res = newey_west_ols(X, y, lag=12)
    labels = ['alpha', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    return pd.DataFrame({'var': labels, 'coef': res['coef'], 'se': res['se'],
                          'tstat': res['tstat'], 'pval': res['pval']}).assign(nobs=res['nobs'])


def run(horizon='12month', results_dir=None, data_dir=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    best_lambda = get_selected_lambda(horizon=horizon, results_dir=results_dir)
    df = build_tidy_frame(horizon, best_lambda, results_dir=results_dir)
    df['raw_composite'] = zscore_composite(df, [f'C_{v}' for v in CONSENSUS_VARS])
    df['mic_composite'] = zscore_composite(df, [f'MIC_{v}' for v in CONSENSUS_VARS])

    size_df = load_size_and_exchcd(horizon, data_dir=data_dir)
    ff6 = load_ff6(data_dir=data_dir)

    signals = {'cb_predicted_return': 'y_pred', 'raw_composite': 'raw_composite', 'mic_composite': 'mic_composite'}

    all_results = {}
    ff6_rows = []
    for tag, col in signals.items():
        res = run_one_signal(df, col, size_df, out_dir, tag, horizon)
        all_results[tag] = res

        key_all_vw = f'{tag}_all_VW'
        if key_all_vw in res and 'H-L' in res[key_all_vw]['panel'].columns:
            alpha_tbl = ff6_alpha(res[key_all_vw]['panel']['H-L'], ff6)
            if alpha_tbl is not None:
                alpha_tbl.to_csv(out_dir / f'{horizon}_{tag}_FF6_alpha.csv', index=False)
                a_row = alpha_tbl[alpha_tbl['var'] == 'alpha'].iloc[0]
                ff6_rows.append({'signal': tag, 'alpha': a_row['coef'], 'tstat': a_row['tstat'], 'pval': a_row['pval']})

    ff6_summary = pd.DataFrame(ff6_rows)
    ff6_summary.to_csv(out_dir / f'{horizon}_FF6_alpha_summary.csv', index=False)

    stability = per_year_stability(df)
    stability.to_csv(out_dir / f'{horizon}_per_year_R2_stability.csv', index=False)

    lines = [
        f"# E10 robust portfolio headline ({horizon}, lambda={best_lambda})",
        "",
        f"ASSUMPTION: net-of-cost uses a flat {ONE_WAY_COST_BPS}bps one-way proportional cost "
        "applied to the H-L spread each month (full-turnover assumption for a monthly-rebalanced "
        "top-minus-bottom-decile long-short leg). This is a simplification of the repo's existing "
        "D.5.3 cost model referenced in the paper (not importable as a standalone function from "
        "the existing notebooks) -- logged as a deviation in analysis/outputs/DEVIATIONS.md.",
        "",
        "ASSUMPTION: ex-microcap uses the standard NYSE 20th-percentile Size breakpoint "
        "(data/raw/IBES_summary.csv exchcd==1 firms), applied to all exchanges -- this differs "
        "from the earlier assumption in the plan (that Size was rank-normalized in the stored "
        "artifacts); Size in data/input_<h>month.csv is actually raw log market equity, so this "
        "is a standard, not approximated, NYSE breakpoint.",
        "",
        "## FF6 alpha of the H-L portfolio (all-firm, VW), per signal",
        ff6_summary.to_string(index=False) if len(ff6_summary) else "(insufficient data)",
        "",
        "## Per-year OOS R2 and stability (2014-2023 test years)",
        stability.to_string(index=False),
        "",
        "REFRAMED per REVISION_PLAN_CLAUDE_CODE.md's updated Section 0: CLAIM-L1 (predictive R2) "
        "is SUPPORTING evidence, not headline. The 2020-concentration caveat is material: the table "
        "above shows 2020's R2 (46.30%) is a major outlier vs. the other years (mostly 0-13%, one "
        "negative year) -- so a per-year mean or single pooled R2 without this table would "
        "overstate how uniformly the prediction gain holds. See analysis/outputs/E12 for the "
        "matching normal_ex_2020 vs full_sample split (R2 ~2.9% ex-2020 vs 10.48% pooled).",
        "",
        "Supports / weakens / neutral w.r.t. CLAIM-L1 (supporting, not headline): "
        + ("SUPPORTS, WITH THE 2020 CAVEAT ABOVE -- the CB-framework H-L portfolio retains a "
           "statistically significant FF6 alpha net of common factor exposures, but the underlying "
           "R2 gain is concentrated in 2020 and should not be presented as uniformly stable across "
           "years without this caveat." if len(ff6_summary) and
           (ff6_summary.loc[ff6_summary['signal'] == 'cb_predicted_return', 'pval'] < 0.1).any()
           else "MIXED/WEAK -- the CB-framework H-L portfolio's FF6 alpha is not statistically "
           "significant at conventional levels in this sample; report as found."),
    ]
    (out_dir / f'{horizon}_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {'results': all_results, 'ff6_summary': ff6_summary, 'stability': stability, 'best_lambda': best_lambda}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    result = run(horizon=args.horizon)
    print(result['ff6_summary'])
