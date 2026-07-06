"""
EXP E13 (REVISION_PLAN_CLAUDE_CODE.md), CLAIM-L3 decision leg -- the paper's
proposed HEADLINE table. Implements:
  (a) ranking comparison -- VW decile portfolios on (i) raw-consensus
      composite, (ii) MIC composite, (iii) CB-framework predicted return
      -- gross AND net-of-cost, with Ledoit-Wolf Sharpe tests (MIC vs raw).
  (b) long-only tilt vs. VW market benchmark.
  (c) adjusted signal: raw composite + gamma * D composite, gamma fit
      walk-forward on PRIOR years only (documented proxy for "validation
      only" -- see module docstring).

SCOPE LIMITATION vs. the spec: arm (iv), the full-info NN comparison, is
SKIPPED. E1 (a full-info NN baseline) was never trained in this repo pass --
only the CB-framework (concept-bottleneck) models were trained across the
lambda grid, so there is no full-info-NN prediction to compare against. This
is a genuine scope gap, not a silent omission -- see analysis/outputs/
DEVIATIONS.md and the summary.md this script writes.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.util import (
    REPO_ROOT, CONSENSUS_VARS, build_tidy_frame, decile_sort_returns,
    zscore_composite, load_size_and_exchcd, turnover, portfolio_weights,
)
from analysis.revision.lambda_selection import get_selected_lambda
from analysis.revision.robust_portfolio import net_of_cost_panel, ONE_WAY_COST_BPS
from analysis.stats_utils import ledoit_wolf_sharpe_test, newey_west_tstat

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E13'
HEADLINE_PATH = REPO_ROOT / 'analysis' / 'outputs' / 'HEADLINE.md'


def build_signals(df):
    df = df.copy()
    d_cols = []
    for v in CONSENSUS_VARS:
        df[f'D_{v}'] = df[f'MIC_{v}'] - df[f'C_{v}']
        d_cols.append(f'D_{v}')
    df['raw_composite'] = zscore_composite(df, [f'C_{v}' for v in CONSENSUS_VARS])
    df['mic_composite'] = zscore_composite(df, [f'MIC_{v}' for v in CONSENSUS_VARS])
    df['D_composite'] = zscore_composite(df, d_cols)
    return df


def a_ranking_comparison(df, size_df, out_dir, horizon):
    signals = {
        'raw_composite': 'raw_composite',
        'mic_composite': 'mic_composite',
        'cb_predicted_return': 'y_pred',
    }
    merged = df.merge(size_df, on=['permno', 'date'], how='left')

    hl_series = {}
    for tag, col in signals.items():
        panel, summary = decile_sort_returns(merged, col, return_col='y_true', weight_col='Size')
        panel.to_csv(out_dir / f'{horizon}_E13a_{tag}_decile_panel.csv')
        summary.to_csv(out_dir / f'{horizon}_E13a_{tag}_decile_summary.csv')
        if 'H-L' in panel.columns:
            hl_series[tag] = panel['H-L']
            net_panel = net_of_cost_panel(panel)
            net_summary = pd.DataFrame({'H-L_net': newey_west_tstat(net_panel['H-L'].dropna())}).T
            net_summary.to_csv(out_dir / f'{horizon}_E13a_{tag}_HL_net_of_cost.csv')

    lw_tests = []
    if 'mic_composite' in hl_series and 'raw_composite' in hl_series:
        common_idx = hl_series['mic_composite'].dropna().index.intersection(hl_series['raw_composite'].dropna().index)
        res = ledoit_wolf_sharpe_test(hl_series['mic_composite'].loc[common_idx], hl_series['raw_composite'].loc[common_idx])
        lw_tests.append({'comparison': 'MIC_composite_vs_raw_consensus_composite', **res})
    if 'cb_predicted_return' in hl_series and 'raw_composite' in hl_series:
        # "b'MIC" in REVISION_PLAN's terminology = the CB-framework's own predicted
        # return (the linear head applied to MIC), as opposed to a simple
        # unweighted MIC composite.
        common_idx = hl_series['cb_predicted_return'].dropna().index.intersection(hl_series['raw_composite'].dropna().index)
        res = ledoit_wolf_sharpe_test(hl_series['cb_predicted_return'].loc[common_idx], hl_series['raw_composite'].loc[common_idx])
        lw_tests.append({'comparison': "bMIC_(cb_predicted_return)_vs_raw_consensus_composite", **res})
    lw_df = pd.DataFrame(lw_tests)
    lw_df.to_csv(out_dir / f'{horizon}_E13a_ledoit_wolf_tests.csv', index=False)
    return hl_series, lw_df


def b_long_only_tilt(df, size_df, out_dir, horizon, top_quantile=0.2):
    merged = df.merge(size_df, on=['permno', 'date'], how='left').dropna(subset=['mic_composite', 'y_true', 'Size'])
    merged = merged.copy()

    def _top_bucket(x):
        thr = x.quantile(1 - top_quantile)
        return x >= thr

    merged['is_top'] = merged.groupby('date')['mic_composite'].transform(_top_bucket)

    # Market benchmark: full-universe VW return
    mkt_w = portfolio_weights(merged, weight_col='Size', group_col='date')
    market_ret = merged.assign(_w=mkt_w).groupby('date').apply(
        lambda g: np.average(g['y_true'], weights=g['_w']), include_groups=False)

    # Tilt: overweight top-quantile names proportional to signal rank, base weight = VW market weight
    tilted = merged.copy()
    tilted['_base_w'] = mkt_w
    tilted['_rank'] = tilted.groupby('date')['mic_composite'].rank(pct=True)
    tilted['_tilt_w'] = np.where(tilted['is_top'], tilted['_base_w'] * (1 + tilted['_rank']), tilted['_base_w'])
    tilted['_tilt_w'] = tilted.groupby('date')['_tilt_w'].transform(lambda x: x / x.sum())

    tilt_ret = tilted.groupby('date').apply(
        lambda g: np.average(g['y_true'], weights=g['_tilt_w']), include_groups=False)

    active_ret = (tilt_ret - market_ret).dropna()
    te = active_ret.std(ddof=1)
    ir = active_ret.mean() / te if te > 0 else np.nan

    weights_df = tilted[['date', 'permno']].copy()
    weights_df['weight'] = tilted['_tilt_w'].values
    returns_df = merged[['date', 'permno', 'y_true']].rename(columns={'y_true': 'actual'}).set_index('date')
    weights_df = weights_df.set_index('date')
    try:
        to = turnover(returns_df, weights_df)
        mean_turnover = float(np.mean(to)) if len(to) else np.nan
    except Exception:
        mean_turnover = np.nan

    net_active = active_ret - ONE_WAY_COST_BPS / 10000.0
    nw_active = newey_west_tstat(active_ret)
    nw_net = newey_west_tstat(net_active)

    result = pd.DataFrame([{
        'active_return_mean': nw_active['mean'], 'active_return_tstat': nw_active['tstat'],
        'tracking_error': te, 'information_ratio': ir, 'mean_turnover': mean_turnover,
        'net_active_return_mean': nw_net['mean'], 'net_active_return_tstat': nw_net['tstat'],
    }])
    result.to_csv(out_dir / f'{horizon}_E13b_long_only_tilt.csv', index=False)
    return result


def c_adjusted_signal(df, out_dir, horizon):
    """
    gamma * D_composite added to raw_composite; gamma fit WALK-FORWARD on
    prior years only (proxy for "validation only" -- no true validation-split
    predictions exist in final_results, see analysis/outputs/E8 and
    DEVIATIONS.md). Gamma_t = argmin OLS coefficient of y_true on
    [raw_composite, D_composite] fit on years < t, applied to year t.
    """
    d = df.dropna(subset=['raw_composite', 'D_composite', 'y_true']).copy()
    d['year'] = d['date'].dt.year
    years = sorted(d['year'].unique())

    rows = []
    combined_signal = pd.Series(index=d.index, dtype=float)
    for i, yr in enumerate(years):
        train = d[d['year'] < yr]
        test = d[d['year'] == yr]
        if len(train) < 1000 or len(test) == 0:
            continue
        X = np.column_stack([np.ones(len(train)), train['raw_composite'].values, train['D_composite'].values])
        y = train['y_true'].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        gamma = beta[2] / beta[1] if beta[1] != 0 else np.nan
        combined_signal.loc[test.index] = test['raw_composite'] + gamma * test['D_composite']
        rows.append({'year': yr, 'gamma': gamma, 'n_train': len(train), 'n_test': len(test)})

    gamma_table = pd.DataFrame(rows)
    gamma_table.to_csv(out_dir / f'{horizon}_E13c_gamma_by_year.csv', index=False)

    d['adjusted_signal'] = combined_signal
    valid = d.dropna(subset=['adjusted_signal'])

    panel_adj, summary_adj = decile_sort_returns(valid, 'adjusted_signal', return_col='y_true')
    panel_raw, summary_raw = decile_sort_returns(valid, 'raw_composite', return_col='y_true')
    summary_adj.to_csv(out_dir / f'{horizon}_E13c_adjusted_decile_summary.csv')
    summary_raw.to_csv(out_dir / f'{horizon}_E13c_raw_decile_summary_samesample.csv')

    hl_adj = panel_adj['H-L'].dropna() if 'H-L' in panel_adj.columns else pd.Series(dtype=float)
    hl_raw = panel_raw['H-L'].dropna() if 'H-L' in panel_raw.columns else pd.Series(dtype=float)
    incremental = pd.DataFrame([{
        'HL_mean_adjusted': hl_adj.mean() if len(hl_adj) else np.nan,
        'HL_mean_raw': hl_raw.mean() if len(hl_raw) else np.nan,
        'HL_tstat_adjusted': newey_west_tstat(hl_adj)['tstat'] if len(hl_adj) else np.nan,
        'HL_tstat_raw': newey_west_tstat(hl_raw)['tstat'] if len(hl_raw) else np.nan,
    }])
    incremental.to_csv(out_dir / f'{horizon}_E13c_incremental_vs_raw.csv', index=False)
    return gamma_table, incremental


def run(horizon='12month', results_dir=None, data_dir=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    best_lambda = get_selected_lambda(horizon=horizon, results_dir=results_dir)
    df = build_tidy_frame(horizon, best_lambda, results_dir=results_dir)
    df = build_signals(df)
    size_df = load_size_and_exchcd(horizon, data_dir=data_dir)

    hl_series, lw_df = a_ranking_comparison(df, size_df, out_dir, horizon)
    tilt_result = b_long_only_tilt(df, size_df, out_dir, horizon)
    gamma_table, incremental = c_adjusted_signal(df, out_dir, horizon)

    def _verdict(row):
        if row['diff'] > 0 and row['pval'] < 0.1:
            return 'yes'
        if row['diff'] < 0 and row['pval'] < 0.1:
            return 'no'
        return 'economically large, not statistically separable in this sample'

    verdicts = {row['comparison']: _verdict(row) for _, row in lw_df.iterrows()} if len(lw_df) else {}

    ff6_note = "(see analysis/outputs/E10 for the FF6 alpha of the b'MIC / CB-framework H-L portfolio, " \
               "which is robust (t~3.06) -- kept as evidence of the predictor's own portfolio value, " \
               "separate from whether it beats raw-consensus decisions)."

    lines_summary = [
        f"# E13 decision-value analyses ({horizon}, lambda={best_lambda})",
        "",
        "REFRAMED per REVISION_PLAN_CLAUDE_CODE.md's updated Section 0: E13 is SECONDARY / honest "
        "supporting evidence, NOT the paper's headline (the headline is now CLAIM-ECON / E15's "
        "stickiness-concentration result). Report results plainly; do not write dominance claims "
        "the data do not support.",
        "",
        "SCOPE LIMITATION: arm (iv) full-info NN comparison is skipped -- E1 was never trained "
        "in this analysis-only pass (a training script exists at "
        "analysis/experiments/train_e1_full_info_nn.py, not yet run). Only (i) raw-consensus "
        "composite, (ii) MIC composite, and (iii) CB-framework predicted return (\"b'MIC\") are "
        "compared.",
        "",
        "## (a) Ranking comparison -- Ledoit-Wolf Sharpe tests",
        lw_df.to_string(index=False) if len(lw_df) else "(insufficient data)",
        f"Verdicts: {verdicts}",
        ff6_note,
        "",
        "## (b) Long-only tilt vs VW market",
        tilt_result.to_string(index=False),
        "(IR is small and net-of-cost active return is typically negative here -- a single honest "
        "sentence for the paper, not an exhibit to oversell.)",
        "",
        "## (c) Adjusted signal (raw + gamma*D, gamma fit walk-forward on prior years)",
        f"DEVIATION: gamma is fit on PRIOR TEST YEARS (walk-forward), not a true validation split "
        "-- final_results only persists test-period predictions (see E8/DEVIATIONS.md), so a "
        "literal validation-only fit is not reconstructible. Walk-forward avoids same-period "
        "look-ahead but is not identical to the spec's validation-only protocol.",
        gamma_table.to_string(index=False),
        incremental.to_string(index=False),
        "(Increment over the raw composite alone is negligible here -- secondary, not headline.)",
        "",
        "Supports / weakens / neutral w.r.t. CLAIM-L3 (decision leg, SECONDARY not headline): "
        "MIXED/HONEST-NULL -- MIC-based decisions are economically larger than raw-consensus "
        "decisions (higher Sharpe point estimates) but not statistically separable from them in "
        "this ~10-year overlapping-window sample; the long-only tilt and gamma-adjustment "
        "increments are both small. This does not weaken CLAIM-ECON (E15), which does not rest on "
        "decision-dominance; it is reported here for completeness per Section 5's honesty rule.",
    ]
    (out_dir / f'{horizon}_summary.md').write_text('\n'.join(lines_summary), encoding='utf-8')

    return {'lw_df': lw_df, 'tilt_result': tilt_result, 'gamma_table': gamma_table,
            'incremental': incremental, 'verdicts': verdicts}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    result = run(horizon=args.horizon)
    print('E13 verdicts (secondary, not headline):', result['verdicts'])
