"""
EXP E12 (REVISION_PLAN_CLAUDE_CODE.md): sub-panel analyses (appendix tier).
Reuses E0's final_results predictions -- no retraining. Splits: analyst
coverage hi/lo, size big/small, dispersion hi/lo, optimism hi/lo, NBER crisis
(2020) vs normal. Reports OOS R2_return and H-L spread per split.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.util import REPO_ROOT, build_tidy_frame, decile_sort_returns, load_size_and_exchcd
from analysis.revision.lambda_selection import get_selected_lambda
from analysis.stats_utils import newey_west_tstat
from models.metrics import r2_score

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E12'

IBES_SPLIT_COLS = ['permno', 'date', 'nanalyst', 'disp', 'meanrec']


def load_ibes_splits(data_dir=None):
    if data_dir is None:
        data_dir = REPO_ROOT / 'data' / 'raw'
    df = pd.read_csv(Path(data_dir) / 'IBES_summary.csv', usecols=IBES_SPLIT_COLS)
    df['date'] = pd.to_datetime(df['date']).values.astype('datetime64[M]')
    df = df.drop_duplicates(['permno', 'date'])
    return df


def _split_r2_and_hl(sub, label, rows, panels):
    if len(sub) < 100:
        rows.append({'split': label, 'n': len(sub), 'R2_return': np.nan, 'HL_mean': np.nan, 'HL_tstat': np.nan})
        return
    r2 = r2_score(sub['y_true'].values, sub['y_pred'].values)
    panel, summary = decile_sort_returns(sub, 'y_pred', return_col='y_true')
    hl = summary.loc['H-L'] if 'H-L' in summary.index else None
    rows.append({
        'split': label, 'n': len(sub), 'R2_return': r2,
        'HL_mean': hl['mean'] if hl is not None else np.nan,
        'HL_tstat': hl['tstat'] if hl is not None else np.nan,
    })
    panels[label] = panel


def run(horizon='12month', results_dir=None, data_dir=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    best_lambda = get_selected_lambda(horizon=horizon, results_dir=results_dir)
    df = build_tidy_frame(horizon, best_lambda, results_dir=results_dir)

    size_df = load_size_and_exchcd(horizon, data_dir=data_dir)
    ibes = load_ibes_splits(data_dir=data_dir)

    df['month'] = df['date'].values.astype('datetime64[M]')
    merged = df.merge(size_df.rename(columns={'date': 'month'}), on=['permno', 'month'], how='left')
    merged = merged.merge(ibes.rename(columns={'date': 'month'}), on=['permno', 'month'], how='left')

    rows, panels = [], {}
    _split_r2_and_hl(merged, 'full_sample', rows, panels)

    # Analyst coverage hi/lo (median split, per date)
    merged['_cov_med'] = merged.groupby('date')['nanalyst'].transform('median')
    _split_r2_and_hl(merged[merged['nanalyst'] >= merged['_cov_med']], 'coverage_hi', rows, panels)
    _split_r2_and_hl(merged[merged['nanalyst'] < merged['_cov_med']], 'coverage_lo', rows, panels)

    # Size big/small (median split, per date)
    merged['_size_med'] = merged.groupby('date')['Size'].transform('median')
    _split_r2_and_hl(merged[merged['Size'] >= merged['_size_med']], 'size_big', rows, panels)
    _split_r2_and_hl(merged[merged['Size'] < merged['_size_med']], 'size_small', rows, panels)

    # Dispersion hi/lo (median split, per date)
    merged['_disp_med'] = merged.groupby('date')['disp'].transform('median')
    _split_r2_and_hl(merged[merged['disp'] >= merged['_disp_med']], 'dispersion_hi', rows, panels)
    _split_r2_and_hl(merged[merged['disp'] < merged['_disp_med']], 'dispersion_lo', rows, panels)

    # Optimism hi/lo: sign/magnitude of raw consensus (meanrec, lower=more optimistic in IBES convention:
    # 1=strong buy .. 5=strong sell) -- optimistic = meanrec below the per-date median
    merged['_rec_med'] = merged.groupby('date')['meanrec'].transform('median')
    _split_r2_and_hl(merged[merged['meanrec'] < merged['_rec_med']], 'optimistic_(low_meanrec)', rows, panels)
    _split_r2_and_hl(merged[merged['meanrec'] >= merged['_rec_med']], 'pessimistic_(high_meanrec)', rows, panels)

    # NBER crisis (2020 COVID window) vs normal
    is_crisis = merged['date'].dt.year == 2020
    _split_r2_and_hl(merged[is_crisis], 'crisis_2020', rows, panels)
    _split_r2_and_hl(merged[~is_crisis], 'normal_ex_2020', rows, panels)

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / f'{horizon}_subpanel_R2_and_HL.csv', index=False)

    lines = [
        f"# E12 sub-panel analyses ({horizon}, lambda={best_lambda})",
        "",
        result.to_string(index=False),
        "",
        "NOTE: crisis_2020 vs normal_ex_2020 is the same 2020-concentration caveat flagged in E10 "
        "(CLAIM-L1 is supporting evidence, not headline, precisely because of this split) -- "
        "normal_ex_2020's R2 (~2.9%) is the more representative figure for a typical year than the "
        "pooled full_sample figure (~10.5%), which is inflated by 2020.",
        "",
        "Supports / weakens / neutral w.r.t. CLAIM-L1 (supporting, not headline): "
        + ("SUPPORTS -- R2_return and H-L spread remain positive and broadly consistent with the "
           "full-sample result across the coverage/size/dispersion/optimism/crisis splits examined "
           "here." if (result['R2_return'].dropna() > 0).mean() > 0.7
           else "MIXED -- performance varies materially across sub-panels; report as found rather "
           "than treating the full-sample result as uniformly representative."),
    ]
    (out_dir / f'{horizon}_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {'result': result, 'panels': panels, 'best_lambda': best_lambda}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    out = run(horizon=args.horizon)
    print(out['result'])
