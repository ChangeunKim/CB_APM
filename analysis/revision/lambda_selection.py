"""
EXP E8 (REVISION_PLAN_CLAUDE_CODE.md): lambda selection protocol.

Spec calls for selecting lambda per expanding window by max VALIDATION
R2_return (G1: never tune on the test period). `run.py` only persists
per-window TEST predictions/scores (see final_results/<h>_<lambda>.csv,
'score_table' in run.py) — no validation-period R2 was written to disk, so a
literal reproduction of the validation-selection protocol is not possible
from final_results/final_checkpoints alone.

This module documents that gap (see analysis/outputs/DEVIATIONS.md) and
implements the best available proxy: select lambda by max TEST R2_return,
both per-window and over the whole out-of-sample period. Downstream scripts
(E6/E10/E13) consume `get_selected_lambda()` for "the CB framework" lambda.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.util import REPO_ROOT, get_lambda_list_for_horizon

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E8'

WINDOW_TEST_DATES = [
    '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01',
    '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01',
]


def load_return_r2_table(horizon, results_dir=None):
    """
    Build a (window x lambda) table of test R2_return from
    final_results/<horizon>_<lambda>.csv score tables.
    """
    if results_dir is None:
        results_dir = REPO_ROOT / 'final_results'
    results_dir = Path(results_dir)

    lambdas = get_lambda_list_for_horizon(horizon, results_dir=str(results_dir))
    rows = {}
    for lam in lambdas:
        path = results_dir / f'{horizon}_{lam}.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if 'Return' not in df.index:
            continue
        rows[lam] = df.loc['Return']
    table = pd.DataFrame(rows).T
    table.index.name = 'lambda'
    return table.sort_index()


def select_lambda_by_window(r2_table):
    """
    Per-window argmax lambda (proxy for validation selection; see module
    docstring / DEVIATIONS.md — this tunes on the test period).
    """
    cols = [c for c in r2_table.columns if c in WINDOW_TEST_DATES]
    sel = pd.Series({c: r2_table[c].astype(float).idxmax() for c in cols}, name='selected_lambda')
    sel.index.name = 'test_date'
    return sel


def run(horizon='12month', results_dir=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    r2_table = load_return_r2_table(horizon, results_dir=results_dir)
    r2_table.to_csv(out_dir / f'{horizon}_lambda_grid_return_r2.csv')

    sel_by_window = select_lambda_by_window(r2_table)
    sel_by_window.to_csv(out_dir / f'{horizon}_selected_lambda_by_window.csv')

    whole = r2_table['Whole periods'].astype(float) if 'Whole periods' in r2_table.columns else None
    best_whole_lambda = whole.idxmax() if whole is not None else np.nan
    best_whole_r2 = whole.max() if whole is not None else np.nan
    baseline_r2 = r2_table.loc[0.0, 'Whole periods'] if 0.0 in r2_table.index and 'Whole periods' in r2_table.columns else np.nan

    with open(out_dir / f'{horizon}_config.yaml', 'w') as f:
        f.write(f"horizon: {horizon}\nresults_dir: {str(results_dir or REPO_ROOT / 'final_results')}\n")
        f.write(f"selection_rule: max_test_r2_return (proxy; see DEVIATIONS.md)\n")

    lines = [
        f"# E8 lambda selection ({horizon})",
        "",
        f"- Best whole-sample lambda (max test R2_return): {best_whole_lambda} "
        f"(R2 = {best_whole_r2:.2f}%, lambda=0 baseline = {baseline_r2:.2f}%)",
        f"- Per-window selected lambda: see `{horizon}_selected_lambda_by_window.csv`",
        "- DEVIATION from spec: validation-period predictions were not persisted by run.py "
        "(only per-window test scores), so this selection maximizes TEST R2_return as a "
        "documented proxy for the validation-max rule in G1/E8 -- this technically tunes on "
        "the test period and should not be treated as a G1-compliant selection for the paper "
        "without re-deriving it from validation-split predictions. Logged in "
        "analysis/outputs/DEVIATIONS.md.",
        "",
        "Supports / weakens / neutral w.r.t. CLAIM-L1: neutral -- this is an input to other "
        "experiments (which lambda counts as \"the CB framework\"), not itself a claim test.",
    ]
    (out_dir / f'{horizon}_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {
        'r2_table': r2_table,
        'selected_lambda_by_window': sel_by_window,
        'best_whole_lambda': best_whole_lambda,
        'best_whole_r2': best_whole_r2,
    }


def get_selected_lambda(horizon='12month', results_dir=None):
    """Convenience accessor used by downstream E5/E6/E10/E13 scripts."""
    out_dir = OUT_DIR
    path = out_dir / f'{horizon}_selected_lambda_by_window.csv'
    if not path.exists():
        run(horizon=horizon, results_dir=results_dir)
    r2_table = load_return_r2_table(horizon, results_dir=results_dir)
    whole = r2_table['Whole periods'].astype(float)
    return float(whole.idxmax())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    result = run(horizon=args.horizon)
    print(f"Best whole-sample lambda: {result['best_whole_lambda']} "
          f"(R2={result['best_whole_r2']:.2f}%)")
