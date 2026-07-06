"""
EXP E11 (REVISION_PLAN_CLAUDE_CODE.md): interpretability demonstration.
  I1: coefficient (b) sign stability across the lambda grid and across
      expanding windows.
  I2: waterfall figure decomposing one month's H-L predicted return into
      the 9 coordinate contributions (b_k * spread in MIC_k).
  I3: coordinate-contribution time series, 2020 highlighted.

Uses the window-local linear PROJECTION of y_pred onto MIC (see
analysis/mic_decomposition.py::fit_window_projections), not literal network
weights -- see analysis/outputs/DEVIATIONS.md for why raw ensemble-averaged
network weights are not a faithful reconstruction of the stored predictions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.util import REPO_ROOT, CONSENSUS_VARS, build_tidy_frame, get_lambda_list_for_horizon
from analysis.revision.mic_decomposition import fit_window_projections
from analysis.revision.lambda_selection import get_selected_lambda

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E11'


def i1_sign_stability(horizon, lambda_grid, results_dir=None):
    rows = []
    for lam in lambda_grid:
        df = build_tidy_frame(horizon, lam, results_dir=results_dir)
        proj = fit_window_projections(df, horizon, lam, results_dir=results_dir)
        for v in CONSENSUS_VARS:
            if v not in proj.columns:
                continue
            signs = np.sign(proj[v].dropna())
            rows.append({
                'lambda': lam, 'coordinate': v,
                'mean_coef': proj[v].mean(), 'std_coef': proj[v].std(),
                'sign_agreement': (signs == signs.mode().iloc[0]).mean() if len(signs) else np.nan,
                'n_windows': len(proj[v].dropna()),
            })
    return pd.DataFrame(rows)


def i2_waterfall(df, proj, test_date, out_dir, horizon):
    """Waterfall of b_k * (H MIC_k - L MIC_k) for the chosen window's H-L spread."""
    sub = df[df['date'] == pd.Timestamp(test_date)]
    if len(sub) == 0 or test_date not in proj.index:
        return None

    mic_cols = [f'MIC_{v}' for v in CONSENSUS_VARS]
    sub = sub.dropna(subset=['y_pred'])
    if len(sub) < 20:
        return None
    top = sub.nlargest(max(len(sub) // 10, 1), 'y_pred')
    bot = sub.nsmallest(max(len(sub) // 10, 1), 'y_pred')

    spread = top[mic_cols].mean() - bot[mic_cols].mean()
    b = proj.loc[test_date, CONSENSUS_VARS]
    contributions = pd.Series({v: b[v] * spread[f'MIC_{v}'] for v in CONSENSUS_VARS}).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    cum = 0.0
    for i, (name, val) in enumerate(contributions.items()):
        bottom = cum if val >= 0 else cum + val
        ax.bar(i, abs(val), bottom=bottom, color='tab:blue' if val >= 0 else 'tab:red')
        cum += val
    ax.set_xticks(range(len(contributions)))
    ax.set_xticklabels(contributions.index, rotation=45, ha='right')
    ax.set_ylabel('Contribution to H-L predicted return')
    ax.set_title(f'Waterfall: coordinate contributions to H-L spread ({test_date})')
    fig.tight_layout()
    fig.savefig(out_dir / f'{horizon}_I2_waterfall_{test_date}.png', dpi=150)
    plt.close(fig)

    contributions.to_csv(out_dir / f'{horizon}_I2_waterfall_{test_date}.csv')
    return contributions


def i3_contribution_timeseries(df, proj, out_dir, horizon):
    mic_cols = [f'MIC_{v}' for v in CONSENSUS_VARS]
    df = df.copy()
    df['test_date'] = None
    test_dates_sorted = sorted(proj.index)
    for td in test_dates_sorted:
        end = pd.Timestamp(td)
        start = end - pd.DateOffset(years=1)
        mask = (df['date'] >= start) & (df['date'] < end)
        df.loc[mask, 'test_date'] = td

    df = df.dropna(subset=['test_date'])
    rows = []
    for month, sub in df.groupby('date'):
        td = sub['test_date'].iloc[0]
        if td not in proj.index:
            continue
        b = proj.loc[td, CONSENSUS_VARS]
        mean_mic = sub[mic_cols].mean()
        contrib = {v: b[v] * mean_mic[f'MIC_{v}'] for v in CONSENSUS_VARS}
        contrib['date'] = month
        rows.append(contrib)

    ts = pd.DataFrame(rows).set_index('date').sort_index()
    ts.to_csv(out_dir / f'{horizon}_I3_contribution_timeseries.csv')

    fig, ax = plt.subplots(figsize=(10, 5))
    ts.plot(ax=ax, linewidth=1.2)
    ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'), color='grey', alpha=0.2, label='2020')
    ax.set_ylabel('Mean coordinate contribution to predicted return')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f'{horizon}_I3_contribution_timeseries.png', dpi=150)
    plt.close(fig)
    return ts


def run(horizon='12month', results_dir=None, out_dir=None, lambda_grid=None, waterfall_test_date=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if lambda_grid is None:
        lambda_grid = get_lambda_list_for_horizon(horizon, results_dir=str(results_dir or REPO_ROOT / 'final_results'))

    i1 = i1_sign_stability(horizon, lambda_grid, results_dir=results_dir)
    i1.to_csv(out_dir / f'{horizon}_I1_sign_stability.csv', index=False)

    best_lambda = get_selected_lambda(horizon=horizon, results_dir=results_dir)
    df = build_tidy_frame(horizon, best_lambda, results_dir=results_dir)
    proj = fit_window_projections(df, horizon, best_lambda, results_dir=results_dir)
    proj.to_csv(out_dir / f'{horizon}_window_projections_lambda{best_lambda}.csv')

    if waterfall_test_date is None:
        waterfall_test_date = '2021-01-01' if '2021-01-01' in proj.index else proj.index[len(proj) // 2]
    waterfall = i2_waterfall(df, proj, waterfall_test_date, out_dir, horizon)

    ts = i3_contribution_timeseries(df, proj, out_dir, horizon)

    low_stability = i1[(i1['lambda'] == best_lambda) & (i1['sign_agreement'] < 0.8)]['coordinate'].tolist()

    lines = [
        f"# E11 interpretability ({horizon}, lambda={best_lambda})",
        "",
        "NOTE: coefficients here are the window-local linear PROJECTION of the stored ensemble "
        "prediction onto MIC (see analysis/mic_decomposition.py::fit_window_projections), not "
        "literal averaged network weights -- see DEVIATIONS.md.",
        "",
        f"## I1 -- sign stability at lambda={best_lambda}",
        f"Coordinates with sign agreement < 80% across windows: {low_stability}",
        "",
        f"## I2 -- waterfall for {waterfall_test_date}",
        (waterfall.to_string() if waterfall is not None else "(not enough data for this window)"),
        "",
        "## I3 -- coordinate contribution time series",
        f"Written to {horizon}_I3_contribution_timeseries.csv/.png (2020 highlighted).",
        "",
        "Supports / weakens / neutral w.r.t. CLAIM-L1/SCOPE: "
        + ("SUPPORTS partial interpretability -- most coordinates show stable sign across windows, "
           "consistent with the paper's SCOPE claim that high-fidelity coordinates are "
           "semantically interpretable." if len(low_stability) <= 2 else
           "PARTIALLY WEAKENS -- several coordinates show unstable sign across windows, "
           "reinforcing that interpretation should be restricted to the high-fidelity subset "
           "(K1 in E6), not all 9 coordinates uniformly."),
    ]
    (out_dir / f'{horizon}_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {'i1': i1, 'proj': proj, 'waterfall': waterfall, 'ts': ts, 'best_lambda': best_lambda}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    result = run(horizon=args.horizon)
    print(result['i1'].groupby('coordinate')['sign_agreement'].mean().sort_values())
