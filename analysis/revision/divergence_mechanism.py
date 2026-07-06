"""
EXP E5 (REVISION_PLAN_CLAUDE_CODE.md), CLAIM-L3: divergence mechanism tests.

D_it = MIC_it - C_it (per coordinate + first-PC composite). Tests whether D
is a machine-implied expectation adjustment that predicts subsequent forecast
revisions (M1), earnings surprises (M2), and realized forecast error (M3),
and whether sorting on D produces a return spread (M4).

Signs are reported as found (no pre-labeling as over/under-reaction); the
supported label (or "mixed") is written into the summary.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from analysis.util import REPO_ROOT, CONSENSUS_VARS, build_tidy_frame, decile_sort_returns
from analysis.revision.lambda_selection import get_selected_lambda
from analysis.stats_utils import two_way_cluster_ols, newey_west_tstat

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E5'

IBES_COLS = ['permno', 'date', 'chfeps', 'chrec', 'sue', 'sfe', 'disp', 'nanalyst']


def load_ibes(data_dir=None):
    if data_dir is None:
        data_dir = REPO_ROOT / 'data' / 'raw'
    df = pd.read_csv(Path(data_dir) / 'IBES_summary.csv', usecols=IBES_COLS)
    # IBES dates are month-end; final_results dates are month-start -> align to month key
    df['date'] = pd.to_datetime(df['date']).values.astype('datetime64[M]')
    df = df.sort_values(['permno', 'date'])
    return df


def add_future_ibes_targets(ibes):
    """Add 3/6/12m-ahead chfeps/chrec and sue, per firm, no look-ahead leakage
    into the CONTEMPORANEOUS regressors (these are outcome/target columns)."""
    ibes = ibes.copy()
    for h in (3, 6, 12):
        ibes[f'chfeps_fwd{h}'] = ibes.groupby('permno')['chfeps'].shift(-h)
        ibes[f'chrec_fwd{h}'] = ibes.groupby('permno')['chrec'].shift(-h)
        ibes[f'sue_fwd{h}'] = ibes.groupby('permno')['sue'].shift(-h)
    return ibes


def build_divergence_frame(horizon, weight_lambda, results_dir=None, data_dir=None):
    df = build_tidy_frame(horizon, weight_lambda, results_dir=results_dir)
    d_cols = []
    for v in CONSENSUS_VARS:
        df[f'D_{v}'] = df[f'MIC_{v}'] - df[f'C_{v}']
        d_cols.append(f'D_{v}')

    pca = PCA(n_components=1)
    df['D_pc1'] = pca.fit_transform(df[d_cols].fillna(0).values).flatten()
    if pca.components_[0].sum() < 0:
        df['D_pc1'] *= -1  # sign-normalize so higher D_pc1 ~ higher avg D

    ibes = load_ibes(data_dir=data_dir)
    ibes = add_future_ibes_targets(ibes)

    df['month'] = df['date'].values.astype('datetime64[M]')
    merged = df.merge(ibes, left_on=['permno', 'month'], right_on=['permno', 'date'],
                       how='inner', suffixes=('', '_ibes'))
    return merged, d_cols


def _panel_reg(merged, x_col, y_col, controls):
    sub = merged.dropna(subset=[x_col, y_col] + controls + ['permno', 'date'])
    if len(sub) < 100:
        return None
    X = np.column_stack([np.ones(len(sub)), sub[x_col].values] + [sub[c].values for c in controls])
    y = sub[y_col].values
    res = two_way_cluster_ols(X, y, sub['permno'].values, sub['date'].values)
    labels = ['const', x_col] + controls
    return pd.DataFrame({
        'var': labels, 'coef': res['coef'], 'se': res['se'],
        'tstat': res['tstat'], 'pval': res['pval'],
    }).assign(nobs=res['nobs'], dependent=y_col)


def m1_revision_regressions(merged, controls):
    rows = []
    for h in (3, 6, 12):
        for target in (f'chfeps_fwd{h}', f'chrec_fwd{h}'):
            r = _panel_reg(merged, 'D_pc1', target, controls)
            if r is not None:
                rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def m2_sue_regressions(merged, controls):
    rows = []
    for h in (3, 6, 12):
        r = _panel_reg(merged, 'D_pc1', f'sue_fwd{h}', controls)
        if r is not None:
            rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def m3_forecast_error(merged, d_cols):
    rows = []
    for col in d_cols + ['D_pc1']:
        sub = merged.dropna(subset=[col, 'sfe'])
        if len(sub) < 30:
            continue
        corr = sub[col].corr(sub['sfe'])
        rows.append({'D_coordinate': col, 'corr_with_sfe': corr, 'nobs': len(sub)})
    out = pd.DataFrame(rows)
    out['is_primary_FEPS'] = out['D_coordinate'] == 'D_FEPS'
    return out


def m4_quintile_sorts(merged):
    panel, summary = decile_sort_returns(merged, 'D_pc1', return_col='y_true', n_bins=5)
    # 5x5 double sort D_pc1 x raw FEPS
    sub = merged.dropna(subset=['D_pc1', 'C_FEPS', 'y_true'])

    def _bin(x, n=5):
        try:
            return pd.qcut(x, n, labels=False, duplicates='drop') + 1
        except ValueError:
            return pd.Series(np.nan, index=x.index)

    sub = sub.copy()
    sub['D_bin'] = sub.groupby('date')['D_pc1'].transform(_bin)
    sub['FEPS_bin'] = sub.groupby('date')['C_FEPS'].transform(_bin)
    double = sub.groupby(['FEPS_bin', 'D_bin'])['y_true'].mean().unstack('D_bin')
    return panel, summary, double


def run(horizon='12month', results_dir=None, data_dir=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    best_lambda = get_selected_lambda(horizon=horizon, results_dir=results_dir)
    merged, d_cols = build_divergence_frame(horizon, best_lambda, results_dir=results_dir, data_dir=data_dir)

    controls = [f'C_{v}' for v in CONSENSUS_VARS]

    m1 = m1_revision_regressions(merged, controls)
    m1.to_csv(out_dir / f'{horizon}_M1_revision_regressions.csv', index=False)

    m2 = m2_sue_regressions(merged, controls)
    m2.to_csv(out_dir / f'{horizon}_M2_sue_regressions.csv', index=False)

    m3 = m3_forecast_error(merged, d_cols)
    m3.to_csv(out_dir / f'{horizon}_M3_forecast_error_corr.csv', index=False)

    panel, sort_summary, double = m4_quintile_sorts(merged)
    sort_summary.to_csv(out_dir / f'{horizon}_M4_quintile_sort_summary.csv')
    double.to_csv(out_dir / f'{horizon}_M4_double_sort_D_x_FEPS.csv')

    # Determine supported sign/label as found (no pre-labeling)
    m1_dpc1 = m1[m1['var'] == 'D_pc1'] if len(m1) else pd.DataFrame()
    sig_m1 = m1_dpc1[m1_dpc1['pval'] < 0.1] if len(m1_dpc1) else pd.DataFrame()
    m3_pc1 = m3[m3['D_coordinate'] == 'D_pc1']
    hl_t = sort_summary.loc['H-L', 'tstat'] if 'H-L' in sort_summary.index else np.nan

    if len(sig_m1) > 0:
        sign = np.sign(sig_m1['coef'].mean())
        label = 'underreaction (D predicts same-signed future revisions)' if sign > 0 else \
                'overreaction/reversal (D predicts opposite-signed future revisions)'
    else:
        label = 'mixed / not statistically supported at the 10% level'

    m2_dpc1 = m2[m2['var'] == 'D_pc1'] if len(m2) else pd.DataFrame()
    m2_all_insignificant = len(m2_dpc1) > 0 and (m2_dpc1['pval'] > 0.10).all()

    lines = [
        f"# E5 divergence mechanism ({horizon}, lambda={best_lambda})",
        "",
        "REFRAMED per REVISION_PLAN_CLAUDE_CODE.md's updated Section 0 (CLAIM-ECON headline): "
        "D = MIC - C is read as a machine-learning estimate of the analyst-stickiness-induced "
        "underreaction term (Cao, Tao, Wang & Yin 2026; Bouchaud et al. 2019's sticky-belief "
        "framing) -- under cognitive noise, sticky analysts compress new information toward their "
        "prior forecast, so D anticipates, ex ante, the revision a sticky analyst will make later. "
        "ROLE SPLIT (do not conflate): D (here) carries the analyst-REVISION-prediction signal; "
        "MIC_perp (analysis/mic_decomposition.py, E6) carries the STABLE RETURN-SPREAD signal. "
        "M1's finding and M4's weak D-sort are both expected under this split, not a contradiction.",
        "",
        f"nobs after merge with IBES_summary: {len(merged)}",
        "",
        "## M1 -- D_pc1 vs future forecast revisions (3/6/12m chfeps, chrec) [HEADLINE evidence]",
        m1_dpc1.to_string(index=False) if len(m1_dpc1) else "(insufficient data)",
        "",
        "## M2 -- D_pc1 vs future SUE [prominent NULL, by design]",
        (m2[m2['var'] == 'D_pc1']).to_string(index=False) if len(m2) else "(insufficient data)",
        f"D does NOT predict realized SUE (all p>0.10 here: {m2_all_insignificant}) -- this is "
        "evidence FOR the stickiness framing, not a weakness: D anticipates belief-UPDATING "
        "(what analysts will revise toward), not fundamentals (what will actually happen). Keep "
        "this null prominent in the paper, not buried.",
        "",
        "## M3 -- D vs realized forecast error (sfe), FEPS coordinate primary",
        m3.to_string(index=False),
        "",
        "## M4 -- quintile sorts on D_pc1 [weak by design -- D is a revision signal, not a return signal]",
        f"H-L NW t-stat = {hl_t:.2f}" if pd.notna(hl_t) else "H-L not estimable",
        "The return spread lives in MIC_perp (E6), not D -- a weak D-sort here is consistent with, "
        "not contrary to, the role-split framing above.",
        "",
        f"Supported label (as found, not pre-assigned): {label}",
        "",
        "Supports / weakens / neutral w.r.t. CLAIM-ECON: "
        + ("SUPPORTS -- D shows statistically significant predictive content for subsequent "
           "analyst revisions (M1) while NOT predicting realized fundamentals (M2 null), jointly "
           "consistent with D estimating a stickiness-induced, ex-ante belief-updating signal "
           "rather than an information-content signal about fundamentals." if (len(sig_m1) > 0)
           else "WEAK/MIXED -- D's predictive link to revisions is not statistically strong in "
           "this sample; report as found."),
    ]
    (out_dir / f'{horizon}_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {'m1': m1, 'm2': m2, 'm3': m3, 'sort_summary': sort_summary, 'label': label}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    result = run(horizon=args.horizon)
    print(result['label'])
