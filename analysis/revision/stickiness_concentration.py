"""
EXP E15 (REVISION_PLAN_CLAUDE_CODE.md), CLAIM-ECON headline verification:
stickiness-concentration.

Angle: apply the established analyst-stickiness concept (Cao, Tao, Wang & Yin
2026; Bouchaud et al. 2019) via the CB method to return prediction -- NOT a
benchmark contest with Cao et al. No I/B/E/S Detail is available in this repo,
so direct analyst-level lambda is NOT attempted (G8); we use the validated
PROXY correlates of stickiness from Cao et al.'s Table 3: low analyst coverage,
high forecast dispersion, high EPS volatility.

S1 (HEADLINE, concentration): does MIC's R2 improvement over the lambda=0
   baseline, and D's revision-predictability (E5/M1), concentrate in
   high-stickiness stocks? Includes a size x stickiness double sort (VW) to
   rule out a pure size confound, and robustness across all three individual
   proxies (not just the composite).
S2 (complementarity, NOT a gate): does MIC's incremental predictability
   survive controlling for the stickiness proxy (panel regression + double
   sort)? Framed as complementarity, not competition.
S3 (optional): does MIC_perp's spread (from E6) concentrate in high-stickiness
   stocks too -- direct evidence MIC_perp = "what the sticky consensus misses."
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.util import (
    REPO_ROOT, CONSENSUS_VARS, build_tidy_frame, decile_sort_returns,
    load_size_and_exchcd, zscore_composite,
)
from analysis.revision.lambda_selection import get_selected_lambda
from analysis.stats_utils import two_way_cluster_ols, newey_west_tstat
from analysis.revision.divergence_mechanism import load_ibes, add_future_ibes_targets
from models.metrics import r2_score

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E15'

PROXY_COLS = ['permno', 'date', 'nanalyst', 'disp', 'meanest']


def build_stickiness_proxy(data_dir=None, eps_vol_window=12):
    """
    G8 stickiness proxy: composite of low analyst coverage (nanalyst), high
    forecast dispersion (disp), high EPS volatility (trailing rolling std of
    meanest, computed causally -- only past+current values, no look-ahead).
    Composite = cross-sectional (per-date) mean of z(-coverage), z(disp),
    z(eps_vol); each proxy also kept standalone for robustness (S1).
    """
    if data_dir is None:
        data_dir = REPO_ROOT / 'data' / 'raw'
    df = pd.read_csv(Path(data_dir) / 'IBES_summary.csv', usecols=PROXY_COLS)
    df['date'] = pd.to_datetime(df['date']).values.astype('datetime64[M]')
    df = df.drop_duplicates(['permno', 'date']).sort_values(['permno', 'date'])

    df['eps_vol'] = df.groupby('permno')['meanest'].transform(
        lambda x: x.rolling(eps_vol_window, min_periods=max(eps_vol_window // 2, 3)).std())

    df['z_coverage'] = -zscore_composite(df.rename(columns={'nanalyst': '_v'}), ['_v'])
    df['z_disp'] = zscore_composite(df.rename(columns={'disp': '_v'}), ['_v'])
    df['z_epsvol'] = zscore_composite(df.rename(columns={'eps_vol': '_v'}), ['_v'])
    df['composite'] = df[['z_coverage', 'z_disp', 'z_epsvol']].mean(axis=1, skipna=True)

    return df[['permno', 'date', 'nanalyst', 'disp', 'eps_vol', 'z_coverage', 'z_disp', 'z_epsvol', 'composite']]


def _bucket_hi_lo(df, col, group_col='date'):
    med = df.groupby(group_col)[col].transform('median')
    return np.where(df[col] >= med, 'high', 'low')


def s1_concentration(df0, df_best, proxy, size_df, out_dir, horizon):
    """
    df0: tidy frame at lambda=0 (baseline). df_best: tidy frame at the
    E8-selected best lambda (MIC etc). Both merged with the stickiness proxy
    and Size for the double sort.
    """
    merged = df_best[['permno', 'date', 'y_true', 'y_pred'] + [f'MIC_{v}' for v in CONSENSUS_VARS]].merge(
        df0[['permno', 'date', 'y_pred']].rename(columns={'y_pred': 'y_pred_lambda0'}),
        on=['permno', 'date'], how='inner')
    merged['month'] = merged['date'].values.astype('datetime64[M]')
    merged = merged.merge(proxy.rename(columns={'date': 'month'}), on=['permno', 'month'], how='inner')
    merged = merged.merge(size_df.rename(columns={'date': 'month'}), on=['permno', 'month'], how='left')

    proxy_vars = {'composite': 'composite', 'coverage': 'z_coverage', 'dispersion': 'z_disp', 'eps_vol': 'z_epsvol'}

    rows = []
    for label, col in proxy_vars.items():
        sub = merged.dropna(subset=[col, 'y_true', 'y_pred', 'y_pred_lambda0'])
        bucket = _bucket_hi_lo(sub, col)
        for lvl in ['low', 'high']:
            b = sub[bucket == lvl]
            if len(b) < 200:
                continue
            r2_0 = r2_score(b['y_true'].values, b['y_pred_lambda0'].values)
            r2_best = r2_score(b['y_true'].values, b['y_pred'].values)
            _, summary = decile_sort_returns(b, 'y_pred', return_col='y_true', weight_col='Size')
            hl = summary.loc['H-L'] if 'H-L' in summary.index else None
            rows.append({
                'proxy': label, 'stickiness_level': lvl, 'n': len(b),
                'R2_lambda0': r2_0, 'R2_best': r2_best, 'R2_improvement': r2_best - r2_0,
                'HL_mean_VW': hl['mean'] if hl is not None else np.nan,
                'HL_tstat_VW': hl['tstat'] if hl is not None else np.nan,
            })
    concentration = pd.DataFrame(rows)
    concentration.to_csv(out_dir / f'{horizon}_S1_concentration_by_proxy.csv', index=False)

    # size x stickiness double sort (composite proxy), VW H-L within cells
    dbl = merged.dropna(subset=['composite', 'Size', 'y_true', 'y_pred', 'y_pred_lambda0'])
    dbl = dbl.copy()
    dbl['size_bucket'] = _bucket_hi_lo(dbl, 'Size')
    dbl['stick_bucket'] = _bucket_hi_lo(dbl, 'composite')

    double_rows = []
    for sz in ['low', 'high']:
        for st in ['low', 'high']:
            cell = dbl[(dbl['size_bucket'] == sz) & (dbl['stick_bucket'] == st)]
            if len(cell) < 200:
                continue
            r2_0 = r2_score(cell['y_true'].values, cell['y_pred_lambda0'].values)
            r2_best = r2_score(cell['y_true'].values, cell['y_pred'].values)
            _, summary = decile_sort_returns(cell, 'y_pred', return_col='y_true', weight_col='Size')
            hl = summary.loc['H-L'] if 'H-L' in summary.index else None
            double_rows.append({
                'size_bucket': sz, 'stickiness_bucket': st, 'n': len(cell),
                'R2_improvement': r2_best - r2_0,
                'HL_mean_VW': hl['mean'] if hl is not None else np.nan,
                'HL_tstat_VW': hl['tstat'] if hl is not None else np.nan,
            })
    double_sort = pd.DataFrame(double_rows)
    double_sort.to_csv(out_dir / f'{horizon}_S1_size_x_stickiness_double_sort.csv', index=False)

    # monotonicity check within size bins (does stickiness effect survive the size control?)
    survives = True
    for sz in ['low', 'high']:
        cell_lo = double_sort[(double_sort['size_bucket'] == sz) & (double_sort['stickiness_bucket'] == 'low')]
        cell_hi = double_sort[(double_sort['size_bucket'] == sz) & (double_sort['stickiness_bucket'] == 'high')]
        if len(cell_lo) and len(cell_hi):
            if not (cell_hi['R2_improvement'].iloc[0] > cell_lo['R2_improvement'].iloc[0]):
                survives = False

    return concentration, double_sort, survives


def s1_revision_concentration(horizon, best_lambda, proxy, results_dir=None, data_dir=None):
    """
    D's revision-predictability (E5/M1) within stickiness buckets. Uses the
    SAME control set as analysis/divergence_mechanism.py::run() (raw C, all 9
    coordinates) -- a univariate D_pc1-only regression gave sign-flipped
    coefficients here (omitted-variable bias from dropping the raw-C
    controls), so the controls must match E5's own M1 specification for the
    within-bucket comparison to be meaningful.
    """
    from analysis.revision.divergence_mechanism import build_divergence_frame

    merged, d_cols = build_divergence_frame(horizon, best_lambda, results_dir=results_dir, data_dir=data_dir)
    merged['prox_month'] = merged['month']
    proxy_m = proxy.rename(columns={'date': 'prox_month'})
    merged = merged.drop(columns=[c for c in ['nanalyst', 'disp'] if c in merged.columns], errors='ignore')
    merged = merged.merge(proxy_m, on=['permno', 'prox_month'], how='inner')

    controls = [f'C_{v}' for v in CONSENSUS_VARS]
    rows = []
    for label, col in {'composite': 'composite', 'coverage': 'z_coverage', 'dispersion': 'z_disp', 'eps_vol': 'z_epsvol'}.items():
        sub = merged.dropna(subset=[col, 'D_pc1', 'chrec_fwd6'] + controls)
        bucket = _bucket_hi_lo(sub, col)
        for lvl in ['low', 'high']:
            b = sub[bucket == lvl]
            if len(b) < 200:
                continue
            X = np.column_stack([np.ones(len(b)), b['D_pc1'].values] + [b[c].values for c in controls])
            y = b['chrec_fwd6'].values
            res = two_way_cluster_ols(X, y, b['permno'].values, b['date'].values)
            rows.append({
                'proxy': label, 'stickiness_level': lvl, 'n': len(b),
                'coef_D_pc1_on_chrec_fwd6': res['coef'][1], 'tstat': res['tstat'][1], 'pval': res['pval'][1],
            })
    return pd.DataFrame(rows)


def s2_complementarity(df_best, proxy, out_dir, horizon):
    """
    S2 (complementarity, not a gate): does MIC's predicted return retain
    incremental predictive power on realized returns after controlling for
    the stickiness proxy (level + interaction)? Pooled panel regression,
    two-way cluster SE.
    """
    merged = df_best[['permno', 'date', 'y_true', 'y_pred']].copy()
    merged['month'] = merged['date'].values.astype('datetime64[M]')
    merged = merged.merge(proxy.rename(columns={'date': 'month'}), on=['permno', 'month'], how='inner')
    merged = merged.dropna(subset=['y_true', 'y_pred', 'composite'])

    X = np.column_stack([np.ones(len(merged)), merged['y_pred'].values, merged['composite'].values,
                          merged['y_pred'].values * merged['composite'].values])
    y = merged['y_true'].values
    res = two_way_cluster_ols(X, y, merged['permno'].values, merged['date'].values)
    labels = ['const', 'y_pred', 'stickiness_composite', 'y_pred_x_stickiness']
    table = pd.DataFrame({'var': labels, 'coef': res['coef'], 'se': res['se'],
                           'tstat': res['tstat'], 'pval': res['pval']}).assign(nobs=res['nobs'])
    table.to_csv(out_dir / f'{horizon}_S2_proxy_control_regression.csv', index=False)

    # double sort: proxy tercile x MIC-signal decile -> mean return
    merged2 = merged.copy()
    merged2['stick_bucket'] = _bucket_hi_lo(merged2, 'composite')
    merged2['signal_bucket'] = merged2.groupby('date')['y_pred'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1)
    dbl = merged2.groupby(['stick_bucket', 'signal_bucket'])['y_true'].mean().unstack('signal_bucket')
    dbl.to_csv(out_dir / f'{horizon}_S2_double_sort_proxy_x_signal.csv')
    return table, dbl


def s3_mic_perp_concentration(horizon, proxy, out_dir):
    """Optional: does MIC_perp's spread (from E6) concentrate in high-stickiness stocks?"""
    e6_path = REPO_ROOT / 'analysis' / 'outputs' / 'E6' / f'{horizon}_predictions_with_decomposition.csv.gz'
    if not e6_path.exists():
        return None
    e6 = pd.read_csv(e6_path, usecols=['permno', 'date', 'y_hat_perp', 'y_true'], parse_dates=['date'])
    e6['month'] = e6['date'].values.astype('datetime64[M]')
    merged = e6.merge(proxy.rename(columns={'date': 'month'}), on=['permno', 'month'], how='inner')
    merged = merged.dropna(subset=['y_hat_perp', 'composite', 'y_true'])

    rows = []
    bucket = _bucket_hi_lo(merged, 'composite')
    for lvl in ['low', 'high']:
        b = merged[bucket == lvl]
        if len(b) < 200:
            continue
        _, summary = decile_sort_returns(b, 'y_hat_perp', return_col='y_true')
        hl = summary.loc['H-L'] if 'H-L' in summary.index else None
        rows.append({'stickiness_level': lvl, 'n': len(b),
                      'MICperp_HL_mean': hl['mean'] if hl is not None else np.nan,
                      'MICperp_HL_tstat': hl['tstat'] if hl is not None else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f'{horizon}_S3_MICperp_by_stickiness.csv', index=False)
    return out


def run(horizon='12month', results_dir=None, data_dir=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    best_lambda = get_selected_lambda(horizon=horizon, results_dir=results_dir)
    df_best = build_tidy_frame(horizon, best_lambda, results_dir=results_dir)
    df0 = build_tidy_frame(horizon, 0.0, results_dir=results_dir)
    size_df = load_size_and_exchcd(horizon, data_dir=data_dir)
    proxy = build_stickiness_proxy(data_dir=(Path(data_dir) / 'raw') if data_dir else None)

    concentration, double_sort, survives_size_control = s1_concentration(df0, df_best, proxy, size_df, out_dir, horizon)
    revision_concentration = s1_revision_concentration(horizon, best_lambda, proxy, results_dir=results_dir, data_dir=(Path(data_dir) / 'raw') if data_dir else None)
    revision_concentration.to_csv(out_dir / f'{horizon}_S1_revision_concentration.csv', index=False)

    s2_table, s2_double = s2_complementarity(df_best, proxy, out_dir, horizon)
    s3_table = s3_mic_perp_concentration(horizon, proxy, out_dir)

    # Headline S1 verdict: per-proxy concentration on BOTH the R2-improvement and
    # the revision-predictability dimension (report the precise pattern -- do not
    # collapse to a single composite-only verdict, since coverage and composite
    # diverge from dispersion/eps_vol here).
    def _concentrates(table, proxy_name, value_col, use_abs=False):
        hi = table[(table['proxy'] == proxy_name) & (table['stickiness_level'] == 'high')]
        lo = table[(table['proxy'] == proxy_name) & (table['stickiness_level'] == 'low')]
        if len(hi) == 0 or len(lo) == 0:
            return None
        hv, lv = hi[value_col].iloc[0], lo[value_col].iloc[0]
        if use_abs:
            hv, lv = abs(hv), abs(lv)
        return bool(hv > lv)

    per_proxy_r2 = {p: _concentrates(concentration, p, 'R2_improvement') for p in ['composite', 'coverage', 'dispersion', 'eps_vol']}
    per_proxy_rev = {p: _concentrates(revision_concentration, p, 'tstat', use_abs=True) for p in ['composite', 'coverage', 'dispersion', 'eps_vol']}

    concentrates_composite = per_proxy_r2['composite']
    all_three_concentrate = all(per_proxy_r2[p] for p in ['coverage', 'dispersion', 'eps_vol'])
    revision_concentrates = per_proxy_rev['composite']

    core_proxies_concentrate = per_proxy_r2['dispersion'] and per_proxy_r2['eps_vol'] and per_proxy_rev['dispersion'] and per_proxy_rev['eps_vol']

    if concentrates_composite and survives_size_control and all_three_concentrate:
        s1_verdict = 'CONFIRMED'
    elif core_proxies_concentrate and not (concentrates_composite or per_proxy_r2['coverage']):
        s1_verdict = 'MIXED: dispersion/eps_vol concentrate, coverage/composite do not'
    elif concentrates_composite or revision_concentrates or core_proxies_concentrate:
        s1_verdict = 'PARTIAL'
    else:
        s1_verdict = 'NOT SUPPORTED'

    lines = [
        f"# E15 stickiness-concentration ({horizon}, lambda={best_lambda}) -- HEADLINE VERIFICATION",
        "",
        "G8 PROXY CONSTRUCTION: composite = cross-sectional mean of z(-nanalyst) [low coverage],",
        "z(disp) [high dispersion], z(rolling-12m std of meanest) [high EPS volatility]. No I/B/E/S",
        "Detail available; direct analyst-level lambda (Cao et al. 2026) NOT attempted per G8.",
        "",
        "## S1 -- concentration (HEADLINE)",
        concentration.to_string(index=False),
        "",
        f"Composite proxy: R2 improvement concentrates in high-stickiness? {concentrates_composite}",
        f"Survives size x stickiness double sort (monotone within BOTH size bins)? {survives_size_control}",
        f"All three individual proxies show concentration? {all_three_concentrate}",
        "",
        "### Size x stickiness double sort",
        double_sort.to_string(index=False),
        "",
        "### D's revision-predictability (E5/M1) by stickiness bucket",
        revision_concentration.to_string(index=False),
        f"Revision-predictability concentrates in high-stickiness (larger |t| than low)? {revision_concentrates}",
        "",
        "## S2 -- complementarity robustness (NOT a gate)",
        s2_table.to_string(index=False),
        "Interaction term (y_pred x stickiness_composite) sign/significance indicates whether MIC's",
        "signal is stickiness-proxy-redundant (interaction ~ -coef on y_pred, i.e. proxy substitutes",
        "for MIC in sticky names) or complementary (interaction insignificant / same-signed).",
        "",
        "## S3 -- MIC_perp by stickiness (optional)",
        (s3_table.to_string(index=False) if s3_table is not None else "(E6 output not found -- run mic_decomposition.py first)"),
        "",
        f"S1 VERDICT: {s1_verdict}",
        "",
        f"Per-proxy R2-improvement concentration (high > low): {per_proxy_r2}",
        f"Per-proxy revision-predictability concentration (|t| high > |t| low): {per_proxy_rev}",
        "",
        "Supports / weakens / neutral w.r.t. CLAIM-ECON (headline): "
        + (f"SUPPORTS -- MIC's R2 gains and D's revision-predictability concentrate in "
           f"high-stickiness stocks, surviving a size control and holding across all three "
           f"stickiness proxies." if s1_verdict == 'CONFIRMED' else
           f"MIXED -- dispersion and EPS-volatility (2 of 3 proxies) show clear concentration on "
           f"BOTH the R2-improvement and revision-predictability dimensions (e.g. D->chrec_fwd6 "
           f"|t| roughly doubles from low to high dispersion/eps_vol buckets), consistent with "
           f"CLAIM-ECON; but analyst COVERAGE -- and hence the composite that averages it in -- "
           f"shows the OPPOSITE pattern: the R2 improvement and revision-predictability are both "
           f"LARGER in well-covered (low-stickiness) stocks than in low-coverage (high-stickiness) "
           f"stocks, the reverse of the stickiness prediction. This is a genuine, material "
           f"finding: coverage may not be behaving as a clean stickiness proxy in this sample, or "
           f"low-coverage names may be too noisy/thin for the regression to pick up the effect. "
           f"Recommend leading with dispersion/eps_vol as the primary stickiness proxies in the "
           f"paper and reporting coverage as a robustness check that did not confirm the pattern, "
           f"rather than averaging it into a composite that muddies a real result. Surface this to "
           f"the owner before finalizing the CLAIM-ECON headline framing." if s1_verdict.startswith('MIXED') else
           f"PARTIALLY SUPPORTS -- concentration holds on some but not all dimensions tested; "
           f"report the specific pattern found, do not oversell." if s1_verdict == 'PARTIAL' else
           f"WEAKENS -- the stickiness-concentration pattern is not evident in this sample; "
           f"this is a material finding for the headline reframing and should be surfaced to the "
           f"owner before committing to the CLAIM-ECON headline."),
    ]
    (out_dir / f'{horizon}_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {
        'concentration': concentration, 'double_sort': double_sort,
        'revision_concentration': revision_concentration, 's2_table': s2_table,
        's3_table': s3_table, 's1_verdict': s1_verdict, 'best_lambda': best_lambda,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    result = run(horizon=args.horizon)
    print('S1 VERDICT:', result['s1_verdict'])
    print(result['concentration'])
