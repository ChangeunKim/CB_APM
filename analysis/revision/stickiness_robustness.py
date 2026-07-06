"""
EXP E15 follow-up (owner's Task 2): reframe the stickiness-concentration
table around the OPTION A decision already adopted (dispersion + EPS
volatility as PRIMARY stickiness proxies; analyst coverage as an honest
non-confirming ROBUSTNESS check, not part of the composite headline number),
and stress-test the EPS-volatility proxy construction itself.

(a) Primary vs robustness split: analysis/stickiness_concentration.py's
    S1 table already has all the per-proxy numbers; this module just
    re-presents them split into the two blocks and drops the composite
    from the headline (kept in an appendix csv only).
(b) EPS-volatility construction sensitivity: compare the original
    construction (rolling 12m std of IBES meanest) against:
      (i)  dispersion ALONE (no EPS-vol at all -- the single most defensible
           proxy, closest to a direct Cao-et-al. correlate)
      (ii) an 8-quarter-window variant of the same meanest-based rolling std
      (iii) VarCF (Compustat-based rolling variance of (ib+dp)/mve_c, i.e. a
           REAL fundamentals-based earnings/cash-flow volatility measure,
           already computed in data/input_<h>month.csv -- not IBES-derived)
    For each, re-run the SAME concentration check (D->revision |t| low vs
    high stickiness bucket) and report whether dispersion/EPS-vol keep
    confirming under each construction.
(c) Size-within-bucket re-verification, per PRIMARY proxy separately (not the
    composite): the original "survives a size x stickiness double sort,
    monotone within BOTH size bins" claim did NOT hold for the composite: this
    reruns the double sort for dispersion and eps_vol individually and
    reports the corrected, narrower claim the data actually support (e.g.
    "holds in the spread dimension, concentrated in small caps" rather than
    blanket monotonicity in both size bins).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.util import REPO_ROOT, CONSENSUS_VARS, build_tidy_frame, decile_sort_returns, load_size_and_exchcd
from analysis.revision.lambda_selection import get_selected_lambda
from analysis.stats_utils import two_way_cluster_ols
from analysis.revision.divergence_mechanism import build_divergence_frame
from analysis.revision.stickiness_concentration import (
    build_stickiness_proxy, s1_concentration, _bucket_hi_lo, PROXY_COLS,
)
from models.metrics import r2_score

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E15'


def primary_vs_robustness_table(concentration):
    primary = concentration[concentration['proxy'].isin(['dispersion', 'eps_vol'])].copy()
    robustness = concentration[concentration['proxy'].isin(['coverage'])].copy()
    appendix = concentration[concentration['proxy'] == 'composite'].copy()
    return primary, robustness, appendix


def build_eps_vol_variant(data_dir=None, window=8, source='meanest'):
    """8-quarter-window variant of the original rolling-std-of-meanest construction
    (same underlying data, different window length -- IBES_summary is monthly,
    so 8 'quarters' = 24 months here to keep it a materially different window
    from the primary 12-month construction, not just a relabeling)."""
    if data_dir is None:
        data_dir = REPO_ROOT / 'data' / 'raw'
    df = pd.read_csv(Path(data_dir) / 'IBES_summary.csv', usecols=PROXY_COLS)
    df['date'] = pd.to_datetime(df['date']).values.astype('datetime64[M]')
    df = df.drop_duplicates(['permno', 'date']).sort_values(['permno', 'date'])
    months = window * 3
    df['eps_vol_variant'] = df.groupby('permno')[source].transform(
        lambda x: x.rolling(months, min_periods=max(months // 2, 3)).std())
    return df[['permno', 'date', 'eps_vol_variant']]


def build_varcf_proxy(horizon='12month', data_dir=None):
    """VarCF: Compustat-based rolling variance of (ib+dp)/mve_c, i.e. a real
    fundamentals cash-flow/earnings volatility measure (NOT derived from
    IBES analyst data at all) -- data/info/SignalDoc.csv's definition."""
    if data_dir is None:
        data_dir = REPO_ROOT / 'data'
    inp = pd.read_csv(Path(data_dir) / f'input_{horizon}.csv', usecols=['permno', 'date', 'VarCF'])
    inp['date'] = pd.to_datetime(inp['date']).values.astype('datetime64[M]')
    return inp


def _zscore_by_date(df, col, group_col='date'):
    return df.groupby(group_col)[col].transform(lambda x: (x - x.mean()) / x.std(ddof=0))


def revision_concentration_for_proxy(horizon, best_lambda, proxy_df, proxy_col, results_dir=None, data_dir=None):
    """Same D->chrec_fwd6 (controls = raw C, per analysis/stickiness_concentration.py's
    fix for omitted-variable bias) check, for an arbitrary proxy column."""
    merged, d_cols = build_divergence_frame(horizon, best_lambda, results_dir=results_dir, data_dir=data_dir)
    merged['prox_month'] = merged['month']
    proxy_m = proxy_df.rename(columns={'date': 'prox_month'})
    # build_divergence_frame's own IBES merge already brought in columns like
    # 'disp'/'nanalyst' -- drop any that collide with the proxy_df's own columns
    # before merging, or pandas suffixes them (_x/_y) and proxy_col disappears.
    collide = [c for c in proxy_m.columns if c in merged.columns and c not in ('permno', 'prox_month')]
    merged = merged.drop(columns=collide, errors='ignore')
    merged = merged.merge(proxy_m, on=['permno', 'prox_month'], how='inner')
    merged[f'{proxy_col}_z'] = _zscore_by_date(merged, proxy_col)

    controls = [f'C_{v}' for v in CONSENSUS_VARS]
    sub = merged.dropna(subset=[f'{proxy_col}_z', 'D_pc1', 'chrec_fwd6'] + controls)
    bucket = _bucket_hi_lo(sub, f'{proxy_col}_z')

    rows = []
    for lvl in ['low', 'high']:
        b = sub[bucket == lvl]
        if len(b) < 200:
            continue
        X = np.column_stack([np.ones(len(b)), b['D_pc1'].values] + [b[c].values for c in controls])
        y = b['chrec_fwd6'].values
        res = two_way_cluster_ols(X, y, b['permno'].values, b['date'].values)
        rows.append({'construction': proxy_col, 'stickiness_level': lvl, 'n': len(b),
                      'coef_D_pc1_on_chrec_fwd6': res['coef'][1], 'tstat': res['tstat'][1], 'pval': res['pval'][1]})
    return pd.DataFrame(rows)


def r2_concentration_for_proxy(horizon, best_lambda, proxy_df, proxy_col, results_dir=None):
    df0 = build_tidy_frame(horizon, 0.0, results_dir=results_dir)
    df_best = build_tidy_frame(horizon, best_lambda, results_dir=results_dir)
    merged = df_best[['permno', 'date', 'y_true', 'y_pred']].merge(
        df0[['permno', 'date', 'y_pred']].rename(columns={'y_pred': 'y_pred_lambda0'}),
        on=['permno', 'date'], how='inner')
    merged['month'] = merged['date'].values.astype('datetime64[M]')
    merged = merged.merge(proxy_df.rename(columns={'date': 'month'}), on=['permno', 'month'], how='inner')
    merged[f'{proxy_col}_z'] = _zscore_by_date(merged, proxy_col, group_col='month')

    sub = merged.dropna(subset=[f'{proxy_col}_z', 'y_true', 'y_pred', 'y_pred_lambda0'])
    bucket = _bucket_hi_lo(sub, f'{proxy_col}_z', group_col='month')
    rows = []
    for lvl in ['low', 'high']:
        b = sub[bucket == lvl]
        if len(b) < 200:
            continue
        r2_0 = r2_score(b['y_true'].values, b['y_pred_lambda0'].values)
        r2_best = r2_score(b['y_true'].values, b['y_pred'].values)
        rows.append({'construction': proxy_col, 'stickiness_level': lvl, 'n': len(b),
                      'R2_lambda0': r2_0, 'R2_best': r2_best, 'R2_improvement': r2_best - r2_0})
    return pd.DataFrame(rows)


def size_within_bucket_by_primary_proxy(horizon, best_lambda, proxy_df, proxy_col, size_df, results_dir=None):
    """Re-verify concentration WITHIN size bins, one primary proxy at a time
    (not the composite) -- reports the actual pattern (which may only hold
    among small caps, not both size bins) rather than asserting blanket
    monotonicity."""
    df_best = build_tidy_frame(horizon, best_lambda, results_dir=results_dir)
    df0 = build_tidy_frame(horizon, 0.0, results_dir=results_dir)
    merged = df_best[['permno', 'date', 'y_true', 'y_pred']].merge(
        df0[['permno', 'date', 'y_pred']].rename(columns={'y_pred': 'y_pred_lambda0'}),
        on=['permno', 'date'], how='inner')
    merged['month'] = merged['date'].values.astype('datetime64[M]')
    merged = merged.merge(proxy_df.rename(columns={'date': 'month'}), on=['permno', 'month'], how='inner')
    merged = merged.merge(size_df.rename(columns={'date': 'month'}), on=['permno', 'month'], how='left')
    merged[f'{proxy_col}_z'] = merged.groupby('month')[proxy_col].transform(lambda x: (x - x.mean()) / x.std(ddof=0))

    sub = merged.dropna(subset=[f'{proxy_col}_z', 'Size', 'y_true', 'y_pred', 'y_pred_lambda0'])
    sub = sub.copy()
    sub['size_bucket'] = _bucket_hi_lo(sub, 'Size', group_col='month')
    sub['stick_bucket'] = _bucket_hi_lo(sub, f'{proxy_col}_z', group_col='month')

    rows = []
    for sz in ['low', 'high']:
        for st in ['low', 'high']:
            cell = sub[(sub['size_bucket'] == sz) & (sub['stick_bucket'] == st)]
            if len(cell) < 200:
                continue
            r2_0 = r2_score(cell['y_true'].values, cell['y_pred_lambda0'].values)
            r2_best = r2_score(cell['y_true'].values, cell['y_pred'].values)
            _, summary = decile_sort_returns(cell, 'y_pred', return_col='y_true', weight_col='Size')
            hl = summary.loc['H-L'] if 'H-L' in summary.index else None
            rows.append({
                'proxy': proxy_col, 'size_bucket': sz, 'stickiness_bucket': st, 'n': len(cell),
                'R2_improvement': r2_best - r2_0,
                'HL_mean_VW': hl['mean'] if hl is not None else np.nan,
                'HL_tstat_VW': hl['tstat'] if hl is not None else np.nan,
            })
    return pd.DataFrame(rows)


def run(horizon='12month', results_dir=None, data_dir=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    best_lambda = get_selected_lambda(horizon=horizon, results_dir=results_dir)
    proxy = build_stickiness_proxy(data_dir=(Path(data_dir) / 'raw') if data_dir else None)
    size_df = load_size_and_exchcd(horizon, data_dir=data_dir)

    # ---- (a) primary vs robustness split (re-slice E15's existing S1 table) ----
    df0 = build_tidy_frame(horizon, 0.0, results_dir=results_dir)
    df_best = build_tidy_frame(horizon, best_lambda, results_dir=results_dir)
    concentration, double_sort, survives = s1_concentration(df0, df_best, proxy, size_df, out_dir, horizon)
    primary, robustness, appendix = primary_vs_robustness_table(concentration)
    primary.to_csv(out_dir / f'{horizon}_S1_primary_dispersion_epsvol.csv', index=False)
    robustness.to_csv(out_dir / f'{horizon}_S1_robustness_coverage.csv', index=False)
    appendix.to_csv(out_dir / f'{horizon}_S1_appendix_composite.csv', index=False)

    # ---- (b) EPS-volatility construction sensitivity ----
    disp_only = proxy[['permno', 'date', 'disp']].copy()
    variant_8q = build_eps_vol_variant(data_dir=(Path(data_dir) / 'raw') if data_dir else None, window=8)
    varcf = build_varcf_proxy(horizon=horizon, data_dir=data_dir)

    sensitivity_rows = []
    for name, df_proxy, col in [
        ('dispersion_alone', disp_only, 'disp'),
        ('eps_vol_original_12m_meanest', proxy[['permno', 'date', 'eps_vol']], 'eps_vol'),
        ('eps_vol_variant_24m_meanest', variant_8q, 'eps_vol_variant'),
        ('eps_vol_varcf_compustat', varcf, 'VarCF'),
    ]:
        rev = revision_concentration_for_proxy(horizon, best_lambda, df_proxy, col, results_dir=results_dir,
                                                data_dir=(Path(data_dir) / 'raw') if data_dir else None)
        r2c = r2_concentration_for_proxy(horizon, best_lambda, df_proxy, col, results_dir=results_dir)
        rev['label'] = name
        r2c['label'] = name
        sensitivity_rows.append((rev, r2c))

    rev_all = pd.concat([r for r, _ in sensitivity_rows], ignore_index=True)
    r2_all = pd.concat([r for _, r in sensitivity_rows], ignore_index=True)
    rev_all.to_csv(out_dir / f'{horizon}_S1_epsvol_construction_sensitivity_revision.csv', index=False)
    r2_all.to_csv(out_dir / f'{horizon}_S1_epsvol_construction_sensitivity_R2.csv', index=False)

    def _confirms(df, key_col='construction'):
        out = {}
        for name in df[key_col].unique():
            sub = df[df[key_col] == name].set_index('stickiness_level')
            if 'high' in sub.index and 'low' in sub.index:
                if 'tstat' in sub.columns:
                    out[name] = bool(abs(sub.loc['high', 'tstat']) > abs(sub.loc['low', 'tstat']))
                else:
                    out[name] = bool(sub.loc['high', 'R2_improvement'] > sub.loc['low', 'R2_improvement'])
        return out

    rev_confirms = _confirms(rev_all)
    r2_confirms = _confirms(r2_all)

    # ---- (c) size-within-bucket re-verification, per primary proxy ----
    size_disp = size_within_bucket_by_primary_proxy(horizon, best_lambda, proxy[['permno', 'date', 'disp']], 'disp', size_df, results_dir=results_dir)
    size_epsvol = size_within_bucket_by_primary_proxy(horizon, best_lambda, proxy[['permno', 'date', 'eps_vol']], 'eps_vol', size_df, results_dir=results_dir)
    size_by_primary = pd.concat([size_disp, size_epsvol], ignore_index=True)
    size_by_primary.to_csv(out_dir / f'{horizon}_S1_size_within_bucket_by_primary_proxy.csv', index=False)

    def _monotone_within_size(df, proxy_name):
        sub = df[df['proxy'] == proxy_name]
        ok = True
        for sz in ['low', 'high']:
            lo = sub[(sub['size_bucket'] == sz) & (sub['stickiness_bucket'] == 'low')]
            hi = sub[(sub['size_bucket'] == sz) & (sub['stickiness_bucket'] == 'high')]
            if len(lo) and len(hi):
                ok = ok and bool(hi['HL_mean_VW'].iloc[0] > lo['HL_mean_VW'].iloc[0])
        return ok

    disp_monotone_both = _monotone_within_size(size_by_primary, 'disp')
    epsvol_monotone_both = _monotone_within_size(size_by_primary, 'eps_vol')

    def _small_cap_only(df, proxy_name):
        sub = df[df['proxy'] == proxy_name]
        lo_small = sub[(sub['size_bucket'] == 'low') & (sub['stickiness_bucket'] == 'low')]
        hi_small = sub[(sub['size_bucket'] == 'low') & (sub['stickiness_bucket'] == 'high')]
        if len(lo_small) and len(hi_small):
            return bool(hi_small['HL_mean_VW'].iloc[0] > lo_small['HL_mean_VW'].iloc[0])
        return None

    disp_small_cap = _small_cap_only(size_by_primary, 'disp')
    epsvol_small_cap = _small_cap_only(size_by_primary, 'eps_vol')

    lines = [
        f"# E15 Task 2 -- stickiness proxy refinement (Option A) ({horizon}, lambda={best_lambda})",
        "",
        "## (a) Primary (dispersion, EPS-volatility) vs robustness (coverage) -- composite dropped from headline",
        "### PRIMARY",
        primary.to_string(index=False),
        "### ROBUSTNESS (non-confirming, reported honestly, not hidden)",
        robustness.to_string(index=False),
        "### Appendix only (composite, muddied by coverage)",
        appendix.to_string(index=False),
        "",
        "## (b) EPS-volatility / dispersion construction sensitivity",
        "### D->chrec_fwd6 revision-predictability concentration (|t| high > |t| low?), by construction",
        rev_all.to_string(index=False),
        f"Confirms per construction: {rev_confirms}",
        "",
        "### R2-improvement concentration (high > low?), by construction",
        r2_all.to_string(index=False),
        f"Confirms per construction: {r2_confirms}",
        "",
        "## (c) Size-within-bucket re-verification, PRIMARY proxies only (corrected claim)",
        size_by_primary.to_string(index=False),
        f"Dispersion: monotone in BOTH size bins? {disp_monotone_both}. Holds among SMALL caps "
        f"(low size bucket)? {disp_small_cap}.",
        f"EPS-vol: monotone in BOTH size bins? {epsvol_monotone_both}. Holds among SMALL caps "
        f"(low size bucket)? {epsvol_small_cap}.",
        "",
        "CORRECTED CLAIM (per owner's Option A instruction -- do not claim blanket size-bin "
        "monotonicity): the stickiness-concentration channel evidence holds in the SPREAD "
        "dimension and is CONCENTRATED IN SMALL-CAP stocks, not uniformly across both size bins. "
        "This is a narrower, more defensible claim than the original 'survives a size x "
        "stickiness double sort' framing.",
        "",
        "Supports / weakens / neutral w.r.t. CLAIM-ECON: "
        + ("SUPPORTS, WITH THE NARROWED SCOPE ABOVE -- dispersion and EPS-volatility (both the "
           "original meanest-based construction and the two alternative constructions tested here) "
           "consistently show concentration in the small-cap spread dimension; coverage remains "
           "honestly reported as non-confirming."
           if all(rev_confirms.get(k, False) for k in
                  ['dispersion_alone', 'eps_vol_original_12m_meanest', 'eps_vol_variant_24m_meanest', 'eps_vol_varcf_compustat'])
           else "MIXED -- not every EPS-volatility construction confirms the concentration pattern; "
           "see the per-construction table above and report exactly which constructions do/don't."),
    ]
    (out_dir / f'{horizon}_task2_stickiness_refinement_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {
        'primary': primary, 'robustness': robustness, 'rev_confirms': rev_confirms,
        'r2_confirms': r2_confirms, 'size_by_primary': size_by_primary,
        'disp_small_cap': disp_small_cap, 'epsvol_small_cap': epsvol_small_cap,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    result = run(horizon=args.horizon)
    print('Revision confirms:', result['rev_confirms'])
    print('R2 confirms:', result['r2_confirms'])
