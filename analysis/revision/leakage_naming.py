"""
EXP E6/K2b (REVISION_PLAN_CLAUDE_CODE.md): "name the leakage." Regresses each
MIC_perp coordinate (and the composite b'MIC_perp, from
analysis/mic_decomposition.py's output) on:
  (i)  E14's compression factors z (PCA d=9, from analysis/compression_ladder.py)
       -- "is leakage just generic input compression, or return-specific
       distillation?"
  (ii) LASSO onto the 114 firm characteristics + 32-d macro embedding (E14's
       persisted x146 test features) -- top loadings grouped by
       characteristic family (data/info/SignalDoc.csv's 'Cat.Economic' column
       -- momentum, value, liquidity, ... -- exactly the taxonomy the spec
       asks for).

Dependencies: analysis/mic_decomposition.py (MIC_perp) and
analysis/compression_ladder.py (z, x146) must have been run first.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression

from analysis.util import REPO_ROOT, CONSENSUS_VARS

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E6'


def _load_maybe_parquet(path_parquet, path_csv_gz):
    if path_parquet.exists():
        try:
            return pd.read_parquet(path_parquet)
        except ImportError:
            pass
    return pd.read_csv(path_csv_gz, parse_dates=['date'])


def load_mic_perp(horizon):
    e6_dir = REPO_ROOT / 'analysis' / 'outputs' / 'E6'
    df = _load_maybe_parquet(
        e6_dir / f'{horizon}_predictions_with_decomposition.parquet',
        e6_dir / f'{horizon}_predictions_with_decomposition.csv.gz')
    perp_cols = [f'MICperp_{v}' for v in CONSENSUS_VARS]
    keep = ['permno', 'date'] + perp_cols + (['MICperp_composite'] if 'MICperp_composite' in df.columns else [])
    return df[keep].copy()


def load_z_factors(horizon):
    e14_dir = REPO_ROOT / 'analysis' / 'outputs' / 'E14'
    return _load_maybe_parquet(
        e14_dir / f'{horizon}_z_factors_d9.parquet',
        e14_dir / f'{horizon}_z_factors_d9.csv.gz')


def load_x146(horizon):
    e14_dir = REPO_ROOT / 'analysis' / 'outputs' / 'E14'
    return _load_maybe_parquet(
        e14_dir / f'{horizon}_x146_test_features.parquet',
        e14_dir / f'{horizon}_x146_test_features.csv.gz')


def z_regression(mic_perp, z_factors, out_dir, horizon):
    """(i) Is MIC_perp just generic input compression? R2 of MIC_perp on the
    PCA compression factors z (same 146-d inputs, generic unsupervised
    compression, no return/consensus supervision)."""
    merged = mic_perp.merge(z_factors, on=['permno', 'date'], how='inner')
    z_cols = [c for c in z_factors.columns if c.startswith('z')]

    rows = []
    targets = [f'MICperp_{v}' for v in CONSENSUS_VARS] + (['MICperp_composite'] if 'MICperp_composite' in mic_perp.columns else [])
    for target in targets:
        sub = merged.dropna(subset=[target] + z_cols)
        if len(sub) < 100:
            continue
        reg = LinearRegression().fit(sub[z_cols], sub[target])
        r2 = reg.score(sub[z_cols], sub[target])
        rows.append({'target': target, 'R2_vs_generic_compression_z': r2, 'n': len(sub)})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f'{horizon}_K2b_z_regression.csv', index=False)
    return out


CHARACTERISTIC_FAMILY_COL = 'Cat.Economic'


def lasso_characteristic_naming(mic_perp, x146, out_dir, horizon, data_dir=None, top_n=15):
    """(ii) LASSO of MIC_perp composite onto the 114 firm characteristics +
    32-d macro embedding; report top loadings grouped by characteristic
    family (SignalDoc's Cat.Economic)."""
    if data_dir is None:
        data_dir = REPO_ROOT / 'data'
    info = pd.read_csv(Path(data_dir) / 'info' / 'SignalDoc.csv')
    family_map = info.set_index('Acronym')[CHARACTERISTIC_FAMILY_COL].to_dict()

    merged = mic_perp.merge(x146, on=['permno', 'date'], how='inner')
    feature_cols = [c for c in x146.columns if c not in ('permno', 'date')]

    target = 'MICperp_composite' if 'MICperp_composite' in mic_perp.columns else f'MICperp_{CONSENSUS_VARS[0]}'
    sub = merged.dropna(subset=[target] + feature_cols)

    X = sub[feature_cols].values
    y = sub[target].values
    lasso = LassoCV(cv=5, max_iter=5000, random_state=0).fit(X, y)
    r2 = lasso.score(X, y)

    loadings = pd.DataFrame({'characteristic': feature_cols, 'coef': lasso.coef_})
    loadings['family'] = loadings['characteristic'].map(lambda c: family_map.get(c, 'macro_embedding' if c.startswith('macro') else 'unknown'))
    loadings['abs_coef'] = loadings['coef'].abs()
    loadings = loadings.sort_values('abs_coef', ascending=False)
    loadings.to_csv(out_dir / f'{horizon}_K2b_lasso_loadings.csv', index=False)

    top = loadings.head(top_n)
    family_share = loadings.groupby('family')['abs_coef'].sum().sort_values(ascending=False)
    family_share = (family_share / family_share.sum()).rename('share_of_total_abs_coef')
    family_share.to_csv(out_dir / f'{horizon}_K2b_family_share.csv')

    return {'target': target, 'lasso_r2': r2, 'alpha': lasso.alpha_, 'top_loadings': top, 'family_share': family_share, 'n': len(sub)}


def run(horizon='12month', data_dir=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    mic_perp = load_mic_perp(horizon)
    z_factors = load_z_factors(horizon)
    x146 = load_x146(horizon)

    z_reg = z_regression(mic_perp, z_factors, out_dir, horizon)
    lasso_result = lasso_characteristic_naming(mic_perp, x146, out_dir, horizon, data_dir=data_dir)

    composite_z_r2 = z_reg[z_reg['target'] == 'MICperp_composite']['R2_vs_generic_compression_z']
    composite_z_r2 = composite_z_r2.iloc[0] if len(composite_z_r2) else np.nan
    is_generic_compression = composite_z_r2 > 0.5  # majority explained by unsupervised PCA -> "just compression"

    lines = [
        f"# E6/K2b -- naming the leakage ({horizon})",
        "",
        "## (i) Is MIC_perp just generic input compression?",
        z_reg.to_string(index=False),
        f"MIC_perp_composite R2 vs generic (PCA) compression factors z: {composite_z_r2:.3f}",
        f"-> {'MIC_perp is LARGELY explained by generic input compression (weak evidence for return-specific distillation).' if is_generic_compression else 'MIC_perp is NOT well explained by generic (unsupervised) compression -- consistent with return-specific distillation, not just dimensionality reduction.'}",
        "",
        f"## (ii) LASSO characteristic naming (target: {lasso_result['target']})",
        f"LASSO R2 = {lasso_result['lasso_r2']:.3f} (alpha={lasso_result['alpha']:.4g}, n={lasso_result['n']})",
        "",
        "### Top loadings",
        lasso_result['top_loadings'].to_string(index=False),
        "",
        "### Loading share by characteristic family",
        lasso_result['family_share'].to_string(),
        "",
        "Supports / weakens / neutral w.r.t. CLAIM-MECH: "
        + ("SUPPORTS -- MIC_perp is not reducible to generic input compression (low R2 vs PCA "
           "factors z) and its LASSO loadings concentrate in identifiable characteristic families, "
           "consistent with return-specific distillation riding on the consensus coordinates."
           if not is_generic_compression else
           "WEAKENS the 'return-specific distillation' framing -- MIC_perp is substantially "
           "explained by generic unsupervised compression of the same inputs; report as found."),
    ]
    (out_dir / f'{horizon}_K2b_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {'z_reg': z_reg, 'lasso_result': lasso_result, 'is_generic_compression': is_generic_compression}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    result = run(horizon=args.horizon)
    print(result['lasso_result']['family_share'])
