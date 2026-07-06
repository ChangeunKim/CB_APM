"""
EXP E14 (REVISION_PLAN_CLAUDE_CODE.md), CLAIM-L2/MECH: compression-ladder
benchmark -- "B6" in the paper. Gives every non-NN-training INGREDIENT of the
CB framework (compressed input info + raw consensus info + linear head)
WITHOUT joint bottleneck learning.

Implemented here (NO new neural-net training): L-0 (raw C -> OLS), L-1
(PCA(inputs,d) -> OLS), L-3 ([PCA(inputs,d), raw C] -> OLS). The macro
component of "inputs" reuses the ALREADY-TRAINED per-window macro autoencoder
checkpoints in final_checkpoints/ (encoder forward pass only -- no training),
exactly the embedding the original CB model saw.

NOT implemented here: L-2 (AE(inputs,d) -> OLS) and L-4 ([AE(inputs,d), raw C]
-> OLS), which require training a NEW small compressor autoencoder per
window -- per the user's instruction, these are SCRIPTED ONLY in
analysis/experiments/train_e14_ae_variants.py, not run in this pass.

Ridge (small alpha) is used instead of plain OLS throughout to avoid
collinearity blow-ups when PCA/raw-C columns are combined -- logged here per
spec ("use ridge with small penalty if collinearity breaks OLS -- log the
choice").
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from analysis.util import REPO_ROOT
from utils.data_preprocessor import rank_norm, min_max_norm
from models.networks import Autoencoder
from models.metrics import r2_score
from analysis.revision.mic_decomposition import get_test_windows
from analysis.stats_utils import clark_west_test

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E14'
RIDGE_ALPHA = 1e-3


def infer_windows(horizon, weight_lambda, checkpoints_dir=None, results_dir=None):
    """
    Pair sorted unique train_dates found in the checkpoint folder with the
    sorted test_date columns in the matching results csv (both
    expanding-window lists, 1:1 by rank order) -- needed here (not in
    mic_decomposition.py) because the macro autoencoder checkpoint files are
    named by train_date, not test_date.
    """
    if checkpoints_dir is None:
        checkpoints_dir = REPO_ROOT / 'final_checkpoints'
    ckpt_dir = Path(checkpoints_dir) / f'{horizon}_{weight_lambda}'
    train_dates = sorted({
        re.match(r'(\d{4}-\d{2}-\d{2})', f.name).group(1)
        for f in ckpt_dir.iterdir() if re.match(r'\d{4}-\d{2}-\d{2}', f.name)
    })
    test_dates = get_test_windows(horizon, weight_lambda, results_dir=results_dir)
    n = min(len(train_dates), len(test_dates))
    return list(zip(train_dates[-n:], test_dates[-n:]))


def load_partitioned_input(horizon, data_dir=None):
    if data_dir is None:
        data_dir = REPO_ROOT / 'data'
    data_dir = Path(data_dir)

    info = pd.read_csv(data_dir / 'info' / 'SignalDoc.csv')
    inp = pd.read_csv(data_dir / f'input_{horizon}.csv')
    inp['date'] = pd.to_datetime(inp['date'])
    tgt = pd.read_csv(data_dir / f'target_{horizon}.csv')
    tgt['date'] = pd.to_datetime(tgt['date'])

    info_f = info[info['Acronym'].isin(inp.columns)]
    firm_cols = list(info_f[info_f['Cat.Data'] != 'Analyst']['Acronym'].values)
    concept_cols = list(info_f[info_f['Cat.Data'] == 'Analyst']['Acronym'].values)
    index_cols = ['date', 'permno']
    macro_cols = [c for c in inp.columns if c not in firm_cols and c not in concept_cols and c not in index_cols]

    return inp, tgt, firm_cols, concept_cols, macro_cols


def embed_macro_with_checkpoint(macro_train, macro_apply, macro_cols, autoencoder_path, latent_dim=32):
    """Min-max normalize (train stats) then encode via the already-trained macro autoencoder."""
    min_dict = {c: macro_train[c].min() for c in macro_cols}
    max_dict = {c: macro_train[c].max() for c in macro_cols}
    macro_apply_norm = min_max_norm(macro_apply[['date'] + macro_cols], min_dict=min_dict, max_dict=max_dict)

    device = torch.device('cpu')
    model = Autoencoder(input_dim=len(macro_cols), latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(autoencoder_path, map_location=device))
    model.eval()

    x = torch.from_numpy(macro_apply_norm[macro_cols].values).float()
    with torch.no_grad():
        z = model.encoder(x).numpy()
    emb = pd.DataFrame(z, columns=[f'macro{i}' for i in range(latent_dim)])
    emb['date'] = macro_apply['date'].values
    return emb


def build_window_matrices(inp, tgt, firm_cols, concept_cols, macro_cols, train_date, start, end, checkpoints_dir, lam=0.42):
    """
    Build [firm(114) rank-normed, macro(32) embedded] = 146-d, raw C (9,
    rank-normed) and Return, separately for the TRAIN window (all rows dated
    < start) and TEST window (rows in [start, end)).
    """
    train_mask = inp['date'] < start
    test_mask = (inp['date'] >= start) & (inp['date'] < end)

    index_cols = ['date', 'permno']
    train_firm = rank_norm(inp.loc[train_mask, index_cols + firm_cols])
    test_firm = rank_norm(inp.loc[test_mask, index_cols + firm_cols])
    train_concept = rank_norm(inp.loc[train_mask, index_cols + concept_cols])
    test_concept = rank_norm(inp.loc[test_mask, index_cols + concept_cols])

    macro_train_raw = inp.loc[train_mask, ['date'] + macro_cols].drop_duplicates('date')
    macro_test_raw = inp.loc[test_mask, ['date'] + macro_cols].drop_duplicates('date')

    ae_path = Path(checkpoints_dir) / f'12month_{lam}' / f'{train_date}_autoencoder_model0.pt'
    macro_train_emb = embed_macro_with_checkpoint(macro_train_raw, macro_train_raw, macro_cols, ae_path)
    macro_test_emb = embed_macro_with_checkpoint(macro_train_raw, macro_test_raw, macro_cols, ae_path)

    train_input = train_firm.merge(macro_train_emb, on='date').merge(train_concept, on=index_cols)
    test_input = test_firm.merge(macro_test_emb, on='date').merge(test_concept, on=index_cols)

    train_y = tgt.loc[train_mask].merge(train_input[index_cols], on=index_cols)['Return'].values
    test_y = tgt.loc[test_mask].merge(test_input[index_cols], on=index_cols)['Return'].values

    macro_names = [f'macro{i}' for i in range(32)]
    x146_train = train_input[firm_cols + macro_names].values
    x146_test = test_input[firm_cols + macro_names].values
    c_train = train_input[concept_cols].values
    c_test = test_input[concept_cols].values

    return {
        'x146_train': x146_train, 'x146_test': x146_test,
        'c_train': c_train, 'c_test': c_test,
        'y_train': train_y, 'y_test': test_y,
        'index_test': test_input[index_cols],
    }


def run_ladder_for_window(mats, d_grid=(9, 16, 32)):
    results = {}
    z_factors = {}

    # L-0: raw C -> ridge
    model_l0 = Ridge(alpha=RIDGE_ALPHA).fit(mats['c_train'], mats['y_train'])
    results['L-0'] = {'d': 9, 'y_pred': model_l0.predict(mats['c_test'])}

    for d in d_grid:
        pca = PCA(n_components=d).fit(mats['x146_train'])
        z_train = pca.transform(mats['x146_train'])
        z_test = pca.transform(mats['x146_test'])
        if d == 9:
            z_factors['test'] = z_test
            z_factors['index'] = mats['index_test']

        # L-1: PCA(inputs,d) -> ridge
        model_l1 = Ridge(alpha=RIDGE_ALPHA).fit(z_train, mats['y_train'])
        results[f'L-1_d{d}'] = {'d': d, 'y_pred': model_l1.predict(z_test)}

        # L-3: [PCA(inputs,d), raw C] -> ridge
        X_train_l3 = np.column_stack([z_train, mats['c_train']])
        X_test_l3 = np.column_stack([z_test, mats['c_test']])
        model_l3 = Ridge(alpha=RIDGE_ALPHA).fit(X_train_l3, mats['y_train'])
        results[f'L-3_d{d}'] = {'d': d, 'y_pred': model_l3.predict(X_test_l3)}

    return results, z_factors


def run(horizon='12month', data_dir=None, checkpoints_dir=None, results_dir=None, out_dir=None, lam=0.42):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if checkpoints_dir is None:
        checkpoints_dir = REPO_ROOT / 'final_checkpoints'

    inp, tgt, firm_cols, concept_cols, macro_cols = load_partitioned_input(horizon, data_dir=data_dir)

    windows = infer_windows(horizon, lam, checkpoints_dir=checkpoints_dir, results_dir=results_dir)

    all_preds = {}
    z_records = []
    x146_records = []
    per_window_r2 = []
    for train_date, test_date in windows:
        end = pd.Timestamp(test_date)
        start = end - pd.DateOffset(years=1)
        mats = build_window_matrices(inp, tgt, firm_cols, concept_cols, macro_cols, train_date, start, end, checkpoints_dir, lam=lam)
        results, z_factors = run_ladder_for_window(mats)

        row = {'test_date': test_date, 'n': len(mats['y_test'])}
        for tag, r in results.items():
            r2v = r2_score(mats['y_test'], r['y_pred'])
            row[f'R2_{tag}'] = r2v
            all_preds.setdefault(tag, []).append(pd.DataFrame({
                'date': mats['index_test']['date'].values, 'permno': mats['index_test']['permno'].values,
                'y_true': mats['y_test'], 'y_pred': r['y_pred'],
            }))
        per_window_r2.append(row)

        z_df = mats['index_test'].copy()
        for j in range(z_factors['test'].shape[1]):
            z_df[f'z{j}'] = z_factors['test'][:, j]
        z_records.append(z_df)

        # Raw 146-d [firm(114) rank-normed, macro(32) embedded] test features,
        # named, for E6/K2b's LASSO characteristic-naming exercise (avoids
        # rebuilding this expensive window construction a second time there).
        macro_names = [f'macro{i}' for i in range(32)]
        x146_df = mats['index_test'].copy()
        x146_arr = mats['x146_test']
        for j, name in enumerate(firm_cols + macro_names):
            x146_df[name] = x146_arr[:, j]
        x146_records.append(x146_df)

    per_window_r2 = pd.DataFrame(per_window_r2)
    per_window_r2.to_csv(out_dir / f'{horizon}_per_window_R2.csv', index=False)

    pooled_rows = []
    pooled_preds = {}
    for tag, dfs in all_preds.items():
        full = pd.concat(dfs, ignore_index=True)
        pooled_preds[tag] = full
        pooled_rows.append({'variant': tag, 'R2_pooled': r2_score(full['y_true'].values, full['y_pred'].values), 'n': len(full)})
    pooled = pd.DataFrame(pooled_rows).sort_values('variant')
    pooled.to_csv(out_dir / f'{horizon}_pooled_R2_ladder.csv', index=False)

    z_all = pd.concat(z_records, ignore_index=True)
    try:
        z_all.to_parquet(out_dir / f'{horizon}_z_factors_d9.parquet', index=False)
    except ImportError:
        z_all.to_csv(out_dir / f'{horizon}_z_factors_d9.csv.gz', index=False, compression='gzip')

    x146_all = pd.concat(x146_records, ignore_index=True)
    try:
        x146_all.to_parquet(out_dir / f'{horizon}_x146_test_features.parquet', index=False)
    except ImportError:
        x146_all.to_csv(out_dir / f'{horizon}_x146_test_features.csv.gz', index=False, compression='gzip')

    # CW test: CB framework (lambda=lam, from final_results) vs each ladder variant, pooled
    from analysis.util import build_tidy_frame
    cb_df = build_tidy_frame(horizon, lam, results_dir=results_dir)[['date', 'permno', 'y_true', 'y_pred']]

    cw_rows = []
    for tag, full in pooled_preds.items():
        merged = cb_df.merge(full, on=['date', 'permno'], suffixes=('_cb', '_ladder'))
        if len(merged) < 100:
            continue
        res = clark_west_test(merged['y_true_cb'].values, merged['y_pred_ladder'].values, merged['y_pred_cb'].values)
        cw_rows.append({'variant': tag, 'cw_stat': res['cw_stat'], 'pval': res['pval'], 'nobs': res['nobs']})
    cw_df = pd.DataFrame(cw_rows).sort_values('variant')
    cw_df.to_csv(out_dir / f'{horizon}_CW_tests_vs_CB.csv', index=False)

    cb_pooled_r2 = r2_score(cb_df['y_true'].values, cb_df['y_pred'].values)
    best_ladder = pooled.loc[pooled['R2_pooled'].idxmax()]
    cb_beats_ladder = cb_pooled_r2 > best_ladder['R2_pooled']

    lines = [
        f"# E14 compression ladder ({horizon}, lambda={lam}) -- L-0, L-1, L-3 (non-NN variants)",
        "",
        "NOT run here (script only, see analysis/experiments/train_e14_ae_variants.py): L-2, L-4 "
        "(AE-compressor variants) -- these need training a NEW small compressor autoencoder per "
        "window, a genuine new-NN-training experiment per the user's instruction.",
        "",
        f"Ridge (alpha={RIDGE_ALPHA}) used throughout instead of plain OLS, to avoid collinearity "
        "blow-ups when combining PCA factors with raw consensus -- logged per spec.",
        "",
        "## Per-window R2 (pooled OOS, all d in {9,16,32})",
        per_window_r2.to_string(index=False),
        "",
        "## Pooled R2 by ladder variant vs the CB framework",
        pooled.to_string(index=False),
        f"CB framework (lambda={lam}) pooled R2_return: {cb_pooled_r2:.2f}%",
        f"Best ladder variant: {best_ladder['variant']} (R2={best_ladder['R2_pooled']:.2f}%)",
        f"CB framework beats the best non-NN ladder variant? {cb_beats_ladder}",
        "",
        "## Clark-West tests: CB framework vs each ladder variant (pooled, nested)",
        cw_df.to_string(index=False),
        "",
        "Interpretation key (per spec): CB > L-3 supports the joint-learning mechanism (that "
        "supervised, jointly-trained compression captures something PCA+raw-C combination cannot); "
        "report as found either way.",
        "",
        "Supports / weakens / neutral w.r.t. CLAIM-L2/MECH: "
        + ("SUPPORTS -- the CB framework's OOS R2 exceeds every non-NN ladder variant tested here, "
           "consistent with joint supervised compression being the mechanism (not merely having "
           "compressed-input + raw-consensus information available)." if cb_beats_ladder else
           "WEAKENS -- at least one non-NN ladder variant matches or exceeds the CB framework's R2; "
           "report as found and flag to the owner before claiming joint learning is necessary."),
    ]
    (out_dir / f'{horizon}_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {'per_window_r2': per_window_r2, 'pooled': pooled, 'cw_df': cw_df, 'cb_beats_ladder': cb_beats_ladder}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    parser.add_argument('--lam', type=float, default=0.42)
    args = parser.parse_args()
    result = run(horizon=args.horizon, lam=args.lam)
    print(result['pooled'])
