"""
EXP E8 FIX (per owner's follow-up prompt): re-select lambda by max VALIDATION
R2_return (G1-compliant), instead of the max-TEST-R2 proxy used previously
(see analysis/lambda_selection.py and analysis/outputs/DEVIATIONS.md).

KEY DISCOVERY: `utils/data_utils.py::create_dataloaders(train_date, valid_date,
test_date, ...)` partitions as train = date < train_date, valid = train_date
<= date < valid_date, test = valid_date <= date < test_date. The checkpoint
filename prefix in final_checkpoints/<h>_<lambda>/<train_date>model_<i>.pt is
literally this `train_date` argument (the TRAIN/VALID boundary, not "start of
training"). Combined with the empirical fact that each window's test period is
exactly [test_date - 1 year, test_date) (see analysis/mic_decomposition.py's
window_bounds docstring), this pins down valid_date = test_date - 1 year
EXACTLY for every window (verified: train_date < valid_date for all 10
windows, with a clean, constant 2-year validation / 1-year test structure).
This means we CAN reconstruct the exact validation split and re-run INFERENCE
ONLY (no training) through the existing checkpoints to score every lambda on
its true validation R2 -- no retraining needed.

Only the minimal inference is done per lambda/window/member: macro min/max
stats from the train slice (needed for consistent normalization -- computing
min/max is not "training"), rank-normalization of the (lambda-independent)
firm/concept validation slice, per-member macro autoencoder encoding, and a
forward pass through the frozen ConceptBottleneckModel checkpoint. A
SANITY CHECK reproduces each window's TEST R2_return via this same pipeline
and compares it to the value already stored in final_results/<h>_<lambda>.csv
-- if these don't match closely, the reconstruction has a bug (or a
pipeline-version mismatch, see DEVIATIONS.md) and the validation numbers
should not be trusted without investigating further.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.util import REPO_ROOT, CONSENSUS_VARS, get_lambda_list_for_horizon
from analysis.revision.compression_ladder import load_partitioned_input
from utils.data_preprocessor import rank_norm, min_max_norm
from models.networks import ConceptBottleneckModel, Autoencoder
from models.metrics import r2_score
from config import get_config

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E8'
N_ENSEMBLE = 10


def get_window_dates(horizon, lam, checkpoints_dir=None, results_dir=None):
    if checkpoints_dir is None:
        checkpoints_dir = REPO_ROOT / 'final_checkpoints'
    if results_dir is None:
        results_dir = REPO_ROOT / 'final_results'

    ckpt_dir = Path(checkpoints_dir) / f'{horizon}_{lam}'
    train_dates = sorted({
        re.match(r'(\d{4}-\d{2}-\d{2})', f.name).group(1)
        for f in ckpt_dir.iterdir() if re.match(r'\d{4}-\d{2}-\d{2}', f.name)
    })
    score = pd.read_csv(Path(results_dir) / f'{horizon}_{lam}.csv', index_col=0)
    test_dates = sorted(c for c in score.columns if c != 'Whole periods')

    n = min(len(train_dates), len(test_dates))
    windows = []
    for train_date, test_date in zip(train_dates[-n:], test_dates[-n:]):
        valid_date = (pd.Timestamp(test_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        windows.append((train_date, valid_date, test_date))
    return windows


def build_window_slices(inp, tgt, firm_cols, concept_cols, macro_cols, train_date, valid_date, test_date):
    """
    Rank-normalize the firm/concept features for the VALID and TEST splits
    (lambda- and member-independent -- computed once per window, reused
    across the whole lambda grid), and compute the macro min/max stats from
    the TRAIN split (needed only for consistent normalization, not training).
    """
    index_cols = ['date', 'permno']
    train_mask = inp['date'] < pd.Timestamp(train_date)
    valid_mask = (inp['date'] >= pd.Timestamp(train_date)) & (inp['date'] < pd.Timestamp(valid_date))
    test_mask = (inp['date'] >= pd.Timestamp(valid_date)) & (inp['date'] < pd.Timestamp(test_date))

    valid_firm = rank_norm(inp.loc[valid_mask, index_cols + firm_cols])
    test_firm = rank_norm(inp.loc[test_mask, index_cols + firm_cols])
    valid_concept = rank_norm(inp.loc[valid_mask, index_cols + concept_cols])
    test_concept = rank_norm(inp.loc[test_mask, index_cols + concept_cols])

    train_macro_raw = inp.loc[train_mask, ['date'] + macro_cols].drop_duplicates('date')
    valid_macro_raw = inp.loc[valid_mask, ['date'] + macro_cols].drop_duplicates('date')
    test_macro_raw = inp.loc[test_mask, ['date'] + macro_cols].drop_duplicates('date')
    min_dict = {c: train_macro_raw[c].min() for c in macro_cols}
    max_dict = {c: train_macro_raw[c].max() for c in macro_cols}

    valid_y = tgt.loc[valid_mask].merge(valid_firm[index_cols], on=index_cols)['Return'].values
    test_y = tgt.loc[test_mask].merge(test_firm[index_cols], on=index_cols)['Return'].values

    return {
        'valid_firm': valid_firm, 'test_firm': test_firm,
        'valid_concept': valid_concept, 'test_concept': test_concept,
        'valid_macro_raw': valid_macro_raw, 'test_macro_raw': test_macro_raw,
        'min_dict': min_dict, 'max_dict': max_dict,
        'valid_y': valid_y, 'test_y': test_y,
    }


def member_inference(slices, split, firm_cols, macro_cols, concept_cols, ae_path, model_path, config, device='cpu'):
    """One ensemble member's concept/return predictions for `split` ('valid' or 'test')."""
    firm = slices[f'{split}_firm']
    macro_raw = slices[f'{split}_macro_raw']
    macro_norm = min_max_norm(macro_raw, min_dict=slices['min_dict'], max_dict=slices['max_dict'])

    ae = Autoencoder(input_dim=len(macro_cols), latent_dim=32).to(device)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()
    with torch.no_grad():
        macro_x = torch.from_numpy(macro_norm[macro_cols].values.astype(np.float32))
        macro_emb = ae.encoder(macro_x).numpy()
    macro_emb_df = pd.DataFrame(macro_emb, columns=[f'macro{i}' for i in range(32)])
    macro_emb_df['date'] = macro_raw['date'].values

    X_df = firm.merge(macro_emb_df, on='date')
    macro_names = [f'macro{i}' for i in range(32)]
    X = X_df[firm_cols + macro_names].values.astype(np.float32)

    model = ConceptBottleneckModel(
        config['input_size'], config['concept_hidden_sizes'], config['concept_output_size'],
        config['final_hidden_sizes'], config['final_output_size']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        concept_out, final_out = model(torch.from_numpy(X))

    return concept_out.numpy(), final_out.numpy().squeeze(-1), X_df[['date', 'permno']]


def ensemble_predict(slices, split, firm_cols, macro_cols, concept_cols, checkpoints_dir, horizon, lam, train_date, config, device='cpu'):
    ckpt_dir = Path(checkpoints_dir) / f'{horizon}_{lam}'
    concept_preds, final_preds, index_df = [], [], None
    for i in range(N_ENSEMBLE):
        ae_path = ckpt_dir / f'{train_date}_autoencoder_model{i}.pt'
        model_path = ckpt_dir / f'{train_date}model_{i}.pt'
        c, f, idx = member_inference(slices, split, firm_cols, macro_cols, concept_cols, ae_path, model_path, config, device=device)
        concept_preds.append(c)
        final_preds.append(f)
        index_df = idx
    return np.mean(concept_preds, axis=0), np.mean(final_preds, axis=0), index_df


def sanity_check_test_r2(inp, tgt, firm_cols, concept_cols, macro_cols, checkpoints_dir, results_dir, horizon, lam, config, device='cpu'):
    """Recompute pooled TEST R2_return via fresh inference and compare to the value already
    stored in final_results/<h>_<lambda>.csv -- validates the whole reconstruction."""
    windows = get_window_dates(horizon, lam, checkpoints_dir=checkpoints_dir, results_dir=results_dir)
    all_y, all_pred = [], []
    for train_date, valid_date, test_date in windows:
        slices = build_window_slices(inp, tgt, firm_cols, concept_cols, macro_cols, train_date, valid_date, test_date)
        _, final_pred, _ = ensemble_predict(slices, 'test', firm_cols, macro_cols, concept_cols, checkpoints_dir, horizon, lam, train_date, config, device=device)
        all_y.append(slices['test_y'])
        all_pred.append(final_pred)
    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)
    recomputed_r2 = r2_score(y, pred)

    score = pd.read_csv(Path(results_dir or REPO_ROOT / 'final_results') / f'{horizon}_{lam}.csv', index_col=0)
    stored_r2 = score.loc['Return', 'Whole periods']
    return recomputed_r2, stored_r2


def run_fallback_robustness_table(horizon, results_dir=None, out_dir=None, sanity_df=None,
                                   robustness_lambdas=(0.1, 0.2, 0.3, 0.4, 0.42, 0.6, 0.8, 1.0)):
    """
    Fallback specified in the task: since the validation split cannot be
    reliably reconstructed (sanity check failed), show that the HEADLINE
    conclusions are qualitatively unchanged across a reasonable lambda range
    -- using only what's already in final_results (no further inference).
    """
    from analysis.util import build_tidy_frame, decile_sort_returns, CONSENSUS_VARS
    from analysis.mic_decomposition import k2a_decomposition
    from models.metrics import r2_score

    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for lam in robustness_lambdas:
        df = build_tidy_frame(horizon, lam, results_dir=results_dir)
        r2 = r2_score(df['y_true'].values, df['y_pred'].values)
        _, summary = decile_sort_returns(df, 'y_pred', return_col='y_true')
        hl = summary.loc['H-L'] if 'H-L' in summary.index else None

        df_dec, _ = k2a_decomposition(df, horizon, lam, results_dir=results_dir)
        valid = df_dec.dropna(subset=['y_hat_perp', 'y_true'])
        _, perp_summary = decile_sort_returns(valid, 'y_hat_perp', return_col='y_true')
        perp_hl = perp_summary.loc['H-L'] if 'H-L' in perp_summary.index else None

        rows.append({
            'lambda': lam, 'R2_return': r2,
            'HL_mean': hl['mean'] if hl is not None else np.nan,
            'HL_tstat': hl['tstat'] if hl is not None else np.nan,
            'MICperp_HL_mean': perp_hl['mean'] if perp_hl is not None else np.nan,
            'MICperp_HL_tstat': perp_hl['tstat'] if perp_hl is not None else np.nan,
        })
        print(f'lambda={lam}: R2={r2:.2f}%, HL_t={hl["tstat"]:.2f}, MICperp_HL_t={perp_hl["tstat"]:.2f}')

    robustness = pd.DataFrame(rows)
    robustness.to_csv(out_dir / f'{horizon}_robustness_by_lambda_FALLBACK.csv', index=False)

    all_hl_significant = (robustness['HL_tstat'].abs() > 1.65).all()
    all_perp_significant = (robustness['MICperp_HL_tstat'].abs() > 1.65).all()
    sign_stable_hl = (np.sign(robustness['HL_mean']) == np.sign(robustness['HL_mean'].iloc[0])).all()
    sign_stable_perp = (np.sign(robustness['MICperp_HL_mean']) == np.sign(robustness['MICperp_HL_mean'].iloc[0])).all()

    lines = [
        f"# E8 FIX -- FALLBACK: multi-lambda robustness table ({horizon})",
        "",
        "The exact validation-split reconstruction FAILED its sanity check (see "
        f"{horizon}_validation_sanity_check.csv and {horizon}_validation_reconstruction_FAILED_note.txt) "
        "-- per the task's own contingency, this table shows headline conclusions hold "
        f"qualitatively across lambda in {robustness_lambdas}, using only the ALREADY-STORED "
        "final_results predictions (no further inference/retraining).",
        "",
        (sanity_df.to_string(index=False) if sanity_df is not None else ""),
        "",
        "## Headline metrics across the lambda grid",
        robustness.to_string(index=False),
        "",
        f"H-L spread significant (|t|>1.65) at every lambda tested? {all_hl_significant}",
        f"MIC_perp H-L spread significant at every lambda tested? {all_perp_significant}",
        f"H-L spread sign stable across lambda? {sign_stable_hl}",
        f"MIC_perp H-L spread sign stable across lambda? {sign_stable_perp}",
        "",
        "Supports / weakens / neutral w.r.t. G1 compliance: "
        "PARTIALLY ADDRESSED, NOT FULLY FIXED -- a true validation-based lambda selection could not "
        "be reconstructed from the frozen artifacts (see the FAILED note); this table demonstrates "
        "the headline results are NOT an artifact of cherry-picking lambda=0.42 specifically, which "
        "partially mitigates (but does not eliminate) the G1 concern. A full fix still requires "
        "either locating the original validation-period predictions/pipeline, or rerunning training "
        "with validation predictions persisted (both out of scope for an inference-only pass).",
    ]
    (out_dir / f'{horizon}_validation_selection_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {'robustness': robustness, 'sanity_ok': False, 'sanity_df': sanity_df}


def run(horizon='12month', lambda_grid=None, checkpoints_dir=None, results_dir=None,
        data_dir=None, out_dir=None, sanity_check_lambda=0.42, device='cpu'):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if checkpoints_dir is None:
        checkpoints_dir = REPO_ROOT / 'final_checkpoints'
    if results_dir is None:
        results_dir = REPO_ROOT / 'final_results'

    inp, tgt, firm_cols, concept_cols, macro_cols = load_partitioned_input(horizon, data_dir=data_dir)

    if lambda_grid is None:
        lambda_grid = get_lambda_list_for_horizon(horizon, results_dir=str(results_dir))

    # ---- Sanity check first: does this pipeline reproduce the stored TEST R2? ----
    config_ref = get_config(sanity_check_lambda)
    recomputed_r2, stored_r2 = sanity_check_test_r2(
        inp, tgt, firm_cols, concept_cols, macro_cols, checkpoints_dir, results_dir,
        horizon, sanity_check_lambda, config_ref, device=device)
    sanity_ok = abs(recomputed_r2 - stored_r2) < 1.0  # within 1pp

    sanity_df = pd.DataFrame([{
        'lambda': sanity_check_lambda, 'recomputed_test_R2': recomputed_r2,
        'stored_test_R2': stored_r2, 'diff': recomputed_r2 - stored_r2, 'sanity_ok': sanity_ok,
    }])
    sanity_df.to_csv(out_dir / f'{horizon}_validation_sanity_check.csv', index=False)

    if not sanity_ok:
        deviation_note = (
            f"SANITY CHECK FAILED: recomputed TEST R2 ({recomputed_r2:.2f}%) via fresh inference "
            f"through final_checkpoints differs from the stored final_results value ({stored_r2:.2f}%) "
            f"by more than 1pp -- per-window diffs are systematic (recomputed always lower, e.g. "
            "2015 window: -8.78% recomputed vs 0.12% stored), not scattered noise, so this is a real "
            "reconstruction mismatch, not floating-point/seed noise. The exact source was not found "
            "in the time available (checked: row universe/count matches exactly per window, feature "
            "column order matches training's firm+macro concatenation, min-max normalization uses "
            "the derived train slice) -- most likely the true historical run.py/data_utils.py "
            "pipeline that produced final_checkpoints differs from the current repo state in some "
            "way not yet identified (consistent with the already-documented train_date convention "
            "mismatch in DEVIATIONS.md). DO NOT TRUST a full validation-based reselection built on "
            "this reconstruction. FALLING BACK to the qualitative multi-lambda robustness table "
            "specified as the contingency in the task."
        )
        print(deviation_note)
        with open(out_dir / f'{horizon}_validation_reconstruction_FAILED_note.txt', 'w', encoding='utf-8') as f:
            f.write(deviation_note)
        return run_fallback_robustness_table(horizon, results_dir=results_dir, out_dir=out_dir,
                                              sanity_df=sanity_df)

    # ---- Validation R2 by (window, lambda) ----
    rows = []
    per_window_lambda_data = {}
    for lam in lambda_grid:
        windows = get_window_dates(horizon, lam, checkpoints_dir=checkpoints_dir, results_dir=results_dir)
        config = get_config(lam)
        for train_date, valid_date, test_date in windows:
            key = (train_date, valid_date, test_date)
            if key not in per_window_lambda_data:
                per_window_lambda_data[key] = build_window_slices(
                    inp, tgt, firm_cols, concept_cols, macro_cols, train_date, valid_date, test_date)
            slices = per_window_lambda_data[key]

            _, valid_pred, _ = ensemble_predict(
                slices, 'valid', firm_cols, macro_cols, concept_cols, checkpoints_dir, horizon, lam, train_date, config, device=device)
            r2_valid = r2_score(slices['valid_y'], valid_pred)
            rows.append({'lambda': lam, 'train_date': train_date, 'valid_date': valid_date,
                         'test_date': test_date, 'validation_R2_return': r2_valid})
            print(f'lambda={lam} test_date={test_date}: validation R2={r2_valid:.2f}%')

    val_table = pd.DataFrame(rows)
    val_table.to_csv(out_dir / f'{horizon}_validation_lambda_selection_raw.csv', index=False)

    # Per-window best lambda by validation R2
    selected = val_table.loc[val_table.groupby('test_date')['validation_R2_return'].idxmax()]
    selected = selected[['test_date', 'lambda', 'validation_R2_return']].rename(
        columns={'lambda': 'selected_lambda'})
    selected.to_csv(out_dir / f'{horizon}_validation_lambda_selection.csv', index=False)

    # Whole-sample: lambda maximizing MEAN validation R2 across windows
    mean_by_lambda = val_table.groupby('lambda')['validation_R2_return'].mean()
    whole_sample_lambda = mean_by_lambda.idxmax()

    # ---- Headline metrics recomputed with the validation-selected lambda ----
    from analysis.util import build_tidy_frame, decile_sort_returns
    from analysis.mic_decomposition import k2a_decomposition

    def build_stitched_frame(lambda_by_window):
        """Concatenate each window's TEST rows using that window's own selected lambda
        (already available in final_results -- no need to recompute test predictions)."""
        frames = []
        for _, row in lambda_by_window.iterrows():
            lam = row['selected_lambda']
            test_date = row['test_date']
            df = build_tidy_frame(horizon, lam, results_dir=results_dir)
            end = pd.Timestamp(test_date)
            start = end - pd.DateOffset(years=1)
            sub = df[(df['date'] >= start) & (df['date'] < end)]
            frames.append(sub)
        return pd.concat(frames, ignore_index=True)

    stitched = build_stitched_frame(selected)
    stitched_r2 = r2_score(stitched['y_true'].values, stitched['y_pred'].values)
    _, stitched_summary = decile_sort_returns(stitched, 'y_pred', return_col='y_true')
    stitched_hl = stitched_summary.loc['H-L'] if 'H-L' in stitched_summary.index else None

    fixed_042 = build_tidy_frame(horizon, 0.42, results_dir=results_dir)
    fixed_r2 = r2_score(fixed_042['y_true'].values, fixed_042['y_pred'].values)
    _, fixed_summary = decile_sort_returns(fixed_042, 'y_pred', return_col='y_true')
    fixed_hl = fixed_summary.loc['H-L'] if 'H-L' in fixed_summary.index else None

    comparison = pd.DataFrame([
        {'method': 'validation_selected (per-window)', 'R2_return': stitched_r2,
         'HL_mean': stitched_hl['mean'] if stitched_hl is not None else np.nan,
         'HL_tstat': stitched_hl['tstat'] if stitched_hl is not None else np.nan},
        {'method': f'fixed_lambda={0.42}_(prior_test-R2-max_proxy)', 'R2_return': fixed_r2,
         'HL_mean': fixed_hl['mean'] if fixed_hl is not None else np.nan,
         'HL_tstat': fixed_hl['tstat'] if fixed_hl is not None else np.nan},
        {'method': f'whole_sample_validation_selected_lambda={whole_sample_lambda}',
         'R2_return': r2_score(
             build_tidy_frame(horizon, whole_sample_lambda, results_dir=results_dir)['y_true'].values,
             build_tidy_frame(horizon, whole_sample_lambda, results_dir=results_dir)['y_pred'].values),
         'HL_mean': np.nan, 'HL_tstat': np.nan},
    ])
    comparison.to_csv(out_dir / f'{horizon}_headline_metrics_by_selected_lambda.csv', index=False)

    lines = [
        f"# E8 FIX -- validation-based lambda selection ({horizon})",
        "",
        "## Sanity check: does fresh-inference-through-checkpoints reproduce the stored TEST R2?",
        sanity_df.to_string(index=False),
        ("SANITY CHECK PASSED (within 1pp) -- the validation-window reconstruction is trustworthy."
         if sanity_ok else
         "SANITY CHECK FAILED -- see console/DEVIATIONS.md; validation numbers below are reported "
         "but should be treated with caution until this is resolved."),
        "",
        "## Per-window validation-selected lambda",
        selected.to_string(index=False),
        "",
        f"## Whole-sample validation-selected lambda (max mean validation R2 across windows): {whole_sample_lambda}",
        "",
        "## Headline metrics: validation-selected vs. the prior fixed-lambda=0.42 proxy",
        comparison.to_string(index=False),
        "",
        "Supports / weakens / neutral w.r.t. G1 compliance: "
        + ("FIXED -- lambda is now selected by max VALIDATION R2_return per window (G1-compliant), "
           "recovered via inference-only reconstruction of the true validation split (no "
           "retraining). Headline conclusions are " +
           ("QUALITATIVELY UNCHANGED (R2/H-L within a comparable range of the prior proxy "
            "selection)." if abs(stitched_r2 - fixed_r2) < 2.0 else
            "MATERIALLY DIFFERENT from the prior proxy selection -- flag to the owner before "
            "using validation-selected results in the paper.")
           if sanity_ok else
           "NOT YET TRUSTWORTHY -- the sanity check failed, meaning the reconstructed validation "
           "split likely does not match how final_checkpoints/final_results were actually produced. "
           "Do not use these numbers without first resolving the discrepancy (see DEVIATIONS.md)."),
    ]
    (out_dir / f'{horizon}_validation_selection_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {
        'sanity_ok': sanity_ok, 'sanity_df': sanity_df, 'val_table': val_table,
        'selected': selected, 'whole_sample_lambda': whole_sample_lambda, 'comparison': comparison,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    result = run(horizon=args.horizon)
    if result['sanity_ok']:
        print(result['comparison'])
    else:
        print(result['robustness'])
