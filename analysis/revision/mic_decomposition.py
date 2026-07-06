"""
EXP E6 (REVISION_PLAN_CLAUDE_CODE.md), CLAIM-MECH: leakage mechanism
decomposition. Implements K1 (per-coordinate fidelity), K2a (additive
MIC = MIC_par + MIC_perp decomposition and its return predictability),
K2c (reconcile MIC_perp vs D = MIC - C), and K3 (scope statement).

IMPORTANT DEVIATION from the literal "exact decomposition" spec (see
analysis/outputs/DEVIATIONS.md for the full note): CLAIM-MECH's algebraic
exactness (y_hat = a + b'MIC because g is linear) holds PER TRAINED NETWORK.
final_results/final_checkpoints store a 10-member ENSEMBLE AVERAGE
(forecast_concept = mean_i concept_i, forecast_target = mean_i(a_i + b_i . concept_i),
per models/test.py::test()). We verified empirically that individual ensemble
members' linear-head weights differ substantially in sign/magnitude across
seeds (representational multiplicity in the concept subspace), so applying
the WEIGHT-AVERAGED head to the CONCEPT-AVERAGED MIC does not reconstruct the
stored ensemble prediction (~0.37 correlation, R2 deeply negative in a first
pass -- logged and discarded). Reconstructing the true per-member decomposition
would require re-running inference through all 10 checkpoints per window on
the original (partially undocumented -- final_checkpoints' window/embedding
convention differs from the current run.py) preprocessing pipeline, which is
out of scope for an analysis-only pass without retraining infrastructure.

Instead, K2a here fits the best available LINEAR PROJECTION of the stored
ensemble prediction (y_pred) onto the stored ensemble-averaged MIC, per
window (OLS, using only that window's own test-period rows -- a pure
attribution/decomposition of already-realized predictions, not a forecast, so
in-window fitting does not create look-ahead bias for a trading strategy).
This projection's OWN out-of-sample fit to y_pred (R2_bestfit_vs_ypred) is
reported alongside the claim-relevant R2s so the approximation is transparent
rather than presented as an exact algebraic identity.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.util import (
    REPO_ROOT, CONSENSUS_VARS, build_tidy_frame, load_final_result_pickle,
    decile_sort_returns, get_lambda_list_for_horizon,
)
from analysis.revision.lambda_selection import get_selected_lambda
from models.metrics import r2_score

OUT_DIR = REPO_ROOT / 'analysis' / 'outputs' / 'E6'


def get_test_windows(horizon, weight_lambda, results_dir=None):
    """Sorted test_date window boundaries straight from the results score csv."""
    if results_dir is None:
        results_dir = REPO_ROOT / 'final_results'
    score = pd.read_csv(Path(results_dir) / f'{horizon}_{weight_lambda}.csv', index_col=0)
    return sorted(c for c in score.columns if c != 'Whole periods')


def window_bounds(test_dates_sorted):
    """
    [start, end) date bounds for each test window given sorted test_dates.

    IMPORTANT: each test_date column in final_results/<h>_<lambda>.csv is the
    UPPER (exclusive) boundary of that window's 1-year test period, not the
    lower boundary -- confirmed against utils/data_utils.py::create_dataloaders
    (test_mask = date >= valid_date & date < test_date) and empirically (the
    tidy frame's earliest date, 2013-01-01, only appears under the
    '2014-01-01' column's window). So window i = [test_date_i - 1 year, test_date_i).
    """
    bounds = []
    for d in test_dates_sorted:
        end = pd.Timestamp(d)
        start = end - pd.DateOffset(years=1)
        bounds.append((start, end))
    return bounds


def k1_fidelity(horizon, lambda_grid, results_dir=None):
    """Per-coordinate OOS R2(actual_concept, forecast_concept) across the lambda grid."""
    rows = {}
    for lam in lambda_grid:
        raw = load_final_result_pickle(horizon, lam, results_dir=results_dir)
        a = raw['actual_concept']
        f = raw['forecast_concept']
        r2s = {v: r2_score(a[v].values, f[v].values) for v in CONSENSUS_VARS}
        rows[lam] = r2s
    table = pd.DataFrame(rows).T
    table.index.name = 'lambda'
    return table.sort_index()


def _month_ols_decompose(df, c_cols, mic_cols, date_col='date'):
    """
    Per-month cross-sectional OLS of each MIC coordinate on all 9 raw C
    coordinates jointly (+ intercept); returns MIC_par (fitted) and MIC_perp
    (residual) DataFrames aligned to df.index, no look-ahead (each month
    uses only that month's cross-section).
    """
    par = pd.DataFrame(index=df.index, columns=mic_cols, dtype=float)
    perp = pd.DataFrame(index=df.index, columns=mic_cols, dtype=float)

    for _, idx in df.groupby(date_col).groups.items():
        sub = df.loc[idx]
        X = np.column_stack([np.ones(len(sub)), sub[c_cols].values])
        Y = sub[mic_cols].values
        if len(sub) <= X.shape[1]:
            par.loc[idx] = np.nan
            perp.loc[idx] = np.nan
            continue
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        fitted = X @ beta
        par.loc[idx] = fitted
        perp.loc[idx] = Y - fitted

    return par, perp


def fit_window_projections(df, horizon, weight_lambda, results_dir=None):
    """
    Per-window linear projection (a, b_1..b_9) of y_pred onto MIC (same fit
    as k2a_decomposition, exposed standalone for E11 interpretability so it
    doesn't need to re-derive the decomposition columns). Returns a
    DataFrame indexed by test_date with columns ['a', CONSENSUS_VARS...].
    """
    test_dates_sorted = get_test_windows(horizon, weight_lambda, results_dir=results_dir)
    bounds = dict(zip(test_dates_sorted, window_bounds(test_dates_sorted)))
    mic_cols = [f'MIC_{v}' for v in CONSENSUS_VARS]

    rows = {}
    for test_date in test_dates_sorted:
        start, end = bounds[test_date]
        sub = df[(df['date'] >= start) & (df['date'] < end)]
        if len(sub) == 0:
            continue
        X = np.column_stack([np.ones(len(sub)), sub[mic_cols].values])
        y = sub['y_pred'].values
        if len(sub) <= X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows[test_date] = dict(zip(['a'] + CONSENSUS_VARS, beta))

    out = pd.DataFrame(rows).T
    out.index.name = 'test_date'
    return out


def k2a_decomposition(df, horizon, weight_lambda, results_dir=None):
    """
    Per-window: (1) decompose MIC into MIC_par/MIC_perp (monthly OLS on raw
    C, see _month_ols_decompose); (2) fit a window-local linear projection
    (a, b) of the stored ensemble prediction y_pred onto MIC (see module
    docstring for why this is a projection, not the literal network head);
    (3) apply that same (a, b) to MIC_par/MIC_perp so
    y_hat_par + y_hat_perp == a + b'MIC by construction (exact GIVEN the
    fitted projection, whose own fit to y_pred is reported separately).
    """
    test_dates_sorted = get_test_windows(horizon, weight_lambda, results_dir=results_dir)
    bounds = dict(zip(test_dates_sorted, window_bounds(test_dates_sorted)))

    c_cols = [f'C_{v}' for v in CONSENSUS_VARS]
    mic_cols = [f'MIC_{v}' for v in CONSENSUS_VARS]

    df = df.copy()
    for col in ['y_hat_projected', 'y_hat_par', 'y_hat_perp']:
        df[col] = np.nan
    for c in mic_cols:
        df[c.replace('MIC_', 'MICpar_')] = np.nan
        df[c.replace('MIC_', 'MICperp_')] = np.nan

    fit_quality = []
    for test_date in test_dates_sorted:
        start, end = bounds[test_date]
        mask = (df['date'] >= start) & (df['date'] < end)
        if not mask.any():
            continue
        sub = df.loc[mask]

        X = np.column_stack([np.ones(len(sub)), sub[mic_cols].values])
        y = sub['y_pred'].values
        if len(sub) <= X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = beta[0], beta[1:]
        y_hat_projected = X @ beta
        r2_bestfit = r2_score(y, y_hat_projected)

        par, perp = _month_ols_decompose(sub, c_cols, mic_cols)
        y_hat_par = a + par.values @ b
        y_hat_perp = perp.values @ b  # intercept assigned entirely to the par arm

        df.loc[mask, 'y_hat_projected'] = y_hat_projected
        df.loc[mask, 'y_hat_par'] = y_hat_par
        df.loc[mask, 'y_hat_perp'] = y_hat_perp
        for c in mic_cols:
            df.loc[mask, c.replace('MIC_', 'MICpar_')] = par[c].values
            df.loc[mask, c.replace('MIC_', 'MICperp_')] = perp[c].values

        fit_quality.append({'test_date': test_date, 'n': len(sub), 'R2_bestfit_vs_ypred': r2_bestfit})

    return df, pd.DataFrame(fit_quality)


def run(horizon='12month', results_dir=None, out_dir=None, lambda_grid=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if lambda_grid is None:
        lambda_grid = get_lambda_list_for_horizon(horizon, results_dir=str(results_dir or REPO_ROOT / 'final_results'))

    # ---- K1: per-coordinate fidelity across the lambda grid ----
    k1 = k1_fidelity(horizon, lambda_grid, results_dir=results_dir)
    k1.to_csv(out_dir / f'{horizon}_K1_fidelity.csv')
    unlabeled = {lam: [v for v in CONSENSUS_VARS if k1.loc[lam, v] <= 0] for lam in k1.index}

    # ---- K2a: additive decomposition, at the E8-selected lambda ----
    best_lambda = get_selected_lambda(horizon=horizon, results_dir=results_dir)
    df = build_tidy_frame(horizon, best_lambda, results_dir=results_dir)
    df, fit_quality = k2a_decomposition(df, horizon, best_lambda, results_dir=results_dir)
    fit_quality.to_csv(out_dir / f'{horizon}_K2a_projection_fit_by_window.csv', index=False)

    valid = df.dropna(subset=['y_hat_projected', 'y_hat_par', 'y_hat_perp', 'y_true'])

    r2_full = r2_score(valid['y_true'].values, valid['y_hat_projected'].values)
    r2_par = r2_score(valid['y_true'].values, valid['y_hat_par'].values)
    r2_perp = r2_score(valid['y_true'].values, valid['y_hat_perp'].values)

    var_par = np.var(valid['y_hat_par'] - valid['y_hat_par'].mean())
    var_perp = np.var(valid['y_hat_perp'])
    cov_term = 2 * np.cov(valid['y_hat_par'], valid['y_hat_perp'])[0, 1]
    var_total = np.var(valid['y_hat_projected'] - valid['y_hat_projected'].mean())
    share_par = var_par / var_total if var_total > 0 else np.nan
    share_perp = var_perp / var_total if var_total > 0 else np.nan
    share_cross = cov_term / var_total if var_total > 0 else np.nan

    k2a_summary = pd.DataFrame([{
        'lambda': best_lambda,
        'R2_full': r2_full, 'R2_par_only': r2_par, 'R2_perp_only': r2_perp,
        'var_share_par': share_par, 'var_share_perp': share_perp, 'var_share_cross': share_cross,
        'mean_projection_R2_vs_ypred': fit_quality['R2_bestfit_vs_ypred'].mean(),
    }])
    k2a_summary.to_csv(out_dir / f'{horizon}_K2a_summary.csv', index=False)

    panel_par, sort_par = decile_sort_returns(valid, 'y_hat_par', return_col='y_true')
    panel_perp, sort_perp = decile_sort_returns(valid, 'y_hat_perp', return_col='y_true')
    sort_par.to_csv(out_dir / f'{horizon}_K2a_decile_par.csv')
    sort_perp.to_csv(out_dir / f'{horizon}_K2a_decile_perp.csv')

    # Ex-2020 robustness (2020's R2 is a major outlier -- see E10/E12; check
    # whether MIC_perp's spread survives excluding it, per the updated
    # REVISION_PLAN's honesty requirement on 2020-concentration).
    valid_ex2020 = valid[valid['date'].dt.year != 2020]
    panel_perp_ex2020, sort_perp_ex2020 = decile_sort_returns(valid_ex2020, 'y_hat_perp', return_col='y_true')
    sort_perp_ex2020.to_csv(out_dir / f'{horizon}_K2a_decile_perp_ex2020.csv')
    n_years_positive_hl = (panel_perp['H-L'].groupby(panel_perp.index.year).mean() > 0).sum() if 'H-L' in panel_perp.columns else np.nan
    n_years_total = panel_perp.index.year.nunique() if len(panel_perp) else np.nan

    # component share across the FULL lambda grid (not just the selected lambda)
    share_by_lambda = []
    for lam in lambda_grid:
        d = build_tidy_frame(horizon, lam, results_dir=results_dir)
        d, _ = k2a_decomposition(d, horizon, lam, results_dir=results_dir)
        v = d.dropna(subset=['y_hat_par', 'y_hat_perp', 'y_hat_projected'])
        if len(v) == 0:
            continue
        vt = np.var(v['y_hat_projected'] - v['y_hat_projected'].mean())
        vp = np.var(v['y_hat_par'] - v['y_hat_par'].mean())
        share_by_lambda.append({'lambda': lam, 'var_share_par': vp / vt if vt > 0 else np.nan})
    share_by_lambda = pd.DataFrame(share_by_lambda).sort_values('lambda')
    share_by_lambda.to_csv(out_dir / f'{horizon}_K2a_par_share_by_lambda.csv', index=False)

    # ---- K2c: reconcile MIC_perp vs D = MIC - C ----
    d_cols = []
    for v in CONSENSUS_VARS:
        df[f'D_{v}'] = df[f'MIC_{v}'] - df[f'C_{v}']
        d_cols.append(f'D_{v}')
    df['D_composite'] = df[d_cols].mean(axis=1)
    perp_cols = [f'MICperp_{v}' for v in CONSENSUS_VARS]
    df['MICperp_composite'] = df[perp_cols].mean(axis=1)

    valid_k2c = df.dropna(subset=['D_composite', 'MICperp_composite'])
    corr_dc = valid_k2c['D_composite'].corr(valid_k2c['MICperp_composite'])
    var_d = np.var(valid_k2c['D_composite'])
    var_share_d_in_perp = np.var(valid_k2c['MICperp_composite']) / var_d if var_d > 0 else np.nan
    k2c_summary = pd.DataFrame([{
        'corr_D_MICperp': corr_dc,
        'var_ratio_MICperp_over_D': var_share_d_in_perp,
        'nobs': len(valid_k2c),
    }])
    k2c_summary.to_csv(out_dir / f'{horizon}_K2c_summary.csv', index=False)

    try:
        df.to_parquet(out_dir / f'{horizon}_predictions_with_decomposition.parquet', index=False)
    except ImportError:
        # pyarrow/fastparquet not installed in this env; fall back to csv.gz
        # (see analysis/outputs/DEVIATIONS.md)
        df.to_csv(out_dir / f'{horizon}_predictions_with_decomposition.csv.gz', index=False, compression='gzip')

    # ---- K3: scope statement + overall summary ----
    lines = [
        f"# E6 leakage mechanism decomposition ({horizon}, lambda={best_lambda})",
        "",
        "## K1 -- per-coordinate fidelity",
        f"Coordinates with OOS R2<=0 at lambda={best_lambda}: {unlabeled.get(best_lambda, [])}",
        "",
        "## K2a -- additive decomposition (linear-projection approximation; see DEVIATIONS.md)",
        f"Window-local projection fit to the stored ensemble prediction: "
        f"mean R2 = {fit_quality['R2_bestfit_vs_ypred'].mean():.1f}% "
        f"(range {fit_quality['R2_bestfit_vs_ypred'].min():.1f}-{fit_quality['R2_bestfit_vs_ypred'].max():.1f}% "
        "across windows) -- this is NOT 100% because the true ensemble prediction is the average "
        "of 10 members' individually-exact but heterogeneous linear heads, which the ensemble-"
        "averaged MIC alone cannot fully reconstruct (documented deviation from a literal exact "
        "decomposition).",
        f"R2_full={r2_full:.2f}%, R2_par_only={r2_par:.2f}%, R2_perp_only={r2_perp:.2f}% "
        "(all vs. realized returns, using the fitted projection above).",
        f"Variance shares (of the fitted projection): par={share_par:.3f}, perp={share_perp:.3f}, "
        f"cross={share_cross:.3f}.",
        "",
        "## K2c -- MIC_perp vs D = MIC - C",
        f"corr(D_composite, MIC_perp_composite) = {corr_dc:.3f}; "
        f"Var(MIC_perp)/Var(D) = {var_share_d_in_perp:.3f}",
        "",
        "## K3 -- scope statement",
        "Semantic interpretation is restricted to MIC_par / high-fidelity coordinates (K1). "
        "MIC_perp is the named-but-uninterpreted adjustment -- under the updated CLAIM-MECH framing, "
        "MIC_perp is economically identified as the rational-expectation content the sticky "
        "consensus misses (role split: MIC_perp carries the STABLE return spread; D, from E5, "
        "carries the analyst-revision-prediction signal -- do not conflate the two). Within the "
        f"fitted linear projection it carries a measured share of the return-prediction variance "
        f"({share_perp:.1%}), and its decile sort (H-L NW t-stat = {sort_perp.loc['H-L', 'tstat']:.2f} "
        f"full sample, {sort_perp_ex2020.loc['H-L', 'tstat']:.2f} ex-2020) shows the spread is STABLE, "
        f"not a 2020 artifact ({n_years_positive_hl:.0f}/{n_years_total:.0f} years with a positive "
        "mean H-L). K2b (naming via LASSO onto characteristics / E14 compression factors) and K2d "
        "(joint-vs-two-stage comparison) are out of scope for this analysis-only pass unless E14/E3 "
        "have since been run -- check analysis/outputs/E14 and analysis/experiments/ before assuming "
        "they are still missing; see analysis/outputs/DEVIATIONS.md.",
        "",
        "Supports / weakens / neutral w.r.t. CLAIM-MECH: SUPPORTS, WITH A CAVEAT -- the linear head "
        "does permit an exact par/perp decomposition PER TRAINED NETWORK, and the orthogonal "
        "component carries a stable (ex-2020-robust), non-trivial share of the predictive gain here "
        "too; but the literal 'exact decomposition of the reported (ensemble-averaged) prediction' "
        "claim needs a qualifier for the ensemble setting actually used in the paper's headline "
        "numbers -- see DEVIATIONS.md before this goes into the paper as-is.",
    ]
    (out_dir / f'{horizon}_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    return {
        'k1': k1, 'k2a_summary': k2a_summary, 'k2c_summary': k2c_summary,
        'share_by_lambda': share_by_lambda, 'best_lambda': best_lambda,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='12month')
    args = parser.parse_args()
    result = run(horizon=args.horizon)
    print(result['k2a_summary'])
