"""
Shared statistical inference module (REVISION_PLAN_CLAUDE_CODE.md, EXP E9).

Pure numpy/pandas/scipy functions, no training-time dependencies, so every
other analysis/*.py module can import from here without pulling in torch.
"""
import numpy as np
import pandas as pd
from scipy import stats


def newey_west_tstat(series, lag=12):
    """
    Newey-West (HAC) t-statistic for the mean of a series (e.g. a monthly
    H-L spread or an information-ratio numerator).

    Parameters
    ----------
    series : array-like of monthly observations
    lag : int, HAC bandwidth (G6 default: 12, for overlapping annual returns)

    Returns
    -------
    dict with 'mean', 'se', 'tstat', 'pval', 'nobs'
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {'mean': np.nan, 'se': np.nan, 'tstat': np.nan, 'pval': np.nan, 'nobs': n}

    mean = x.mean()
    u = x - mean

    # Bartlett-kernel HAC variance of the mean
    gamma0 = np.mean(u ** 2)
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        gamma_k = np.mean(u[k:] * u[:-k])
        var += 2 * w * gamma_k
    se = np.sqrt(max(var, 0.0) / n)

    tstat = mean / se if se > 0 else np.nan
    pval = 2 * (1 - stats.norm.cdf(abs(tstat))) if np.isfinite(tstat) else np.nan
    return {'mean': mean, 'se': se, 'tstat': tstat, 'pval': pval, 'nobs': n}


def _hac_var_of_mean(u, lag):
    n = len(u)
    gamma0 = np.mean(u ** 2)
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        gamma_k = np.mean(u[k:] * u[:-k])
        var += 2 * w * gamma_k
    return max(var, 0.0) / n


def clark_west_test(y_true, pred_restricted, pred_unrestricted, lag=12):
    """
    Clark-West (2007) test for comparing nested OOS R2's: is the larger
    (unrestricted, e.g. CB-framework) model's forecast significantly better
    than the smaller (restricted, e.g. lambda=0 / raw-consensus) model's,
    after adjusting for the noise added by extra parameters?

    Monthly-pooled loss differentials with a Newey-West (lag=12) t-test on
    the adjusted MSE differential, per REVISION_PLAN_CLAUDE_CODE.md G6/E9.

    Returns
    -------
    dict with 'cw_stat', 'pval' (one-sided, unrestricted better), 'nobs'
    """
    y = np.asarray(y_true, dtype=float)
    pr = np.asarray(pred_restricted, dtype=float)
    pu = np.asarray(pred_unrestricted, dtype=float)

    e_r = y - pr
    e_u = y - pu
    adj = (pr - pu) ** 2
    f = e_r ** 2 - e_u ** 2 + adj

    f = f[~np.isnan(f)]
    n = len(f)
    if n < 2:
        return {'cw_stat': np.nan, 'pval': np.nan, 'nobs': n}

    mean_f = f.mean()
    var_f = _hac_var_of_mean(f - mean_f, lag)
    se = np.sqrt(var_f)
    cw_stat = mean_f / se if se > 0 else np.nan
    pval = 1 - stats.norm.cdf(cw_stat) if np.isfinite(cw_stat) else np.nan
    return {'cw_stat': cw_stat, 'pval': pval, 'nobs': n}


def diebold_mariano_test(y_true, pred_a, pred_b, h=1, lag=None):
    """
    Diebold-Mariano test for two non-nested forecasts (e.g. CB-framework vs.
    full-info NN). Loss differential d_t = e_a_t^2 - e_b_t^2; two-sided test
    of H0: E[d_t] = 0. Negative DM stat -> model B (pred_b) has lower loss.

    lag defaults to h-1 (standard DM bandwidth) unless overridden (E9 uses
    lag=12 for consistency with the rest of the module when h is unknown).
    """
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)

    d = (y - a) ** 2 - (y - b) ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return {'dm_stat': np.nan, 'pval': np.nan, 'nobs': n}

    bw = lag if lag is not None else max(h - 1, 0)
    mean_d = d.mean()
    var_d = _hac_var_of_mean(d - mean_d, bw) if bw > 0 else np.var(d, ddof=1) / n
    se = np.sqrt(var_d)
    dm_stat = mean_d / se if se > 0 else np.nan
    pval = 2 * (1 - stats.norm.cdf(abs(dm_stat))) if np.isfinite(dm_stat) else np.nan
    return {'dm_stat': dm_stat, 'pval': pval, 'nobs': n}


def ledoit_wolf_sharpe_test(returns_a, returns_b):
    """
    Ledoit-Wolf (2008) test for the equality of two Sharpe ratios computed
    from paired (same-dates) return series, e.g. MIC-based vs raw-consensus
    decile-sort H-L returns. Uses the HAC-robust asymptotic variance of the
    Sharpe-ratio difference (Bartlett kernel, same lag convention as the rest
    of this module -> lag=12 for monthly overlapping-horizon returns).

    Returns
    -------
    dict with 'sharpe_a', 'sharpe_b', 'diff', 'tstat', 'pval'
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 2:
        return {'sharpe_a': np.nan, 'sharpe_b': np.nan, 'diff': np.nan, 'tstat': np.nan, 'pval': np.nan}

    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(ddof=1), b.std(ddof=1)
    sr_a = mu_a / sig_a if sig_a > 0 else np.nan
    sr_b = mu_b / sig_b if sig_b > 0 else np.nan

    # Delta-method gradient of SR_a - SR_b w.r.t. (mu_a, mu_b, sig2_a, sig2_b)
    var_a, var_b = sig_a ** 2, sig_b ** 2
    grad = np.array([
        1.0 / sig_a if sig_a > 0 else 0.0,
        -1.0 / sig_b if sig_b > 0 else 0.0,
        -mu_a / (2 * sig_a ** 3) if sig_a > 0 else 0.0,
        mu_b / (2 * sig_b ** 3) if sig_b > 0 else 0.0,
    ])

    moments = np.column_stack([
        a - mu_a,
        b - mu_b,
        (a - mu_a) ** 2 - var_a,
        (b - mu_b) ** 2 - var_b,
    ])

    lag = 12
    k = moments.shape[1]
    omega = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            u = moments[:, i]
            v = moments[:, j]
            gamma0 = np.mean(u * v)
            s = gamma0
            for lg in range(1, min(lag, n - 1) + 1):
                w = 1 - lg / (lag + 1)
                s += w * (np.mean(u[lg:] * v[:-lg]) + np.mean(u[:-lg] * v[lg:]))
            omega[i, j] = s
    var_diff = grad @ omega @ grad / n
    se = np.sqrt(max(var_diff, 0.0))

    diff = sr_a - sr_b
    tstat = diff / se if se > 0 else np.nan
    pval = 2 * (1 - stats.norm.cdf(abs(tstat))) if np.isfinite(tstat) else np.nan
    return {'sharpe_a': sr_a, 'sharpe_b': sr_b, 'diff': diff, 'tstat': tstat, 'pval': pval}


def _cluster_robust_cov(X, resid, cluster_ids):
    """
    One-way cluster-robust (sandwich) covariance of OLS coefficients,
    vectorized via a group-sum of per-observation scores (avoids an O(n*G)
    python loop, which is prohibitive for high-cardinality clusters like a
    firm x month interaction with tens of thousands of groups).
    """
    xtx_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]  # (n, k) per-observation score contributions

    _, inverse = np.unique(cluster_ids, return_inverse=True)
    n_groups = inverse.max() + 1
    group_scores = np.zeros((n_groups, X.shape[1]))
    np.add.at(group_scores, inverse, scores)  # (G, k) sum of scores per cluster

    meat = group_scores.T @ group_scores
    return xtx_inv @ meat @ xtx_inv


def two_way_cluster_ols(X, y, cluster1, cluster2):
    """
    OLS with two-way cluster-robust standard errors (Cameron, Gelbach & Miller
    2011): V = V_cluster1 + V_cluster2 - V_cluster1xcluster2. Used by E5's
    panel regressions (SE clustered by firm and month, per
    REVISION_PLAN_CLAUDE_CODE.md M1/M2).

    Parameters
    ----------
    X : (n, k) array, should already include an intercept column if desired.
    y : (n,) array
    cluster1, cluster2 : (n,) arrays of cluster ids (e.g. firm id, month id)

    Returns
    -------
    dict with 'coef', 'se', 'tstat', 'pval' (each length-k arrays) and 'nobs'
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y, cluster1, cluster2 = X[mask], y[mask], cluster1[mask], cluster2[mask]
    n, k = X.shape

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    v1 = _cluster_robust_cov(X, resid, cluster1)
    v2 = _cluster_robust_cov(X, resid, cluster2)
    # Vectorized intersection-cluster id (avoids a per-row python string-format loop)
    _, codes1 = np.unique(cluster1, return_inverse=True)
    _, codes2 = np.unique(cluster2, return_inverse=True)
    cluster12 = codes1.astype(np.int64) * (codes2.max() + 1) + codes2.astype(np.int64)
    v12 = _cluster_robust_cov(X, resid, cluster12)
    vcov = v1 + v2 - v12

    se = np.sqrt(np.clip(np.diag(vcov), 0, None))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pval = 2 * (1 - stats.norm.cdf(np.abs(tstat)))
    return {'coef': beta, 'se': se, 'tstat': tstat, 'pval': pval, 'nobs': n}


def newey_west_ols(X, y, lag=12):
    """
    Time-series OLS with Newey-West HAC standard errors (Bartlett kernel).
    Used for the FF6 time-series alpha regression of an H-L portfolio (E10).
    X must already include an intercept column if an alpha is wanted; assumes
    rows are time-ordered (no cluster ids needed -- single HAC cluster).

    Returns
    -------
    dict with 'coef', 'se', 'tstat', 'pval' (length-k arrays) and 'nobs'
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]
    n, k = X.shape

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    xtx_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]  # (n, k)

    s = scores.T @ scores
    for lg in range(1, min(lag, n - 1) + 1):
        w = 1 - lg / (lag + 1)
        gamma = scores[lg:].T @ scores[:-lg]
        s += w * (gamma + gamma.T)

    vcov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.clip(np.diag(vcov), 0, None))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pval = 2 * (1 - stats.norm.cdf(np.abs(tstat)))
    return {'coef': beta, 'se': se, 'tstat': tstat, 'pval': pval, 'nobs': n}


def patton_timmermann_bootstrap_test(decile_returns, n_boot=2000, block_size=12, seed=0):
    """
    Patton & Timmermann (2010) "MR" monotonicity test via MOVING-BLOCK
    BOOTSTRAP (closer to their actual procedure than the parametric
    Newey-West-t version in patton_timmermann_test, which is a simplified
    stand-in -- see its docstring). Resamples overlapping blocks of
    consecutive months (preserving serial/autocorrelation structure) to build
    the null distribution of the min-adjacent-difference statistic, then
    reports a bootstrap p-value for H1: returns increase monotonically from
    decile 1 to decile K.

    Parameters
    ----------
    decile_returns : array-like, shape (T, K), columns ordered low to high.
    n_boot : number of bootstrap resamples.
    block_size : block length (months) for the moving-block bootstrap.
    seed : int, RNG seed (deterministic; np.random.default_rng not used
        elsewhere in this module so no cross-call contamination).

    Returns
    -------
    dict with 'min_diff_stat' (observed), 'boot_pval' (P(bootstrap min-diff
    <= 0), one-sided), 'n_deciles', 'nobs'
    """
    arr = np.asarray(decile_returns, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return {'min_diff_stat': np.nan, 'boot_pval': np.nan, 'n_deciles': np.nan, 'nobs': np.nan}

    T, k = arr.shape
    diffs_obs = np.diff(arr, axis=1).mean(axis=0)  # mean of each adjacent decile-pair difference
    observed_min_diff = diffs_obs.min()

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_size))
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        start_idx = rng.integers(0, max(T - block_size, 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_size, T)) for s in start_idx])[:T]
        resampled = arr[idx]
        diffs = np.diff(resampled, axis=1).mean(axis=0)
        boot_stats[b] = diffs.min()

    # Center the bootstrap distribution at zero (test statistic under H0: no
    # monotonic increase) by removing the observed mean, then see how often a
    # centered bootstrap draw exceeds the observed statistic.
    centered = boot_stats - boot_stats.mean()
    boot_pval = float(np.mean(centered >= observed_min_diff))

    return {'min_diff_stat': observed_min_diff, 'boot_pval': boot_pval, 'n_deciles': k, 'nobs': T}


def patton_timmermann_test(decile_returns, lag=12):
    """
    SIMPLIFIED variant (parametric, NW-t based) of the Patton-Timmermann
    (2010) "MR" monotonicity test -- see patton_timmermann_bootstrap_test
    above for the moving-block-bootstrap version closer to their actual
    procedure. Kept for backward compatibility / speed; prefer the bootstrap
    version for anything going into the paper.

    Over K decile-portfolio monthly return series. H0: no monotonic
    (increasing) pattern across deciles; H1: returns increase monotonically
    from decile 1 to decile K.

    Test statistic = min_k [ NW-tstat( decile_{k+1,t} - decile_{k,t} ) ],
    i.e. the weakest (smallest) HAC t-stat among all adjacent successive
    differences, one-sided (reject H0 / support monotonicity iff min_t > 0
    and the associated p-value is small).

    Parameters
    ----------
    decile_returns : array-like, shape (T, K)
        Monthly return series for K decile portfolios (columns ordered low
        to high), or a pandas DataFrame with one column per decile.
    lag : int, HAC bandwidth (matches newey_west_tstat default)

    Returns
    -------
    dict with 'min_diff_tstat', 'pval', 'n_deciles'
    """
    arr = np.asarray(decile_returns, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return {'min_diff_tstat': np.nan, 'pval': np.nan, 'n_deciles': arr.shape[-1] if arr.ndim == 2 else np.nan}

    k = arr.shape[1]
    tstats = []
    for j in range(k - 1):
        diff_series = arr[:, j + 1] - arr[:, j]
        res = newey_west_tstat(diff_series, lag=lag)
        tstats.append(res['tstat'])

    tstats = np.array(tstats, dtype=float)
    if np.all(np.isnan(tstats)):
        return {'min_diff_tstat': np.nan, 'pval': np.nan, 'n_deciles': k}

    min_t = np.nanmin(tstats)
    pval = 1 - stats.norm.cdf(min_t) if np.isfinite(min_t) else np.nan
    return {'min_diff_tstat': min_t, 'pval': pval, 'n_deciles': k}
