"""
transfer_entropy.py — Stage 4 (correlation pre-filter): lagged Transfer Entropy.

Contains the two TE estimators (histogram and KNN/KSG), the best-lag
search weighted by the lag penalty, the permutation test, and the
function executable in parallel used by discovery.filter_causal_candidates().
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional

from .penalties import LagPenaltyFn, DEFAULT_LAG_PENALTY


def _te_histogram(x: np.ndarray, y: np.ndarray, lag: int, bins: int = 8) -> float:
    """TE(X->Y, lag) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})."""
    T = len(y)
    if T - lag < 20:
        return 0.0

    yt  = y[lag:]
    yt1 = y[lag - 1:-1]
    xl  = x[:T - lag]

    valid = ~(np.isnan(yt) | np.isnan(yt1) | np.isnan(xl))
    yt, yt1, xl = yt[valid], yt1[valid], xl[valid]
    n = len(yt)
    if n < 15:
        return 0.0

    def qbin(v, b):
        lo, hi = np.nanmin(v), np.nanmax(v)
        if hi == lo:
            return np.zeros(len(v), dtype=int)
        edges = np.linspace(lo, hi, b + 1)
        return np.clip(np.digitize(v, edges) - 1, 0, b - 1)

    qyt  = qbin(yt,  bins)
    qyt1 = qbin(yt1, bins)
    qxl  = qbin(xl,  bins)

    def entropy_joint(*arrs):
        # Each input array is a digit in [0, bins) (output of qbin
        # above) -- encoding the tuple as a single integer in base
        # `bins` is a bijection (collision-free) over
        # pd.Series(...).value_counts(), and counting it via
        # np.bincount avoids the heavy overhead of building a pandas
        # Series plus hash/groupby on every call. The summation order of
        # the terms of -sum(p*log2(p)) is not guaranteed to match a
        # groupby-based computation, producing floating-point noise of
        # ~1e-15 (irrelevant -- te_val is rounded to 6 decimal places
        # before use).
        codes = np.zeros(len(arrs[0]), dtype=np.int64)
        for a in arrs:
            codes = codes * bins + a
        counts = np.bincount(codes)
        counts = counts[counts > 0]
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-12))

    h_yt_yt1     = entropy_joint(qyt,  qyt1)
    h_yt1        = entropy_joint(qyt1,)
    h_yt_yt1_xl  = entropy_joint(qyt,  qyt1, qxl)
    h_yt1_xl     = entropy_joint(qyt1, qxl)

    te = h_yt_yt1 - h_yt1 - h_yt_yt1_xl + h_yt1_xl
    return max(float(te), 0.0)


def _te_knn(x: np.ndarray, y: np.ndarray, lag: int, k: int = 4) -> float:
    """Transfer Entropy via the Kraskov-Stoegbauer-Grassberger KNN estimator."""
    T = len(y)
    if T - lag < 20:
        return 0.0

    yt  = y[lag:].copy()
    yt1 = y[lag - 1:-1].copy()
    xl  = x[:T - lag].copy()

    valid = ~(np.isnan(yt) | np.isnan(yt1) | np.isnan(xl))
    yt, yt1, xl = yt[valid], yt1[valid], xl[valid]
    n = len(yt)
    if n < max(20, k + 2):
        return 0.0

    def _norm(v):
        s = v.std()
        return (v - v.mean()) / s if s > 1e-10 else v - v.mean()

    yt, yt1, xl = _norm(yt), _norm(yt1), _norm(xl)

    W = np.column_stack([yt, yt1, xl])
    Z = np.column_stack([yt, yt1])
    Y = yt1.reshape(-1, 1)
    V = np.column_stack([yt1, xl])

    def count_neighbors(space, eps):
        nn = NearestNeighbors(metric='chebyshev', algorithm='ball_tree')
        nn.fit(space)
        counts = np.array([
            nn.radius_neighbors([p], radius=e, return_distance=False)[0].shape[0] - 1
            for p, e in zip(space, eps)
        ])
        return np.maximum(counts, 1)

    nn_w = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev', algorithm='ball_tree')
    nn_w.fit(W)
    dist_w, _ = nn_w.kneighbors(W)
    eps = dist_w[:, -1]

    n_z = count_neighbors(Z, eps)
    n_y = count_neighbors(Y, eps)
    n_v = count_neighbors(V, eps)

    from scipy.special import digamma as psi
    te = (psi(k)
          - np.mean(psi(n_z + 1))
          + np.mean(psi(n_y + 1))
          - np.mean(psi(n_v + 1)))

    return max(float(te), 0.0)


def _te_best_over_lags(x: np.ndarray, y: np.ndarray,
                        max_lag: int, method: str, bins: int, k: int,
                        lag_penalty_fn: LagPenaltyFn) -> tuple:
    """Scans lag=1..max_lag and returns (best_score, best_te, best_lag)."""
    best_score, best_te, best_lag = 0.0, 0.0, 1
    te_fn = _te_histogram if method == 'histogram' else _te_knn
    te_kwargs = {'bins': bins} if method == 'histogram' else {'k': k}

    for lag in range(1, min(max_lag + 1, len(x) // 4)):
        te_val  = te_fn(x, y, lag, **te_kwargs)
        score   = te_val * lag_penalty_fn(lag)
        if score > best_score:
            best_score, best_te, best_lag = score, te_val, lag

    return best_score, best_te, best_lag


def best_lag_te(x: np.ndarray, y: np.ndarray,
                 max_lag: int, method: str = 'histogram',
                 bins: int = 8, k: int = 4,
                 lag_penalty_fn: Optional[LagPenaltyFn] = None) -> tuple:
    if lag_penalty_fn is None:
        lag_penalty_fn = DEFAULT_LAG_PENALTY
    _, best_te, best_lag = _te_best_over_lags(x, y, max_lag, method, bins, k, lag_penalty_fn)
    return best_te, best_lag


def max_stat_permutation_pvalue(x: np.ndarray, y: np.ndarray,
                                 max_lag: int, observed_score: float,
                                 lag_penalty_fn: LagPenaltyFn,
                                 n_perms: int = 50,
                                 method: str = 'histogram',
                                 bins: int = 8, k: int = 4,
                                 rng_seed: int = 42) -> float:
    """
    P-value corrected for the multiplicity of lags tested by argmax over
    [1, max_lag]. The
    permutation_pvalue() below only tests whether the TE of the
    ALREADY-CHOSEN lag is surprising against chance -- it does not
    account for the fact that up to max_lag lags were scanned to find it
    (look-elsewhere effect). Here, the null distribution is that of the
    MAXIMUM penalized score (TE * lambda(lag)) over ALL lags, in the
    same spirit as the max-statistic used in multi-hypothesis
    permutation tests; complementary to (not a replacement for) the
    batch FDR-BH correction over pairs done by the caller
    (discovery.filter_causal_candidates).
    """
    if observed_score <= 0.0:
        return 1.0
    rng = np.random.default_rng(rng_seed)
    null_max = np.empty(n_perms)
    for i in range(n_perms):
        x_perm = rng.permutation(x)
        best_null, _, _ = _te_best_over_lags(x_perm, y, max_lag, method, bins, k, lag_penalty_fn)
        null_max[i] = best_null
    pval = float(np.mean(null_max >= observed_score))
    return max(pval, 1.0 / n_perms)


def permutation_pvalue(x: np.ndarray, y: np.ndarray,
                        lag: int, observed_te: float,
                        n_perms: int = 100,
                        method: str = 'histogram',
                        bins: int = 8, k: int = 4) -> float:
    rng = np.random.default_rng(42)
    te_fn = _te_histogram if method == 'histogram' else _te_knn
    te_kwargs = {'bins': bins} if method == 'histogram' else {'k': k}

    null_dist = []
    for _ in range(n_perms):
        x_perm = rng.permutation(x)
        te_null = te_fn(x_perm, y, lag, **te_kwargs)
        null_dist.append(te_null)

    null_dist = np.array(null_dist)
    pval = float(np.mean(null_dist >= observed_te))
    return max(pval, 1.0 / n_perms)


def te_filter_pair(args):
    """
    Tests a pair (src->tgt) with Transfer Entropy. Executable in parallel.

    Always returns, even below min_te or the nominal alpha -- the accept
    decision is made in batch by discovery.filter_causal_candidates()
    after multiple-comparison correction over 'pvalue_discovery'.
    """
    (src_key, tgt_key, x, y,
     max_lag, alpha, min_te,
     method, bins, k, n_perms,
     lag_penalty_fn,
     correct_lag_multiplicity, n_perms_lag) = args

    te_val, best_lag = best_lag_te(x, y, max_lag, method, bins, k,
                                    lag_penalty_fn=lag_penalty_fn)
    pen_score = lag_penalty_fn(best_lag)

    if te_val < min_te:
        return {
            'source':            src_key,
            'target':            tgt_key,
            'lag':               best_lag,
            'te':                round(te_val, 6),
            'pvalue_discovery':  1.0,
            'filter_method':     f'TE_{method}',
            'lag_penalty_score': round(pen_score, 4),
        }

    pval_lag = permutation_pvalue(
        x, y, best_lag, te_val,
        n_perms=n_perms, method=method, bins=bins, k=k
    )

    pval_report = pval_lag
    if correct_lag_multiplicity:
        observed_score = te_val * pen_score
        pval_report = max_stat_permutation_pvalue(
            x, y, max_lag, observed_score, lag_penalty_fn,
            n_perms=n_perms_lag, method=method, bins=bins, k=k,
        )

    return {
        'source':            src_key,
        'target':            tgt_key,
        'lag':               best_lag,
        'te':                round(te_val, 6),
        'pvalue_discovery':  round(pval_report, 4),   # RAW -- batch correction applied by the caller
        'pvalue_discovery_lag_raw': round(pval_lag, 4),  # permutation on the winning lag only (no lag-multiplicity correction)
        'filter_method':     f'TE_{method}',
        'lag_penalty_score': round(pen_score, 4),
    }

