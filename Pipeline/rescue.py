"""
rescue.py — Stage 5 (correlation pre-filter): rescues applied to pairs
discarded by xcorr/TE.

    granger_rescue_pair : Granger Causality F-test.
    pearson_rescue_pair : Pearson correlation with block bootstrap.

Both run for ALL pairs rejected by xcorr (not just the "observed"/watch_pairs
ones) — see discovery.py, rescue section.

LAG MULTIPLICITY CORRECTION (LOOK-ELSEWHERE) IN THE RESCUES
-------------------------------------------------------------
Both rescues pick the lag by ARGMIN of the p-value over up to max_lag
candidates and compare the winning lag's RAW p-value against alpha. This
is the winner's curse: with ~24 lags scanned, a pair with no real
relationship at all has a high probability of showing at least one lag
with p < alpha by pure sampling chance.

Stages 3-4 (xcorr/TE) already handle this with TWO layers
(discovery.py): batch FDR-BH over the pairs + max-statistic permutation
p-value over the lags (correct_lag_multiplicity). Without
`correct_lag_multiplicity` the rescues have NEITHER of the two, so edges
entering via the rescues are expected to carry more false positives than
edges entering via TE, purely from this uncorrected lag search.

`correct_lag_multiplicity` applies the Sidak correction over the number
of lags ACTUALLY evaluated (not over the nominal max_lag, which may be
larger than the lags feasible given the series length):

    corrected_p = 1 - (1 - raw_p) ** n_lags_evaluated

It neither caps nor penalizes any lag -- the rescue remains free to pick
any lag; it only pays the correct statistical price for the
search. Same correction family already used by the DML cascade for the
selection among N models (see dml_engine.py, "Sidak correction").

Off by default; enabled via
discovery.filter_causal_candidates(correct_lag_multiplicity_rescue=True)
-> run_pipeline(rescue_correct_lag_multiplicity=True).
"""

import numpy as np
from scipy import stats
from typing import Optional


def _sidak(p: float, n_tests: int) -> float:
    """P-value corrected for selecting the BEST of `n_tests` (Sidak)."""
    if n_tests <= 1:
        return float(p)
    p = min(max(float(p), 0.0), 1.0)
    return float(1.0 - (1.0 - p) ** n_tests)


def granger_rescue_pair(args) -> Optional[dict]:
    """Granger Causality test as a rescue for pairs discarded by TE."""
    (src_key, tgt_key, x, y, max_lag, alpha_rescue, lag_penalty_fn,
     correct_lag_multiplicity) = args

    T = len(y)
    best_score = 0.0
    best_pval  = 1.0
    best_lag   = 1
    best_fstat = 0.0
    n_lags_evaluated = 0

    for lag in range(1, min(max_lag + 1, T // 4)):
        y_t   = y[lag:]
        y_lag = y[lag - 1: T - 1]
        x_lag = x[: T - lag]

        valid = ~(np.isnan(y_t) | np.isnan(y_lag) | np.isnan(x_lag))
        n = valid.sum()
        if n < max(20, lag * 3 + 4):
            continue

        yt  = y_t[valid]
        yl  = y_lag[valid]
        xl  = x_lag[valid]

        X_r = np.column_stack([np.ones(n), yl])
        X_u = np.column_stack([np.ones(n), yl, xl])

        try:
            b_r, res_r, _, _ = np.linalg.lstsq(X_r, yt, rcond=None)
            b_u, res_u, _, _ = np.linalg.lstsq(X_u, yt, rcond=None)

            rss_r = float(np.sum((yt - X_r @ b_r) ** 2))
            rss_u = float(np.sum((yt - X_u @ b_u) ** 2))

            k_r, k_u = X_r.shape[1], X_u.shape[1]
            df1 = k_u - k_r
            df2 = n - k_u

            if df2 < 1 or rss_u < 1e-12:
                continue

            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
            pval   = float(1 - stats.f.cdf(f_stat, df1, df2))
            n_lags_evaluated += 1

            if pval < best_pval:
                score = (1.0 - pval) * lag_penalty_fn(lag)
                if score > best_score:
                    best_score = score
                    best_pval  = pval
                    best_lag   = lag
                    best_fstat = f_stat

        except (np.linalg.LinAlgError, ValueError):
            continue

    # Look-elsewhere correction over the lags scanned by the argmin
    # above (see module docstring). n_lags_evaluated only counts the
    # lags that actually produced a valid test.
    effective_pvalue = (_sidak(best_pval, n_lags_evaluated)
                         if correct_lag_multiplicity else best_pval)

    if effective_pvalue < alpha_rescue:
        pen_score = lag_penalty_fn(best_lag)
        return {
            'source':           src_key,
            'target':           tgt_key,
            'lag':              best_lag,
            'te':               0.0,
            'granger_fstat':    round(best_fstat, 4),
            'pvalue_discovery': round(effective_pvalue, 4),
            'pvalue_discovery_raw': round(best_pval, 4),
            'n_lags_evaluated': n_lags_evaluated,
            'filter_method':    'Granger_rescue',
            'lag_penalty_score': round(pen_score, 4),
        }
    return None


def pearson_rescue_pair(args) -> Optional[dict]:
    """Rescue via Pearson cross-correlation with block bootstrap."""
    (src_key, tgt_key, x, y,
     max_lag, alpha_pearson, min_corr_rescue, n_boot, block_size,
     lag_penalty_fn, correct_lag_multiplicity) = args

    T = len(y)
    rng = np.random.default_rng(42)

    best_score, best_corr, best_lag, best_pval = 0.0, 0.0, 1, 1.0
    n_lags_evaluated = 0

    for lag in range(1, min(max_lag + 1, T // 4)):
        x_lag = np.roll(x, lag).astype(float)
        x_lag[:lag] = np.nan
        valid = ~(np.isnan(x_lag) | np.isnan(y))
        n_valid = valid.sum()
        if n_valid < 15:
            continue

        xv = x_lag[valid]
        yv = y[valid]

        try:
            corr_obs, _ = stats.pearsonr(xv, yv)
        except Exception:
            continue

        if abs(corr_obs) < min_corr_rescue:
            continue

        bs = max(2, block_size if block_size else max(3, n_valid // 10))
        n_blocks = int(np.ceil(n_valid / bs))

        # Vectorized: since the upper bound of rng.integers is
        # n_valid - bs + 1, every drawn start satisfies start + bs <= n_valid,
        # so every block is built at full length `bs` and only the
        # concatenation is truncated to n_valid.
        starts_all = rng.integers(0, max(1, n_valid - bs + 1), size=(n_boot, n_blocks))
        offsets    = np.arange(bs)
        idx_all    = (starts_all[:, :, None] + offsets[None, None, :]).reshape(
            n_boot, n_blocks * bs)[:, :n_valid]
        Y_boot = yv[idx_all]

        xc  = xv - xv.mean()
        sxx = float(np.sum(xc ** 2))
        yc  = Y_boot - Y_boot.mean(axis=1, keepdims=True)
        syy = np.sum(yc ** 2, axis=1)
        sxy = yc @ xc

        denom = np.sqrt(sxx * syy)
        with np.errstate(invalid='ignore', divide='ignore'):
            corr_boot = np.where(denom > 1e-12, sxy / denom, np.nan)
        # NaN (degenerate block, e.g. constant y_boot) never satisfies
        # >=: not counted in count_ge, but still in the n_boot denominator.
        count_ge = int(np.sum(np.abs(corr_boot) >= abs(corr_obs)))

        pval_boot = max(float(count_ge) / n_boot, 1.0 / n_boot)
        n_lags_evaluated += 1

        score = abs(corr_obs) * lag_penalty_fn(lag)
        if score > best_score:
            best_score = score
            best_corr  = corr_obs
            best_lag   = lag
            best_pval  = pval_boot

    # Look-elsewhere correction over the scanned lags (see module
    # docstring). NOTE: the bootstrap p-value already has a floor of
    # 1/n_boot, so the correction here is applied on top of that floor
    # -- with a small n_boot the effect is stronger, which is the
    # correct behavior (less p-value resolution = less evidence to back
    # the search).
    effective_pvalue = (_sidak(best_pval, n_lags_evaluated)
                         if correct_lag_multiplicity else best_pval)

    if effective_pvalue < alpha_pearson and abs(best_corr) >= min_corr_rescue:
        pen_score = lag_penalty_fn(best_lag)
        return {
            'source':           src_key,
            'target':           tgt_key,
            'lag':              best_lag,
            'te':               0.0,
            'corr':             round(float(best_corr), 6),
            'pvalue_discovery': round(effective_pvalue, 4),
            'pvalue_discovery_raw': round(best_pval, 4),
            'n_lags_evaluated': n_lags_evaluated,
            'filter_method':    'Pearson_rescue',
            'lag_penalty_score': round(pen_score, 4),
        }
    return None

