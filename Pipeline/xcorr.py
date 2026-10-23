"""
xcorr.py — Stage 3 (correlation pre-filter): lagged cross-correlation.

Always returns the best result found for the pair (even below
alpha/min_corr) — the accept/reject decision is made in batch by
discovery.filter_causal_candidates(), AFTER the multiple-comparison
correction (see causal_common._correct_pvalues).

LAG MULTIPLICITY CORRECTION (optional, correct_lag_multiplicity):
the parametric p-value from pearsonr above is computed only at the winning
lag from the argmax, without accounting for the fact that up to max_lag lags
were tested to get there (look-elsewhere effect). When enabled,
max_stat_permutation_pvalue() replaces that
p-value with a permutation p-value based on the MAXIMUM score over all lags,
complementing (not replacing) the batch FDR-BH correction over pairs done by
the caller.
"""

import numpy as np
from scipy import stats


def _xcorr_score_all_lags(x, y, max_lag, lag_penalty_fn):
    """Scans lag=1..max_lag and returns the best (score, corr, lag, pval)."""
    best_score, best_corr, best_lag, best_pval = 0.0, 0.0, 1, 1.0
    for lag in range(1, min(max_lag + 1, len(x) // 4)):
        x_lag = np.roll(x, lag).astype(float)
        x_lag[:lag] = np.nan
        valid = ~np.isnan(x_lag) & ~np.isnan(y)
        if valid.sum() < 15:
            continue
        try:
            corr, pval = stats.pearsonr(x_lag[valid], y[valid])
        except Exception:
            continue
        score = abs(corr) * lag_penalty_fn(lag)
        if score > best_score:
            best_score, best_corr, best_lag, best_pval = score, corr, lag, pval
    return best_score, best_corr, best_lag, best_pval


def max_stat_permutation_pvalue(x, y, max_lag, observed_score, lag_penalty_fn,
                                 n_perms=50, rng_seed=42):
    """
    P-value corrected for the multiplicity of lags tested by the argmax over
    [1, max_lag]. Builds the null distribution of the MAXIMUM penalized score
    (|corr| * lambda(lag)) over ALL lags — not just the winning lag — by
    permuting the source series, and compares the observed score against that
    null maximum. n_perms permutations x up to max_lag lags each.
    """
    if observed_score <= 0.0:
        return 1.0
    rng = np.random.default_rng(rng_seed)
    null_max = np.empty(n_perms)
    for i in range(n_perms):
        x_perm = rng.permutation(x)
        best_null, _, _, _ = _xcorr_score_all_lags(x_perm, y, max_lag, lag_penalty_fn)
        null_max[i] = best_null
    pval = float(np.mean(null_max >= observed_score))
    return max(pval, 1.0 / n_perms)


def xcorr_filter_pair(args):
    (src_key, tgt_key, x, y, max_lag, alpha, min_corr, lag_penalty_fn,
     correct_lag_multiplicity, n_perms_lag) = args

    best_score, best_corr, best_lag, best_pval = _xcorr_score_all_lags(x, y, max_lag, lag_penalty_fn)
    pen_score = lag_penalty_fn(best_lag)

    pval_report = best_pval
    if correct_lag_multiplicity:
        pval_report = max_stat_permutation_pvalue(
            x, y, max_lag, best_score, lag_penalty_fn, n_perms=n_perms_lag,
        )

    return {
        'source':            src_key,
        'target':            tgt_key,
        'lag':               best_lag,
        'corr':              round(best_corr, 6),
        'pvalue_discovery':  round(pval_report, 4),   # RAW — batch correction applied by the caller
        'pvalue_discovery_lag_raw': round(best_pval, 4),  # parametric, winning lag only (no lag-multiplicity correction)
        'filter_method':     'xcorr',
        'lag_penalty_score': round(pen_score, 4),
    }

