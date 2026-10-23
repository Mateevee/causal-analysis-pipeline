"""
results.py — Stage 6 output: builds the results DataFrame and applies consolidated
statistical checks (t-stat, CI coefficient of variation, asymmetry, BH
correction).

Also passes through extra diagnostic fields produced by the DML cascade
(dml_engine.py) without breaking pipelines/models that do not produce
them — see EXTRA_FIELDS below.
"""

import numpy as np
import pandas as pd

try:
    from statsmodels.stats.multitest import multipletests
    HAS_MULTIPLETESTS = True
except ImportError:
    HAS_MULTIPLETESTS = False


# Extra fields that the DML cascade (dml_engine.py) may attach to each
# pair's result; must match diagnostics_aux.py's dict keys verbatim.
EXTRA_FIELDS = (
    'cascade_confidence', 'cascade_log',
    'n_models_agree', 'models_accepted',
    'robustness_value', 'robustness_value_qa',
    'dcor_residuals', 'corr_linear_residuals',
    'reset_fstat', 'reset_pvalue', 'nonlinear_suspected',
    'spearman_residuals', 'spearman_pvalue',
    'hsic_residuals',
    'chow_fstat', 'chow_pvalue', 'chow_breakpoint',
    'permutation_pvalue', 'theta_permutation_std',
    'ci_lower_boot', 'ci_upper_boot', 'theta_boot_mean',
    'ess_treatment', 'ess_outcome',
    'theta_forward_lite', 'theta_reverse_lite', 'causal_asymmetry',
    'ccm_skill', 'ccm_convergent', 'ccm_skill_by_L',
)


def build_results(candidates: list, estimates: list) -> pd.DataFrame:
    """Combines candidates and estimates into a clean DataFrame."""
    rows = []
    for cand, est in zip(candidates, estimates):
        src, tgt = cand['source'], cand['target']
        sw, sf = (src.split('__', 1) + ['?'])[:2]
        tw, tf = (tgt.split('__', 1) + ['?'])[:2]
        lag_val   = cand['lag']
        lag_score = round(cand.get('lag_penalty_score', np.nan), 4)
        row = {
            'source_entity':      sw,
            'source_property':    sf,
            'target_entity':      tw,
            'target_property':    tf,
            'lag':            lag_val,
            'lag_penalty_score': lag_score,
            'transfer_entropy': round(cand.get('te', cand.get('corr', 0.0)), 6),
            'filter_method':  cand.get('filter_method', 'TE'),
            'pvalue_discovery_raw':       cand.get('pvalue_discovery', np.nan),
            'pvalue_discovery_corrected': cand.get('pvalue_corrected', np.nan),
            'causal_effect':  round(est['effect'], 4)   if not np.isnan(est['effect'])   else np.nan,
            'standard_error': round(est['se'],     4)   if not np.isnan(est['se'])        else np.nan,
            'direction':      est['direction'],
            'ci_95_lower':    round(est['ci_lower'], 4) if not np.isnan(est['ci_lower']) else np.nan,
            'ci_95_upper':    round(est['ci_upper'], 4) if not np.isnan(est['ci_upper']) else np.nan,
            'pvalue':         round(est['pvalue'], 4)   if not np.isnan(est['pvalue'])   else np.nan,
            'significant':    est['significant'],
            'strength':       abs(est['effect'])         if not np.isnan(est['effect'])   else 0.0,
            'dml_model_used': est.get('dml_model_used', '?'),
        }

        for extra_key in EXTRA_FIELDS:
            if extra_key in est:
                row[extra_key] = est[extra_key]

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
              .sort_values('strength', ascending=False)
              .reset_index(drop=True))


def apply_statistical_checks(results: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
        t_stat, cv_ci, asymmetry_ci, pvalue_bh, significant_bh
    """
    if results.empty:
        return results

    df = results.copy()

    df['t_stat'] = (df['causal_effect'] / df['standard_error']).round(4)

    ci_width = df['ci_95_upper'] - df['ci_95_lower']
    df['cv_ci'] = (ci_width / df['causal_effect'].abs()).round(4)

    dist_lower = df['causal_effect'] - df['ci_95_lower']
    dist_upper = df['ci_95_upper'] - df['causal_effect']
    df['asymmetry_ci'] = (dist_lower - dist_upper).abs().round(4)

    pvals = df['pvalue'].fillna(1.0).values
    if HAS_MULTIPLETESTS:
        reject, pvals_bh, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        df['pvalue_bh'] = pvals_bh.round(4)
        df['significant_bh'] = reject
    else:
        n = len(pvals)
        order = np.argsort(pvals)
        pvals_bh = np.zeros(n)
        for i, idx in enumerate(order):
            pvals_bh[idx] = min(pvals[idx] * n / (i + 1), 1.0)
        for i in range(len(order) - 2, -1, -1):
            pvals_bh[order[i]] = min(pvals_bh[order[i]], pvals_bh[order[i + 1]])
        df['pvalue_bh'] = np.round(pvals_bh, 4)
        df['significant_bh'] = pvals_bh < 0.05

    return df


# What "the pair comes with its effect estimate" means: all five present
# and finite.
ESTIMATE_COLUMNS = ('causal_effect', 'standard_error', 'ci_95_lower',
                    'ci_95_upper', 'pvalue')


def require_effect_estimate(results: pd.DataFrame) -> tuple:
    """
    Output invariant: a pair is only reported (significant=True) if it
    carries its effect estimate (ESTIMATE_COLUMNS). The predictive
    competition (8a-bis) and the lag-0 skeleton (8b) accept a pair on
    evidence other than the DML estimate, so a pair whose estimation failed
    (too few valid samples, theta not computable in any cascade model)
    could otherwise leave the pipeline unmeasured.

    Such rows are marked significant=False with
    structural_rejection_reason='no_effect_estimate'; no row is removed.

    Returns (results, number of pairs rejected).
    """
    if results.empty or 'significant' not in results.columns:
        return results, 0
    measured = pd.Series(True, index=results.index)
    for col in ESTIMATE_COLUMNS:
        if col not in results.columns:
            measured[:] = False
            break
        measured &= np.isfinite(pd.to_numeric(results[col], errors='coerce'))
    unmeasured = (results['significant'] == True) & ~measured  # noqa: E712
    n = int(unmeasured.sum())
    if n:
        if 'structural_rejection_reason' not in results.columns:
            results['structural_rejection_reason'] = None
        results.loc[unmeasured, 'structural_rejection_reason'] = 'no_effect_estimate'
        results.loc[unmeasured, 'significant'] = False
    return results, n


# Stage 7 output and the estimate column each one is the penalized form of.
PENALIZED_COLUMNS = (('penalized_effect', 'causal_effect'),
                     ('ci_lower_pen', 'ci_95_lower'),
                     ('ci_upper_pen', 'ci_95_upper'))


def ensure_penalized_effect(results: pd.DataFrame) -> tuple:
    """
    Output invariant: a reported pair (significant=True) always carries a
    penalized effect (PENALIZED_COLUMNS). Stage 7 writes it as estimate *
    causal_penalty_score, but a pair can be reported without that product
    being meaningful:

      - the lag-0 links of Stage 8b are appended after Stage 7 (no score);
      - the Stage 8 guards (incoherent-evidence fallback, 8a-bis) accept
        pairs Stage 7 had given weight 0, so their product is 0;
      - with UniformCausalPenalty Stage 7 does not run (no column at all).

    In all three the penalty was not applied to the pair, i.e. its weight is
    1: the penalized columns receive the unpenalized estimate.
    'causal_penalty_score' is left as Stage 7 wrote it (the fusion verdict),
    and 'penalized_effect_source' records which rows were filled here
    ('unpenalized') and which come from Stage 7 ('stage7').

    Returns (results, number of pairs filled).
    """
    if results.empty or 'significant' not in results.columns \
            or 'causal_effect' not in results.columns:
        return results, 0
    score = (pd.to_numeric(results['causal_penalty_score'], errors='coerce')
             if 'causal_penalty_score' in results.columns
             else pd.Series(np.nan, index=results.index))
    penalized = (pd.to_numeric(results['penalized_effect'], errors='coerce')
                 if 'penalized_effect' in results.columns
                 else pd.Series(np.nan, index=results.index))
    fill = (results['significant'] == True) & (penalized.isna() | ~(score > 0))  # noqa: E712
    for pen_col, est_col in PENALIZED_COLUMNS:
        if est_col not in results.columns:
            continue
        if pen_col not in results.columns:
            results[pen_col] = np.nan
        results.loc[fill, pen_col] = pd.to_numeric(
            results.loc[fill, est_col], errors='coerce').round(4)
    results['penalized_effect_source'] = np.where(
        fill, 'unpenalized', np.where(penalized.notna(), 'stage7', None))
    return results, int(fill.sum())
