"""
dml_engine.py -- Stage 6: DML with dynamic confounding.

Native cascade: bayesian_ridge -> random_forest -> kernel_ridge ->
nadaraya_watson, with acceptance by agreement (cascade_confidence), Sidak
correction, nonlinearity diagnostics (dcor/RESET), and injection of
'lag_median' before the causal penalty.

SIDAK CORRECTION: the cascade tests N models; accepting the first one that
passes every criterion would be equivalent to testing N hypotheses and
reporting the first "significant" one (true FWER = 1-(1-pval_max)^N, not
pval_max). When dml_model in ('auto', 'auto_v2'), the effective pval_max
used in the criteria is Sidak-corrected:
    effective_pval_max = 1 - (1 - pval_max) ** (1 / N)
'significant' can only be True when the result was actually accepted by
the cascade (cascade_accepted=True) -- never on a fallback or when models
disagree in sign.

THETA STABILITY: each fold computes its own theta_fold; cv_theta =
std(thetas_fold)/|mean(thetas_fold)| is logged and (optionally) used as a
rejection criterion (cfg.cv_theta_max). r2_nuisance_T/O (the nuisance's
out-of-fold R^2) is also logged.

ACCEPTANCE BY AGREEMENT: ALL models in the cascade run; the final result
carries a confidence level (cascade_confidence):
    'high'   : 2+ models passed every criterion AND agree on the sign of
               the effect
    'medium' : exactly 1 model passed (no other model left to confirm it)
    'low'    : no model passed (fallback), OR 2+ passed but disagree in
               sign (contradictory evidence)
'significant' can only be True when confidence in ('high', 'medium').

LAG-MEDIAN INJECTION: apply_causal_penalty writes, BEFORE causal_penalty_fn
is applied to each row, a 'lag_median' column holding the median of the
'lag' column over the ENTIRE run's DataFrame -- available both to
causal_penalty_fn (e.g. to penalize pairs whose lag strays far from the
run's typical lag) and to any downstream code that reads the output
DataFrame, even if causal_penalty_fn does not use it.

Only has a practical effect when this function is actually called by
run_pipeline -- which only happens when causal_penalty_fn !=
UniformCausalPenalty. Running run_pipeline without a custom causal penalty
skips this entire stage, and 'lag_median' does not appear in the final
DataFrame.

NONLINEARITY DIAGNOSTICS: 'theta' comes from a LINEAR regression of the
residuals (res_O ~ res_T) -- adequate when the effect of T on O is
approximately linear, but in quadratic relationships (e.g.
O_t = a*T_(t-lag)^2 + noise) theta can end up close to zero even with a
genuinely strong dependence. This is not corrected here -- doing so would
require changing the estimator itself. What exists instead are diagnostics
computed on the SAME res_T/res_O that produce theta, logged per model in
cascade_log and attached to the winning model's final_result, purely
informative (they never enter _evaluate, never decide
accepted/fails/cascade_confidence/significant):
  - dcor_residuals        : distance correlation (Szekely, Rizzo &
    Bakirov, 2007) between res_T and res_O -- dCor(x,y)=0 ONLY when x and y
    are genuinely independent, so it captures nonlinear/non-monotonic
    dependence such as y=x^2.
  - reset_fstat/pvalue   : a RESET-type test (Ramsey, 1969) restricted to
    the quadratic term -- a low p-value = evidence that the linear model is
    misspecified for this relationship.
  - corr_linear_residuals : Pearson(res_T, res_O), kept as a reference -- a
    large GAP between dCor and |corr_linear_residuals| is the direct
    symptom of the nonlinearity that theta is missing.
  - nonlinear_suspected   : convenience flag, True when dcor_residuals >
    |corr_linear_residuals| + 0.15 AND reset_pvalue < 0.05 simultaneously.
On/off switch: cfg.compute_nonlinear_diagnostics (default True). Cost: dCor
is O(n^2), subsampled (without replacement, fixed seed) up to
cfg.dcor_max_n points.

EXTRA DIAGNOSTICS: a second batch of diagnostics, also strictly
informative, coming from diagnostics_aux.py (defensive import -- if the
file does not exist, the extra fields come out as None/absent without
breaking anything):
  RESIDUAL (over the SAME res_T/res_O of each model in the cascade):
    spearman_residuals/pvalue, hsic_residuals, chow_fstat/pvalue/
    breakpoint, permutation_pvalue, theta_permutation_std,
    ci_lower_boot/upper_boot, theta_boot_mean.
  RAW (once per pair, over T_arr/O_arr/W_arr BEFORE the cascade's
  residualization): ess_treatment/outcome, theta_forward_lite/reverse_lite,
  causal_asymmetry, ccm_skill/convergent/by_L (CCM off by default --
  cfg.enable_ccm).
Run automatically as soon as diagnostics_aux.py is found. Fine-tune via
configure_extra_diagnostics(...).

CONTROL WINDOW: `max_lag` as received by estimate_all_effects is a
SWEEP/calibration parameter (used by another pipeline stage to scan lag
candidates), not the ideal control-window size for a pair that already has
a specific `lag` being tested. See _compute_control_window() for the full
formula (it respects both the max_lag_original ceiling and the sample
floor).

SWAP CRITERION -- the model is discarded if ANY condition fails:
    1. var(res_T)   >= var_res_T_threshold
    2. |t_stat|     >= t_min
    3. pvalue       <= effective pval_max (Sidak-corrected in cascade mode)
    4. the CI does not cross zero ambiguously
    5. cv_ci        <= cv_ci_max
    6. sign(theta)  == sign(raw_corr)
    7. strength     >= strength_min
    8. cv_theta     <= cv_theta_max
    9. r2_nuisance  >= r2_nuisance_min       [optional, default off]
   (The nonlinearity and extra diagnostics above are NOT part of this list
   and cannot reject/accept a model.)

FALLBACK: if no model passes every criterion, returns the "closest"
    model's result (proximity score), always with significant=False and
    cascade_confidence='low'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Library of extra diagnostics (Spearman, HSIC, Chow, permutation,
# bootstrap, ESS, reverse causality, CCM). Defensive import: if the file
# is not present, the extra fields come out absent/None without breaking
# anything -- see the EXTRA DIAGNOSTICS section in the module docstring.
try:
    from .diagnostics_aux import (
        ExtraDiagnosticsConfig,
        DEFAULT_EXTRA_CONFIG,
        extra_residual_diagnostics,
        extra_raw_diagnostics,
    )
    HAS_DIAG_EXTRA = True
except ImportError:
    HAS_DIAG_EXTRA = False

# Extra-diagnostics configuration as a MODULE-LEVEL GLOBAL, independent of
# CascadeConfig. Runs automatically (ExtraDiagnosticsConfig defaults --
# CCM off, everything else on) as soon as diagnostics_aux.py is found. To
# tweak something specific (e.g. turn CCM on), call
# configure_extra_diagnostics(...) at any point before estimate_all_effects.
_ACTIVE_DIAG_EXTRA_CFG = DEFAULT_EXTRA_CONFIG if HAS_DIAG_EXTRA else None


def configure_extra_diagnostics(diag_cfg: "ExtraDiagnosticsConfig") -> None:
    """
    Sets the extra-diagnostics configuration globally. Can be
    called at any point before estimate_all_effects (reading the global
    happens at run time, inside _dml_single). If diagnostics_aux.py is
    not available, this is a no-op (a warning is printed).
    """
    global _ACTIVE_DIAG_EXTRA_CFG
    if not HAS_DIAG_EXTRA:
        print(
            "[dml_engine] WARNING: diagnostics_aux.py not found -- "
            "configure_extra_diagnostics() has no effect."
        )
        return
    _ACTIVE_DIAG_EXTRA_CFG = diag_cfg


# apply_causal_penalty needs to be available as an ATTRIBUTE of this
# module (not merely imported) so that pipeline.py calls it via
# `dml_engine.apply_causal_penalty(...)` rather than via
# `from dml_engine import apply_causal_penalty` -- see the docstring of
# Pipeline/pipeline.py.
#
# penalties.apply_causal_penalty has NO drop_zero parameter -- that
# function never removes any row, ever (see penalties.py). The
# physical removal of non-significant rows happens AFTER Stage 8 -- see
# pipeline._finalize, which removes rows with significant=False using
# the value ALREADY RESOLVED by Stage 8 (not the raw value from this
# stage).
#
# Rationale: removing rows BEFORE Stage 8a runs erases real competitors
# from the candidate graph -- this (a) contaminates the topology
# auto-detection used by optimize_graph_structure/apply_structural_filter
# (a dense/coupled domain can be misclassified as sparse once its real
# competitors are missing) and (b) removes real competitors from the
# reverse-edge contest and from the BIC/ARD parent competition, changing
# Stage 8a's result unpredictably.
from .penalties import apply_causal_penalty as _apply_causal_penalty_raw


def apply_causal_penalty(results, causal_penalty_fn, inplace_effect: bool = False):
    """
    Wrapper around penalties.apply_causal_penalty used at Stage 7
    (BEFORE the structural pruning of Stage 8a).

    In addition to forwarding to penalties.apply_causal_penalty, it
    writes, BEFORE that, a 'lag_median' column holding the median of the
    'lag' column over the ENTIRE run's DataFrame (every pair that
    survived up to this point) -- available both to causal_penalty_fn
    (e.g. to penalize pairs whose lag strays far from the run's typical
    lag) and to any downstream code that reads the output DataFrame, even
    if causal_penalty_fn does not use it.

    Only has a practical effect when this function is actually called by
    run_pipeline -- which only happens when causal_penalty_fn !=
    UniformCausalPenalty. Running run_pipeline without a custom causal
    penalty skips this entire stage, and 'lag_median' does not appear in
    the final DataFrame.
    """
    out = results.copy()
    if 'lag' in out.columns and not out.empty:
        median = float(pd.to_numeric(out['lag'], errors='coerce').median())
    else:
        median = float('nan')
    out['lag_median'] = median

    return _apply_causal_penalty_raw(
        out, causal_penalty_fn=causal_penalty_fn,
        inplace_effect=inplace_effect,
    )


# =============================================================================
# Nadaraya-Watson Regressor
# =============================================================================

class NadarayaWatsonRegressor(BaseEstimator, RegressorMixin):
    """
    Nadaraya-Watson kernel regression -- local density estimation.

    Estimates E[Y|X=x] as a weighted average of the training y values,
    with weights proportional to the Gaussian kernel evaluated at the
    distance ||x - xi||.

    Bandwidth via Scott's rule: h = n^(-1/(d+4)) in the standardized
    space. Falls back to the mean when the sum of the weights collapses
    (isolated point).
    """

    def __init__(self, bandwidth=None, min_weight_sum=1e-6):
        self.bandwidth = bandwidth
        self.min_weight_sum = min_weight_sum

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._scaler  = StandardScaler()
        self.X_train_ = self._scaler.fit_transform(X)
        self.y_train_ = y.copy()
        self.y_mean_  = float(np.mean(y))
        n, d = self.X_train_.shape
        self.h_ = (max(0.5, n ** (-1.0 / (d + 4)))
                   if self.bandwidth is None else float(self.bandwidth))
        return self

    def predict(self, X):
        X_sc = self._scaler.transform(np.asarray(X, dtype=float))
        # ||x - xi||^2 = ||x||^2 + ||xi||^2 - 2 x.xi
        sq_x    = np.sum(X_sc ** 2, axis=1)[:, np.newaxis]
        sq_xt   = np.sum(self.X_train_ ** 2, axis=1)[np.newaxis, :]
        sq_dist = sq_x + sq_xt - 2.0 * (X_sc @ self.X_train_.T)
        np.clip(sq_dist, 0, None, out=sq_dist)
        W       = np.exp(-sq_dist / (2.0 * self.h_ ** 2))
        W_sum   = W.sum(axis=1)
        preds   = np.full(len(X_sc), self.y_mean_)
        valid   = W_sum > self.min_weight_sum
        if valid.any():
            preds[valid] = (W[valid] @ self.y_train_) / W_sum[valid]
        return preds


# =============================================================================
# Cascade configuration
# =============================================================================

@dataclass
class CascadeConfig:
    """
    Thresholds governing when a model is accepted or rejected. All have
    conservative defaults, adjustable via the constructor + calling
    configure_cascade(cfg) to swap the active config globally.
    """
    var_res_T_threshold: float = 0.05
    t_min: float = 1.645
    pval_max: float = 0.10
    allow_ic_zero_cross: bool = True
    ci_zero_ratio_max: float = 0.30
    cv_ci_max: float = 8.0
    raw_corr_min: float = 0.05
    strength_min: float = 0.0
    ci_cross_penalty_max: float = 0.5

    cv_theta_max: Optional[float] = 2.5
    r2_nuisance_min: Optional[float] = None
    require_confirmation: bool = True
    sidak_correction: bool = True

    # Nonlinearity diagnostics -- purely informative, never used in
    # _evaluate. Turning this off (False) skips the computation entirely
    # (zero extra cost).
    compute_nonlinear_diagnostics: bool = True
    dcor_max_n: int = 500

    # control_window_margin: extra controls beyond the tested `lag`
    # itself (desired window = lag + margin). See
    # _compute_control_window() for the full formula.
    control_window_margin: int = 2

    # estimate_selection: when 2+ cascade models are accepted with agreeing
    # signs, which one supplies theta/SE and the residual diagnostics.
    #   'cascade_order' -- the first accepted in cascade order (since
    #       BayesianRidge comes first, a linear model whenever it passes).
    #   'nuisance_loss' -- the one with the smallest out-of-fold nuisance
    #       error product (1 - R2_T) * (1 - R2_O), see _nuisance_loss(). DML's
    #       bias is driven by the product of the two nuisance errors
    #       (Chernozhukov et al., 2018), so this picks, per pair and from the
    #       data alone, the learner that best fits that pair's nuisances --
    #       linear where the relation is linear, flexible where it is not.
    # Consensus (n_models_agree / cascade_confidence) is unaffected.
    estimate_selection: str = 'cascade_order'

    # cascade_decision: which acceptances make the pair significant.
    #   'any' -- one accepting model is enough. A learner whose
    #       nuisances explain nothing (out-of-fold R2 ~ 0) leaves the raw
    #       treatment/outcome as "residuals", so its acceptance is a
    #       marginal-association test; under 'any' it overrides the rejection
    #       of learners that did partial the controls out.
    #   'best_nuisance' -- the verdict is that of the model with the smallest
    #       out-of-fold nuisance error product (_nuisance_loss): the pair is
    #       significant only if THAT model accepts it. The other acceptances
    #       only add to the consensus (n_models_agree).
    #   'majority' -- at least half of the models with a computable estimate
    #       must accept.
    cascade_decision: str = 'any'


DEFAULT_CONFIG = CascadeConfig()
_ACTIVE_CFG: CascadeConfig = DEFAULT_CONFIG


def configure_cascade(cfg: CascadeConfig) -> None:
    """Sets the active CascadeConfig globally (default: DEFAULT_CONFIG)."""
    global _ACTIVE_CFG
    _ACTIVE_CFG = cfg


# =============================================================================
# Empty result (pair without enough data)
# =============================================================================

def _empty_result(**extra_diag):
    """
    extra_diag (optional): extra diagnostic fields to merge into the
    empty result -- e.g. _empty_result(T_v=16, control_window_used=24,
    max_lag_original=24) so that even discarded pairs can be audited for
    whether the cause was insufficient T_v and which window was tried.
    """
    base = {
        'effect':             np.nan,
        'se':                 np.nan,
        'ci_lower':           np.nan,
        'ci_upper':           np.nan,
        'pvalue':             np.nan,
        'direction':          'indeterminate',
        'significant':        False,
        'dml_model_used':    'none',
        'cascade_log':        [],
        '_cascade_accepted':  False,
        'cascade_confidence': 'low',
        'n_models_agree':     0,
        'models_accepted':    [],
        # keeps the same keys present in valid results, so that any
        # DataFrame built from a mix of empty/valid results has
        # consistent columns.
        'dcor_residuals':        None,
        'corr_linear_residuals': None,
        'reset_fstat':          None,
        'reset_pvalue':         None,
        'nonlinear_suspected':   False,
        'spearman_residuals':     None,
        'spearman_pvalue':       None,
        'hsic_residuals':         None,
        'chow_fstat':            None,
        'chow_pvalue':           None,
        'chow_breakpoint':       None,
        'permutation_pvalue':     None,
        'theta_permutation_std':  None,
        'ci_lower_boot':         None,
        'ci_upper_boot':         None,
        'theta_boot_mean':       None,
        'ess_treatment':         None,
        'ess_outcome':           None,
        'theta_forward_lite':    None,
        'theta_reverse_lite':    None,
        'causal_asymmetry':     None,
        'ccm_skill':             None,
        'ccm_convergent':       None,
        'ccm_skill_by_L':       None,
        'control_window_used': None,
        'max_lag_original':       None,
        'T_v':                    None,
    }
    base.update(extra_diag)
    return base


# =============================================================================
# Proximity score ("closest" fallback)
# =============================================================================

def _proximity_score(result: dict, var_res_T: float, cfg: CascadeConfig) -> float:
    theta = result.get('effect', np.nan)
    se    = result.get('se',     np.nan)
    pval  = result.get('pvalue', 1.0)

    if np.isnan(theta) or np.isnan(se) or se < 1e-12:
        return -1.0

    t_stat = abs(theta) / se
    ci_lo  = result.get('ci_lower', theta - 1.96 * se)
    ci_hi  = result.get('ci_upper', theta + 1.96 * se)
    cv_ci  = (ci_hi - ci_lo) / max(abs(theta), 1e-9)

    s_var  = min(var_res_T / max(cfg.var_res_T_threshold * 4, 1e-9), 1.0)
    s_t    = min(t_stat    / max(cfg.t_min * 2,               1e-9), 1.0)
    s_pval = 1.0 - min(pval, 1.0)
    s_cv   = 1.0 - min(cv_ci / max(cfg.cv_ci_max * 2, 1e-9),  1.0)

    base_score = (s_var + s_t + s_pval + s_cv) / 4.0

    ci_cross_penalty = 0.0
    if ci_lo * ci_hi < 0 and cfg.ci_cross_penalty_max > 0.0:
        neg_side = abs(min(ci_lo, 0.0))
        pos_side = abs(max(ci_hi, 0.0))
        total = neg_side + pos_side
        crossing_depth = (2.0 * min(neg_side, pos_side)) / max(total, 1e-9)
        ci_cross_penalty = cfg.ci_cross_penalty_max * crossing_depth

    return max(0.0, base_score - ci_cross_penalty)


# =============================================================================
# Nonlinearity diagnostics -- INFORMATIVE ONLY
# =============================================================================
# The two functions below measure whether the relationship between res_T
# and res_O (the residuals that produce theta via linear regression) has
# NONLINEAR structure that the linear theta is missing. Neither takes
# part in _evaluate: they are computed, logged in cascade_log per model,
# and attached to the winning model's final_result, for auditing/later
# use.

def _distance_correlation(x: np.ndarray, y: np.ndarray, max_n: int = 500) -> float:
    """
    Distance correlation (Szekely, Rizzo & Bakirov, 2007) between two
    1-D vectors. Unlike Pearson/Spearman, dCor(x,y) == 0 ONLY when x and
    y are genuinely independent -- it captures any form of dependence,
    including non-monotonic relationships such as y = x^2 with x centered
    (where Pearson and Spearman are both close to zero despite total
    deterministic dependence).

    Returns a value in [0, 1]. NaN if n < 10 (insufficient data).

    O(n^2) cost in time/memory -- to keep it predictable on long panels,
    subsamples (without replacement, fixed seed) down to `max_n` points
    before computing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 10:
        return np.nan

    if n > max_n:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=max_n, replace=False)
        x = x[idx]
        y = y[idx]
        n = max_n

    def _dcov_sq(a: np.ndarray, b: np.ndarray) -> float:
        A = np.abs(a[:, None] - a[None, :])
        B = np.abs(b[:, None] - b[None, :])
        A = A - A.mean(axis=0, keepdims=True) - A.mean(axis=1, keepdims=True) + A.mean()
        B = B - B.mean(axis=0, keepdims=True) - B.mean(axis=1, keepdims=True) + B.mean()
        return float((A * B).mean())

    try:
        dxy2 = _dcov_sq(x, y)
        dxx2 = _dcov_sq(x, x)
        dyy2 = _dcov_sq(y, y)
    except Exception:
        return np.nan

    denom = np.sqrt(max(dxx2, 0.0) * max(dyy2, 0.0))
    if denom < 1e-12:
        return 0.0

    dcor_sq = max(dxy2, 0.0) / denom
    return float(np.sqrt(min(max(dcor_sq, 0.0), 1.0)))


def _quadratic_reset_test(res_T: np.ndarray, res_O: np.ndarray) -> Tuple[float, float]:
    """
    A RESET-type test (Ramsey, 1969), restricted to the quadratic term:
    compares the fit of res_O ~ res_T (the SAME linear model that
    produces `theta` in the cascade) against res_O ~ res_T + res_T^2 via
    the classic F-test for comparing nested models.

    A low p-value (< 0.05, by convention) is evidence that the linear
    model is misspecified for this relationship -- i.e. there is
    systematic curvature in the residuals that theta (linear) is not
    capturing.

    Returns (f_stat, pvalue). (nan, nan) if there are not enough degrees
    of freedom (n < 15) or the regression is degenerate.
    """
    res_T = np.asarray(res_T, dtype=float)
    res_O = np.asarray(res_O, dtype=float)
    n = len(res_T)
    if n < 15:
        return np.nan, np.nan

    ones = np.ones(n)
    X1 = np.column_stack([ones, res_T])
    X2 = np.column_stack([ones, res_T, res_T ** 2])

    try:
        b1, *_ = np.linalg.lstsq(X1, res_O, rcond=None)
        b2, *_ = np.linalg.lstsq(X2, res_O, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    rss1 = float(np.sum((res_O - X1 @ b1) ** 2))
    rss2 = float(np.sum((res_O - X2 @ b2) ** 2))

    dof2 = n - 3
    if dof2 <= 0 or rss2 < 1e-12:
        return np.nan, np.nan

    f_stat = ((rss1 - rss2) / 1.0) / (rss2 / dof2)
    if np.isnan(f_stat) or np.isinf(f_stat) or f_stat < 0:
        return np.nan, np.nan

    pvalue = float(1.0 - stats.f.cdf(f_stat, 1, dof2))
    return float(f_stat), pvalue


def _nonlinearity_diagnostics(res_T: np.ndarray, res_O: np.ndarray,
                               cfg: CascadeConfig) -> dict:
    """
    Aggregates the three nonlinearity diagnostics into a single dict,
    already in the format ready to enter cascade_log / final_result.
    Returns all fields as None if cfg.compute_nonlinear_diagnostics=False
    (zero extra cost in that case).
    """
    if not cfg.compute_nonlinear_diagnostics:
        empty_result_dict = {
            'dcor_residuals':        None,
            'corr_linear_residuals': None,
            'reset_fstat':          None,
            'reset_pvalue':         None,
            'nonlinear_suspected':   False,
        }
        # Even with compute_nonlinear_diagnostics=False, the extra
        # residual diagnostics have their OWN on/off switch --
        # independent of the RESET/dCor test -- so we still try to
        # compute them here.
        empty_result_dict.update(_extra_residual_diagnostics_v83(res_T, res_O))
        return empty_result_dict

    dcor_res = _distance_correlation(res_T, res_O, max_n=cfg.dcor_max_n)
    reset_f, reset_p = _quadratic_reset_test(res_T, res_O)

    if len(res_T) >= 10:
        try:
            corr_res_lin, _ = stats.pearsonr(res_T, res_O)
        except Exception:
            corr_res_lin = np.nan
    else:
        corr_res_lin = np.nan

    nonlinear_suspected = bool(
        not np.isnan(dcor_res)
        and not np.isnan(corr_res_lin)
        and not np.isnan(reset_p)
        and dcor_res > (abs(corr_res_lin) + 0.15)
        and reset_p < 0.05
    )

    result_dict = {
        'dcor_residuals':        round(dcor_res, 4) if not np.isnan(dcor_res) else None,
        'corr_linear_residuals': round(corr_res_lin, 4) if not np.isnan(corr_res_lin) else None,
        'reset_fstat':          round(reset_f, 4) if not np.isnan(reset_f) else None,
        'reset_pvalue':         round(reset_p, 5) if not np.isnan(reset_p) else None,
        'nonlinear_suspected':   nonlinear_suspected,
    }

    # Extra residual diagnostics (spearman/hsic/chow/permutation/
    # bootstrap). Same defensive treatment: never brings down the
    # computation of the diagnostics above, even if something here fails.
    result_dict.update(_extra_residual_diagnostics_v83(res_T, res_O))
    return result_dict


def _extra_residual_diagnostics_v83(res_T: np.ndarray, res_O: np.ndarray) -> dict:
    """
    Defensive bridge to diagnostics_aux.extra_residual_diagnostics().
    Uses the module-level global _ACTIVE_DIAG_EXTRA_CFG (adjustable via
    configure_extra_diagnostics()) -- runs automatically with the
    defaults if nobody configures anything. Returns an empty dict (keys
    absent) if diagnostics_aux.py cannot be imported or any exception
    occurs during the computation -- never propagates an error to the
    main cascade.
    """
    if not HAS_DIAG_EXTRA:
        return {}
    try:
        return extra_residual_diagnostics(res_T, res_O, cfg=_ACTIVE_DIAG_EXTRA_CFG)
    except Exception:
        return {}


def _extra_raw_diagnostics_v83(T_arr: np.ndarray, O_arr: np.ndarray, W_arr: np.ndarray) -> dict:
    """
    Defensive bridge to diagnostics_aux.extra_raw_diagnostics() (ESS,
    lite reverse causality via Ridge, CCM). Uses the module-level global
    _ACTIVE_DIAG_EXTRA_CFG -- runs automatically with the defaults if
    nobody configures anything. Returns an empty dict if
    diagnostics_aux.py cannot be imported or any exception occurs --
    never propagates an error to the main cascade, and never echoes back
    into _evaluate/accepted/fails.
    """
    if not HAS_DIAG_EXTRA:
        return {}
    try:
        return extra_raw_diagnostics(T_arr, O_arr, W_arr, cfg=_ACTIVE_DIAG_EXTRA_CFG)
    except Exception:
        return {}


# =============================================================================
# Confidence evaluation for a candidate
# =============================================================================

def _evaluate(result: dict, var_res_T: float, raw_corr: float,
              cfg: CascadeConfig,
              cv_theta: Optional[float] = None,
              r2_T: Optional[float] = None,
              r2_O: Optional[float] = None) -> List[str]:
    theta = result['effect']
    se    = result['se']
    ci_lo = result['ci_lower']
    ci_hi = result['ci_upper']
    pval  = result.get('pvalue', 1.0)

    t_stat = abs(theta) / max(se, 1e-9)
    cv_ci  = (ci_hi - ci_lo) / max(abs(theta), 1e-9)

    fails: List[str] = []

    if var_res_T < cfg.var_res_T_threshold:
        fails.append(
            f"var(res_T)={var_res_T:.4f} < {cfg.var_res_T_threshold}"
            " [nuisance memorized the treatment]"
        )
    if t_stat < cfg.t_min:
        fails.append(f"|t_stat|={t_stat:.3f} < {cfg.t_min} [weak evidence]")
    if pval is not None and not np.isnan(pval) and pval > cfg.pval_max:
        fails.append(f"pvalue={pval:.4f} > {cfg.pval_max} [not significant]")

    if ci_lo * ci_hi < 0:
        if not cfg.allow_ic_zero_cross:
            fails.append(f"CI crosses zero ci=[{ci_lo:.4f}, {ci_hi:.4f}] [ambiguous CI]")
        else:
            neg_side = abs(min(ci_lo, 0.0))
            pos_side = abs(max(ci_hi, 0.0))
            ratio = min(neg_side, pos_side) / max(neg_side, pos_side, 1e-9)
            if ratio > cfg.ci_zero_ratio_max:
                fails.append(
                    f"CI crosses zero ratio={ratio:.3f} > {cfg.ci_zero_ratio_max}"
                    " [symmetric crossing]"
                )

    if cv_ci > cfg.cv_ci_max:
        fails.append(f"cv_ci={cv_ci:.2f} > {cfg.cv_ci_max} [CI too wide]")

    if abs(raw_corr) > cfg.raw_corr_min:
        if np.sign(theta) != np.sign(raw_corr):
            fails.append(
                f"sign(theta)={int(np.sign(theta))} != "
                f"sign(raw_corr)={int(np.sign(raw_corr))} [inconsistent direction]"
            )

    strength = abs(theta) / max(se, 1e-9)
    if cfg.strength_min > 0 and strength < cfg.strength_min:
        fails.append(f"strength={strength:.3f} < {cfg.strength_min} [weak effect]")

    if (cfg.cv_theta_max is not None and cv_theta is not None
            and not np.isnan(cv_theta)):
        if cv_theta > cfg.cv_theta_max:
            fails.append(
                f"cv_theta={cv_theta:.3f} > {cfg.cv_theta_max}"
                " [theta unstable across folds]"
            )

    if cfg.r2_nuisance_min is not None:
        if r2_T is not None and not np.isnan(r2_T) and r2_T < cfg.r2_nuisance_min:
            fails.append(
                f"r2_nuisance_T={r2_T:.3f} < {cfg.r2_nuisance_min}"
                " [weak nuisance for T]"
            )
        if r2_O is not None and not np.isnan(r2_O) and r2_O < cfg.r2_nuisance_min:
            fails.append(
                f"r2_nuisance_O={r2_O:.3f} < {cfg.r2_nuisance_min}"
                " [weak nuisance for O]"
            )

    # NOTE: the nonlinearity diagnostics (dcor_residuals, reset_fstat/
    # pvalue, nonlinear_suspected) are deliberately NOT included here --
    # they are computed outside this function and merely logged/attached
    # to the result.

    return fails


# =============================================================================
# Model instantiation
# =============================================================================

def list_dml_models() -> list:
    """Lists the values accepted by the dml_model parameter (native cascade)."""
    print("Default cascade ('auto'): bayesian_ridge -> random_forest -> "
          "kernel_ridge -> nadaraya_watson")
    print("Each of these models is also accepted on its own via "
          "dml_model=<name>.")
    return [
        'auto', 'bayesian_ridge', 'random_forest', 'kernel_ridge',
        'nadaraya_watson',
    ]


def _make_nuisance_model(model_name: str, n_train: int, nuisance_weights=None):
    n_est = max(50, min(200, n_train * 2))
    msl   = max(3, n_train // 20)

    if model_name == 'bayesian_ridge':
        return BayesianRidge(max_iter=300, tol=1e-3, compute_score=False)

    elif model_name == 'random_forest':
        return RandomForestRegressor(
            n_estimators=n_est, max_features='sqrt',
            min_samples_leaf=msl, random_state=42, n_jobs=1,
        )

    elif model_name == 'kernel_ridge':
        alpha_kr = max(0.01, 1.0 / max(n_train, 1))
        return Pipeline([
            ('scaler', StandardScaler()),
            ('kr', KernelRidge(kernel='rbf', alpha=alpha_kr, gamma=None)),
        ])

    elif model_name == 'nadaraya_watson':
        return NadarayaWatsonRegressor(bandwidth=None, min_weight_sum=1e-6)

    else:
        raise ValueError(
            f"model_name='{model_name}' unknown. "
            f"Valid values: 'bayesian_ridge', 'random_forest', 'kernel_ridge', "
            f"'nadaraya_watson'."
        )


# =============================================================================
# Cross-fitting
# =============================================================================

def _fit_predict_nuisance(model_name: str, W_train, y_train, W_test, n_train: int):
    model = _make_nuisance_model(model_name, n_train)
    try:
        model.fit(W_train, y_train)
        pred = model.predict(W_test)
        if hasattr(pred, 'ndim') and pred.ndim > 1:
            pred = pred.ravel()
        return pred
    except Exception:
        return np.full(len(W_test), float(np.mean(y_train)))


def _run_crossfit(
    model_name: str, T_arr, O_arr, W_arr, n_splits: int
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Returns (res_T, res_O, fold_info).

    fold_info: a list of dicts, one per successfully executed fold,
    containing 'fold', 'n_test', 'r2_T', 'r2_O' (the nuisance's
    out-of-fold R^2) and 'theta_fold' (theta computed from only that
    fold's residuals -- used to measure the effect's stability across
    folds, see cv_theta in _dml_single).
    """
    T_v   = len(T_arr)
    block = max(T_v // n_splits, 1)
    res_T = np.zeros(T_v)
    res_O = np.zeros(T_v)
    fold_info: List[dict] = []

    for fold in range(n_splits):
        t0    = fold * block
        t1    = (fold + 1) * block if fold < n_splits - 1 else T_v
        test  = np.arange(t0, t1)
        train = np.concatenate([np.arange(0, t0), np.arange(t1, T_v)])
        if len(train) < 10 or len(test) < 2:
            continue
        try:
            pred_T = _fit_predict_nuisance(model_name, W_arr[train], T_arr[train], W_arr[test], len(train))
            pred_O = _fit_predict_nuisance(model_name, W_arr[train], O_arr[train], W_arr[test], len(train))

            res_T[test] = T_arr[test] - pred_T
            res_O[test] = O_arr[test] - pred_O

            ss_res_T = float(np.sum((T_arr[test] - pred_T) ** 2))
            ss_tot_T = float(np.sum((T_arr[test] - np.mean(T_arr[test])) ** 2))
            r2_T = 1.0 - ss_res_T / max(ss_tot_T, 1e-12)

            ss_res_O = float(np.sum((O_arr[test] - pred_O) ** 2))
            ss_tot_O = float(np.sum((O_arr[test] - np.mean(O_arr[test])) ** 2))
            r2_O = 1.0 - ss_res_O / max(ss_tot_O, 1e-12)

            denom_fold = float(res_T[test] @ res_T[test])
            theta_fold = (float(res_T[test] @ res_O[test]) / denom_fold
                          if denom_fold > 1e-12 else np.nan)

            fold_info.append({
                'fold': fold, 'n_test': int(len(test)),
                'r2_T': r2_T, 'r2_O': r2_O, 'theta_fold': theta_fold,
            })
        except Exception:
            continue

    return res_T, res_O, fold_info


def _summarize_fold_info(fold_info: List[dict]):
    """
    Aggregates fold_info into (cv_theta, r2_T_mean, r2_O_mean).
    cv_theta = std(thetas_fold) / |mean(thetas_fold)|, computed only when
    there are >=2 folds with a computable theta_fold; otherwise nan (does
    not penalize for missing data, it simply does not filter on this
    criterion).
    """
    thetas = [f['theta_fold'] for f in fold_info
              if f.get('theta_fold') is not None and not np.isnan(f['theta_fold'])]
    r2_Ts  = [f['r2_T'] for f in fold_info if not np.isnan(f.get('r2_T', np.nan))]
    r2_Os  = [f['r2_O'] for f in fold_info if not np.isnan(f.get('r2_O', np.nan))]

    if len(thetas) >= 2:
        mean_theta = np.mean(thetas)
        cv_theta = (float(np.std(thetas)) / max(abs(mean_theta), 1e-9)
                    if abs(mean_theta) > 1e-9 else np.nan)
    else:
        cv_theta = np.nan

    r2_T_mean = float(np.mean(r2_Ts)) if r2_Ts else np.nan
    r2_O_mean = float(np.mean(r2_Os)) if r2_Os else np.nan

    return cv_theta, r2_T_mean, r2_O_mean


# =============================================================================
# Robustness Value (Cinelli & Hazlett, 2020, JRSS-B)
# =============================================================================

def _partial_f(t_stat: float, dof: float) -> float:
    if dof <= 0 or np.isnan(t_stat) or np.isinf(t_stat):
        return np.nan
    return abs(t_stat) / np.sqrt(dof)


def _robustness_value(t_stat: float, dof: float, q: float = 1.0,
                       alpha: Optional[float] = None) -> float:
    f = _partial_f(t_stat, dof)
    if np.isnan(f):
        return np.nan

    if alpha is None:
        fq = q * f
        rv = 0.5 * (np.sqrt(fq**4 + 4 * fq**2) - fq**2)
        return float(rv)

    t_crit = float(stats.t.ppf(1 - alpha / 2, dof))
    if abs(t_stat) <= t_crit:
        return 0.0
    fq = q * f
    f_crit = t_crit / np.sqrt(dof)
    rv = 0.5 * (
        np.sqrt((fq**2 - f_crit**2) ** 2 + 4 * (fq**2 - f_crit**2 * fq**2))
        - (fq**2 - f_crit**2)
    ) if fq**2 > f_crit**2 else 0.0
    return float(rv)


# =============================================================================
# _compute_theta_and_se
# =============================================================================

def _compute_theta_and_se(res_T, res_O, T_v, hac_lags):
    denom = float(res_T @ res_T)
    if denom < 1e-12:
        return None

    theta = float(res_T @ res_O) / denom
    eta   = res_O - theta * res_T

    if hac_lags is None:
        hac_lags = max(1, int(np.floor(4 * (T_v / 100) ** (2 / 9))))

    X_reg  = res_T.reshape(-1, 1)
    XtX    = X_reg.T @ X_reg / T_v
    scores = eta[:, None] * X_reg
    S = scores.T @ scores / T_v
    for h in range(1, hac_lags + 1):
        w = 1 - h / (hac_lags + 1)
        g = scores[h:].T @ scores[:-h] / T_v
        S += w * (g + g.T)

    try:
        V  = np.linalg.inv(XtX) @ S @ np.linalg.inv(XtX) / T_v
        se = float(np.sqrt(abs(V[0, 0])))
    except np.linalg.LinAlgError:
        return None

    ci_lo  = theta - 1.96 * se
    ci_hi  = theta + 1.96 * se
    t_stat = theta / se if se > 1e-12 else np.nan
    pval   = (float(2 * (1 - stats.norm.cdf(abs(t_stat))))
              if not np.isnan(t_stat) else np.nan)

    dof = max(T_v - 1, 1)
    rv_q1   = _robustness_value(t_stat, dof, q=1.0)
    rv_qa05 = _robustness_value(t_stat, dof, q=1.0, alpha=0.05)

    return {
        'effect':      float(theta),
        'se':          se,
        'ci_lower':    float(ci_lo),
        'ci_upper':    float(ci_hi),
        'pvalue':      pval,
        'direction':   'positive' if theta > 0 else 'negative',
        'significant': (not np.isnan(pval)) and pval < 0.05,
        'robustness_value':    round(rv_q1, 4) if not np.isnan(rv_q1) else None,
        'robustness_value_qa': round(rv_qa05, 4) if not np.isnan(rv_qa05) else None,
    }


# =============================================================================
# Control window
# =============================================================================

def _compute_control_window(
    lag: int,
    N: int,
    max_lag_original: int,
    n_splits: int,
    margin: int = 2,
) -> int:
    """
    Decides how many lags (1..window) go into the control matrix W.

    Tries `window = max_lag_original` first. The window is only reduced
    if that would drop T_v below the cascade's sample floor -- and it is
    reduced only down to the minimum needed to avoid starving the sample
    (lag + margin, respecting the sample ceiling).

    Formula:
      1. window = max_lag_original.
      2. resulting T_v = N - lag - window. If T_v >= floor
         (max(20, n_splits*4)), return the window as-is.
      3. Only if T_v < floor, reduce the window to the minimum needed:
         min(lag + margin, sample_ceiling), where
         sample_ceiling = max(1, N - lag - floor).
      4. never smaller than 1.

    Example: N=40, lag=2, max_lag_original=24, n_splits=3, margin=2.
        window=24 -> T_v = 40 - max(2,24) = 16 < floor=20 -> would starve.
        Reduce: sample_ceiling = 40-20 = 20, window = min(4, 20) = 4 ->
        T_v = 40 - max(2,4) = 36 >= 20 -> runs.

    Longer series: N=100, lag=2, max_lag_original=24, n_splits=3.
        window=24 -> T_v = 100 - max(2,24) = 76 >= floor=20 -> does not
        starve, window stays 24, no loss of confounding controls.

    Note on T_v: the value actually computed in _dml_single is not
    N-lag-window -- the `valid` mask requires index >= max(lag, window),
    since the treatment's NaN padding goes up to `lag` and W's (the
    longest lagged column) goes up to `window`. Since window >= lag in
    every case of interest, T_v = N - max(lag, window) simplifies to
    T_v = N - window in practice -- that is the approximation used in
    this function.
    """
    floor = max(20, n_splits * 4)

    # 1) tries exactly the "full" value, with no extra ceiling.
    window = int(max_lag_original)
    T_v = N - max(lag, window)

    # 2) only touches it if this would actually starve the sample.
    if T_v < floor:
        sample_ceiling = max(1, N - floor)
        effective_margin = margin if lag > 0 else max(margin, 1)
        window = min(lag + effective_margin, sample_ceiling)

    return max(1, int(window))


def _nuisance_loss(r2_T: float, r2_O: float) -> float:
    """
    Out-of-fold nuisance error product used by estimate_selection=
    'nuisance_loss': (1 - R2_T) * (1 - R2_O), i.e. the product of the two
    nuisances' MSEs normalized by var(T) and var(O). Lower is better. A
    negative R2 (worse than predicting the mean) gives a factor > 1, so it
    is penalized rather than clipped. Missing R2 -> +inf, so such a model
    is only picked if no accepted model has both R2s.
    """
    if r2_T is None or r2_O is None or np.isnan(r2_T) or np.isnan(r2_O):
        return float('inf')
    return float((1.0 - r2_T) * (1.0 - r2_O))


# =============================================================================
# _dml_single -- model cascade (BR -> RF -> KRR -> NadarayaWatson)
# =============================================================================

def _dml_single(args):
    """
    DML for a single pair (src, tgt) -- executable in parallel.

    GLOBAL NAME OF THIS MODULE: `estimate_all_effects` calls
    `_dml_single(t)` as a global name of this module (not imported from
    anywhere else).
    """
    cfg = _ACTIVE_CFG

    (src_key, tgt_key, lag,
     treatment_raw, outcome, controls,
     max_lag, n_splits, hac_lags,
     dml_model, var_res_T_threshold, estimate_selection) = args[:12]
    # optional 13th element: (T, k) array of CONTEMPORANEOUS controls
    # (values at time t, not lagged) -- used only by the lag-0 estimates of
    # Stage 8b (estimate_lag0_effects); None for every lagged pair.
    contemp_controls = args[12] if len(args) > 12 else None
    # optional 14th element: dict of per-task options (None = defaults):
    #   'W'                : (T, k) control matrix ALREADY lagged and
    #                        NaN-padded (control_set='parents', see
    #                        _parent_controls); replaces the blanket W below.
    #   'cascade_decision' : see CascadeConfig.cascade_decision.
    opts = (args[13] if len(args) > 13 else None) or {}
    W_prebuilt = opts.get('W')
    cascade_decision = opts.get('cascade_decision') or cfg.cascade_decision

    # passed per task (not read from the global cfg) so it survives the
    # joblib worker processes and can differ between model variants.
    if estimate_selection is None:
        estimate_selection = cfg.estimate_selection

    effective_var_thresh = (
        var_res_T_threshold
        if var_res_T_threshold != 0.05
        else cfg.var_res_T_threshold
    )

    treatment = np.roll(treatment_raw, lag).astype(float)
    treatment[:lag] = np.nan

    # `max_lag` is a sweep/calibration parameter from another stage, not
    # the right window size for this pair (which already has a specific
    # `lag` defined) -- see _compute_control_window().
    control_window = _compute_control_window(
        lag=lag,
        N=len(treatment_raw),
        max_lag_original=max_lag,
        n_splits=n_splits,
        margin=getattr(cfg, 'control_window_margin', 2),
    )

    # The source's own lag l == `lag` IS the treatment (np.roll(treatment_raw,
    # lag)), so it must never enter W: with it in W the nuisance for T can
    # copy the treatment column, res_T collapses towards zero ("nuisance
    # memorized the treatment") and whatever residual survives is just
    # the model's copying error, so the cascade's acceptance would depend
    # on how hard that column is to find among the others (i.e. on
    # max_lag). Every other lag of the source stays in W, as in a
    # full-conditional time-series graph (condition on the whole past
    # except the link being tested).
    ctrl_cols = []
    if W_prebuilt is not None:
        ctrl_cols = [W_prebuilt[:, k] for k in range(W_prebuilt.shape[1])]
    else:
        for l in range(1, control_window + 1):
            for arr in [treatment_raw, outcome]:
                if arr is treatment_raw and l == lag:
                    continue
                lag_arr = np.roll(arr, l).astype(float)
                lag_arr[:l] = np.nan
                ctrl_cols.append(lag_arr)
            for k in range(controls.shape[1]):
                lag_arr = np.roll(controls[:, k], l).astype(float)
                lag_arr[:l] = np.nan
                ctrl_cols.append(lag_arr)
    if contemp_controls is not None:
        for k in range(contemp_controls.shape[1]):
            ctrl_cols.append(np.asarray(contemp_controls[:, k], dtype=float))

    if not ctrl_cols:
        return _empty_result()

    W = np.column_stack(ctrl_cols)
    valid = (~np.isnan(treatment) & ~np.isnan(outcome)
             & ~np.any(np.isnan(W), axis=1))
    T_v = int(valid.sum())

    if T_v < max(20, n_splits * 4):
        return _empty_result(
            T_v=T_v,
            control_window_used=control_window,
            max_lag_original=max_lag,
        )

    T_arr = treatment[valid]
    O_arr = outcome[valid]
    W_arr = W[valid]

    valid_corr = ~(np.isnan(treatment) | np.isnan(outcome))
    if valid_corr.sum() >= 10:
        try:
            raw_corr, _ = stats.pearsonr(treatment[valid_corr], outcome[valid_corr])
        except Exception:
            raw_corr = 0.0
    else:
        raw_corr = 0.0

    if dml_model in ('auto', 'auto_v2'):
        cascade = [
            'bayesian_ridge',
            'random_forest',
            'kernel_ridge',
            'nadaraya_watson',
        ]
    else:
        cascade = [dml_model]

    n_models_cascade = len(cascade)

    if cfg.sidak_correction and n_models_cascade > 1:
        pval_max_base = 1.0 - (1.0 - cfg.pval_max) ** (1.0 / n_models_cascade)
    else:
        pval_max_base = cfg.pval_max

    FLEXIBLE_MODELS      = {'kernel_ridge', 'nadaraya_watson'}
    T_MIN_FACTOR         = 1.15
    PVAL_MAX_FACTOR      = 0.5
    CV_CI_MAX_FACTOR     = 0.5
    IC_RATIO_MAX_FACTOR  = 0.7
    CV_THETA_MAX_FACTOR  = 0.6

    local_cfg = CascadeConfig(
        var_res_T_threshold  = effective_var_thresh,
        t_min                = cfg.t_min,
        pval_max             = pval_max_base,
        allow_ic_zero_cross  = cfg.allow_ic_zero_cross,
        ci_zero_ratio_max    = cfg.ci_zero_ratio_max,
        cv_ci_max            = cfg.cv_ci_max,
        raw_corr_min         = cfg.raw_corr_min,
        strength_min         = cfg.strength_min,
        ci_cross_penalty_max = cfg.ci_cross_penalty_max,
        cv_theta_max         = cfg.cv_theta_max,
        r2_nuisance_min      = cfg.r2_nuisance_min,
        compute_nonlinear_diagnostics = cfg.compute_nonlinear_diagnostics,
        dcor_max_n           = cfg.dcor_max_n,
    )

    local_cfg_flexible = CascadeConfig(
        var_res_T_threshold  = local_cfg.var_res_T_threshold,
        t_min                = local_cfg.t_min * T_MIN_FACTOR,
        pval_max             = local_cfg.pval_max * PVAL_MAX_FACTOR,
        allow_ic_zero_cross  = local_cfg.allow_ic_zero_cross,
        ci_zero_ratio_max    = local_cfg.ci_zero_ratio_max * IC_RATIO_MAX_FACTOR,
        cv_ci_max            = local_cfg.cv_ci_max * CV_CI_MAX_FACTOR,
        raw_corr_min         = local_cfg.raw_corr_min,
        strength_min         = local_cfg.strength_min,
        ci_cross_penalty_max = local_cfg.ci_cross_penalty_max,
        cv_theta_max         = (None if local_cfg.cv_theta_max is None
                                 else local_cfg.cv_theta_max * CV_THETA_MAX_FACTOR),
        r2_nuisance_min      = local_cfg.r2_nuisance_min,
        compute_nonlinear_diagnostics = local_cfg.compute_nonlinear_diagnostics,
        dcor_max_n           = local_cfg.dcor_max_n,
    )

    cascade_log: List[dict] = []
    accepted_list: List[Tuple[str, dict]] = []
    nuisance_loss_by_model: Dict[str, float] = {}
    # every model with a computable theta (accepted or not), for
    # cascade_decision != 'any'.
    evaluated: Dict[str, dict] = {}
    nuisance_loss_all: Dict[str, float] = {}
    best_fallback_score:  float          = -1.0
    best_fallback_result: Optional[dict] = None
    best_fallback_model:  str            = cascade[-1]

    # holds the nonlinearity diagnostics computed for each model in the
    # cascade, to later attach to the winning model's final_result
    # (accepted_list[0], fallback, or sign conflict).
    diag_by_model: Dict[str, dict] = {}

    for model_name in cascade:
        res_T, res_O, fold_info = _run_crossfit(model_name, T_arr, O_arr, W_arr, n_splits)
        result = _compute_theta_and_se(res_T, res_O, T_v, hac_lags)

        if result is None:
            # even without a computable theta/SE, res_T/res_O exist --
            # compute the nonlinearity diagnostics here too (and store
            # them in diag_by_model) to keep the SAME columns across all
            # cascade_log entries, and in case (rare) this model still
            # ends up as best_fallback via another path.
            diag_nl_empty = _nonlinearity_diagnostics(res_T, res_O, cfg)
            diag_by_model[model_name] = diag_nl_empty
            log_entry = {
                'model':     model_name,
                'accepted':  False,
                'reason':    'theta/SE not computable (null denominator)',
                'var_res_T': 0.0,
                'score':     -1.0,
                'dcor_residuals':        diag_nl_empty['dcor_residuals'],
                'corr_linear_residuals': diag_nl_empty['corr_linear_residuals'],
                'reset_fstat':          diag_nl_empty['reset_fstat'],
                'reset_pvalue':         diag_nl_empty['reset_pvalue'],
                'nonlinear_suspected':   diag_nl_empty['nonlinear_suspected'],
            }
            # includes any extra residual fields (spearman/hsic/chow/
            # permutation/bootstrap) that _nonlinearity_diagnostics
            # computed, without overwriting the keys already defined.
            for k, v in diag_nl_empty.items():
                log_entry.setdefault(k, v)
            cascade_log.append(log_entry)
            continue

        model_cfg = (
            local_cfg_flexible if model_name in FLEXIBLE_MODELS else local_cfg
        )

        var_res = float(np.var(res_T))
        cv_theta, r2_T_mean, r2_O_mean = _summarize_fold_info(fold_info)

        # nonlinearity diagnostics on the SAME res_T/res_O used to
        # compute this model's theta. Purely informative -- does not
        # enter fails/score.
        diag_nl = _nonlinearity_diagnostics(res_T, res_O, model_cfg)
        diag_by_model[model_name] = diag_nl

        fails = _evaluate(result, var_res, raw_corr, model_cfg,
                           cv_theta=cv_theta, r2_T=r2_T_mean, r2_O=r2_O_mean)
        score = _proximity_score(result, var_res, model_cfg)

        ci_lo = result['ci_lower']
        ci_hi = result['ci_upper']
        if ci_lo * ci_hi < 0:
            neg_side = abs(min(ci_lo, 0.0))
            pos_side = abs(max(ci_hi, 0.0))
            total = neg_side + pos_side
            ci_cross_ratio   = (2.0 * min(neg_side, pos_side)) / max(total, 1e-9)
            ci_cross_penalty = round(model_cfg.ci_cross_penalty_max * ci_cross_ratio, 4)
        else:
            ci_cross_ratio   = None
            ci_cross_penalty = 0.0

        if score > best_fallback_score:
            best_fallback_score  = score
            best_fallback_result = result
            best_fallback_model  = model_name

        log_entry = {
            'model':            model_name,
            'accepted':         len(fails) == 0,
            'reasons_fail':     fails,
            'var_res_T':        round(var_res, 5),
            't_stat':           round(abs(result['effect']) / max(result['se'], 1e-9), 4),
            'pvalue':           round(result['pvalue'] if result['pvalue'] is not None
                                      and not np.isnan(result['pvalue']) else 1.0, 5),
            'pval_max_used':   round(model_cfg.pval_max, 5),
            'cv_ci':            round((result['ci_upper'] - result['ci_lower'])
                                      / max(abs(result['effect']), 1e-9), 3),
            'cv_theta':         round(cv_theta, 4) if not np.isnan(cv_theta) else None,
            'r2_nuisance_T':    round(r2_T_mean, 4) if not np.isnan(r2_T_mean) else None,
            'r2_nuisance_O':    round(r2_O_mean, 4) if not np.isnan(r2_O_mean) else None,
            'n_folds_ok':       len(fold_info),
            'ci_cross_ratio':   round(ci_cross_ratio, 4) if ci_cross_ratio is not None else None,
            'ci_cross_penalty': ci_cross_penalty,
            'score':            round(score, 4),
            'dcor_residuals':        diag_nl['dcor_residuals'],
            'corr_linear_residuals': diag_nl['corr_linear_residuals'],
            'reset_fstat':          diag_nl['reset_fstat'],
            'reset_pvalue':         diag_nl['reset_pvalue'],
            'nonlinear_suspected':   diag_nl['nonlinear_suspected'],
        }
        # includes any extra residual fields computed by
        # _nonlinearity_diagnostics, without overwriting the keys already
        # defined above.
        for k, v in diag_nl.items():
            log_entry.setdefault(k, v)
        cascade_log.append(log_entry)

        evaluated[model_name] = result
        nuisance_loss_all[model_name] = _nuisance_loss(r2_T_mean, r2_O_mean)
        if len(fails) == 0:
            accepted_list.append((model_name, result))
            nuisance_loss_by_model[model_name] = _nuisance_loss(r2_T_mean, r2_O_mean)

    # Cascade decision rule (CascadeConfig.cascade_decision). 'any' keeps
    # every acceptance. The other rules can VETO the pair even
    # though some model accepted it; the vetoed pair is reported with the
    # estimate of the model that decided, marked "[vetoed]".
    veto_model: Optional[str] = None
    if accepted_list and cascade_decision != 'any' and evaluated:
        accepted_names = [name for name, _ in accepted_list]
        if cascade_decision == 'best_nuisance':
            best = min(evaluated, key=lambda m: nuisance_loss_all[m])
            if np.isfinite(nuisance_loss_all[best]) and best not in accepted_names:
                veto_model = best
        elif cascade_decision == 'majority':
            if 2 * len(accepted_names) < len(evaluated):
                veto_model = min(
                    (m for m in evaluated if m not in accepted_names),
                    key=lambda m: nuisance_loss_all[m],
                )
        else:
            raise ValueError(
                f"cascade_decision='{cascade_decision}' unknown. "
                f"Valid values: 'any', 'best_nuisance', 'majority'."
            )

    if veto_model is not None:
        final_result     = evaluated[veto_model]
        final_model      = f"{veto_model}[vetoed]"
        cascade_accepted = False
        confidence       = 'low'
        n_agree          = 0
        models_ok        = [name for name, _ in accepted_list]

    elif len(accepted_list) == 0:
        if best_fallback_result is None:
            return _empty_result()
        final_result     = best_fallback_result
        final_model      = f"{best_fallback_model}[fallback]"
        cascade_accepted = False
        confidence       = 'low'
        n_agree          = 0
        models_ok        = []

    else:
        signs = [np.sign(res['effect']) for _, res in accepted_list]
        signs_agree = all(s == signs[0] for s in signs)
        name0, res0 = accepted_list[0]
        models_ok = [name for name, _ in accepted_list]

        if len(accepted_list) >= 2 and signs_agree:
            # which agreeing model supplies theta/SE/diagnostics -- see
            # CascadeConfig.estimate_selection.
            if estimate_selection == 'nuisance_loss':
                name0, res0 = min(
                    accepted_list,
                    key=lambda item: nuisance_loss_by_model[item[0]],
                )
            final_result, final_model = res0, name0
            cascade_accepted = True
            confidence       = 'high'
            n_agree          = len(accepted_list)

        elif len(accepted_list) >= 2 and not signs_agree:
            final_result = res0
            final_model  = f"{name0}[sign_conflict]"
            cascade_accepted = False
            confidence        = 'low'
            n_agree           = 0

        else:
            final_result, final_model = res0, name0
            cascade_accepted = True
            confidence       = 'medium'
            n_agree          = 1

    final_result['significant'] = bool(
        cascade_accepted
        and final_result.get('significant', False)
        and confidence in ('high', 'medium')
    )

    final_result['dml_model_used']    = final_model
    final_result['cascade_log']        = cascade_log
    final_result['_cascade_accepted']  = cascade_accepted
    final_result['cascade_confidence'] = confidence
    final_result['n_models_agree']     = n_agree
    final_result['models_accepted']    = models_ok

    # attaches to final_result the nonlinearity diagnostics of the model
    # actually chosen (stripping suffixes such as "[fallback]" /
    # "[sign_conflict]" to find the key in diag_by_model).
    base_model_name = final_model.split('[')[0]
    diag_final = diag_by_model.get(base_model_name, {
        'dcor_residuals':        None,
        'corr_linear_residuals': None,
        'reset_fstat':          None,
        'reset_pvalue':         None,
        'nonlinear_suspected':   False,
    })
    final_result.update(diag_final)

    # extra RAW diagnostics (ESS, reverse causality, CCM) -- computed
    # ONCE per pair over T_arr/O_arr/W_arr (before the cascade's
    # residualization), not repeated per model, since they do not depend
    # on which nuisance model won.
    final_result.update(_extra_raw_diagnostics_v83(T_arr, O_arr, W_arr))

    # exposes what was actually used to build W, for auditing (compare
    # against max_lag_original: if control_window_used <
    # max_lag_original often, it is a sign that the configured max_lag is
    # larger than the sample allows).
    final_result['control_window_used'] = control_window
    final_result['max_lag_original']       = max_lag

    return final_result


# =============================================================================
# AUTOMATIC NUISANCE TRIAGE
# =============================================================================
#
# Filtering Stage 6's `controls` down to the NUISANCE_SCREENING_K most
# correlated (when there are enough entities) trades statistical power
# for a smaller nuisance-regression problem -- worthwhile when most of
# the trimmed controls carry little real confounding, but potentially
# harmful in a densely, genuinely correlated system (e.g. a real
# biological signaling network, where practically every entity is
# correlated with every other one via the same pathway, and controls
# ranked 3rd/4th/5th by correlation still carry real confounding).
#
# The AVERAGE absolute correlation across all of a dataset's series
# (without looking at ground truth) distinguishes the two regimes: a
# genuinely, densely correlated system sits well above the range seen in
# datasets where correlation drops off quickly after the first few
# entities, which is where screening is safe.
NUISANCE_SCREENING_MIN_ENTITIES = 10
NUISANCE_SCREENING_MAX_AVG_CORR = 0.25
NUISANCE_SCREENING_K = 2


def _detect_nuisance_regime(processed: dict, all_keys: list, verbose: bool = True) -> Optional[int]:
    """
    Decides (without ground truth) whether Stage 6 should screen the
    nuisance controls down to the `NUISANCE_SCREENING_K` most correlated,
    based on two raw properties of the dataset: number of entities and
    the AVERAGE absolute correlation across all series. Returns
    `NUISANCE_SCREENING_K` if both conditions (>= min entities AND < max
    avg_corr) are satisfied, otherwise None (blanket conditioning, no
    screening).
    """
    n = len(all_keys)
    if n < NUISANCE_SCREENING_MIN_ENTITIES:
        if verbose:
            print(f"  [auto nuisance screening] n_entities={n} < "
                  f"{NUISANCE_SCREENING_MIN_ENTITIES} -> no screening (blanket conditioning)")
        return None

    corrs = []
    for i, ki in enumerate(all_keys):
        si = processed[ki]
        for kj in all_keys[i + 1:]:
            sj = processed[kj]
            mask = ~np.isnan(si) & ~np.isnan(sj)
            if mask.sum() < 10:
                continue
            try:
                c = abs(np.corrcoef(si[mask], sj[mask])[0, 1])
            except Exception:
                continue
            if np.isfinite(c):
                corrs.append(c)

    avg_corr = float(np.mean(corrs)) if corrs else 0.0
    should_screen = avg_corr < NUISANCE_SCREENING_MAX_AVG_CORR
    if verbose:
        print(f"  [auto nuisance screening] n_entities={n}, "
              f"avg_absolute_corr={avg_corr:.3f} (floor={NUISANCE_SCREENING_MAX_AVG_CORR}) "
              f"-> {'SCREEN to k=' + str(NUISANCE_SCREENING_K) if should_screen else 'no screening (blanket conditioning)'}")
    return NUISANCE_SCREENING_K if should_screen else None


# =============================================================================
# PARENT-BASED CONTROL SET (control_set='parents')
# =============================================================================
#
# Blanket conditioning puts lags 1..window of EVERY series in W: N x window
# columns (hundreds) against a few hundred rows, so the nuisance learners
# cannot fit (out-of-fold R2 ~ 0 for the flexible ones) and nothing is
# actually partialled out. The parent-based set conditions each pair on the
# candidate parents the screening stages already selected, each at its own
# lag -- the conditioning set of PCMCI's MCI test (Runge et al., Sci. Adv.
# 2019), built here from the Stage 3-5 candidates instead of PC1.

# Observations per control column (the usual rule of thumb for regression;
# the same ratio as the sample check of Pipeline/structural.py's parent
# competition).
PARENT_CONTROLS_ROWS_PER_COLUMN = 10


def _ar_order_bic(x: np.ndarray, max_p: int) -> int:
    """Autoregressive order of `x` (1..max_p) by BIC, on a common sample."""
    x = np.asarray(x, dtype=float)
    max_p = int(max(1, min(max_p, len(x) // 10)))
    cols = []
    for l in range(1, max_p + 1):
        c = np.full(len(x), np.nan)
        c[l:] = x[:-l]
        cols.append(c)
    L = np.column_stack(cols)
    ok = ~np.isnan(x) & ~np.any(np.isnan(L), axis=1)
    n = int(ok.sum())
    if n < max_p + 10:
        return 1
    y = x[ok]
    best_p, best_bic = 1, np.inf
    for p in range(1, max_p + 1):
        A = np.column_stack([np.ones(n), L[ok, :p]])
        rss = float(np.sum((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) ** 2))
        bic = n * np.log(max(rss, 1e-300) / n) + (p + 1) * np.log(n)
        if bic < best_bic:
            best_p, best_bic = p, bic
    return best_p


def _lagged(x: np.ndarray, l: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if 0 < l < len(x):
        out[l:] = x[:-l]
    return out


def _parent_controls(
    src: str, tgt: str, lag: int,
    processed: dict,
    parents_of: dict,
    own_order: dict,
    n_splits: int,
    source_block: bool,
) -> np.ndarray:
    """
    Control matrix W (already lagged, NaN-padded) for the pair
    src(t-lag) -> tgt(t), with time t the outcome's:

      1. the target's own past, lags 1..p (p = its AR order by BIC);
      2. the OTHER candidate parents of the target, each at its own lag;
      3. unless `source_block`: the conditioning set of the treatment, as in
         the MCI test -- the source's own past OLDER than the treatment
         (lags lag+1..lag+p_src; never a more recent lag, which would be a
         mediator of the tested link) and the source's candidate parents,
         shifted by `lag`.

    source_block=True drops (3): no lag of the source enters W, so the
    treatment (the source at its selected lag) stands for the source's whole
    lag block and the test is a Granger-type one -- "does the source's past
    matter for the target given the target's past and its other parents?" --
    instead of "does this one lag matter given the source's other lags?",
    which has no power when the source is smooth (its neighbouring lags
    predict the treatment almost perfectly).

    Columns are capped at one per PARENT_CONTROLS_ROWS_PER_COLUMN training
    rows: own lags first, then the other parents by decreasing |correlation|
    with the outcome or the treatment.
    """
    y = processed[tgt]
    x = processed[src]
    T = len(y)
    treatment = _lagged(x, lag)
    self_pair = (src == tgt)

    fixed, ranked, seen = [], [], set()

    def _add(bucket, key, col):
        if key in seen or np.all(np.isnan(col)):
            return
        seen.add(key)
        bucket.append(col)

    seen.add((src, lag))
    for l in range(1, own_order.get(tgt, 1) + 1):
        _add(fixed, (tgt, l), _lagged(y, l))
    if not source_block and not self_pair:
        for l in range(1, own_order.get(src, 1) + 1):
            _add(fixed, (src, lag + l), _lagged(x, lag + l))

    for p_src, p_lag in parents_of.get(tgt, ()):
        if p_src in (src, tgt):
            continue
        _add(ranked, (p_src, p_lag), _lagged(processed[p_src], p_lag))
    if not source_block and not self_pair:
        for p_src, p_lag in parents_of.get(src, ()):
            if p_src == src:
                continue
            _add(ranked, (p_src, lag + p_lag), _lagged(processed[p_src], lag + p_lag))

    n_train = (T - lag) * (n_splits - 1) // max(n_splits, 1)
    max_cols = max(2, n_train // PARENT_CONTROLS_ROWS_PER_COLUMN)
    fixed = fixed[:max(1, max_cols // 2)]
    room = max_cols - len(fixed)
    if len(ranked) > room:
        def _relevance(col):
            best = 0.0
            for ref in (y, treatment):
                m = ~np.isnan(col) & ~np.isnan(ref)
                if m.sum() < 10:
                    continue
                with np.errstate(invalid='ignore', divide='ignore'):
                    c = abs(np.corrcoef(col[m], ref[m])[0, 1])
                if np.isfinite(c):
                    best = max(best, c)
            return best
        ranked = sorted(ranked, key=_relevance, reverse=True)[:max(room, 0)]

    return np.column_stack(fixed + ranked)


def median_outcome_r2(estimates: list) -> float:
    """
    Median, over the estimated pairs, of the best out-of-fold R2 any cascade
    model reached for the OUTCOME nuisance -- how much of the target the
    control set explains before the treatment enters. NaN if no pair has one.
    """
    best = []
    for est in estimates:
        r2 = [m.get('r2_nuisance_O') for m in (est.get('cascade_log') or [])]
        r2 = [v for v in r2 if v is not None and np.isfinite(v)]
        if r2:
            best.append(max(r2))
    return float(np.median(best)) if best else float('nan')


def estimate_effects_by_control_set(
    candidates: list,
    processed: dict,
    all_keys: list,
    control_set: str = 'blanket',
    source_block: bool = False,
    auto_max_outcome_r2: float = 0.5,
    **kwargs,
) -> tuple:
    """
    Stage 6 entry point: estimate_all_effects under the requested control
    set, resolving control_set='auto' from the data.

    'auto' runs the lean design first (candidate parents + source block). A
    conditional test attributes to the source the outcome variation its
    controls leave unexplained; when the controls already explain most of it
    out-of-fold (near-deterministic dynamics: every series is a function of
    the others) there is nothing left to attribute and the lean test loses
    the true parents. The dataset is then re-estimated with the blanket
    design. Decided from the nuisance fits alone.

    Returns (estimates, lean_design, near_deterministic).
    """
    if control_set != 'auto':
        estimates = estimate_all_effects(
            candidates, processed, all_keys,
            control_set=control_set, source_block=source_block, **kwargs)
        return estimates, (control_set == 'parents' and source_block), False

    estimates = estimate_all_effects(
        candidates, processed, all_keys,
        control_set='parents', source_block=True, **kwargs)
    r2_outcome = median_outcome_r2(estimates)
    lean = bool(np.isfinite(r2_outcome) and r2_outcome < auto_max_outcome_r2)
    print(f"  [DML controls] auto: median out-of-fold outcome R2 under the "
          f"parent-based design = {r2_outcome:.2f} "
          f"({'<' if lean else '>='} {auto_max_outcome_r2}) -> "
          f"{'parent-based + source block' if lean else 'blanket'}")
    near_deterministic = bool(np.isfinite(r2_outcome)) and not lean
    if not lean:
        estimates = estimate_all_effects(
            candidates, processed, all_keys, control_set='blanket', **kwargs)
    return estimates, lean, near_deterministic


def estimate_all_effects(
    candidates: list,
    processed: dict,
    all_keys: list,
    max_lag: int = 8,
    n_splits: int = 5,
    n_jobs: int = 1,
    dml_model: str = 'auto',
    var_res_T_threshold: float = 0.05,
    nuisance_max_controls: Optional[int] = None,
    auto_nuisance_screening: bool = False,
    estimate_selection: Optional[str] = None,
    control_set: str = 'blanket',
    source_block: bool = False,
    cascade_decision: Optional[str] = None,
) -> list:
    """
    Runs DML for each candidate pair. Controls = every series except the
    pair being analyzed. Calls `_dml_single` as a global name of this
    module.

    nuisance_max_controls : EXPERIMENTAL. None (default):
        `controls` = ALL other series ("blanket
        conditioning"). When set (an integer), reduces `controls` to the
        `nuisance_max_controls` most correlated series (raw |corr|, max
        between treatment and outcome) BEFORE building the matrix -- the
        same estimator (BayesianRidge/cascade) as always, just with fewer
        irrelevant nuisance columns. The selection is done by a simple,
        cheap filter (correlation), outside the model, so the estimator
        itself does not change. IGNORED if
        `auto_nuisance_screening=True` (the automatic decision prevails).
    auto_nuisance_screening : if True, calls `_detect_nuisance_regime`
        (number of entities + the dataset's average absolute correlation,
        see the section above) to decide `nuisance_max_controls`
        automatically per dataset, instead of using the fixed value
        passed in. Same spirit as `auto_detect_topology` in
        Pipeline/structural.py.
    estimate_selection : see CascadeConfig.estimate_selection. None falls
        back to the active CascadeConfig's value.
    control_set : 'blanket' (lags 1..window of every series) or
        'parents' (the candidate parents of the pair, each at its own lag --
        see _parent_controls). With 'parents', nuisance_max_controls and
        auto_nuisance_screening do not apply.
    source_block : only with control_set='parents'. True keeps every lag of
        the source out of W, so the source is tested as a block of lags
        (see _parent_controls).
    cascade_decision : see CascadeConfig.cascade_decision. None falls back
        to the active CascadeConfig's value.
    """
    if control_set not in ('blanket', 'parents'):
        raise ValueError(
            f"control_set='{control_set}' unknown. Valid values: 'blanket', 'parents'."
        )
    opts = {'cascade_decision': cascade_decision}

    if control_set == 'parents':
        parents_of: Dict[str, list] = {}
        for cand in candidates:
            parents_of.setdefault(cand['target'], []).append((cand['source'], cand['lag']))
        involved = ({c['source'] for c in candidates} | {c['target'] for c in candidates})
        own_order = {k: _ar_order_bic(processed[k], max_lag) for k in involved}
        n_cols = []
        tasks = []
        for cand in candidates:
            src, tgt, lag = cand['source'], cand['target'], cand['lag']
            W = _parent_controls(src, tgt, lag, processed, parents_of, own_order,
                                 n_splits, source_block)
            n_cols.append(W.shape[1])
            tasks.append((
                src, tgt, lag,
                processed[src].copy(), processed[tgt].copy(), np.zeros((len(processed[tgt]), 0)),
                max_lag, n_splits, None, dml_model, var_res_T_threshold,
                estimate_selection, None, {**opts, 'W': W},
            ))
        print(f"  [DML controls] parent-based{' + source block' if source_block else ''}: "
              f"median {int(np.median(n_cols))} control columns per pair "
              f"(max {max(n_cols)}); blanket would use "
              f"{len(all_keys) * max_lag}")
        return _run_dml_tasks(tasks, n_jobs, dml_model)

    if auto_nuisance_screening:
        nuisance_max_controls = _detect_nuisance_regime(processed, all_keys)

    all_matrix = np.column_stack([processed[k] for k in all_keys])

    tasks = []
    for cand in candidates:
        src, tgt, lag = cand['source'], cand['target'], cand['lag']
        ctrl_keys = [k for k in all_keys if k not in [src, tgt]]
        if nuisance_max_controls is not None and len(ctrl_keys) > nuisance_max_controls:
            t_raw, o_raw = processed[src], processed[tgt]
            scored = []
            for k in ctrl_keys:
                ck = processed[k]
                mask = ~np.isnan(t_raw) & ~np.isnan(o_raw) & ~np.isnan(ck)
                if mask.sum() < 10:
                    scored.append((k, 0.0))
                    continue
                try:
                    c_t = abs(np.corrcoef(ck[mask], t_raw[mask])[0, 1])
                except Exception:
                    c_t = 0.0
                try:
                    c_o = abs(np.corrcoef(ck[mask], o_raw[mask])[0, 1])
                except Exception:
                    c_o = 0.0
                score = max(
                    c_t if np.isfinite(c_t) else 0.0,
                    c_o if np.isfinite(c_o) else 0.0,
                )
                scored.append((k, score))
            scored.sort(key=lambda x: -x[1])
            ctrl_keys = [k for k, _ in scored[:nuisance_max_controls]]
        controls  = (np.column_stack([processed[k] for k in ctrl_keys])
                     if ctrl_keys else np.zeros((all_matrix.shape[0], 1)))
        tasks.append((
            src, tgt, lag,
            processed[src].copy(), processed[tgt].copy(), controls,
            max_lag, n_splits, None, dml_model, var_res_T_threshold,
            estimate_selection, None, opts,
        ))

    return _run_dml_tasks(tasks, n_jobs, dml_model)


def _run_dml_tasks(tasks: list, n_jobs: int, dml_model: str) -> list:
    if HAS_JOBLIB and n_jobs != 1:
        estimates = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_dml_single)(t) for t in tasks)
    else:
        estimates = []
        for i, t in enumerate(tasks):
            est = _dml_single(t)
            estimates.append(est)
            src_w, src_f = t[0].split('__', 1)
            tgt_w, tgt_f = t[1].split('__', 1)
            flag  = "OK" if est['significant'] else "--"
            model = est.get('dml_model_used', dml_model)
            print(f"  [{i+1:02d}/{len(tasks)}] [{flag}] "
                  f"{src_w}({src_f}) -> {tgt_w}({tgt_f}) | "
                  f"lag={t[2]} | theta={est['effect']:+.3f} | "
                  f"se={est['se']:.3f} | p={est['pvalue']:.3f} | model={model}")

    return estimates


# =============================================================================
# LAG-0 (CONTEMPORANEOUS) EFFECT ESTIMATES -- Stage 8b quantification
# =============================================================================

def estimate_lag0_effects(
    links: list,
    processed: dict,
    all_keys: list,
    max_lag: int,
    n_splits: int = 5,
    n_jobs: int = 1,
    dml_model: str = 'auto',
    var_res_T_threshold: float = 0.05,
    estimate_selection: Optional[str] = None,
) -> list:
    """
    DML estimate of the contemporaneous effect for each Stage 8b link, with
    the SAME engine and acceptance rules as the lagged pairs (_dml_single):

        treatment T = X_source(t)          (lag 0)
        outcome   O = X_target(t)
        controls  W = lags 1..window of source, target and every other series
                      (as in Stage 6, blanket conditioning)
                    + the CONTEMPORANEOUS values X_k(t) of the other nodes
                      adjacent to source or target in the lag-0 skeleton
                      (the conditioning set PCMCI+ uses for lag-0 links).

    `links` is a list of dicts {'source', 'target', 'neighbors'}, keys in
    `processed`. A lag-0 direction is not identified from time order, so
    each direction of an unoriented adjacency is estimated separately and
    must be read as "effect of source on target, ASSUMING source -> target".
    Returns one _dml_single result dict per link, in order.
    """
    tasks = []
    for l in links:
        src, tgt = l['source'], l['target']
        ctrl_keys = [k for k in all_keys if k not in (src, tgt)]
        T = len(processed[src])
        controls = (np.column_stack([processed[k] for k in ctrl_keys])
                    if ctrl_keys else np.zeros((T, 1)))
        nb = [k for k in l.get('neighbors', []) if k not in (src, tgt)]
        contemp = np.column_stack([processed[k] for k in nb]) if nb else None
        tasks.append((
            src, tgt, 0,
            processed[src].copy(), processed[tgt].copy(), controls,
            max_lag, n_splits, None, dml_model, var_res_T_threshold,
            estimate_selection, contemp,
        ))
    if HAS_JOBLIB and n_jobs != 1 and len(tasks) > 1:
        return Parallel(n_jobs=n_jobs)(delayed(_dml_single)(t) for t in tasks)
    return [_dml_single(t) for t in tasks]
