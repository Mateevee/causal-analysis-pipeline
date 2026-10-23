"""
diagnostics_aux.py
===================
Library of ADDITIONAL diagnostics for the DML cascade (dml_engine.py) and
for the evidence-fusion layer (SelectorFunc.py).

PHILOSOPHY (same as the rest of the codebase): everything here is
STRICTLY INFORMATIONAL. No function in this module decides whether to
accept/discard a pair, changes theta/se, enters the cascade's
_evaluate(), or plugs into any SelectorFunc.py expert automatically. The
values computed here only become "evidence" if explicitly plugged in as
a new expert in the fusion layer.

This file performs NO monkey-patching and does not import anything from
the pipeline modules — it is just a library of pure functions, testable
in isolation. dml_engine.py imports it directly, via a defensive import:
if this file is absent, the extra fields come out as None/absent without
breaking anything.

Diagnostics provided, grouped by what kind of nonlinearity/anomaly each
targets:

  - effective_sample_size, block_bootstrap_theta, convergent_cross_mapping:
        chaotic/strongly autocorrelated regimes, where the "nominal"
        sample size overstates how much genuinely independent information
        is available.
  - hsic_residuals: a second, independent kernel-based opinion on
        residual dependence, complementing the dCor/RESET diagnostics
        already computed in dml_engine.py's cascade — useful for
        symmetric curvature where Pearson correlation can be near zero
        despite a deterministic dependence.
  - spearman_correlation_residuals: catches monotonic non-linearity
        (e.g. sigmoid/log-type saturation) at no O(n^2) cost.
  - structural_break_test (Chow / simplified Quandt-Andrews): catches
        discontinuities/kinks, which a quadratic RESET term approximates
        poorly.
  - permutation_test_theta: a specificity/false-positive check for pairs
        with no real causal link.
  - reverse_causality_lite: causal asymmetry via a direction test, to
        help distinguish a direct path from an indirect one in larger
        networks (does not replace a full mediation analysis, which would
        require columns labeled as a candidate mediator — not available
        here).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExtraDiagnosticsConfig:
    """
    Turns each diagnostic on/off and parametrizes it. Defaults are meant
    to run on moderately sized panels (hundreds to a few thousand points
    per pair) without exploding cost. CCM is by far the most expensive —
    it is off by default; turn it on only after a first pass without it,
    and preferably only on small, low-noise dynamical systems.
    """
    # -- residual-based (computed on the cascade's res_T/res_O) --
    enable_spearman:    bool = True
    enable_hsic:        bool = True
    hsic_max_n:         int = 500

    enable_chow:        bool = True
    chow_n_candidates:  int = 9
    chow_trim:          float = 0.15

    enable_permutation:        bool = True
    permutation_n:             int = 200
    permutation_shift_min_frac: float = 0.05

    enable_bootstrap:   bool = True
    bootstrap_n:        int = 300
    bootstrap_block_size: Optional[int] = None  # None = auto (n**(1/3))

    # -- raw (computed on T_arr/O_arr/W_arr, before residualization) --
    enable_ess:         bool = True

    enable_reverse_causality: bool = True
    reverse_ridge_alpha: float = 1.0

    enable_ccm:         bool = False   # expensive — enable with caution
    ccm_E:              int = 3
    ccm_tau:            int = 1
    ccm_max_n:          int = 300
    ccm_n_samples:      int = 8
    ccm_n_library_sizes: int = 5

    seed: int = 42


DEFAULT_EXTRA_CONFIG = ExtraDiagnosticsConfig()


# =============================================================================
# 1) Effective Sample Size (ESS) via autocorrelation
# =============================================================================

def effective_sample_size(x: np.ndarray, max_lag: Optional[int] = None) -> float:
    """
    EFFECTIVE sample size of a series, corrected for autocorrelation
    (Geyer's 1992 "initial positive sequence" rule — the same one used
    for MCMC diagnostics). In chaotic series with strong short-term
    autocorrelation, the
    "nominal" T_v used in the cascade's HAC can be WAY above the number
    of genuinely independent observations — this does not invalidate the
    HAC (which already corrects the VARIANCE), but explains, for
    example, why few cascade models agree (low n_models_agree) without
    that indicating anything spurious: it is genuine estimation
    difficulty, not an absence of signal.

    ess = n / (1 + 2*sum(rho_k)), summing consecutive positive
    autocorrelations (cuts off at the first non-positive one, k>1).

    Returns a float in (0, n]. Never fails (fallback = n if variance
    ~0 or n is small).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return float(n)

    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return float(n)

    if max_lag is None:
        max_lag = min(n // 4, 100)

    acf_sum = 0.0
    for k in range(1, max_lag + 1):
        num = np.sum(x[k:] * x[:-k]) / n
        rho = num / var
        if rho <= 0 and k > 1:
            break
        acf_sum += rho

    ess = n / (1.0 + 2.0 * acf_sum)
    return float(max(1.0, min(ess, n)))


# =============================================================================
# 2) Spearman on the residuals (non-linear monotonic — e.g. saturation)
# =============================================================================

def spearman_correlation_residuals(res_T: np.ndarray, res_O: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Spearman(res_T, res_O), alongside the Pearson already computed in
    the cascade (corr_linear_residuals). Intended difference in usage:

      - a large gap between |dcor_residuals| and |spearman_residuals| ->
        NON-MONOTONIC non-linearity (e.g. quadratic)
      - |spearman_residuals| >> |corr_linear_residuals| but dcor not much
        larger than spearman -> MONOTONIC non-linearity (e.g. saturation)
        — in that case the effect is real and
        monotonic, just poorly approximated by a straight line.

    Returns None if n < 10 or a numerical error occurs.
    """
    res_T = np.asarray(res_T, dtype=float)
    res_O = np.asarray(res_O, dtype=float)
    if len(res_T) < 10:
        return {'spearman_residuals': None, 'spearman_pvalue': None}
    try:
        rho, pval = stats.spearmanr(res_T, res_O)
        if np.isnan(rho):
            return {'spearman_residuals': None, 'spearman_pvalue': None}
        return {
            'spearman_residuals': round(float(rho), 4),
            'spearman_pvalue':    round(float(pval), 5),
        }
    except Exception:
        return {'spearman_residuals': None, 'spearman_pvalue': None}


# =============================================================================
# 3) HSIC (Hilbert-Schmidt Independence Criterion) — second kernel opinion
# =============================================================================

def hsic_residuals(res_T: np.ndarray, res_O: np.ndarray,
                    max_n: int = 500, seed: int = 42) -> Optional[float]:
    """
    Normalized HSIC (Gretton et al., 2005) between res_T and res_O, with
    a Gaussian kernel and bandwidth set by the median heuristic. Returns
    a value approximately in [0, 1] (CKA-style normalization:
    HSIC(x,y) / sqrt(HSIC(x,x)*HSIC(y,y))), comparable in scale to the
    dcor_residuals already computed in the cascade.

    Motivation for having BOTH (dCor and HSIC) instead of just one: they
    are distinct independence statistics (dCor uses Euclidean distances;
    HSIC uses inner products in an RKHS via a kernel) with different
    asymptotic properties. When the two agree (both high with low
    Pearson), the evidence of non-linearity is much more robust than
    relying on a single statistic.

    Cost O(n²) — subsample (without replacement, fixed seed) down to
    max_n points, the same strategy as _distance_correlation in the
    cascade.
    """
    x = np.asarray(res_T, dtype=float)
    y = np.asarray(res_O, dtype=float)
    n = len(x)
    if n < 10:
        return None

    if n > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        x = x[idx]
        y = y[idx]
        n = max_n

    def _gram(v: np.ndarray) -> np.ndarray:
        d = np.abs(v[:, None] - v[None, :])
        nonzero = d[d > 0]
        sigma = np.median(nonzero) if nonzero.size > 0 else 1.0
        sigma = max(sigma, 1e-9)
        return np.exp(-(d ** 2) / (2.0 * sigma ** 2))

    try:
        K = _gram(x)
        L = _gram(y)
        H = np.eye(n) - np.ones((n, n)) / n
        Kc = H @ K @ H
        Lc = H @ L @ H

        hsic_xy = float(np.trace(Kc @ Lc)) / ((n - 1) ** 2)
        hsic_xx = float(np.trace(Kc @ Kc)) / ((n - 1) ** 2)
        hsic_yy = float(np.trace(Lc @ Lc)) / ((n - 1) ** 2)

        denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
        if denom < 1e-12:
            return 0.0
        val = hsic_xy / denom
        return round(float(max(0.0, min(1.0, val))), 4)
    except Exception:
        return None


# =============================================================================
# 4) Structural break test (Chow / simplified Quandt-Andrews)
# =============================================================================

def structural_break_test(res_T: np.ndarray, res_O: np.ndarray,
                           n_candidates: int = 9,
                           trim: float = 0.15) -> Dict[str, Optional[float]]:
    """
    Complements the quadratic RESET test from the cascade, which is weak
    for discontinuities/kinks — an abrupt break is not well approximated
    by a T² term. Here, we test whether splitting the data into TWO
    regimes (at a candidate cut point in res_T) and fitting separate
    lines with distinct intercept and slope reduces the RSS enough
    compared to a single line, via an F-test of nested models (the same
    skeleton as _reset_test_quadratico in the cascade, swapping the
    quadratic term for a regime dummy + interaction).

    Searches for the best cut point on a grid of `n_candidates` points
    between the [trim, 1-trim] percentiles of res_T (Quandt-Andrews) and
    reports the maximum F-stat found.

    IMPORTANT CAVEAT: since the cut point is chosen by the search itself
    (sup-F), the p-value from a plain F(2, n-4) is ANTI-CONSERVATIVE —
    the correct distribution under H0 is that of the supremum of the F
    process over the grid (Andrews, 1993), which requires its own
    tables. Here we report the "naive" p-value only as an ORDER-OF-
    MAGNITUDE / exploratory diagnostic — never use chow_pvalue as a hard
    cutoff criterion without the proper correction.

    Returns (chow_fstat, chow_pvalue, chow_breakpoint) — all None if
    n < 30 or the problem is degenerate.
    """
    res_T = np.asarray(res_T, dtype=float)
    res_O = np.asarray(res_O, dtype=float)
    n = len(res_T)

    empty_result = {'chow_fstat': None, 'chow_pvalue': None, 'chow_breakpoint': None}
    if n < 30:
        return empty_result

    order = np.argsort(res_T)
    T_sorted = res_T[order]
    O_sorted = res_O[order]

    lo = int(n * trim)
    hi = int(n * (1 - trim))
    if hi - lo < 5:
        return empty_result

    candidates = np.unique(np.linspace(lo, hi, n_candidates, dtype=int))
    ones = np.ones(n)
    X0 = np.column_stack([ones, T_sorted])

    try:
        b0, *_ = np.linalg.lstsq(X0, O_sorted, rcond=None)
    except np.linalg.LinAlgError:
        return empty_result
    rss0 = float(np.sum((O_sorted - X0 @ b0) ** 2))

    best_f = -1.0
    best_breakpoint = None
    idx_range = np.arange(n)

    for c in candidates:
        dummy = (idx_range >= c).astype(float)
        X1 = np.column_stack([ones, T_sorted, dummy, T_sorted * dummy])
        try:
            b1, *_ = np.linalg.lstsq(X1, O_sorted, rcond=None)
        except np.linalg.LinAlgError:
            continue
        rss1 = float(np.sum((O_sorted - X1 @ b1) ** 2))
        dof2 = n - 4
        if dof2 <= 0 or rss1 < 1e-12:
            continue
        f_stat = ((rss0 - rss1) / 2.0) / (rss1 / dof2)
        if np.isnan(f_stat) or np.isinf(f_stat):
            continue
        if f_stat > best_f:
            best_f = f_stat
            best_breakpoint = float(T_sorted[min(int(c), n - 1)])

    if best_f < 0 or best_breakpoint is None:
        return empty_result

    dof2 = n - 4
    pvalue = float(1.0 - stats.f.cdf(best_f, 2, dof2))

    return {
        'chow_fstat':      round(float(best_f), 4),
        'chow_pvalue':     round(pvalue, 5),
        'chow_breakpoint': round(best_breakpoint, 5),
    }


# =============================================================================
# 5) Permutation test (circular shift) for theta
# =============================================================================

def permutation_test_theta(res_T: np.ndarray, res_O: np.ndarray,
                            n_perm: int = 200, seed: int = 42,
                            shift_min_frac: float = 0.05) -> Dict[str, Optional[float]]:
    """
    Empirical p-value for theta via circular-shift permutation of res_T
    — this ENTIRELY preserves the autocorrelation structure of res_T (it
    is a rotation, not an i.i.d. shuffle) while breaking the temporal
    alignment with res_O. It is the analogue, for the specificity/FP
    diagnostic, of what the HAC's asymptotic normal assumes by
    approximation — more robust when that approximation is questionable
    (chaotic regimes, short series).

    Each permutation: shifts res_T by a random amount s (uniform over
    [shift_min_frac*n, n - shift_min_frac*n], avoiding shifts that are
    too small and would leave residual overlap with the original
    alignment), recomputes theta_perm = (shifted_res_T @ res_O) /
    (shifted_res_T @ shifted_res_T). p-value = fraction of permutations
    with |theta_perm| >= |theta_observed|.

    Returns None if n < 20 or the denominator is null.
    """
    res_T = np.asarray(res_T, dtype=float)
    res_O = np.asarray(res_O, dtype=float)
    n = len(res_T)

    empty_result = {'permutation_pvalue': None, 'theta_permutation_std': None}
    denom = float(res_T @ res_T)
    if denom < 1e-12 or n < 20:
        return empty_result

    theta_obs = float(res_T @ res_O) / denom

    rng = np.random.RandomState(seed)
    shift_min = max(1, int(n * shift_min_frac))
    if n - shift_min <= shift_min:
        return empty_result

    # Vectorized: rng.randint(...,
    # size=n_perm) consumes the RandomState in the same sequence as
    # n_perm scalar calls. np.roll(res_T, s)[k] == res_T[(k - s) % n],
    # and the sum of squares of a roll is invariant to the shift (same
    # values, just reordered) — so the denominator is the SAME `denom`
    # already validated (>1e-12) by the caller for theta_obs, with no
    # need to recompute it per permutation.
    shifts      = rng.randint(shift_min, n - shift_min, size=n_perm)
    idx_shift   = (np.arange(n)[None, :] - shifts[:, None]) % n
    RT_shift    = res_T[idx_shift]
    thetas_perm = (RT_shift @ res_O) / denom

    pvalue = float(np.mean(np.abs(thetas_perm) >= abs(theta_obs)))
    return {
        'permutation_pvalue':     round(pvalue, 5),
        'theta_permutation_std':  round(float(np.std(thetas_perm)), 6),
    }


# =============================================================================
# 6) Moving-block bootstrap for theta's CI
# =============================================================================

def block_bootstrap_theta(res_T: np.ndarray, res_O: np.ndarray,
                           n_boot: int = 300,
                           block_size: Optional[int] = None,
                           seed: int = 42) -> Dict[str, Optional[float]]:
    """
    CI for theta via block bootstrap (Künsch, 1989) on the SAME
    res_T/res_O used by the cascade's winning model — a second
    uncertainty estimate that does not assume the normal approximation
    used in _compute_theta_and_se (HAC). A large GAP between
    [ci_lower, ci_upper] (HAC, analytical) and [ci_lower_boot,
    ci_upper_boot] (bootstrap) is a sign that the normal approximation
    may be poorly calibrated — typically in chaotic regimes with strong
    short-term dependence.

    Default block_size = n**(1/3) (a common rule of thumb for block
    bootstrap; adjust if the autocorrelation horizon is much longer/
    shorter than that — see effective_sample_size for a complementary
    estimate of the dependence horizon).

    Returns None for all fields if n < 20 or fewer than 20 valid
    replicates.
    """
    res_T = np.asarray(res_T, dtype=float)
    res_O = np.asarray(res_O, dtype=float)
    n = len(res_T)

    empty_result = {'ci_lower_boot': None, 'ci_upper_boot': None, 'theta_boot_mean': None}
    if n < 20:
        return empty_result

    if block_size is None:
        block_size = max(2, int(round(n ** (1.0 / 3.0))))
    block_size = min(block_size, max(2, n // 2))

    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(n / block_size))

    # Vectorized: each replicate draws exactly n_blocks block starts
    # (rng.randint(..., size=(n_boot, n_blocks))). Every block is built at
    # full block_size and the concatenation is truncated to the first `n`
    # indices, so only the LAST block of a replicate may be shorter than
    # block_size (when n is not a multiple of block_size).
    starts_all = rng.randint(0, n - block_size + 1, size=(n_boot, n_blocks))
    offsets    = np.arange(block_size)
    idx_all    = (starts_all[:, :, None] + offsets[None, None, :]).reshape(
        n_boot, n_blocks * block_size)[:, :n]

    RT = res_T[idx_all]
    RO = res_O[idx_all]
    denom = np.sum(RT * RT, axis=1)
    numer = np.sum(RT * RO, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        thetas = np.where(denom > 1e-12, numer / denom, np.nan)

    valid_thetas = thetas[~np.isnan(thetas)]
    if len(valid_thetas) < 20:
        return empty_result

    return {
        'ci_lower_boot':   round(float(np.percentile(valid_thetas, 2.5)), 6),
        'ci_upper_boot':   round(float(np.percentile(valid_thetas, 97.5)), 6),
        'theta_boot_mean': round(float(np.mean(valid_thetas)), 6),
    }


# =============================================================================
# 7) Reverse causality (asymmetry) — light residualization with Ridge
# =============================================================================

def _residualize_ridge_lite(target: np.ndarray, W: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    FAST residualization, independent of the nuisance model chosen by
    the cascade (which can be RF/KRR/Nadaraya-Watson — too expensive to
    run again hundreds of times just for a direction test). Uses simple
    linear Ridge. This is a DELIBERATELY simpler estimate than the
    cascade's — the goal here is not to reproduce theta with precision,
    it is to give a FAST, cheap signal of causal-direction asymmetry,
    not a re-estimate of the reported effect.
    """
    from sklearn.linear_model import Ridge
    try:
        model = Ridge(alpha=alpha)
        model.fit(W, target)
        pred = model.predict(W)
        return target - pred
    except Exception:
        return target - float(np.mean(target))


def reverse_causality_lite(T_arr: np.ndarray, O_arr: np.ndarray, W_arr: np.ndarray,
                            alpha: float = 1.0) -> Dict[str, Optional[float]]:
    """
    Causal-direction asymmetry test: re-estimates a "reverse" theta
    (treating O as if it were the treatment and T as if it were the
    outcome, controlling for the SAME W — i.e. the same lags of T, O and
    controls used as the nuisance) using a FAST linear estimator (Ridge,
    see _residualize_ridge_lite), and compares its magnitude with theta
    in the original direction (also recomputed the fast way, so the
    comparison is fair — this is NOT the cascade's official theta, which
    uses a non-linear nuisance model).

    Rationale: in systems with a reasonably strong genuine T->O causal
    relationship, it is common (though not guaranteed — this is not a
    formal proof, it is a screening heuristic in the spirit of
    Convergent Cross Mapping/bidirectional Granger) that
    |theta_forward| > |theta_reverse|. Weak or inverted asymmetry is a
    reason to suspect confounding from a common trend not fully removed
    by the lag controls (particularly relevant in chains and dense
    graphs, where indirect effects can generate correlation in both
    directions).

    Returns theta_forward_lite, theta_reverse_lite and
    causal_asymmetry = |theta_forward_lite| - |theta_reverse_lite|
    (positive = more evidence in the direction tested; negative = more
    evidence in the opposite direction, a warning sign).
    """
    empty_result = {
        'theta_forward_lite': None,
        'theta_reverse_lite': None,
        'causal_asymmetry': None,
    }
    try:
        res_T_fwd = _residualize_ridge_lite(T_arr, W_arr, alpha=alpha)
        res_O_fwd = _residualize_ridge_lite(O_arr, W_arr, alpha=alpha)
        denom_fwd = float(res_T_fwd @ res_T_fwd)
        theta_fwd = float(res_T_fwd @ res_O_fwd) / denom_fwd if denom_fwd > 1e-12 else np.nan

        res_O_rev = _residualize_ridge_lite(O_arr, W_arr, alpha=alpha)
        res_T_rev = _residualize_ridge_lite(T_arr, W_arr, alpha=alpha)
        denom_rev = float(res_O_rev @ res_O_rev)
        theta_rev = float(res_O_rev @ res_T_rev) / denom_rev if denom_rev > 1e-12 else np.nan

        if np.isnan(theta_fwd) or np.isnan(theta_rev):
            return empty_result

        return {
            'theta_forward_lite': round(theta_fwd, 6),
            'theta_reverse_lite': round(theta_rev, 6),
            'causal_asymmetry':   round(abs(theta_fwd) - abs(theta_rev), 6),
        }
    except Exception:
        return empty_result


# =============================================================================
# 8) Convergent Cross Mapping (Sugihara et al., 2012)
# =============================================================================

def _build_manifold(series: np.ndarray, E: int, tau: int) -> Optional[np.ndarray]:
    """
    State-space reconstruction via delay embedding (Takens). Row i of
    the manifold corresponds to the vector [series[t], series[t-tau], ...,
    series[t-(E-1)*tau]] where t = i + (E-1)*tau in the original series.
    """
    n = len(series)
    n_points = n - (E - 1) * tau
    if n_points <= 0:
        return None
    M = np.empty((n_points, E))
    for i in range(n_points):
        t0 = i + (E - 1) * tau
        for e in range(E):
            M[i, e] = series[t0 - e * tau]
    return M


def convergent_cross_mapping(cause: np.ndarray, effect: np.ndarray,
                              E: int = 3, tau: int = 1,
                              n_library_sizes: int = 5,
                              n_samples: int = 8,
                              max_n: int = 300,
                              seed: int = 42) -> Dict[str, object]:
    """
    CCM (Sugihara et al., 2012, Science) — a causality method DESIGNED
    for deterministic-chaotic dynamical systems, where classical
    statistical separability (DML, Granger)
    can fail because the variables are strongly coupled on the
    attractor, not because there is no causality.

    DIRECTION CONVENTION (often counter-intuitive — pay attention): to
    test "cause → effect", one CROSS-MAPS FROM the EFFECT manifold to
    estimate CAUSE. The logic: if cause influences effect, then
    information about cause gets "recorded" in the dynamics of effect —
    hence the effect manifold can recover cause. Cross-mapping in the
    wrong direction (cause->cause) would test nothing.

    Skill = correlation between the values of `cause` predicted via
    k-nearest neighbors on the `effect` manifold (k=E+1, exponential
    weights by normalized distance) and the actual values of `cause`.
    The SIGNATURE of genuine causality in dynamical systems (distinct
    from spurious correlation) is CONVERGENCE: skill should INCREASE
    with the library size (the subsample used for the neighbors) —
    hence the method's name.

    COST: O(n_points² * n_samples * n_library_sizes) in the worst case
    (L = n_points). Subsamples (fixed seed) down to max_n points BEFORE
    building the manifold, to keep it predictable. Even so, this is the
    most expensive diagnostic in this module — off by default in
    ExtraDiagnosticsConfig; enable it selectively on small dynamical
    systems, not on large panels.

    Returns:
        ccm_skill        : skill at the largest library size tested
        ccm_convergent   : bool — skill at the largest L exceeded that
                            of the smallest L by at least 0.05 (a simple
                            convergence heuristic, not a formal test)
        ccm_skill_by_L   : dict {L: skill} to inspect the whole
                            convergence curve, not just the binary summary
    """
    cause = np.asarray(cause, dtype=float)
    effect = np.asarray(effect, dtype=float)
    n = min(len(cause), len(effect))
    cause, effect = cause[:n], effect[:n]

    empty_result = {'ccm_skill': None, 'ccm_convergent': None, 'ccm_skill_by_L': None}

    # remove NaNs pairwise (keeping alignment)
    valid = ~(np.isnan(cause) | np.isnan(effect))
    cause, effect = cause[valid], effect[valid]
    n = len(cause)

    if n > max_n:
        # central contiguous window instead of random subsample — CCM
        # depends on the local temporal/dynamical structure, random
        # sampling would break the trajectory on the attractor
        start = (n - max_n) // 2
        cause = cause[start:start + max_n]
        effect = effect[start:start + max_n]
        n = max_n

    try:
        M_effect = _build_manifold(effect, E, tau)
    except Exception:
        return empty_result
    if M_effect is None:
        return empty_result

    n_points = M_effect.shape[0]
    offset = (E - 1) * tau
    cause_target = cause[offset:offset + n_points]

    if n_points < 30:
        return empty_result

    L_min = E + 2
    L_max = n_points
    if L_max <= L_min:
        return empty_result
    library_sizes = sorted(set(np.linspace(L_min, L_max, n_library_sizes, dtype=int)))

    rng = np.random.RandomState(seed)
    skill_by_L: Dict[int, float] = {}

    for L in library_sizes:
        if L > n_points or L < E + 1:
            continue
        skills = []
        for _ in range(n_samples):
            lib_idx = rng.choice(n_points, size=L, replace=False)
            M_lib = M_effect[lib_idx]
            lib_cause = cause_target[lib_idx]

            preds = np.empty(n_points)
            k = E + 1
            for i in range(n_points):
                dists = np.sqrt(np.sum((M_lib - M_effect[i]) ** 2, axis=1))
                if len(dists) <= k:
                    preds[i] = np.mean(lib_cause)
                    continue
                nn_idx = np.argpartition(dists, k)[:k]
                d_nn = dists[nn_idx]
                d0 = max(d_nn.min(), 1e-9)
                w = np.exp(-d_nn / d0)
                w_sum = w.sum()
                if w_sum < 1e-12:
                    preds[i] = float(np.mean(lib_cause[nn_idx]))
                else:
                    preds[i] = float(np.sum(w * lib_cause[nn_idx]) / w_sum)

            if np.std(preds) > 1e-9 and np.std(cause_target) > 1e-9:
                try:
                    skill = float(np.corrcoef(preds, cause_target)[0, 1])
                except Exception:
                    skill = np.nan
            else:
                skill = np.nan
            skills.append(skill)

        valid_values = [s for s in skills if not np.isnan(s)]
        skill_by_L[int(L)] = float(np.mean(valid_values)) if valid_values else np.nan

    Ls = sorted(skill_by_L.keys())
    vals = [skill_by_L[L] for L in Ls]
    valid_vals_idx = [i for i, v in enumerate(vals) if not np.isnan(v)]

    if len(valid_vals_idx) < 2:
        return {
            'ccm_skill': None, 'ccm_convergent': None,
            'ccm_skill_by_L': {k: (round(v, 4) if not np.isnan(v) else None)
                                for k, v in skill_by_L.items()},
        }

    first = vals[valid_vals_idx[0]]
    last = vals[valid_vals_idx[-1]]
    convergent = bool(last > first + 0.05)

    return {
        'ccm_skill':      round(float(last), 4),
        'ccm_convergent': convergent,
        'ccm_skill_by_L': {k: (round(v, 4) if not np.isnan(v) else None)
                            for k, v in skill_by_L.items()},
    }


# =============================================================================
# Aggregators — used by dml_engine.py
# =============================================================================

def extra_residual_diagnostics(res_T: np.ndarray, res_O: np.ndarray,
                                cfg: ExtraDiagnosticsConfig = DEFAULT_EXTRA_CONFIG) -> Dict[str, object]:
    """
    Aggregates all diagnostics that operate on res_T/res_O (the same
    residuals the cascade uses to compute theta) — Spearman, HSIC, Chow,
    permutation, bootstrap. Each one runs in its own isolated try/except:
    a failure in one diagnostic does not bring down the others or the
    pipeline.
    """
    output: Dict[str, object] = {}

    if cfg.enable_spearman:
        try:
            output.update(spearman_correlation_residuals(res_T, res_O))
        except Exception:
            output.update({'spearman_residuals': None, 'spearman_pvalue': None})

    if cfg.enable_hsic:
        try:
            output['hsic_residuals'] = hsic_residuals(res_T, res_O, max_n=cfg.hsic_max_n, seed=cfg.seed)
        except Exception:
            output['hsic_residuals'] = None

    if cfg.enable_chow:
        try:
            output.update(structural_break_test(
                res_T, res_O, n_candidates=cfg.chow_n_candidates, trim=cfg.chow_trim))
        except Exception:
            output.update({'chow_fstat': None, 'chow_pvalue': None, 'chow_breakpoint': None})

    if cfg.enable_permutation:
        try:
            output.update(permutation_test_theta(
                res_T, res_O, n_perm=cfg.permutation_n, seed=cfg.seed,
                shift_min_frac=cfg.permutation_shift_min_frac))
        except Exception:
            output.update({'permutation_pvalue': None, 'theta_permutation_std': None})

    if cfg.enable_bootstrap:
        try:
            output.update(block_bootstrap_theta(
                res_T, res_O, n_boot=cfg.bootstrap_n,
                block_size=cfg.bootstrap_block_size, seed=cfg.seed))
        except Exception:
            output.update({'ci_lower_boot': None, 'ci_upper_boot': None, 'theta_boot_mean': None})

    return output


def extra_raw_diagnostics(T_arr: np.ndarray, O_arr: np.ndarray, W_arr: np.ndarray,
                           cfg: ExtraDiagnosticsConfig = DEFAULT_EXTRA_CONFIG) -> Dict[str, object]:
    """
    Aggregates the diagnostics that need the RAW arrays (before the
    cascade's residualization) — ESS, reverse causality, CCM.
    """
    output: Dict[str, object] = {}

    if cfg.enable_ess:
        try:
            output['ess_treatment'] = round(effective_sample_size(T_arr), 2)
            output['ess_outcome']   = round(effective_sample_size(O_arr), 2)
        except Exception:
            output['ess_treatment'] = None
            output['ess_outcome'] = None

    if cfg.enable_reverse_causality:
        try:
            output.update(reverse_causality_lite(T_arr, O_arr, W_arr, alpha=cfg.reverse_ridge_alpha))
        except Exception:
            output.update({'theta_forward_lite': None, 'theta_reverse_lite': None,
                           'causal_asymmetry': None})

    if cfg.enable_ccm:
        try:
            output.update(convergent_cross_mapping(
                T_arr, O_arr, E=cfg.ccm_E, tau=cfg.ccm_tau,
                n_library_sizes=cfg.ccm_n_library_sizes, n_samples=cfg.ccm_n_samples,
                max_n=cfg.ccm_max_n, seed=cfg.seed))
        except Exception:
            output.update({'ccm_skill': None, 'ccm_convergent': None, 'ccm_skill_by_L': None})

    return output
