# -*- coding: utf-8 -*-
"""
structural.py
============================================================================
PHASE 2 of the causal pipeline — Structural Graph Optimization.

Motivation:

    Phase 1 (Stages 1-7, Pipeline/pipeline.py) answers, for EACH edge in isolation:
        "Does this relationship have sufficient statistical evidence to exist?"

    This maximizes Recall, but produces False Positives that are
    statistically indistinguishable from True Positives when evaluated
    pairwise — because, in a good share of the cases, they are NOT
    statistical errors. They are structurally redundant relationships:
        - indirect effects (A->B->C explaining A->C)
        - common causes
        - multiple parents explaining the same target
        - dependencies that disappear after conditioning

    Phase 2 answers, for each edge, GIVEN THE REST OF THE GRAPH:
        "Is this relationship still necessary once all other relationships
         in the system have already been accounted for?"

Design rules (deliberate, not accidental):
    - Phase 2 NEVER discovers a new edge. It can only remove edges that
      Phase 1 has already accepted. (This module is step 8a of the
      structural adjustment; the only step that adds edges is 8b,
      Pipeline/contemporaneous.py, which runs after it and only appends
      lag-0 links for pairs this module left without a significant edge.)
    - Phase 2 NEVER recomputes TE, Bootstrap, or retrains the DML's
      nuisance models.
    - Phase 2 does NOT use Ground Truth, TP/FP, benchmark, or any label
      at all. Every decision comes from: graph + time series + lags +
      coefficients + metrics already produced by Phase 1.
    - Phase 2 is an additive/optional layer. Running without it returns
      exactly Phase 1's output.

Architecture implemented (document sections -> function):
    1. Parent Competition           -> _parent_competition_for_target()
                                        (method='ard', default) or
                                        _parent_competition_for_target_cmi()
                                        (method='cmi', alternative)
    2. Conditional re-estimation     -> _hac_ols()
    3. Structural Graph Pruning      -> implicit in the conditional
                                        competition (see note in
                                        _parent_competition_for_target)
    4. Information criterion         -> _bic_backward_elimination()
    5. Structural bootstrap          -> _stability_selection() (via ARD) or
                                        _cmi_significance() (via CMI + shuffle test)
    6. Global optimization (BIC)     -> _bic_backward_elimination()
    Iterative process                -> optimize_graph_structure(..., method=...)

Parent competition method (`method` parameter, in optimize_graph_structure
and apply_structural_filter):
    'ard' (default of optimize_graph_structure): structural bootstrap with Sparse Bayesian Learning
        (ARDRegression) — assumes an approximately linear relationship
        between each (standardized) parent and the target, conditional on
        the other parents.
    'cmi' (default of apply_structural_filter, the mode run_pipeline
        uses): directly tests I(parent ; target | other parents) via a
        non-parametric k-NN estimator (Kraskov / Frenzel-Pompe), with
        significance obtained via a local permutation test (a shuffle
        test that preserves the dependency structure with the
        conditioning set). Assumes no functional form at all — detects
        redundancy (including non-linear) without depending on
        OLS/ElasticNet/ARD. More computationally expensive.

Typical usage:

    from Pipeline.pipeline import run_pipeline
    from Pipeline.pivot import load_and_pivot
    from Pipeline.preprocessing import preprocess_all
    from Pipeline.structural import optimize_graph_structure

    results_phase1 = run_pipeline(df, ...)

    df_wide, pairs, entities, properties, dates = load_and_pivot(df, ...)
    processed = preprocess_all(df_wide[pairs])

    results_phase2, log = optimize_graph_structure(
        results_phase1, processed, max_lag=10,
    )

Requirements: numpy, pandas, scikit-learn, scipy (already used by the pipeline).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import ARDRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sklearn.neighbors import KDTree
    from scipy.special import digamma
    HAS_CMI_DEPS = True
except ImportError:
    HAS_CMI_DEPS = False


# =============================================================================
# 0. TEMPORAL ALIGNMENT UTILITIES
# =============================================================================

def _shifted(series: np.ndarray, lag: int) -> np.ndarray:
    """source[t - lag] aligned to index t of the original series (NaN at the edges)."""
    if lag <= 0:
        return series.copy()
    out = np.full_like(series, np.nan, dtype=float)
    out[lag:] = series[:-lag]
    return out


def _valid_rows(*arrays) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= ~np.isnan(a)
    return mask


def _standardize(X: np.ndarray) -> tuple:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-10] = 1.0
    return (X - mu) / sd, mu, sd


# =============================================================================
# 1. NEWEY-WEST (HAC) MULTIVARIATE OLS — conditional re-estimation
# =============================================================================

def _hac_ols(X: np.ndarray, y: np.ndarray, hac_lags: Optional[int] = None) -> Optional[dict]:
    """
    Multivariate OLS with heteroskedasticity- and autocorrelation-robust
    (Newey-West) standard errors, without an intercept (series already
    standardized).

    Used to re-estimate the effect of EACH candidate parent while
    simultaneously conditioning on all other parents of the same target —
    the practical equivalent of the "Conditional DML / Partial
    Correlation" recommended in the document, keeping the same HAC spirit
    already used in Phase 1 (_compute_theta_and_se in dml_engine.py).

    Returns a dict with 'theta' (vector), 'se' (vector), 'pvalue' (vector),
    or None if n <= k+2 (too few observations) or pinv() raises a
    LinAlgError (rare -- severe collinearity between candidates does NOT
    trigger this: pinv() degrades silently via a truncated SVD instead of
    failing, producing numerically unstable theta/pvalue). No explicit
    condition-number guard is applied on purpose: it would reject some
    genuinely coupled (but collinear) topologies along with the truly
    non-identifiable ones.
    """
    n, k = X.shape
    if n <= k + 2:
        return None

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError:
        return None

    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    if hac_lags is None:
        hac_lags = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))

    scores = X * resid[:, None]
    S = scores.T @ scores / n
    for h in range(1, hac_lags + 1):
        w = 1 - h / (hac_lags + 1)
        g = scores[h:].T @ scores[:-h] / n
        S += w * (g + g.T)

    # Robust sandwich estimator (Newey-West): V = (X'X)^-1 (n*S) (X'X)^-1
    V = XtX_inv @ (n * S) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    se[se < 1e-12] = np.nan

    t_stat = beta / se
    pval = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        'theta': beta, 'se': se, 'pvalue': pval,
        'resid': resid, 'ss_res': ss_res, 'r2': r2, 'n': n, 'k': k,
    }


def _bic(ss_res: float, n: int, k: int, penalty_scale: float = 1.0) -> float:
    """
    Bayesian Information Criterion for Gaussian linear regression.

    penalty_scale : float
        Multiplier of the complexity-penalty term (k * log(n)). 1.0 =
        classic BIC. Values < 1 make pruning LESS aggressive (0.5 sits
        between BIC and AIC; ~2/log(n) reproduces AIC). In dense graphs
        (many candidate parents per target), log(n) per parameter is too
        severe and drops true parents along with the redundant ones --
        this knob exists to loosen that without turning pruning off
        entirely.
    """
    if n <= 0:
        return np.inf
    sigma2 = max(ss_res / n, 1e-12)
    return n * np.log(sigma2) + penalty_scale * k * np.log(n)


# =============================================================================
# 2. PARENT COMPETITION (with Stability Selection via Sparse Bayesian
#    Learning / ARD)
# =============================================================================

def _stability_selection(X: np.ndarray, y: np.ndarray,
                          n_boot: int = 100,
                          sample_frac: float = 0.7,
                          threshold_lambda: float = 1e4,
                          max_iter: int = 300,
                          random_state: int = 42) -> np.ndarray:
    """
    Structural bootstrap (document item 5): resamples the pair (X, y)
    `n_boot` times, fits a Sparse Bayesian Learning model (Automatic
    Relevance Determination, `ARDRegression`) on each resample, and
    counts the fraction of resamples in which each column of X received
    a non-zero coefficient. Returns the vector of selection frequencies
    (0..1) per column.

    Why ARD instead of Elastic Net:
        ARD estimates an independent precision (1/variance) per
        coefficient under a hierarchical Gaussian prior; coefficients
        whose estimated precision exceeds `threshold_lambda` are zeroed
        out automatically ("relevance determination"). This gives the
        same practical function as Elastic Net (selecting/zeroing out
        uninformative parents) without requiring a manual choice of
        alpha/l1_ratio — sparsity emerges from the evidence in the data
        (Bayesian Occam's razor), which tends to be more stable when the
        candidate parents of the same target are correlated with each
        other (same entity/region), the typical case here.
    """
    n, k = X.shape
    if k == 0:
        return np.zeros(0)
    rng = np.random.RandomState(random_state)
    counts = np.zeros(k)
    n_sub = max(int(n * sample_frac), k + 5, 10)
    n_sub = min(n_sub, n)

    for _ in range(n_boot):
        idx = rng.choice(n, size=n_sub, replace=False)
        Xs, ys = X[idx], y[idx]
        try:
            model = _make_ard(max_iter=max_iter, threshold_lambda=threshold_lambda)
            model.fit(Xs, ys)
            counts += (np.abs(model.coef_) > 1e-8)
        except Exception:
            continue

    return counts / n_boot


def _make_ard(max_iter: int, threshold_lambda: float) -> "ARDRegression":
    """
    Instantiates ARDRegression covering the renaming of the
    number-of-iterations parameter across scikit-learn versions: `n_iter`
    (old versions, <1.3) was renamed to `max_iter` (new versions). Avoids
    breaking the module depending on the version installed in the
    production environment.
    """
    try:
        return ARDRegression(max_iter=max_iter,
                              threshold_lambda=threshold_lambda,
                              fit_intercept=False)
    except TypeError:
        return ARDRegression(n_iter=max_iter,
                              threshold_lambda=threshold_lambda,
                              fit_intercept=False)


def _parent_competition_for_target(target: str,
                                    parent_edges: pd.DataFrame,
                                    processed: dict,
                                    n_boot: int = 100,
                                    stability_threshold: float = 0.6,
                                    hac_lags: Optional[int] = None) -> pd.DataFrame:
    """
    Runs the parent competition for ONE target (document item 1).

    Instead of validating each candidate parent in isolation, all
    candidate parents of `target` compete simultaneously to explain the
    target series. Only the ones that add explanatory power when all
    others are already present in the model survive.

    NOTE on "Structural Graph Pruning" (item 3) and "Exclusive
    Information Criterion" (item 4): when an effect A->C is, in fact,
    mediated by B (A->B->C, with B also a candidate parent of C), the
    simultaneous inclusion of A and B as regressors of C makes A's
    coefficient collapse to near zero (all the covariance between A and
    C is already "explained" by B). This is exactly the transitive-
    reduction and "exclusive information" effect that items 3 and 4 call
    for — no separate graph algorithm is needed once the conditional
    regression problem is solved with all candidate parents
    simultaneously. An explicit path check (for reporting/diagnostics) is
    done separately in `_explain_by_transitive_path()`.

    Returns a copy of `parent_edges` with additional columns:
        stability_score, conditional_theta, conditional_se,
        conditional_pvalue, survives_phase2
    """
    out = parent_edges.copy()
    if out.empty:
        return out

    y = processed[target]
    T = len(y)

    # Builds the candidate-parent matrix, each shifted by its own lag
    cols, valid_edge_idx = [], []
    for i, row in out.iterrows():
        src = row['_src_key']
        lag = int(row['lag'])
        if src not in processed:
            continue
        x_shift = _shifted(processed[src], lag)
        cols.append(x_shift)
        valid_edge_idx.append(i)

    if not cols:
        # FAIL-SAFE: no source had a series available in `processed` --
        # keep the edges (survives_phase2=True) rather than removing them,
        # since a technical inability to evaluate a pair is not evidence
        # against it (removing here would be a silent source of FN).
        out['stability_score'] = np.nan
        out['conditional_theta'] = out['causal_effect']
        out['conditional_se'] = out['standard_error']
        out['conditional_pvalue'] = out['pvalue']
        out['survives_phase2'] = True
        return out

    X_raw = np.column_stack(cols)
    mask = _valid_rows(y, *[X_raw[:, j] for j in range(X_raw.shape[1])])
    X_valid, y_valid = X_raw[mask], y[mask]

    if len(y_valid) < max(20, X_valid.shape[1] + 5) or not HAS_SKLEARN:
        # Insufficient data for competition — keeps everything as in
        # Phase 1 (fail-safe: never removes an edge due to a technical
        # inability to evaluate it).
        out.loc[valid_edge_idx, 'stability_score'] = 1.0
        out['conditional_theta'] = out['causal_effect']
        out['conditional_se'] = out['standard_error']
        out['conditional_pvalue'] = out['pvalue']
        out['survives_phase2'] = True
        return out

    X_std, _, sd_x = _standardize(X_valid)
    y_std, _, sd_y = _standardize(y_valid.reshape(-1, 1))
    y_std = y_std.ravel()

    freq = _stability_selection(X_std, y_std, n_boot=n_boot)

    ols = _hac_ols(X_std, y_std, hac_lags=hac_lags)

    out['stability_score'] = np.nan
    out['conditional_theta'] = np.nan
    out['conditional_se'] = np.nan
    out['conditional_pvalue'] = np.nan
    out['survives_phase2'] = False

    for j, idx in enumerate(valid_edge_idx):
        out.at[idx, 'stability_score'] = round(float(freq[j]), 4)
        if ols is not None:
            # un-standardizes the coefficient back to the original scale
            theta_scaled = ols['theta'][j] * (sd_y[0] / sd_x[j])
            se_scaled = ols['se'][j] * (sd_y[0] / sd_x[j]) if not np.isnan(ols['se'][j]) else np.nan
            out.at[idx, 'conditional_theta'] = round(float(theta_scaled), 4)
            out.at[idx, 'conditional_se'] = round(float(se_scaled), 4) if not np.isnan(se_scaled) else np.nan
            out.at[idx, 'conditional_pvalue'] = round(float(ols['pvalue'][j]), 4)
        survives = freq[j] >= stability_threshold
        out.at[idx, 'survives_phase2'] = bool(survives)

    # FAIL-SAFE: edges whose source was not in `processed` were never
    # evaluated -- keep them rather than defaulting to False, since a
    # technical inability to evaluate a pair is not evidence against it.
    _not_evaluated = [i for i in out.index if i not in set(valid_edge_idx)]
    if _not_evaluated:
        out.loc[_not_evaluated, 'survives_phase2'] = True

    return out


# =============================================================================
# 2b. PARENT COMPETITION VIA CONDITIONAL MUTUAL INFORMATION (CMI)
# =============================================================================
#
# Alternative method to block 2 (ARD). Instead of assuming a functional
# form (linear after standardization, as OLS/ARD do) and testing
# coefficient sparsity, this block directly tests the statistical
# dependence
#
#     I(parent ; target | other_parents)
#
# using a non-parametric k-NN estimator (Kraskov / Frenzel-Pompe), the
# same "nearest neighbors in the joint space" principle used by causal
# discovery tools such as PCMCI (Runge et al.). If the effect of A on C
# is entirely mediated by B (A->B->C, with B among the "other parents"
# used as the conditioning set Z), then A and C become (approximately)
# conditionally independent given B, and the estimated CMI drops toward
# zero and stops being statistically significant — exactly the same role
# that ARD-based competition fulfills, but without assuming linearity.
#
# Cost: k-NN in spaces of dimension d(Z)+1 and the permutation-based
# significance test (which needs to retrain/requery the tree on every
# resample) make this method more expensive than ARD. `method='cmi'` is
# recommended when non-linearity is suspected in the relationships, or as
# a cross-check of the decision made by the 'ard' method.

def _as_2d(a: np.ndarray) -> np.ndarray:
    """Ensures a 1D array becomes a column (n, 1); (n, d) arrays pass through unchanged."""
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _knn_cmi(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None,
             k: int = 5) -> float:
    """
    Estimates I(X;Y) (if z is None) or I(X;Y|Z) via the Kraskov/
    Frenzel-Pompe k-NN estimator, with distances in the max (Chebyshev)
    norm in the joint space — the same estimator used by CMI-based causal
    discovery libraries (e.g., tigramite's CMIknn).

    Idea: for each point i, epsilon_i is the distance (max norm) to the
    k-th nearest neighbor in the full joint space (X,Y,Z). Then, in the
    marginal subspaces (X,Z), (Y,Z) and (Z), we count how many points
    fall strictly within epsilon_i. CMI is obtained from the average of
    these counts via the digamma function.

    Returns a value >= 0 (truncates negative estimates, which can occur
    from estimation noise when the true dependency is ~0).
    """
    if not HAS_CMI_DEPS:
        raise ImportError("scikit-learn and scipy are required for _knn_cmi.")

    x = _as_2d(x)
    y = _as_2d(y)
    n = x.shape[0]
    if n <= k + 2:
        return 0.0

    if z is None or (hasattr(z, 'shape') and z.shape[1] == 0):
        xyz = np.hstack([x, y])
        tree_xyz = KDTree(xyz, metric='chebyshev')
        dist, _ = tree_xyz.query(xyz, k=k + 1)
        eps = dist[:, -1]

        tree_x = KDTree(x, metric='chebyshev')
        tree_y = KDTree(y, metric='chebyshev')
        n_x = tree_x.query_radius(x, r=eps - 1e-12, count_only=True) - 1
        n_y = tree_y.query_radius(y, r=eps - 1e-12, count_only=True) - 1

        mi = digamma(k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
        return float(max(mi, 0.0))

    z = _as_2d(z)
    xyz = np.hstack([x, y, z])
    xz = np.hstack([x, z])
    yz = np.hstack([y, z])

    tree_xyz = KDTree(xyz, metric='chebyshev')
    dist, _ = tree_xyz.query(xyz, k=k + 1)
    eps = dist[:, -1]

    tree_xz = KDTree(xz, metric='chebyshev')
    tree_yz = KDTree(yz, metric='chebyshev')
    tree_z = KDTree(z, metric='chebyshev')
    n_xz = tree_xz.query_radius(xz, r=eps - 1e-12, count_only=True) - 1
    n_yz = tree_yz.query_radius(yz, r=eps - 1e-12, count_only=True) - 1
    n_z = tree_z.query_radius(z, r=eps - 1e-12, count_only=True) - 1

    cmi = digamma(k) - np.mean(digamma(n_xz + 1) + digamma(n_yz + 1) - digamma(n_z + 1))
    return float(max(cmi, 0.0))


def _local_permutation(x: np.ndarray, z: Optional[np.ndarray], rng: np.random.RandomState,
                        n_neighbors_shuffle: int = 5) -> np.ndarray:
    """
    Shuffles `x` "locally": instead of a global permutation (which would
    also destroy the dependency of X on Z, inflating the null CMI in
    high-dimensional conditioning sets), swaps each point only with one
    of its `n_neighbors_shuffle` nearest neighbors in Z's space. This
    preserves P(X|Z) under the null hypothesis that X and Y are
    conditionally independent given Z, following the shuffle test of
    Runge (2018) used in CMIknn/PCMCI.

    If `z` is None (no conditioning set), performs an ordinary global
    permutation.
    """
    n = len(x)
    if z is None or (hasattr(z, 'shape') and z.shape[1] == 0):
        return x[rng.permutation(n)]

    tree_z = KDTree(z, metric='chebyshev')
    k_query = min(n_neighbors_shuffle + 1, n)
    _, neighbor_idx = tree_z.query(z, k=k_query)
    neighbor_idx = neighbor_idx[:, 1:]  # removes the point itself (1st neighbor = self)

    perm = np.arange(n)
    used = np.zeros(n, dtype=bool)
    order = rng.permutation(n)
    for i in order:
        if used[i]:
            continue
        candidates = [j for j in neighbor_idx[i] if not used[j] and j != i]
        if candidates:
            j = candidates[rng.randint(len(candidates))]
            perm[i], perm[j] = perm[j], perm[i]
            used[i] = True
            used[j] = True
    return x[perm]


def _cmi_significance(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray],
                       k: int = 5, n_perm: int = 200,
                       n_neighbors_shuffle: int = 5,
                       random_state: int = 42) -> tuple:
    """
    Tests H0: X and Y are conditionally independent given Z, via a local
    shuffle test (document item 5, CMI version of the structural
    bootstrap): re-estimates the null CMI `n_perm` times by shuffling X
    within Z's neighborhoods (preserves the X-Z relationship, destroys
    only the X-Y relationship beyond what Z explains) and compares it to
    the observed value.

    Returns (cmi_observed, p_value). A small p-value => rejects H0 => the
    X->target edge carries information that Z (the other parents) does
    not explain (edge survives). A large p-value => no evidence of
    dependency beyond what Z already explains (redundant edge, removed).
    """
    rng = np.random.RandomState(random_state)
    cmi_obs = _knn_cmi(x, y, z, k=k)

    null_dist = np.empty(n_perm)
    for p in range(n_perm):
        x_shuf = _local_permutation(_as_2d(x), z, rng, n_neighbors_shuffle=n_neighbors_shuffle)
        null_dist[p] = _knn_cmi(x_shuf, y, z, k=k)

    pval = (np.sum(null_dist >= cmi_obs) + 1) / (n_perm + 1)
    return cmi_obs, float(pval)


def _parent_competition_for_target_cmi(target: str,
                                        parent_edges: pd.DataFrame,
                                        processed: dict,
                                        k: int = 5,
                                        n_perm: int = 200,
                                        alpha: float = 0.05,
                                        n_neighbors_shuffle: int = 5,
                                        max_cond: Optional[int] = None,
                                        linear_rescue_alpha: float = 0.0,
                                        random_state: int = 42,
                                        block: bool = False) -> pd.DataFrame:
    """
    Variant of `_parent_competition_for_target` (document item 1) that
    decides the survival of each candidate parent by directly testing
    I(parent ; target | other_candidate_parents) instead of fitting a
    sparse linear model.

    block=True tests each candidate as a BLOCK of lags and conditions on
    the target's own past:
        X = the source at lags {1, its selected lag, and the one halfway}
            (the selected lag is the argmax of a pairwise scan, not
            necessarily the lag of the mechanism);
        Z = the other candidates (as below) PLUS the target's own past,
            lags 1..p with p its autoregressive order by BIC -- without it
            a source that only shares the target's autocorrelation survives.
    The conditional OLS reported alongside is unchanged.

    For each edge (src -> target, lag), Z is formed by ALL other
    candidate parents of `target` (each shifted by its own lag), exactly
    like the design matrix X used by `_parent_competition_for_target`. If
    there is only one candidate parent (Z ends up empty), there is
    nothing to condition on — the edge is kept by definition (same logic
    as `_bic_backward_elimination` when len(active)<=1).

    Returns a copy of `parent_edges` with additional columns:
        cmi_value, cmi_pvalue, stability_score (=1 - p-value, to be on
        the same 0..1 scale and comparable to the 'ard' method),
        conditional_theta, conditional_se, conditional_pvalue
        (re-estimated via conditional HAC OLS, for reporting only —
        the survival decision uses cmi_pvalue, not conditional_pvalue),
        survives_phase2
    """
    out = parent_edges.copy()
    if out.empty:
        return out

    y = processed[target]

    series_by_edge = {}
    for i, row in out.iterrows():
        src = row['_src_key']
        lag = int(row['lag'])
        if src not in processed:
            continue
        series_by_edge[i] = _shifted(processed[src], lag)

    valid_edge_idx = list(series_by_edge.keys())

    out['cmi_value'] = np.nan
    out['cmi_pvalue'] = np.nan
    out['cmi_n_cond'] = np.nan
    out['stability_score'] = np.nan
    out['conditional_theta'] = np.nan
    out['conditional_se'] = np.nan
    out['conditional_pvalue'] = np.nan
    out['survives_phase2'] = False

    if not valid_edge_idx:
        return out

    if not HAS_CMI_DEPS:
        # Fail-safe: without scikit-learn/scipy for k-NN, keep everything
        # as it was in Phase 1 (never remove due to a technical
        # inability).
        out.loc[valid_edge_idx, 'stability_score'] = 1.0
        out['conditional_theta'] = out['causal_effect']
        out['conditional_se'] = out['standard_error']
        out['conditional_pvalue'] = out['pvalue']
        out['survives_phase2'] = True
        return out

    # A single candidate parent: no conditioning set possible -> keep.
    if len(valid_edge_idx) == 1:
        idx = valid_edge_idx[0]
        out.at[idx, 'stability_score'] = 1.0
        out.at[idx, 'survives_phase2'] = True
        out.at[idx, 'conditional_theta'] = out.at[idx, 'causal_effect']
        out.at[idx, 'conditional_se'] = out.at[idx, 'standard_error']
        out.at[idx, 'conditional_pvalue'] = out.at[idx, 'pvalue']
        return out

    X_all = np.column_stack([series_by_edge[i] for i in valid_edge_idx])
    mask = _valid_rows(y, *[X_all[:, j] for j in range(X_all.shape[1])])

    # block=True: extra lags of each source (its block) and the target's own
    # past. Built here so the valid-row mask covers them.
    block_extra, own_past = {}, None
    if block:
        from .dml_engine import _ar_order_bic
        lags_of = {i: int(out.at[i, 'lag']) for i in valid_edge_idx}
        self_lags = {lags_of[i] for i in valid_edge_idx if out.at[i, '_src_key'] == target}
        p_own = _ar_order_bic(y, max(max(lags_of.values()), 1))
        own_cols = [_shifted(y, l) for l in range(1, p_own + 1) if l not in self_lags]
        for i in valid_edge_idx:
            lag_i = lags_of[i]
            extra = sorted({1, (1 + lag_i) // 2} - {lag_i, 0})
            if out.at[i, '_src_key'] == target:
                extra = []
            block_extra[i] = [_shifted(processed[out.at[i, '_src_key']], l) for l in extra]
        every = own_cols + [c for cols in block_extra.values() for c in cols]
        if every:
            mask &= _valid_rows(*every)
        if own_cols:
            own_past = np.column_stack(own_cols)

    if mask.sum() < max(30, X_all.shape[1] * 10):
        # Insufficient data for a reliable k-NN estimate in high
        # dimension — fail-safe, keep everything (same policy as the ARD
        # method).
        out.loc[valid_edge_idx, 'stability_score'] = 1.0
        out['conditional_theta'] = out['causal_effect']
        out['conditional_se'] = out['standard_error']
        out['conditional_pvalue'] = out['pvalue']
        out['survives_phase2'] = True
        return out

    X_valid = X_all[mask]
    y_valid = y[mask]
    X_std, _, sd_x = _standardize(X_valid)
    y_std, _, sd_y = _standardize(y_valid.reshape(-1, 1))
    y_std = y_std.ravel()

    own_past_std = _standardize(own_past[mask])[0] if own_past is not None else None
    block_extra_std = {
        i: (_standardize(np.column_stack(cols)[mask])[0] if cols else None)
        for i, cols in block_extra.items()
    }

    # Conditional HAC OLS only to fill in theta/se/pvalue for reporting
    # (auditing), on the same scale as the 'ard' method. The survival
    # decision does NOT depend on this.
    ols = _hac_ols(X_std, y_std, hac_lags=None)

    # Relevance of each competitor to the target -- used to choose WHICH
    # ones enter Z when max_cond limits the conditioning set's dimension.
    with np.errstate(invalid='ignore', divide='ignore'):
        _rel = np.array([abs(np.corrcoef(X_std[:, c], y_std)[0, 1])
                         for c in range(X_std.shape[1])])
    _rel = np.nan_to_num(_rel, nan=0.0)

    for j, idx in enumerate(valid_edge_idx):
        x_j = X_std[:, j]
        others = [c for c in range(X_std.shape[1]) if c != j]
        # ── Dimension-limited Z (PC1/PCMCI-style) ─────────────────────
        # Keeps only the `max_cond` most relevant competitors to the
        # target as the conditioning set Z.
        if max_cond is not None and 0 <= max_cond < len(others):
            others = sorted(others, key=lambda c: -_rel[c])[:max_cond]
        z_j = X_std[:, others] if others else None

        if block:
            if own_past_std is not None:
                z_j = own_past_std if z_j is None else np.column_stack([z_j, own_past_std])
            if block_extra_std.get(idx) is not None:
                x_j = np.column_stack([x_j, block_extra_std[idx]])

        cmi_val, pval = _cmi_significance(
            x_j, y_std, z_j, k=k, n_perm=n_perm,
            n_neighbors_shuffle=n_neighbors_shuffle,
            random_state=random_state + j,
        )
        out.at[idx, 'cmi_value'] = round(cmi_val, 5)
        out.at[idx, 'cmi_pvalue'] = round(pval, 5)
        out.at[idx, 'cmi_n_cond'] = len(others)
        out.at[idx, 'stability_score'] = round(1.0 - pval, 4)

        survives = bool(pval <= alpha)

        # ── Rescue via conditional linear evidence ─────────────────────
        # The CMI shuffle test is non-parametric (assumes no functional
        # form) but, precisely because of that, has low power with
        # moderate n / large Z / finite n_perm: the p-value floor is
        # 1/(n_perm+1) and the k-NN estimator's variance is high. The
        # conditional OLS-HAC already computed above (same Z of ALL
        # competitors, hence a test at least as demanding in terms of
        # confounding) costs nothing extra. If it flags a strong direct
        # effect AND with the SAME sign as the Phase 1 effect, the edge
        # is kept: sign agreement is required precisely so as not to
        # rescue an association induced by other parents (which is what
        # step 2d removes).
        if (not survives) and linear_rescue_alpha > 0 and ols is not None:
            p_lin = float(ols['pvalue'][j]) if np.isfinite(ols['pvalue'][j]) else np.nan
            th_lin = float(ols['theta'][j])
            ef1 = _num(out.at[idx, 'causal_effect']) if 'causal_effect' in out.columns else np.nan
            sign_ok = (not np.isfinite(ef1)) or abs(ef1) < 1e-12 or \
                       (np.sign(th_lin) == np.sign(ef1))
            if np.isfinite(p_lin) and p_lin <= linear_rescue_alpha and sign_ok:
                survives = True
                if 'phase2_removal_reason' in out.columns:
                    out.at[idx, 'phase2_removal_reason'] = (
                        f"conditional_linear_rescue (cmi_p={pval:.4g} > {alpha}, "
                        f"but p_conditional_OLS={p_lin:.4g} <= {linear_rescue_alpha})"
                    )

        out.at[idx, 'survives_phase2'] = survives

        if ols is not None:
            theta_scaled = ols['theta'][j] * (sd_y[0] / sd_x[j])
            se_scaled = ols['se'][j] * (sd_y[0] / sd_x[j]) if not np.isnan(ols['se'][j]) else np.nan
            out.at[idx, 'conditional_theta'] = round(float(theta_scaled), 4)
            out.at[idx, 'conditional_se'] = round(float(se_scaled), 4) if not np.isnan(se_scaled) else np.nan
            out.at[idx, 'conditional_pvalue'] = round(float(ols['pvalue'][j]), 4)

    # FAIL-SAFE: edges whose source was not in `processed` were never
    # tested -- keep them rather than defaulting to survives_phase2=False,
    # since a technical inability to evaluate a pair is not evidence
    # against it.
    _not_evaluated = [i for i in out.index if i not in set(valid_edge_idx)]
    if _not_evaluated:
        out.loc[_not_evaluated, 'survives_phase2'] = True

    return out


# =============================================================================
# 2c. RECIPROCAL EDGE RESOLUTION (ORIENTATION)
# =============================================================================
#
# WHY THIS STEP EXISTS
# -------------------------
# All of this module's parent competition (ARD, CMI and BIC pruning) is
# based on conditional (in)dependence, which is a SYMMETRIC relationship.
# If A->B is a true relationship, then B also carries information about A
# beyond what A's other parents explain — so the reverse edge B->A is
# conditionally dependent and survives ANY conditional independence test,
# by construction. No adjustment of stability_threshold, cmi_alpha or BIC
# removes this type of false positive; an ASYMMETRY criterion is needed.
#
# This step uses the asymmetry that Phase 1 has already measured (Transfer
# Entropy, which is directional by definition; and the magnitude of the
# DML effect) plus, when available, Phase 2's conditional CMI — four
# signals produced by independent stages of the pipeline.

_ORIENT_CRITERIA = ('transfer_entropy', 'strength', 'cmi_value', 'conditional_theta')


def _num(v) -> float:
    """Converts to a finite float or NaN (never raises)."""
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _orient_margin(a: float, b: float):
    """Relative margin of `a` over `b` in [-1, 1]; None if the criterion does not apply."""
    if not (np.isfinite(_num(a)) and np.isfinite(_num(b))):
        return None
    a, b = _num(a), _num(b)
    denom = max(abs(a), abs(b))
    if denom <= 1e-12:
        # both ~0 (e.g., transfer_entropy == 0 on Granger-rescue edges)
        return None
    return (a - b) / denom


def _resolve_reciprocal_edges(graph: pd.DataFrame,
                              min_margin: float = 0.10,
                              min_criteria: int = 2,
                              criteria: tuple = _ORIENT_CRITERIA,
                              max_lag_diff: Optional[int] = None,
                              protect_pvalue: Optional[float] = None,
                              max_degree: Optional[int] = None,
                              only_lag_zero: bool = False,
                              allow_unanimity_override: bool = False,
                              verbose: bool = False) -> pd.DataFrame:
    """
    For every pair of variables (u, v) where the current graph
    SIMULTANEOUSLY contains u->v and v->u, decides whether one of the
    directions is an artificial "mirror" of the other, and if so, marks
    the weaker direction with survives_phase2 = False.

    Decision rule (deliberately conservative):
      - each criterion in `criteria` present in BOTH directions votes for
        the direction with the higher value, but ONLY if the relative
        margin between them is >= `min_margin`; smaller margins => the
        criterion abstains;
      - the losing direction only falls if there are >= `min_criteria`
        votes and NO vote for the opposite direction (unanimity among
        those who voted);
      - any other situation (tie, disagreement, missing data) KEEPS both
        directions — same fail-safe policy as the rest of this module.

    Safety: a genuine physical A<->B feedback tends to
    produce similar strengths in both directions (small margins =>
    abstention) or disagreeing criteria => nothing is removed. The mirror
    of a strong unidirectional relationship, on the other hand, produces
    large and concordant margins. Increasing `min_margin` or
    `min_criteria` only makes this step MORE conservative; it never
    causes the true direction to be removed instead of the false one. To
    turn it off completely: resolve_reciprocal=False.

    If one direction has several edges (different lags), the BEST edge of
    each direction is compared; if it loses, ALL edges of that direction
    in the pair are removed.

    only_lag_zero : bool
        If True, only CONTEMPORANEOUS pairs compete (lag==0 in BOTH
        directions) -- lagged pairs (lag>=1 on either side) NEVER
        compete, they are always kept. Rationale (PCMCI/PCMCI+, Runge et
        al. 2020): X(t-tau)->Y(t) and Y(t-tau)->X(t) are distinct
        statistical hypotheses, independently testable -- the
        identification ambiguity that motivates "choosing one direction"
        only genuinely exists for contemporaneous links (same instant,
        Markov equivalence). For lagged links, competing is an
        unnecessary bias that discards genuine bidirectional coupling.
        Default False (competes by strength/TE regardless of lag).
    allow_unanimity_override : bool
        If True, ignores `protect_pvalue` when at least 2 evaluable
        criteria voted for the winning direction and at most 1 evaluable
        criterion abstains or disagrees. The p-value veto can mistakenly
        protect spurious mirrors when the data is nearly deterministic
        (conditional_pvalue saturates at ~0 for both the real direction
        and the mirror) -- but in those cases strength/TE/cmi_value/
        conditional_theta agree almost unanimously (sometimes only 2 of
        the 4 are even computable, when one side is a root with no
        competitors), with a large margin, exactly the pattern that the
        veto's own premise (docstring of the VETO section below)
        describes as evidence AGAINST genuine bidirectionality (which
        tends to abstain in >=2 criteria, not in at most 1). In a
        dense/coupled regime, however, this same override can remove
        GENUINELY bidirectional pairs that only appear unanimous due to
        estimation noise -- hence False by default here; callers should
        only enable it in the sparse regime (see
        detect_topology_and_adjust_parameters).
    """
    if graph.empty or 'survives_phase2' not in graph.columns:
        return graph

    live = graph[graph['survives_phase2'] == True]  # noqa: E712
    if live.empty:
        return graph

    by_dir = {}
    for i, r in live.iterrows():
        by_dir.setdefault((r['_src_key'], r['_tgt_key']), []).append(i)

    seen = set()
    for (u, v) in list(by_dir.keys()):
        if (v, u) not in by_dir:
            continue
        key = tuple(sorted((u, v)))
        if key in seen:
            continue
        seen.add(key)

        idx_uv, idx_vu = by_dir[(u, v)], by_dir[(v, u)]
        best_uv = max(idx_uv, key=lambda i: abs(_num(graph.at[i, 'strength'])))
        best_vu = max(idx_vu, key=lambda i: abs(_num(graph.at[i, 'strength'])))

        if max_lag_diff is not None:
            if abs(int(graph.at[best_uv, 'lag']) - int(graph.at[best_vu, 'lag'])) > max_lag_diff:
                continue

        if only_lag_zero:
            if int(graph.at[best_uv, 'lag']) != 0 or int(graph.at[best_vu, 'lag']) != 0:
                continue

        # conditional_theta only carries information INDEPENDENT of
        # 'strength' when real conditioning took place (the target had >=2
        # candidates in the parent competition). With only 1 candidate,
        # the competition engine has nothing to condition on and
        # conditional_theta collapses to (nearly) the same value as
        # strength — counting it as a vote in that case would double-count
        # the SAME evidence, not add new evidence.
        tgt_uv, tgt_vu = graph.at[best_uv, '_tgt_key'], graph.at[best_vu, '_tgt_key']
        n_parents_uv = int((live['_tgt_key'] == tgt_uv).sum())
        n_parents_vu = int((live['_tgt_key'] == tgt_vu).sum())
        theta_informative = n_parents_uv >= 2 and n_parents_vu >= 2

        # ── density limit ──────────────────────────────────────────
        # Skips resolution when either target's parent count exceeds
        # max_degree.
        if max_degree is not None and (n_parents_uv > max_degree or n_parents_vu > max_degree):
            continue

        votes_uv, votes_vu, n_evaluated, details = 0, 0, 0, []
        for c in criteria:
            if c == 'conditional_theta':
                if not theta_informative or c not in graph.columns:
                    continue
                a_val = abs(_num(graph.at[best_uv, c]))
                b_val = abs(_num(graph.at[best_vu, c]))
            else:
                if c not in graph.columns:
                    continue
                a_val, b_val = graph.at[best_uv, c], graph.at[best_vu, c]
            m = _orient_margin(a_val, b_val)
            if m is None:
                continue
            n_evaluated += 1
            if m >= min_margin:
                votes_uv += 1
                details.append(f"{c}:{m:+.2f}->{u}")
            elif m <= -min_margin:
                votes_vu += 1
                details.append(f"{c}:{m:+.2f}->{v}")

        if votes_uv >= min_criteria and votes_vu == 0:
            winner, loser_idx = (u, v), idx_vu
        elif votes_vu >= min_criteria and votes_uv == 0:
            winner, loser_idx = (v, u), idx_uv
        else:
            continue  # ambiguous -> keeps both directions

        # Near-total unanimity: at most 1 evaluable criterion abstains
        # or disagrees with the winning direction.
        n_won = votes_uv if winner == (u, v) else votes_vu
        near_total_unanimity = (allow_unanimity_override and n_evaluated >= 2 and
                              n_won >= n_evaluated - 1)

        # ── VETO: genuine bidirectional coupling ────────────────────
        # This step's premise is that the dominated direction is the
        # statistical MIRROR of the dominant one. That is false in
        # dynamical systems with physical neighborhoods (coupled
        # rings/chains, flow between adjacent entities),
        # where X_i -> X_j AND X_j -> X_i are both true edges -- and one
        # of them becomes an FN here. If the losing direction has, by
        # itself, strong conditional evidence (it survived the parent
        # competition with a low p, i.e., it carries information that the
        # OTHER parents of its target do not explain), the magnitude
        # asymmetry is not enough to eliminate it.
        if protect_pvalue is not None and near_total_unanimity and verbose:
            print(f"      [reciprocal] p-value veto IGNORED between {u} and {v}: "
                  f"{n_won}/{n_evaluated} criteria agree with "
                  f"{winner[0]} -> {winner[1]} (at most 1 abstention/disagreement)")
        if protect_pvalue is not None and not near_total_unanimity:
            _strong = []
            for i in loser_idx:
                _pv = _num(graph.at[i, 'conditional_pvalue']) if 'conditional_pvalue' in graph.columns else np.nan
                _cp = _num(graph.at[i, 'cmi_pvalue']) if 'cmi_pvalue' in graph.columns else np.nan
                if not np.isfinite(_pv) and not np.isfinite(_cp):
                    # Lone parent of its target (no other competitor): the
                    # conditional competition never runs on that side, so
                    # conditional_pvalue/cmi_pvalue are NaN due to absence
                    # of a test, not absence of evidence. Without a
                    # fallback, this direction would be left with NO
                    # protection at all just for having been the only
                    # candidate of its target -- falls back to Phase 1's
                    # (DML) raw p-value, already computed on the same row.
                    _pv_phase1 = _num(graph.at[i, 'pvalue']) if 'pvalue' in graph.columns else np.nan
                    _strong.append(np.isfinite(_pv_phase1) and _pv_phase1 <= protect_pvalue)
                else:
                    _strong.append((np.isfinite(_pv) and _pv <= protect_pvalue) or
                                   (np.isfinite(_cp) and _cp <= protect_pvalue))
            if _strong and all(_strong):
                if verbose:
                    print(f"      [reciprocal] kept BOTH directions between {u} and {v}: "
                          f"the dominated one has its own conditional evidence (p <= {protect_pvalue})")
                continue

        reason = (f"reverse_edge (dominant direction {winner[0]} -> {winner[1]}; "
                  f"{'; '.join(details)})")
        for i in loser_idx:
            graph.at[i, 'survives_phase2'] = False
            if 'phase2_removal_reason' in graph.columns:
                graph.at[i, 'phase2_removal_reason'] = reason
        if verbose:
            print(f"      [reciprocal] kept {winner[0]} -> {winner[1]}; "
                  f"removed the opposite direction ({'; '.join(details)})")

    return graph


# =============================================================================
# 2d. SIGN CONSISTENCY: PHASE 1 EFFECT vs. CONDITIONAL EFFECT
# =============================================================================
#
# WHY THIS STEP EXISTS
# -------------------------
# `conditional_theta` (the parent's coefficient when ALL other parents of
# the same target are in the model, via _hac_ols) is computed in both
# competition methods. It carries, for free, the most discriminative
# information that Phase 2 produces: if, upon conditioning on the other
# parents, the direct effect flips to the OPPOSITE SIGN with statistical
# confidence, Phase 1's marginal association was not a direct effect — it
# was induced by the other parents (mediation, confounding, collider, or
# collinearity among parents).
#
# A genuine direct effect can change in MAGNITUDE when conditioning
# (shrink or grow); it does not flip sign with p ~ 0.

def _apply_sign_consistency(graph: pd.DataFrame,
                            alpha: float = 0.05,
                            min_ratio: float = 0.0,
                            verbose: bool = False) -> pd.DataFrame:
    """
    Marks survives_phase2 = False for edges where
    sign(conditional_theta) != sign(causal_effect) AND the conditional
    estimate is significant (conditional_pvalue <= alpha).

    Fail-safe: if conditional_theta / conditional_pvalue are missing, or
    if the flip is not significant (i.e., it is within noise), the edge
    is KEPT. Zero extra computational cost — uses columns already filled
    in by the parent competition.
    """
    if graph.empty:
        return graph
    for col in ('conditional_theta', 'conditional_pvalue', 'causal_effect'):
        if col not in graph.columns:
            return graph

    for i, r in graph.iterrows():
        if not bool(r.get('survives_phase2', False)):
            continue
        th, ef = _num(r['conditional_theta']), _num(r['causal_effect'])
        pv = _num(r['conditional_pvalue'])
        if not (np.isfinite(th) and np.isfinite(ef) and np.isfinite(pv)):
            continue
        if abs(th) < 1e-12 or abs(ef) < 1e-12:
            continue
        if np.sign(th) == np.sign(ef):
            continue
        if pv > alpha:
            continue  # flip within noise -> keep
        if min_ratio > 0 and abs(th) < min_ratio * abs(ef):
            # Sign flip with negligible magnitude relative to the Phase 1
            # effect: the conditional effect COLLAPSED toward ~zero and
            # the sign is just numerical residue from that collapse --
            # could be collinearity among parents, not necessarily
            # mediation. Conservative: keep.
            continue
        graph.at[i, 'survives_phase2'] = False
        if 'phase2_removal_reason' in graph.columns:
            graph.at[i, 'phase2_removal_reason'] = (
                f"conditional_sign_inversion (Phase 1 effect {ef:+.4f} -> "
                f"conditional theta {th:+.4f}, p={pv:.4g})"
            )
        if verbose:
            print(f"      [sign] removed {r['_src_key']} -> {r['_tgt_key']}: "
                  f"{ef:+.4f} -> {th:+.4f} (p={pv:.4g})")

    return graph


def _warn_cmi_pvalue_floor(graph: pd.DataFrame, n_perm: int, verbose: bool = True) -> None:
    """
    Diagnostic: warns when ALL of the CMI p-values are tied at the floor
    1/(n_perm+1). In that regime the shuffle test discriminates nothing —
    every parent "survives" and the CMI-based parent competition becomes a
    silent no-op.
    """
    if not verbose or 'cmi_pvalue' not in graph.columns:
        return
    p = pd.to_numeric(graph['cmi_pvalue'], errors='coerce').dropna()
    if len(p) < 2:
        return
    floor = 1.0 / (n_perm + 1)
    if np.allclose(p.values, floor, atol=1e-6):
        print(f"      [WARNING] all {len(p)} CMI p-values are at the floor "
              f"{floor:.5f} (n_perm={n_perm}): the shuffle test is not "
              f"discriminating edges. Consider method='ard', increasing "
              f"cmi_n_perm, or applying an effect-size threshold on cmi_value.")


# =============================================================================
# 2e. ROBUSTNESS GATE FOR UNCONTESTED EDGES (target with only 1 candidate)
# =============================================================================
#
# WHY THIS STEP EXISTS
# --------------------------
# Steps 2/2b (ARD/CMI) and BIC pruning only remove an edge when it loses
# to ANOTHER candidate of the same target. When a target has exactly ONE
# significant candidate, the entire competition engine keeps it "by
# definition" (there is nothing to condition on) — and steps 2c/2d also do
# not apply (2c needs two directions; 2d needs a conditional_theta
# computed from other parents, which only exists when there was
# competition). Without this step, an isolated edge, even a weak one,
# passes through the entirety of Phase 2 untouched.
#
# Phase 1 already computes, for every edge, robustness diagnostics that
# are NEVER used to decide 'significant' (which normally comes only from
# the DML's 95% CI not containing zero): cascade_confidence/n_models_agree
# (how many independent model families agree with the estimate),
# robustness_value (composite robustness score), cv_ci (CI width relative
# to the effect — the larger, the noisier the estimate) and
# significant_bh (significance after multiple-testing correction).
#
# This step uses those signals ONLY on uncontested edges (the only place
# where the rest of Phase 2 has no chance to act) and ONLY rejects when at
# least `min_weak_signals` of the AVAILABLE signals independently point to
# weak evidence — the same "vote by agreement between independent
# criteria, abstain if data is missing" principle from step 2c. An
# isolated but robust-on-every-signal edge is never touched.

def _lone_edge_weak_votes(row: pd.Series,
                          robustness_floor: float,
                          cv_ci_ceiling: float) -> tuple:
    """
    Counts how many signals (among those available) indicate weak
    evidence for an uncontested edge. Returns (weak_votes,
    possible_votes). Each criterion abstains (does not count for either
    side) if the column does not exist or the value is NaN — fail-safe.
    """
    weak, possible = 0, 0

    cc = row.get('cascade_confidence', None)
    if isinstance(cc, str) and cc.strip():
        possible += 1
        if cc.strip().lower() != 'high':
            weak += 1

    rv = _num(row.get('robustness_value', np.nan))
    if np.isfinite(rv):
        possible += 1
        if rv < robustness_floor:
            weak += 1

    cv = _num(row.get('cv_ci', np.nan))
    if np.isfinite(cv):
        possible += 1
        if cv > cv_ci_ceiling:
            weak += 1

    bh = row.get('significant_bh', None)
    if isinstance(bh, (bool, np.bool_)):
        possible += 1
        if not bool(bh):
            weak += 1

    return weak, possible


def _apply_lone_edge_quality_gate(graph: pd.DataFrame,
                                  robustness_floor: float = 0.16,
                                  cv_ci_ceiling: float = 1.3,
                                  min_weak_signals: int = 2,
                                  require_unanimity: bool = False,
                                  majority_fallback: bool = True,
                                  veto_bh: bool = False,
                                  veto_pvalue: Optional[float] = None,
                                  dry_run: bool = False,
                                  verbose: bool = False) -> pd.DataFrame:
    """
    *** IMPORTANT NOTE about veto_pvalue in this function ***
    This gate only runs over edges whose target has exactly 1 candidate
    (by definition -- see the body below). For those edges,
    `_parent_competition_for_target[_cmi]` sets
    `conditional_pvalue = pvalue` from Phase 1, since there is nothing to
    condition on with a single candidate. In other words: `veto_pvalue`
    compared to `conditional_pvalue` here is the SAME p-value that already
    made Phase 1 mark `significant=True` -- a tautological veto that
    disarms the gate for exactly the case it was designed to act on
    (Phase 1 already filters by p-value; this gate exists because p-value
    alone is not sufficient for edges without a competitor). That's why
    `veto_pvalue` is ignored when the edge is truly a lone edge (always,
    given this function's scope) -- see `_lone_edge_weak_votes`/loop
    below. Accepted as a parameter, but has no effect here.
    """
    """
    For every target whose current graph has EXACTLY ONE candidate parent
    (no parent competition happened for it), re-evaluates that isolated
    edge against robustness signals already computed by Phase 1
    (cascade_confidence, robustness_value, cv_ci, significant_bh — no
    recomputation). Only marks survives_phase2 = False if >=
    `min_weak_signals` AVAILABLE signals independently indicate weak
    evidence.

    Fail-safe: targets with >=2 candidates are not touched (already
    covered by the normal parent competition); if fewer than
    `min_weak_signals` signals are available for an edge, it is KEPT.

    dry_run : bool
        If True, does NOT remove anything — only fills in
        'lone_edge_gate_would_fail' (bool) and 'lone_edge_gate_reason'
        with what the step WOULD have done. Use to audit the impact of
        the thresholds on a new dataset before letting the step actually
        remove edges.
    """
    if graph.empty or 'survives_phase2' not in graph.columns or '_tgt_key' not in graph.columns:
        return graph

    live = graph[graph['survives_phase2'] == True]  # noqa: E712
    if live.empty:
        return graph

    counts = live['_tgt_key'].value_counts()
    lone_targets = set(counts[counts == 1].index)
    if not lone_targets:
        return graph

    if dry_run:
        graph['lone_edge_gate_would_fail'] = False
        graph['lone_edge_gate_reason'] = ''

    for i, r in live.iterrows():
        if r['_tgt_key'] not in lone_targets:
            continue
        weak, possible = _lone_edge_weak_votes(r, robustness_floor, cv_ci_ceiling)
        reject = possible >= min_weak_signals and weak >= min_weak_signals

        # ── (fractional) unanimity ────────────────────────────────
        # "2 of 4" (majority_fallback=False) rejects even on an exact tie --
        # it drops edges with genuinely mixed evidence, not just weak
        # ones. This is good when the 2 weak signals ARE the majority
        # (e.g., 3 of 4), but a pure 2-2 tie is AMBIGUOUS by definition --
        # the available signals do not have enough discriminative power
        # there, and no fixed threshold gets both sides right. The default
        # is therefore asymmetric: require a MAJORITY (more than half of
        # those available, not just a fixed >= min_weak_signals) instead
        # of total unanimity -- unanimity (require_unanimity=True) remains
        # available for those who prefer the conservative extreme.
        if reject and require_unanimity and weak < possible:
            reject = False
        elif reject and majority_fallback and not require_unanimity \
                and possible >= 3 and weak * 2 <= possible:
            # tie or weak minority among >=3 available signals -> abstain.
            # majority_fallback=False always rejects when
            # weak >= min_weak_signals -- used by PRESET_PHASE2_BASE.
            reject = False

        # ── vetoes from direct statistical evidence ────────────────────
        # The gate's 4 signals are cascade robustness heuristics; none of
        # them is a formal hypothesis test on the edge. If the formal test
        # (Phase 1's BH / Phase 2's conditional p-value) says the edge is
        # significant, it takes precedence over the heuristics.
        if reject and veto_bh:
            _bh = r.get('significant_bh', None)
            if isinstance(_bh, (bool, np.bool_)) and bool(_bh):
                reject = False
        # veto_pvalue: has NO EFFECT by construction in this function --
        # see the note in the docstring (conditional_pvalue ==
        # Phase 1's pvalue on a lone edge; comparing a p-value with
        # itself vetoes nothing informative).

        if not reject:
            continue
        reason = (
            f"weak_uncontested_edge ({weak}/{possible} robustness signals "
            f"indicate weak evidence: cascade_confidence/"
            f"robustness_value/cv_ci/significant_bh)"
        )
        if dry_run:
            graph.at[i, 'lone_edge_gate_would_fail'] = True
            graph.at[i, 'lone_edge_gate_reason'] = reason
            if verbose:
                print(f"      [lone-edge gate | DRY RUN] WOULD REJECT "
                      f"{r['_src_key']} -> {r['_tgt_key']}: {weak}/{possible} "
                      f"weak signals (not removed)")
            continue
        graph.at[i, 'survives_phase2'] = False
        if 'phase2_removal_reason' in graph.columns:
            graph.at[i, 'phase2_removal_reason'] = reason
        if verbose:
            print(f"      [lone-edge gate] removed {r['_src_key']} -> "
                  f"{r['_tgt_key']}: {weak}/{possible} weak signals")

    return graph

# =============================================================================
# 2f. EXACT-LAG-MATCH MEDIATION GATE  (default: OFF)
# =============================================================================
#
# *** OFF BY DEFAULT (exact_mediation_gate=False) — ENABLE WITH CAUTION ***
# This gate can remove genuine direct edges in COUPLED dynamical systems,
# where a direct edge and a 2-hop path can both be real, independent
# causal mechanisms rather than one being an artifact of the other (see
# WHY THIS HAPPENS below) — only enable it after confirming the candidate
# graph has no chain/ring propagation among the entities involved.
#
# WHY THIS HAPPENS
# ----------------------
# The premise "a path A->B->C exists with an exact lag sum => A->C is an
# artifact of the path" only holds in SPARSE causal graphs, where
# mediation genuinely excludes a direct effect. In COUPLED dynamical
# systems (chains, rings, any topology where information propagates step
# by step between neighbors with an ~constant lag per hop — any physical
# system with additive temporal propagation), it is EXPECTED for the lag sum of a
# 2-hop path to exactly match the lag of an equally real direct edge: the
# direct edge and the indirect path are two genuine, simultaneous causal
# mechanisms, not one being an artifact of the other. In these systems, an
# exact lag match does NOT discriminate redundancy from genuine
# coexistence.
#
# WHEN TO ENABLE
# ------------
# Safe in graphs where causal sparsity is expected and there is no
# chain/ring propagation among the candidates themselves. Risky whenever
# the domain is a dynamical system with a physical neighborhood
# (flow between adjacent entities, ecological networks,
# spatially coupled climate series, etc.) — in those cases, keep it off.
#
# WHY THIS STEP EXISTS
# ------------------------------------------------
# `_explain_by_transitive_path` (right below) already computes, for every
# edge A->C, whether there is a path A->B->C within the graph with
# compatible lags — but only for reporting (tolerance lag_AC ~=
# lag_AB+lag_BC ± 1), never to decide removal. This ±1 tolerance is
# deliberate for the report (it catches more approximate cases), but has a
# cost: in dense graphs it can also flag genuine DIRECT edges whose lag
# sum from another path collides by coincidence within ±1.
#
# This step uses the SAME computation, but with ZERO tolerance (lag_AC ==
# lag_AB+lag_BC exactly) as the removal criterion — an exact lag match
# between a 2-hop path and the direct edge is stronger evidence of
# redundancy than the tolerance used in the report, BUT still insufficient
# on its own in systems with coupled propagation (see above). Runs only on
# edges that survived the current cycle's parent competition + BIC pruning
# (same spirit as 2c/2d/2e: acts last, on what is left).

def _apply_exact_mediation_gate(graph: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    For every surviving direct edge A->C, looks for a path A->B->C (B
    also surviving, same direction) whose lag sum is EXACTLY equal to the
    direct lag of A->C. If found, marks the DIRECT A->C edge as not
    surviving (edges B->C and A->B remain — they are the ones that
    explain the path).

    Fail-safe: no exact match found => edge kept. Does not remove B->C nor
    A->B, only the redundant A->C edge.
    """
    if graph.empty or 'survives_phase2' not in graph.columns:
        return graph

    live = graph[graph['survives_phase2'] == True]  # noqa: E712
    if live.empty:
        return graph

    by_src = {}
    for i, r in live.iterrows():
        by_src.setdefault(r['_src_key'], []).append((i, r))

    for i, row in live.iterrows():
        a, c, lag_ac = row['_src_key'], row['_tgt_key'], int(row['lag'])
        found = None
        for _, mid in by_src.get(a, []):
            b = mid['_tgt_key']
            if b == c or b == a:
                continue
            lag_ab = int(mid['lag'])
            for j, r2 in by_src.get(b, []):
                if r2['_tgt_key'] != c:
                    continue
                lag_bc = int(r2['lag'])
                if (lag_ab + lag_bc) == lag_ac:
                    found = f"{a} -> {b} -> {c} (lags {lag_ab}+{lag_bc}=={lag_ac}, exact)"
                    break
            if found:
                break
        if found:
            graph.at[i, 'survives_phase2'] = False
            if 'phase2_removal_reason' in graph.columns:
                graph.at[i, 'phase2_removal_reason'] = f"exact_mediation ({found})"
            if verbose:
                print(f"      [exact mediation] removed {a} -> {c}: {found}")

    return graph


# =============================================================================
# 3b. SPURIOUS HUB GATE — statistically atypical out-degree (domain-
#     agnostic, see full docstring below)
# =============================================================================
#
# *** OFF BY DEFAULT (spurious_hub_gate=False) ***
#
# Flags source entities whose out-degree in the current candidate graph is
# a statistical outlier (MAD z-score) relative to the rest of the graph --
# domain-agnostic, since it works purely from the graph's own degree
# distribution, without hardcoding any entity name.
#
# Why it runs HERE (inside apply_structural_filter's cycle loop), and not
# in Stage 7 (causal_pen/AdaptiveCausalPenalty): the hub pattern only
# becomes statistically visible after the parent
# competition + BIC pruning + reciprocal resolution have already run for
# at least one cycle: the hub is an EMERGENT effect of the structural
# competition itself, not
# something detectable in the input candidates -- calibrating this at
# Stage 7 would be circular (the signal does not exist yet at the time
# Stage 7 runs). Running it inside this function's iterative loop resolves the
# circularity without needing a second full pass through the pipeline:
# each cycle sees the out-degree left over from the previous cycle
# (already passed through competition+BIC+reciprocals) and prunes hubs
# before the NEXT cycle competes again -- the same "runs last, on what is
# left" logic that governs the other 2c-2g gates in this loop.
def _apply_spurious_hub_gate(
    graph: pd.DataFrame,
    zscore_threshold: float = 3.5,
    min_origins: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Marks survives_phase2=False for every edge whose source (_src_key)
    has an out-degree (number of distinct _tgt_key among the edges STILL
    surviving in this cycle) that is a statistical outlier relative to
    the other sources in the SAME graph -- MAD modified z-score
    (Iglewicz & Hoaglin, 1993), without reading any entity name or any
    metadata external to the candidate graph itself.

    zscore_threshold=3.5: standard literature default (conservative). A
    lower threshold risks flagging a topology where a genuinely high,
    homogeneous out-degree is an intrinsic property of the true causal
    structure (e.g. every node in a ring/lattice system causing the same
    number of neighbors), not a symptom of confounding.

    min_origins: MAD over fewer than `min_origins` distinct sources is not
    reliable (few data points, a single high value becomes an "outlier"
    by chance) -- below that, nothing is flagged.

    Fail-safe: MAD==0 (all sources with the same out-degree) => flags
    nothing.
    """
    if graph.empty or 'survives_phase2' not in graph.columns:
        return graph

    live = graph[graph['survives_phase2'] == True]  # noqa: E712
    if live.empty:
        return graph

    outdeg = live.groupby('_src_key')['_tgt_key'].nunique()
    if len(outdeg) < min_origins:
        return graph

    vals = outdeg.values.astype(float)
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    if mad <= 1e-9:
        return graph

    z = 0.6745 * (vals - median) / mad
    hubs = set(outdeg.index[z > zscore_threshold])
    if not hubs:
        return graph

    mask = graph['survives_phase2'] & graph['_src_key'].isin(hubs)
    for i in graph.index[mask]:
        graph.at[i, 'survives_phase2'] = False
        if 'phase2_removal_reason' in graph.columns:
            src = graph.at[i, '_src_key']
            graph.at[i, 'phase2_removal_reason'] = (
                f"spurious_hub (source '{src}' with a statistically "
                f"outlier out-degree among this cycle's candidates)"
            )
    if verbose:
        print(f"      [spurious hub] flagged sources: {sorted(hubs)} "
              f"({int(mask.sum())} edges removed)")

    return graph


def _explain_by_transitive_path(graph_edges: pd.DataFrame) -> pd.DataFrame:
    """
    Marks, for reporting/auditing purposes, direct edges A->C for which
    there is an alternative path A->B->C within the current graph (same
    direction, compatible lags: lag_AC ~= lag_AB + lag_BC).

    This does NOT decide removal (that is decided by the conditional
    parent competition, which already "knows" this implicitly) — it
    serves to explain to the user WHY an edge was removed in Phase 2
    (document item 3: "Structural Graph Pruning"), making the result
    interpretable.
    """
    edges = graph_edges[graph_edges['survives_phase2']].copy()
    if edges.empty:
        graph_edges['alternative_path'] = ''
        return graph_edges

    by_src = {}
    for _, r in edges.iterrows():
        by_src.setdefault(r['_src_key'], []).append(r)

    explanations = {i: '' for i in graph_edges.index}
    for i, row in graph_edges.iterrows():
        a, c, lag_ac = row['_src_key'], row['_tgt_key'], int(row['lag'])
        for mid in by_src.get(a, []):
            b = mid['_tgt_key']
            if b == c or b == a:
                continue
            lag_ab = int(mid['lag'])
            for _, r2 in edges[edges['_src_key'] == b].iterrows():
                if r2['_tgt_key'] != c:
                    continue
                lag_bc = int(r2['lag'])
                if abs((lag_ab + lag_bc) - lag_ac) <= 1:
                    explanations[i] = f"{a} -> {b} -> {c} (lags {lag_ab}+{lag_bc}~={lag_ac})"
                    break
            if explanations[i]:
                break

    graph_edges = graph_edges.copy()
    graph_edges['alternative_path'] = pd.Series(explanations)
    return graph_edges


# =============================================================================
# 4. GLOBAL PRUNING BY INFORMATION CRITERION (BIC)
# =============================================================================

def _bic_backward_elimination(target: str, edges: pd.DataFrame,
                               processed: dict,
                               hac_lags: Optional[int] = None,
                               protect_pvalue: Optional[float] = None,
                               penalty_scale: float = 1.0,
                               max_degree: Optional[int] = None,
                               min_collinearity_to_prune: Optional[float] = None) -> pd.DataFrame:
    """
    Item 6 (Global Optimization): given the set of parents that survived
    the conditional competition, performs greedy backward elimination
    guided by BIC — at each step, removes the parent whose removal most
    reduces the target's model BIC, until no removal improves the
    criterion.

    This seeks the smallest subset of relationships capable of explaining
    the target variable (parsimony), complementing the stability-based
    selection (which is more permissive) with an explicit information-gain
    requirement (Sparse Graph Optimization / MDL-like).

    min_collinearity_to_prune : float | None
        If set, a removal only happens when the candidate has
        correlation >= this value with SOME other surviving parent at the
        time of removal (direct evidence of redundancy). Without this
        evidence, the candidate stays protected at that step even if its
        removal would reduce the BIC -- the marginal drop is treated as
        estimation instability in a dense target, not as a duplicate.
        More surgical than `max_degree` (which turns off the whole pruning
        step above a given degree): it keeps removing genuine duplicates
        at any density, only protecting candidates that have no collinear
        pair. The two can be used together (defense in depth) or just
        this one. None (default) disables the collinearity check.
    """
    active = edges[edges['survives_phase2']].copy()
    if len(active) <= 1:
        return edges  # nothing to prune (0 or 1 parent has no redundancy to test)

    if min_collinearity_to_prune is not None and 'bic_protected_collinearity' not in edges.columns:
        edges['bic_protected_collinearity'] = False

    # ── density limit ──────────────────────────────────────────────
    # BIC pruning behaves differently depending on the target's density:
    # at high density it can remove true edges. The reason is structural,
    # not a threshold issue: BIC decides by the
    # JOINT fit, and in a target with many mutually correlated parents
    # (strongly coupled systems) each true parent's marginal
    # contribution becomes small even when all of them are real -- the
    # criterion starts rewarding parsimony against the graph's truth.
    if max_degree is not None and len(active) > max_degree:
        return edges

    y = processed[target]
    parents = list(active.index)

    def _fit_bic(parent_idx_list):
        if not parent_idx_list:
            ss_res = float(((y - np.nanmean(y)) ** 2).sum())
            n = np.sum(~np.isnan(y))
            return _bic(ss_res, n, 0)
        cols = []
        for idx in parent_idx_list:
            row = active.loc[idx]
            cols.append(_shifted(processed[row['_src_key']], int(row['lag'])))
        X = np.column_stack(cols)
        mask = _valid_rows(y, *[X[:, j] for j in range(X.shape[1])])
        if mask.sum() < len(parent_idx_list) + 5:
            return -np.inf  # insufficient data: does not penalize (keeps the edge)
        X_std, _, _ = _standardize(X[mask])
        y_std, _, _ = _standardize(y[mask].reshape(-1, 1))
        ols = _hac_ols(X_std, y_std.ravel(), hac_lags=hac_lags)
        if ols is None:
            return -np.inf
        return _bic(ols['ss_res'], ols['n'], ols['k'], penalty_scale=penalty_scale)

    # ── protected parents ──────────────────────────────────────────────
    # BIC decides by the JOINT fit and does not distinguish "redundant
    # parent" from "real parent with a noisy estimate" -- both have a
    # weak marginal contribution to the model. A parent whose conditional
    # effect (already controlling for all other candidates) is
    # statistically significant is NOT redundant by definition: no
    # combination of the others explains what it explains. These are
    # excluded from backward elimination.
    protected = set()
    if protect_pvalue is not None and 'conditional_pvalue' in active.columns:
        for idx in parents:
            pv = _num(active.at[idx, 'conditional_pvalue'])
            if np.isfinite(pv) and pv <= protect_pvalue:
                protected.add(idx)

    current = parents[:]
    current_bic = _fit_bic(current)

    # ── real collinearity as a pre-condition for removal ────────────
    # Removing `best_drop` (whichever most reduces BIC) without checking
    # WHETHER the reason for the marginal drop is genuine redundancy
    # fails as described above: in a dense target, the parent with the weakest marginal
    # contribution can be a REAL parent whose information simply gets
    # diluted among many candidates — not a duplicate of anyone.
    #
    # min_collinearity_to_prune makes the removal condition EXPLICIT: only
    # removes a candidate if it has high correlation with at least one
    # other CURRENT survivor (the same metric already used by the post-hoc
    # rescue in `_rescue_bic_pruned_low_collinearity`, but applied BEFORE
    # removal instead of after). Among the candidates that BIC would
    # approve removing at a step, the loop picks the one with the lowest
    # resulting BIC that ALSO passes this test; those without a collinear
    # pair stay protected at that step (they can be re-evaluated in a
    # later step if some other parent leaves and changes the collinearity
    # picture — hence the outer loop keeps running normally).
    #
    # None (default) always removes best_drop, without a collinearity
    # check.
    def _max_collinearity(idx, pairs):
        row = active.loc[idx]
        s1 = _shifted(processed[row['_src_key']], int(row['lag']))
        best = 0.0
        for p2 in pairs:
            if p2 == idx:
                continue
            row2 = active.loc[p2]
            s2 = _shifted(processed[row2['_src_key']], int(row2['lag']))
            mask = _valid_rows(s1, s2)
            if mask.sum() < 10:
                continue
            c = abs(np.corrcoef(s1[mask], s2[mask])[0, 1])
            if np.isfinite(c):
                best = max(best, c)
        return best

    improved = True
    while improved and len(current) > 1:
        improved = False
        candidates = []
        for idx in current:
            if idx in protected:
                continue
            candidate = [p for p in current if p != idx]
            bic_c = _fit_bic(candidate)
            if bic_c < current_bic - 1e-6:
                candidates.append((bic_c, idx))
        candidates.sort(key=lambda t: t[0])  # lowest resulting BIC first

        best_bic, best_drop = None, None
        for bic_c, idx in candidates:
            if min_collinearity_to_prune is None:
                best_bic, best_drop = bic_c, idx
                break
            if _max_collinearity(idx, current) >= min_collinearity_to_prune:
                best_bic, best_drop = bic_c, idx
                break
            # no collinear pair: protected AT THIS STEP (evidence of
            # estimation instability in a dense target, not of
            # redundancy) -- tries the next candidate on the list.
            if 'bic_protected_collinearity' in edges.columns:
                edges.at[idx, 'bic_protected_collinearity'] = True

        if best_drop is not None:
            current.remove(best_drop)
            current_bic = best_bic
            improved = True

    removed_by_bic = set(parents) - set(current)
    for idx in removed_by_bic:
        edges.at[idx, 'survives_phase2'] = False
        edges.at[idx, 'phase2_removal_reason'] = 'global_bic_pruning'

    return edges


# =============================================================================
# 2g. BIC-PRUNING RESCUE FOR LOW COLLINEARITY  (default: ON)
# =============================================================================
#
# WHY THIS STEP EXISTS
# --------------------------
# `_bic_backward_elimination` above decides removal by the target model's
# JOINT fit, not by each edge's individual evidence. This is correctly
# conservative when the removed candidates carry information that
# REALLY overlaps with the survivors (genuine collinearity: two mutually
# correlated candidates, BIC keeps the best and drops the other — correct
# decision, don't touch).
#
# But there is a second regime: a candidate can be removed by the joint
# fit not because it is redundant with the survivors, but because its own
# estimate is noisy (large standard error relative to the effect, small
# sample) -- even when it does not actually overlap with any of them. BIC
# does not distinguish "redundant candidate" from "individually unstable
# but uncorrelated with the rest candidate" -- both produce the same
# symptom (weak marginal contribution in the joint model), but only the
# first is real evidence of a spurious edge.
#
# This step distinguishes the two cases with a direct test: it computes
# the correlation between the removed candidate's (lagged) series and the
# (lagged) series of EACH parent that survived on the same target. If the
# maximum correlation is LOW (the candidate does not actually overlap with
# what remained) AND Phase 1's individual evidence for that candidate is
# still strong (significant_bh, cascade_confidence, robustness_value —
# the same signals from step 2e, here used in the opposite direction:
# requiring them to be STRONG to rescue, not weak to remove), the edge is
# returned to the graph. Fail-safe: any missing signal, or correlation
# above the threshold, or weak individual evidence => keeps BIC's
# decision.
#
# On by default (bic_rescue=True); bic_rescue=False keeps BIC's decision
# unchanged.

def _rescue_bic_pruned_low_collinearity(edges: pd.DataFrame,
                                        processed: dict,
                                        max_collinearity: float = 0.3,
                                        min_robustness: float = 0.16,
                                        verbose: bool = False) -> pd.DataFrame:
    """
    Re-evaluates, within a single target, the edges that
    `global_bic_pruning` just removed
    (edges['phase2_removal_reason']=='global_bic_pruning',
    survives_phase2=False). Rescues (survives_phase2=True again) when:
      1. correlation(candidate's lagged series, EACH surviving parent's
         lagged series on the same target) < max_collinearity for all of
         them — i.e., the candidate genuinely does not overlap with what
         remained;
      2. cascade_confidence == 'high' (if the column exists);
      3. robustness_value >= min_robustness (if the column exists);
      4. significant_bh == True (if the column exists).
    Criteria 2-4 only need to satisfy the ones that are AVAILABLE; if NONE
    is available, it does not rescue (fail-safe — no strong individual
    evidence to justify overturning BIC).
    """
    removed = edges[(edges.get('phase2_removal_reason') == 'global_bic_pruning') &
                      (edges['survives_phase2'] == False)]  # noqa: E712
    if removed.empty:
        return edges

    survivors = edges[edges['survives_phase2'] == True]  # noqa: E712

    for idx, row in removed.iterrows():
        cand_series = _shifted(processed[row['_src_key']], int(row['lag']))

        max_corr = 0.0
        for _, surv in survivors.iterrows():
            surv_series = _shifted(processed[surv['_src_key']], int(surv['lag']))
            mask = _valid_rows(cand_series, surv_series)
            if mask.sum() < 10:
                max_corr = np.inf  # insufficient data to evaluate -> does not rescue
                break
            c = abs(np.corrcoef(cand_series[mask], surv_series[mask])[0, 1])
            if np.isfinite(c):
                max_corr = max(max_corr, c)
        if max_corr >= max_collinearity:
            continue  # genuinely collinear with something that survived -> BIC was right

        checks, available = [], 0
        if 'cascade_confidence' in edges.columns:
            cc = row.get('cascade_confidence')
            if isinstance(cc, str) and cc.strip():
                available += 1
                checks.append(cc.strip().lower() == 'high')
        if 'robustness_value' in edges.columns:
            rv = _num(row.get('robustness_value', np.nan))
            if np.isfinite(rv):
                available += 1
                checks.append(rv >= min_robustness)
        if 'significant_bh' in edges.columns:
            bh = row.get('significant_bh')
            if isinstance(bh, (bool, np.bool_)):
                available += 1
                checks.append(bool(bh))

        if available == 0 or not all(checks):
            continue

        edges.at[idx, 'survives_phase2'] = True
        edges.at[idx, 'phase2_removal_reason'] = (
            f"rescued_post_bic (max collinearity={max_corr:.2f} < {max_collinearity}, "
            f"strong individual evidence in {available}/{available} available signals)"
        )
        if verbose:
            print(f"      [BIC rescue] restored {row['_src_key']} -> "
                  f"{row['_tgt_key']}: max collinearity={max_corr:.2f}")

    return edges


def _apply_lag_ceiling_gate(graph: pd.DataFrame,
                            lag_max: Optional[int],
                            verbose: bool = False) -> pd.DataFrame:
    """
    Marks survives_phase2 = False for edges with lag > `lag_max`.

    WHY THIS STEP EXISTS
    -------------------------
    Lag tends to be a highly discriminative signal between a true edge and
    a false positive: beyond a certain horizon, a detected association is
    increasingly likely to be sampling coincidence rather than causal
    propagation, so capping lag can remove a disproportionate share of
    false positives relative to true edges.

    *** WARNING: THIS THRESHOLD IS DOMAIN-DEPENDENT ***
    The useful value depends on the system's physical time scale and the
    sampling granularity. In daily series with slow propagation
    between distant entities, real lags can easily exceed 4 -- in that
    case a low ceiling removes genuine TPs. The default is None (off) in
    this module; whoever enables it knows their domain's scale. Calibrate
    by looking at the lag distribution of the relationships you KNOW to be
    true before fixing a value.
    """
    if graph.empty or lag_max is None or 'survives_phase2' not in graph.columns:
        return graph
    n_rm = 0
    for i, r in graph.iterrows():
        if not bool(r.get('survives_phase2', False)):
            continue
        try:
            lag = int(r['lag'])
        except (TypeError, ValueError, KeyError):
            continue          # fail-safe: unreadable lag -> keep
        if lag > lag_max:
            graph.at[i, 'survives_phase2'] = False
            n_rm += 1
            if 'phase2_removal_reason' in graph.columns:
                graph.at[i, 'phase2_removal_reason'] = (
                    f"lag_above_ceiling (lag={lag} > lag_max={lag_max})")
    if verbose and n_rm:
        print(f"      [lag ceiling] {n_rm} edge(s) removed with lag > {lag_max}")
    return graph


def _run_parent_competition(method: str, tgt: str, parent_edges: pd.DataFrame, processed: dict,
                             n_boot: int, stability_threshold: float, hac_lags: Optional[int],
                             cmi_k: int, cmi_n_perm: int, cmi_alpha: float,
                             cmi_n_neighbors_shuffle: int,
                             cmi_max_cond: Optional[int] = None,
                             cmi_linear_rescue_alpha: float = 0.0,
                             cmi_block: bool = False) -> pd.DataFrame:
    """
    Single dispatch point between the two parent-competition methods
    (document item 1): 'ard' (structural bootstrap via ARDRegression,
    section 2) or 'cmi' (conditional independence test via k-NN
    Conditional Mutual Information, section 2b). Kept separate from the
    orchestration to avoid duplicating the loop in
    optimize_graph_structure() and apply_structural_filter().
    """
    if method == 'cmi':
        return _parent_competition_for_target_cmi(
            tgt, parent_edges, processed,
            k=cmi_k, n_perm=cmi_n_perm, alpha=cmi_alpha,
            n_neighbors_shuffle=cmi_n_neighbors_shuffle,
            max_cond=cmi_max_cond,
            linear_rescue_alpha=cmi_linear_rescue_alpha,
            block=cmi_block,
        )
    elif method == 'ard':
        return _parent_competition_for_target(
            tgt, parent_edges, processed,
            n_boot=n_boot, stability_threshold=stability_threshold,
            hac_lags=hac_lags,
        )
    raise ValueError(f"invalid method: {method!r}. Use 'ard' or 'cmi'.")


def _process_target_structural(args: tuple) -> pd.DataFrame:
    """
    Per-target work body (parent competition + BIC pruning + rescue),
    factored out as a top-level module function so it can be dispatched
    via `joblib.Parallel` (separate processes require a picklable
    callable — closures/lambdas don't work). Each target is independent
    of the others within the same cycle, so the loop in
    optimize_graph_structure() and apply_structural_filter() can call this
    in parallel, one target per task.
    """
    (method, tgt, parent_edges, processed, n_boot, stability_threshold, hac_lags,
     cmi_k, cmi_n_perm, cmi_alpha, cmi_n_neighbors_shuffle, cmi_max_cond,
     cmi_linear_rescue_alpha, use_bic_pruning, bic_protect_pvalue, bic_penalty_scale,
     bic_max_degree, bic_min_collinearity_to_prune, bic_rescue,
     bic_rescue_max_collinearity, bic_rescue_min_robustness, verbose,
     cmi_block) = args

    competed = _run_parent_competition(
        method, tgt, parent_edges, processed,
        n_boot=n_boot, stability_threshold=stability_threshold,
        hac_lags=hac_lags,
        cmi_k=cmi_k, cmi_n_perm=cmi_n_perm, cmi_alpha=cmi_alpha,
        cmi_n_neighbors_shuffle=cmi_n_neighbors_shuffle,
        cmi_max_cond=cmi_max_cond,
        cmi_linear_rescue_alpha=cmi_linear_rescue_alpha,
        cmi_block=cmi_block,
    )
    if use_bic_pruning:
        competed = _bic_backward_elimination(
            tgt, competed, processed, hac_lags=hac_lags,
            protect_pvalue=bic_protect_pvalue,
            penalty_scale=bic_penalty_scale,
            max_degree=bic_max_degree,
            min_collinearity_to_prune=bic_min_collinearity_to_prune,
        )
        if bic_rescue:
            competed = _rescue_bic_pruned_low_collinearity(
                competed, processed,
                max_collinearity=bic_rescue_max_collinearity,
                min_robustness=bic_rescue_min_robustness, verbose=verbose,
            )
    return competed


def _run_targets_structural(targets: list, graph: pd.DataFrame, processed: dict,
                             method: str, n_boot: int, stability_threshold: float,
                             hac_lags: Optional[int], cmi_k: int, cmi_n_perm: int,
                             cmi_alpha: float, cmi_n_neighbors_shuffle: int,
                             cmi_max_cond: Optional[int], cmi_linear_rescue_alpha: float,
                             use_bic_pruning: bool, bic_protect_pvalue: Optional[float],
                             bic_penalty_scale: float, bic_max_degree: Optional[int],
                             bic_min_collinearity_to_prune: Optional[float],
                             bic_rescue: bool, bic_rescue_max_collinearity: float,
                             bic_rescue_min_robustness: float, verbose: bool,
                             n_jobs: int, cmi_block: bool = False) -> list:
    """
    Assembles the current cycle's per-target tasks and dispatches them —
    in parallel via joblib (processes, 'loky' backend) when `n_jobs != 1`
    and there is more than one target, sequentially otherwise (without
    joblib installed or with n_jobs=1).
    """
    tasks = [
        (method, tgt, graph[graph['_tgt_key'] == tgt], processed, n_boot,
         stability_threshold, hac_lags, cmi_k, cmi_n_perm, cmi_alpha,
         cmi_n_neighbors_shuffle, cmi_max_cond, cmi_linear_rescue_alpha,
         use_bic_pruning, bic_protect_pvalue, bic_penalty_scale, bic_max_degree,
         bic_min_collinearity_to_prune, bic_rescue, bic_rescue_max_collinearity,
         bic_rescue_min_robustness, verbose, cmi_block)
        for tgt in targets
    ]
    if HAS_JOBLIB and n_jobs != 1 and len(tasks) > 1:
        return Parallel(n_jobs=n_jobs)(delayed(_process_target_structural)(t) for t in tasks)
    return [_process_target_structural(t) for t in tasks]


# =============================================================================
# 5. ORCHESTRATION — ITERATIVE CYCLE
# =============================================================================

# =============================================================================
# 5d. TOPOLOGY AUTO-DETECTION (average competitor degree per target)
# =============================================================================
# Motivation: global_bic_pruning and reverse_edge behave differently
# depending on the density of the candidate graph (how many competing
# parents each target has after Phase 1). In a dense target, BIC's joint
# fit dilutes each true parent's marginal contribution (see
# _bic_backward_elimination), and reciprocal pairs are more often
# genuinely bidirectional, so reverse_edge's mirror premise holds less
# often.
#
# That's why the auto-detection below turns off BIC pruning in the dense
# regime, but does NOT turn off reciprocal resolution -- instead it makes
# its voting criteria more demanding.
#
# Signal used: average degree of candidate parents per target, computed
# from ALL candidate rows of Phase 1 (the raw candidate graph, see
# _average_competitor_degree) -- WITHOUT using any label/ground truth
# (same design constraint as the rest of the module).
# cmi_max_cond is NOT routed by this same regime signal.

AVERAGE_DEGREE_DENSE_THRESHOLD = 6.0

# Ceiling of the DENSE/COUPLED regime -- see AVERAGE_DEGREE_ULTRA_DENSE_THRESHOLD
# below (defined after the DENSE/SPARSE parameter blocks, which it reuses).

# Parameters applied when the graph is classified as DENSE/COUPLED
# (AVERAGE_DEGREE_DENSE_THRESHOLD <= avg_degree < AVERAGE_DEGREE_ULTRA_DENSE_THRESHOLD).
# See AVERAGE_DEGREE_ULTRA_DENSE_THRESHOLD below for the ceiling.
BIC_USE_PRUNING_DENSE = False           # global_bic_pruning: off
RECIPROCAL_MIN_MARGIN_DENSE = 0.20      # stricter than SPARSE (0.10) -- requires greater asymmetry
RECIPROCAL_MIN_CRITERIA_DENSE = 3       # stricter than SPARSE (2) -- requires more concordant criteria
# Genuine bidirectional coupling veto (protect_pvalue): in the dense
# regime, genuine reciprocal pairs can have cmi_pvalue saturated near the
# permutation test's floor -- a veto stricter than the pipeline's own
# standard alpha (0.05) would discard the genuine half of the pair, so
# the DENSE floor is aligned with that same alpha instead of an arbitrary
# stricter number.
RECIPROCAL_PROTECT_PVALUE_DENSE = 0.05
# Unanimity override (see _resolve_reciprocal_edges) deliberately OFF
# here: in the dense regime, estimation noise alone can produce
# "apparent" unanimity for pairs that ARE genuinely bidirectional --
# exactly the scenario the p-value veto exists to protect. Turning the
# override on here would destroy that protection.
RECIPROCAL_ALLOW_UNANIMITY_OVERRIDE_DENSE = False
# cmi_linear_rescue_alpha: OLS-conditional p-value ceiling to rescue an
# edge rejected by CMI (see the parameter's docstring in
# apply_structural_filter). Same strict value as the SPARSE regime: a
# looser value would also return edges CMI rejected for structural
# redundancy.
CMI_LINEAR_RESCUE_ALPHA_DENSE = 0.0001

# Parameters applied in the SPARSE regime.
BIC_USE_PRUNING_SPARSE = True
RECIPROCAL_MIN_MARGIN_SPARSE = 0.10
RECIPROCAL_MIN_CRITERIA_SPARSE = 2
# Kept strict: in the sparse regime, the same cmi_pvalue saturation near
# the permutation floor ALSO appears on spurious mirrors of true
# unidirectional edges -- loosening this here would mistakenly protect
# those mirrors.
RECIPROCAL_PROTECT_PVALUE_SPARSE = 0.001
# Unanimity override ON here: this is the mechanism that resolves
# spurious-mirror cases in the sparse regime without needing to loosen
# the fixed p-value floor above.
RECIPROCAL_ALLOW_UNANIMITY_OVERRIDE_SPARSE = True
# Strict cmi_linear_rescue_alpha: with small samples, the OLS-conditional
# p-value alone is not enough to overturn a CMI rejection.
CMI_LINEAR_RESCUE_ALPHA_SPARSE = 0.0001

# Ceiling of the DENSE/COUPLED regime -- above this, falls back to the
# SPARSE parameters (BIC_USE_PRUNING on, strict reciprocal veto) instead
# of applying the DENSE loosening.
#
# The DENSE loosening assumes candidate density reflects real causality
# (genuinely dense coupling, where turning off BIC pruning recovers true
# pairs it would otherwise mistakenly cut). Above the ultra-dense
# ceiling, high density instead tends to come from an excess of
# correlated/redundant candidates without being causal -- exactly what
# BIC pruning exists to filter -- so the SPARSE (pruning-on) behavior is
# applied there instead.
AVERAGE_DEGREE_ULTRA_DENSE_THRESHOLD = 15.0


def _average_competitor_degree(results: pd.DataFrame) -> float:
    """
    Average degree of candidate parents per target, using ALL rows of
    `results` -- i.e., the RAW topology of the candidate graph, BEFORE any
    significance filtering.

    IMPORTANT: using `results['significant'] == True` instead (the same
    cand_mask that optimize_graph_structure/apply_structural_filter use to
    decide who enters Phase 2) SEEMS more "correct" at first glance, but
    measures density AFTER the fusion stage has already filtered out a
    good share of the spurious candidates -- and this filtering removes
    proportionally MORE candidates in dense domains, masking the true
    density and risking misclassifying a genuinely dense graph as
    'sparse'. The regime thresholds are defined on the RAW graph, which
    is why this function uses all rows.
    """
    if results is None or results.empty:
        return 0.0
    tgt_key = results['target_entity'].astype(str) + '__' + results['target_property'].astype(str)
    degrees = tgt_key.value_counts()
    return float(degrees.mean()) if len(degrees) else 0.0


def detect_topology_and_adjust_parameters(
    results: pd.DataFrame,
    dense_threshold: float = AVERAGE_DEGREE_DENSE_THRESHOLD,
    ultra_dense_threshold: float = AVERAGE_DEGREE_ULTRA_DENSE_THRESHOLD,
    verbose: bool = True,
) -> dict:
    """
    Computes the average competitor degree per target and returns a dict
    of kwargs ready for `optimize_graph_structure(results, processed,
    **kwargs)`, adjusting use_bic_pruning / reciprocal_min_margin /
    reciprocal_min_criteria / reciprocal_protect_pvalue /
    reciprocal_allow_unanimity_override / cmi_linear_rescue_alpha
    according to the detected regime (see the section above for the
    rationale behind each choice). Does NOT use ground truth --
    only the candidate graph's own structure from Phase 1.

    Three regimes (not two): avg_degree < dense_threshold -> SPARSE;
    dense_threshold <= avg_degree < ultra_dense_threshold -> DENSE/COUPLED
    (loosened parameters); avg_degree >= ultra_dense_threshold -> falls
    back to the SPARSE parameters (see the docstring of
    AVERAGE_DEGREE_ULTRA_DENSE_THRESHOLD — extreme density from an excess
    of redundant candidates, not from genuine causal coupling).

    Can be called manually (to inspect/adjust before running) or
    automatically via optimize_graph_structure(...,
    auto_detect_topology=True).
    """
    avg_degree = _average_competitor_degree(results)
    dense = dense_threshold <= avg_degree < ultra_dense_threshold

    if dense:
        params = dict(
            use_bic_pruning=BIC_USE_PRUNING_DENSE,
            reciprocal_min_margin=RECIPROCAL_MIN_MARGIN_DENSE,
            reciprocal_min_criteria=RECIPROCAL_MIN_CRITERIA_DENSE,
            reciprocal_protect_pvalue=RECIPROCAL_PROTECT_PVALUE_DENSE,
            reciprocal_allow_unanimity_override=RECIPROCAL_ALLOW_UNANIMITY_OVERRIDE_DENSE,
            cmi_linear_rescue_alpha=CMI_LINEAR_RESCUE_ALPHA_DENSE,
        )
        regime = "DENSE/COUPLED"
    else:
        params = dict(
            use_bic_pruning=BIC_USE_PRUNING_SPARSE,
            reciprocal_min_margin=RECIPROCAL_MIN_MARGIN_SPARSE,
            reciprocal_min_criteria=RECIPROCAL_MIN_CRITERIA_SPARSE,
            reciprocal_protect_pvalue=RECIPROCAL_PROTECT_PVALUE_SPARSE,
            reciprocal_allow_unanimity_override=RECIPROCAL_ALLOW_UNANIMITY_OVERRIDE_SPARSE,
            cmi_linear_rescue_alpha=CMI_LINEAR_RESCUE_ALPHA_SPARSE,
        )
        regime = "ULTRA-DENSE (falls back to SPARSE)" if avg_degree >= ultra_dense_threshold else "SPARSE (default)"

    if verbose:
        print(f"  [auto-topology] avg_competitor_degree={avg_degree:.2f} "
              f"(thresholds={dense_threshold}-{ultra_dense_threshold}) -> regime {regime}")
        print(f"  [auto-topology] adjusted parameters: {params}")

    return {**params, "_avg_competitor_degree": avg_degree, "_detected_regime": regime}


def optimize_graph_structure(results: pd.DataFrame,
                              processed: dict,
                              max_iter: int = 5,
                              method: str = 'ard',
                              n_boot: int = 100,
                              stability_threshold: float = 0.6,
                              use_bic_pruning: bool = True,
                              hac_lags: Optional[int] = None,
                              cmi_k: int = 5,
                              cmi_n_perm: int = 200,
                              cmi_alpha: float = 0.05,
                              cmi_n_neighbors_shuffle: int = 5,
                              cmi_max_cond: Optional[int] = 4,
                              cmi_block: bool = False,
                              cmi_linear_rescue_alpha: float = 0.01,
                              resolve_reciprocal: bool = True,
                              reciprocal_min_margin: float = 0.10,
                              reciprocal_min_criteria: int = 2,
                              reciprocal_protect_pvalue: Optional[float] = None,   # a p-value veto disarms the gate
                              reciprocal_max_lag_diff: Optional[int] = None,
                              reciprocal_max_degree: Optional[int] = 6,
                              reciprocal_only_lag_zero: bool = False,
                              reciprocal_allow_unanimity_override: bool = False,
                              sign_consistency: bool = True,
                              sign_alpha: float = 0.05,
                              sign_min_ratio: float = 0.0,
                              lone_edge_gate: bool = True,
                              lone_edge_robustness_floor: float = 0.16,
                              lone_edge_cv_ci_ceiling: float = 1.3,
                              lone_edge_min_weak_signals: int = 2,
                              lone_edge_majority_fallback: bool = False,
                              lone_edge_require_unanimity: bool = False,
                              lone_edge_veto_bh: bool = False,
                              lone_edge_veto_pvalue: Optional[float] = None,  # no effect on lone edges -- see docstring
                              lone_edge_dry_run: bool = False,
                              exact_mediation_gate: bool = False,
                              lag_max_significant: Optional[int] = None,
                              bic_protect_pvalue: Optional[float] = None,          # a p-value veto disarms pruning
                              bic_penalty_scale: float = 1.0,
                              bic_max_degree: Optional[int] = 3,
                              bic_min_collinearity_to_prune: Optional[float] = None,
                              bic_rescue: bool = True,
                              bic_rescue_max_collinearity: float = 0.3,
                              bic_rescue_min_robustness: float = 0.16,
                              auto_detect_topology: bool = True,
                              n_jobs: int = 1,
                              verbose: bool = True) -> tuple:
    """
    Complete Phase 2: Structural Graph Optimization.

    Parameters
    ----------
    results : pd.DataFrame
        Phase 1 output (run_pipeline / build_results), already with the
        columns source_entity, source_property, target_entity, target_property, lag,
        causal_effect, standard_error, pvalue, strength, ...
    processed : dict
        {"entity__property": np.ndarray} — the SAME preprocessed series
        used by Phase 1 (output of Pipeline/preprocessing.py::preprocess_all).
        Reusing them (instead of recomputing TE/DML) is deliberate.
    max_iter : int
        Maximum number of cycles (discovery -> optimization ->
        re-estimation -> new optimization) described in the document.
    method : str
        'ard' (default): parent competition via structural bootstrap
            with ARDRegression (section 2) — assumes an approx. linear
            relationship.
        'cmi': parent competition via conditional independence test
            with k-NN Conditional Mutual Information (section 2b) — does
            not assume a functional form, more computationally
            expensive.
    n_boot : int
        [method='ard'] Structural Bootstrap resamples (item 5) per
        target/iteration.
    stability_threshold : float
        Minimum fraction of resamples in which an edge must survive to
        be kept (0.6 = must appear in >=60% of the bootstraps).
    use_bic_pruning : bool
        If True, also applies global BIC pruning (item 6) after the
        stability competition, per target. Only makes sense together
        with method='ard' (BIC uses the same linear conditional OLS);
        when method='cmi', BIC pruning remains available but is
        independent of the CMI survival decision (runs afterward, on
        whoever survived the CMI competition).
    cmi_k : int
        [method='cmi'] Number of neighbors (k) for the k-NN CMI
        estimator.
    cmi_n_perm : int
        [method='cmi'] Number of resamples for the CMI significance
        shuffle test (equivalent to the 'ard' method's n_boot).
    cmi_alpha : float
        [method='cmi'] Significance level: an edge survives if the CMI
        p-value <= cmi_alpha.
    cmi_n_neighbors_shuffle : int
        [method='cmi'] Number of neighbors (in the conditioning set Z's
        space) used for the local permutation in the shuffle test.
    cmi_max_cond : int | None
        [method='cmi'] MAXIMUM dimension of the conditioning set Z in the
        test I(parent ; target | Z). None = conditions on all other
        candidates. The k-NN CMI estimator quickly loses
        power with dim(Z): in targets with many candidates, TRUE edges
        would get p > alpha and be removed. With an integer here, only
        the `cmi_max_cond` competitors most correlated with the target
        enter Z -- the only ones capable of explaining the tested
        association (same idea as PCMCI's PC1, which also limits the
        conditioning set). Default 4.
    cmi_linear_rescue_alpha : float
        [method='cmi'] If > 0, an edge rejected by CMI is kept when the
        conditional OLS-HAC (already computed, zero extra cost,
        conditions on ALL candidates) gives p <= this value AND the
        conditional coefficient's sign matches the Phase 1 effect.
        Compensates for the non-parametric test's low power with
        moderate n. 0 turns it off. Default 0.01.
    reciprocal_protect_pvalue : float | None
        (2c) Does not remove the dominated direction of a reciprocal pair
        if it itself has strong conditional evidence (conditional_pvalue
        or cmi_pvalue <= this value). Protects genuine bidirectional
        coupling (physical rings/chains, where A->B and B->A are both
        true). None (default) turns off the veto.
    reciprocal_max_lag_diff : int | None
        (2c) If set, only compares two directions whose lags differ by at
        most this value. None = always compares.
    sign_min_ratio : float
        (2d) Only removes for sign inversion if |conditional_theta| >=
        sign_min_ratio * |causal_effect|. Inversions with negligible
        magnitude are a collapse-to-zero with a residual sign (typical of
        collinearity among parents), not evidence of mediation. 0 turns
        it off.
    lone_edge_require_unanimity : bool
        (2e) Requires that ALL available signals vote "weak" (not just
        `lone_edge_min_weak_signals` of them) to reject the lone edge.
        With 4 signals, "2 of 4" alone rejects even with the other 2
        indicating strong evidence. Default False.
    lone_edge_veto_bh : bool
        (2e) If True, an edge with significant_bh == True is never
        removed by the lone-edge gate: a formal hypothesis test takes
        precedence over the cascade robustness heuristics. Default False.
    lone_edge_veto_pvalue : float | None
        (2e) Same, for conditional_pvalue (or Phase 1's pvalue, if the
        conditional one doesn't exist) <= this value. None (default)
        turns it off; has no effect on lone edges (see
        _apply_lone_edge_quality_gate).
    bic_protect_pvalue : float | None
        (item 6) Parents with conditional_pvalue <= this value stay OUT
        of BIC backward elimination. A parent whose effect, already
        conditioned on all other candidates, is significant is not
        redundant by definition -- if BIC removes it, it is due to
        estimate instability, not information overlap. None (default)
        turns it off.
    bic_penalty_scale : float
        (item 6) Multiplier of BIC's k*log(n) term. 1.0 = classic BIC;
        < 1 prunes less (useful in dense graphs). Default 1.0.
    resolve_reciprocal : bool
        (Step 2c) If True, when the graph contains both directions of the
        same pair (A->B and B->A), decides the orientation by asymmetry
        (transfer_entropy, strength, cmi_value) and removes the dominated
        direction. Exists because conditional independence tests
        (ARD/CMI/BIC) are SYMMETRIC and can never remove the reverse edge
        of a true relationship. Turn it off if the domain has genuine
        physical feedback.
    reciprocal_min_margin : float
        Minimum relative margin between the two directions for a
        criterion to vote (below that it abstains). Higher = more
        conservative.
    reciprocal_min_criteria : int
        Minimum number of criteria that must vote for the same direction,
        with no opposing vote, for the dominated direction to be removed.
    sign_consistency : bool
        (Step 2d) If True, removes edges whose conditional_theta flips
        sign relative to Phase 1's causal_effect with significance
        (conditional_pvalue <= sign_alpha) — the signature of an
        association induced by other parents (mediation/confounding/
        collider), not of a direct effect. Zero computational cost.
    sign_alpha : float
        Significance level required to consider the sign flip reliable.
        Smaller = more conservative.
    lone_edge_gate : bool
        (Step 2e) If True, re-evaluates edges that are the ONLY
        significant candidate of their target (parent competition never
        acts on them, by definition) against robustness signals that
        Phase 1 has already computed but does not use to decide
        significance (cascade_confidence, robustness_value, cv_ci,
        significant_bh). Only removes when >=
        lone_edge_min_weak_signals AVAILABLE signals agree that the
        evidence is weak; missing signals abstain (fail-safe).
    lone_edge_robustness_floor : float
        robustness_value floor below which the signal votes "weak".
    lone_edge_cv_ci_ceiling : float
        cv_ci ceiling (CI width / |effect|) above which the signal votes
        "weak".
    lone_edge_min_weak_signals : int
        How many of the available signals need to vote "weak" for the
        lone edge to be removed.
    lone_edge_dry_run : bool
        If True, step 2e does NOT remove anything -- only audits, filling
        in 'lone_edge_gate_would_fail'/'lone_edge_gate_reason' with what
        would have been removed. Use to validate the thresholds on a new
        dataset before trusting the actual removal.
    exact_mediation_gate : bool
        (Step 2f) OFF BY DEFAULT. If True, removes direct edges A->C when
        there is a path A->B->C in the current graph whose lag sum
        matches EXACTLY (not +-1) the direct lag. Risky in systems with
        coupled propagation (chain/ring), where the direct edge and the
        2-hop path can genuinely coexist as real, independent effects --
        an exact lag match does NOT imply redundancy in that kind of
        topology. Only enable it if the domain is sparse and without
        chain propagation among candidates (see the comment on the
        _apply_exact_mediation_gate function).
    bic_rescue : bool
        (Step 2g) ON BY DEFAULT. If True, re-evaluates every edge
        removed by `global_bic_pruning` and returns it to the graph when
        (a) the correlation between its lagged series and that of ALL
        parents that survived on the same target is low (<
        bic_rescue_max_collinearity -- i.e., it genuinely does not
        overlap with what remained, so the pruning was not due to real
        redundancy) AND (b) Phase 1's individual evidence for that edge
        is still strong (high cascade_confidence, robustness_value above
        bic_rescue_min_robustness, significant_bh True -- requires all
        available signals, does not rescue if none are available). Exists
        because BIC decides by the model's JOINT fit, which can prune a
        real, individually strong edge that is statistically unstable
        (large standard error / small sample), without it actually being
        redundant with anything that remained.
    bic_rescue_max_collinearity : float
        Maximum correlation (absolute value) tolerated with any surviving
        parent of the same target to consider the candidate "not
        collinear" and eligible for rescue.
    bic_rescue_min_robustness : float
        robustness_value floor required (when the column exists) to
        rescue.
    verbose : bool
        Prints progress cycle by cycle.
    auto_detect_topology : bool
        If True, BEFORE anything else, computes the average competitor
        degree per target (see detect_topology_and_adjust_parameters) and
        OVERRIDES use_bic_pruning, reciprocal_min_margin,
        reciprocal_min_criteria, reciprocal_protect_pvalue,
        reciprocal_allow_unanimity_override and cmi_linear_rescue_alpha
        according to the detected regime (dense -> BIC pruning off +
        more conservative reciprocal resolution + more permissive
        genuine-bidirectional veto + unanimity override OFF + more
        permissive linear rescue; sparse -> the default values
        + unanimity override ON + stricter linear rescue). Any
        explicit value passed for these 6 parameters is ignored when
        auto_detect_topology=True (default) -- for manual control, set it
        False and pass the 6 parameters yourself. Does NOT use ground
        truth.
    n_jobs : int
        Number of processes to parallelize the parent competition across
        targets WITHIN each cycle (each target is independent of the
        others in the same cycle). 1 (default) = sequential. -1 = uses
        all cores. Requires joblib to be
        installed; without it, falls back to sequential even with
        n_jobs != 1.

    Returns
    -------
    (results_phase2, log) : tuple
        results_phase2 : pd.DataFrame with the SAME columns as `results`
            plus: stability_score, conditional_theta, conditional_se,
            conditional_pvalue, alternative_path, phase2_removal_reason —
            containing only the edges that survived.
        log : dict with the iterative cycle's history (for auditing):
            {'n_edges_per_iter': [...], 'removed_per_iter': [...],
             'converged_at': int}
    """
    if results is None or results.empty:
        return results, {'n_edges_per_iter': [], 'removed_per_iter': [], 'converged_at': 0}

    if auto_detect_topology:
        _topology = detect_topology_and_adjust_parameters(results, verbose=verbose)
        use_bic_pruning = _topology['use_bic_pruning']
        reciprocal_min_margin = _topology['reciprocal_min_margin']
        reciprocal_min_criteria = _topology['reciprocal_min_criteria']
        reciprocal_protect_pvalue = _topology['reciprocal_protect_pvalue']
        reciprocal_allow_unanimity_override = _topology['reciprocal_allow_unanimity_override']
        cmi_linear_rescue_alpha = _topology['cmi_linear_rescue_alpha']
        topology_log = {
            'average_competitor_degree': _topology['_avg_competitor_degree'],
            'detected_regime': _topology['_detected_regime'],
        }
    else:
        topology_log = None

    graph = results.copy().reset_index(drop=True)
    graph['_src_key'] = graph['source_entity'] + '__' + graph['source_property']
    graph['_tgt_key'] = graph['target_entity'] + '__' + graph['target_property']
    graph['phase2_removal_reason'] = ''

    log = {'n_edges_per_iter': [len(graph)], 'removed_per_iter': [], 'converged_at': None}

    if verbose:
        print("=" * 65)
        print("PHASE 2 — STRUCTURAL GRAPH OPTIMIZATION")
        print("=" * 65)
        print(f"  Input edges (Phase 1): {len(graph)}")

    for it in range(1, max_iter + 1):
        if verbose:
            print(f"\n  [cycle {it}/{max_iter}] parent competition per target...")

        targets = graph['_tgt_key'].unique().tolist()
        graph['survives_phase2'] = True
        pieces = _run_targets_structural(
            targets, graph, processed,
            method=method, n_boot=n_boot, stability_threshold=stability_threshold,
            hac_lags=hac_lags, cmi_k=cmi_k, cmi_n_perm=cmi_n_perm, cmi_alpha=cmi_alpha,
            cmi_n_neighbors_shuffle=cmi_n_neighbors_shuffle, cmi_max_cond=cmi_max_cond,
            cmi_linear_rescue_alpha=cmi_linear_rescue_alpha,
            use_bic_pruning=use_bic_pruning, bic_protect_pvalue=bic_protect_pvalue,
            bic_penalty_scale=bic_penalty_scale, bic_max_degree=bic_max_degree,
            bic_min_collinearity_to_prune=bic_min_collinearity_to_prune,
            bic_rescue=bic_rescue, bic_rescue_max_collinearity=bic_rescue_max_collinearity,
            bic_rescue_min_robustness=bic_rescue_min_robustness, verbose=verbose,
            n_jobs=n_jobs, cmi_block=cmi_block,
        )
        if not pieces:
            # Nothing to compete (the graph became empty in an earlier
            # cycle) -- avoid calling pd.concat on an empty list of pieces.
            break
        graph = pd.concat(pieces, axis=0).sort_index()

        # ── Steps 2c and 2d ─────────────────────────────────────
        # Run AFTER the parent competition + BIC pruning, on the columns
        # they have already filled in (conditional_theta, cmi_value).
        # They can only mark survives_phase2 = False; the existing
        # iterative loop takes care of re-competing with the already
        # reduced parent sets in the next cycle and of detecting
        # convergence.
        if method == 'cmi':
            _warn_cmi_pvalue_floor(graph, cmi_n_perm, verbose=verbose)
        if resolve_reciprocal:
            graph = _resolve_reciprocal_edges(
                graph, min_margin=reciprocal_min_margin,
                min_criteria=reciprocal_min_criteria,
                max_lag_diff=reciprocal_max_lag_diff,
                protect_pvalue=reciprocal_protect_pvalue,
                max_degree=reciprocal_max_degree,
                only_lag_zero=reciprocal_only_lag_zero,
                allow_unanimity_override=reciprocal_allow_unanimity_override,
                verbose=verbose,
            )
        if sign_consistency:
            graph = _apply_sign_consistency(graph, alpha=sign_alpha,
                                            min_ratio=sign_min_ratio, verbose=verbose)
        if lone_edge_gate:
            graph = _apply_lone_edge_quality_gate(
                graph, robustness_floor=lone_edge_robustness_floor,
                cv_ci_ceiling=lone_edge_cv_ci_ceiling,
                min_weak_signals=lone_edge_min_weak_signals,
                require_unanimity=lone_edge_require_unanimity,
                majority_fallback=lone_edge_majority_fallback,
                veto_bh=lone_edge_veto_bh,
                veto_pvalue=lone_edge_veto_pvalue,
                dry_run=lone_edge_dry_run, verbose=verbose,
            )
        if lag_max_significant is not None:
            graph = _apply_lag_ceiling_gate(graph, lag_max_significant, verbose=verbose)
        if exact_mediation_gate:
            graph = _apply_exact_mediation_gate(graph, verbose=verbose)

        n_before = log['n_edges_per_iter'][-1]
        survivors = graph[graph['survives_phase2']].copy()
        n_after = len(survivors)
        removed = n_before - n_after

        log['n_edges_per_iter'].append(n_after)
        log['removed_per_iter'].append(removed)

        if verbose:
            print(f"    edges before: {n_before} | after: {n_after} | removed: {removed}")

        if removed == 0:
            log['converged_at'] = it
            graph = survivors
            if verbose:
                print(f"\n  Converged at cycle {it} (no further removal).")
            break

        # Re-estimates the next cycle only over the surviving edges
        graph = survivors.drop(columns=['survives_phase2'], errors='ignore')
        graph['survives_phase2'] = True
    else:
        log['converged_at'] = None
        graph = graph[graph['survives_phase2']].copy()
        if verbose:
            print(f"\n  Reached max_iter={max_iter} without full convergence.")

    graph = _explain_by_transitive_path(graph.assign(survives_phase2=True))

    # Replaces the original (Phase 1) causal_effect/standard_error/pvalue
    # with the conditional (Phase 2) values when available, preserving the
    # originals in *_phase1 columns for auditing.
    graph['causal_effect_phase1'] = graph['causal_effect']
    graph['standard_error_phase1'] = graph['standard_error']
    graph['pvalue_phase1'] = graph['pvalue']
    cond_ok = graph['conditional_theta'].notna()
    graph.loc[cond_ok, 'causal_effect'] = graph.loc[cond_ok, 'conditional_theta']
    graph.loc[cond_ok, 'standard_error'] = graph.loc[cond_ok, 'conditional_se']
    graph.loc[cond_ok, 'pvalue'] = graph.loc[cond_ok, 'conditional_pvalue']
    graph['strength'] = graph['causal_effect'].abs()

    graph = (graph.drop(columns=['_src_key', '_tgt_key', 'survives_phase2'], errors='ignore')
                   .sort_values('strength', ascending=False)
                   .reset_index(drop=True))

    if verbose:
        total_in = log['n_edges_per_iter'][0]
        total_out = len(graph)
        pct = 100 * (1 - total_out / total_in) if total_in else 0.0
        print(f"\n  Phase 2 result: {total_out}/{total_in} edges kept "
              f"({pct:.1f}% removed for structural redundancy).")

    if topology_log is not None:
        log['auto_topology'] = topology_log

    return graph, log


# =============================================================================
# 5b. "FILTER" MODE — meant to be plugged directly inside run_pipeline()
# =============================================================================

def apply_structural_filter(results: pd.DataFrame,
                             processed: dict,
                             max_iter: int = 5,
                             method: str = 'cmi',
                             n_boot: int = 100,
                             stability_threshold: float = 0.6,
                             use_bic_pruning: bool = True,
                             hac_lags: Optional[int] = None,
                             cmi_k: int = 5,
                             cmi_n_perm: int = 200,
                             cmi_alpha: float = 0.05,
                             cmi_n_neighbors_shuffle: int = 5,
                             cmi_max_cond: Optional[int] = 4,
                             cmi_block: bool = False,
                             cmi_linear_rescue_alpha: float = 0.01,
                             resolve_reciprocal: bool = True,
                             reciprocal_min_margin: float = 0.10,
                             reciprocal_min_criteria: int = 2,
                             reciprocal_protect_pvalue: Optional[float] = None,   # a p-value veto disarms the gate
                             reciprocal_max_lag_diff: Optional[int] = None,
                             reciprocal_max_degree: Optional[int] = 6,
                             reciprocal_only_lag_zero: bool = False,
                             reciprocal_allow_unanimity_override: bool = False,
                             sign_consistency: bool = True,
                             sign_alpha: float = 0.05,
                             sign_min_ratio: float = 0.0,
                             lone_edge_gate: bool = True,
                             lone_edge_robustness_floor: float = 0.16,
                             lone_edge_cv_ci_ceiling: float = 1.3,
                             lone_edge_min_weak_signals: int = 2,
                             lone_edge_majority_fallback: bool = False,
                             lone_edge_require_unanimity: bool = False,
                             lone_edge_veto_bh: bool = False,
                             lone_edge_veto_pvalue: Optional[float] = None,  # no effect on lone edges -- see docstring
                             lone_edge_dry_run: bool = False,
                             exact_mediation_gate: bool = False,
                             spurious_hub_gate: bool = False,
                             hub_zscore_threshold: float = 3.5,
                             hub_min_origins: int = 5,
                             lag_max_significant: Optional[int] = None,
                             bic_protect_pvalue: Optional[float] = None,          # a p-value veto disarms pruning
                             bic_penalty_scale: float = 1.0,
                             bic_max_degree: Optional[int] = 3,
                             bic_min_collinearity_to_prune: Optional[float] = None,
                             bic_rescue: bool = True,
                             bic_rescue_max_collinearity: float = 0.3,
                             bic_rescue_min_robustness: float = 0.16,
                             auto_detect_topology: bool = True,
                             n_jobs: int = 1,
                             verbose: bool = True) -> pd.DataFrame:
    """
    Variant of `optimize_graph_structure` designed to be called INSIDE the
    pipeline (run_pipeline), right after the causal-effect penalty
    (Stage 7), acting as an additional filter — not as a reconstruction of
    the DataFrame.

    `method` ('ard' or 'cmi') and the cmi_* parameters have the same
    meaning as in `optimize_graph_structure` (see the docstring there).

    Differences from optimize_graph_structure():
        - Does NOT remove rows from the DataFrame. The input `results` is
          returned with the SAME number of rows and the SAME order.
        - Does NOT overwrite causal_effect/standard_error/pvalue with the
          conditional values — these are made available in new columns
          (conditional_theta, conditional_se, conditional_pvalue,
          stability_score) for auditing/diagnostics only.
        - Only evaluates rows with significant == True (and these
          compete against each other, grouped by target variable). Rows
          that Phase 1 already marked as non-significant are ignored by
          this filter (they remain as they were, without participating in
          the parent competition and without receiving any rejection
          marker).
        - For the candidate rows (significant == True) that do NOT
          survive the structural competition (stability + BIC pruning),
          sets:
              significant            = False
              rejected_phase2_structural = True
          The remaining surviving candidate rows get
          rejected_phase2_structural = False.

    resolve_reciprocal : bool
        (Step 2c) If True, when the graph contains both directions of the
        same pair (A->B and B->A), decides the orientation by asymmetry
        (transfer_entropy, strength, cmi_value) and removes the dominated
        direction. Exists because conditional independence tests
        (ARD/CMI/BIC) are SYMMETRIC and can never remove the reverse edge
        of a true relationship. Turn it off if the domain has genuine
        physical feedback.
    reciprocal_min_margin : float
        Minimum relative margin between the two directions for a
        criterion to vote (below that it abstains). Higher = more
        conservative.
    reciprocal_min_criteria : int
        Minimum number of criteria that must vote for the same direction,
        with no opposing vote, for the dominated direction to be removed.
    sign_consistency : bool
        (Step 2d) If True, removes edges whose conditional_theta flips
        sign relative to Phase 1's causal_effect with significance
        (conditional_pvalue <= sign_alpha) — the signature of an
        association induced by other parents (mediation/confounding/
        collider), not of a direct effect. Zero computational cost.
    sign_alpha : float
        Significance level required to consider the sign flip reliable.
        Smaller = more conservative.

    cmi_max_cond : int | None
        [method='cmi'] MAXIMUM dimension of the conditioning set Z in the
        test I(parent ; target | Z). None = conditions on all other
        candidates. The k-NN CMI estimator quickly loses
        power with dim(Z): in targets with many candidates, TRUE edges
        would get p > alpha and be removed. With an integer here, only
        the `cmi_max_cond` competitors most correlated with the target
        enter Z -- the only ones capable of explaining the tested
        association (same idea as PCMCI's PC1, which also limits the
        conditioning set). Default 4.
    cmi_linear_rescue_alpha : float
        [method='cmi'] If > 0, an edge rejected by CMI is kept when the
        conditional OLS-HAC (already computed, zero extra cost,
        conditions on ALL candidates) gives p <= this value AND the
        conditional coefficient's sign matches the Phase 1 effect.
        Compensates for the non-parametric test's low power with
        moderate n. 0 turns it off. Default 0.01.
    reciprocal_protect_pvalue : float | None
        (2c) Does not remove the dominated direction of a reciprocal pair
        if it itself has strong conditional evidence (conditional_pvalue
        or cmi_pvalue <= this value). Protects genuine bidirectional
        coupling (physical rings/chains, where A->B and B->A are both
        true). None (default) turns off the veto.
    reciprocal_max_lag_diff : int | None
        (2c) If set, only compares two directions whose lags differ by at
        most this value. None = always compares.
    sign_min_ratio : float
        (2d) Only removes for sign inversion if |conditional_theta| >=
        sign_min_ratio * |causal_effect|. Inversions with negligible
        magnitude are a collapse-to-zero with a residual sign (typical of
        collinearity among parents), not evidence of mediation. 0 turns
        it off.
    lone_edge_require_unanimity : bool
        (2e) Requires that ALL available signals vote "weak" (not just
        `lone_edge_min_weak_signals` of them) to reject the lone edge.
        With 4 signals, "2 of 4" alone rejects even with the other 2
        indicating strong evidence. Default False.
    lone_edge_veto_bh : bool
        (2e) If True, an edge with significant_bh == True is never
        removed by the lone-edge gate: a formal hypothesis test takes
        precedence over the cascade robustness heuristics. Default False.
    lone_edge_veto_pvalue : float | None
        (2e) Same, for conditional_pvalue (or Phase 1's pvalue, if the
        conditional one doesn't exist) <= this value. None (default)
        turns it off; has no effect on lone edges (see
        _apply_lone_edge_quality_gate).
    bic_protect_pvalue : float | None
        (item 6) Parents with conditional_pvalue <= this value stay OUT
        of BIC backward elimination. A parent whose effect, already
        conditioned on all other candidates, is significant is not
        redundant by definition -- if BIC removes it, it is due to
        estimate instability, not information overlap. None (default)
        turns it off.
    bic_penalty_scale : float
        (item 6) Multiplier of BIC's k*log(n) term. 1.0 = classic BIC;
        < 1 prunes less (useful in dense graphs). Default 1.0.

    lone_edge_gate : bool
        (Step 2e) If True, re-evaluates edges that are the ONLY
        significant candidate of their target (parent competition never
        acts on them, by definition) against robustness signals that
        Phase 1 has already computed but does not use to decide
        significance (cascade_confidence, robustness_value, cv_ci,
        significant_bh). Only removes when >=
        lone_edge_min_weak_signals AVAILABLE signals agree that the
        evidence is weak; missing signals abstain (fail-safe).
    lone_edge_robustness_floor : float
        robustness_value floor below which the signal votes "weak".
    lone_edge_cv_ci_ceiling : float
        cv_ci ceiling (CI width / |effect|) above which the signal votes
        "weak".
    lone_edge_min_weak_signals : int
        How many of the available signals need to vote "weak" for the
        lone edge to be removed.
    lone_edge_dry_run : bool
        If True, step 2e does NOT remove anything -- only audits, filling
        in 'lone_edge_gate_would_fail'/'lone_edge_gate_reason' with what
        would have been removed. Use to validate the thresholds on a new
        dataset before trusting the actual removal.
    exact_mediation_gate : bool
        (Step 2f) OFF BY DEFAULT. If True, removes direct edges A->C when
        there is a path A->B->C in the current graph whose lag sum
        matches EXACTLY (not +-1) the direct lag. Risky in systems with
        coupled propagation (chain/ring), where the direct edge and the
        2-hop path can genuinely coexist as real, independent effects --
        an exact lag match does NOT imply redundancy in that kind of
        topology. Only enable it if the domain is sparse and without
        chain propagation among candidates (see the comment on the
        _apply_exact_mediation_gate function).
    bic_rescue : bool
        (Step 2g) ON BY DEFAULT. If True, re-evaluates every edge
        removed by `global_bic_pruning` and returns it to the graph when
        (a) the correlation between its lagged series and that of ALL
        parents that survived on the same target is low (<
        bic_rescue_max_collinearity -- i.e., it genuinely does not
        overlap with what remained, so the pruning was not due to real
        redundancy) AND (b) Phase 1's individual evidence for that edge
        is still strong (high cascade_confidence, robustness_value above
        bic_rescue_min_robustness, significant_bh True -- requires all
        available signals, does not rescue if none are available). Exists
        because BIC decides by the model's JOINT fit, which can prune a
        real, individually strong edge that is statistically unstable
        (large standard error / small sample), without it actually being
        redundant with anything that remained.
    bic_rescue_max_collinearity : float
        Maximum correlation (absolute value) tolerated with any surviving
        parent of the same target to consider the candidate "not
        collinear" and eligible for rescue.
    bic_rescue_min_robustness : float
        robustness_value floor required (when the column exists) to
        rescue.

    Returns the same `results`, with new columns added:
        stability_score, conditional_theta, conditional_se,
        conditional_pvalue, alternative_structural_path,
        structural_rejection_reason, rejected_phase2_structural
    auto_detect_topology : bool
        If True, BEFORE anything else, computes the average competitor
        degree per target (see detect_topology_and_adjust_parameters,
        same logic as optimize_graph_structure) and OVERRIDES
        use_bic_pruning, reciprocal_min_margin, reciprocal_min_criteria,
        reciprocal_protect_pvalue, reciprocal_allow_unanimity_override
        and cmi_linear_rescue_alpha according to the detected regime. Any
        explicit value passed for these 6 parameters is ignored when
        True. Does NOT use ground truth.
    n_jobs : int
        Number of processes to parallelize the parent competition across
        targets WITHIN each cycle (each target is independent of the
        others in the same cycle). 1 (default) = sequential. -1 = uses
        all cores. Requires joblib to be
        installed; without it, falls back to sequential even with
        n_jobs != 1.
    """
    if auto_detect_topology:
        _topology = detect_topology_and_adjust_parameters(results, verbose=verbose)
        use_bic_pruning = _topology['use_bic_pruning']
        reciprocal_min_margin = _topology['reciprocal_min_margin']
        reciprocal_min_criteria = _topology['reciprocal_min_criteria']
        reciprocal_protect_pvalue = _topology['reciprocal_protect_pvalue']
        reciprocal_allow_unanimity_override = _topology['reciprocal_allow_unanimity_override']
        cmi_linear_rescue_alpha = _topology['cmi_linear_rescue_alpha']

    out = results.copy()

    # New columns, with neutral defaults for ALL rows (including the ones
    # this filter does not evaluate).
    out['stability_score'] = np.nan
    out['conditional_theta'] = np.nan
    out['conditional_se'] = np.nan
    out['conditional_pvalue'] = np.nan
    out['cmi_value'] = np.nan
    out['cmi_pvalue'] = np.nan
    out['alternative_structural_path'] = ''
    out['structural_rejection_reason'] = ''
    out['rejected_phase2_structural'] = False
    out['lone_edge_gate_would_fail'] = False
    out['lone_edge_gate_reason'] = ''
    if auto_detect_topology:
        out['auto_topology_regime'] = _topology['_detected_regime']
        out['auto_topology_average_degree'] = round(_topology['_avg_competitor_degree'], 2)

    if out.empty or 'significant' not in out.columns:
        return out

    cand_mask = out['significant'] == True  # noqa: E712 — explicit and intentional comparison
    if not cand_mask.any():
        if verbose:
            print("\n  [Phase 2 — structural filter] No significant row to evaluate.")
        return out

    graph = out.loc[cand_mask].copy()
    graph['_src_key'] = graph['source_entity'] + '__' + graph['source_property']
    graph['_tgt_key'] = graph['target_entity'] + '__' + graph['target_property']
    graph['phase2_removal_reason'] = ''

    if verbose:
        print(f"\n  [Phase 2 — structural filter] Evaluating {len(graph)} "
              f"significant relationships (of {len(out)} total)...")

    _removed_reasons = {}

    for it in range(1, max_iter + 1):
        targets = graph['_tgt_key'].unique().tolist()
        graph['survives_phase2'] = True
        pieces = _run_targets_structural(
            targets, graph, processed,
            method=method, n_boot=n_boot, stability_threshold=stability_threshold,
            hac_lags=hac_lags, cmi_k=cmi_k, cmi_n_perm=cmi_n_perm, cmi_alpha=cmi_alpha,
            cmi_n_neighbors_shuffle=cmi_n_neighbors_shuffle, cmi_max_cond=cmi_max_cond,
            cmi_linear_rescue_alpha=cmi_linear_rescue_alpha,
            use_bic_pruning=use_bic_pruning, bic_protect_pvalue=bic_protect_pvalue,
            bic_penalty_scale=bic_penalty_scale, bic_max_degree=bic_max_degree,
            bic_min_collinearity_to_prune=bic_min_collinearity_to_prune,
            bic_rescue=bic_rescue, bic_rescue_max_collinearity=bic_rescue_max_collinearity,
            bic_rescue_min_robustness=bic_rescue_min_robustness, verbose=verbose,
            n_jobs=n_jobs, cmi_block=cmi_block,
        )
        if not pieces:
            # Nothing to compete (the graph became empty in an earlier
            # cycle) -- avoid calling pd.concat on an empty list of pieces.
            break
        graph = pd.concat(pieces, axis=0).sort_index()

        # ── Steps 2c and 2d ─────────────────────────────────────
        # Run AFTER the parent competition + BIC pruning, on the columns
        # they have already filled in (conditional_theta, cmi_value).
        # They can only mark survives_phase2 = False; the existing
        # iterative loop takes care of re-competing with the already
        # reduced parent sets in the next cycle and of detecting
        # convergence.
        if method == 'cmi':
            _warn_cmi_pvalue_floor(graph, cmi_n_perm, verbose=verbose)

        if resolve_reciprocal:
            graph = _resolve_reciprocal_edges(
                graph, min_margin=reciprocal_min_margin,
                min_criteria=reciprocal_min_criteria,
                max_lag_diff=reciprocal_max_lag_diff,
                protect_pvalue=reciprocal_protect_pvalue,
                max_degree=reciprocal_max_degree,
                only_lag_zero=reciprocal_only_lag_zero,
                allow_unanimity_override=reciprocal_allow_unanimity_override,
                verbose=verbose,
            )
        if sign_consistency:
            graph = _apply_sign_consistency(graph, alpha=sign_alpha,
                                            min_ratio=sign_min_ratio, verbose=verbose)
        if lone_edge_gate:
            graph = _apply_lone_edge_quality_gate(
                graph, robustness_floor=lone_edge_robustness_floor,
                cv_ci_ceiling=lone_edge_cv_ci_ceiling,
                min_weak_signals=lone_edge_min_weak_signals,
                require_unanimity=lone_edge_require_unanimity,
                majority_fallback=lone_edge_majority_fallback,
                veto_bh=lone_edge_veto_bh,
                veto_pvalue=lone_edge_veto_pvalue,
                dry_run=lone_edge_dry_run, verbose=verbose,
            )
        if lag_max_significant is not None:
            graph = _apply_lag_ceiling_gate(graph, lag_max_significant, verbose=verbose)
        if exact_mediation_gate:
            graph = _apply_exact_mediation_gate(graph, verbose=verbose)
        if spurious_hub_gate:
            graph = _apply_spurious_hub_gate(
                graph, zscore_threshold=hub_zscore_threshold,
                min_origins=hub_min_origins, verbose=verbose,
            )

        survivors = graph[graph['survives_phase2']].copy()
        removed = len(graph) - len(survivors)

        if verbose:
            print(f"    cycle {it}/{max_iter}: {len(graph)} -> {len(survivors)} "
                  f"({removed} removed)")

        for _i in graph.index.difference(survivors.index):
            _removed_reasons[_i] = graph.at[_i, 'phase2_removal_reason']

        if removed == 0:
            graph = survivors
            break
        graph = survivors.drop(columns=['survives_phase2'], errors='ignore')
        graph['survives_phase2'] = True

    graph = graph.assign(survives_phase2=True)
    graph = _explain_by_transitive_path(graph)

    # `graph` only covers the candidate rows (significant == True on
    # input). Any index from `cand_mask` that no longer appears in `graph`
    # was rejected by the structural filter.
    idx_survivors = set(graph.index)
    idx_candidates = set(out.index[cand_mask])
    idx_rejected = idx_candidates - idx_survivors

    for idx in idx_survivors:
        out.at[idx, 'stability_score'] = graph.at[idx, 'stability_score']
        out.at[idx, 'conditional_theta'] = graph.at[idx, 'conditional_theta']
        out.at[idx, 'conditional_se'] = graph.at[idx, 'conditional_se']
        out.at[idx, 'conditional_pvalue'] = graph.at[idx, 'conditional_pvalue']
        out.at[idx, 'alternative_structural_path'] = graph.at[idx, 'alternative_path']
        out.at[idx, 'rejected_phase2_structural'] = False
        if 'cmi_value' in graph.columns:
            out.at[idx, 'cmi_value'] = graph.at[idx, 'cmi_value']
            out.at[idx, 'cmi_pvalue'] = graph.at[idx, 'cmi_pvalue']
        if 'lone_edge_gate_would_fail' in graph.columns:
            out.at[idx, 'lone_edge_gate_would_fail'] = graph.at[idx, 'lone_edge_gate_would_fail']
            out.at[idx, 'lone_edge_gate_reason'] = graph.at[idx, 'lone_edge_gate_reason']

    for idx in idx_rejected:
        out.at[idx, 'significant'] = False
        out.at[idx, 'rejected_phase2_structural'] = True
        reason = ''
        if idx in _removed_reasons:
            reason = str(_removed_reasons.get(idx) or '')
        out.at[idx, 'structural_rejection_reason'] = reason or (
            'structural_redundancy (did not survive the parent competition '
            '+ stability bootstrap / BIC pruning)'
        )

    if verbose:
        print(f"\n  [Phase 2 — structural filter] {len(idx_rejected)} of "
              f"{len(idx_candidates)} significant relationships were "
              f"marked significant=False due to structural redundancy.")

    return out


# =============================================================================
# 5c. BASE PRESET — Phase 2 with every refinement turned off
# =============================================================================
#
# The defaults prioritize RECALL (fewer FN) while keeping the gates on.
# To run Phase 2 without its refinements (e.g., for a side-by-side
# ablation in the benchmark runner), just unpack this dict:
#
#     from Pipeline.structural import PRESET_PHASE2_BASE
#     run_pipeline(df, ..., **{f'structural_{k}': v
#                              for k, v in PRESET_PHASE2_BASE.items()})
#
# Each key below turns OFF ONE gate/pruning refinement, so an ablation can
# isolate its individual contribution.

PRESET_PHASE2_BASE = {
    'cmi_max_cond': None,             # Z = all competitors
    'cmi_linear_rescue_alpha': 0.0,   # no linear rescue
    'reciprocal_min_margin': 0.10,
    'reciprocal_min_criteria': 2,
    'reciprocal_protect_pvalue': None,
    'reciprocal_max_degree': None,     # resolves reciprocals at any density
    'sign_alpha': 0.05,
    'sign_min_ratio': 0.0,
    'lone_edge_require_unanimity': False,
    'lone_edge_majority_fallback': False,
    'lone_edge_veto_bh': False,
    'lone_edge_veto_pvalue': None,
    'bic_protect_pvalue': None,
    'bic_penalty_scale': 1.0,
    'bic_max_degree': None,           # BIC pruning at any density
    'bic_min_collinearity_to_prune': None,  # always removes, without checking collinearity
    'lag_max_significant': None,    # no lag ceiling
    'bic_rescue': False,
}
