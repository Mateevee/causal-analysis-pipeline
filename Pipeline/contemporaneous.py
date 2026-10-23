"""
contemporaneous.py — Stage 8b (optional): contemporaneous (lag-0) links.

Second step of Stage 8, the structural adjustment. 8a (structural.py)
prunes the lagged graph; 8b identifies the lag-0 part of the structural
VAR -- the contemporaneous matrix A0 in A0 y_t = sum_i A_i y_(t-i) + e_t --
from the reduced-form innovations, and appends it only for pairs with no
significant lagged edge left after 8a.

The lag search of Stages 1-8 starts at lag 1, so a relation that acts
within one sampling interval (coarsely sampled aggregates, i.i.d. samples)
is invisible to it by construction. This stage adds
lag-0 links, following the principle of VAR-LiNGAM (Hyvärinen et al.,
JMLR 2010) and of PCMCI+'s contemporaneous step (Runge, UAI 2020):

    1. Innovations. Each series is regressed on the past (lags 1..p) of
       ALL series (ridge, CV-chosen penalty); the out-of-sample residuals
       are the innovations. Whatever dependence is left between two
       innovations cannot come from the past: it is contemporaneous.
    2. Skeleton. Two options (`skeleton`):
       'full' -- partial correlation between each pair of innovations
       given ALL the other innovations (precision matrix), Fisher-z test,
       Benjamini-Hochberg over the N(N-1)/2 pairs. Conditioning on the
       other innovations removes chains (A-B-C does not leave A-C), but
       conditioning on a common EFFECT (collider) of A and B creates a
       spurious A-B dependence.
       'pc' -- PC-stable skeleton search on the innovations (Spirtes,
       Glymour & Scheines 2000), as done on VAR residuals by Swanson &
       Granger (1997) and Demiralp & Hoover (2003): an adjacency is
       removed as soon as SOME conditioning set (of size 0..pc_max_cond,
       drawn from the current neighbours) makes the Fisher-z partial
       correlation test non-significant, so a collider is never forced
       into the conditioning set. Each pair's p-value is the maximum over
       the tests it went through, and Benjamini-Hochberg runs over the
       N(N-1)/2 pairs on those maxima -- the same decision PCMCI+ makes
       on its p_matrix.
    3. Orientation. Time order cannot orient a lag-0 link.
       orientation='both' (pipeline default) reports each adjacency as an
       UNORIENTED link, i.e. in both directions -- the CPDAG convention for
       an undirected edge. orientation='lingam' instead orients it by the
       pairwise likelihood ratio for non-Gaussian SEMs (Hyvärinen & Smith,
       JMLR 2013), tanh approximation:
            R = rho * E[ x*tanh(y) - tanh(x)*y ],  R > 0  =>  x -> y
       on skewness-sign-corrected innovations, keeping only signs stable
       over bootstrap resamples.

GATE (max_past_r2). The stage only runs when the past explains little of
the series (median out-of-sample R2 of the innovation regressions <=
max_past_r2). In a strongly dynamic, nonlinear system the linear ridge on
the past leaves unmodelled
nonlinear dynamics in the residuals, which shows up as spurious lag-0
dependence; when the past explains little (i.i.d. samples, noise-driven
systems) residual dependence can only be contemporaneous.
"""

import itertools
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .discovery import _detect_time_axis

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False


def _innovations(X: np.ndarray, p: int, n_splits: int = 5) -> Optional[np.ndarray]:
    """Out-of-fold residuals of each column of X on lags 1..p of all columns."""
    T, N = X.shape
    if p < 1 or T - p < max(30, 2 * N):
        return None
    lagged = np.column_stack([X[p - l:T - l] for l in range(1, p + 1)])
    Y = X[p:]
    E = np.empty_like(Y)
    cv = KFold(n_splits=n_splits, shuffle=False)
    alphas = np.logspace(-2, 4, 13)
    for j in range(N):
        pred = cross_val_predict(RidgeCV(alphas=alphas), lagged, Y[:, j], cv=cv)
        E[:, j] = Y[:, j] - pred
    return E


def _partial_correlations(E: np.ndarray) -> Optional[np.ndarray]:
    C = np.corrcoef(E, rowvar=False)
    try:
        P = np.linalg.pinv(C)
    except np.linalg.LinAlgError:
        return None
    d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
    R = -P / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -0.999999, 0.999999)


def _pc_skeleton_pvalues(E: np.ndarray, alpha: float, max_cond: int,
                         max_subsets: int) -> np.ndarray:
    """
    PC-stable skeleton search on the innovations E (n x N) with Fisher-z
    partial-correlation tests. Returns, for each pair i < j in
    np.triu_indices order, the MAXIMUM p-value over the conditional
    independence tests the pair went through (a removed pair has one
    above `alpha`). The caller applies Benjamini-Hochberg on these maxima.

    Levels 0..max_cond; at each level the neighbour sets are frozen first
    (PC-stable, order-independent), and at most `max_subsets` conditioning
    sets are tried per pair and level.
    """
    n, N = E.shape
    C = np.corrcoef(E, rowvar=False)
    adj = {i: set(range(N)) - {i} for i in range(N)}
    pmax = np.zeros((N, N))

    def _pval(i, j, S):
        idx = [i, j] + list(S)
        P = np.linalg.pinv(C[np.ix_(idx, idx)])
        r = -P[0, 1] / np.sqrt(max(P[0, 0] * P[1, 1], 1e-12))
        r = float(np.clip(r, -0.999999, 0.999999))
        dof = max(n - len(S) - 3, 1)
        return float(2.0 * stats.norm.sf(abs(np.arctanh(r)) * np.sqrt(dof)))

    for level in range(0, max_cond + 1):
        frozen = {i: set(a) for i, a in adj.items()}
        testable = False
        for i in range(N):
            for j in sorted(frozen[i]):
                if j not in adj[i]:
                    continue
                cand = sorted(frozen[i] - {j})
                if len(cand) < level:
                    continue
                testable = True
                a, b = min(i, j), max(i, j)
                for S in itertools.islice(itertools.combinations(cand, level), max_subsets):
                    q = _pval(i, j, S)
                    pmax[a, b] = max(pmax[a, b], q)
                    if q > alpha:
                        adj[i].discard(j)
                        adj[j].discard(i)
                        break
        if not testable:
            break

    return pmax[np.triu_indices(N, k=1)]


def _lr_tanh(x: np.ndarray, y: np.ndarray) -> float:
    """Hyvärinen & Smith (2013) pairwise LR, tanh approximation. >0 => x->y."""
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    # skewness-sign correction so the measure is not fooled by the sign
    # convention of a skewed variable
    x = x * (np.sign(stats.skew(x)) or 1.0)
    y = y * (np.sign(stats.skew(y)) or 1.0)
    rho = float(np.mean(x * y))
    return rho * float(np.mean(x * np.tanh(y) - np.tanh(x) * y))


# How much of the series the past explains -- read by the regime guards of
# Pipeline/pipeline.py (_stage8_regime_guards, evidence_coherence_fallback).
def past_explained_variance(processed: dict, keys: List[str],
                            max_lag: int) -> Optional[float]:
    """
    Median out-of-sample R2 of each series regressed on the past (lags
    1..p) of ALL series -- the quantity discover_contemporaneous compares
    with `max_past_r2` to decide whether Stage 8b runs, computed the same
    way. Returns None when it cannot be computed (sklearn missing, fewer
    than two series, or a series too short for the number of series).
    """
    if not HAS_SKLEARN or len(keys) < 2:
        return None
    X = np.column_stack([np.asarray(processed[k], dtype=float) for k in keys])
    X = X[~np.any(np.isnan(X), axis=1)]
    T, N = X.shape
    p = int(max(1, min(max_lag, (T // 3) // max(N, 1))))
    E = _innovations(X, p)
    if E is None:
        return None
    return float(np.median(1.0 - E.var(axis=0) / np.maximum(X[p:].var(axis=0), 1e-12)))


def own_past_explained_variance(processed: dict, keys: List[str],
                                max_lag: int) -> Optional[float]:
    """
    Same quantity as past_explained_variance, but each series is regressed
    on its OWN lags 1..p only (same p, same ridge, same folds).

    The past of all series contains each series' own past, so it cannot
    explain LESS of it. When the cross-validated all-series estimate falls
    below this one, the gap is estimation error -- too few points for the
    N*p regressors (short series) -- not absence of dynamics. Used as a
    floor for the past R2 (evidence_fallback_own_past_floor).

    WHY THE TWO ESTIMATES CAN DISAGREE
        In population the prediction-error variance given a larger
        information set is never above the one given a subset of it; that
        comparison is what Granger causality is defined on (Granger, 1969).
        In a finite sample the regression has to be estimated, and with
        estimated coefficients the one-step forecast MSE of a VAR(p) on N
        series is inflated by about (T + N*p + 1) / T under least squares
        (Lütkepohl, 2005, Section 3.5, forecasting with estimated
        processes): the all-series regression pays for N*p coefficients,
        the own-past regression for p. Ridge shrinkage reduces that cost
        but does not remove it; unrestricted VARs are over-parameterized
        when N*p is large relative to T (Bańbura, Giannone & Reichlin,
        2010). Taking the larger of the two out-of-sample R2 therefore
        restores, in the sample, an ordering that holds in population.

        Bańbura, M., Giannone, D. & Reichlin, L. (2010). Large Bayesian
            vector auto regressions. Journal of Applied Econometrics,
            25(1), 71-92.
        Granger, C. W. J. (1969). Investigating causal relations by
            econometric models and cross-spectral methods. Econometrica,
            37(3), 424-438.
        Lütkepohl, H. (2005). New Introduction to Multiple Time Series
            Analysis. Springer, Berlin.
    """
    if not HAS_SKLEARN or len(keys) < 2:
        return None
    X = np.column_stack([np.asarray(processed[k], dtype=float) for k in keys])
    X = X[~np.any(np.isnan(X), axis=1)]
    T, N = X.shape
    p = int(max(1, min(max_lag, (T // 3) // max(N, 1))))
    if p < 1 or T - p < max(30, 2 * N):
        return None
    cv = KFold(n_splits=5, shuffle=False)
    alphas = np.logspace(-2, 4, 13)
    r2 = []
    for j in range(N):
        own = np.column_stack([X[p - l:T - l, j] for l in range(1, p + 1)])
        y = X[p:, j]
        pred = cross_val_predict(RidgeCV(alphas=alphas), own, y, cv=cv)
        r2.append(1.0 - np.var(y - pred) / max(np.var(y), 1e-12))
    return float(np.median(r2))


def _bh(pvals: np.ndarray, alpha: float) -> np.ndarray:
    m = len(pvals)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = pvals[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:k]] = True
    return out


def discover_contemporaneous(
    processed: dict,
    keys: List[str],
    max_lag: int,
    alpha: float = 0.05,
    n_boot: int = 200,
    min_orientation_stability: float = 0.9,
    seed: int = 0,
    orientation: str = 'lingam',
    max_past_r2: Optional[float] = None,
    skeleton: str = 'full',
    pc_max_cond: int = 4,
    pc_max_subsets: int = 300,
    verbose: bool = True,
) -> List[dict]:
    """
    Returns a list of oriented lag-0 links:
        {'source', 'target', 'lag': 0, 'pcorr', 'pvalue', 'pvalue_corrected',
         'orientation_lr', 'orientation_stability'}
    See the module docstring for the method.
    """
    if not HAS_SKLEARN or len(keys) < 2:
        return []
    X = np.column_stack([np.asarray(processed[k], dtype=float) for k in keys])
    X = X[~np.any(np.isnan(X), axis=1)]
    T, N = X.shape
    p = int(max(1, min(max_lag, (T // 3) // max(N, 1))))
    E = _innovations(X, p)
    if E is None:
        if verbose:
            print(f"  [contemporaneous] skipped: T={T} too short for N={N}, p={p}")
        return []
    past_r2 = float(np.median(1.0 - E.var(axis=0) / np.maximum(X[p:].var(axis=0), 1e-12)))
    if max_past_r2 is not None and past_r2 > max_past_r2:
        if verbose:
            print(f"  [contemporaneous] skipped: the past explains median R2={past_r2:.2f} "
                  f"> {max_past_r2} of the series (dynamic system; lag-0 residual "
                  f"dependence would mostly be unmodelled dynamics)")
        return []
    n = E.shape[0]
    dof = n - (N - 2) - 3
    if dof < 10:
        if verbose:
            print(f"  [contemporaneous] skipped: dof={dof} too small")
        return []
    R = _partial_correlations(E)
    if R is None:
        return []

    iu = np.triu_indices(N, k=1)
    r = R[iu]
    if skeleton == 'pc':
        pv = _pc_skeleton_pvalues(E, alpha, pc_max_cond, pc_max_subsets)
    elif skeleton == 'full':
        z = np.arctanh(r) * np.sqrt(dof)
        pv = 2.0 * stats.norm.sf(np.abs(z))
    else:
        raise ValueError(f"contemporaneous: unknown skeleton={skeleton!r} (use 'full' or 'pc')")
    keep = _bh(pv, alpha)
    p_corr = np.minimum(1.0, pv * len(pv) / (np.argsort(np.argsort(pv)) + 1))

    rng = np.random.default_rng(seed)
    links = []
    n_skel = int(keep.sum())
    for idx in np.where(keep)[0]:
        a, b = iu[0][idx], iu[1][idx]
        if orientation == 'both':
            for src, tgt in ((a, b), (b, a)):
                links.append({
                    'source': keys[src], 'target': keys[tgt], 'lag': 0,
                    'pcorr': float(r[idx]), 'pvalue': float(pv[idx]),
                    'pvalue_corrected': float(p_corr[idx]),
                    'orientation_lr': float('nan'), 'orientation_stability': float('nan'),
                })
            continue
        lr = _lr_tanh(E[:, a], E[:, b])
        signs = []
        for _ in range(n_boot):
            s = rng.integers(0, n, n)
            signs.append(np.sign(_lr_tanh(E[s, a], E[s, b])))
        stab = float(np.mean(np.array(signs) == np.sign(lr))) if lr != 0 else 0.0
        if stab < min_orientation_stability:
            continue
        src, tgt = (a, b) if lr > 0 else (b, a)
        links.append({
            'source': keys[src], 'target': keys[tgt], 'lag': 0,
            'pcorr': float(r[idx]), 'pvalue': float(pv[idx]),
            'pvalue_corrected': float(p_corr[idx]),
            'orientation_lr': float(abs(lr)), 'orientation_stability': stab,
        })
    if verbose:
        detail = (f"{len(links)} directed rows (unoriented, both directions)"
                  if orientation == 'both' else
                  f"{len(links)} oriented (bootstrap stability >= {min_orientation_stability})")
        print(f"  [contemporaneous] N={N}, T={T}, innovations on lags 1..{p} "
              f"(past R2 median={past_r2:.2f}), skeleton={skeleton}: {n_skel} lag-0 adjacencies "
              f"(BH alpha={alpha}), {detail}")
    return links


def drop_lagged_without_time_axis(results: pd.DataFrame, processed: dict) -> tuple:
    """
    Without a genuine time axis (i.i.d. rows -- e.g. data sampled row by row
    from a Bayesian network) no lag is real: every lagged edge is an artefact
    of row order. They are dropped here, and the contemporaneous step then
    describes the whole graph. Decided from the data only
    (discovery._detect_time_axis: median |ACF(1)| against the white-noise
    band).

    Returns (results, number of edges dropped -- None when the data has a
    time axis and nothing was touched).
    """
    if _detect_time_axis(processed):
        return results, None
    sig_mask = results['significant'] == True  # noqa: E712
    n_drop = int(sig_mask.sum())
    if 'structural_rejection_reason' not in results.columns:
        results['structural_rejection_reason'] = None
    results.loc[sig_mask, 'structural_rejection_reason'] = 'no_time_axis'
    results.loc[sig_mask, 'significant'] = False
    return results, n_drop


def append_contemporaneous_links(
    results: pd.DataFrame,
    processed: dict,
    keys: List[str],
    dml_engine,
    max_lag: int,
    alpha: float = 0.01,
    max_past_r2: Optional[float] = 0.25,
    orientation: str = 'both',
    skeleton: str = 'full',
    pc_max_cond: int = 4,
    require_dml: bool = False,
    dml_splits: int = 5,
    dml_model: str = 'auto',
    var_res_T_threshold: float = 0.05,
    estimate_selection: Optional[str] = None,
    n_jobs: int = 1,
) -> tuple:
    """
    Runs discover_contemporaneous and APPENDS the lag-0 links to `results`
    (rows with lag=0, filter_method='contemporaneous'), so the calibration
    of Stages 7/8a is untouched. Pairs already significant (either
    direction) at some lag, and intra-entity pairs, are skipped.

    Every appended link gets a DML effect estimate -- the same cascade as
    Stage 6, with the contemporaneous neighbours of the pair in the lag-0
    skeleton as extra controls (dml_engine.estimate_lag0_effects). A link
    whose estimation fails does not leave the pipeline
    (results.require_effect_estimate). The estimate quantifies only; with
    require_dml=True a link is kept only if it is also significant.

    Returns (results, number of links appended).
    """
    links = discover_contemporaneous(
        processed, list(keys), max_lag=max_lag, alpha=alpha,
        max_past_r2=max_past_r2, orientation=orientation,
        skeleton=skeleton, pc_max_cond=pc_max_cond,
    )
    taken = set()
    if not results.empty and 'significant' in results.columns:
        sig = results[results['significant'] == True]  # noqa: E712
        for _, row in sig.iterrows():
            a = f"{row['source_entity']}__{row['source_property']}"
            b = f"{row['target_entity']}__{row['target_property']}"
            taken.add((a, b)); taken.add((b, a))
    new_rows = []
    for l in links:
        if (l['source'], l['target']) in taken:
            continue
        se, sp = l['source'].split('__', 1)
        te, tp = l['target'].split('__', 1)
        if se == te:
            continue
        new_rows.append({
            'source_entity': se, 'source_property': sp,
            'target_entity': te, 'target_property': tp,
            'lag': 0,
            # the skeleton statistics are a DEPENDENCE measure, not an
            # effect: kept in their own columns; causal_effect is filled
            # below by the lag-0 DML estimate
            'causal_effect': np.nan,
            'lag0_partial_corr': l['pcorr'],
            'lag0_skeleton_pvalue': l['pvalue'],
            'lag0_skeleton_pvalue_bh': l['pvalue_corrected'],
            'significant': True, 'filter_method': 'contemporaneous',
            '_src_key': l['source'], '_tgt_key': l['target'],
        })
    if new_rows:
        adj = {}
        for l in links:
            adj.setdefault(l['source'], set()).add(l['target'])
            adj.setdefault(l['target'], set()).add(l['source'])
        est = dml_engine.estimate_lag0_effects(
            [{'source': r['_src_key'], 'target': r['_tgt_key'],
              'neighbors': sorted((adj.get(r['_src_key'], set()) |
                                   adj.get(r['_tgt_key'], set())))}
             for r in new_rows],
            processed, list(keys), max_lag=max_lag, n_splits=dml_splits,
            n_jobs=n_jobs, dml_model=dml_model,
            var_res_T_threshold=var_res_T_threshold,
            estimate_selection=estimate_selection,
        )
        for r, e in zip(new_rows, est):
            r.update({
                'causal_effect': e.get('effect', np.nan),
                'standard_error': e.get('se', np.nan),
                'ci_95_lower': e.get('ci_lower', np.nan),
                'ci_95_upper': e.get('ci_upper', np.nan),
                'pvalue': e.get('pvalue', np.nan),
                'dml_model_used': e.get('dml_model_used'),
                'n_models_agree': e.get('n_models_agree'),
                'cascade_confidence': e.get('cascade_confidence'),
                'lag0_dml_significant': bool(e.get('significant', False)),
            })
            if require_dml:
                r['significant'] = bool(e.get('significant', False))
        n_q = sum(r['lag0_dml_significant'] for r in new_rows)
        print(f"  [Stage 8b] lag-0 DML: {n_q} of {len(new_rows)} directed "
              f"links with a significant effect estimate")
    for r in new_rows:
        r.pop('_src_key', None); r.pop('_tgt_key', None)
    if new_rows:
        results = pd.concat([results, pd.DataFrame(new_rows)], ignore_index=True)
    return results, len(new_rows)
