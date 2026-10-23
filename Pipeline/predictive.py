"""
predictive.py — Stage 8a-bis (optional): predictive parent competition.

Stages 6-8a test each candidate pair conditionally on the others. In a
near-deterministic system (dynamics with little or no process noise) that
fails in both directions: every series is a function of the
others, so a conditional test has little outcome variation left to attribute
to a true parent, while indirect neighbours (two steps of the dynamics inside
one sampling interval) remain statistically significant.

In such a system the parents are better read from a multivariate predictive
model than from pairwise tests. For each target:

    1. Model. The target at t is predicted from lags 1..L of itself and of
       every screened candidate source (the Stage 3-5 pairs), with the
       pipeline's own flexible learners (random forest, kernel ridge --
       dml_engine._make_nuisance_model); the learner and L (out of `lags`)
       are the ones with the best out-of-fold R2, contiguous folds.
    2. Importance of a source = relative increase of the out-of-fold squared
       error when ALL its lags are permuted within the test fold (grouped
       permutation importance, Breiman 2001) -- a nonlinear, multivariate
       Granger-type measure.
    3. Significance. One-sided paired t-test on block means of the loss
       differential (Diebold & Mariano, 1995, with non-overlapping blocks for
       the serial dependence), Benjamini-Hochberg over all candidates.

Significance alone does not separate parents from indirect neighbours here
(the latter are real, only weaker), so the decision is a RELATIVE one:

    4. A one- and a two-component Gaussian mixture are fitted to the
       log-importance of the significant candidates. The competition is
       APPLIED only if the importances form two populations -- the
       two-component model is preferred with BIC difference > `min_delta_bic`
       (6 = "strong" evidence, Kass & Raftery 1995) and both components hold
       at least two pairs. The pairs of the upper component are the parents;
       every other lagged candidate is rejected.
    5. Otherwise the competition abstains and the pipeline's decision stands.

Nothing here uses ground truth or the dataset's identity; with few
candidates the mixture comparison has no power and the stage abstains.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:  # pragma: no cover
    HAS_JOBLIB = False

try:
    from sklearn.mixture import GaussianMixture
    from statsmodels.stats.multitest import multipletests
    HAS_DEPS = True
except ImportError:  # pragma: no cover
    HAS_DEPS = False

from .dml_engine import _make_nuisance_model

_LEARNERS = ('random_forest', 'kernel_ridge')


def _lag_block(x: np.ndarray, L: int) -> np.ndarray:
    cols = []
    for l in range(1, L + 1):
        c = np.full(len(x), np.nan)
        c[l:] = x[:-l]
        cols.append(c)
    return np.column_stack(cols)


def _fit_target(y: np.ndarray, sources: list, L: int, n_splits: int) -> Optional[dict]:
    """Best learner (by out-of-fold R2) for one target and one lag window."""
    blocks = [_lag_block(y, L)] + [_lag_block(x, L) for x in sources]
    Z = np.column_stack(blocks)
    ok = ~np.isnan(y) & ~np.any(np.isnan(Z), axis=1)
    Z, Y = Z[ok], y[ok]
    n = len(Y)
    if n < max(30, 4 * n_splits) or float(np.var(Y)) < 1e-12:
        return None
    size = n // n_splits
    folds = []
    for f in range(n_splits):
        test = np.arange(f * size, (f + 1) * size if f < n_splits - 1 else n)
        gap = np.arange(max(0, test[0] - L), min(n, test[-1] + 1 + L))
        folds.append((np.setdiff1d(np.arange(n), gap), test))
    best = None
    for name in _LEARNERS:
        models, err = [], np.empty(n)
        try:
            for train, test in folds:
                m = _make_nuisance_model(name, len(train)).fit(Z[train], Y[train])
                models.append(m)
                err[test] = (Y[test] - m.predict(Z[test])) ** 2
        except Exception:
            continue
        r2 = 1.0 - float(err.mean()) / float(np.var(Y))
        if best is None or r2 > best['r2']:
            best = {'r2': r2, 'learner': name, 'models': models, 'err': err}
    if best is None:
        return None
    best.update(Z=Z, Y=Y, folds=folds, L=L)
    return best


def _compete_target(args: tuple) -> list:
    """Importance and p-value of every candidate source of one target."""
    target, y, src_keys, src_series, lags, n_splits, n_rep, block = args
    fits = [f for f in (_fit_target(y, src_series, L, n_splits) for L in lags) if f is not None]
    if not fits:
        return []
    # best out-of-fold R2; ties (to 3 decimals) go to the shorter window
    fit = max(fits, key=lambda f: (round(f['r2'], 3), -f['L']))
    Z, Y, L, err = fit['Z'], fit['Y'], fit['L'], fit['err']
    n = len(Y)
    rng = np.random.RandomState(0)
    out = []
    for i, src in enumerate(src_keys):
        cols = list(range((i + 1) * L, (i + 2) * L))
        diff = np.zeros(n)
        for (train, test), model in zip(fit['folds'], fit['models']):
            for _ in range(n_rep):
                Zp = Z[test].copy()
                Zp[:, cols] = Zp[rng.permutation(len(test))][:, cols]
                diff[test] += ((Y[test] - model.predict(Zp)) ** 2 - err[test]) / n_rep
        n_blocks = n // block
        if n_blocks >= 3:
            means = diff[:n_blocks * block].reshape(n_blocks, block).mean(axis=1)
            se = means.std(ddof=1) / np.sqrt(n_blocks)
            pval = float(stats.t.sf(means.mean() / se, n_blocks - 1)) if se > 0 else 1.0
        else:
            pval = 1.0
        out.append({
            'source': src, 'target': target,
            'importance': float(diff.mean() / max(err.mean(), 1e-300)),
            'pvalue': pval, 'r2': fit['r2'], 'lags': L, 'learner': fit['learner'],
        })
    return out


def _two_populations(log_importance: np.ndarray, min_delta_bic: float) -> tuple:
    """(upper-component mask or None, delta BIC) for the log-importances."""
    x = log_importance.reshape(-1, 1)
    one = GaussianMixture(1, random_state=0).fit(x)
    two = GaussianMixture(2, n_init=5, random_state=0).fit(x)
    delta = float(one.bic(x) - two.bic(x))
    upper = two.predict(x) == int(np.argmax(two.means_.ravel()))
    if delta <= min_delta_bic or upper.sum() < 2 or (~upper).sum() < 2:
        return None, delta
    return upper, delta


def predictive_competition(
    results: pd.DataFrame,
    processed: dict,
    lags: Sequence[int] = (1, 2, 3, 5),
    n_splits: int = 5,
    n_rep: int = 10,
    block: int = 10,
    alpha: float = 0.05,
    min_delta_bic: float = 6.0,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple:
    """
    Runs the competition over the lagged candidate pairs of `results`.

    Returns (results, info). `results` gains the columns predictive_importance,
    predictive_pvalue, predictive_lags, predictive_r2; if the competition is
    applied (see the module docstring) `significant` becomes its decision for
    every lagged cross pair and the previous value is kept in
    'significant_before_predictive'. info = {'applied', 'delta_bic',
    'n_candidates', 'n_significant', 'n_parents'}.
    """
    info = {'applied': False, 'delta_bic': float('nan'), 'n_candidates': 0,
            'n_significant': 0, 'n_parents': 0}
    if not HAS_DEPS or results is None or results.empty:
        return results, info

    res = results.copy()
    src = res['source_entity'].astype(str) + '__' + res['source_property'].astype(str)
    tgt = res['target_entity'].astype(str) + '__' + res['target_property'].astype(str)
    lag = pd.to_numeric(res['lag'], errors='coerce')
    cand = (lag >= 1) & (src != tgt) & src.isin(processed.keys()) & tgt.isin(processed.keys())
    info['n_candidates'] = int(cand.sum())
    if not cand.any():
        return results, info

    tasks = []
    for target, idx in res[cand].groupby(tgt[cand]).groups.items():
        keys = sorted(set(src.loc[idx]))
        tasks.append((target, processed[target], keys, [processed[k] for k in keys],
                      tuple(lags), n_splits, n_rep, block))
    if HAS_JOBLIB and n_jobs != 1 and len(tasks) > 1:
        pieces = Parallel(n_jobs=n_jobs)(delayed(_compete_target)(t) for t in tasks)
    else:
        pieces = [_compete_target(t) for t in tasks]
    table = {(r['source'], r['target']): r for piece in pieces for r in piece}
    if not table:
        return results, info

    pair = list(zip(src, tgt))
    get = lambda field: [table[p][field] if (c and p in table) else np.nan
                         for p, c in zip(pair, cand)]
    res['predictive_importance'] = get('importance')
    res['predictive_pvalue'] = get('pvalue')
    res['predictive_lags'] = get('lags')
    res['predictive_r2'] = get('r2')

    keys = list(table)
    pvals = np.array([table[k]['pvalue'] for k in keys])
    imp = np.array([table[k]['importance'] for k in keys])
    sig = multipletests(pvals, alpha=alpha, method='fdr_bh')[0] & (imp > 0)
    info['n_significant'] = int(sig.sum())
    if sig.sum() >= 6:
        upper, delta = _two_populations(np.log(imp[sig]), min_delta_bic)
        info['delta_bic'] = delta
        if upper is not None:
            sig_keys = [k for k, s in zip(keys, sig) if s]
            parents = {k for k, up in zip(sig_keys, upper) if up}
            res['significant_before_predictive'] = res['significant']
            decided = [bool(c) and p in table for p, c in zip(pair, cand)]
            res.loc[decided, 'significant'] = [p in parents for p, d in zip(pair, decided) if d]
            info.update(applied=True, n_parents=len(parents))

    if verbose:
        dbic = 'n/a' if not np.isfinite(info['delta_bic']) else f"{info['delta_bic']:.1f}"
        print(f"  [predictive competition] {info['n_candidates']} candidates, "
              f"{info['n_significant']} with significant importance, "
              f"two-population delta BIC = {dbic} (needs > {min_delta_bic}) -> "
              + (f"APPLIED: {info['n_parents']} parents" if info['applied']
                 else "abstains (the pipeline's decision stands)"))
    return res, info
