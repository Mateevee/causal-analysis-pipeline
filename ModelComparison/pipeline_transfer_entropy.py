"""
pipeline_transfer_entropy.py — Transfer Entropy on its own, as a baseline
============================================================================
Exposes run_pipeline_transfer_entropy with the SAME call signature used by
the runner. Reuses _pcmci_load_and_pivot and _correct_pvalues from
causal_common (same preprocessing as the other baselines) and the TE
estimator of the main pipeline (Pipeline/transfer_entropy.py).

The main pipeline uses Transfer Entropy (Schreiber, PRL 2000) as one of its
ingredients (Stage 4). This baseline answers "what does TE do by itself":
the SAME estimator -- TE(X->Y, lag) = I(Y_t ; X_{t-lag} | Y_{t-1}), with the
method/bins the run sets for the pipeline (te_method, te_bins, te_k) -- over
every ordered pair of series, with nothing from the pipeline around it: no
cross-correlation screen before it, no lag penalty, no rescue, no DML, no
structural stage.

For every ordered pair (i -> j), i != j:

    statistic : max over lag = 1..tau_max of TE(X_i -> X_j, lag), every lag
                with the same weight;
    p-value   : permutation test of that MAXIMUM (the source series is
                shuffled, the whole lag scan repeated), so the search over
                lags is already paid for -- the same max-statistic test as
                the pipeline's correct_lag_multiplicity.

Decision: the p-values of ALL ordered pairs are corrected together
(transfer_entropy_correction, default 'fdr_bh') and compared with
transfer_entropy_alpha_level -- the same rule as the PCMCI baseline
(alpha = 0.05 with FDR-BH over every tested hypothesis).

NUMBER OF PERMUTATIONS. A permutation p-value cannot be smaller than
1 / (n_perms + 1), and FDR-BH over m pairs only rejects k hypotheses when
their p-values are <= k * alpha / m. With few permutations a sparse graph
could therefore never be detected, whatever the data (m = 1560 pairs and
alpha = 0.05 at 200 permutations would need more than 150 simultaneous
discoveries). To afford a low floor, the test is sequential (Besag &
Clifford, Biometrika 1991): a pair stops as soon as
transfer_entropy_min_exceed permuted statistics reach the observed one
(p = min_exceed / permutations drawn), so only the pairs that look
significant run the full transfer_entropy_n_perms
(p = (exceedances + 1) / (n_perms + 1)). Both forms are valid p-values.

Like the pipeline's estimator, TE here conditions on ONE lag of the target
and on nothing else: it does not separate a direct link from an indirect
one or from a common driver.

One row per ordered pair: 'value' is the TE (bits) at the best lag, 'lag'
that lag, 'n_perms' the permutations drawn. Lagged-only by definition
(lag >= 1).
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional, List

from Auxiliary.causal_common import _pcmci_load_and_pivot, _correct_pvalues
from Pipeline.penalties import UniformLagPenalty
from Pipeline.transfer_entropy import _te_best_over_lags

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

_UNIFORM_LAG = UniformLagPenalty()


def _qbin(v: np.ndarray, bins: int) -> np.ndarray:
    lo, hi = v.min(), v.max()
    if hi == lo:
        return np.zeros(len(v), dtype=np.int64)
    return np.clip(np.digitize(v, np.linspace(lo, hi, bins + 1)) - 1, 0, bins - 1)


def _entropy(codes: np.ndarray) -> float:
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))


def _te_lag_scanner(y: np.ndarray, tau_max: int, method: str, bins: int, k: int) -> Callable:
    """
    Returns scan(x) -> (max over lags of TE(x -> y, lag), that lag).

    For the histogram estimator this is _te_best_over_lags (uniform lag
    weights) with everything that depends only on the target computed once,
    since the permutation test calls it thousands of times per pair with
    the same y: same binning, same entropies, same result bit for bit.
    Assumes no NaN (guaranteed by _pcmci_load_and_pivot). Other estimators
    go through _te_best_over_lags itself.
    """
    if method != "histogram":
        return lambda x: _te_best_over_lags(x, y, tau_max, method, bins, k, _UNIFORM_LAG)[1:]

    T = len(y)
    per_lag = []
    for lag in range(1, min(tau_max + 1, T // 4)):
        if T - lag < 20:
            continue
        q_yt, q_yt1 = _qbin(y[lag:], bins), _qbin(y[lag - 1:-1], bins)
        joint_y = q_yt * bins + q_yt1
        per_lag.append((lag, joint_y * bins, q_yt1 * bins,
                        _entropy(joint_y) - _entropy(q_yt1)))

    def scan(x: np.ndarray) -> tuple:
        # x[:n] is binned on its own range; that is the range of the whole
        # series unless its minimum or maximum lies in the last T - n points.
        q_full = _qbin(x, bins)
        first_min, first_max = int(x.argmin()), int(x.argmax())
        best_te, best_lag = 0.0, 1
        for lag, joint_y_shifted, q_yt1_shifted, h_y in per_lag:
            n = T - lag
            q_x = q_full[:n] if (first_min < n and first_max < n) else _qbin(x[:n], bins)
            te = max(float(h_y - _entropy(joint_y_shifted + q_x)
                           + _entropy(q_yt1_shifted + q_x)), 0.0)
            if te > best_te:
                best_te, best_lag = te, lag
        return best_te, best_lag

    return scan


def _te_test_target(j: int, X: np.ndarray, tau_max: int, method: str, bins: int, k: int,
                    n_perms: int, min_exceed: int, seed: int) -> list:
    """
    Tests every source i != j against target j. Executable in parallel.
    Returns [(i, j, lag, te, pvalue, permutations drawn), ...].
    """
    scan = _te_lag_scanner(X[:, j], tau_max, method, bins, k)
    out = []
    for i in range(X.shape[1]):
        if i == j:
            continue
        x = X[:, i]
        te, lag = scan(x)
        pval, drawn = 1.0, 0
        if te > 0.0:
            rng = np.random.default_rng([seed, i, j])
            exceed = 0
            for drawn in range(1, n_perms + 1):
                if scan(rng.permutation(x))[0] >= te:
                    exceed += 1
                    if exceed >= min_exceed:
                        break
            pval = (exceed / drawn if exceed >= min_exceed
                    else (exceed + 1) / (n_perms + 1))
        out.append((i, j, lag, te, pval, drawn))
    return out


def run_pipeline_transfer_entropy(
    df: pd.DataFrame,
    date_col: str = "Data",
    entity_col: str = "Nome",
    property_cols: Optional[List[str]] = None,
    max_lag: int = 24,
    alpha_discovery: float = 0.20,
    te_method: str = "histogram",
    te_bins: int = 6,
    te_k: int = 4,
    n_jobs: int = 1,
    transfer_entropy_tau_max: Optional[int] = None,
    transfer_entropy_alpha_level: Optional[float] = None,
    transfer_entropy_correction: str = "fdr_bh",
    transfer_entropy_n_perms: int = 5000,
    transfer_entropy_min_exceed: int = 10,
    transfer_entropy_seed: int = 42,
    **_kwargs,
) -> pd.DataFrame:
    """
    Pairwise Transfer Entropy with a max-over-lags permutation test —
    drop-in in the runner: pipeline(df, **cfg) -> DataFrame with
    'significant'.

    Shared with the main pipeline (read from the same config keys, so the
    estimator is the one Stage 4 uses in that run):
      - te_method : 'histogram' | 'knn'.
      - te_bins   : bins per variable of the histogram estimator.
      - te_k      : neighbours of the KNN estimator.
      - n_jobs    : parallel workers (one task per target series).

    Method-specific parameters (prefix 'transfer_entropy_'):
      - transfer_entropy_tau_max     : maximum lag scanned. Default:
                                       min(max_lag, 10). The estimator
                                       itself stops at T // 4 - 1.
      - transfer_entropy_alpha_level : significance level, applied AFTER
                                       the multiple-comparisons correction.
                                       Default: alpha_discovery.
      - transfer_entropy_correction  : 'none' | 'bonferroni' | 'fdr_bh'
                                       (default), over the p-values of all
                                       ordered pairs (m = N * (N - 1)).
      - transfer_entropy_n_perms     : maximum permutations per pair; sets
                                       the smallest attainable p-value,
                                       1 / (n_perms + 1). Default: 5000.
      - transfer_entropy_min_exceed  : exceedances that stop a pair early
                                       (see NUMBER OF PERMUTATIONS in the
                                       module docstring). Default: 10.
      - transfer_entropy_seed        : seed of the permutations (each pair
                                       has its own stream, so the result
                                       does not depend on n_jobs).
    """
    tau_max     = (transfer_entropy_tau_max if transfer_entropy_tau_max is not None
                   else min(int(max_lag), 10))
    alpha_level = (transfer_entropy_alpha_level if transfer_entropy_alpha_level is not None
                   else alpha_discovery)

    pivoted = _pcmci_load_and_pivot(df, date_col=date_col, entity_col=entity_col,
                                     property_cols=property_cols)
    var_names = list(pivoted.columns)

    # per-column z-score standardization
    X = pivoted.values.astype(float)
    dev = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(dev < 1e-12, 1.0, dev)
    T, N = X.shape

    if N < 2:
        raise ValueError(f"Transfer Entropy: only {N} valid series — need >= 2.")
    if T < 21:
        raise ValueError(
            f"Transfer Entropy: time series too short (T={T}) — the estimator "
            f"needs >= 20 points after the lag."
        )

    n_lags = min(tau_max + 1, T // 4) - 1
    print(f"    [transfer_entropy] N={N} variables, T={T} points, lags 1..{n_lags}, "
          f"estimator={te_method} (bins={te_bins}, k={te_k}), "
          f"n_perms<={transfer_entropy_n_perms} (stop at {transfer_entropy_min_exceed} "
          f"exceedances), alpha_level={alpha_level}, "
          f"correction={transfer_entropy_correction}")

    args = (X, tau_max, te_method, te_bins, te_k,
            transfer_entropy_n_perms, transfer_entropy_min_exceed, transfer_entropy_seed)
    if HAS_JOBLIB and n_jobs != 1:
        per_target = Parallel(n_jobs=n_jobs)(
            delayed(_te_test_target)(j, *args) for j in range(N)
        )
    else:
        per_target = [_te_test_target(j, *args) for j in range(N)]
    tested = [t for target in per_target for t in target]

    corrected = _correct_pvalues([t[4] for t in tested], transfer_entropy_correction)

    rows = []
    for (i, j, lag, te, pval, drawn), pval_corr in zip(tested, corrected):
        src_entity, src_property = var_names[i].split("__", 1)
        dest_entity, dest_property = var_names[j].split("__", 1)
        rows.append({
            "source_entity":    src_entity,
            "source_property":  src_property,
            "target_entity":    dest_entity,
            "target_property":  dest_property,
            "lag":              lag,
            "value":            float(te),
            "n_perms":          drawn,
            "pvalue":           float(pval),
            "pvalue_corrected": float(pval_corr),
            "significant":      bool(pval_corr < alpha_level),
            "method":           "transfer_entropy",
        })

    return pd.DataFrame(rows, columns=[
        "source_entity", "source_property", "target_entity", "target_property",
        "lag", "value", "n_perms", "pvalue", "pvalue_corrected",
        "significant", "method",
    ])
