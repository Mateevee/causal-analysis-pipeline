"""
pipeline_granger.py — Pairwise Granger causality on its own, as a baseline
============================================================================
Exposes run_pipeline_granger with the SAME call signature used by the
runner. Reuses from causal_common: _pcmci_load_and_pivot (same preprocessing
as the other baselines), _ngc_build_lagged, _ngc_ftest_pair and
_correct_pvalues.

The main pipeline uses a Granger test as one of its ingredients (Stage 5
rescue, Pipeline/rescue.py). This baseline answers "what does a Granger test
do by itself": the textbook bivariate test (Granger, Econometrica 1969; in
its F-test form, Hamilton 1994, Sec. 11.2), with nothing from the pipeline
around it -- no screening, no lag penalty, no DML, no structural stage.

For every ordered pair of series (i -> j), i != j:

    restricted   : X_j(t) ~ 1 + X_j(t-1..t-p)
    unrestricted : X_j(t) ~ 1 + X_j(t-1..t-p) + X_i(t-1..t-p)

    F = ((RSS_r - RSS_u) / p) / (RSS_u / (n - 2p - 1))  ~  F(p, n - 2p - 1)

i.e. the p lags of X_i are tested as a block, given the target's own past.
The order p is that of the bivariate VAR of (X_i, X_j), chosen per pair by
an information criterion over 1..tau_max (granger_criterion, default 'aic',
the same criterion the VARLiNGAM baseline uses for its VAR order), with all
orders compared on the same sample (t = tau_max..T-1); the test itself then
uses every point available at the chosen order (t = p..T-1). The criterion
is the one of the two-equation system (ln det of the residual covariance),
not of the tested equation alone: choosing p on the equation whose
coefficients are then tested roughly doubles the size of the test (about
0.10 at a nominal 0.05 on independent AR(1) series, against 0.06-0.08 for
the system criterion), while 'bic' keeps the nominal size but misses weak
effects at lags >= 3 because it rarely selects that many lags.

The test is BIVARIATE on purpose: it conditions on the target's past only,
not on the other series, so it does not separate a direct link from an
indirect one or from a common driver. That is the method as defined, and
the reason the conditional methods (PCMCI, VARLiNGAM, ...) exist.

Decision: the p-values of ALL ordered pairs are corrected together
(granger_correction, default 'fdr_bh') and compared with
granger_alpha_level -- the same rule as the PCMCI baseline (alpha = 0.05
with FDR-BH over every tested hypothesis).

One row per ordered pair: 'value' is the F statistic, 'var_order' the
selected p, and 'lag' the lag of X_i with the largest absolute coefficient
in the unrestricted regression (the test is on the block; that lag is
reported for information). Lagged-only by definition (lag >= 1).
"""

import numpy as np
import pandas as pd
from typing import Optional, List

from Auxiliary.causal_common import (
    _pcmci_load_and_pivot,
    _correct_pvalues,
    _ngc_build_lagged,
    _ngc_ftest_pair,
    MIN_VALID_SAMPLES,
)


def _granger_max_feasible_lag(T: int, tau_max: int) -> int:
    """
    Largest order p <= tau_max whose unrestricted regression keeps at least
    MIN_VALID_SAMPLES residual degrees of freedom: (T - p) - (2p + 1).
    """
    return min(tau_max, (T - 1 - MIN_VALID_SAMPLES) // 3)


def _granger_select_order(lags_x: np.ndarray, lags_y: np.ndarray, xy: np.ndarray,
                          criterion: Optional[str]) -> int:
    """
    Order p in 1..P of the bivariate VAR xy(t) ~ 1 + lags_x[:, :p] +
    lags_y[:, :p], minimizing ln det(Sigma_p) + c * k / n, with Sigma_p the
    residual covariance, k = 2 * (2p + 1) coefficients and c = 2 ('aic') or
    ln n ('bic'). lags_*[:, tau-1] is the series at t - tau and xy the two
    series at t, all orders on the same n rows. criterion=None -> P.
    """
    n, P = lags_y.shape
    if criterion is None:
        return P
    ones = np.ones((n, 1))
    penalty = 2.0 if criterion == "aic" else np.log(n)
    best_p, best_ic = 1, np.inf
    for p in range(1, P + 1):
        design = np.column_stack([ones, lags_x[:, :p], lags_y[:, :p]])
        coef, *_ = np.linalg.lstsq(design, xy, rcond=None)
        resid = xy - design @ coef
        sign, logdet = np.linalg.slogdet(resid.T @ resid / n)
        ic = (logdet if sign > 0 else -np.inf) + penalty * 2 * (2 * p + 1) / n
        if ic < best_ic:
            best_p, best_ic = p, ic
    return best_p


def run_pipeline_granger(
    df: pd.DataFrame,
    date_col: str = "Data",
    entity_col: str = "Nome",
    property_cols: Optional[List[str]] = None,
    max_lag: int = 24,
    alpha_discovery: float = 0.20,
    granger_tau_max: Optional[int] = None,
    granger_criterion: Optional[str] = "aic",
    granger_alpha_level: Optional[float] = None,
    granger_correction: str = "fdr_bh",
    **_kwargs,
) -> pd.DataFrame:
    """
    Pairwise (bivariate) Granger causality F-test — drop-in in the runner:
    pipeline(df, **cfg) -> DataFrame with 'significant'.

    Method-specific parameters (prefix 'granger_'):
      - granger_tau_max     : maximum order searched. Default:
                              min(max_lag, 10), further capped by
                              _granger_max_feasible_lag.
      - granger_criterion   : 'aic' (default) | 'bic' | None — order of
                              the bivariate VAR within [1, tau_max], per
                              pair; None tests exactly tau_max lags.
      - granger_alpha_level : significance level, applied AFTER the
                              multiple-comparisons correction. Default:
                              alpha_discovery.
      - granger_correction  : 'none' | 'bonferroni' | 'fdr_bh' (default),
                              over the p-values of all ordered pairs
                              (m = N * (N - 1)).
    """
    criterion = granger_criterion.lower() if granger_criterion else None
    if criterion not in ("aic", "bic", None):
        raise ValueError(f"Granger: unknown granger_criterion '{granger_criterion}' "
                         f"(use 'aic', 'bic' or None).")

    tau_max     = granger_tau_max if granger_tau_max is not None else min(int(max_lag), 10)
    alpha_level = granger_alpha_level if granger_alpha_level is not None else alpha_discovery

    pivoted = _pcmci_load_and_pivot(df, date_col=date_col, entity_col=entity_col,
                                     property_cols=property_cols)
    var_names = list(pivoted.columns)

    # per-column z-score standardization
    X = pivoted.values.astype(float)
    dev = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(dev < 1e-12, 1.0, dev)
    T, N = X.shape

    if N < 2:
        raise ValueError(f"Granger: only {N} valid series — need >= 2.")

    lag_cap = _granger_max_feasible_lag(T, tau_max)
    if lag_cap < 1:
        raise ValueError(
            f"Granger: time series too short (T={T}) — even order 1 leaves fewer "
            f"than {MIN_VALID_SAMPLES} residual degrees of freedom."
        )
    if lag_cap < tau_max:
        print(f"    [granger] tau_max capped {tau_max} -> {lag_cap} "
              f"(T={T}: keeps >= {MIN_VALID_SAMPLES} residual dof in the unrestricted regression)")

    print(f"    [granger] N={N} variables, T={T} points, max order {lag_cap} "
          f"(criterion={criterion}), alpha_level={alpha_level}, "
          f"correction={granger_correction}")

    # Common sample of the order selection: t = lag_cap..T-1, columns grouped
    # by source (lags 1..lag_cap of series g at g*lag_cap .. (g+1)*lag_cap - 1).
    X_lag = _ngc_build_lagged(X, lag_cap)
    Y     = X[lag_cap:]

    tested = []
    for j in range(N):
        lags_j = X_lag[:, j * lag_cap: (j + 1) * lag_cap]
        for i in range(N):
            if i == j:
                continue
            p = _granger_select_order(X_lag[:, i * lag_cap: (i + 1) * lag_cap],
                                      lags_j, Y[:, [i, j]], criterion)
            # Bivariate system (X_i, X_j) at the chosen order: lags of X_i
            # tested as a block, conditioned on the lags of X_j.
            pair_lag = _ngc_build_lagged(X[:, [i, j]], p)
            f_stat, pval, dominant_lag = _ngc_ftest_pair(0, 1, pair_lag, X[p:, j], 2, p, [1])
            tested.append((i, j, p, dominant_lag, f_stat, pval))

    corrected = _correct_pvalues([t[5] for t in tested], granger_correction)

    rows = []
    for (i, j, p, dominant_lag, f_stat, pval), pval_corr in zip(tested, corrected):
        src_entity, src_property = var_names[i].split("__", 1)
        dest_entity, dest_property = var_names[j].split("__", 1)
        rows.append({
            "source_entity":    src_entity,
            "source_property":  src_property,
            "target_entity":    dest_entity,
            "target_property":  dest_property,
            "lag":              dominant_lag,
            "value":            float(f_stat),
            "var_order":        p,
            "pvalue":           float(pval),
            "pvalue_corrected": float(pval_corr),
            "significant":      bool(pval_corr < alpha_level),
            "method":           "granger",
        })

    return pd.DataFrame(rows, columns=[
        "source_entity", "source_property", "target_entity", "target_property",
        "lag", "value", "var_order", "pvalue", "pvalue_corrected",
        "significant", "method",
    ])
