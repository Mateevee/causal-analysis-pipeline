"""
pipeline_varlingam.py — Wrapper for the official VARLiNGAM (lingam library)
============================================================================
Exposes run_pipeline_varlingam with the SAME call signature used by the
runner. Reuses from causal_common: _pcmci_load_and_pivot (same preprocessing
as the PCMCIs), _ngc_build_lagged and _ngc_ftest_pair (same post-selection
F-test as dynotears/cuts_plus).

VARLiNGAM (Hyvärinen, Zhang, Shimizu & Hoyer, JMLR 2010) fits a VAR model,
runs DirectLiNGAM on its residuals to identify the contemporaneous matrix
B0 via non-Gaussianity, and derives the lagged matrices
B_tau = (I - B0) M_tau; with prune=True the B_tau are then sparsified by an
adaptive Lasso (lingam's default). adjacency_matrices_[tau][j, i] is the
effect of X_i(t - tau) on X_j(t).

The method has no native p-values. As for dynotears/cuts_plus, a pair
i -> j is 'discovered' when some lagged coefficient B_tau[j, i] (tau >= 1)
survives pruning, and its p-value comes from an OLS F-test of the lags of
X_i conditioned on the other discovered lagged parents of j — here
including j's own lags when VARLiNGAM kept them (the model is a VAR, so its
own autoregressive terms are part of the discovered parent set).
'significant' = discovered AND pvalue < alpha_discovery, after
varlingam_correction (varlingam_ftest_gate=False uses the pruned support
alone -- see _posthoc_gate).

B0 is always estimated; its edges are REPORTED only with
varlingam_include_contemporaneous=True (default False: lagged rows only),
which the main run sets (Config/execution.py) because the pipeline itself
reports lag-0 links (Stage 8b) -- same flag as PCMCI+ and DYNOTEARS. A
lag-0 edge i -> j (B0[j, i] != 0) gets its p-value from
_ngc_contemp_ttest: a t-test of X_i(t) conditioned on j's other
B0-parents at t and on the lags of j's lagged parents (the lag-0
counterpart of the lagged F-test above); varlingam_ftest_gate applies to
it the same way. The lagged rows are identical with the flag on or off.

Requires `pip install lingam`.
"""

import numpy as np
import pandas as pd
from typing import Optional, List

from Auxiliary.causal_common import (
    _pcmci_load_and_pivot,
    _ngc_build_lagged,
    _ngc_ftest_pair,
    _ngc_contemp_ttest,
    _posthoc_gate,
)


def _varlingam_max_feasible_lag(T: int, N: int, tau_max: int) -> int:
    """
    Largest VAR order whose per-equation OLS still has enough residual
    degrees of freedom: (T - L) - N*L >= max(2N, 30).
    """
    min_dof = max(2 * N, 30)
    L = tau_max
    while L >= 1 and (T - L) - N * L < min_dof:
        L -= 1
    return L


def run_pipeline_varlingam(
    df: pd.DataFrame,
    date_col: str = "Data",
    entity_col: str = "Nome",
    property_cols: Optional[List[str]] = None,
    max_lag: int = 24,
    alpha_discovery: float = 0.20,
    varlingam_tau_max: Optional[int] = None,
    varlingam_criterion: Optional[str] = "aic",
    varlingam_prune: bool = True,
    varlingam_ftest_gate: bool = True,
    varlingam_include_contemporaneous: bool = False,
    varlingam_correction: str = "none",
    **_kwargs,
) -> pd.DataFrame:
    """
    OFFICIAL VARLiNGAM (lingam.VARLiNGAM) — drop-in in the runner:
    pipeline(df, **cfg) -> DataFrame with 'significant'.

    Method-specific parameters (prefix 'varlingam_'):
      - varlingam_tau_max    : maximum VAR order searched. Default:
                               min(max_lag, 10), further capped by
                               _varlingam_max_feasible_lag.
      - varlingam_criterion  : 'aic' | 'fpe' | 'hqic' | 'bic' | None — lag
                               selection within [1, tau_max]; None fits
                               exactly tau_max. Default: 'aic' — the
                               criterion the paper itself uses to pick the
                               VAR order (Hyvarinen et al. 2010, Sec. 7:
                               "we determined the order k of our SVAR
                               model by minimizing the AIC criterion").
                               lingam's own default is 'bic'.
      - varlingam_prune      : adaptive-Lasso pruning of B_tau (lingam's
                               default). Default: True.
      - varlingam_ftest_gate : require the conditioned F-test p-value
                               (corrected with varlingam_correction) to be
                               below alpha_discovery (same post-selection
                               inference as dynotears/cuts_plus). Default:
                               True. False = NATIVE output: every edge kept
                               by the adaptive-Lasso pruning is significant.
      - varlingam_correction : 'none' (default) | 'bonferroni' | 'fdr_bh',
                               over the edges of the pruned graph (see
                               _posthoc_gate).
      - varlingam_include_contemporaneous : also report the lag-0 edges of
                               B0, each tested with _ngc_contemp_ttest.
                               Default: False (the main run sets True).

    Requires `pip install lingam`.
    """
    try:
        from lingam import VARLiNGAM
    except ImportError:
        raise ImportError(
            "varlingam: package 'lingam' not found. Install with:\n"
            "    pip install lingam"
        )

    tau_max = varlingam_tau_max if varlingam_tau_max is not None else min(int(max_lag), 10)

    pivoted = _pcmci_load_and_pivot(df, date_col=date_col, entity_col=entity_col,
                                     property_cols=property_cols)
    var_names = list(pivoted.columns)

    # per-column z-score standardization
    X = pivoted.values.astype(float)
    dev = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(dev < 1e-12, 1.0, dev)
    T, N = X.shape

    if T <= tau_max + 5:
        raise ValueError(
            f"VARLiNGAM: time series too short (T={T}) for tau_max={tau_max}."
        )
    if N < 2:
        raise ValueError(f"VARLiNGAM: only {N} valid series — need >= 2.")

    lag_cap = _varlingam_max_feasible_lag(T, N, tau_max)
    if lag_cap < 1:
        raise ValueError(
            f"VARLiNGAM: T={T} too short for a VAR(1) on N={N} series "
            f"(needs (T-1) - N >= max(2N, 30)). Not applicable here: without "
            f"this guard lingam's own adaptive-Lasso pruning (LassoLarsIC) "
            f"also refuses n_samples < n_features."
        )
    if lag_cap < tau_max:
        print(f"    [varlingam] tau_max capped {tau_max} -> {lag_cap} "
              f"(N={N}, T={T}: keeps >= max(2N, 30) residual dof per VAR equation)")

    model = VARLiNGAM(lags=lag_cap, criterion=varlingam_criterion,
                      prune=varlingam_prune, random_state=0)
    model.fit(X)
    B = model.adjacency_matrices_          # (1 + p, N, N); B[tau][target, source]
    p = B.shape[0] - 1

    print(f"    [varlingam] N={N} variables, T={T} points, max lag {lag_cap} "
          f"-> VAR order p={p} (criterion={varlingam_criterion}), prune={varlingam_prune}, "
          f"ftest_gate={varlingam_ftest_gate}, correction={varlingam_correction}")

    B_lagged = B[1:]                        # (p, N, N)
    X_lag = _ngc_build_lagged(X, p)         # (T - p, N*p), grouped by source
    Y     = X[p:]

    rows = []
    for j in range(N):
        # lagged parents of j kept by VARLiNGAM (including j itself)
        parents_j = [i for i in range(N) if np.any(np.abs(B_lagged[:, j, i]) > 0.0)]
        for i in range(N):
            if i == j:
                continue
            coefs = B_lagged[:, j, i]
            k = int(np.argmax(np.abs(coefs)))
            discovered = bool(np.abs(coefs[k]) > 0.0)

            if discovered:
                cond = [q for q in parents_j if q != i]
                _, pval, _ = _ngc_ftest_pair(i, j, X_lag, Y[:, j], N, p, cond)
            else:
                pval = 1.0

            src_entity, src_property = var_names[i].split("__", 1)
            dest_entity, dest_property = var_names[j].split("__", 1)
            rows.append({
                "source_entity":    src_entity,
                "source_property":  src_property,
                "target_entity":    dest_entity,
                "target_property":  dest_property,
                "lag":              k + 1,
                "value":            float(coefs[k]),
                "pvalue":           float(pval),
                "_discovered":      discovered,
                "method":           "varlingam",
            })

    if varlingam_include_contemporaneous:
        B0 = B[0]                           # B0[target, source]
        for j in range(N):
            parents0_j = [i for i in range(N) if i != j and B0[j, i] != 0.0]
            parents_lag_j = [i for i in range(N) if np.any(np.abs(B_lagged[:, j, i]) > 0.0)]
            for i in parents0_j:
                t_val, pval = _ngc_contemp_ttest(i, j, X, p, parents0_j, parents_lag_j)

                src_entity, src_property = var_names[i].split("__", 1)
                dest_entity, dest_property = var_names[j].split("__", 1)
                rows.append({
                    "source_entity":    src_entity,
                    "source_property":  src_property,
                    "target_entity":    dest_entity,
                    "target_property":  dest_property,
                    "lag":              0,
                    "value":            float(B0[j, i]),
                    "pvalue":           float(pval),
                    "method":           "varlingam",
                })

    _posthoc_gate(rows, alpha_discovery, varlingam_ftest_gate, varlingam_correction)

    return pd.DataFrame(rows, columns=[
        "source_entity", "source_property", "target_entity", "target_property",
        "lag", "value", "pvalue", "pvalue_corrected", "significant", "method",
    ])
