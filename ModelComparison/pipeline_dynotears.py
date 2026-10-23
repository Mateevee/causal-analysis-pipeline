"""
pipeline_dynotears.py — Wrapper for the OFFICIAL DYNOTEARS optimization core
================================================================================
Exposes run_pipeline_dynotears with the SAME call signature used by the
runner. Reuses from causal_common: _pcmci_load_and_pivot (same preprocessing
as the PCMCIs), _ngc_build_lagged, _ngc_ftest_pair and _split_sample.

Uses the OFFICIAL numerical core of DYNOTEARS (Pamfil,
Sriwattanaworachai, Desai, Pilgerstorfer, Georgatzis, Beaumont & Aragam,
"DYNOTEARS: Structure Learning from Time-Series Data", AISTATS 2020),
vendored verbatim (functions `_dynotears_official_reshape_wa` and
`_dynotears_official_learn_structure` below) from the `causalnex` package
(McKinsey & Company / QuantumBlack Labs), file
`causalnex/structure/dynotears.py`, at
https://github.com/mckinsey/causalnex — only the private numerical routines
are vendored (no dependency on the rest of the causalnex package, which
pulls in an older networkx/pgmpy stack incompatible with this project's
environment); everything around them (data preparation, F-test gating,
output formatting) is this project's own wrapper code, following the same
pattern already used for the official PCMCI/PCMCI+ (tigramite) and
VARLiNGAM (lingam) baselines.

    Copyright 2019-2020 QuantumBlack Visual Analytics Limited
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

DYNOTEARS jointly estimates, via continuous optimization (dual ascent + an
augmented-Lagrangian acyclicity constraint on W, exactly as in the original
paper):
  - a contemporaneous matrix W (N x N), subject to h(W) = tr(e^{W∘W}) - N = 0;
  - lagged matrices A_1..A_tau_max (N x N each), unconstrained (a purely
    lagged DAG is already acyclic by temporal construction).

Both W and A are fit JOINTLY by the vendored optimizer exactly as in the
official method — W is not fixed at zero here. Whether the contemporaneous
(lag-0) edges of W are REPORTED is controlled by
dynotears_include_contemporaneous (default False: lagged rows only). The
main run sets it True (Config/execution.py), because the pipeline itself
reports lag-0 links (Stage 8b) — the same flag is set for PCMCI+ and
VARLiNGAM. The flag changes only the reporting: W is always estimated, and
the lagged rows are identical with the flag on or off.

DYNOTEARS has no native p-value. As for the other baselines in this runner
without one (VARLiNGAM, CUTS+), a pair i -> j is "discovered" when some
lagged coefficient A_k[i, j] (k >= 1) survives a_threshold pruning, and its
p-value comes from an OLS F-test of the tau_max lags of X_i conditioned on
the other discovered lagged parents of j (_ngc_ftest_pair) — the same
post-selection inference already used for PCMCI's MCI-style conditioning.
'significant' = discovered AND pvalue < alpha_discovery (after
dynotears_correction; dynotears_ftest_gate=False uses the native pruned
graph alone -- see _posthoc_gate). A reported lag-0
edge i -> j (W[i, j] != 0) gets its p-value from the contemporaneous
counterpart, _ngc_contemp_ttest: a t-test of X_i(t) conditioned on j's
other W-parents at t and on the lags of j's A-parents.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import scipy.linalg as _slin
import scipy.optimize as _sopt

from Auxiliary.causal_common import (
    _pcmci_load_and_pivot,
    _ngc_build_lagged,
    _ngc_ftest_pair,
    _ngc_contemp_ttest,
    _split_sample,
    _posthoc_gate,
)


# =============================================================================
# Vendored (verbatim, cosmetically renamed) from causalnex/structure/dynotears.py
# Copyright QuantumBlack Visual Analytics Limited — Apache License 2.0.
# See file docstring above for provenance and license terms.
# =============================================================================

def _dynotears_official_reshape_wa(
    wa_vec: np.ndarray, d_vars: int, p_orders: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Verbatim from causalnex's `_reshape_wa`: unpacks the signed-split
    optimization vector back into the (W, A) matrix pair."""
    w_tilde = wa_vec.reshape([2 * (p_orders + 1) * d_vars, d_vars])
    w_plus = w_tilde[:d_vars, :]
    w_minus = w_tilde[d_vars: 2 * d_vars, :]
    w_mat = w_plus - w_minus
    a_plus = (
        w_tilde[2 * d_vars:]
        .reshape(2 * p_orders, d_vars ** 2)[::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_minus = (
        w_tilde[2 * d_vars:]
        .reshape(2 * p_orders, d_vars ** 2)[1::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_mat = a_plus - a_minus
    return w_mat, a_mat


def _dynotears_official_learn_structure(
    X: np.ndarray,
    Xlags: np.ndarray,
    bnds: list,
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verbatim from causalnex's `_learn_dynamic_structure` (the DYNOTEARS
    dual-ascent optimizer): minimizes F(W, A) — reconstruction loss + L1
    penalties on W and A — subject to the NOTEARS acyclicity constraint
    h(W) = tr(e^{W∘W}) - N = 0 on the contemporaneous matrix W, via an
    augmented-Lagrangian schedule (rho, alpha) solved with L-BFGS-B at each
    step. See Pamfil et al. (2020), AISTATS, for the full derivation.

    Args:
        X: (n, d) current-slice data (already lag-aligned with Xlags).
        Xlags: (n, p*d) lagged data, Xlags = [shift(X,1)|...|shift(X,p)].
        bnds: box constraints for L-BFGS-B (bans W self-loops; enforces the
            non-negativity of the split positive/negative parts).
        lambda_w, lambda_a: L1 penalty on W and A respectively.
        max_iter: max number of dual-ascent steps.
        h_tol: acyclicity tolerance to stop early.

    Returns:
        (W, A): (d, d) contemporaneous and (d*p, d) lagged adjacency matrices.
    """
    if X.size == 0:
        raise ValueError("Input data X is empty, cannot learn any structure")
    if Xlags.size == 0:
        raise ValueError("Input data Xlags is empty, cannot learn any structure")
    if X.shape[0] != Xlags.shape[0]:
        raise ValueError("Input data X and Xlags must have the same number of rows")
    if Xlags.shape[1] % X.shape[1] != 0:
        raise ValueError("Number of columns of Xlags must be a multiple of number of columns of X")

    n, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars

    def _h(wa_vec: np.ndarray) -> float:
        _w_mat, _ = _dynotears_official_reshape_wa(wa_vec, d_vars, p_orders)
        return np.trace(_slin.expm(_w_mat * _w_mat)) - d_vars

    def _func(wa_vec: np.ndarray) -> float:
        _w_mat, _a_mat = _dynotears_official_reshape_wa(wa_vec, d_vars, p_orders)
        loss = (
            0.5 / n * np.square(np.linalg.norm(
                X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat), "fro"
            ))
        )
        _h_value = _h(wa_vec)
        l1_penalty = lambda_w * (wa_vec[: 2 * d_vars ** 2].sum()) + lambda_a * (
            wa_vec[2 * d_vars ** 2:].sum()
        )
        return loss + 0.5 * rho * _h_value * _h_value + alpha * _h_value + l1_penalty

    def _grad(wa_vec: np.ndarray) -> np.ndarray:
        _w_mat, _a_mat = _dynotears_official_reshape_wa(wa_vec, d_vars, p_orders)
        e_mat = _slin.expm(_w_mat * _w_mat)
        loss_grad_w = (
            -1.0 / n * (X.T.dot(X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat)))
        )
        obj_grad_w = (
            loss_grad_w + (rho * (np.trace(e_mat) - d_vars) + alpha) * e_mat.T * _w_mat * 2
        )
        obj_grad_a = (
            -1.0 / n * (Xlags.T.dot(X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat)))
        )
        grad_vec_w = np.append(obj_grad_w, -obj_grad_w, axis=0).flatten() + lambda_w * np.ones(2 * d_vars ** 2)
        grad_vec_a = obj_grad_a.reshape(p_orders, d_vars ** 2)
        grad_vec_a = np.hstack((grad_vec_a, -grad_vec_a)).flatten() + lambda_a * np.ones(2 * p_orders * d_vars ** 2)
        return np.append(grad_vec_w, grad_vec_a, axis=0)

    wa_est = np.zeros(2 * (p_orders + 1) * d_vars ** 2)
    wa_new = np.zeros(2 * (p_orders + 1) * d_vars ** 2)
    rho, alpha, h_value, h_new = 1.0, 0.0, np.inf, np.inf

    for n_iter in range(max_iter):
        while (rho < 1e20) and (h_new > 0.25 * h_value or h_new == np.inf):
            wa_new = _sopt.minimize(_func, wa_est, method="L-BFGS-B", jac=_grad, bounds=bnds).x
            h_new = _h(wa_new)
            if h_new > 0.25 * h_value:
                rho *= 10
        wa_est = wa_new
        h_value = h_new
        alpha += rho * h_value
        if h_value <= h_tol:
            break
        if h_value > h_tol and n_iter == max_iter - 1:
            warnings.warn("DYNOTEARS: failed to converge within dynotears_max_iter "
                           "dual-ascent steps — consider increasing it.")
    return _dynotears_official_reshape_wa(wa_est, d_vars, p_orders)


def _dynotears_official_bounds(d_vars: int, p_orders: int) -> list:
    """
    Verbatim (tabu-free specialization) of the bound construction inside
    causalnex's `from_numpy_dynamic`: bans self-loops in W, enforces
    non-negativity of the split positive/negative parts of W and A.
    """
    bnds_w = 2 * [
        (0, 0) if i == j else (0, None)
        for i in range(d_vars) for j in range(d_vars)
    ]
    bnds_a = []
    for _k in range(1, p_orders + 1):
        bnds_a.extend(2 * [(0, None) for _i in range(d_vars) for _j in range(d_vars)])
    return bnds_w + bnds_a


# =============================================================================
# This project's own wrapper code (data prep, F-test gating, output format)
# =============================================================================

def _dynotears_build_xlags(X: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds (X_used, Xlags) in the exact layout `_dynotears_official_learn_structure`
    (== causalnex's `from_numpy_dynamic`) expects: Xlags = [shift(X,1) | ... |
    shift(X,p)], row-aligned with X_used = X[p:] so Xlags[t] holds X[t-1..t-p]
    for the same time index as X_used[t]. This column layout (grouped BY LAG)
    is DIFFERENT from _ngc_build_lagged's (grouped BY SOURCE VARIABLE) — kept
    local to this module since only the vendored optimizer consumes it; the
    post-hoc F-test further below builds its own _ngc_build_lagged matrix.
    """
    T, d = X.shape
    X_used = X[p:]
    blocks = [X[p - k: T - k] for k in range(1, p + 1)]
    Xlags = np.concatenate(blocks, axis=1)
    return X_used, Xlags


def run_pipeline_dynotears(
    df: pd.DataFrame,
    date_col: str = "Data",
    entity_col: str = "Nome",
    property_cols: Optional[List[str]] = None,
    max_lag: int = 24,
    alpha_discovery: float = 0.20,
    dynotears_tau_max: Optional[int] = None,
    dynotears_lambda_w: float = 0.05,
    dynotears_lambda_a: float = 0.05,
    dynotears_max_iter: int = 100,
    dynotears_h_tol: float = 1e-8,
    dynotears_w_threshold: float = 0.3,
    dynotears_a_threshold: float = 0.1,
    dynotears_sample_split: bool = False,
    dynotears_include_contemporaneous: bool = False,
    dynotears_ftest_gate: bool = True,
    dynotears_correction: str = "none",
    **_kwargs,
) -> pd.DataFrame:
    """
    OFFICIAL DYNOTEARS (Pamfil et al., AISTATS 2020) — drop-in in the
    runner: pipeline(df, **cfg) -> DataFrame with 'significant'.

    Method-specific parameters (prefix 'dynotears_'):
        dynotears_tau_max        : maximum lag tested (default: min(max_lag, 10)).
        dynotears_lambda_w       : L1 penalty on the contemporaneous matrix W
                                    (lambda_W in the paper). Default: 0.05 —
                                    the value the paper states for its own
                                    experiments (Sec. 4.1: "regularization
                                    parameters lambda_W = lambda_A = 0.05").
                                    NOTE: causalnex's library default is 0.1;
                                    the paper's value is used here so the
                                    pipeline reproduces the published setup.
        dynotears_lambda_a       : L1 penalty on the lagged matrices A_1..A_p
                                    (lambda_A in the paper). Default: 0.05,
                                    same source/rationale as lambda_w above.
        dynotears_max_iter       : max dual-ascent steps of the augmented
                                    Lagrangian schedule. Default: 100
                                    (causalnex's own default).
        dynotears_h_tol          : acyclicity tolerance to stop early.
                                    Default: 1e-8 (causalnex's own default).
        dynotears_w_threshold    : post-hoc pruning of small |W| entries
                                    (contemporaneous) — the weight threshold
                                    tau_W of the paper. Default: 0.3, the
                                    value of the paper's benchmark protocol
                                    (App. B.3: "we apply the weight
                                    thresholds tau_W = 0.3 and tau_A = 0.1",
                                    tau_W chosen to match static NOTEARS).
                                    The tau_W = tau_A = 0.01 of Sec. 4.1 /
                                    Fig. 2 is only the illustrative example,
                                    whose low values the authors call
                                    "intentional". NOTE: causalnex has a
                                    single w_threshold for both matrices
                                    (library default 0.0 = no pruning); the
                                    paper's two thresholds are applied
                                    separately here.
        dynotears_a_threshold    : post-hoc pruning of small |A| entries
                                    (lagged) — tau_A of the paper. Default:
                                    0.1 (App. B.3, same source as above).
        dynotears_sample_split   : if True, splits the series into two
                                    temporal halves — the first for fitting
                                    (W, A), the second for the significance
                                    F-test — eliminating the double-dipping
                                    of testing significance on the same data
                                    used for structure learning. Default False.
        dynotears_include_contemporaneous : if True, also reports lag-0
                                    (contemporaneous) edges from W, each
                                    tested with _ngc_contemp_ttest. Default
                                    False; the main run sets True (same as
                                    PCMCI+ and VARLiNGAM), since the pipeline
                                    reports lag-0 links (Stage 8b). W is
                                    always estimated jointly with A regardless
                                    of this flag; only the reporting changes.
        dynotears_ftest_gate     : True (default) requires the post-hoc
                                    F-/t-test p-value (corrected with
                                    dynotears_correction) to be < alpha;
                                    False keeps the NATIVE output — every
                                    edge surviving dynotears_w_threshold /
                                    dynotears_a_threshold is
                                    significant, as in the paper (the
                                    p-value is still reported).
        dynotears_correction     : 'none' (default) | 'bonferroni' |
                                    'fdr_bh', over the edges of the graph
                                    (see _posthoc_gate).

    Flow:
        1) Loads and pivots the df (same as PCMCI) via _pcmci_load_and_pivot.
        2) [If dynotears_sample_split] splits the series into a selection
           half (structure learning) and an inference half (F-test).
        3) Runs the OFFICIAL DYNOTEARS optimizer (_dynotears_official_learn_structure)
           to jointly estimate W (contemporaneous, acyclicity-constrained)
           and A_1..A_tau_max (lagged), on the selection half.
        4) Prunes |W| entries below dynotears_w_threshold (tau_W) and |A|
           entries below dynotears_a_threshold (tau_A).
        5) For each lagged pair (i, j, lag=k) with A_k[i, j] != 0: OLS
           F-test conditioning on the other discovered lagged parents ->
           (f_stat, pvalue, dom_lag); and, if dynotears_include_contemporaneous,
           for each contemporaneous pair with W[i, j] != 0: OLS t-test of
           X_i(t) (_ngc_contemp_ttest) -> a lag-0 row. Both on the
           inference half when dynotears_sample_split.
        6) Returns a DataFrame in the pipeline's standard format.
    """
    tau_max = dynotears_tau_max if dynotears_tau_max is not None else min(int(max_lag), 10)

    pivoted = _pcmci_load_and_pivot(
        df, date_col=date_col, entity_col=entity_col, property_cols=property_cols
    )
    var_names = list(pivoted.columns)
    N = len(var_names)
    X = pivoted.values.astype(float)
    dev = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(dev < 1e-12, 1.0, dev)
    T = X.shape[0]

    if T <= tau_max + 5:
        raise ValueError(f"DYNOTEARS: time series too short (T={T}) for tau_max={tau_max}.")
    if N < 2:
        raise ValueError(f"DYNOTEARS: only {N} valid series — need >= 2.")

    # -- Data for the OFFICIAL optimizer (grouped BY LAG; see _dynotears_build_xlags) --
    X_off, Xlags_off = _dynotears_build_xlags(X, tau_max)

    # -- Data for the post-hoc F-test (grouped BY SOURCE VARIABLE; same convention
    #    as the other baselines in this runner) --
    X_lag_ftest = _ngc_build_lagged(X, tau_max)
    Y_ftest = X[tau_max:]

    if dynotears_sample_split:
        X_off_sel, Xlags_off_sel, _, _, _, _ = _split_sample(X_off, Xlags_off)
        X_ftest_sel, Y_ftest_sel, _, X_ftest_inf, Y_ftest_inf, _ = _split_sample(X_lag_ftest, Y_ftest)
        if (X_off_sel.shape[0] <= tau_max + 5 or Xlags_off_sel.shape[0] <= tau_max + 5
                or X_ftest_inf.shape[0] <= tau_max + 5):
            print(f"    [dynotears] WARNING: the split sample would be too short "
                  f"(T_eff={X_lag_ftest.shape[0]}) — ignoring dynotears_sample_split "
                  f"for this scenario.")
            dynotears_sample_split = False
            X_off_sel = X_off
            Xlags_off_sel = Xlags_off
            X_ftest_inf, Y_ftest_inf = X_lag_ftest, Y_ftest
    else:
        X_off_sel, Xlags_off_sel = X_off, Xlags_off
        X_ftest_inf, Y_ftest_inf = X_lag_ftest, Y_ftest

    # Raw rows behind the inference sample, for the lag-0 t-test: its
    # targets X_contemp[tau_max:] are exactly the rows of Y_ftest_inf.
    X_contemp = X[(X_lag_ftest.shape[0] // 2):] if dynotears_sample_split else X

    print(
        f"    [dynotears] N={N} vars, T={T}, tau_max={tau_max}, "
        f"lambda_w={dynotears_lambda_w}, lambda_a={dynotears_lambda_a}, "
        f"max_iter={dynotears_max_iter}, w_threshold={dynotears_w_threshold}, "
        f"a_threshold={dynotears_a_threshold}, "
        f"sample_split={dynotears_sample_split}, ftest_gate={dynotears_ftest_gate}, "
        f"correction={dynotears_correction}, official optimizer (causalnex core)"
    )

    bnds = _dynotears_official_bounds(N, tau_max)
    W, A = _dynotears_official_learn_structure(
        X_off_sel, Xlags_off_sel, bnds,
        lambda_w=dynotears_lambda_w, lambda_a=dynotears_lambda_a,
        max_iter=dynotears_max_iter, h_tol=dynotears_h_tol,
    )
    W[np.abs(W) < dynotears_w_threshold] = 0.0
    A[np.abs(A) < dynotears_a_threshold] = 0.0

    n_active_lagged = int((np.abs(A) > 0).sum())
    n_active_contemp = int((np.abs(W) > 0).sum())
    print(f"    [dynotears] official fit: {n_active_lagged} active lagged edge(s) in A, "
          f"{n_active_contemp} active contemporaneous edge(s) in W "
          f"({'reported' if dynotears_include_contemporaneous else 'not reported'})")

    # -- Candidate-pair support: A (lag >= 1) and, if reported, W (lag 0) --
    # lagged_by_target[j] / contemp_by_target[j] = source indices with a
    # nonzero lagged (A) / contemporaneous (W) coefficient into target j
    lagged_by_target = {j: set() for j in range(N)}
    for k in range(1, tau_max + 1):
        block = A[(k - 1) * N: k * N, :]           # (N, N): block[i, j] = source i at lag k -> target j
        for i, j in zip(*np.nonzero(block)):
            if i != j:
                lagged_by_target[int(j)].add(int(i))
    contemp_by_target = {j: set() for j in range(N)}
    if dynotears_include_contemporaneous:
        for i, j in zip(*np.nonzero(W)):
            if i != j:
                contemp_by_target[int(j)].add(int(i))

    # -- One row per discovered (pair, kind): lagged F-test / lag-0 t-test --
    # Pairs not discovered at any lag get no row (same convention as
    # varlingam/cuts_plus).
    rows = []
    for j in range(N):
        for i in range(N):
            if i == j:
                continue
            src_entity, src_property = var_names[i].split("__", 1)
            dest_entity, dest_property = var_names[j].split("__", 1)
            if src_entity == dest_entity and src_property == dest_property:
                continue
            base = {
                "source_entity":    src_entity,
                "source_property":  src_property,
                "target_entity":   dest_entity,
                "target_property": dest_property,
                "method":         "dynotears",
            }
            if i in lagged_by_target[j]:
                cond = [p for p in lagged_by_target[j] if p != i]
                f_val, pval, dom_lag = _ngc_ftest_pair(
                    i, j, X_ftest_inf, Y_ftest_inf[:, j], N, tau_max, cond
                )
                rows.append({**base, "lag": dom_lag, "value": f_val, "pvalue": pval})
            if i in contemp_by_target[j]:
                t_val, pval0 = _ngc_contemp_ttest(
                    i, j, X_contemp, tau_max,
                    sorted(contemp_by_target[j]), sorted(lagged_by_target[j]),
                )
                rows.append({**base, "lag": 0, "value": t_val, "pvalue": pval0})

    _posthoc_gate(rows, alpha_discovery, dynotears_ftest_gate, dynotears_correction)

    return pd.DataFrame(rows, columns=[
        "source_entity", "source_property", "target_entity", "target_property",
        "lag", "value", "pvalue", "pvalue_corrected", "significant", "method",
    ])
