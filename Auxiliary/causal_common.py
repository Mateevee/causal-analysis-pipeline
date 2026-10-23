"""
causal_common.py — Shared numerical helpers used across pipelines
=====================================================================
Gathers low-level functions used by more than one causal-discovery
pipeline in this project:

    _correct_pvalues       : correction for multiple comparisons (FDR-BH /
                              Bonferroni) — used by pcmci_tigramite and by
                              Pipeline/discovery.py.
    _pcmci_load_and_pivot  : converts the LONG dataframe (Date, Name, property
                              columns) into the WIDE format (entity__property)
                              used by pcmci_tigramite and dynotears.
    _ngc_relu / _ngc_drelu / _ngc_build_lagged / _ngc_prox_group_lasso /
    _ngc_ftest_pair        : Neural-Granger-Causality-style helpers,
                              shared by dynotears and cuts_plus.
    _theoretical_lasso_lambda : theoretical LASSO regularization scale, used
                              by dynotears and cuts_plus to auto-scale
                              hyperparameters by problem dimension.
    _split_sample          : temporal split (sample splitting) for
                              post-selection inference, used by dynotears and
                              cuts_plus.
    _tigramite_graph_to_rows : converts a tigramite `graph`/`p_matrix`/
                              `val_matrix` triple into the pipelines' standard
                              row format, used by pcmciplus.
    _cap_tigramite_tau_max : caps tau_max so tigramite's 2*tau_max cut-off
                              keeps >= MIN_VALID_SAMPLES points, used by the
                              PCMCI / PCMCI+ pipelines (parcorr and GPDC).

None of the functions here are specific to a single pipeline — if a helper
is used by only one run_pipeline_*, it should live in that pipeline's own
file, not here.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from scipy import stats as _scipy_stats

def _correct_pvalues(pvalues: list, method: str = "fdr_bh") -> list:
    """
    Correction for multiple comparisons over a list of p-values.

    method:
        'none'        — no correction (raw p-value).
        'bonferroni'  — corrected_p = min(1, p * n_tests).
        'fdr_bh'      — Benjamini-Hochberg (FDR control), default.
                        corrected_p_(i) = min_{j>=i}( p_(j) * n / j ), with
                        p-values sorted p_(1) <= p_(2) <= ... <= p_(n).

    Implemented without depending on statsmodels (numpy only), so as not
    to add a new dependency to the runner.

    NaN p-values (undefined tests) stay NaN and are left out of the family.
    """
    n = len(pvalues)
    if n == 0:
        return []
    method = (method or "none").lower()

    if method == "none":
        return list(pvalues)

    pv_all = np.asarray(pvalues, dtype=float)
    finite = ~np.isnan(pv_all)
    if not finite.all():
        out = np.full(n, np.nan)
        out[finite] = _correct_pvalues(pv_all[finite].tolist(), method)
        return out.tolist()

    if method == "bonferroni":
        return [float(min(1.0, p * n)) for p in pvalues]

    if method == "fdr_bh":
        pv = np.asarray(pvalues, dtype=float)
        order = np.argsort(pv)
        pv_sorted = pv[order]
        ranks = np.arange(1, n + 1)
        # p_(i) * n / i, then enforce monotonicity from the end
        adjusted = pv_sorted * n / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        output = np.empty(n)
        output[order] = adjusted
        return output.tolist()

    raise ValueError(f"Unknown correction method: '{method}' "
                      f"(use 'none', 'bonferroni' or 'fdr_bh').")


def _posthoc_gate(rows: list, alpha: float, ftest_gate: bool, correction: str = "none") -> list:
    """
    Decision rule shared by the baselines without a native p-value
    (DYNOTEARS, VARLiNGAM, CUTS+). Each row is one edge in the method's own
    native graph, with the post-hoc F-/t-test p-value in 'pvalue'. A row
    with a falsy '_discovered' key (VARLiNGAM's rows for pruned pairs) is
    not an edge of the graph: it is never significant and is left out of
    the correction family.

        ftest_gate=False : NATIVE output — significant = edge in the graph;
                           the p-value is kept for information only.
        ftest_gate=True  : significant = edge AND corrected p < alpha, with
                           `correction` ('none' | 'bonferroni' | 'fdr_bh')
                           applied over the edges of the graph (the tested
                           hypotheses, lagged and lag-0 rows together).

    Fills 'pvalue_corrected' and 'significant' in place and drops
    '_discovered'. Returns the same list.
    """
    tested = [r for r in rows if r.get("_discovered", True)]
    corrected = _correct_pvalues([r["pvalue"] for r in tested], correction)
    for r, pc in zip(tested, corrected):
        r["pvalue_corrected"] = float(pc)
        r["significant"] = bool(pc < alpha) if ftest_gate else True
    for r in rows:
        if not r.pop("_discovered", True):
            r["pvalue_corrected"] = 1.0
            r["significant"] = False
    return rows



def _pcmci_load_and_pivot(df: pd.DataFrame, date_col: str, entity_col: str,
                           property_cols: Optional[List[str]]) -> pd.DataFrame:
    """
    Converts the LONG dataframe (date_col, entity_col, property columns) into
    the WIDE format (one column per entity__property combination, indexed by
    date) used by the PCMCI engine.
    """
    dfx = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(dfx[date_col]):
        dfx[date_col] = pd.to_datetime(dfx[date_col], dayfirst=True, errors='coerce')
    dfx = dfx.dropna(subset=[date_col]).sort_values([date_col, entity_col])
    if dfx.empty:
        raise ValueError(
            f"PCMCI custom: dataframe empty after converting '{date_col}' to datetime."
        )

    if property_cols is None:
        resolved_property_cols = [c for c in dfx.columns if c not in [date_col, entity_col]]
    else:
        resolved_property_cols = list(property_cols)
    if not resolved_property_cols:
        raise ValueError("PCMCI custom: resolved property_cols is empty.")

    for col in resolved_property_cols:
        if col not in dfx.columns:
            raise KeyError(f"PCMCI custom: property column '{col}' does not exist in the dataframe.")
        dfx[col] = pd.to_numeric(dfx[col], errors='coerce')

    pivoted = dfx.pivot_table(
        index=date_col, columns=entity_col, values=resolved_property_cols, aggfunc='mean',
    )
    if pivoted.shape[1] == 0:
        raise ValueError("PCMCI custom: pivot_table resulted in 0 columns (entity x property).")

    # pivot_table columns are (property, entity) tuples -> rename to "entity__property"
    pivoted.columns = [f"{w}__{f}" for f, w in pivoted.columns]
    pivoted = pivoted.sort_index()

    # drop 100% NaN columns (entity/property combination that never occurs,
    # e.g. a property only some entities measure)
    pivoted = pivoted.dropna(axis=1, how='all')
    if pivoted.shape[1] == 0:
        raise ValueError("PCMCI custom: all columns ended up empty after pivoting.")

    # interpolate/fill remaining NaNs (needed for linear regression)
    pivoted = pivoted.interpolate(limit_direction='both').ffill().bfill()
    pivoted = pivoted.dropna(axis=1, how='any')
    if pivoted.shape[1] == 0:
        raise ValueError("PCMCI custom: all series ended up empty after interpolation.")

    return pivoted



def _ngc_relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, Z)


def _ngc_drelu(Z: np.ndarray) -> np.ndarray:
    return (Z > 0.0).astype(np.float64)


def _ngc_build_lagged(X: np.ndarray, tau_max: int) -> np.ndarray:
    """
    Builds the lagged feature matrix.

    Returns X_lag with shape (T-tau_max, N*tau_max).
    Columns grouped by source variable:
        [X_0(t-1), X_0(t-2), ..., X_0(t-τ),
         X_1(t-1), ...,
         X_{N-1}(t-1), ..., X_{N-1}(t-τ)]

    Grouping BY VARIABLE is essential both for the Group LASSO proximal
    operator (CUTS+) and for the block-wise norm extraction (DYNOTEARS).
    """
    T, N = X.shape
    blocks = []
    for g in range(N):
        for tau in range(1, tau_max + 1):
            blocks.append(X[tau_max - tau: T - tau, g: g + 1])
    return np.concatenate(blocks, axis=1)          # (T_eff, N*tau_max)


def _theoretical_lasso_lambda(X_lag: np.ndarray, Y: np.ndarray) -> float:
    """
    Theoretical L1 regularization scale for multivariate Lasso regression:

        lambda ~ sigma_hat * sqrt(2*log(D) / T)

    where D = number of features (N*tau_max) and sigma_hat is the residual
    standard deviation, roughly estimated by the target's standard
    deviation (already z-score standardized in this code, so sigma_hat ~ 1
    typically, but kept explicit for robustness). This is the classic LASSO
    consistency rate (Bickel, Ritov & Tsybakov, 2009; Hastie, Tibshirani &
    Wainwright, 2015, ch. 2) — it grows with the number of candidate
    variables and shrinks with sample size, unlike a fixed value shared
    across scenarios of very different sizes.

    Returns the BASE multiplier (T_eff and D already baked in); the final
    value used by the pipelines is this multiplied by the user parameter
    (dynotears_lambda_w / cuts_lambda_gl), which acts as a fine-tuning
    factor rather than an absolute scale.
    """
    T_eff, D = X_lag.shape
    sigma_hat = float(np.std(Y)) if np.std(Y) > 1e-12 else 1.0
    return float(sigma_hat * np.sqrt(2.0 * np.log(max(D, 2)) / max(T_eff, 1)))


def _split_sample(X_lag: np.ndarray, Y: np.ndarray, mask: Optional[np.ndarray] = None):
    """
    Splits (X_lag, Y[, mask]) into two contiguous TEMPORAL halves — the
    first half for selection (Lasso/Group LASSO), the second half for
    inference (F-test) — so that the same points are never used in both
    steps ("sample splitting"), eliminating post-selection inference
    double-dipping. Temporal (not random) split, to preserve the serial
    dependence structure within each half.

    Returns (X_sel, Y_sel, mask_sel, X_inf, Y_inf, mask_inf).
    """
    T_eff = X_lag.shape[0]
    cutoff = T_eff // 2
    m = mask if mask is not None else np.ones(T_eff)
    return (X_lag[:cutoff], Y[:cutoff], m[:cutoff],
            X_lag[cutoff:], Y[cutoff:], m[cutoff:])


# Minimum number of valid samples a baseline must keep after its lag cut-off.
# Same floor the main pipeline uses for the DML control window
# (max(20, 4*n_splits) in Pipeline/dml_engine.py), so the protocol is symmetric.
MIN_VALID_SAMPLES = 20


def _cap_tigramite_tau_max(tau_max: int, T: int, label: str,
                           min_samples: int = MIN_VALID_SAMPLES) -> int:
    """
    Largest tau <= tau_max that leaves at least `min_samples` valid samples
    under tigramite's default cut_off='2xtau_max' (PCMCI / PCMCI+ discard the
    first 2*tau_max time points). Without this cap, a series with
    T <= 2*tau_max leaves no samples.

    Raises ValueError when not even tau=1 satisfies the floor.
    """
    cap = (T - min_samples) // 2
    if cap < 1:
        raise ValueError(
            f"{label}: time series too short (T={T}) — even tau_max=1 leaves "
            f"fewer than {min_samples} valid samples after the 2*tau_max cut-off."
        )
    if cap < tau_max:
        print(f"    [{label}] tau_max capped {tau_max} -> {cap} "
              f"(T={T}, keeps >= {min_samples} valid samples after 2*tau_max cut-off)")
        return cap
    return tau_max


def _tigramite_graph_to_rows(
    graph: np.ndarray,
    p_matrix: np.ndarray,
    val_matrix: np.ndarray,
    var_names: List[str],
    tau_max: int,
    accepted_marks: tuple,
    correction: str,
    alpha_level: float,
    method: str,
    include_contemporaneous: bool = False,
    contemporaneous_marks: tuple = ("-->", "o-o", "x-x"),
) -> pd.DataFrame:
    """
    Converts the output of a graph-based tigramite method (currently
    PCMCI+) into the pipelines' standard row format — one row per
    (i, j, lag), i != j, lag in [lag_min, tau_max].

    Unlike plain PCMCI, the decision of these methods is the `graph` array
    (string edge marks), not a p-value threshold. A row is 'significant'
    when graph[i, j, lag] is one of `accepted_marks` (the mark that asserts
    (or does not rule out) i -> j) — the method's NATIVE output, already
    decided at pc_alpha (and, if requested, with tigramite's own
    fdr_method inside the skeleton phase). With correction='none' (the
    main run) that is the whole rule and alpha_level is not used. Any other
    `correction` adds a post-hoc cut the published method does not make
    (corrected p < alpha_level), kept only as an optional sensitivity
    variant.

    p-values of links absent from the graph are forced to 1.0 before the
    correction (PCMCI+ already does this internally, but the guard is kept
    general for any future graph-based method that might not).

    include_contemporaneous: lag 0 is only reported when True; the methods
    always run with tau_min=0, so contemporaneous parents are used as
    conditions for the lagged links either way. The main run turns it on
    (Config/execution.py): the pipeline reports lag-0 links (Stage 8b), so
    every baseline that estimates contemporaneous structure reports it too.
    The `correction` runs over every reported hypothesis, so with lag 0
    included the family is larger and the significance of a few lagged
    rows can shift (BH adapts to the number of small p-values).

    contemporaneous_marks: marks accepted at lag 0 (lagged rows use
    `accepted_marks`). Default ('-->', 'o-o', 'x-x'): an oriented link
    counts in its own direction, while an unoriented ('o-o') or
    orientation-conflicting ('x-x') adjacency counts in BOTH directions --
    the graph is symmetric at lag 0, so rows (i, j, 0) and (j, i, 0) carry
    the same mark. This is the mixed-graph convention of the pipeline's
    Stage 8b (orientation='both'), so an unoriented lag-0 adjacency is
    scored the same way whichever method reports it.
    """
    N = len(var_names)
    lag_min = 0 if include_contemporaneous else 1

    tested = []
    raw_pvals = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for lag in range(lag_min, tau_max + 1):
                mark = str(graph[i, j, lag])
                p = 1.0 if mark == "" else float(p_matrix[i, j, lag])
                tested.append((i, j, lag, mark))
                raw_pvals.append(p)

    corrected = _correct_pvalues(raw_pvals, correction)
    native = (correction or "none").lower() == "none"

    rows = []
    for (i, j, lag, mark), pval, pval_corr in zip(tested, raw_pvals, corrected):
        marks_ok = contemporaneous_marks if lag == 0 else accepted_marks
        src_entity, src_property = var_names[i].split("__", 1)
        dest_entity, dest_property = var_names[j].split("__", 1)
        rows.append({
            "source_entity":    src_entity,
            "source_property":  src_property,
            "target_entity":    dest_entity,
            "target_property":  dest_property,
            "lag":              lag,
            "value":            float(val_matrix[i, j, lag]),
            "graph_mark":       mark,
            "pvalue":           pval,
            "pvalue_corrected": pval_corr,
            "significant":      bool(mark in marks_ok and (native or pval_corr < alpha_level)),
            "method":           method,
        })

    return pd.DataFrame(rows, columns=[
        "source_entity", "source_property", "target_entity", "target_property",
        "lag", "value", "graph_mark", "pvalue", "pvalue_corrected",
        "significant", "method",
    ])


def _ngc_prox_group_lasso(
    W1: np.ndarray,
    step: float,
    lambda_gl: float,
    N: int,
    tau_max: int,
) -> np.ndarray:
    """
    Group LASSO proximal operator applied to W1 (shape: N*tau_max × hidden).

    For each group g (variable g, tau_max rows):
        W1_g ← max(0, 1 − step·λ / ‖W1_g‖_F) · W1_g
    """
    W_out = W1.copy()
    threshold = step * lambda_gl
    for g in range(N):
        sl = slice(g * tau_max, (g + 1) * tau_max)
        Wg = W_out[sl]
        norm_g = np.linalg.norm(Wg, 'fro')
        if norm_g > threshold:
            W_out[sl] = Wg * (1.0 - threshold / norm_g)
        else:
            W_out[sl] = 0.0
    return W_out


def _ngc_ftest_pair(
    i: int,
    j: int,
    X_lag: np.ndarray,
    Y_j: np.ndarray,
    N: int,
    tau_max: int,
    cond_indices: list,
) -> tuple:
    """
    F-test for the null hypothesis: "the tau_max lags of X_i do not
    contribute to predicting Y_j, controlling for the parents in
    cond_indices".

    Analogous to the PCMCI MCI step: uses the parents discovered during the
    screening/regularization phase as the conditioning set.

    Returns (f_stat, p_value, dominant_lag).
    """
    T_eff = X_lag.shape[0]
    sl_i  = slice(i * tau_max, (i + 1) * tau_max)

    cond_cols = []
    for p in cond_indices:
        if p != i:
            cond_cols.extend(range(p * tau_max, (p + 1) * tau_max))

    Z   = X_lag[:, cond_cols] if cond_cols else None
    X_i = X_lag[:, sl_i]                          # (T_eff, tau_max)

    # Residualize Y_j and the tau_max lags of X_i on Z in ONE least-squares
    # solve (multi-column right-hand side): same SVD of the design, same
    # solution per column as solving them one by one, with tau_max+1 times
    # fewer factorizations.
    V = np.column_stack([Y_j, X_i])
    if Z is None or Z.shape[1] == 0:
        V_r = V - V.mean(axis=0)
    else:
        des = np.column_stack([np.ones(T_eff), Z])
        c, *_ = np.linalg.lstsq(des, V, rcond=None)
        V_r = V - des @ c
    Y_r, X_r = V_r[:, 0], V_r[:, 1:]

    design = np.column_stack([np.ones(T_eff), X_r])
    coef, *_ = np.linalg.lstsq(design, Y_r, rcond=None)
    resid    = Y_r - design @ coef

    RSS_full = float((resid ** 2).sum())
    RSS_null = float((Y_r  ** 2).sum())
    # The full model is the regression of Y_j on [1, Z, X_i lags] (the
    # two-step residualization above is Frisch-Waugh-Lovell), so its
    # residual dof discount the conditioning columns too.
    df1 = tau_max
    df2 = T_eff - len(cond_cols) - tau_max - 1

    if df2 <= 0 or RSS_full < 1e-15:
        return float("nan"), float("nan"), 1   # test undefined, not "p = 1"
    if RSS_null <= RSS_full:
        return 0.0, 1.0, 1

    F    = ((RSS_null - RSS_full) / df1) / (RSS_full / df2)
    pval = float(1 - _scipy_stats.f.cdf(max(F, 0.0), df1, df2))
    dominant_lag = int(np.argmax(np.abs(coef[1:]))) + 1

    return float(max(F, 0.0)), pval, dominant_lag


def _ngc_contemp_ttest(
    i: int,
    j: int,
    X: np.ndarray,
    n_lags: int,
    contemp_parents: list,
    lagged_parents: list,
) -> tuple:
    """
    t-test for the null hypothesis: "X_i(t) does not contribute to
    predicting X_j(t), controlling for j's other contemporaneous parents
    (at time t) and for lags 1..n_lags of j's lagged parents".

    Lag-0 counterpart of _ngc_ftest_pair, for the baselines that estimate a
    contemporaneous matrix (DYNOTEARS' W, VARLiNGAM's B0) and report its
    edges: a reported lag-0 edge gets its p-value from the same kind of
    post-selection OLS test as the lagged ones, conditioned on the parents
    the method itself discovered for j. (Testing a W/B0 edge with
    _ngc_ftest_pair would test the LAGS of X_i instead, i.e. the wrong
    hypothesis.)

    X is the (T, N) standardized data; targets are rows n_lags..T-1, so the
    sample matches the lagged test's.

    Returns (t_stat, p_value).
    """
    T = X.shape[0]
    y = X[n_lags:, j]
    cols = [X[n_lags:, i]]
    cols += [X[n_lags:, q] for q in contemp_parents if q not in (i, j)]
    for q in lagged_parents:
        for lag in range(1, n_lags + 1):
            cols.append(X[n_lags - lag: T - lag, q])

    design = np.column_stack([np.ones(len(y))] + cols)
    dof = len(y) - design.shape[1]
    if dof <= 0:
        return float("nan"), float("nan")      # test undefined, not "p = 1"

    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    sigma2 = float(resid @ resid) / dof
    var_i = sigma2 * float(np.linalg.pinv(design.T @ design)[1, 1])
    if not np.isfinite(var_i) or var_i <= 0.0:
        return float("nan"), float("nan")

    t_stat = float(coef[1] / np.sqrt(var_i))
    pval = float(2.0 * _scipy_stats.t.sf(abs(t_stat), dof))
    return t_stat, pval
