"""
pipeline_pcmci_tigramite.py — Wrapper for the official PCMCI (tigramite library)
====================================================================================
Exposes run_pipeline_pcmci_tigramite with the SAME call signature used by
the runner. Reuses _pcmci_load_and_pivot and _correct_pvalues from
causal_common for data loading and multiple-comparisons correction, using
the same z-score standardization as the other pipelines in this package.

Structural property of the algorithm (not a bug): tigramite returns an
EXHAUSTIVE p_matrix/val_matrix — a p-value for EVERY pair of variables
(i, j) with i != j, at EVERY lag from 1 to tau_max — rather than only the
candidates that survive a prior screen. This tends to produce a larger
output table (and potentially more false positives under the same alpha,
since more tests are reported). This is structural to the "textbook"
PCMCI algorithm and is made explicit here on purpose, rather than hidden.

Precisely because it exhaustively tests every pair x lag with no prior
screening, this is exactly where a correction for multiple comparisons
matters most — without it, the expected number of false positives under
H0 grows linearly with N²*tau_max, even at a low nominal alpha. Correction
over the full p_matrix (Benjamini-Hochberg/FDR or Bonferroni) is
controllable via pcmci_correction (default 'fdr_bh').

Requires `pip install tigramite`. The ImportError is only raised when the
function is actually called, not on import of this module — the runner
(run_all) already catches and logs this error per model/dataset without
interrupting the others.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from scipy import stats as _scipy_stats

from Auxiliary.causal_common import _pcmci_load_and_pivot, _correct_pvalues, _cap_tigramite_tau_max


def _pcmci_tigramite_get_cit(name: str):
    """Instantiates tigramite's conditional-independence test."""
    name = (name or "parcorr").lower()
    if name == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr
        return ParCorr()
    elif name == "gpdc":
        try:
            from tigramite.independence_tests.gpdc import GPDC
            return GPDC()
        except ImportError:
            print("    [pcmci_tigramite][WARNING] GPDC unavailable, using ParCorr.")
            from tigramite.independence_tests.parcorr import ParCorr
            return ParCorr()
    elif name == "cmiknn":
        try:
            from tigramite.independence_tests.cmiknn import CMIknn
            return CMIknn()
        except ImportError:
            print("    [pcmci_tigramite][WARNING] CMIknn unavailable, using ParCorr.")
            from tigramite.independence_tests.parcorr import ParCorr
            return ParCorr()
    else:
        print(f"    [pcmci_tigramite][WARNING] unknown CIT '{name}', using ParCorr.")
        from tigramite.independence_tests.parcorr import ParCorr
        return ParCorr()


def run_pipeline_pcmci_tigramite(
    df: pd.DataFrame,
    date_col: str = "Data",
    entity_col: str = "Nome",
    property_cols: Optional[List[str]] = None,
    max_lag: int = 24,
    alpha_discovery: float = 0.20,
    pcmci_tau_max: Optional[int] = None,
    pcmci_pc_alpha: Optional[float] = None,
    pcmci_alpha_level: Optional[float] = None,
    pcmci_cond_ind_test: str = "parcorr",
    pcmci_correction: str = "fdr_bh",
    **_kwargs,
) -> pd.DataFrame:
    """
    OFFICIAL PCMCI (the PCMCI class from the tigramite library) — drop-in
    in the runner: pipeline(df, **cfg) -> DataFrame with 'significant'.

    Method-specific parameters (prefix 'pcmci_'):
      - pcmci_tau_max       : maximum lag tested. Default: min(max_lag, 10).
      - pcmci_pc_alpha      : alpha of the PC step (parent selection).
                               Default: alpha_discovery.
      - pcmci_alpha_level   : final alpha (MCI significance), applied
                               AFTER the multiple-comparisons correction.
                               Default: alpha_discovery.
      - pcmci_cond_ind_test : tigramite's conditional-independence
                               test — 'parcorr' (linear, fast),
                               'gpdc' (non-linear GP, slow) or 'cmiknn'
                               (CMI via KNN, very slow). Default: 'parcorr'.
      - pcmci_correction    : multiple-comparisons correction applied
                               to the full p_matrix (every pair x lag) before
                               deciding 'significant'. 'none' |
                               'bonferroni' | 'fdr_bh'. 'fdr_bh' is
                               tigramite's own BH over every lagged link,
                               auto-links included (m = N^2 * tau_max, as
                               in the paper). Default: 'fdr_bh' --
                               without correction, this path is exposed to
                               false positives from multiple comparisons,
                               since it exhaustively tests every pair x lag
                               with no prior screening.

    Requires `pip install tigramite`.
    """
    try:
        import tigramite.data_processing as _tg_pp
        from tigramite.pcmci import PCMCI as _TigramitePCMCI
    except ImportError:
        raise ImportError(
            "pcmci_tigramite: package 'tigramite' not found. Install with:\n"
            "    pip install tigramite"
        )

    tau_max     = pcmci_tau_max if pcmci_tau_max is not None else min(int(max_lag), 10)
    pc_alpha    = pcmci_pc_alpha if pcmci_pc_alpha is not None else alpha_discovery
    alpha_level = pcmci_alpha_level if pcmci_alpha_level is not None else alpha_discovery

    pivoted = _pcmci_load_and_pivot(df, date_col=date_col, entity_col=entity_col,
                                     property_cols=property_cols)
    var_names = list(pivoted.columns)

    # per-column z-score standardization
    X = pivoted.values.astype(float)
    dev = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(dev < 1e-12, 1.0, dev)
    T, N = X.shape

    tau_max = _cap_tigramite_tau_max(tau_max, T, "pcmci_tigramite")
    if N < 2:
        raise ValueError(f"PCMCI tigramite: only {N} valid series — need >= 2.")

    print(f"    [pcmci_tigramite] N={N} variables, T={T} points, tau_max={tau_max}, "
          f"pc_alpha={pc_alpha}, alpha_level={alpha_level}, "
          f"cond_ind_test={pcmci_cond_ind_test}, correction={pcmci_correction}")

    cit = _pcmci_tigramite_get_cit(pcmci_cond_ind_test)
    dataframe = _tg_pp.DataFrame(X, var_names=var_names)
    pcmci_obj = _TigramitePCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=0)

    result     = pcmci_obj.run_pcmci(tau_max=tau_max, tau_min=1,
                                      pc_alpha=pc_alpha, alpha_level=alpha_level)
    p_matrix   = result["p_matrix"]    # (N, N, tau_max+1)
    val_matrix = result["val_matrix"]  # (N, N, tau_max+1)

    # -- Confidence intervals (90%) via Fisher transform ----------------------
    # Computed analytically from val_matrix — the same method that
    # tigramite's ParCorr uses internally in get_confidence(), but without
    # depending on the package's internals or version.
    #
    # For the partial correlation r with degrees of freedom df:
    #   z(r)    = arctanh(r)           (Fisher z-transform)
    #   SE      = 1 / sqrt(df - 1)
    #   90% CI  = [tanh(z - z_crit*SE), tanh(z + z_crit*SE)]
    #
    # Conservative df: T - tau_max - 2  (ignores the dimension of the
    # conditioning set, which varies per pair — slightly underestimates the
    # real CI width, but is always finite and computable without access to
    # the MCI parents).
    _conf_lev  = 0.9
    _z_crit    = float(_scipy_stats.norm.ppf(0.5 + _conf_lev / 2.0))
    _df_approx = max(T - tau_max - 2, 1)
    _se_z      = 1.0 / max(np.sqrt(_df_approx - 1), 1e-12)

    # -- Collect all p-values (i,j,lag; i!=j) to correct as a block -----------
    tested_pairs   = []
    raw_pvals      = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for lag in range(1, tau_max + 1):
                tested_pairs.append((i, j, lag))
                raw_pvals.append(float(p_matrix[i, j, lag]))

    # 'fdr_bh' = tigramite's OWN Benjamini-Hochberg (get_corrected_pvalues,
    # what run_pcmci(fdr_method='fdr_bh') applies): the family is every
    # lagged link incl. the auto-links, m = N^2 * tau_max, as in the paper
    # (Runge et al. 2019, Sec. S1 "False discovery rate control").
    if (pcmci_correction or "none").lower() == "fdr_bh":
        q_matrix = pcmci_obj.get_corrected_pvalues(
            p_matrix=p_matrix, fdr_method="fdr_bh", exclude_contemporaneous=True,
            tau_min=1, tau_max=tau_max)
        corrected_pvals = [float(q_matrix[i, j, lag]) for (i, j, lag) in tested_pairs]
    else:
        corrected_pvals = _correct_pvalues(raw_pvals, pcmci_correction)
    corrected_pval_of = dict(zip(tested_pairs, corrected_pvals))

    rows = []
    for idx, (i, j, lag) in enumerate(tested_pairs):
        src_entity, src_property = var_names[i].split("__", 1)
        dest_entity, dest_property = var_names[j].split("__", 1)
        val  = float(val_matrix[i, j, lag])
        pval = raw_pvals[idx]
        pval_corr = corrected_pval_of[(i, j, lag)]
        r_clip = float(np.clip(val, -0.999999, 0.999999))
        z_r    = float(np.arctanh(r_clip))
        ci_low  = float(np.tanh(z_r - _z_crit * _se_z))
        ci_high = float(np.tanh(z_r + _z_crit * _se_z))
        rows.append({
            "source_entity":    src_entity,
            "source_property":  src_property,
            "target_entity":   dest_entity,
            "target_property": dest_property,
            "lag":            lag,
            "value":          val,
            "ci_lower":         ci_low,
            "ci_upper":        ci_high,
            "pvalue":        pval,
            "pvalue_corrected": pval_corr,
            "significant":  bool(pval_corr < alpha_level),
            "method":         "pcmci_tigramite",
        })

    return pd.DataFrame(rows, columns=[
        "source_entity", "source_property", "target_entity", "target_property",
        "lag", "value", "ci_lower", "ci_upper", "pvalue", "pvalue_corrected",
        "significant", "method",
    ])
