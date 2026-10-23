"""
pipeline_pcmci_gpdc.py — PCMCI (tigramite) with the non-linear GPDC test
========================================================================
Variant of pipeline_pcmci_tigramite.py for the NON-LINEAR scenarios. Two
things differ from the ParCorr wrapper, and nothing else:

  1) Conditional-independence test: GPDC (Gaussian-process regression of
     X and Y on the conditioning set Z, then distance correlation of the
     residuals) with significance='analytic', instead of ParCorr.
  2) Lag window: tau_max is EXACTLY the pipeline's `max_lag` (no
     min(max_lag, 10) cap, and no pcmci_tau_max override), so the
     baseline searches the same lags 1..max_lag as the DML pipeline.

Fixed configuration (identical in every scenario — nothing here is tuned
per dataset): GPDC_CONFIG below is the single source of truth and is
shared with pipeline_pcmciplus_gpdc.py. The GP hyper-parameters are
tigramite's defaults (gp_params=None) and the null distribution is the
library's analytic (pre-sampled) one.

Differences from the ParCorr wrapper that are consequences of GPDC, not
choices:
  - GPDC is STRICT here: if it cannot be built (missing `dcor` /
    scikit-learn) the call raises. The ParCorr wrapper silently falls back
    to ParCorr; doing that here would quietly turn this "non-linear"
    baseline back into the linear one.
  - The GPDC statistic is a one-sided distance correlation in [0, 1], not
    a partial correlation, so the Fisher-z confidence interval of the
    ParCorr wrapper is not valid. ci_lower/ci_upper are kept in the output
    for schema compatibility but are NaN.
  - The analytic p-value comes from a sig_samples=500 null sample, so its
    resolution is 1/500 (p can be exactly 0.0). This is tigramite's
    default and is left untouched to keep the configuration fixed.

Multiple-comparisons correction, alpha handling and the exhaustive
(pair x lag) p_matrix are the same as in pcmci_tigramite — see that
module's docstring.

Requires `pip install tigramite dcor scikit-learn`.
"""

import numpy as np
import pandas as pd
from typing import Optional, List

from Auxiliary.causal_common import _pcmci_load_and_pivot, _correct_pvalues, _cap_tigramite_tau_max


# Single, scenario-independent GPDC configuration (shared by pcmci_gpdc and
# pcmciplus_gpdc). Do not vary per dataset.
GPDC_CONFIG = dict(
    significance="analytic",
    gp_params=None,        # tigramite's default GP (RBF + white-noise kernel)
)


def _make_gpdc():
    """Builds the GPDC test with the fixed GPDC_CONFIG. Never falls back."""
    try:
        from tigramite.independence_tests.gpdc import GPDC
    except ImportError as e:
        raise ImportError(
            "pcmci_gpdc/pcmciplus_gpdc: GPDC unavailable "
            f"({e}). Install with:\n    pip install tigramite dcor scikit-learn"
        )
    return GPDC(**GPDC_CONFIG)


def run_pipeline_pcmci_gpdc(
    df: pd.DataFrame,
    date_col: str = "Data",
    entity_col: str = "Nome",
    property_cols: Optional[List[str]] = None,
    max_lag: int = 24,
    alpha_discovery: float = 0.20,
    pcmci_gpdc_pc_alpha: Optional[float] = None,
    pcmci_gpdc_alpha_level: Optional[float] = None,
    pcmci_gpdc_correction: str = "fdr_bh",
    **_kwargs,
) -> pd.DataFrame:
    """
    PCMCI + GPDC(significance='analytic') — drop-in in the runner:
    pipeline(df, **cfg) -> DataFrame with 'significant'.

    tau_max is not a parameter: it is always int(max_lag), the same lag
    window the DML pipeline uses.

    Method-specific parameters (prefix 'pcmci_gpdc_'):
      - pcmci_gpdc_pc_alpha    : alpha of the PC step (parent selection).
                                  Default: alpha_discovery.
      - pcmci_gpdc_alpha_level : final alpha, applied AFTER the
                                  multiple-comparisons correction.
                                  Default: alpha_discovery.
      - pcmci_gpdc_correction  : 'none' | 'bonferroni' | 'fdr_bh' over the
                                  full p_matrix. Default: 'fdr_bh'.

    Requires `pip install tigramite dcor scikit-learn`.
    """
    try:
        import tigramite.data_processing as _tg_pp
        from tigramite.pcmci import PCMCI as _TigramitePCMCI
    except ImportError:
        raise ImportError(
            "pcmci_gpdc: package 'tigramite' not found. Install with:\n"
            "    pip install tigramite dcor scikit-learn"
        )

    tau_max     = int(max_lag)
    pc_alpha    = pcmci_gpdc_pc_alpha if pcmci_gpdc_pc_alpha is not None else alpha_discovery
    alpha_level = pcmci_gpdc_alpha_level if pcmci_gpdc_alpha_level is not None else alpha_discovery

    pivoted = _pcmci_load_and_pivot(df, date_col=date_col, entity_col=entity_col,
                                     property_cols=property_cols)
    var_names = list(pivoted.columns)

    # per-column z-score standardization
    X = pivoted.values.astype(float)
    dev = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(dev < 1e-12, 1.0, dev)
    T, N = X.shape

    tau_max = _cap_tigramite_tau_max(tau_max, T, "pcmci_gpdc")
    if N < 2:
        raise ValueError(f"PCMCI GPDC: only {N} valid series — need >= 2.")

    print(f"    [pcmci_gpdc] N={N} variables, T={T} points, tau_max={tau_max} (= max_lag), "
          f"pc_alpha={pc_alpha}, alpha_level={alpha_level}, "
          f"cond_ind_test=gpdc({GPDC_CONFIG['significance']}), "
          f"correction={pcmci_gpdc_correction}")

    cit = _make_gpdc()
    dataframe = _tg_pp.DataFrame(X, var_names=var_names)
    pcmci_obj = _TigramitePCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=0)

    result     = pcmci_obj.run_pcmci(tau_max=tau_max, tau_min=1,
                                      pc_alpha=pc_alpha, alpha_level=alpha_level)
    p_matrix   = result["p_matrix"]    # (N, N, tau_max+1)
    val_matrix = result["val_matrix"]  # (N, N, tau_max+1)

    # -- Collect all p-values (i,j,lag; i!=j) to correct as a block -----------
    tested_pairs = []
    raw_pvals    = []
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
    if (pcmci_gpdc_correction or "none").lower() == "fdr_bh":
        q_matrix = pcmci_obj.get_corrected_pvalues(
            p_matrix=p_matrix, fdr_method="fdr_bh", exclude_contemporaneous=True,
            tau_min=1, tau_max=tau_max)
        corrected_pvals = [float(q_matrix[i, j, lag]) for (i, j, lag) in tested_pairs]
    else:
        corrected_pvals = _correct_pvalues(raw_pvals, pcmci_gpdc_correction)

    rows = []
    for (i, j, lag), pval, pval_corr in zip(tested_pairs, raw_pvals, corrected_pvals):
        src_entity, src_property = var_names[i].split("__", 1)
        dest_entity, dest_property = var_names[j].split("__", 1)
        rows.append({
            "source_entity":    src_entity,
            "source_property":  src_property,
            "target_entity":    dest_entity,
            "target_property":  dest_property,
            "lag":              lag,
            "value":            float(val_matrix[i, j, lag]),
            "ci_lower":         np.nan,   # no valid analytic CI for GPDC's dCor
            "ci_upper":         np.nan,
            "pvalue":           pval,
            "pvalue_corrected": pval_corr,
            "significant":      bool(pval_corr < alpha_level),
            "method":           "pcmci_gpdc",
        })

    return pd.DataFrame(rows, columns=[
        "source_entity", "source_property", "target_entity", "target_property",
        "lag", "value", "ci_lower", "ci_upper", "pvalue", "pvalue_corrected",
        "significant", "method",
    ])
