"""
pipeline_pcmciplus_gpdc.py — PCMCI+ (tigramite) with the non-linear GPDC test
============================================================================
Variant of pipeline_pcmciplus.py for the NON-LINEAR scenarios. Two things
differ from the ParCorr wrapper, and nothing else:

  1) Conditional-independence test: GPDC with significance='analytic'
     (instead of ParCorr), built by the same strict factory used by
     pipeline_pcmci_gpdc.py (fixed GPDC_CONFIG, identical in every
     scenario, no silent ParCorr fallback).
  2) Lag window: tau_max is EXACTLY the pipeline's `max_lag` (no
     min(max_lag, 10) cap, and no pcmciplus_tau_max override), so the
     baseline searches the same lags 1..max_lag as the DML pipeline.

Everything else — loading/z-score, tau_min=0 (contemporaneous
conditioning), collider rule, reset_lagged_links=False (tigramite's
default), and the graph-mark + corrected-p decision in
Auxiliary.causal_common._tigramite_graph_to_rows — is the same as in
pcmciplus. Only lagged links are reported unless
pcmciplus_gpdc_include_contemporaneous=True ('-->' at lag 0 in its
direction, 'o-o'/'x-x' in both directions -- same convention as pcmciplus
and the pipeline's Stage 8b). main_gpdc.py sets it True.

See pipeline_pcmci_gpdc.py for the GPDC-specific caveats (one-sided
distance-correlation statistic, p-value resolution of 1/500).

Requires `pip install tigramite dcor scikit-learn`.
"""

import numpy as np
import pandas as pd
from typing import Optional, List

from Auxiliary.causal_common import _pcmci_load_and_pivot, _tigramite_graph_to_rows, _cap_tigramite_tau_max
from ModelComparison.pipeline_pcmci_gpdc import _make_gpdc, GPDC_CONFIG


def run_pipeline_pcmciplus_gpdc(
    df: pd.DataFrame,
    date_col: str = "Data",
    entity_col: str = "Nome",
    property_cols: Optional[List[str]] = None,
    max_lag: int = 24,
    alpha_discovery: float = 0.20,
    pcmciplus_gpdc_pc_alpha: Optional[float] = None,
    pcmciplus_gpdc_alpha_level: Optional[float] = None,
    pcmciplus_gpdc_correction: str = "none",
    pcmciplus_gpdc_fdr_method: str = "none",
    pcmciplus_gpdc_contemp_collider_rule: str = "majority",
    pcmciplus_gpdc_reset_lagged_links: bool = False,
    pcmciplus_gpdc_include_contemporaneous: bool = False,
    **_kwargs,
) -> pd.DataFrame:
    """
    PCMCI+ (PCMCI.run_pcmciplus) + GPDC(significance='analytic') — drop-in
    in the runner: pipeline(df, **cfg) -> DataFrame with 'significant'.

    tau_max is not a parameter: it is always int(max_lag), the same lag
    window the DML pipeline uses.

    Method-specific parameters (prefix 'pcmciplus_gpdc_'), same meaning and
    defaults as the 'pcmciplus_' ones in pipeline_pcmciplus.py:
      - pcmciplus_gpdc_pc_alpha, pcmciplus_gpdc_alpha_level
            (default: alpha_discovery)
      - pcmciplus_gpdc_correction                : 'none' (native graph)
      - pcmciplus_gpdc_fdr_method                : 'none' (tigramite's own FDR)
      - pcmciplus_gpdc_contemp_collider_rule     : 'majority'
      - pcmciplus_gpdc_reset_lagged_links        : False
      - pcmciplus_gpdc_include_contemporaneous   : False (main_gpdc.py sets True)

    Requires `pip install tigramite dcor scikit-learn`.
    """
    try:
        import tigramite.data_processing as _tg_pp
        from tigramite.pcmci import PCMCI as _TigramitePCMCI
    except ImportError:
        raise ImportError(
            "pcmciplus_gpdc: package 'tigramite' not found. Install with:\n"
            "    pip install tigramite dcor scikit-learn"
        )

    tau_max     = int(max_lag)
    pc_alpha    = (pcmciplus_gpdc_pc_alpha if pcmciplus_gpdc_pc_alpha is not None
                   else alpha_discovery)
    alpha_level = (pcmciplus_gpdc_alpha_level if pcmciplus_gpdc_alpha_level is not None
                   else alpha_discovery)

    pivoted = _pcmci_load_and_pivot(df, date_col=date_col, entity_col=entity_col,
                                     property_cols=property_cols)
    var_names = list(pivoted.columns)

    # per-column z-score standardization
    X = pivoted.values.astype(float)
    dev = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(dev < 1e-12, 1.0, dev)
    T, N = X.shape

    tau_max = _cap_tigramite_tau_max(tau_max, T, "pcmciplus_gpdc")
    if N < 2:
        raise ValueError(f"PCMCI+ GPDC: only {N} valid series — need >= 2.")

    print(f"    [pcmciplus_gpdc] N={N} variables, T={T} points, tau_max={tau_max} (= max_lag), "
          f"pc_alpha={pc_alpha}, alpha_level={alpha_level}, "
          f"cond_ind_test=gpdc({GPDC_CONFIG['significance']}), "
          f"correction={pcmciplus_gpdc_correction}, fdr_method={pcmciplus_gpdc_fdr_method}, "
          f"collider_rule={pcmciplus_gpdc_contemp_collider_rule}, "
          f"reset_lagged_links={pcmciplus_gpdc_reset_lagged_links}, "
          f"include_contemporaneous={pcmciplus_gpdc_include_contemporaneous}")

    cit = _make_gpdc()
    dataframe = _tg_pp.DataFrame(X, var_names=var_names)
    pcmci_obj = _TigramitePCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=0)

    result = pcmci_obj.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        contemp_collider_rule=pcmciplus_gpdc_contemp_collider_rule,
        reset_lagged_links=pcmciplus_gpdc_reset_lagged_links,
        fdr_method=pcmciplus_gpdc_fdr_method,
    )

    return _tigramite_graph_to_rows(
        graph=result["graph"],
        p_matrix=result["p_matrix"],
        val_matrix=result["val_matrix"],
        var_names=var_names,
        tau_max=tau_max,
        accepted_marks=("-->",),
        correction=pcmciplus_gpdc_correction,
        alpha_level=alpha_level,
        method="pcmciplus_gpdc",
        include_contemporaneous=pcmciplus_gpdc_include_contemporaneous,
    )
