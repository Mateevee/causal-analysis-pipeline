"""
pipeline_pcmciplus.py — Wrapper for the official PCMCI+ (tigramite library)
============================================================================
Exposes run_pipeline_pcmciplus with the SAME call signature used by the
runner. Same loading/standardization as pcmci_tigramite
(_pcmci_load_and_pivot + per-column z-score) and the same conditional-
independence test factory (_pcmci_tigramite_get_cit).

PCMCI+ (Runge, UAI 2020) extends PCMCI to contemporaneous links: the MCI
tests of each lagged link also condition on the contemporaneous adjacencies,
followed by collider/Meek orientation of the lag-0 edges. Its relevant
output is the `graph` array, not the p_matrix — see
Auxiliary.causal_common._tigramite_graph_to_rows for how it is mapped to
'significant' (the native graph mark '-->'; a post-hoc corrected-p cut is
added only when pcmciplus_correction != 'none').

The algorithm always runs with tau_min=0 (otherwise it is not PCMCI+); only
lagged links are reported unless pcmciplus_include_contemporaneous=True,
in which case lag-0 links are reported as well: '-->' in its direction,
unoriented 'o-o' and conflicting 'x-x' adjacencies in both directions (the
Stage 8b convention; see _tigramite_graph_to_rows). The main run turns it on
(Config/execution.py), since the pipeline itself reports lag-0 links.

Requires `pip install tigramite`.
"""

import numpy as np
import pandas as pd
from typing import Optional, List

from Auxiliary.causal_common import _pcmci_load_and_pivot, _tigramite_graph_to_rows, _cap_tigramite_tau_max
from ModelComparison.pipeline_pcmci_tigramite import _pcmci_tigramite_get_cit


def run_pipeline_pcmciplus(
    df: pd.DataFrame,
    date_col: str = "Data",
    entity_col: str = "Nome",
    property_cols: Optional[List[str]] = None,
    max_lag: int = 24,
    alpha_discovery: float = 0.20,
    pcmciplus_tau_max: Optional[int] = None,
    pcmciplus_pc_alpha: Optional[float] = None,
    pcmciplus_alpha_level: Optional[float] = None,
    pcmciplus_cond_ind_test: str = "parcorr",
    pcmciplus_correction: str = "none",
    pcmciplus_fdr_method: str = "none",
    pcmciplus_contemp_collider_rule: str = "majority",
    pcmciplus_reset_lagged_links: bool = False,
    pcmciplus_include_contemporaneous: bool = False,
    **_kwargs,
) -> pd.DataFrame:
    """
    OFFICIAL PCMCI+ (PCMCI.run_pcmciplus from tigramite) — drop-in in the
    runner: pipeline(df, **cfg) -> DataFrame with 'significant'.

    Method-specific parameters (prefix 'pcmciplus_'):
      - pcmciplus_tau_max        : maximum lag. Default: min(max_lag, 10).
      - pcmciplus_pc_alpha       : alpha of the PC1 + skeleton phases (the
                                   algorithm's only significance level).
                                   Default: alpha_discovery.
      - pcmciplus_alpha_level    : alpha of the optional post-hoc cut
                                   (used only when pcmciplus_correction
                                   != 'none'). Default: alpha_discovery.
      - pcmciplus_cond_ind_test  : 'parcorr' | 'gpdc' | 'cmiknn'.
                                   Default: 'parcorr'.
      - pcmciplus_correction     : 'none' (default) = NATIVE output: the
                                   decision is PCMCI+'s own graph, decided
                                   at pc_alpha. 'bonferroni' | 'fdr_bh' add a
                                   post-hoc cut (graph AND corrected p <
                                   alpha_level) that the published method
                                   does not make — sensitivity use only.
      - pcmciplus_fdr_method     : tigramite's NATIVE FDR option of
                                   run_pcmciplus ('none' | 'fdr_bh'),
                                   applied inside the skeleton phase.
                                   Default: 'none' (tigramite's default).
      - pcmciplus_contemp_collider_rule : 'majority' | 'conservative' |
                                   'none'. Default: 'majority' (tigramite's
                                   default; order-independent).
      - pcmciplus_reset_lagged_links : re-test every lagged link in the
                                   skeleton phase, not only the PC1 parents
                                   (more power, slower). Default: False
                                   (tigramite's own default). This is a
                                   structural trait of PCMCI+, not a bug:
                                   with the default, a lagged link that the
                                   conservative PC1 condition-selection
                                   (pc_alpha) drops never reaches the MCI
                                   test, so PCMCI+ naturally reports far
                                   fewer candidate pairs — and lower recall
                                   — than pcmci_tigramite, which tests every
                                   (i, j, lag) exhaustively regardless of
                                   PC1 (see that module's own docstring).
                                   Kept False to match the library's
                                   intended default rather than tune away
                                   an authentic property of the algorithm.
      - pcmciplus_include_contemporaneous : also report lag-0 links
                                   ('-->' at lag 0 in its direction; 'o-o'
                                   and 'x-x' in both directions). Default:
                                   False (the main run sets True).

    Requires `pip install tigramite`.
    """
    try:
        import tigramite.data_processing as _tg_pp
        from tigramite.pcmci import PCMCI as _TigramitePCMCI
    except ImportError:
        raise ImportError(
            "pcmciplus: package 'tigramite' not found. Install with:\n"
            "    pip install tigramite"
        )

    tau_max     = pcmciplus_tau_max if pcmciplus_tau_max is not None else min(int(max_lag), 10)
    pc_alpha    = pcmciplus_pc_alpha if pcmciplus_pc_alpha is not None else alpha_discovery
    alpha_level = pcmciplus_alpha_level if pcmciplus_alpha_level is not None else alpha_discovery

    pivoted = _pcmci_load_and_pivot(df, date_col=date_col, entity_col=entity_col,
                                     property_cols=property_cols)
    var_names = list(pivoted.columns)

    # per-column z-score standardization
    X = pivoted.values.astype(float)
    dev = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(dev < 1e-12, 1.0, dev)
    T, N = X.shape

    tau_max = _cap_tigramite_tau_max(tau_max, T, "pcmciplus")
    if N < 2:
        raise ValueError(f"PCMCI+: only {N} valid series — need >= 2.")

    print(f"    [pcmciplus] N={N} variables, T={T} points, tau_max={tau_max}, "
          f"pc_alpha={pc_alpha}, alpha_level={alpha_level}, "
          f"cond_ind_test={pcmciplus_cond_ind_test}, correction={pcmciplus_correction}, "
          f"fdr_method={pcmciplus_fdr_method}, "
          f"collider_rule={pcmciplus_contemp_collider_rule}, "
          f"reset_lagged_links={pcmciplus_reset_lagged_links}, "
          f"include_contemporaneous={pcmciplus_include_contemporaneous}")

    cit = _pcmci_tigramite_get_cit(pcmciplus_cond_ind_test)
    dataframe = _tg_pp.DataFrame(X, var_names=var_names)
    pcmci_obj = _TigramitePCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=0)

    result = pcmci_obj.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        contemp_collider_rule=pcmciplus_contemp_collider_rule,
        reset_lagged_links=pcmciplus_reset_lagged_links,
        fdr_method=pcmciplus_fdr_method,
    )

    # Lagged links are always oriented by time order, so '-->' is the only
    # mark that can appear at lag >= 1. At lag 0 (reported only with
    # include_contemporaneous=True) '-->' is a link oriented i -> j, and the
    # unoriented o-o / conflicting x-x adjacencies count in both directions
    # (contemporaneous_marks default of _tigramite_graph_to_rows).
    return _tigramite_graph_to_rows(
        graph=result["graph"],
        p_matrix=result["p_matrix"],
        val_matrix=result["val_matrix"],
        var_names=var_names,
        tau_max=tau_max,
        accepted_marks=("-->",),
        correction=pcmciplus_correction,
        alpha_level=alpha_level,
        method="pcmciplus",
        include_contemporaneous=pcmciplus_include_contemporaneous,
    )
