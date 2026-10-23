"""
pipeline.py — MAIN PIPELINE (high-level orchestration).

This file contains NO math for any stage — `run_pipeline` only runs the
stage functions below one after the other, each delegating to its own
module. The numbering is the one of the paper's pipeline figure:

    (input)   structuring              pivot.py             load_and_pivot   (before Stage 1)
    Stage 1   active-property detection property_scanner.py scan_entity_properties
    Stage 2   preprocessing            preprocessing.py     preprocess_all
    Stages 3-5 correlation pre-filter  discovery.py         filter_causal_candidates
              (3 cross-correlation, 4 Transfer Entropy, 5 rescues)
    Stage 6   cascaded DML             dml_engine.py        estimate_effects_by_control_set
                                       results.py           build_results / apply_statistical_checks
                                       coherence.py         assess_evidence_coherence
    Stage 7   evidence-based filtering dml_engine.py        apply_causal_penalty
    Stage 8   graph-level structural adjustment
        8a    structural pruning       structural.py        apply_structural_filter
        8a-bis                         predictive.py        predictive_competition
        8-guard regime guards          coherence.py         keep_stage6_pairs
                                       contemporaneous.py   drop_lagged_without_time_axis
        8b    contemporaneous completion contemporaneous.py   append_contemporaneous_links
        8-out output invariant         results.py           require_effect_estimate

Three data-driven regime checks (no ground truth, no dataset name) decide
how the stages above behave on a given dataset:

    near-deterministic dynamics  (Stage 6)  the DML control set falls back
        from parents + source block to the blanket design, and the
        predictive competition is attempted after 8a;
    incoherent evidence          (Stage 6)  the Stage 7-8a rejections are
        not applied: the pairs with a significant Stage 6 effect are kept
        and the output is flagged;
    no time axis                 (8-guard)  every lagged edge is dropped and
        8b describes the whole graph.

`dml_engine.py` natively runs the BR -> RF -> KRR -> NadarayaWatson
cascade (with nonlinearity diagnostics, lag median, etc. — see that
module's own docstring). The `dml_engine_module` parameter exists only for
advanced use cases (e.g. tests) where one wants to inject an alternative
dml_engine implementation; in normal use it stays None.
"""

from typing import Optional
import pandas as pd

from . import pivot
from . import property_scanner
from . import preprocessing
from . import discovery
from . import dml_engine as _default_dml_engine
from . import results as results_mod
from . import coherence as coherence_mod
from . import predictive as predictive_mod
from . import contemporaneous as contemp_mod
from .structural import apply_structural_filter

from .penalties import (
    LagPenaltyFn, CausalPenaltyFn,
    UniformCausalPenalty, DEFAULT_CAUSAL_PENALTY,
)
from .SelectorFunc import AdaptiveCausalPenalty


def run_pipeline(
    df: pd.DataFrame,
    date_col: str = "Date",
    entity_col: str = "Name",
    property_cols: Optional[list] = None,
    max_lag: int = 10,
    alpha_discovery: float = 0.10,
    # ── Stage 1: active-property detection ───────────────────────────
    use_property_scanner: bool = True,
    scanner_min_active_pct: float = 0.09,
    scanner_zero_threshold: float = 1e-3,
    # ── Stage 2: pre-processing ──────────────────────────────────────
    gap_limit: Optional[int] = 3,
    zero_run_min_length: Optional[int] = 2,
    # ── Stage 3: cross-correlation ───────────────────────────────────
    use_xcorr: bool = True,
    min_corr: float = 0.05,
    # ── Stage 4: Transfer Entropy ────────────────────────────────────
    use_te: bool = True,
    min_te: float = 0.005,
    te_method: str = 'histogram',
    te_bins: int = 6,
    te_k: int = 4,
    n_perms: int = 50,
    xcorr_te_correction: str = 'fdr_bh',
    correct_lag_multiplicity: bool = False,
    n_perms_lag_correction: Optional[int] = None,
    # ── Stage 5: rescues ─────────────────────────────────────────────
    use_granger_rescue: bool = True,
    alpha_granger: Optional[float] = None,
    use_pearson_rescue: bool = True,
    alpha_pearson: Optional[float] = None,
    min_corr_rescue: float = 0.20,
    n_boot: int = 200,
    boot_block_size: Optional[int] = None,
    rescue_correct_lag_multiplicity: bool = False,
    # ── Stages 3-5: lag penalty lambda(tau) and pair universe ────────
    lag_penalty_fn: Optional[LagPenaltyFn] = None,
    use_lag_penalty: bool = True,
    lag_knee1: int = 4,
    lag_knee2: int = 6,
    lag_rate_zone1: float = 0.02,
    lag_rate_zone2: float = 0.15,
    lag_rate_zone3: float = 0.50,
    block_same_property: bool = True,
    focus_entity: Optional[object] = None,
    include_self: bool = False,
    target_entity: Optional[object] = None,
    # ── Stage 6: cascaded DML ────────────────────────────────────────
    dml_splits: int = 5,
    dml_model: str = 'auto',
    var_res_T_threshold: float = 0.05,
    dml_nuisance_max_controls: Optional[int] = None,
    dml_auto_nuisance_screening: bool = False,
    dml_estimate_selection: Optional[str] = None,
    # Control set of the DML nuisances: 'blanket' (lags 1..window of every
    # series), 'parents' (candidate parents of the pair, each at its own
    # lag) or 'auto' (parents + source block, falling back to blanket when
    # the outcome is near-deterministic);
    # dml_source_block tests the source as a block of lags. See
    # dml_engine.estimate_effects_by_control_set / _parent_controls /
    # CascadeConfig.cascade_decision.
    dml_control_set: str = 'blanket',
    dml_source_block: bool = False,
    dml_cascade_decision: Optional[str] = None,
    dml_auto_max_outcome_r2: float = 0.5,
    # Evidence-coherence check (Pipeline/coherence.py): assessed at the end
    # of Stage 6, applied by the regime guards of Stage 8 (incoherent ->
    # the decision falls back to the Stage 6 effect). Off by default.
    evidence_coherence_guard: bool = False,
    evidence_coherence_min_rho: float = 0.3,
    evidence_coherence_alpha: float = 0.05,
    # What the incoherent regime falls back to. 'stage6' (default): every
    # pair with a significant Stage 6 effect is kept. 'auto': the same, EXCEPT when the past explains little
    # of the series (median out-of-sample R2 <= contemporaneous_max_past_r2,
    # the gate of Stage 8b) -- there the Stage 6 fallback is not applied and
    # the Stage 7/8a decision stands. See _stage8_regime_guards.
    evidence_coherence_fallback: str = 'stage6',
    # A refinement of 'auto' (off by default; it does nothing under
    # 'stage6'): evidence_fallback_own_past_floor -- the R2 compared with
    # the gate is the larger of the all-series estimate and the own-past
    # estimate (contemporaneous.own_past_explained_variance), so a short
    # series, where the all-series regression underestimates the dynamics,
    # is not mistaken for one without dynamics (rationale and references in
    # that function's docstring: Granger 1969; Lütkepohl 2005, Sec. 3.5;
    # Bańbura, Giannone & Reichlin 2010).
    evidence_fallback_own_past_floor: bool = False,
    # ── Stage 7: evidence-based filtering ────────────────────────────
    causal_penalty_fn: Optional[CausalPenaltyFn] = None,
    causal_drop_zero: bool = False,
    # ── Stage 8a: structural pruning (structural.py) ─────────────────
    use_structural_filter: bool = True,
    structural_max_iter: int = 5,
    structural_n_boot: int = 100,
    structural_stability_threshold: float = 0.6,
    structural_use_bic_pruning: bool = True,
    structural_hac_lags: Optional[int] = None,
    structural_method: str = 'cmi',
    # -- conditional-test power (Z dimensionality + linear rescue) --
    structural_cmi_max_cond: Optional[int] = 4,
    # Test each candidate as a block of lags, conditioning also on the
    # target's own past (structural._parent_competition_for_target_cmi).
    # 'auto' follows Stage 6: block testing whenever the DML ran with the
    # parent-based + source-block design.
    structural_cmi_block: object = False,
    structural_cmi_linear_rescue_alpha: float = 0.0001,
    structural_cmi_n_perm: int = 200,
    # -- reciprocal-edge orientation + sign consistency --
    structural_resolve_reciprocal: bool = True,
    structural_reciprocal_min_margin: float = 0.10,
    structural_reciprocal_min_criteria: int = 2,
    structural_reciprocal_protect_pvalue: Optional[float] = None,
    structural_reciprocal_max_lag_diff: Optional[int] = None,
    structural_reciprocal_max_degree: Optional[int] = 6,
    structural_reciprocal_only_lag_zero: bool = False,
    structural_sign_consistency: bool = True,
    structural_sign_alpha: float = 0.05,
    structural_sign_min_ratio: float = 0.0,
    # -- robustness gate for edges with no competitor (target=1 parent) --
    structural_lone_edge_gate: bool = True,
    structural_lone_edge_robustness_floor: float = 0.18,
    structural_lone_edge_cv_ci_ceiling: float = 1.2,
    structural_lone_edge_min_weak_signals: int = 2,
    structural_lone_edge_require_unanimity: bool = False,
    structural_lone_edge_majority_fallback: bool = False,
    structural_lone_edge_veto_bh: bool = False,
    structural_lone_edge_veto_pvalue: Optional[float] = None,  # no effect for lone edges -- see docstring
    structural_lone_edge_dry_run: bool = False,
    structural_exact_mediation_gate: bool = False,
    structural_spurious_hub_gate: bool = False,
    structural_hub_zscore_threshold: float = 3.5,
    structural_hub_min_origins: int = 5,
    structural_lag_max_significant: Optional[int] = None,
    # -- BIC pruning (protection + penalty scale + rescue) --
    structural_bic_protect_pvalue: Optional[float] = None,
    structural_bic_penalty_scale: float = 1.0,
    structural_bic_max_degree: Optional[int] = 3,
    structural_bic_min_collinearity_to_prune: Optional[float] = None,
    structural_bic_rescue: bool = True,
    structural_bic_rescue_max_collinearity: float = 0.3,
    structural_bic_rescue_min_robustness: float = 0.16,
    # Predictive parent competition (8a-bis, Pipeline/predictive.py), last
    # step of 8a. False = off, True = always attempted, 'auto' = attempted on the
    # datasets where dml_control_set='auto' found near-deterministic
    # dynamics (blanket fallback). It only replaces the decision when its
    # importances split into two populations; otherwise it abstains.
    predictive_competition: object = False,
    predictive_min_delta_bic: float = 6.0,
    # ── Stage 8 regime guards ────────────────────────────────────────
    # Drop every lagged edge when the data has no genuine time axis
    # (discovery._detect_time_axis). Off by default.
    lagged_requires_time_axis: bool = False,
    # ── Stage 8b: contemporaneous completion (lag-0 links) ───────────
    # See Pipeline/contemporaneous.py. Off by default.
    contemporaneous_links: bool = False,
    contemporaneous_alpha: float = 0.01,
    contemporaneous_max_past_r2: Optional[float] = 0.25,
    contemporaneous_orientation: str = 'both',
    contemporaneous_skeleton: str = 'full',
    contemporaneous_pc_max_cond: int = 4,
    # Every 8b link gets a DML effect estimate (lag-0 treatment, past +
    # contemporaneous-neighbour controls). It quantifies only; with
    # contemporaneous_require_dml=True a link is kept only if that
    # estimate is also significant.
    contemporaneous_require_dml: bool = False,
    # ── Output ───────────────────────────────────────────────────────
    top_n: Optional[int] = None,
    n_jobs: int = 1,
    # ── Pair tracking ────────────────────────────────────────────────
    watch_pairs: Optional['object'] = None,
    # ── Alternative DML engine (advanced/testing use — see module
    #    docstring) ──
    dml_engine_module=None,
) -> pd.DataFrame:
    """
    Full all-to-all pipeline: the stages of `_STAGES`, in order (see the
    module docstring for the stage map).

    DML cascade: BR -> RF -> KRR -> NadarayaWatson, native to
    dml_engine.py — see that module's docstring for the
    acceptance/rejection criteria and the diagnostics attached to the
    result.
    """
    params = dict(locals())  # every argument, by name -- must stay first

    print("=" * 65)
    print("CAUSAL DISCOVERY PIPELINE — all-to-all (modular)")
    print("=" * 65)

    run = _Run(params)
    run.causal_pen_fn = _resolve_causal_penalty(causal_penalty_fn)

    for stage in _STAGES:
        stage(run)
        if run.stopped:
            return pd.DataFrame()

    return _finalize(run)


# =============================================================================
# RUN STATE
# =============================================================================

class _Run:
    """What one pipeline run carries from stage to stage."""

    def __init__(self, params: dict):
        self.p = params                    # run_pipeline arguments, by name
        self.dml_engine = (params.get('dml_engine_module')
                           or _default_dml_engine)
        self.causal_pen_fn = None
        self.df_wide = None
        self.pairs = None                  # active "ENTITY__property" series
        self.processed = None              # series after Stage 2
        self.candidates = None             # pairs confirmed by Stages 3-5
        self.results = pd.DataFrame()      # one row per candidate (Stage 6 on)
        # Regime decided from the data at Stage 6:
        self.lean_design = False           # DML ran with parents + source block
        self.near_deterministic = False    # ... and fell back to blanket
        self.coherence = None              # assess_evidence_coherence output
        # Pair-space progression by stage.
        self.counts = {}
        self.stopped = False               # a stage left nothing to analyze

    def has_decisions(self) -> bool:
        return not self.results.empty and 'significant' in self.results.columns

    def n_significant(self) -> int:
        return int((self.results['significant'] == True).sum())  # noqa: E712


# =============================================================================
# STAGES
# =============================================================================

def _stage0_structure(run: _Run) -> None:
    p = run.p
    print("\n[Stage 0] Structuring data...")
    run.df_wide, run.pairs, _entities, _properties, _dates = pivot.load_and_pivot(
        p['df'], p['date_col'], p['entity_col'], p['property_cols']
    )
    n_series = len(run.pairs)
    raw_pair_space = n_series * (n_series - 1) if n_series > 1 else 0
    run.counts['0_series_universe'] = n_series
    run.counts['0_raw_pair_space'] = raw_pair_space
    print(f"  Series (entity,property): {n_series:,} "
          f"-> raw directional pair space: {raw_pair_space:,}")


def _stage1_active_properties(run: _Run) -> None:
    p = run.p
    if p['use_property_scanner']:
        print("\n[Stage 1] Identifying characteristic properties per entity...")
        scan_df = property_scanner.scan_entity_properties(
            p['df'], date_col=p['date_col'], entity_col=p['entity_col'],
            property_cols=p['property_cols'],
            zero_threshold=p['scanner_zero_threshold'],
            min_active_pct=p['scanner_min_active_pct'], verbose=True,
        )
        active_pairs = property_scanner.get_active_pairs(scan_df)
        pairs_before = len(run.pairs)
        run.pairs = [s for s in run.pairs if s in active_pairs]
        removed = pairs_before - len(run.pairs)
        if removed > 0:
            print(f"  {removed} (entity, property) pairs removed for inactivity.")
        n = len(run.pairs)
        print(f"  Active series: {n:,} of {pairs_before:,} "
              f"-> post-Stage-1 directional pair space: "
              f"{n * (n - 1) if n > 1 else 0:,}")
        if not run.pairs:
            print("  No active pair found. Adjust scanner_min_active_pct.")
            run.stopped = True
            return
    run.counts['1_active_series'] = len(run.pairs)


def _stage2_preprocessing(run: _Run) -> None:
    p = run.p
    print("\n[Stage 2] Preprocessing series...")
    cols_wide = [c for c in run.df_wide.columns if c in run.pairs]
    run.df_wide = run.df_wide[cols_wide]
    run.processed, _zero_run_flags = preprocessing.preprocess_all(
        run.df_wide, gap_limit=p['gap_limit'], zero_run_min_length=p['zero_run_min_length'],
    )


def _stage3to5_prefilter(run: _Run) -> None:
    p = run.p
    print("\n[Stages 3-5] Correlation pre-filter (xcorr + TE + rescues)...")
    run.candidates = discovery.filter_causal_candidates(
        run.processed, run.pairs,
        max_lag=p['max_lag'], alpha=p['alpha_discovery'],
        use_xcorr=p['use_xcorr'], min_corr=p['min_corr'],
        use_te=p['use_te'], min_te=p['min_te'], te_method=p['te_method'],
        te_bins=p['te_bins'], te_k_knn=p['te_k'], te_n_perms=p['n_perms'],
        n_jobs=p['n_jobs'], block_same_property=p['block_same_property'],
        focus_entity=p['focus_entity'], include_self=p['include_self'],
        target_entity=p['target_entity'],
        use_granger_rescue=p['use_granger_rescue'], alpha_granger=p['alpha_granger'],
        use_pearson_rescue=p['use_pearson_rescue'], alpha_pearson=p['alpha_pearson'],
        min_corr_rescue=p['min_corr_rescue'], n_boot=p['n_boot'],
        boot_block_size=p['boot_block_size'],
        use_lag_penalty=p['use_lag_penalty'], lag_knee1=p['lag_knee1'],
        lag_knee2=p['lag_knee2'], lag_rate_zone1=p['lag_rate_zone1'],
        lag_rate_zone2=p['lag_rate_zone2'], lag_rate_zone3=p['lag_rate_zone3'],
        lag_penalty_fn=p['lag_penalty_fn'],
        watch_pairs=p['watch_pairs'], xcorr_te_correction=p['xcorr_te_correction'],
        correct_lag_multiplicity=p['correct_lag_multiplicity'],
        n_perms_lag_correction=p['n_perms_lag_correction'],
        correct_lag_multiplicity_rescue=p['rescue_correct_lag_multiplicity'],
    )
    run.counts['3-5_prefilter_confirmed'] = len(run.candidates)
    if not run.candidates:
        print("\n  No candidates found.")
        print("  Suggestions: increase alpha_discovery, reduce min_corr/min_te, "
              "increase max_lag, or try xcorr_te_correction='none'.")
        run.stopped = True


def _stage6_dml(run: _Run) -> None:
    """Effect estimate for every candidate, plus the two regime checks that
    are read from the estimates: near-deterministic dynamics (control set)
    and evidence coherence (applied later, by the Stage 8 guards)."""
    p = run.p
    print(f"\n[Stage 6] Estimating causal effects via DML ({len(run.candidates)} pairs)...")
    estimates, run.lean_design, run.near_deterministic = \
        run.dml_engine.estimate_effects_by_control_set(
            run.candidates, run.processed, run.pairs,
            control_set=p['dml_control_set'], source_block=p['dml_source_block'],
            auto_max_outcome_r2=p['dml_auto_max_outcome_r2'],
            max_lag=p['max_lag'], n_splits=p['dml_splits'], n_jobs=p['n_jobs'],
            dml_model=p['dml_model'], var_res_T_threshold=p['var_res_T_threshold'],
            nuisance_max_controls=p['dml_nuisance_max_controls'],
            auto_nuisance_screening=p['dml_auto_nuisance_screening'],
            estimate_selection=p['dml_estimate_selection'],
            cascade_decision=p['dml_cascade_decision'],
        )
    run.counts['6_dml_evaluated'] = len(estimates)

    run.results = results_mod.build_results(run.candidates, estimates)
    if not run.results.empty:
        run.results = results_mod.apply_statistical_checks(run.results)
    if run.has_decisions():
        run.counts['6_dml_significant'] = int(run.results['significant'].sum())

    # Do the directional (TE) and the conditional (DML) evidence agree on
    # which candidates are strong?
    if p['evidence_coherence_guard'] and not run.results.empty:
        run.coherence = coherence_mod.assess_evidence_coherence(
            run.results, min_rho=p['evidence_coherence_min_rho'],
            alpha=p['evidence_coherence_alpha'],
        )
        run.results['evidence_regime'] = run.coherence['regime']
        run.results['evidence_coherence_rho'] = run.coherence['rho']
        # The decision the Stage 8 guards fall back to if incoherent.
        run.results['significant_stage6'] = run.results['significant'].astype(bool)


def _stage7_causal_filter(run: _Run) -> None:
    """Causal penalty (Dempster-Shafer evidence fusion) over the DML output.

    Rows are never dropped here, even with causal_drop_zero=True: removing
    them before Stage 8a would erase real competitors from the topology it
    sees. The drop happens in _finalize, on the `significant` flag as
    resolved by Stage 8."""
    pen_fn = run.causal_pen_fn
    if isinstance(pen_fn, UniformCausalPenalty) or run.results.empty:
        return
    # Adaptive Support floor (Pipeline.SelectorFunc.AdaptiveCausalPenalty):
    # calibrates the floor from THIS dataset's empirical Support
    # distribution (without ground truth), instead of the fixed global
    # floor — see the class docstring. Opt-in via config.
    if isinstance(pen_fn, AdaptiveCausalPenalty):
        calibrated_floor = pen_fn.calibrate(run.results)
        print(f"  [adaptive causal_penalty] floor calibrated from "
              f"{pen_fn.n_calibration_candidates_} candidates "
              f"(percentile {pen_fn.percentile}): {calibrated_floor:.4f}")
    print(f"\n[Stage 7] Applying causal penalty ({pen_fn})...")
    run.results = run.dml_engine.apply_causal_penalty(
        run.results, causal_penalty_fn=pen_fn,
    )
    if 'significant' in run.results.columns:
        n_post_pen = int(run.results['significant'].sum())
        run.counts['7_causal_penalty_significant'] = n_post_pen
        print(f"  [Stage 7] {n_post_pen} of {len(run.results)} pairs significant "
              f"after causal penalty")


def _stage8a_structural_pruning(run: _Run) -> None:
    """Prunes redundant lagged edges over the complete, already-penalized
    set. Only marks significant=False on rows that do not survive; it never
    removes rows nor discovers new relationships."""
    p = run.p
    if p['use_structural_filter'] and not run.results.empty:
        print("\n[Stage 8a] Graph-level structural adjustment -- structural pruning...")
        prefix = 'structural_'
        kwargs = {k[len(prefix):]: v for k, v in p.items() if k.startswith(prefix)}
        if kwargs['cmi_block'] == 'auto':
            kwargs['cmi_block'] = run.lean_design
        else:
            kwargs['cmi_block'] = bool(kwargs['cmi_block'])
        run.results = apply_structural_filter(
            run.results, run.processed, n_jobs=p['n_jobs'], verbose=True, **kwargs)
        if 'significant' in run.results.columns:
            n_post_structural = int(run.results['significant'].sum())
            run.counts['8_structural_significant'] = n_post_structural
            print(f"  [Stage 8a] {n_post_structural} of {len(run.results)} pairs "
                  f"significant after structural pruning")

    # 8a-bis. Predictive parent competition (Pipeline/predictive.py): in a
    # near-deterministic system the parents are read from a multivariate
    # predictive model of each target instead of from conditional tests.
    # Replaces the Stage 6-8a decision only when its importances form two
    # populations; otherwise it abstains.
    attempt = (run.near_deterministic if p['predictive_competition'] == 'auto'
               else bool(p['predictive_competition']))
    if attempt and run.has_decisions():
        print("\n[Stage 8a-bis] Predictive parent competition...")
        run.results, pred_info = predictive_mod.predictive_competition(
            run.results, run.processed, n_splits=5,
            min_delta_bic=p['predictive_min_delta_bic'], n_jobs=p['n_jobs'],
        )
        if pred_info['applied']:
            run.counts['8_predictive_parents'] = run.n_significant()


def _stage8_regime_guards(run: _Run) -> None:
    """Two rules that overrule the lagged graph of Stages 6-8a when the data
    is outside the regime those stages were built for."""
    p = run.p
    if not run.has_decisions():
        return

    # Incoherent evidence (assessed at Stage 6): the Stage 7-8a rejections
    # are not applied; the decision falls back to the Stage 6 test, and
    # 'evidence_regime' marks the whole graph as not refined by Stages 7-8a.
    # Low-predictability exception to the incoherent fallback
    # (evidence_coherence_fallback='auto'). The Stage 6 fallback trusts the
    # lagged DML tests alone. When the past of ALL series explains little of
    # each series, the dependence between them is mostly contemporaneous --
    # what temporal aggregation does to lagged causality (Breitung & Swanson,
    # J. Time Series Analysis 23(6), 2002; Gong et al., UAI 2017; lagged vs
    # instantaneous feedback in Geweke, JASA 77, 1982) -- so a large set of
    # Stage-6-significant lagged pairs is not credible there: the fallback
    # is skipped, the Stage 7/8a decision stands, and Stage 8b adds the
    # lag-0 links. The threshold is the existing Stage 8b gate
    # (contemporaneous_max_past_r2); when the R2 cannot be computed the
    # default fallback applies.
    skip_stage6_fallback = False
    if p['evidence_coherence_fallback'] not in ('stage6', 'auto'):
        raise ValueError("evidence_coherence_fallback must be 'stage6' or 'auto'.")
    if (p['evidence_coherence_fallback'] == 'auto'
            and run.coherence is not None
            and run.coherence['regime'] == 'incoherent'
            and p['contemporaneous_max_past_r2'] is not None):
        past_r2_all = contemp_mod.past_explained_variance(
            run.processed, list(run.pairs), p['max_lag'])
        past_r2 = past_r2_all
        if p['evidence_fallback_own_past_floor'] and past_r2_all is not None:
            past_r2_own = contemp_mod.own_past_explained_variance(
                run.processed, list(run.pairs), p['max_lag'])
            if past_r2_own is not None:
                past_r2 = max(past_r2_all, past_r2_own)
            run.results['evidence_past_r2_all'] = past_r2_all
            run.results['evidence_past_r2_own'] = past_r2_own
            own_txt = 'n/a' if past_r2_own is None else f'{past_r2_own:.2f}'
            print(f"  [evidence coherence] past R2: all series "
                  f"{past_r2_all:.2f}, own past {own_txt} "
                  f"-> {past_r2:.2f} used")
        skip_stage6_fallback = (past_r2 is not None
                                and past_r2 <= p['contemporaneous_max_past_r2'])
        run.results['evidence_past_r2'] = past_r2
        run.results['evidence_fallback'] = ('stage7_8a' if skip_stage6_fallback
                                            else 'stage6')
        if skip_stage6_fallback:
            # Same audit column keep_stage6_pairs writes: what Stages 7-8a kept.
            run.results['significant_before_guard'] = run.results['significant']
            n_kept = int((run.results['significant'] == True).sum())  # noqa: E712
            run.counts['8_incoherent_low_past_kept'] = n_kept
            print(f"  [evidence coherence] INCOHERENT, but the past explains "
                  f"median R2={past_r2:.2f} <= "
                  f"{p['contemporaneous_max_past_r2']} of the series: Stage 6 "
                  f"fallback NOT applied -- the {n_kept} pairs kept by Stages "
                  f"7-8a stand")

    # Incoherent evidence and no exception: fall back to the Stage 6 test.
    if (run.coherence is not None and run.coherence['regime'] == 'incoherent'
            and not skip_stage6_fallback):
        run.results, n_before, n_kept = coherence_mod.keep_stage6_pairs(run.results)
        run.counts['8_incoherent_kept'] = n_kept
        print(f"  [evidence coherence] INCOHERENT: Stage 7-8a rejections not "
              f"applied -- {n_kept} of {len(run.results)} pairs kept on their "
              f"Stage 6 effect (Stages 7-8a had kept {n_before})")

    # No genuine time axis: no lag is real, so 8b describes the whole graph.
    if p['lagged_requires_time_axis']:
        run.results, n_drop = contemp_mod.drop_lagged_without_time_axis(
            run.results, run.processed)
        if n_drop is not None:
            run.counts['8b_no_time_axis_dropped'] = n_drop
            print(f"  [Stage 8] no time axis detected: {n_drop} lagged edges dropped")


def _stage8b_contemporaneous(run: _Run) -> None:
    """Lag-0 links -- the contemporaneous part of a structural VAR,
    identified from the innovations. Only APPENDS rows, so the calibration
    of Stages 7/8a is untouched."""
    p = run.p
    if not p['contemporaneous_links']:
        return
    print("\n[Stage 8b] Structural adjustment -- contemporaneous (lag-0) links...")
    run.results, n_added = contemp_mod.append_contemporaneous_links(
        run.results, run.processed, list(run.pairs), run.dml_engine,
        max_lag=p['max_lag'],
        alpha=p['contemporaneous_alpha'],
        max_past_r2=p['contemporaneous_max_past_r2'],
        orientation=p['contemporaneous_orientation'],
        skeleton=p['contemporaneous_skeleton'],
        pc_max_cond=p['contemporaneous_pc_max_cond'],
        require_dml=p['contemporaneous_require_dml'],
        dml_splits=p['dml_splits'], dml_model=p['dml_model'],
        var_res_T_threshold=p['var_res_T_threshold'],
        estimate_selection=p['dml_estimate_selection'],
        n_jobs=p['n_jobs'],
    )
    run.counts['8b_contemporaneous_added'] = n_added
    print(f"  [Stage 8b] {n_added} lag-0 links added")


def _stage8_output_invariant(run: _Run) -> None:
    """A reported pair always carries its effect estimate: whatever Stage 8
    accepted without one (8a-bis and 8b do not decide on the DML estimate)
    is rejected here. It also always carries its penalized effect: pairs
    Stage 8 accepted outside the Stage 7 penalty get the unpenalized
    estimate."""
    if not run.has_decisions():
        return
    run.results, n_rejected = results_mod.require_effect_estimate(run.results)
    if n_rejected:
        run.counts['8_no_estimate_rejected'] = n_rejected
        print(f"\n  [Stage 8] {n_rejected} pairs rejected: no effect estimate")
    run.results, n_filled = results_mod.ensure_penalized_effect(run.results)
    if n_filled:
        run.counts['8_unpenalized_effect'] = n_filled
        print(f"  [Stage 8] {n_filled} reported pairs outside the Stage 7 "
              f"penalty: penalized effect = unpenalized estimate")


_STAGES = (
    _stage0_structure,
    _stage1_active_properties,
    _stage2_preprocessing,
    _stage3to5_prefilter,
    _stage6_dml,
    _stage7_causal_filter,
    _stage8a_structural_pruning,
    _stage8_regime_guards,
    _stage8b_contemporaneous,
    _stage8_output_invariant,
)


def _finalize(run: _Run) -> pd.DataFrame:
    p = run.p
    results = run.results

    # Final drop of non-significant pairs (if requested), on the
    # `significant` flag as resolved by Stage 8 (which reflects both the
    # causal-penalty rejection and any structural rejection).
    if p['causal_drop_zero'] and run.has_decisions():
        before = len(results)
        results = results[results['significant'] == True].reset_index(drop=True)  # noqa: E712
        removed = before - len(results)
        if removed:
            print(f"  [causal_drop_zero] {removed} pairs removed "
                  f"(significant=False after Stage 8)")

    if p['top_n'] and not results.empty:
        results = results.head(p['top_n'])

    watch_pairs = p['watch_pairs']
    if watch_pairs is not None and len(watch_pairs) > 0:
        watch_pairs.print_report()

    _print_stage_pair_summary(run.counts)

    return results


def _print_stage_pair_summary(stage_pair_counts: dict) -> None:
    """Summary of the pair-space progression by stage — only prints the
    keys that were actually populated in this run (disabled stages simply
    do not appear)."""
    labels = {
        '0_series_universe':            'Series (entity,property)',
        '0_raw_pair_space':             'Raw directional pair space',
        '1_active_series':              'Active series (post-Stage 1)',
        '3-5_prefilter_confirmed':      'Confirmed pairs (Stages 3-5: xcorr+TE+rescue)',
        '6_dml_evaluated':              'Pairs evaluated by DML (Stage 6)',
        '6_dml_significant':            'Significant pairs after DML (pre-Stage 7)',
        '7_causal_penalty_significant': 'Significant pairs after Stage 7 (causal penalty)',
        '8_structural_significant':     'Significant pairs after Stage 8a (structural pruning)',
        '8_predictive_parents':         'Significant pairs after the predictive competition (8a-bis)',
        '8_incoherent_kept':            'Pairs kept on the Stage 6 effect (incoherent evidence)',
        '8_incoherent_low_past_kept':   'Pairs kept by Stages 7-8a (incoherent, low past R2)',
        '8b_no_time_axis_dropped':      'Lagged edges dropped (Stage 8, no time axis)',
        '8b_contemporaneous_added':     'Lag-0 links added (Stage 8b, contemporaneous)',
        '8_no_estimate_rejected':       'Pairs rejected for lack of an effect estimate (Stage 8)',
        '8_unpenalized_effect':         'Reported pairs with the unpenalized effect (Stage 8)',
    }
    print("\n" + "=" * 65)
    print("SUMMARY — pairs tested/confirmed by stage")
    print("=" * 65)
    for key, label in labels.items():
        if key in stage_pair_counts:
            print(f"  {label:<58} {stage_pair_counts[key]:>6,}")
    print("=" * 65)


def _resolve_causal_penalty(causal_penalty_fn) -> CausalPenaltyFn:
    """Resolves which causal penalty function to use."""
    if causal_penalty_fn is not None:
        print(f"\n  [causal_penalty] {causal_penalty_fn}")
        return causal_penalty_fn

    return DEFAULT_CAUSAL_PENALTY
