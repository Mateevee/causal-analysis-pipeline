"""
SelectorFunc.py
=======================
Modular 7-stage architecture for the causal_pen fusion decision:

    1. Structural Gates       — deterministic, not evidence
    2. Feature Engineering    — only computes, decides nothing
    3. Primary Evidence       — 9 experts, 1 family each
    4. Meta Evidence          — experts that summarize the primary ones
    5. Interaction Specialists— experts that cross distinct families
    6. Evidence Fusion        — Dempster-Shafer, Yager's rule (gated hybrid)
    7. Final Decision         — Support/Opposition/Uncertainty/Conflict/
                                Consensus/Evidence Quality/Final Confidence

PRIMARY EVIDENCE: raw pvalue, pvalue_bh, direct BH significance,
CI-crosses-zero and cv_ci are not standalone experts, since they are all
deterministic transformations of the same statistical information (theta,
se, under the normal approximation used in _compute_theta_and_se). cv_ci
and CI width remain available as INPUT to Feature Engineering (stage 2) and
feed a COHERENCE expert in Meta Evidence (meta_consistency_statistical)
instead of voting independently. Batch-level FDR control (pvalue_bh) is
conceptually distinct from a single pair's theta/se and, if needed, belongs
in a dedicated Meta Evidence expert (e.g. 'batch_fdr_control') rather than
the pointwise statistical expert.

RESIDUAL QUALITY: RESET, dCor, HSIC and Spearman are combined into ONE
diagnostic of "how much the residuals appear misspecified by the linear
form", treated mostly as UNCERTAINTY (Θ) rather than direct evidence
against H, since uncaptured curvature does not imply the absence of
causality, only that linear theta may be poorly calibrated.

META-EVIDENCE LAYER: consistency among primary experts, bootstrap/
permutation/theta consistency, internal statistical coherence (observed CI
vs. CI implied by se) and causal consistency (TE/forward/reverse/asymmetry).

INTERACTION SPECIALISTS: capture patterns that independent fusion of
primary experts does NOT capture well — especially CONJUNCTION cases where
one axis looks good but another contradicts it (e.g. stable bootstrap +
misspecified residuals -> reduce confidence), which independent-evidence
fusion tends to dilute rather than flag.

EXPERT-COUNT CONTROL: the full architecture enumerates ~19 possible experts
(9 primary + 4 meta + 6 interaction), more than the recommended "10-12"
range for a Dempster-Shafer fusion. All ~19 are implemented as independent,
testable, individually documented functions (for modularity and
auditability), but only a CURATED SUBSET enters the fusion by default — the
9 primary (mandatory) ones plus 2 additional ones chosen for being the
least redundant with what the independent-evidence fusion already captures
on its own (see EXPERTS_CFG). The rest are available via a flag, for anyone
who wants to validate them on a benchmark before enabling them in
production.

COLUMN-NAME COMPATIBILITY: stage 2 (Feature Engineering) uses a helper
(_get) that accepts multiple possible names for the same field (e.g.
'causal_effect' OR 'effect'), accepting both the causal_pen output
format and the DML output format (dml_engine.py) — this is schema
tolerance only.

DOMAIN-AGNOSTIC DESIGN: the pipeline operates over ENTITIES x PROPERTIES
generically, with no assumption about what they represent.
_stage1_structural_gates() contains only the truly universal
gates (effect ~= 0, minimum statistical data missing, marginal variance for
T or O ~= 0 — a practically constant entity/property does not support a
causal claim in any domain). A domain-specific admissibility rule (e.g.
"both properties share a blocked prefix") is available, opt-in, as a plugin
via create_blocked_property_gate(), injected through the optional
domain_gate_fn parameter of causal_pen()/causal_pen_diagnostics() — never
applied unless explicitly passed.

Column names follow a single unified contract across the whole Pipeline
package: 'source_entity', 'target_entity', 'source_property',
'target_property'.

DECISION CRITERION — RATIO, NOT RAW DIFFERENCE
------------------------------------------------------------------
The final decision compares the RATIO Support/(Support+Opposition) —
"among what was actually decided by the experts, what fraction favors H" —
instead of the raw difference Support - Opposition. When combining many
experts via Dempster-Shafer, accumulated conflict (K) grows with the number
of independently opinionated sources, and Yager's rule redirects that
conflict to Θ (uncertainty), which crushes the raw difference even when
Opposition is negligible relative to Support. The ratio is invariant to how
much conflict/uncertainty has accumulated. An absolute floor
(MIN_ABSOLUTE_SUPPORT_THRESHOLD) prevents accepting on a high ratio with
negligible decided mass, and the Opposition ceiling
(MAX_OPPOSITION_THRESHOLD) is preserved as a safeguard. See comments
next to NET_RATIO_MIN_THRESHOLD and in _stage7_final_decision.

NOTE — expert_statistical uses abs(feat.t_stat)
------------------------------------------------------------------
expert_statistical() measures the STRENGTH/significance of the estimate, not
the DIRECTION of the effect (direction is already assessed by other experts:
sign agreement with bootstrap, causal asymmetry, etc.), so it evaluates
abs(feat.t_stat) against the threshold rather than the signed value. Every
other call to _mass_by_threshold() in the module uses values that are
non-negative by construction or already enter as correctly formed
differences/residuals, so this is the only expert where the sign needs to be
stripped explicitly.

CAVEAT: the `temporal_structure` expert uses `lag - median_lag` WITH its
sign against a threshold of 0.0 with favors_larger=False — this treats a lag
much SMALLER than the run's typical lag as just as favorable as a lag equal
to the typical one (only a lag LARGER than typical is penalized), when the
more defensible approach would be to penalize the deviation in either
direction (abs(lag - median_lag)). Recorded here as a candidate for future
review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import numpy as np


# =============================================================================
# Calibrated thresholds shared across experts.
# =============================================================================

TE_VETO_MIN = 0.24

LAG_VETO_MIN = 4

RESID_PROTECTION_MIN = 0.010
ROB_CVCI_INTERCEPT = 0.3853
ROB_CVCI_SLOPE = -0.1639

# Dempster-Shafer combination rule. 'yager' preserves the conflict by
# redirecting it to uncertainty (Θ), instead of discarding it via
# normalization (classic Dempster rule). 'hybrid_gated' (default) is the
# Yager chain with the causal_soft cluster merged into one Murphy-average
# pseudo-expert when the DML cascade is reliable (see
# _combine_gated_hybrid); 'hybrid_light' and 'hierarchical' are the other
# options.
COMBINATION_RULE = "hybrid_gated"

# Final decision: TWO criteria together.
#
# MAX_OPPOSITION_THRESHOLD is the ABSOLUTE safety ceiling on Belief(notH).
MAX_OPPOSITION_THRESHOLD = 0.45

# Acceptance is decided by a RATIO criterion: among the mass the experts
# actually committed to (Support + Opposition, ignoring Θ), what fraction
# favors H — rather than the raw difference Support - Opposition. When
# combining many experts via Dempster-Shafer, accumulated conflict (K) grows
# with the number of independently opinionated sources, and Yager's rule
# redirects that conflict to Θ, which would crush a raw difference even when
# Opposition is negligible. The ratio is invariant to the absolute amount of
# accumulated conflict/uncertainty.
NET_RATIO_MIN_THRESHOLD = 0.55

# ABSOLUTE floor on Support — needed so that the ratio above does not
# accept pairs whose committed mass is negligible (e.g. Support=0.001,
# Opposition=0.0, ratio=1.0 but with no real information behind it).
MIN_ABSOLUTE_SUPPORT_THRESHOLD = 0.10

# If True, causal_pen returns the continuous final_confidence in [0,1]
# instead of the binary 0/1. Default False to preserve the interface
# contract (causal_pen(row, target_fn=None) -> float).
CONTINUOUS_RETURN = False


# =============================================================================
# Configuration of which Meta Evidence and Interaction experts actually
# enter the fusion (stage 6). See section 5 of the module docstring for
# the justification of the default curation.
# =============================================================================

@dataclass
class ExpertsConfig:
    # -- Meta Evidence --
    enable_evidence_consistency:  bool = False  # see note (a) below
    enable_bootstrap_consistency:   bool = False  # see note (a) below
    enable_statistical_consistency: bool = False  # see note (a) below
    enable_causal_consistency:      bool = True   # active by default — see note (b)
    enable_minimum_causal_confirmation: bool = True  # see note (e)

    # -- Interaction Specialists --
    enable_te_x_robustness:      bool = False  # see note (c) below
    enable_te_x_bootstrap:     bool = False  # see note (c) below
    enable_bootstrap_x_residuals: bool = True   # see note (f)
    enable_robustness_x_reset:   bool = True     # see note (f)
    enable_asymmetry_x_te:    bool = False    # see note (c) below
    enable_ess_x_bootstrap:    bool = True     # active by default — see note (d)
    enable_weak_consensus_x_missing_te: bool = True  # see note (g)

    # NOTE (a): evidence_consistency/bootstrap_consistency/
    # statistical_consistency are OFF by default to keep the total
    # number of experts in the fusion within ~10-12. They continue to
    # be computed and available in causal_pen_diagnostics() for
    # audit/benchmark purposes even when off in the fusion — see
    # _evaluate_pair_v2.
    #
    # NOTE (b): causal_consistency is ON by default because it is the
    # meta-evidence with the least overlap with the primary experts —
    # no single primary expert checks DIRECTION coherence between TE
    # and causal asymmetry simultaneously.
    #
    # NOTE (c): TE x Robustness, TE x Bootstrap and Asymmetry x TE are
    # off by default because, when both sides of an interaction
    # SUPPORT H in the same direction, the Dempster-Shafer fusion of
    # the primary experts already amplifies that automatically (this is
    # the nature of combining independent evidence) — an additional
    # interaction expert for the same case ("both agree in favor")
    # risks double-counting the same thing. They remain implemented and
    # available for anyone who wants to validate whether, on their own
    # benchmark, they actually add incremental signal.
    #
    # NOTE (d): ESS x Bootstrap is ON by default because it is the
    # clearest NON-redundant case: low ESS alone and unstable bootstrap
    # alone are each, on their own, weak evidence (high uncertainty of
    # their own); together, they indicate that the estimate is
    # fundamentally unreliable in a way that neither primary expert
    # communicates with sufficient force in isolation.
    #
    # NOTE (e): minimum_causal_confirmation is ON by default. It closes
    # the gap where the statistical experts (t_stat/bootstrap/permutation
    # — which fundamentally test the same thing, "is this coefficient
    # different from zero?") can be strong WITHOUT any causal-specific
    # confirmation (TE/robustness/cascade consensus/asymmetry), which does
    # not distinguish causality from spurious correlation/confounding.
    #
    # NOTE (f): bootstrap_x_residuals and robustness_x_reset are on by
    # default, same motivation as (e). They capture the "good on one axis,
    # bad on another" pattern — apparently stable bootstrap or apparently
    # high robustness, but misspecified residuals (RESET/dCor)
    # contradicting it.
    #
    # NOTE (g): weak_consensus_x_missing_te is on by default. See the
    # docstring of interaction_weak_consensus_x_missing_te() for the full
    # analysis of the pattern this expert captures.


DEFAULT_EXPERTS_CFG = ExpertsConfig()


# =============================================================================
# Belief-mass (BBA) helpers.
# =============================================================================

def _sigmoid(z: float) -> float:
    if z > 40:
        return 1.0
    if z < -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _mass_by_threshold(
    value: Optional[float],
    threshold: float,
    scale: float,
    base_uncertainty: float,
    favors_larger: bool = True,
) -> Dict[str, float]:
    """
    Converts a continuous value, relative to a threshold, into a belief
    mass over {H, notH, Theta}.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {"H": 0.0, "notH": 0.0, "Theta": 1.0}

    z = (value - threshold) / max(scale, 1e-9)
    if not favors_larger:
        z = -z

    strength = _sigmoid(z)
    m_theta = base_uncertainty
    m_h = strength * (1.0 - m_theta)
    m_nh = (1.0 - strength) * (1.0 - m_theta)
    return {"H": m_h, "notH": m_nh, "Theta": m_theta}


# =============================================================================
# Dempster-Shafer / Yager's rule combination.
# =============================================================================

def _combine_raw_pair(
    m1: Dict[str, float], m2: Dict[str, float]
) -> Tuple[Dict[str, float], float]:
    h1, nh1, th1, k1 = m1["H"], m1["notH"], m1["Theta"], m1.get("K", 0.0)
    h2, nh2, th2, k2 = m2["H"], m2["notH"], m2["Theta"], m2.get("K", 0.0)

    new_h = h1 * h2 + h1 * th2 + th1 * h2
    new_nh = nh1 * nh2 + nh1 * th2 + th1 * nh2
    new_th = th1 * th2
    direct_k = h1 * nh2 + nh1 * h2
    new_k = direct_k + k1 + k2 * (1.0 - k1)

    return {"H": new_h, "notH": new_nh, "Theta": new_th, "K": new_k}, direct_k


def _combine_all(
    masses: List[Tuple[str, Dict[str, float]]], rule: str = COMBINATION_RULE
) -> Tuple[Dict[str, float], float, List[Tuple[str, float]]]:
    """
    Combines N experts keeping the accumulated conflict in its own
    bucket (K) throughout the whole sequence — see the docstring of
    _combine_raw_pair for the combination math.
    """
    current = {"H": 0.0, "notH": 0.0, "Theta": 1.0, "K": 0.0}
    conflicts: List[Tuple[str, float]] = []

    for name, mass in masses:
        mass_with_k = dict(mass)
        mass_with_k.setdefault("K", 0.0)
        current, direct_k = _combine_raw_pair(current, mass_with_k)
        conflicts.append((name, direct_k))

    total_k = current["K"]

    if rule == "dempster_classic":
        denom = max(1.0 - total_k, 1e-9)
        final_mass = {
            "H": current["H"] / denom,
            "notH": current["notH"] / denom,
            "Theta": current["Theta"] / denom,
        }
    else:  # yager
        final_mass = {
            "H": current["H"],
            "notH": current["notH"],
            "Theta": current["Theta"] + total_k,
        }

    return final_mass, total_k, conflicts


# =============================================================================
# HIERARCHICAL FUSION BY FAMILIES (opt-in)
# =============================================================================
# Motivation: with plain Yager combination (rule='yager') and ~15-17 partially
# correlated experts combined sequentially, accumulated conflict (K)
# grows with cardinality even when the disagreement is noise between
# correlated sources, not genuine signal — Yager itself then throws all
# that K to Θ, crushing real net support in pairs where the
# causal-specific family is weak-but-agreeing (not weak-and-contrary).
#
# Strategy (does not change any individual expert):
#   1. Groups the active experts into 3 correlated families:
#      'statistical' (t_stat/bootstrap/permutation — fundamentally test
#      "effect != 0"), 'diagnostic' (residual quality/ESS and their
#      interactions — test "is the estimate reliable?") and 'causal'
#      (everything else: TE, robustness, temporal structure, asymmetry,
#      cascade consensus, causal consistency, minimum causal
#      confirmation, context/mode — test "is this causal, not just
#      correlation?").
#   2. WITHIN each family: Murphy Average Rule (average of the BBAs,
#      then self-combined via Dempster N-1 times) — tolerates internal
#      correlation without generating artificial conflict, but still
#      lets genuine consensus WITHIN the family assert itself.
#   3. BETWEEN the (up to 3) resulting families: classic Dempster if the
#      inter-family conflict is low (most cases — cardinality 3, not
#      15), with a fallback to Yager only if that conflict is high
#      enough to be genuine disagreement between "there is a
#      statistical effect" and "this is causal" (the scenario Yager was
#      designed for).
#
# Only enabled when `fusion_rule='hierarchical'` is explicitly passed to
# causal_pen()/causal_pen_diagnostics()/_evaluate_pair_v2(); the
# default is COMBINATION_RULE ('hybrid_gated').

STATISTICAL_FAMILY = {"statistical", "bootstrap", "permutation"}

DIAGNOSTIC_FAMILY = {
    "residual_quality", "ess", "bootstrap_x_residuals",
    "robustness_x_reset", "ess_x_bootstrap", "weak_consensus_x_missing_te",
}

SOFT_CAUSAL_FAMILY = {
    "transfer_entropy", "robustness", "temporal_structure", "asymmetry",
    "te_x_robustness", "te_x_bootstrap", "asymmetry_x_te",
}

# GATE experts: already operate as near-vetoes (strongly asymmetric,
# near-binary mass by design — see docstrings of
# meta_minimum_causal_confirmation and expert_cascade_consensus).
# Averaging this inside a family dilutes exactly the mechanism that
# makes them effective (e.g. minimum_causal_confirmation is designed
# to PUSH Support below the minimum floor when no causal source
# confirms — Murphy-averaging with ~5 other members of the causal
# family would water that notH=0.75 down to a small fraction, undoing
# the precision protection it provides).
# That is why they stay OUT of Murphy: they enter via direct conjunction
# against the families' aggregated result.
GATE_EXPERTS = {"minimum_causal_confirmation", "cascade_consensus", "causal_consistency"}

INTERFAMILY_YAGER_CONFLICT_THRESHOLD = 0.30


def _family_name(expert_name: str) -> str:
    base = expert_name.split(" ")[0]  # 'context_mode [rescue]' -> 'context_mode'
    if base in GATE_EXPERTS:
        return "gate"
    if base in STATISTICAL_FAMILY:
        return "statistical"
    if base in DIAGNOSTIC_FAMILY:
        return "diagnostic"
    if base in SOFT_CAUSAL_FAMILY:
        return "causal_soft"
    return "causal_soft"  # context_mode[...] and any new unmapped expert


def _combine_murphy_family(
    masses: List[Tuple[str, Dict[str, float]]]
) -> Tuple[Dict[str, float], float]:
    """
    Murphy Average Rule: arithmetic average of the family's BBAs,
    followed by self-combination via classic Dempster (N-1 times) —
    instead of combining the N correlated sources directly against each
    other (which artificially inflates conflict when they are
    redundant).
    """
    if not masses:
        return {"H": 0.0, "notH": 0.0, "Theta": 1.0}, 0.0

    n = len(masses)
    h_avg = float(np.mean([m["H"] for _, m in masses]))
    nh_avg = float(np.mean([m["notH"] for _, m in masses]))
    th_avg = max(0.0, 1.0 - h_avg - nh_avg)
    average = {"H": h_avg, "notH": nh_avg, "Theta": th_avg, "K": 0.0}

    if n == 1:
        return {"H": h_avg, "notH": nh_avg, "Theta": th_avg}, 0.0

    current = dict(average)
    for _ in range(n - 1):
        current, _ = _combine_raw_pair(current, average)
    total_k = current["K"]
    denom = max(1.0 - total_k, 1e-9)
    final = {
        "H": current["H"] / denom,
        "notH": current["notH"] / denom,
        "Theta": current["Theta"] / denom,
    }
    return final, total_k


def _combine_light_hybrid(
    masses: List[Tuple[str, Dict[str, float]]],
) -> Tuple[Dict[str, float], float, List[Tuple[str, float]]]:
    """
    Light hybrid: the sequential Yager chain of _combine_all (statistical,
    diagnostic and gate experts enter one at a time), with the members of
    the 'causal_soft' cluster (TE, robustness, temporal_structure,
    asymmetry) replaced by ONE Murphy-average pseudo-expert of those four.
    Reduces the chain's cardinality by ~3 (from ~15-17 to ~12-14)
    specifically where the spurious conflict between
    TE-below-but-near-the-veto vs. other weak-but-agreeing causal experts
    originates, without touching any other expert or its behavior in the
    chain.
    """
    others: List[Tuple[str, Dict[str, float]]] = []
    soft_members: List[Tuple[str, Dict[str, float]]] = []
    for name, mass in masses:
        if _family_name(name) == "causal_soft":
            soft_members.append((name, mass))
        else:
            others.append((name, mass))

    if soft_members:
        m_soft, _ = _combine_murphy_family(soft_members)
        others.append((f"causal_soft_combined({len(soft_members)})", m_soft))

    return _combine_all(others, rule="yager")


def _combine_gated_hybrid(
    masses: List[Tuple[str, Dict[str, float]]],
    feat: "FeatureSet",
) -> Tuple[Dict[str, float], float, List[Tuple[str, float]]]:
    """
    Same grouping as _combine_light_hybrid (only groups causal_soft via
    Murphy), but GATED by the reliability of the underlying DML estimate
    — discounting, not a blind rule swap: only relaxes the internal
    conjunction of the causal_soft cluster when the nuisance-model
    cascade has already strongly corroborated the estimate
    (cascade_confidence='high' AND n_models_agree>=3 — a minimum of 3
    independent models agreeing on sign).
    """
    reliable = (
        feat.cascade_confidence == "high"
        and feat.n_models_agree is not None
        and feat.n_models_agree >= 3
    )
    if not reliable:
        return _combine_all(masses, rule="yager")
    return _combine_light_hybrid(masses)


def _combine_hierarchical(
    masses: List[Tuple[str, Dict[str, float]]],
) -> Tuple[Dict[str, float], float, List[Tuple[str, float]]]:
    """
    Two-level fusion:
      - the 'statistical'/'diagnostic'/'causal_soft' families (genuinely
        correlated evidence) combined via Murphy Average within each
        family;
      - 'gate' experts (minimum_causal_confirmation, cascade_consensus,
        causal_consistency — near-vetoes by design) do NOT enter the
        Murphy average, they enter via direct conjunction;
      - all resulting sources (up to 3 families + up to 3 gates) are
        combined at the top always via YAGER (not classic Dempster) —
        cardinality drops from ~15-17 to ~4-6, which already resolves
        most of the correlation-driven conflict explosion, without
        giving up Yager's safety margin against genuine conflict (this
        is what keeps the 'gate' experts' veto intact).
    """
    families: Dict[str, List[Tuple[str, Dict[str, float]]]] = {
        "statistical": [], "diagnostic": [], "causal_soft": [],
    }
    gates: List[Tuple[str, Dict[str, float]]] = []
    for name, mass in masses:
        fam = _family_name(name)
        if fam == "gate":
            gates.append((name, mass))
        else:
            families[fam].append((name, mass))

    top_sources: List[Tuple[str, Dict[str, float]]] = []
    for fam, members in families.items():
        if not members:
            continue
        m_fam, _ = _combine_murphy_family(members)
        top_sources.append((f"family::{fam}({len(members)})", m_fam))
    top_sources.extend(gates)

    if not top_sources:
        return {"H": 0.0, "notH": 0.0, "Theta": 1.0}, 0.0, []

    current = {"H": 0.0, "notH": 0.0, "Theta": 1.0, "K": 0.0}
    conflicts: List[Tuple[str, float]] = []
    for name, mass in top_sources:
        mass_with_k = dict(mass)
        mass_with_k.setdefault("K", 0.0)
        current, direct_k = _combine_raw_pair(current, mass_with_k)
        conflicts.append((name, direct_k))
    total_k = current["K"]

    final_mass = {
        "H": current["H"], "notH": current["notH"],
        "Theta": current["Theta"] + total_k,
    }
    return final_mass, total_k, conflicts


# =============================================================================
# Column-name-tolerant reading helper
# =============================================================================

def _get(row: Any, *keys: str, default: Any = None) -> Any:
    """
    Reads the first existing, non-null key from `row` (dict or
    pandas.Series) among `keys`, in the order given. Exists to accept
    both the causal_pen output naming scheme ('causal_effect',
    'standard_error', 'ci_95_lower'/'ci_95_upper') and the native scheme
    of the DML output ('effect', 'se', 'ci_lower'/'ci_upper').
    """
    for key in keys:
        try:
            value = row[key]
        except (KeyError, TypeError, IndexError):
            continue
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return default


# =============================================================================
# STAGE 1 — STRUCTURAL GATES (deterministic, NOT evidence)
# =============================================================================
# NOTE: the pipeline operates over ENTITIES x PROPERTIES, with no
# assumption about what they represent. This function contains ONLY the
# universal gates (apply to any domain, no dataset-specific assumption);
# see create_blocked_property_gate() further below for the opt-in plugin
# that adds a domain-specific admissibility rule.

def _stage1_structural_gates(row: Any) -> Optional[str]:
    """
    Conditions where the question "is H valid?" does not even make
    sense for this row — these are not evaluated by evidence fusion,
    they are deterministic admissibility gates. Returns None if the pair
    passes (should proceed to fusion), or a string with the reason for
    the cut.

    Contains only the universal gates (effect ~= 0, minimum statistical
    data missing, marginal variance ~= 0) — no domain-specific rule.
    """
    # -- UNIVERSAL (apply to any domain) --
    effect = _get(row, "causal_effect", "effect")
    if effect is None:
        return "effect_missing"
    if abs(float(effect)) < 1e-4:
        return "effect_numerically_zero"

    se = _get(row, "standard_error", "se")
    if se is None or float(se) <= 0:
        return "standard_error_invalid_or_missing"

    ci_lo = _get(row, "ci_95_lower", "ci_lower")
    ci_hi = _get(row, "ci_95_upper", "ci_upper")
    if ci_lo is None or ci_hi is None:
        return "confidence_interval_missing"

    # Marginal variance of T or O ~= 0 — universal, does not
    # depend on domain. A practically constant entity/property does not
    # support any causal claim, regardless of what it represents. Only
    # evaluated if the information is available on the row (expected
    # column names: 'treatment_variance' / 'outcome_variance', computed
    # upstream if the pipeline exposes this information; if missing,
    # this gate is skipped at no cost).
    var_t = _get(row, "treatment_variance", "var_treatment")
    var_o = _get(row, "outcome_variance", "var_outcome")
    if var_t is not None and float(var_t) < 1e-8:
        return "treatment_marginal_variance_near_zero"
    if var_o is not None and float(var_o) < 1e-8:
        return "outcome_marginal_variance_near_zero"

    return None


def create_blocked_property_gate(
    blocked_prefix: str,
    source_property_key: str = "source_property",
    target_property_key: str = "target_property",
):
    """
    Factory for an OPTIONAL `domain_gate_fn`: blocks any entity x property
    pair where BOTH properties (source and target) start with the same
    blocked prefix (e.g. two exogenous inputs that cannot cause each
    other). NEVER applied by default by the module.

    Usage:
        gate_entities = create_blocked_property_gate(blocked_prefix="U_")
        causal_pen(row, domain_gate_fn=gate_entities)
    """
    def _gate(row: Any) -> Optional[str]:
        prop_source = _get(row, source_property_key, default="")
        prop_target = _get(row, target_property_key, default="")
        if str(prop_source).startswith(blocked_prefix) and str(prop_target).startswith(blocked_prefix):
            return f"both_properties_with_prefix_{blocked_prefix}"
        return None
    return _gate


# =============================================================================
# STAGE 2 — FEATURE ENGINEERING (only computes, never decides)
# =============================================================================

@dataclass
class FeatureSet:
    # core statistics
    theta: float
    se: float
    t_stat: float
    ci_lo: float
    ci_hi: float
    ci_width: float
    cv_ci: float
    ci_asymmetry: Optional[float]

    # bootstrap
    boot_mean: Optional[float] = None
    boot_std: Optional[float] = None
    boot_width: Optional[float] = None
    boot_stability: Optional[float] = None
    boot_overlap: Optional[float] = None

    # permutation
    perm_pvalue: Optional[float] = None
    perm_theta_std: Optional[float] = None
    perm_distance: Optional[float] = None

    # transfer entropy
    te: Optional[float] = None
    te_normalized: Optional[float] = None

    # robustness
    robustness: Optional[float] = None
    robustness_qa: Optional[float] = None
    robustness_residual: Optional[float] = None

    # residual quality
    reset_pvalue: Optional[float] = None
    reset_fstat: Optional[float] = None
    dcor: Optional[float] = None
    hsic: Optional[float] = None
    spearman: Optional[float] = None
    corr_linear: Optional[float] = None

    # temporal structure
    lag: Optional[float] = None
    lag_median: Optional[float] = None
    lag_penalty: Optional[float] = None

    # ESS
    ess_treatment: Optional[float] = None
    ess_outcome: Optional[float] = None
    ess_min: Optional[float] = None

    # causal asymmetry
    theta_forward: Optional[float] = None
    theta_reverse: Optional[float] = None
    causal_asymmetry: Optional[float] = None

    # context (mode prior — not one of the 9 primary experts; see
    # expert_context_mode)
    mode_txt: str = ""
    prior_strength_nh: float = 0.0

    # Nuisance-model cascade consensus: whether the result fell to a
    # fallback model (no main model in the cascade passed the criteria),
    # the cascade_confidence and n_models_agree.
    cascade_confidence: Optional[str] = None
    n_models_agree: Optional[int] = None
    fallback_model: bool = False


def _stage2_feature_engineering(row: Any) -> FeatureSet:
    """
    Builds ALL derived metrics used by the experts. No decision is made
    here — only transformation/derivation. Assumes
    _stage1_structural_gates already returned None (pair passed).
    """
    theta = float(_get(row, "causal_effect", "effect"))
    se = float(_get(row, "standard_error", "se"))
    ci_lo = float(_get(row, "ci_95_lower", "ci_lower"))
    ci_hi = float(_get(row, "ci_95_upper", "ci_upper"))

    raw_t_stat = _get(row, "t_stat")
    t_stat = float(raw_t_stat) if raw_t_stat is not None else abs(theta) / max(se, 1e-9)

    ci_width = ci_hi - ci_lo
    cv_ci = ci_width / max(abs(theta), 1e-9)

    ci_asymmetry = None
    if ci_width > 1e-9:
        dist_hi = ci_hi - theta
        dist_lo = theta - ci_lo
        ci_asymmetry = (dist_hi - dist_lo) / ci_width

    # -- bootstrap --
    boot_mean = _get(row, "theta_boot_mean")
    boot_lo = _get(row, "ci_lower_boot")
    boot_hi = _get(row, "ci_upper_boot")
    boot_width = (float(boot_hi) - float(boot_lo)) if (boot_lo is not None and boot_hi is not None) else None
    # approximate standard deviation from the 2.5-97.5 percentile CI
    # width (~3.92 standard deviations under normality) — avoids
    # requiring the raw bootstrap draws here, since only the summarized
    # CI is available.
    boot_std = (boot_width / 3.92) if boot_width is not None else None
    boot_stability = None
    if boot_std is not None and boot_mean is not None:
        cv_boot = boot_std / max(abs(float(boot_mean)), 1e-9)
        boot_stability = max(0.0, 1.0 - min(cv_boot, 3.0) / 3.0)
    boot_overlap = None
    if boot_lo is not None and boot_hi is not None:
        boot_lo_f, boot_hi_f = float(boot_lo), float(boot_hi)
        inter = max(0.0, min(ci_hi, boot_hi_f) - max(ci_lo, boot_lo_f))
        union = max(max(ci_hi, boot_hi_f) - min(ci_lo, boot_lo_f), 1e-9)
        boot_overlap = inter / union

    # -- permutation --
    perm_pvalue = _get(row, "permutation_pvalue")
    perm_theta_std = _get(row, "theta_permutation_std")
    perm_distance = None
    if perm_theta_std is not None:
        perm_distance = abs(theta) / max(float(perm_theta_std), 1e-9)

    # -- transfer entropy --
    te = _get(row, "transfer_entropy")
    te_normalized = None
    if te is not None:
        te_normalized = _sigmoid((float(te) - TE_VETO_MIN) / 0.08)

    # -- robustness --
    robustness = _get(row, "robustness_value")
    robustness_qa = _get(row, "robustness_value_qa")
    robustness_residual = None
    if robustness is not None and cv_ci:
        expected_robustness = ROB_CVCI_INTERCEPT + ROB_CVCI_SLOPE * cv_ci
        robustness_residual = float(robustness) - expected_robustness

    # -- residual quality --
    reset_pvalue = _get(row, "reset_pvalue")
    reset_fstat = _get(row, "reset_fstat")
    dcor = _get(row, "dcor_residuals")
    hsic = _get(row, "hsic_residuals")
    spearman = _get(row, "spearman_residuals")
    corr_linear = _get(row, "corr_linear_residuals")

    # -- temporal structure --
    lag = _get(row, "lag")
    lag_median = _get(row, "lag_median")
    lag_penalty = _get(row, "lag_penalty_score")

    # -- ESS --
    ess_treatment = _get(row, "ess_treatment")
    ess_outcome = _get(row, "ess_outcome")
    ess_min = None
    if ess_treatment is not None and ess_outcome is not None:
        ess_min = min(float(ess_treatment), float(ess_outcome))

    # -- causal asymmetry --
    theta_forward = _get(row, "theta_forward_lite")
    theta_reverse = _get(row, "theta_reverse_lite")
    causal_asymmetry = _get(row, "causal_asymmetry")
    if causal_asymmetry is None and theta_forward is not None and theta_reverse is not None:
        causal_asymmetry = abs(float(theta_forward)) - abs(float(theta_reverse))

    # -- context/mode prior --
    # is_rescue covers both Granger_rescue and Pearson_rescue (the level-2
    # rescue, more permissive than Granger, only triggered when Granger
    # also fails) — both get the same skepticism
    # discount via prior_strength_nh (there is no distinct empirical
    # calibration by rescue path yet; if one is introduced, differentiate
    # here).
    filter_method = _get(row, "filter_method", default="")
    is_rescue = filter_method in ("Granger_rescue", "Pearson_rescue")
    old_sig = bool(_get(row, "old_sig", default=False))
    if is_rescue and old_sig:
        prior_strength_nh, mode_txt = 0.25, f"[rescue:{filter_method}+old_sig]"
    elif is_rescue:
        prior_strength_nh, mode_txt = 0.40, f"[rescue:{filter_method}]"
    else:
        prior_strength_nh, mode_txt = 0.0, ""

    # -- nuisance-model cascade consensus --
    cascade_confidence_val = _get(row, "cascade_confidence")
    n_models_agree_val = _get(row, "n_models_agree")
    dml_model_used_val = _get(row, "dml_model_used", default="")
    fallback_model_val = "[fallback]" in str(dml_model_used_val)

    return FeatureSet(
        theta=theta, se=se, t_stat=t_stat, ci_lo=ci_lo, ci_hi=ci_hi,
        ci_width=ci_width, cv_ci=cv_ci, ci_asymmetry=ci_asymmetry,
        boot_mean=float(boot_mean) if boot_mean is not None else None,
        boot_std=boot_std, boot_width=boot_width,
        boot_stability=boot_stability, boot_overlap=boot_overlap,
        perm_pvalue=float(perm_pvalue) if perm_pvalue is not None else None,
        perm_theta_std=float(perm_theta_std) if perm_theta_std is not None else None,
        perm_distance=perm_distance,
        te=float(te) if te is not None else None, te_normalized=te_normalized,
        robustness=float(robustness) if robustness is not None else None,
        robustness_qa=float(robustness_qa) if robustness_qa is not None else None,
        robustness_residual=robustness_residual,
        reset_pvalue=float(reset_pvalue) if reset_pvalue is not None else None,
        reset_fstat=float(reset_fstat) if reset_fstat is not None else None,
        dcor=float(dcor) if dcor is not None else None,
        hsic=float(hsic) if hsic is not None else None,
        spearman=float(spearman) if spearman is not None else None,
        corr_linear=float(corr_linear) if corr_linear is not None else None,
        lag=float(lag) if lag is not None else None,
        lag_median=float(lag_median) if lag_median is not None else None,
        lag_penalty=float(lag_penalty) if lag_penalty is not None else None,
        ess_treatment=float(ess_treatment) if ess_treatment is not None else None,
        ess_outcome=float(ess_outcome) if ess_outcome is not None else None,
        ess_min=ess_min,
        theta_forward=float(theta_forward) if theta_forward is not None else None,
        theta_reverse=float(theta_reverse) if theta_reverse is not None else None,
        causal_asymmetry=float(causal_asymmetry) if causal_asymmetry is not None else None,
        mode_txt=mode_txt, prior_strength_nh=prior_strength_nh,
        cascade_confidence=str(cascade_confidence_val) if cascade_confidence_val is not None else None,
        n_models_agree=int(n_models_agree_val) if n_models_agree_val is not None else None,
        fallback_model=fallback_model_val,
    )


# =============================================================================
# STAGE 3 — PRIMARY EVIDENCE (9 experts, 1 family each)
# =============================================================================

_NEUTRAL_MASS = {"H": 0.0, "notH": 0.0, "Theta": 1.0}


def expert_transfer_entropy(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """Exclusive input: transfer entropy."""
    if feat.te is None:
        return ("transfer_entropy", dict(_NEUTRAL_MASS))
    if feat.te == 0:
        # TE==0 EXACTLY is treated as "not measured" (common in
        # rescue/Granger pairs, where TE is typically not computed), not
        # as an actual measurement of the absence of information
        # transfer.
        return ("transfer_entropy", {"H": 0.05, "notH": 0.05, "Theta": 0.90})
    m = _mass_by_threshold(feat.te, TE_VETO_MIN, scale=0.08,
                            base_uncertainty=0.20, favors_larger=True)
    return ("transfer_entropy", m)


def expert_statistical(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """
    Exclusive input: t_stat (which already incorporates theta and
    standard error). Does NOT use pvalue, pvalue_bh, significant,
    significant_bh nor cv_ci — all are deterministic transformations
    of the same pair (theta, se) under the normal approximation, and
    counting them separately would be multiplying the same evidence. CI
    width/coherence enters a distinct META-evidence expert
    (meta_consistency_statistical), not here, because there the question
    is different: "is the observed CI coherent with what the normal
    approximation would imply?", not "is the effect strong?".

    Uses |t_stat|, not signed t_stat: this expert measures the
    STRENGTH/significance of the estimate (is |t_stat| large enough?), not
    the DIRECTION of the effect — direction is the responsibility of other
    experts (sign agreement with bootstrap, causal asymmetry, etc.).
    """
    m = _mass_by_threshold(abs(feat.t_stat), 1.645, scale=0.6,
                            base_uncertainty=0.10, favors_larger=True)
    return ("statistical", m)


def expert_bootstrap(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """Input: bootstrap mean, bootstrap std (via width), overlap."""
    if feat.boot_mean is None or feat.boot_stability is None:
        return ("bootstrap", dict(_NEUTRAL_MASS))

    signs_agree = (feat.theta * feat.boot_mean) > 0
    strength = _sigmoid((feat.boot_stability - 0.5) / 0.2)
    if not signs_agree:
        strength *= 0.3  # sign disagreement between theta and bootstrap mean is a strong red flag

    m_theta = 0.15
    m_h = strength * (1.0 - m_theta)
    m_nh = (1.0 - strength) * (1.0 - m_theta)
    return ("bootstrap", {"H": m_h, "notH": m_nh, "Theta": m_theta})


def expert_permutation(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """Exclusive input: permutation p-value (empirical, via circular shift)."""
    if feat.perm_pvalue is None:
        return ("permutation", dict(_NEUTRAL_MASS))
    m = _mass_by_threshold(feat.perm_pvalue, 0.10, scale=0.04,
                            base_uncertainty=0.10, favors_larger=False)
    return ("permutation", m)


def expert_robustness(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """
    Input: robustness_value, via the RESIDUAL relative to what is
    expected from cv_ci (Cinelli & Hazlett, 2020).
    """
    if feat.robustness_residual is None:
        return ("robustness", dict(_NEUTRAL_MASS))
    m = _mass_by_threshold(
        feat.robustness_residual, RESID_PROTECTION_MIN,
        scale=max(abs(RESID_PROTECTION_MIN) * 3, 0.01),
        base_uncertainty=0.30, favors_larger=True,
    )
    return ("robustness", m)


def expert_residual_quality(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """
    Combines RESET, dCor, HSIC and Spearman into ONE SINGLE diagnostic:
    "do the residuals indicate that the linear functional form may be
    misspecified?". Treated mostly as UNCERTAINTY (Θ), not as direct
    evidence against H — uncaptured curvature does not mean the absence
    of causality, only that linear theta may be poorly calibrated.
    """
    available_signals = []
    if feat.reset_pvalue is not None:
        available_signals.append(feat.reset_pvalue < 0.05)
    if feat.dcor is not None and feat.corr_linear is not None:
        available_signals.append(feat.dcor > (abs(feat.corr_linear) + 0.15))
    if feat.hsic is not None:
        available_signals.append(feat.hsic > 0.30)

    if not available_signals:
        return ("residual_quality", dict(_NEUTRAL_MASS))

    n_signals = len(available_signals)
    suspicious_fraction = sum(available_signals) / n_signals

    m_theta = 0.70 + 0.25 * suspicious_fraction
    m_h = 0.10 * (1.0 - suspicious_fraction)
    m_nh = max(0.0, 1.0 - m_theta - m_h)
    return ("residual_quality", {"H": m_h, "notH": m_nh, "Theta": min(m_theta, 1.0)})


def expert_ess(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """
    Input: ESS treatment, ESS outcome (uses the minimum of the two — the
    "bottleneck" of effective sample size determines the estimate's
    reliability). Modest vote (high base uncertainty) — low ESS alone is
    WEAK, not strong, evidence of estimation difficulty; see
    interaction_ess_x_bootstrap for the more informative combined case.
    """
    if feat.ess_min is None:
        return ("ess", dict(_NEUTRAL_MASS))
    m = _mass_by_threshold(feat.ess_min, 50.0, scale=25.0,
                            base_uncertainty=0.35, favors_larger=True)
    return ("ess", m)


def expert_temporal_structure(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """
    Combines lag (vs. the veto threshold), lag vs. the run's
    median_lag, and lag_penalty_score (domain heuristic, external to the
    DML cascade) into ONE temporal-structure expert — average of the
    available individual masses.
    """
    signals: List[Dict[str, float]] = []
    if feat.lag is not None:
        signals.append(_mass_by_threshold(feat.lag, LAG_VETO_MIN, scale=1.5,
                                           base_uncertainty=0.20, favors_larger=False))
    if feat.lag is not None and feat.lag_median is not None:
        signals.append(_mass_by_threshold(feat.lag - feat.lag_median, 0.0, scale=2.0,
                                           base_uncertainty=0.25, favors_larger=False))
    if feat.lag_penalty is not None:
        signals.append(_mass_by_threshold(feat.lag_penalty, 0.90, scale=0.08,
                                           base_uncertainty=0.35, favors_larger=True))

    if not signals:
        return ("temporal_structure", dict(_NEUTRAL_MASS))

    m_h = float(np.mean([s["H"] for s in signals]))
    m_nh = float(np.mean([s["notH"] for s in signals]))
    m_theta = max(0.0, 1.0 - m_h - m_nh)
    return ("temporal_structure", {"H": m_h, "notH": m_nh, "Theta": m_theta})


def expert_asymmetry(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """Exclusive input: causal_asymmetry (|theta_forward| - |theta_reverse|)."""
    if feat.causal_asymmetry is None:
        return ("asymmetry", dict(_NEUTRAL_MASS))
    m = _mass_by_threshold(feat.causal_asymmetry, 0.0, scale=0.05,
                            base_uncertainty=0.30, favors_larger=True)
    return ("asymmetry", m)


def expert_context_mode(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """
    NOT part of the 9 primary experts — a more skeptical prior for
    "rescued" pairs, via Granger OR Pearson rescue (see
    _stage2_feature_engineering/is_rescue). Documented separately to make
    clear this is an ADDITION beyond the core set, easy to remove without
    affecting the rest.
    """
    if not feat.mode_txt:
        return ("context_mode", dict(_NEUTRAL_MASS))
    m_nh = feat.prior_strength_nh
    return ("context_mode" + f" {feat.mode_txt}", {"H": 0.0, "notH": m_nh, "Theta": 1.0 - m_nh})


def expert_cascade_consensus(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """
    NOT part of the 9 primary experts. Input: cascade_confidence,
    n_models_agree, and whether the winning model was a fallback
    (dml_model_used contains "[fallback]" — no main model in the cascade
    passed the acceptance/rejection criteria and the returned result is
    only "the closest one").

    Captures a higher-level discriminant not captured by any other expert
    (statistical, robustness, or residual_quality): "no cascade model
    agreed" is information that does not directly appear in isolated
    theta/se/robustness_value.

    DELIBERATE ASYMMETRIC CALIBRATION: the "against H" side (fallback +
    low) is stronger and more consistent than the "in favor of H" side
    (high + high n_models_agree), which gets moderate weight so as not
    to over-reinforce a less discriminating signal.
    """
    if feat.cascade_confidence is None:
        return ("cascade_consensus", dict(_NEUTRAL_MASS))

    if feat.fallback_model and feat.cascade_confidence == "low":
        # notH=0.90: combined via Dempster-Shafer with the other ~9 experts
        # (most favoring H), most of this opposition turns into CONFLICT ->
        # uncertainty (Yager's rule), not net opposition. The mechanism
        # that rejects these cases is Support falling below the absolute
        # minimum floor (MIN_ABSOLUTE_SUPPORT_THRESHOLD) because of the
        # inflated uncertainty, which requires a high notH.
        return ("cascade_consensus", {"H": 0.0, "notH": 0.90, "Theta": 0.10})

    if feat.cascade_confidence == "high":
        # 'high' always supports H (never zeroed) since it IS, by the
        # cascade's own definition (dml_engine.py),
        # n_models_agree>=2 WITH sign agreement between models; strength is
        # modulated, not penalized, by n_models_agree.
        h_strength = 0.35 if (feat.n_models_agree is not None and feat.n_models_agree >= 3) else 0.22
        return ("cascade_consensus", {"H": h_strength, "notH": 0.0, "Theta": 1.0 - h_strength})

    if feat.cascade_confidence == "medium":
        return ("cascade_consensus", {"H": 0.10, "notH": 0.0, "Theta": 0.90})

    # low but not a pure fallback (e.g. low without explicit fallback)
    return ("cascade_consensus", {"H": 0.0, "notH": 0.25, "Theta": 0.75})


PRIMARY_EXPERTS = [
    expert_transfer_entropy,
    expert_statistical,
    expert_bootstrap,
    expert_permutation,
    expert_robustness,
    expert_residual_quality,
    expert_ess,
    expert_temporal_structure,
    expert_asymmetry,
    expert_cascade_consensus,
]


# =============================================================================
# STAGE 4 — META EVIDENCE (summarizes the joint behavior of the primaries)
# =============================================================================

def meta_consistency_evidence(
    primary_masses: List[Tuple[str, Dict[str, float]]]
) -> Tuple[str, Dict[str, float]]:
    """
    How much the primary experts agree with each other. For each expert
    with a genuine opinion (Theta < 0.95), computes lean = H - notH in
    [-1, 1]; high consistency = low standard deviation of the leans.
    Vote direction follows the mean sign of the leans.
    """
    leans = [m["H"] - m["notH"] for _, m in primary_masses if m["Theta"] < 0.95]
    if len(leans) < 2:
        return ("evidence_consistency", dict(_NEUTRAL_MASS))

    std = float(np.std(leans))
    mean_lean = float(np.mean(leans))
    consistency = max(0.0, 1.0 - std / 0.5)

    if mean_lean >= 0:
        m_h, m_nh = consistency * abs(mean_lean), 0.0
    else:
        m_h, m_nh = 0.0, consistency * abs(mean_lean)
    m_theta = max(0.0, 1.0 - m_h - m_nh)
    return ("evidence_consistency", {"H": m_h, "notH": m_nh, "Theta": m_theta})


def meta_consistency_bootstrap(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """Checks whether theta, bootstrap and permutation point in the same direction/strength."""
    signs_agree: List[bool] = []
    if feat.boot_mean is not None:
        signs_agree.append((feat.theta * feat.boot_mean) > 0)
    if feat.perm_pvalue is not None:
        signs_agree.append(feat.perm_pvalue < 0.10)

    if not signs_agree:
        return ("bootstrap_consistency", dict(_NEUTRAL_MASS))

    fraction = sum(signs_agree) / len(signs_agree)
    strength = 0.35
    if fraction >= 0.5:
        m_h, m_nh = strength * fraction, 0.0
    else:
        m_h, m_nh = 0.0, strength * (1.0 - fraction)
    m_theta = max(0.0, 1.0 - m_h - m_nh)
    return ("bootstrap_consistency", {"H": m_h, "notH": m_nh, "Theta": m_theta})


def meta_consistency_statistical(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """
    Checks coherence between theta, t-stat and the CONFIDENCE INTERVAL:
    compares the OBSERVED CI width against the width the normal
    approximation would imply from se (2*1.96*se). A large gap signals
    that the HAC normal approximation may be poorly calibrated (e.g.
    under strong residual autocorrelation) — information DISTINCT from
    expert_statistical, which only measures the t_stat's strength.
    """
    implied_ic_width = 2 * 1.96 * feat.se
    if implied_ic_width < 1e-9:
        return ("statistical_consistency", dict(_NEUTRAL_MASS))

    ratio = feat.ci_width / implied_ic_width
    gap = abs(ratio - 1.0)
    m = _mass_by_threshold(gap, 0.20, scale=0.15, base_uncertainty=0.40, favors_larger=False)
    return ("statistical_consistency", m)


def meta_consistency_causal(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """Checks coherence between TE, forward, reverse and causal asymmetry."""
    signals: List[bool] = []
    if feat.te is not None:
        signals.append(feat.te >= TE_VETO_MIN)
    if feat.causal_asymmetry is not None:
        signals.append(feat.causal_asymmetry > 0)

    if not signals:
        return ("causal_consistency", dict(_NEUTRAL_MASS))

    fraction = sum(signals) / len(signals)
    strength = 0.15 if len(signals) < 2 else 0.45
    if fraction >= 0.5:
        m_h, m_nh = strength * fraction, 0.0
    else:
        m_h, m_nh = 0.0, strength * (1.0 - fraction)
    m_theta = max(0.0, 1.0 - m_h - m_nh)
    return ("causal_consistency", {"H": m_h, "notH": m_nh, "Theta": m_theta})


def meta_minimum_causal_confirmation(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """
    Motivation: finite-sample noise passes t_stat/bootstrap/
    permutation with room to spare (the three fundamentally test, from
    different angles, "is this coefficient different from zero?" — they
    are not causal-specific evidence), and the fusion accepts even
    without ANY signal that discriminates spurious correlation from a
    genuine causal mechanism.

    This expert checks whether there is confirmation from AT LEAST ONE
    genuinely causal-specific source — TE above the threshold, a
    positive robustness residual (Cinelli & Hazlett), high cascade
    consensus (>=3 models agreeing), or causal asymmetry (forward >
    reverse). If NONE of these four sources support H — even if the
    entire statistical battery (t_stat/bootstrap/permutation) is strong
    — this expert pulls toward notH/uncertainty, because statistical
    significance alone does not distinguish causality from spurious
    correlation/confounding.

    Deliberately does NOT count how many of these four agree with each
    other (that is already the role of meta_consistency_causal) — it
    only checks whether SOME minimum confirmation exists, which is a
    different and complementary question.
    """
    causal_signals_available = 0
    causal_signals_positive = 0

    if feat.te is not None:
        causal_signals_available += 1
        if feat.te >= TE_VETO_MIN:
            causal_signals_positive += 1

    if feat.robustness_residual is not None:
        causal_signals_available += 1
        if feat.robustness_residual >= RESID_PROTECTION_MIN:
            causal_signals_positive += 1

    if feat.cascade_confidence is not None:
        causal_signals_available += 1
        if feat.cascade_confidence == "high":
            causal_signals_positive += 1

    if feat.causal_asymmetry is not None:
        causal_signals_available += 1
        if feat.causal_asymmetry > 0:
            causal_signals_positive += 1

    if causal_signals_available == 0:
        return ("minimum_causal_confirmation", dict(_NEUTRAL_MASS))

    if causal_signals_positive == 0:
        # NO causal-specific source confirms -- pulls strongly toward
        # uncertainty/opposition, regardless of how "significant" the
        # result looks on the statistical tests.
        #
        # notH=0.75: by the "conflict dilution" of Dempster-Shafer/Yager's
        # rule, notH needs to exceed ~0.70 for Support to fall below the
        # absolute minimum floor and the pair to be rejected.
        return ("minimum_causal_confirmation", {"H": 0.0, "notH": 0.75, "Theta": 0.25})

    if causal_signals_positive >= 2:
        # at least two causal-specific sources agree -- reinforcement
        return ("minimum_causal_confirmation", {"H": 0.20, "notH": 0.0, "Theta": 0.80})

    # exactly one source confirms -- neutral/light reinforcement
    return ("minimum_causal_confirmation", {"H": 0.08, "notH": 0.0, "Theta": 0.92})




# =============================================================================
# STAGE 5 — INTERACTION SPECIALISTS (relationships between distinct families)
# =============================================================================

def interaction_te_x_robustness(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """High TE + high robustness -> additional reinforcement of causal evidence."""
    if feat.te is None or feat.robustness is None:
        return ("te_x_robustness", dict(_NEUTRAL_MASS))
    if feat.te >= TE_VETO_MIN and feat.robustness >= 0.10:
        return ("te_x_robustness", {"H": 0.25, "notH": 0.0, "Theta": 0.75})
    return ("te_x_robustness", dict(_NEUTRAL_MASS))


def interaction_te_x_bootstrap(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """High TE + unstable bootstrap -> conflicting evidence."""
    if feat.te is None or feat.boot_stability is None:
        return ("te_x_bootstrap", dict(_NEUTRAL_MASS))
    if feat.te >= TE_VETO_MIN and feat.boot_stability < 0.4:
        return ("te_x_bootstrap", {"H": 0.0, "notH": 0.15, "Theta": 0.85})
    return ("te_x_bootstrap", dict(_NEUTRAL_MASS))


def interaction_bootstrap_x_residuals(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """Excellent bootstrap + bad residuals -> reduce confidence."""
    if feat.boot_stability is None:
        return ("bootstrap_x_residuals", dict(_NEUTRAL_MASS))
    bad_residual_signals = []
    if feat.reset_pvalue is not None:
        bad_residual_signals.append(feat.reset_pvalue < 0.05)
    if feat.dcor is not None and feat.corr_linear is not None:
        bad_residual_signals.append(feat.dcor > (abs(feat.corr_linear) + 0.15))
    if not bad_residual_signals:
        return ("bootstrap_x_residuals", dict(_NEUTRAL_MASS))

    bad_residual = (sum(bad_residual_signals) / len(bad_residual_signals)) >= 0.5
    good_bootstrap = feat.boot_stability > 0.7
    if good_bootstrap and bad_residual:
        return ("bootstrap_x_residuals", {"H": 0.0, "notH": 0.20, "Theta": 0.80})
    return ("bootstrap_x_residuals", dict(_NEUTRAL_MASS))


def interaction_robustness_x_reset(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """High robustness + bad RESET -> increases conflict."""
    if feat.robustness is None or feat.reset_pvalue is None:
        return ("robustness_x_reset", dict(_NEUTRAL_MASS))
    if feat.robustness >= 0.10 and feat.reset_pvalue < 0.05:
        return ("robustness_x_reset", {"H": 0.0, "notH": 0.15, "Theta": 0.85})
    return ("robustness_x_reset", dict(_NEUTRAL_MASS))


def interaction_asymmetry_x_te(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """High asymmetry + high TE -> strong additional causal support."""
    if feat.causal_asymmetry is None or feat.te is None:
        return ("asymmetry_x_te", dict(_NEUTRAL_MASS))
    if feat.causal_asymmetry > 0.05 and feat.te >= TE_VETO_MIN:
        return ("asymmetry_x_te", {"H": 0.25, "notH": 0.0, "Theta": 0.75})
    return ("asymmetry_x_te", dict(_NEUTRAL_MASS))


def interaction_ess_x_bootstrap(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """Low ESS + unstable bootstrap -> strong evidence against (compounds two weaknesses)."""
    if feat.ess_min is None or feat.boot_stability is None:
        return ("ess_x_bootstrap", dict(_NEUTRAL_MASS))
    if feat.ess_min < 50 and feat.boot_stability < 0.4:
        return ("ess_x_bootstrap", {"H": 0.0, "notH": 0.35, "Theta": 0.65})
    return ("ess_x_bootstrap", dict(_NEUTRAL_MASS))


def interaction_weak_consensus_x_missing_te(feat: FeatureSet) -> Tuple[str, Dict[str, float]]:
    """
    Motivation: cascade_confidence='medium' in isolation is not a
    reliable veto across domains — treating it as one would cost recall
    in scenarios where the cascade legitimately converges moderately.
    The genuine signal is not "weak cascade" in isolation, it is "weak
    cascade AND no information-transfer confirmation" — when the cascade
    has already not converged well AND there is no TE sustaining the
    result, there is little left genuinely supporting H.
    """
    if feat.cascade_confidence != "medium" or feat.te is None:
        return ("weak_consensus_x_missing_te", dict(_NEUTRAL_MASS))
    if feat.te < 0.15:
        return ("weak_consensus_x_missing_te", {"H": 0.0, "notH": 0.85, "Theta": 0.15})
    return ("weak_consensus_x_missing_te", dict(_NEUTRAL_MASS))


# =============================================================================
# STAGE 6 — EVIDENCE FUSION (Dempster-Shafer, Yager's rule)
# =============================================================================
# Performed in _evaluate_pair_v2 by the rule in `fusion_rule` (default
# COMBINATION_RULE, see the combination functions above) over the curated
# list of experts assembled there.


# =============================================================================
# STAGE 7 — FINAL DECISION
# =============================================================================

def _stage7_final_decision(
    final_mass: Dict[str, float],
    total_conflict: float,
    feat: FeatureSet,
    min_support_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    The final decision does NOT depend only on Belief(H) and Belief(notH)
    — it also combines consensus among experts and the average quality
    of the available evidence into a Final Confidence.

    min_support_threshold : optional override of the absolute Support
        floor (MIN_ABSOLUTE_SUPPORT_THRESHOLD). None (default) uses the
        module constant. AdaptiveCausalPenalty uses it to pass its
        calibrated floor.
    """
    if min_support_threshold is None:
        min_support_threshold = MIN_ABSOLUTE_SUPPORT_THRESHOLD
    support = final_mass["H"]
    opposition = final_mass["notH"]
    uncertainty = final_mass["Theta"]
    conflict = total_conflict

    consensus_score = max(0.0, 1.0 - conflict)

    # Evidence Quality: average of indicators of "how informative the
    # available evidence is" (not direction, only quality/precision).
    quality_signals: List[float] = []
    if feat.boot_stability is not None:
        quality_signals.append(feat.boot_stability)
    if feat.robustness is not None:
        quality_signals.append(min(max(feat.robustness, 0.0), 1.0))
    if feat.te is not None:
        quality_signals.append(min(feat.te / max(TE_VETO_MIN * 2, 1e-9), 1.0))
    if feat.reset_pvalue is not None:
        quality_signals.append(1.0 if feat.reset_pvalue >= 0.05 else 0.5)
    evidence_quality = float(np.mean(quality_signals)) if quality_signals else 0.5

    net_support = support - opposition
    decided_mass = support + opposition
    net_ratio = (net_support / decided_mass) if decided_mass > 1e-9 else 0.0

    final_confidence = max(0.0, min(1.0,
        0.5 * max(0.0, net_support) + 0.3 * consensus_score + 0.2 * evidence_quality
    ))

    # Decides by RATIO within the decided mass (Support+Opposition), not by
    # the raw difference — see the comment next to NET_RATIO_MIN_THRESHOLD.
    # The absolute floor prevents accepting on a high ratio with negligible
    # decided mass; the Opposition ceiling is a separate safety safeguard.
    accepted = (
        net_ratio >= NET_RATIO_MIN_THRESHOLD
        and support >= min_support_threshold
        and opposition < MAX_OPPOSITION_THRESHOLD
    )

    return {
        "support": round(support, 4),
        "opposition": round(opposition, 4),
        "uncertainty": round(uncertainty, 4),
        "conflict": round(conflict, 4),
        "consensus_score": round(consensus_score, 4),
        "evidence_quality": round(evidence_quality, 4),
        "final_confidence": round(final_confidence, 4),
        "net_support": round(net_support, 4),
        "net_ratio": round(net_ratio, 4),
        "decided_mass": round(decided_mass, 4),
        "accepted": bool(accepted),
    }


# =============================================================================
# Full orchestration (all 7 stages)
# =============================================================================

def _evaluate_pair_v2(
    row: Any,
    target_fn: Optional[Any] = None,
    experts_cfg: ExpertsConfig = DEFAULT_EXPERTS_CFG,
    domain_gate_fn: Optional[Any] = None,
    fusion_rule: str = COMBINATION_RULE,
    min_support_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Core of the selector: runs the 7 stages in sequence and returns the FULL STATE
    for auditing (None if a Structural Gate — universal or domain-
    specific — cut the pair before the fusion started).

    domain_gate_fn : Optional[Callable[[Any], Optional[str]]] — optional
        DOMAIN-SPECIFIC gate plugin, runs AFTER the universal gates in
        _stage1_structural_gates. None by default (no domain rule
        applied). See create_blocked_property_gate() for an example.

    min_support_threshold : see docstring of _stage7_final_decision —
        targeted override of the absolute Support floor, None uses the
        module constant.
    """
    pair_info = (
        f"{_get(row, 'source_entity', default='?')}"
        f"->{_get(row, 'target_entity', default='?')}"
    )

    def log(msg: str) -> None:
        if target_fn:
            print(f"{pair_info} | {msg}")

    # -- Stage 1 (universal gates) --
    gate_reason = _stage1_structural_gates(row)
    if gate_reason is not None:
        log(f"[universal GATE] {gate_reason}")
        return None

    # -- Stage 1b (domain gate, optional — see docstring above) --
    if domain_gate_fn is not None:
        domain_gate_reason = domain_gate_fn(row)
        if domain_gate_reason is not None:
            log(f"[domain GATE] {domain_gate_reason}")
            return None

    # -- Stage 2 --
    feat = _stage2_feature_engineering(row)

    # -- Stage 3 (primaries, always active — mandatory) --
    primary_masses = [fn(feat) for fn in PRIMARY_EXPERTS]
    primary_masses.append(expert_context_mode(feat))  # mode prior, see expert_context_mode

    # -- Stage 4 (meta — curated via experts_cfg) --
    meta_masses: List[Tuple[str, Dict[str, float]]] = []
    if experts_cfg.enable_evidence_consistency:
        meta_masses.append(meta_consistency_evidence(primary_masses))
    if experts_cfg.enable_bootstrap_consistency:
        meta_masses.append(meta_consistency_bootstrap(feat))
    if experts_cfg.enable_statistical_consistency:
        meta_masses.append(meta_consistency_statistical(feat))
    if experts_cfg.enable_causal_consistency:
        meta_masses.append(meta_consistency_causal(feat))
    if experts_cfg.enable_minimum_causal_confirmation:
        meta_masses.append(meta_minimum_causal_confirmation(feat))

    # -- Stage 5 (interaction — curated via experts_cfg) --
    interaction_masses: List[Tuple[str, Dict[str, float]]] = []
    if experts_cfg.enable_te_x_robustness:
        interaction_masses.append(interaction_te_x_robustness(feat))
    if experts_cfg.enable_te_x_bootstrap:
        interaction_masses.append(interaction_te_x_bootstrap(feat))
    if experts_cfg.enable_bootstrap_x_residuals:
        interaction_masses.append(interaction_bootstrap_x_residuals(feat))
    if experts_cfg.enable_robustness_x_reset:
        interaction_masses.append(interaction_robustness_x_reset(feat))
    if experts_cfg.enable_asymmetry_x_te:
        interaction_masses.append(interaction_asymmetry_x_te(feat))
    if experts_cfg.enable_ess_x_bootstrap:
        interaction_masses.append(interaction_ess_x_bootstrap(feat))
    if experts_cfg.enable_weak_consensus_x_missing_te:
        interaction_masses.append(interaction_weak_consensus_x_missing_te(feat))

    experts_active_in_fusion = primary_masses + meta_masses + interaction_masses

    # -- Stage 6 --
    if fusion_rule == "hierarchical":
        final_mass, total_conflict, conflicts_per_pair = _combine_hierarchical(experts_active_in_fusion)
    elif fusion_rule == "hybrid_light":
        final_mass, total_conflict, conflicts_per_pair = _combine_light_hybrid(experts_active_in_fusion)
    elif fusion_rule == "hybrid_gated":
        final_mass, total_conflict, conflicts_per_pair = _combine_gated_hybrid(experts_active_in_fusion, feat)
    else:
        final_mass, total_conflict, conflicts_per_pair = _combine_all(
            experts_active_in_fusion, rule=fusion_rule
        )

    # -- Stage 7 --
    decision = _stage7_final_decision(final_mass, total_conflict, feat, min_support_threshold=min_support_threshold)

    if target_fn:
        log(
            f"[FUSION] n_experts={len(experts_active_in_fusion)} "
            f"Support={decision['support']:.4f} Opposition={decision['opposition']:.4f} "
            f"Uncertainty={decision['uncertainty']:.4f} Conflict={decision['conflict']:.4f} "
            f"Consensus={decision['consensus_score']:.4f} "
            f"EvidenceQuality={decision['evidence_quality']:.4f} "
            f"FinalConfidence={decision['final_confidence']:.4f}"
        )
        for name, k in conflicts_per_pair:
            if k > 0.01:
                log(f"    conflict incorporating '{name}': k={k:.4f}")
        log(f"{'[ACCEPTED]' if decision['accepted'] else '[REJECTED]'}")

    # -- all experts computed in full (even those off in the fusion) for
    # audit via causal_pen_diagnostics --
    full_meta_masses = {
        "evidence_consistency": meta_consistency_evidence(primary_masses),
        "bootstrap_consistency":  meta_consistency_bootstrap(feat),
        "statistical_consistency": meta_consistency_statistical(feat),
        "causal_consistency":     meta_consistency_causal(feat),
        "minimum_causal_confirmation": meta_minimum_causal_confirmation(feat),
    }
    full_interaction_masses = {
        "te_x_robustness":        interaction_te_x_robustness(feat),
        "te_x_bootstrap":       interaction_te_x_bootstrap(feat),
        "bootstrap_x_residuals": interaction_bootstrap_x_residuals(feat),
        "robustness_x_reset":     interaction_robustness_x_reset(feat),
        "asymmetry_x_te":      interaction_asymmetry_x_te(feat),
        "ess_x_bootstrap":      interaction_ess_x_bootstrap(feat),
        "weak_consensus_x_missing_te": interaction_weak_consensus_x_missing_te(feat),
    }

    return {
        **decision,
        "feature_set": feat,
        "primary_experts": primary_masses,
        "active_meta_experts": meta_masses,
        "active_interaction_experts": interaction_masses,
        "full_meta_experts": full_meta_masses,
        "full_interaction_experts": full_interaction_masses,
        "conflicts_by_expert": conflicts_per_pair,
        "n_experts_in_fusion": len(experts_active_in_fusion),
    }


# =============================================================================
# Public interface
# =============================================================================

def causal_pen(
    row: Any,
    target_fn: Optional[Any] = None,
    domain_gate_fn: Optional[Any] = None,
    fusion_rule: str = COMBINATION_RULE,
    min_support_threshold: Optional[float] = None,
) -> float:
    """
    Thin wrapper: delegates to _evaluate_pair_v2 and returns only the
    float (causal_pen(row) or causal_pen(row, target_fn)). Structural
    Gates (universal or domain-specific) return 0.0 directly, without
    going through the fusion.

    domain_gate_fn : see docstring of _evaluate_pair_v2 — optional
        plugin for domain-specific admissibility rules (e.g.
        create_blocked_property_gate()). None by default.

    min_support_threshold : override of the absolute Support floor (see
        docstring of _stage7_final_decision). None (default) uses the
        module constant.
    """
    state = _evaluate_pair_v2(
        row, target_fn=target_fn, domain_gate_fn=domain_gate_fn, fusion_rule=fusion_rule,
        min_support_threshold=min_support_threshold,
    )
    if state is None:
        return 0.0
    if CONTINUOUS_RETURN:
        return state["final_confidence"] if state["accepted"] else 0.0
    return 1.0 if state["accepted"] else 0.0


def causal_pen_diagnostics(
    row: Any,
    target_fn: Optional[Any] = None,
    domain_gate_fn: Optional[Any] = None,
    fusion_rule: str = COMBINATION_RULE,
    min_support_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Audit/benchmark version of causal_pen: returns the FULL state from
    _evaluate_pair_v2 (all 7 stages, including experts off in the fusion
    by default, computed separately for inspection), or a minimal dict
    when a Structural Gate decided before the fusion.
    """
    state = _evaluate_pair_v2(
        row, target_fn=target_fn, domain_gate_fn=domain_gate_fn, fusion_rule=fusion_rule,
        min_support_threshold=min_support_threshold,
    )
    if state is None:
        reason = _stage1_structural_gates(row)
        if reason is None and domain_gate_fn is not None:
            reason = domain_gate_fn(row)
        return {
            "support": 0.0, "opposition": 1.0, "uncertainty": 0.0,
            "conflict": 0.0, "consensus_score": 0.0, "evidence_quality": 0.0,
            "final_confidence": 0.0, "net_support": -1.0,
            "net_ratio": -1.0, "decided_mass": 1.0, "accepted": False,
            "gate_reason": reason,
            "feature_set": None,
            "primary_experts": [],
            "active_meta_experts": [], "active_interaction_experts": [],
            "full_meta_experts": {}, "full_interaction_experts": {},
            "conflicts_by_expert": [], "n_experts_in_fusion": 0,
        }
    return state


# =============================================================================
# ADAPTIVE SUPPORT FLOOR PER DATASET (no ground truth, no hardcoding by
# name).
# =============================================================================
#
# Motivation: a single fixed Support floor (MIN_ABSOLUTE_SUPPORT_THRESHOLD)
# treats every dataset alike, but the Support scale depends on the
# dataset: in weak-signal data (no real time axis, small n) TRUE pairs
# carry structurally lower Support (less evidence mass per pair, not
# because the relationship is spurious), so a fixed floor suited to
# strong, clean signal rejects them.
#
# AdaptiveCausalPenalty calibrates the floor, per dataset, as a
# PERCENTILE of the Support distribution observed among the candidates
# that already pass the other criteria (net_ratio and Opposition) —
# without ever looking at causal_gt. Datasets with many strong-evidence
# pairs (clean synthetic ones) tend to have this percentile still high
# (effective floor close to the fixed one); weak-signal datasets
# tend to have this percentile lower (more permissive effective floor) —
# WITHOUT any dataset name appearing in the code.
class AdaptiveCausalPenalty:
    """
    causal_pen wrapper that replaces the fixed global floor
    (MIN_ABSOLUTE_SUPPORT_THRESHOLD) with a floor calibrated from the
    empirical distribution of Support observed among the candidates of
    the dataset being processed ITSELF — without using ground truth and
    without hardcoding by dataset name (see note above).

    Usage: pass an INSTANCE of this class as `causal_penalty_fn` in the
    pipeline's config_override. `Pipeline.pipeline.run_pipeline` detects
    the type and calls `.calibrate(results)` on the candidates DataFrame
    BEFORE applying the causal penalty (Stage 7) — after that the
    instance behaves like a regular causal_penalty_fn (callable
    row-by-row).

    percentile : which percentile of the Support distribution (among
        candidates that already pass net_ratio/Opposition) becomes the
        floor. A meta-parameter of the calibration MECHANISM (not tuned
        per dataset, nor chosen by looking at any specific dataset's
        result) — 50 (MEDIAN) by default: accepts an evidentially
        unambiguous candidate if its Support is in the upper half of the
        distribution observed in that dataset. It is the only point of
        the distribution with a standard statistical justification
        (center), with no extra degree of freedom to tune.
    min_absolute_floor : safety floor — never calibrates below this,
        even if the computed percentile gives a smaller value (avoids
        accepting by statistical chance in datasets with few candidates
        or a distribution degenerate near zero).
    max_absolute_floor : safety ceiling — never calibrates above this
        (None = no ceiling, only the floor). Exists because the computed
        percentile can end up ABOVE the fixed global threshold
        (0.10) in strong-signal datasets — without a ceiling this would
        make the gate MORE restrictive than the fixed floor there and
        would hurt recall. A ceiling at 0.10 (the fixed global value)
        guarantees the calibrated floor is only EQUAL to or MORE
        PERMISSIVE than the fixed one, never more restrictive —
        reducing the mechanism to an "adaptive floor within a range",
        not a free recalibration.
    min_n_candidates : below this number of "clean" candidates
        (already passing net_ratio/Opposition), the estimated percentile
        is not reliable (e.g. a median of 6 points is noise, not
        signal) — falls back to the fixed global floor
        (MIN_ABSOLUTE_SUPPORT_THRESHOLD) instead of calibrating. Standard
        statistical principle (minimum sample size to trust a
        percentile), not an ad hoc tweak.
    """

    def __init__(
        self,
        percentile: float = 50.0,
        min_absolute_floor: float = 0.02,
        max_absolute_floor: Optional[float] = None,
        min_n_candidates: int = 10,
        domain_gate_fn: Optional[Any] = None,
        fusion_rule: str = COMBINATION_RULE,
    ):
        self.percentile = percentile
        self.min_absolute_floor = min_absolute_floor
        self.max_absolute_floor = max_absolute_floor
        self.min_n_candidates = min_n_candidates
        self.domain_gate_fn = domain_gate_fn
        self.fusion_rule = fusion_rule
        self.calibrated_threshold_: Optional[float] = None
        self.n_calibration_candidates_: Optional[int] = None

    def calibrate(self, results: 'pd.DataFrame') -> float:
        """Computes the floor from `results` (post-DML candidates,
        BEFORE the causal penalty) and stores it in
        self.calibrated_threshold_. Called automatically by run_pipeline;
        can be called manually to inspect the value before running."""
        supports: List[float] = []
        for _, row in results.iterrows():
            state = _evaluate_pair_v2(
                row, domain_gate_fn=self.domain_gate_fn, fusion_rule=self.fusion_rule,
                min_support_threshold=0.0,
            )
            if state is None:
                continue
            if state["net_ratio"] >= NET_RATIO_MIN_THRESHOLD and state["opposition"] < MAX_OPPOSITION_THRESHOLD:
                supports.append(state["support"])

        self.n_calibration_candidates_ = len(supports)
        if len(supports) < self.min_n_candidates:
            self.calibrated_threshold_ = MIN_ABSOLUTE_SUPPORT_THRESHOLD
        else:
            value = max(self.min_absolute_floor, float(np.percentile(supports, self.percentile)))
            if self.max_absolute_floor is not None:
                value = min(value, self.max_absolute_floor)
            self.calibrated_threshold_ = value
        return self.calibrated_threshold_

    def __call__(self, row: Any, target_fn: Optional[Any] = None) -> float:
        return causal_pen(
            row, target_fn=target_fn, domain_gate_fn=self.domain_gate_fn, fusion_rule=self.fusion_rule,
            min_support_threshold=self.calibrated_threshold_,
        )

    def __repr__(self) -> str:
        if self.calibrated_threshold_ is not None:
            return (f"AdaptiveCausalPenalty(percentile={self.percentile}, "
                    f"calibrated_threshold={self.calibrated_threshold_:.4f}, "
                    f"n={self.n_calibration_candidates_})")
        return f"AdaptiveCausalPenalty(percentile={self.percentile}, NOT_CALIBRATED)"


class RankMeanSelector(AdaptiveCausalPenalty):
    """Accepts a candidate whose mean percentile rank over six evidence
    sources, computed within the run itself, exceeds `cut`.

    Sources (all oriented so that larger = more evidence): |t_stat|,
    transfer_entropy, -permutation_pvalue, robustness_value,
    causal_asymmetry, n_models_agree. Ranks are taken over ALL candidates
    of the run (not only the significant ones), ties get the average rank,
    missing sources are skipped in the mean, and a row without t_stat is
    rejected. Transfer Entropy is zero for most rescued pairs, so their
    tied (low) rank acts as a discount for rescued pairs.

    Does NOT apply the deterministic Stage 7 gates nor a minimum sample
    size. Rides on AdaptiveCausalPenalty only so that run_pipeline calls
    .calibrate(results) before applying the selector row by row."""

    def __init__(self, cut: float = 0.5):
        super().__init__()
        self.cut = cut
        self.percentile = cut * 100  # only for the pipeline's log message
        self._score = None

    def calibrate(self, results: 'pd.DataFrame') -> float:
        import pandas as pd

        if not results.index.is_unique:
            raise ValueError("RankMeanSelector needs a unique results index.")
        num = lambda c: (pd.to_numeric(results[c], errors="coerce")
                         if c in results.columns
                         else pd.Series(np.nan, index=results.index))
        t = num("t_stat").abs()
        ranks = pd.DataFrame({
            "t": t,
            "te": num("transfer_entropy"),
            "perm": -num("permutation_pvalue"),
            "rv": num("robustness_value"),
            "asym": num("causal_asymmetry"),
            "agree": num("n_models_agree"),
        }).rank(pct=True)
        self._score = ranks.mean(axis=1, skipna=True).where(t.notna())
        self.n_calibration_candidates_ = int(t.notna().sum())
        self.calibrated_threshold_ = self.cut
        return self.cut

    def __call__(self, row: Any, target_fn: Optional[Any] = None) -> float:
        s = self._score.get(row.name, float("nan"))
        return 1.0 if s > self.cut else 0.0

    def __repr__(self) -> str:
        return f"RankMeanSelector(cut={self.cut})"
