"""
Config.sensitivity
===============================================================================
Protocol of the threshold sensitivity analysis (main_sensitivity.py): which
cut-offs are varied, over which grids, how the joint configurations are drawn,
and what "robust" means. FIXED BEFORE THE RUN: main_sensitivity.py writes this
protocol to <OUTPUT_DIR>/protocol.json when a run starts and refuses to
continue a run whose stored protocol differs. The reported defaults are NOT
changed on the result of this analysis.

Every configuration is the reference model (REFERENCE_MODEL) with the listed
cut-offs replaced; nothing else changes.

Robustness criterion (ROBUSTNESS below). With delta = mean_f1_tolerance, a
configuration is "within tolerance" when all three hold:

  R1  the mean F1 differs from the default's by at most delta, over all
      scenarios, over the development suite and over the held-out datasets,
      each taken separately;
  R2  the number of scenarios within `within_mark` F1 of the best of the six
      methods falls by at most `max_within_drop`;
  R3  the pipeline's mean shortfall to the best of the six methods stays
      below that of every baseline.

A cut-off is ROBUST over its grid when every grid value is within tolerance;
otherwise the largest interval of consecutive grid values around the default
that is within tolerance is reported. The joint variation is robust when at
least `joint_min_fraction` of the drawn configurations are within tolerance.
An interaction between cut-offs is flagged when the mean F1 of a joint
configuration differs by more than delta from the sum of the one-at-a-time
effects (linear interpolation of each one-at-a-time curve).
"""

REFERENCE_MODEL = 'patch_tau10'


def _grid(lo: float, hi: float, step: float) -> list:
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 6) for i in range(n + 1)]


# kind: how main_sensitivity.py applies the value
#   'config'    key `target` of run_pipeline
#   'time_axis' alpha_white_noise of Pipeline.discovery._detect_time_axis
#               (band = value / sqrt(n))
#   'penalty'   attribute `target` of the AdaptiveCausalPenalty instance
#   'selector'  module constant `target` of Pipeline.SelectorFunc
THRESHOLDS = [
    # -- regime cut-offs (one decision per dataset) --
    dict(key='outcome_r2', group='regime', kind='config',
         target='dml_auto_max_outcome_r2',
         label='Outcome R² (Stage 6 control set)',
         default=0.5, grid=_grid(0.30, 0.70, 0.05)),
    dict(key='coherence_rho', group='regime', kind='config',
         target='evidence_coherence_min_rho',
         label='ρ of the coherence guard',
         default=0.3, grid=_grid(0.10, 0.50, 0.05)),
    dict(key='delta_bic', group='regime', kind='config',
         target='predictive_min_delta_bic',
         label='ΔBIC of the predictive competition',
         default=6.0, grid=_grid(2.0, 10.0, 1.0)),
    dict(key='past_r2', group='regime', kind='config',
         target='contemporaneous_max_past_r2',
         label='Past R² (Stage 8b gate; guard exception)',
         default=0.25, grid=_grid(0.10, 0.45, 0.05)),
    dict(key='time_axis_band', group='regime', kind='time_axis',
         target='alpha_white_noise',
         label='Time-axis band (c in c/√n)',
         default=2.0, grid=_grid(1.5, 3.0, 0.25)),
    # -- Stage 7 cut-offs (one decision per candidate pair) --
    dict(key='floor_percentile', group='stage7', kind='penalty',
         target='percentile',
         label='Support-floor percentile',
         default=45.0, grid=_grid(25.0, 65.0, 5.0)),
    dict(key='floor_few_candidates', group='stage7', kind='selector',
         target='MIN_ABSOLUTE_SUPPORT_THRESHOLD',
         label='Support floor with few candidates',
         default=0.10, grid=_grid(0.05, 0.15, 0.025)),
    dict(key='support_ratio', group='stage7', kind='selector',
         target='NET_RATIO_MIN_THRESHOLD',
         label='Support ratio',
         default=0.55, grid=_grid(0.35, 0.75, 0.05)),
    dict(key='opposition_ceiling', group='stage7', kind='selector',
         target='MAX_OPPOSITION_THRESHOLD',
         label='Opposition ceiling',
         default=0.45, grid=_grid(0.25, 0.65, 0.05)),
]

# Joint variation: Latin hypercube over all the cut-offs above, each uniform
# on [min(grid), max(grid)].
JOINT = dict(n=150, seed=20261010)

ROBUSTNESS = dict(
    mean_f1_tolerance=0.02,
    within_mark=0.10,
    max_within_drop=3,
    joint_min_fraction=0.90,
)


def protocol() -> dict:
    """The protocol as plain data (what is locked in protocol.json)."""
    return dict(reference_model=REFERENCE_MODEL, thresholds=THRESHOLDS,
                joint=JOINT, robustness=ROBUSTNESS)
