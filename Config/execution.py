"""
Config.execution
===============================================================================
Run control panel: execution flags, datasets/benchmarks active in this
run, MODELS and PIPELINE_CONFIG — kept in this module so that switching
scenario/config does not require editing main.py (runner/validator)
directly.

main.py imports this module and uses the names directly:

    from Config import execution as cfg_exec
    ...
    if cfg_exec.USE_TIGRAMITE_BENCHMARK:
        ...
    run_all(..., models=cfg_exec.MODELS, config=cfg_exec.PIPELINE_CONFIG, ...)

Sections below:
    1) Execution control (what to run, validation flags)
    2) Datasets/benchmarks active in this run (the development suite)
    3) Models (MODELS)
    4) Base pipeline configuration (PIPELINE_CONFIG)
"""

import math
import os

from Pipeline.pipeline import run_pipeline as rp_v2
from Pipeline.penalties import KneeLagPenalty
from Pipeline.SelectorFunc import causal_pen, AdaptiveCausalPenalty, RankMeanSelector

from ModelComparison.pipeline_pcmci_tigramite import run_pipeline_pcmci_tigramite
from ModelComparison.pipeline_pcmciplus import run_pipeline_pcmciplus
from ModelComparison.pipeline_varlingam import run_pipeline_varlingam
from ModelComparison.pipeline_dynotears import run_pipeline_dynotears
from ModelComparison.pipeline_granger import run_pipeline_granger
from ModelComparison.pipeline_transfer_entropy import run_pipeline_transfer_entropy

# Models actually run by main.py (it reads MODELS, built from ALL_MODELS at the
# end of section 3). ALL models are active in this run, each at two lag
# windows: tau=24 (plain name) and tau=10 ('<name>_tau10'; the baselines'
# variants are generated from their tau=24 entries at the end of section 3).
# The GPDC baselines (PCMCI-GPDC, PCMCI+-GPDC) are defined and run only by
# main_gpdc.py. Remove a name here to skip that model.
ACTIVE_MODEL_NAMES = [
    'patch',           'patch_tau10',
    'patch_rank',      'patch_rank_tau10',
    'pcmci_tigramite','pcmci_tigramite_tau10',
    'pcmciplus',       'pcmciplus_tau10',
    'varlingam',       'varlingam_tau10',
    'dynotears',       'dynotears_tau10',
    'cuts_plus',       'cuts_plus_tau10',
    'granger',         'granger_tau10',
    'transfer_entropy', 'transfer_entropy_tau10',
]

# cuts_plus imports torch at module level, and torch fails to load on some
# machines (e.g. OSError WinError 1114 on c10.dll). Import it only when the
# model is active, so an inactive model can never break unrelated runs.
run_pipeline_cuts_plus = None
if any(n.startswith('cuts_plus') for n in ACTIVE_MODEL_NAMES):
    from ModelComparison.pipeline_cuts_plus import run_pipeline_cuts_plus


# =============================================================================
# 1) EXECUTION CONTROL
# =============================================================================

RUN_RUNNER      = True
RUN_VALIDATION  = True
SHOW_PAIRS      = True  # prints TP_hard/TP_soft/FP_hard/FN to the console

SOFT_THRESHOLD = 1

# DETECT_INTRA_ENTITY: controls whether INTRA-ENTITY pairs (same entity,
# different properties — e.g. F_0 -> F_1 within the same entity, present
# in the tigramite benchmark's 'quadratic'/'threshold'/'saturation'
# scenarios) enter the validation score.
#   True  (default) — these pairs count normally as TP/FP/FN/TN.
#   False            — these pairs are removed from the ground truth and
#                       from every model's predictions before any metric
#                       is computed (neither rewarded nor penalized).
#                       Use when the pipeline being evaluated was not
#                       designed to detect relationships within the same
#                       entity.
# Affects ONLY the VALIDATION step (Step 2) — it does not change what the
# runner runs nor what the pipeline itself tries to detect. Works
# retroactively: it can be toggled and validation re-run over the same
# OUTPUT_DIR without needing to run the runner again.
DETECT_INTRA_ENTITY = True

# USE_CANONICAL_PAIR: controls whether A->B and B->A are treated as the
# SAME pair (True) or as DISTINCT pairs (False
# — the model only gets it right if it detects the correct direction of
# the relationship).
#   True  — correct and recommended for datasets where entities have
#           fixed, disjoint roles (sources vs. targets).
#   False — recommended for the literature benchmarks (Lorenz-96,
#           logistic maps, tigramite scenarios), where variables have no
#           fixed role and the direction of the relationship genuinely
#           matters.
# Like DETECT_INTRA_ENTITY, this affects only the VALIDATION step and
# works retroactively over already-run datasets.
USE_CANONICAL_PAIR = False

# USE_CANONICAL_PAIR_DEFAULT: default used by the entire validation layer
# in main.py (_pair_key, validate_dataset/validate_all, etc.) when
# `use_canonical_pair` is not explicitly passed to the call.
# It is INDEPENDENT of USE_CANONICAL_PAIR above (which is the value
# actually passed by the entry point) — kept separate so that other entry
# points/tests can use a different default without touching the main run.
USE_CANONICAL_PAIR_DEFAULT = True


# =============================================================================
# 2) DATASETS / BENCHMARKS ACTIVE IN THIS RUN
# =============================================================================
#
# Internal synthetics (linear — VAR with lags): generated IN MEMORY by
#   Data/synthetic_var.py (generate_synthetic_var_benchmark_scenarios), via
#   the USE_SYNTHETIC_VAR_BENCHMARK flag + the SYNTHETIC_VAR_SCENARIOS list
#   below.
#
# Established literature benchmarks:
#   USE_TIGRAMITE_BENCHMARK      — nonlinear SCM via tigramite (5 scenarios)
#   USE_LORENZ96_BENCHMARK       — classic chaotic Lorenz-96 system
#   USE_LOGISTIC_BENCHMARK       — classic coupled logistic maps
#   USE_CAUSALTIME_BENCHMARK     — medical/pm25/traffic domains (CausalTime)
#   USE_ALARM_BENCHMARK          — ALARM Bayesian network (bnlearn)
#   USE_NETSIM_BENCHMARK         — NetSim simulated fMRI (Smith et al. 2011)
#
# The suite above is the DEVELOPMENT SUITE (run by main.py). The held-out
# datasets (Sachs, PeMS-BAY, Walker Circulation, PetShop, TEP, BEELINE, Causal
# Chambers, DREAM3, EasyVista, RCAEval) are evaluated OUTSIDE it by
# main_heldout.py — their flags and paths live in that script, not here.

# USE_SYNTHETIC_VAR_BENCHMARK: generates the internal synthetic scenarios
# in memory via Data/synthetic_var.py
# (generate_synthetic_var_benchmark_scenarios). Each tuple in
# SYNTHETIC_VAR_SCENARIOS becomes ONE independent call to the generator
# (name_prefix, seed, n_points, n_entities, prob_edge, max_parents,
# max_lag, min_edges) — one call per tuple, since each scenario has its
# own combination of size/density (the generator's `seeds` parameter only
# varies the seed within the SAME set of the other parameters). The
# ground truth (`causal_gt`) is guaranteed by construction, exactly like
# the other providers.
USE_SYNTHETIC_VAR_BENCHMARK = True

# (name_prefix, seed, n_points, n_entities, prob_edge, max_parents, max_lag, min_edges)
SYNTHETIC_VAR_SCENARIOS = [
    ("benchmark_synthetic_var_1", 11, 180, 5,  0.35, 2, 3, 1),
    ("benchmark_synthetic_var_2", 22, 180, 6,  0.35, 2, 3, 2),
    ("benchmark_synthetic_var_3", 33, 240, 12, 0.30, 3, 3, 5),
    ("benchmark_synthetic_var_4", 44, 240, 22, 0.20, 3, 4, 10),
    ("benchmark_synthetic_var_5", 55, 300, 40, 0.12, 3, 4, 18),
]

USE_TIGRAMITE_BENCHMARK  = True
USE_LORENZ96_BENCHMARK   = True
USE_LOGISTIC_BENCHMARK   = True

# USE_CAUSALTIME_BENCHMARK: enables the Data/causaltime.py provider
# (generate_causaltime_benchmark_scenarios), which reads the CausalTime
# medical/pm25/traffic domains (gen_data.npy + graph.npy) and assembles
# scenarios in the pipeline's standard format (see Data/causaltime.py).
USE_CAUSALTIME_BENCHMARK = True

# Root path of the CausalTime data (medical/pm25/traffic folders, each
# with gen_data.npy + graph.npy). Can be left as None to use
# Data/causaltime.py's default (env CAUSALTIME_DIR or
# Data/CausalTimeData/Dataset) — adjust here if the data lives elsewhere.
CAUSALTIME_DIR_PATH = "Data/CausalTimeData/Dataset"

# USE_ALARM_BENCHMARK: ALARM Bayesian network (Beinlich et al. 1989, 37
#   nodes/46 edges, expert-curated GT via bnlearn; see Data/alarm.py).
#   Automatic download (with cache) -- no manual file needed. NO real time
#   axis (data sampled i.i.d. from the network).
USE_ALARM_BENCHMARK = True

# USE_NETSIM_BENCHMARK: NetSim simulated BOLD fMRI, sim3 (Smith et al. 2011,
#   15 nodes x 200 points per simulated subject; GT = the network used by the
#   simulator; see Data/netsim.py). One scenario per subject of
#   NETSIM_SUBJECTS (sim3 has 50, indices 0..49). Requires sim3.mat in
#   ./external_data/netsim (or env NETSIM_DIR).
USE_NETSIM_BENCHMARK = True
NETSIM_SUBJECTS = (0, 1, 2)


# =============================================================================
# 3) MODELS
# =============================================================================
# pcmci_tigramite : official PCMCI (tigramite library). Requires
#                    `pip install tigramite` — if not installed, the
#                    runner catches the ImportError and skips this model
#                    without interrupting the rest.
# Together with the generic-penalty pipelines (patch/random_forest/
# kernel_ridge/bayesian_ridge) and the two most recent methods from the
# literature (dynotears/cuts_plus).
#
# Native DML cascade in Pipeline/dml_engine.py (BR -> RF -> KRR ->
# NadarayaWatson — see the module docstring). `rp_v2` uses this cascade
# directly.
#
# ADAPTIVE_CAUSAL_PENALTY: replaces Stage 7's fixed global Support floor
# with one calibrated per dataset from the empirical Support distribution
# of that run's own candidates (a percentile of the distribution, no
# ground truth, no dataset name hardcoded). See
# Pipeline.SelectorFunc.AdaptiveCausalPenalty for the full rationale.
#
# Single instance shared across datasets; each dataset calls .calibrate()
# again before use. Not safe for parallel use across datasets.
ADAPTIVE_CAUSAL_PENALTY = AdaptiveCausalPenalty(
    percentile=45.0, min_absolute_floor=0.03, max_absolute_floor=0.10,
    min_n_candidates=10,
)

# Reference lag window and lag-penalty shape of the base pipeline (used by
# PIPELINE_CONFIG in section 4).
REFERENCE_MAX_LAG = 24
REFERENCE_LAG_KNEES = dict(
    knee1=5,
    knee2=10,
    rate_zone1=0.02,
    rate_zone2=0.15,
    rate_zone3=0.50,
)


PATCH_CONFIG_OVERRIDE = {
    'dml_model': 'auto',
    'causal_penalty_fn': ADAPTIVE_CAUSAL_PENALTY,
    # structural_cmi_max_cond: ceiling on the conditioning set |Z| in
    # Stage 8's parent competition (Pipeline/structural.py).
    'structural_cmi_max_cond': 5,
    # dml_auto_nuisance_screening (Stage 6, Pipeline/dml_engine.py::
    # _detect_nuisance_regime): per-dataset, ground-truth-free decision
    # on whether the DML nuisance regression conditions on all other
    # series or only the top-K most correlated with source/target.
    # DISABLED: blanket conditioning on all other series.
    'dml_auto_nuisance_screening': False,
    # contemporaneous_links (Stage 8b of the structural adjustment,
    # Pipeline/contemporaneous.py): adds
    # lag-0 adjacencies (partial correlation of VAR innovations, BH) as
    # unoriented links, only when the past explains little of the series
    # (median innovation R2 <= contemporaneous_max_past_r2 = 0.25).
    'contemporaneous_links': True,
    # contemporaneous_skeleton='pc' (Stage 8b): PC-stable skeleton on the
    # innovations (minimal separating sets, FDR-BH on each pair's max
    # p-value, as PCMCI+ does) instead of conditioning every pair on ALL
    # other innovations, which creates spurious links through colliders.
    # Stage 8b edges are reported unoriented ('both').
    'contemporaneous_skeleton': 'pc',
    'lagged_requires_time_axis': True,
    # structural_spurious_hub_gate (Stage 8, Pipeline/structural.py::
    # _apply_spurious_hub_gate): flags source entities whose out-degree
    # is a statistical outlier (MAD z-score) among the graph's current
    # candidates.
    'structural_spurious_hub_gate': True,
    # rescue_correct_lag_multiplicity='auto' (Pipeline/rescue.py,
    # Pipeline/discovery.py::_detect_time_axis): Sidak-corrects the
    # rescue mechanisms' lag search, gated per dataset by a
    # data-driven check for genuine temporal structure.
    'rescue_correct_lag_multiplicity': 'auto',
    # dml_estimate_selection (Stage 6, Pipeline/dml_engine.py::
    # CascadeConfig.estimate_selection): when several cascade models agree,
    # theta/SE/diagnostics come from the one with the smallest out-of-fold
    # nuisance error product (1-R2_T)(1-R2_O) instead of the first in
    # cascade order -- lets each pair pick a linear or flexible learner from
    # the data. 'cascade_order' takes the first accepted model instead.
    'dml_estimate_selection': 'nuisance_loss',
    # dml_control_set='auto' (Stage 6, Pipeline/dml_engine.py::
    # _parent_controls): the DML conditions each pair on its candidate
    # parents (each at its own lag) and tests the source as a block of lags,
    # instead of on lags 1..window of every series. Falls back to that
    # blanket design, per dataset, when the parent-based controls already
    # explain most of the outcome out-of-fold (median R2 >= 0.5:
    # near-deterministic dynamics, where a conditional test has nothing left
    # to attribute). 'blanket' always uses that blanket design.
    'dml_control_set': 'auto',
    # structural_cmi_block='auto' (Stage 8a, Pipeline/structural.py::
    # _parent_competition_for_target_cmi): block-of-lags test conditioned on
    # the target's own past, on the datasets where Stage 6 ran the
    # parent-based design.
    'structural_cmi_block': 'auto',
    # predictive_competition='auto' (Stage 8a-bis, Pipeline/predictive.py):
    # on the datasets where dml_control_set='auto' found near-deterministic
    # dynamics, the parents are read from a multivariate predictive model of
    # each target (grouped permutation importance). It replaces the Stage
    # 6-8a decision only when the importances split into two populations
    # (two-component mixture preferred, delta BIC > 6); otherwise it abstains.
    'predictive_competition': 'auto',
    # evidence_coherence_guard (Pipeline/coherence.py): when transfer entropy
    # and the DML |t| do not agree on the ranking of the candidates, the
    # Stage 7 fusion and the Stage 8a competition are not applied: the
    # decision falls back to the Stage 6 test (pairs with a significant DML
    # effect are kept) and the output is flagged ('evidence_regime' =
    # 'incoherent'). REPORT those datasets as not refined by Stages 7-8a.
    # See the module docstring for the rationale and references.
    'evidence_coherence_guard': True,
    # evidence_coherence_fallback='auto': exception to the fallback above.
    # In the incoherent regime, when the past explains <= 25% of the series
    # (the Stage 8b gate), the dependence is mostly contemporaneous and the
    # Stage 6 fallback is not applied: the Stage 7/8a decision stands and
    # Stage 8b adds the lag-0 links (Pipeline/pipeline.py,
    # _stage8_regime_guards). 'stage6' applies the fallback always.
    'evidence_coherence_fallback': 'auto',
    # The R2 compared with that gate is the larger of the all-series
    # estimate and each series' own-past estimate, so that a short series,
    # where the all-series regression underestimates the dynamics, is not
    # mistaken for one without dynamics (rationale and references in
    # Pipeline/contemporaneous.py, own_past_explained_variance).
    'evidence_fallback_own_past_floor': True,
}

# LAG-0 REPORTING (fairness). The pipeline reports contemporaneous (lag-0)
# links (Stage 8b, 'contemporaneous_links' above), so every baseline whose
# published method estimates contemporaneous structure reports it too, under
# the same scoring convention:
#   - PCMCI+ / PCMCI+ GPDC (main_gpdc.py): the graph at lag 0 ('-->' in its direction;
#     unoriented 'o-o' and conflicting 'x-x' adjacencies in BOTH directions,
#     exactly like Stage 8b's orientation='both');
#   - DYNOTEARS: the edges of W; VARLiNGAM: the edges of B0 -- both already
#     oriented by the method, each tested with the lag-0 t-test
#     (Auxiliary/causal_common.py::_ngc_contemp_ttest).
# PCMCI (tigramite, and its GPDC variant) and CUTS+ are lagged-only methods
# as published (no contemporaneous step), so they keep reporting lag >= 1.
# So do Granger and Transfer Entropy, which are lagged by definition.
ALL_MODELS = [
    {'name': 'patch',          'pipeline': rp_v2,  'config_override': PATCH_CONFIG_OVERRIDE},
    # 'patch' with the lag sweep / DML control window at tau=10 instead of 24
    # (robustness check: results should not hinge on tau).
    {'name': 'patch_tau10', 'pipeline': rp_v2,
     'config_override': {**PATCH_CONFIG_OVERRIDE, 'max_lag': 10}},
    # 'patch' with Stage 7 replaced by RankMeanSelector (accept when the mean
    # percentile rank over six evidence sources, within the run, exceeds 0.5).
    # Everything else identical to 'patch'. Single shared instance: runs must
    # be sequential (it is recalibrated per dataset).
    {'name': 'patch_rank', 'pipeline': rp_v2,
     'config_override': {**PATCH_CONFIG_OVERRIDE, 'causal_penalty_fn': RankMeanSelector()}},
    {'name': 'patch_rank_tau10', 'pipeline': rp_v2,
     'config_override': {**PATCH_CONFIG_OVERRIDE, 'causal_penalty_fn': RankMeanSelector(),
                         'max_lag': 10}},
    # {'name': 'kernel_ridge',  'pipeline': rp_v2, 'config_override': {'dml_model': 'kernel_ridge'}},
    # {'name': 'bayesian_ridge', 'pipeline': rp_v2, 'config_override': {'dml_model': 'bayesian_ridge'}},
    # tau_max = REFERENCE_MAX_LAG (24) for EVERY baseline in the main run --
    # the same lag window 1..24 as 'patch', instead of each wrapper's own
    # min(max_lag, 10) default.
    #
    # SIGNIFICANCE LEVELS of the PCMCI family (also mirrored in main_gpdc.py).
    # They do NOT inherit alpha_discovery=0.20: for 'patch' that value is only
    # the Stage-1 screening level, whereas in the PCMCI wrappers it would
    # become the final significance level, which the literature does not use.
    # (For reference, the patch's own DML decision is NOT at 0.05: the
    # cascade uses pval_max = 0.10 nominal, Sidak-corrected over its models
    # to ~0.026 per model -- halved again for the flexible ones -- followed
    # by the Stage-7 causal penalty; see Pipeline/dml_engine.py.) Instead,
    # each baseline uses the level of its own published method:
    #   - PCMCI (Runge et al., Sci. Adv. 2019): a fixed alpha for the PC1
    #     parent-selection step, pc_alpha = 0.2 -- the paper's fixed-alpha
    #     variant ("PC1^alpha + MCI", Tab. S1/S2; also tigramite's default)
    #     and its recommended value for GPDC/CMI. For ParCorr the paper's
    #     main variant instead picks alpha by AIC over {0.1, 0.2, 0.3, 0.4};
    #     the fixed value is kept so ParCorr and GPDC share one setting.
    #     Final MCI decision at alpha_level = 0.05 with FDR-BH, which the
    #     paper describes as an option (its own experiments do not apply
    #     it) -- a declared deviation, needed because the MCI step tests
    #     every pair x lag (N^2 * tau_max hypotheses).
    #   - PCMCI+ (Runge, UAI 2020): a single alpha governs skeleton,
    #     orientation and decision; pc_alpha = 0.01, the value of the
    #     paper's main experiments (Fig. 3, ParCorr and GPDC) and
    #     tigramite's default for run_pcmciplus. The decision is the NATIVE
    #     graph (correction='none'): no post-hoc BH on top of it, which the
    #     published method does not do (same principle as the native
    #     outputs of DYNOTEARS / VARLiNGAM / CUTS+). tigramite's own FDR
    #     (fdr_method) is left at its default 'none'.
    # The levels are therefore NOT equal across methods, and are not meant
    # to be: each method keeps its own published decision rule.
    {'name': 'pcmci_tigramite', 'pipeline': run_pipeline_pcmci_tigramite,
     'config_override': {
         'pcmci_tau_max':     REFERENCE_MAX_LAG,
         'pcmci_pc_alpha':    0.2,       # PC1 (liberal parent selection)
         'pcmci_alpha_level': 0.05,      # final MCI decision
         'pcmci_correction':  'fdr_bh',
     }},
    # PCMCI+ (Runge, UAI 2020), from tigramite: run with tau_min=0
    # (contemporaneous conditioning), decision = the native graph edge mark.
    # Lagged AND lag-0 links are reported (see LAG-0 REPORTING above).
    {
        'name': 'pcmciplus',
        'pipeline': run_pipeline_pcmciplus,
        'config_override': {
            'pcmciplus_tau_max':                 REFERENCE_MAX_LAG,  # 24, same window as 'patch'
            'pcmciplus_cond_ind_test':           'parcorr',
            'pcmciplus_pc_alpha':                0.01,    # single alpha; paper's main experiments + tigramite default
            'pcmciplus_correction':              'none',  # native graph (no post-hoc BH)
            'pcmciplus_fdr_method':              'none',  # tigramite's own FDR: off (its default)
            'pcmciplus_include_contemporaneous': True,
        },
    },
    # VARLiNGAM (Hyvarinen et al., JMLR 2010), from lingam: VAR (order chosen
    # by AIC within 1..tau_max, as in the paper's own real-data analysis,
    # Sec. 7: "we determined the order k ... by minimizing the AIC") +
    # DirectLiNGAM on the residuals (lingam's default estimator; the paper
    # used ICA-LiNGAM) + adaptive-Lasso pruning. Decision = the method's
    # NATIVE pruned graph (lagged B_tau and lag-0 B0 edges); the conditioned
    # F-/t-test p-value is still reported, for information only.
    # Requires `pip install lingam`.
    {
        'name': 'varlingam',
        'pipeline': run_pipeline_varlingam,
        'config_override': {
            'varlingam_tau_max':    REFERENCE_MAX_LAG,  # 24; still capped by T/N (dof guard in the wrapper)
            'varlingam_criterion':  'aic',  # paper: VAR order by AIC (lingam's lib default is 'bic')
            'varlingam_prune':      True,
            # Decision rule (see Auxiliary/causal_common.py::_posthoc_gate):
            #   ftest_gate=False -> NATIVE graph, as in the paper;
            #   ftest_gate=True  -> graph AND post-hoc F-test p < alpha,
            #   after correction ('none' | 'bonferroni' | 'fdr_bh').
            'varlingam_ftest_gate': False,  # native: adaptive-Lasso-pruned graph
            'varlingam_correction': 'none',
            'varlingam_include_contemporaneous': True,  # report B0 (lag 0)
        },
    },
    # ── Recent literature methods ──────────────────────────────────
    #
    # Both pipelines below wrap the AUTHORS' OWN numerical core (see each
    # module's docstring for what is vendored and from where). The values
    # set here are therefore the REFERENCE values published by each paper /
    # official config — NOT tuned on this project's benchmarks. Anything
    # not listed keeps the pipeline's own default, which is likewise the
    # reference value. Do not re-tune these without saying so in the
    # paper: the whole point of these two rows is that they reproduce the
    # published methods rather than a local variant of them.
    #
    # DYNOTEARS (Pamfil et al., AISTATS 2020):
    #   Joint (W, A) estimation under the NOTEARS acyclicity constraint
    #   h(W) = tr(e^{W∘W}) - d, augmented Lagrangian + L-BFGS-B, via the
    #   official causalnex optimizer. Decision = the NATIVE thresholded
    #   graph (lagged A and contemporaneous W edges); the conditioned
    #   F-/t-test p-value is still reported, for information only.
    #   lambda_w/lambda_a and the weight thresholds are ABSOLUTE values,
    #   from the paper's benchmark protocol (App. B.3: lambda_W = lambda_A
    #   = 0.05, tau_W = 0.3, tau_A = 0.1). The tau_W = tau_A = 0.01 of
    #   Sec. 4.1 / Fig. 2 belongs to the illustrative example only (the
    #   authors call those low values "intentional").
    {
        'name': 'dynotears',
        'pipeline': run_pipeline_dynotears,
        'config_override': {
            'dynotears_tau_max':      REFERENCE_MAX_LAG,  # 24, same window as 'patch'
            'dynotears_lambda_w':     0.05,   # paper App. B.3 (causalnex's lib default is 0.1)
            'dynotears_lambda_a':     0.05,   # paper App. B.3 (causalnex's lib default is 0.1)
            'dynotears_w_threshold':  0.3,    # tau_W, paper App. B.3 (contemporaneous W)
            'dynotears_a_threshold':  0.1,    # tau_A, paper App. B.3 (lagged A)
            'dynotears_sample_split': False,  # True = avoids double-dipping in the F-test
            'dynotears_include_contemporaneous': True,  # report W (lag 0)
            # Decision rule (see Auxiliary/causal_common.py::_posthoc_gate):
            #   ftest_gate=False -> NATIVE graph, as in the paper;
            #   ftest_gate=True  -> graph AND post-hoc F-test p < alpha,
            #   after correction ('none' | 'bonferroni' | 'fdr_bh').
            'dynotears_ftest_gate':   False,  # native: |W| > tau_W, |A| > tau_A
            'dynotears_correction':   'none',
        },
    },
    # CUTS+ (Cheng et al., AAAI 2024), official architecture:
    #   Alternating Prediction Stage (MPGNN imputation) / Causal Discovery
    #   Stage (Gumbel-softmax-sampled Causal Probability Graph under an L1
    #   sparsity penalty), with Coarse-to-fine Discovery (C2FD) grouping,
    #   Decision = the NATIVE thresholded graph; the post-hoc F-test
    #   p-value is still reported, for information only.
    #   Every hyper-parameter below is the authors' published Lorenz-96
    #   config (CUTS_Plus/opt/lorenz_example.yaml); the ones not listed
    #   here (total_epoch=64, batch_size=128, lambda_s 1e-1 -> 1e-2,
    #   lr_graph 1e-3 -> 1e-4, tau 1 -> 0.1,
    #   n_groups=32 for N >= 64, else no grouping (see the pipeline's
    #   cuts_n_groups_start doc),
    #   group_mul=2, group_every=20, ...) are the pipeline's defaults and
    #   come from that same file.
    #
    {
        'name': 'cuts_plus',
        'pipeline': run_pipeline_cuts_plus,
        'config_override': {
            'cuts_tau_max':          REFERENCE_MAX_LAG,  # 24; post-hoc F-test only
            'cuts_hidden':           32,     # reference config: data_pred.mlp_hid
            'cuts_sample_split':     False,  # True = avoids double-dipping in the F-test
            'cuts_causal_threshold': 0.5,    # official code: raw G @ GT > 0.5
            # Decision rule (see Auxiliary/causal_common.py::_posthoc_gate):
            #   ftest_gate=False -> NATIVE graph, as in the paper;
            #   ftest_gate=True  -> graph AND post-hoc F-test p < alpha,
            #   after correction ('none' | 'bonferroni' | 'fdr_bh').
            'cuts_ftest_gate':       False,  # native: raw G @ GT > 0.5
            'cuts_correction':       'none',
        },
    },
    # ── The pipeline's own ingredients, each on its own ────────────
    #
    # Granger and Transfer Entropy are two of the tests the pipeline combines
    # (Stage 5 rescue and Stage 4). These two rows run each of them ALONE over
    # every ordered pair of series -- no screening, no lag penalty, no DML,
    # no structural stage -- to show what the rest of the pipeline adds.
    # Both are pairwise (they condition on the target's own past only) and
    # decide like the PCMCI row: alpha = 0.05 with FDR-BH over all the pairs
    # tested, not alpha_discovery = 0.20 (see SIGNIFICANCE LEVELS above).
    #
    # Granger (Econometrica 1969): bivariate F-test of the lags 1..p of the
    # source given the lags 1..p of the target, p = order of the pair's
    # bivariate VAR, chosen by AIC within 1..tau_max (same criterion as the
    # VARLiNGAM row; 'bic' is more conservative but misses weak effects at
    # lags >= 3 -- see ModelComparison/pipeline_granger.py).
    {
        'name': 'granger',
        'pipeline': run_pipeline_granger,
        'config_override': {
            'granger_tau_max':     REFERENCE_MAX_LAG,  # 24; still capped by T (dof guard in the wrapper)
            'granger_criterion':   'aic',
            'granger_alpha_level': 0.05,
            'granger_correction':  'fdr_bh',
        },
    },
    # Transfer Entropy (Schreiber, PRL 2000): the pipeline's own estimator
    # (te_method / te_bins of PIPELINE_CONFIG), statistic = max TE over the
    # lags 1..tau_max, p-value by permutation of that maximum.
    # n_perms is the cap of a sequential test (a pair stops at min_exceed
    # exceedances), so only the pairs that look significant pay for it; it
    # sets the smallest attainable p-value, which FDR-BH needs to be low on
    # the larger scenarios (see ModelComparison/pipeline_transfer_entropy.py).
    {
        'name': 'transfer_entropy',
        'pipeline': run_pipeline_transfer_entropy,
        'config_override': {
            'transfer_entropy_tau_max':     REFERENCE_MAX_LAG,  # 24, same window as 'patch'
            'transfer_entropy_alpha_level': 0.05,
            'transfer_entropy_correction':  'fdr_bh',
            'transfer_entropy_n_perms':     5000,
            'transfer_entropy_min_exceed':  10,
        },
    },
]

# tau=10 variant of every baseline ('<name>_tau10'), mirroring
# 'patch' vs 'patch_tau10': identical config except the lag window. Each
# wrapper reads its window from its own key (listed here). max_lag is set to
# the same value in every variant so the window is 10 wherever the wrapper
# reads it.
TAU10_MAX_LAG = 10
_BASELINE_TAU_KEYS = {
    'pcmci_tigramite': 'pcmci_tau_max',
    'pcmciplus':       'pcmciplus_tau_max',
    'varlingam':       'varlingam_tau_max',
    'dynotears':       'dynotears_tau_max',
    'cuts_plus':       'cuts_tau_max',
    'granger':         'granger_tau_max',
    'transfer_entropy': 'transfer_entropy_tau_max',
}


def _tau10_variant(model):
    override = {**model.get('config_override', {}), 'max_lag': TAU10_MAX_LAG}
    tau_key = _BASELINE_TAU_KEYS[model['name']]
    if tau_key is not None:
        override[tau_key] = TAU10_MAX_LAG
    return {**model, 'name': f"{model['name']}_tau10", 'config_override': override}


ALL_MODELS += [_tau10_variant(m) for m in ALL_MODELS if m['name'] in _BASELINE_TAU_KEYS]

# Selection is driven by ACTIVE_MODEL_NAMES, defined at the top of this module.
MODELS = [m for m in ALL_MODELS if m['name'] in ACTIVE_MODEL_NAMES]


# =============================================================================
# 4) BASE PIPELINE CONFIGURATION (PIPELINE_CONFIG)
# =============================================================================
# NOTE: PIPELINE_CONFIG is shared by every model via **cfg. The
# generic-penalty pipelines (patch/random_forest/kernel_ridge/
# bayesian_ridge) use practically every field below; the literature
# baselines (pcmci_tigramite, pcmciplus, varlingam, dynotears,
# cuts_plus, granger, transfer_entropy) actually only use date_col,
# entity_col, property_cols, max_lag and alpha_discovery (plus their own
# prefixed parameters in config_override) — the rest (dml_model,
# lag_penalty_fn, causal_penalty_fn, te_method, n_perms,
# block_same_property, ...) is ignored via **_kwargs.
# pcmci_correction is used by pcmci_tigramite. transfer_entropy also reads
# te_method, te_bins and n_jobs, so that it runs the pipeline's estimator.
PIPELINE_CONFIG = dict(
    dml_model='auto',
    lag_penalty_fn=KneeLagPenalty(**REFERENCE_LAG_KNEES),
    causal_penalty_fn=causal_pen,
    causal_drop_zero=False,
    use_pearson_rescue=True,
    max_lag=REFERENCE_MAX_LAG,
    date_col='Date',
    entity_col='Name',
    property_cols=None,
    alpha_discovery=0.20,
    use_xcorr=True,
    min_corr=0.05,
    min_te=0.005,
    te_method='histogram',
    te_bins=6,
    n_perms=50,
    # LAG multiplicity correction within each pair (max-stat permutation
    # over the up to 24 lags scanned by the argmax) — complements the
    # batch FDR-BH (already active via xcorr_te_correction, default) and
    # accounts for the look-elsewhere effect of the lag argmax.
    # n_perms_lag_correction=200 (not the default 50).
    correct_lag_multiplicity=True,
    n_perms_lag_correction=200,
    dml_splits=3,
    block_same_property=False,
    include_self=True,
    # Up to 20 workers, leaving 2 cores free (at least 1 worker).
    n_jobs=max(1, min(20, (os.cpu_count() or 1) - 2)),
)
