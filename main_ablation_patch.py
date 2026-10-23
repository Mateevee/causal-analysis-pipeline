"""
main_ablation_patch.py -- Additive ablation of the proposed pipeline
('patch'), running ONLY that model (no baselines), over the scenarios
listed in ABLATION_SCENARIOS (one or more, run in sequence).

Reimplements the additive ablation protocol described in the paper:

    1. Isolated DML (single model, no cascade)
    2. + active property filter (Stage 1)
    3. + lag penalty (KneeLagPenalty) -- step 1's config
       (filter + DML(BR), nothing else) already covers the "no lag
       penalty" baseline, so no separate step is needed for it
    4. + auxiliary model cascade (BR->RF->KRR->NW)
    5. + causal effect penalty (Stage 7, Dempster-Shafer fusion), isolated
       -- WITHOUT structural optimization
    6. + structural optimization (the COMPLETE Stage 8), isolated --
       WITHOUT the causal penalty from step 5: 8a (parent competition /
       pruning, Pipeline/structural.py) + 8b (contemporaneous lag-0 links,
       Pipeline/contemporaneous.py, PC-stable skeleton) + the time-axis
       rule (lagged edges dropped when the data has no genuine time axis)
    7. full pipeline: ALL components active at the same time -- property
       filter + lag penalty + cascade + causal penalty + structural
       optimization (reproduces the production PIPELINE_CONFIG,
       Config/execution.py)

followed by three SUBTRACTIVE steps, each starting from step 7 and
removing ONE of the newer components (the difference to step 7 is that
component's contribution with everything else on -- these components only
act together with the others, e.g. the coherence guard suspends the
rejections of Stages 7 and 8a, so testing them additively makes no sense):

    8. full minus the parent-based control set (dml_control_set='blanket';
       with no dataset flagged as near-deterministic, Stage 8a-bis and the
       Stage 8a block test stop firing too -- the step measures all three)
    9. full minus the predictive competition (Stage 8a-bis)
   10. full minus the evidence coherence guard

The lag penalty (KneeLagPenalty) appears in exactly TWO steps: 3 (where it
is the only new component, demonstrating its isolated effect) and 7 (full
pipeline, where everything is on). Steps 4, 5 and 6 go back to
`use_lag_penalty=False` (UniformLagPenalty/pure argmax) and each one adds
exactly ONE component on top of step 4 (auxiliary cascade), in isolation --
step 5 turns on the causal penalty (but not structural optimization), step
6 turns on structural optimization (but not the causal penalty) -- so that
the effect of each one can be read without mixing with the other's. That is
why step 6 is NOT identical to step 7: step 6 isolates only structural
optimization (without lag or causal penalty), while step 7 is the only one
that combines everything -- it, not step 6, is the one that actually
represents the "full pipeline".

Stages 3-5 (cross-correlation + Transfer Entropy + Granger/Pearson rescue)
stay ALWAYS on in every step: they are not one of the ablation axes listed
in the paper -- they are the mechanism that supplies the
candidate pairs to the DML, without which there would be nothing for the
"isolated DML" to estimate.

The protocol runs over the synthetic-VAR generator (Data/synthetic_var.py)
or any other provider registered in Data/base.py.

Reuses the same execution/validation infrastructure as main.py (run_all +
validate_all): each additive step is treated as a distinct "model" that
points to the same pipeline (Pipeline.pipeline.run_pipeline), each with a
different config_override -- the same mechanism Config/execution.py already
uses to compare models/baselines.

Usage
---
No command-line arguments, by design -- edit the constants in the
"EXECUTION CONTROL" section below (same spirit as Config/execution.py) and
run:

    python main_ablation_patch.py

This runs the complete protocol (10 steps) for EACH entry of
ABLATION_SCENARIOS and EACH window in MAX_LAGS, in sequence, each with its
own output folder at OUTPUT_DIR/tau<L>/<scenario_name>/.

To just list the scenarios available from the providers referenced in
ABLATION_SCENARIOS without running anything, set LIST_SCENARIOS = True
below.
"""
from __future__ import annotations

import os
import sys

# Forces UTF-8 on stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import main as main_mod  # reuses run_all/validate_all/_discover_datasets/...
from Config import execution as cfg_exec
from Data import base as data_base
from Pipeline.pipeline import run_pipeline as rp_v2
from Pipeline.penalties import KneeLagPenalty
from Pipeline.SelectorFunc import causal_pen

# Importing each provider populates Data.base's registry via @register(...).
# Each import is protected individually: a provider with a dependency
# missing in this environment (e.g. tigramite not installed) must not
# prevent the use of the others.
for _mod_name in ("synthetic_var", "tigramite_scm", "lorenz96", "logistic_maps",
                   "causaltime", "real_climate"):
    try:
        __import__(f"Data.{_mod_name}")
    except Exception as e:
        print(f"  [WARNING] Data.{_mod_name} could not be imported ({e}); "
              f"the corresponding provider will be unavailable.")


# =============================================================================
# EXECUTION CONTROL -- edit here (same spirit as Config/execution.py).
# =============================================================================

# ABLATION_SCENARIOS: list of base scenarios over which the ablation
# protocol (10 steps) runs, one at a time, in sequence -- no single scenario
# adequately covers the two regimes the paper needs to demonstrate (dense
# linear DGP vs. mixed/nonlinear DGP) with enough statistical resolution to
# support "incremental contribution per stage".
#
# Candidates with too few causal pairs give the metric too little
# resolution for the main table (e.g. 'benchmark_tigramite_quadratic_seed11'
# and 'benchmark_tigramite_baseline_linear_seed11' have only 2 causal pairs
# out of 4 possible, so the metric can only take {0, 0.5, 1.0}). The 2
# chosen below have adequate resolution and cover complementary DGPs:
#
#   - benchmark_synthetic_var_3 (provider 'synthetic_var'): dense linear
#     DGP (12 entities), the best statistical resolution available.
#     correct_lag_multiplicity=True, matching the production default
#     (enabled across every evaluation surface in the paper).
#   - benchmark_tigramite_large_mixed_seed11 (provider 'tigramite'):
#     mixing linear and nonlinear relationships -- the densest scenario of
#     the tigramite family, hence
#     correct_lag_multiplicity=True here (matches the paper's actual
#     protocol for that family).
ABLATION_SCENARIOS = [
    dict(provider='synthetic_var', scenario='benchmark_synthetic_var_3_seed33',
         correct_lag_multiplicity=True),
    dict(provider='tigramite', scenario='benchmark_tigramite_large_mixed_seed11',
         correct_lag_multiplicity=True),
    dict(provider='lorenz96', scenario='benchmark_lorenz96_K10_seed1',
         correct_lag_multiplicity=True),
    # Incoherent-evidence regime: transfer entropy and the DML |t| disagree
    # on the ranking of the candidates, so the evidence coherence guard
    # fires (step 10 removes it). provider_kwargs restricts the provider to
    # the one sample needed (same data path as main.py).
    dict(provider='causaltime', scenario='benchmark_causaltime_medical_sample0',
         correct_lag_multiplicity=True,
         provider_kwargs=dict(domains=['medical'], max_samples=1,
                              base_dir=cfg_exec.CAUSALTIME_DIR_PATH)),
]

# If True, for each provider referenced in ABLATION_SCENARIOS, only
# generate the scenarios and print the available names/causal pairs to the
# console -- do not run any ablation. Use it to find out the exact name
# before editing ABLATION_SCENARIOS above.
LIST_SCENARIOS = False

# Seeds/n_points to generate before filtering by scenario. Ignored for
# provider='synthetic_var', which always uses
# Config.execution.SYNTHETIC_VAR_SCENARIOS (each scenario has its own
# size/density, one tuple per scenario).
SEEDS = None
N_POINTS = None

# Tau_max windows evaluated, one full ablation per window (the same two
# windows as the main comparison: 24 = REFERENCE_MAX_LAG, 10 = the
# '<model>_tau10' robustness check). Each window is written to its own
# folder, OUTPUT_DIR/tau<L>/<scenario>/, and validated separately. Leave a
# single value (e.g. [24]) to run only that window.
MAX_LAGS = [24, 10]

# Treats A->B and B->A as the SAME pair in validation. False (default) =
# directional -- correct for generic benchmarks with no fixed entity role;
# only use True for scenarios with fixed,
# disjoint entity roles.
CANONICAL_PAIR = False

OUTPUT_DIR = './Results_Ablation'


# =============================================================================
# BASE CONFIG -- shared by all the ablation steps. Only the fields touched
# by the additive protocol (property scanner, lag penalty, dml_model,
# causal_penalty_fn, structural filter) change between steps; the rest
# (Stages 3-5, DML splits, etc.) follows the same values as
# Config/execution.py.
# =============================================================================

def _build_base_config(max_lag: int, correct_lag_multiplicity: bool) -> dict:
    return dict(
        date_col='Date', entity_col='Name', property_cols=None,
        max_lag=max_lag,
        alpha_discovery=0.20,
        use_xcorr=True, min_corr=0.05,
        min_te=0.005, te_method='histogram', te_bins=6, n_perms=50,
        correct_lag_multiplicity=correct_lag_multiplicity,
        n_perms_lag_correction=200 if correct_lag_multiplicity else None,
        use_granger_rescue=True, use_pearson_rescue=True,
        dml_splits=3,
        # Generic benchmarks use entities with no fixed role, so pairs
        # sharing the same property are not blocked.
        block_same_property=False,
        include_self=True,
        # Up to 20 workers, leaving 2 cores free (at least 1 worker).
        n_jobs=max(1, min(20, (os.cpu_count() or 1) - 2)),
    )


_KNEE_LAG = KneeLagPenalty(
    knee1=5, knee2=10, rate_zone1=0.02, rate_zone2=0.15, rate_zone3=0.50,
)


def _build_ablation_steps() -> list:
    """Builds the list of 'models' (one per additive step) for run_all()."""
    return [
        {
            'name': '1_isolated_dml',
            'pipeline': rp_v2,
            'config_override': {
                'use_property_scanner': False,
                'use_lag_penalty': False,        # -> UniformLagPenalty (pure argmax)
                'dml_model': 'bayesian_ridge',   # single model, no cascade
                'causal_penalty_fn': None,       # Stage 7 off
                'use_structural_filter': False,  # Stage 8 off
            },
        },
        {
            'name': '2_plus_property_filter',
            'pipeline': rp_v2,
            'config_override': {
                'use_property_scanner': True,
                'use_lag_penalty': False,
                'dml_model': 'bayesian_ridge',
                'causal_penalty_fn': None,
                'use_structural_filter': False,
            },
        },
        {
            # Sole representative of "filter + DML(BR) + lag penalty",
            # cumulative over step 2 -- step 1 already covers the "no lag
            # penalty" baseline (see the module docstring), so no separate
            # step is needed just to isolate the property filter alone.
            'name': '3_prefilter_with_lag_penalty',
            'pipeline': rp_v2,
            'config_override': {
                'use_property_scanner': True,
                'lag_penalty_fn': _KNEE_LAG,
                'dml_model': 'bayesian_ridge',
                'causal_penalty_fn': None,
                'use_structural_filter': False,
            },
        },
        {
            'name': '4_plus_auxiliary_cascade',
            'pipeline': rp_v2,
            'config_override': {
                'use_property_scanner': True,
                'use_lag_penalty': False,
                'dml_model': 'auto',
                'causal_penalty_fn': None,
                'use_structural_filter': False,
            },
        },
        {
            'name': '5_plus_causal_penalty',
            'pipeline': rp_v2,
            'config_override': {
                'use_property_scanner': True,
                'use_lag_penalty': False,
                'dml_model': 'bayesian_ridge',
                'causal_penalty_fn': causal_pen,
                'use_structural_filter': False,
            },
        },
        {
            'name': '6_plus_structural_optimization',
            'pipeline': rp_v2,
            'config_override': {
                'use_property_scanner': True,
                'use_lag_penalty': False,
                'dml_model': 'bayesian_ridge',
                'causal_penalty_fn': None,
                # complete Stage 8 = 8a + 8b + time-axis rule, with the same
                # values as production (Config/execution.py), so step 6
                # isolates the whole structural stage, not only 8a
                'use_structural_filter': True,
                'contemporaneous_links': True,
                'contemporaneous_skeleton': 'pc',
                'lagged_requires_time_axis': True,
            },
        },
        {
            # NOT identical to step 6 (see the note at the top of the
            # file): it is the ONLY step with ALL components active at the
            # same time -- property filter, lag penalty (KneeLagPenalty,
            # the only other place in the table besides step 3), cascade,
            # causal penalty AND structural optimization -- reproducing the
            # production PIPELINE_CONFIG (Config/execution.py). Steps 4-6
            # keep testing the cascade/causal penalty/structural
            # optimization in isolation (no lag penalty, not combined with
            # each other) so that each one's marginal effect can be read
            # without mixing with the others'; only here do they appear
            # together.
            #
            # This step inherits the production 'patch' model's
            # config_override directly from Config/execution.py (rather
            # than repeating AdaptiveCausalPenalty, structural_cmi_max_cond,
            # dml_auto_nuisance_screening, structural_spurious_hub_gate,
            # rescue_correct_lag_multiplicity by hand), so that step 7 can
            # never silently diverge from production; the two fields below
            # (property scanner, lag penalty) are specific to this ablation
            # protocol, not part of the patch's config_override (they are
            # run_pipeline's own defaults), so they still need to stay
            # explicit here.
            'name': '7_full_pipeline',
            'pipeline': rp_v2,
            'config_override': {
                'use_property_scanner': True,
                'lag_penalty_fn': _KNEE_LAG,
                'use_structural_filter': True,
                **next(m for m in cfg_exec.ALL_MODELS if m['name'] == 'patch')['config_override'],
            },
        },
        # Subtractive steps 8-10: same config as step 7 with ONE of the new
        # components removed. The override comes after the production
        # config_override so that it prevails over it.
        {
            # With the blanket control set no dataset is flagged as
            # near-deterministic, so Stage 8a-bis and the Stage 8a block
            # test ('auto') stop firing too: this step measures the three
            # together.
            'name': '8_full_minus_control_set',
            'pipeline': rp_v2,
            'config_override': {
                'use_property_scanner': True,
                'lag_penalty_fn': _KNEE_LAG,
                'use_structural_filter': True,
                **next(m for m in cfg_exec.ALL_MODELS if m['name'] == 'patch')['config_override'],
                'dml_control_set': 'blanket',
            },
        },
        {
            'name': '9_full_minus_predictive',
            'pipeline': rp_v2,
            'config_override': {
                'use_property_scanner': True,
                'lag_penalty_fn': _KNEE_LAG,
                'use_structural_filter': True,
                **next(m for m in cfg_exec.ALL_MODELS if m['name'] == 'patch')['config_override'],
                'predictive_competition': False,
            },
        },
        {
            'name': '10_full_minus_coherence_guard',
            'pipeline': rp_v2,
            'config_override': {
                'use_property_scanner': True,
                'lag_penalty_fn': _KNEE_LAG,
                'use_structural_filter': True,
                **next(m for m in cfg_exec.ALL_MODELS if m['name'] == 'patch')['config_override'],
                'evidence_coherence_guard': False,
            },
        },
    ]


# =============================================================================
# SCENARIO LOADING
# =============================================================================

def _load_synthetic_var_scenarios(target_scenario_name: str = None) -> list:
    """Special case: each tuple of Config.execution.SYNTHETIC_VAR_SCENARIOS
    requires an isolated call to the provider (its own size/density per
    scenario) -- replicates exactly the loop main.py runs in the entry
    point."""
    result = []
    for (name_prefix, seed, n_points, n_entities, prob_edge,
         max_parents, max_lag, min_edges) in cfg_exec.SYNTHETIC_VAR_SCENARIOS:
        if target_scenario_name is not None and not target_scenario_name.startswith(name_prefix):
            continue
        provider = data_base.get_provider('synthetic_var')
        result += provider(
            seeds=[seed], n_points=n_points, name_prefix=name_prefix,
            n_entities=n_entities, prob_edge=prob_edge,
            max_parents=max_parents, max_lag=max_lag, min_edges=min_edges,
        )
    return result


def _load_scenarios(provider_name: str, seeds, n_points,
                    provider_kwargs: dict = None) -> list:
    if provider_name == 'synthetic_var':
        return _load_synthetic_var_scenarios()
    provider = data_base.get_provider(provider_name)
    kwargs = dict(provider_kwargs or {})
    if seeds is not None:
        kwargs['seeds'] = seeds
    if n_points is not None:
        kwargs['n_points'] = n_points
    return provider(**kwargs)


def _select_scenario(scenarios: list, scenario_name: str) -> dict:
    for c in scenarios:
        if c['name'] == scenario_name:
            return c
    available = sorted(c['name'] for c in scenarios)
    raise KeyError(
        f"Scenario '{scenario_name}' not found. Available with these "
        f"parameters ({len(available)}): {available}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    if LIST_SCENARIOS:
        # First entry per provider supplies its provider_kwargs.
        unique_providers = {}
        for item in ABLATION_SCENARIOS:
            unique_providers.setdefault(item['provider'], item.get('provider_kwargs'))
        for provider_name, provider_kwargs in sorted(unique_providers.items()):
            scenarios = _load_scenarios(provider_name, SEEDS, N_POINTS, provider_kwargs)
            print(f"\nScenarios available for provider = '{provider_name}':")
            for c in sorted(scenarios, key=lambda c: c['name']):
                n_causal = int(c['gt']['causal_gt'].sum()) if c.get('gt') is not None else '?'
                print(f"  {c['name']}  ({n_causal} causal pairs)")
        print("\nAdjust ABLATION_SCENARIOS with the desired names and run again "
              "(with LIST_SCENARIOS = False).")
        return

    ablation_steps = _build_ablation_steps()

    for item in ABLATION_SCENARIOS:
        scenarios = _load_scenarios(item['provider'], SEEDS, N_POINTS,
                                    item.get('provider_kwargs'))
        entry = _select_scenario(scenarios, item['scenario'])
        n_causal = int(entry['gt']['causal_gt'].sum()) if entry.get('gt') is not None else '?'
        print(f"\n[ablation] scenario loaded: {entry['name']} "
              f"({len(entry['df'])} lines, {n_causal} causal pairs)")

        for max_lag in MAX_LAGS:
            print(f"\n[ablation] {entry['name']} -- tau_max = {max_lag}")
            base_config = _build_base_config(max_lag, item['correct_lag_multiplicity'])

            output_dir = os.path.join(OUTPUT_DIR, f"tau{max_lag}", entry['name'])
            # Cleans the scenario's folder BEFORE running, so that
            # validate_all() only reads the folders of the steps defined in
            # _build_ablation_steps().
            if os.path.isdir(output_dir):
                import shutil
                shutil.rmtree(output_dir)
                print(f"  [cleanup] previous output folder removed: {output_dir}/")

            main_mod.run_all(
                dataframes=[entry],
                models=ablation_steps,
                config=base_config,
                output_dir=output_dir,
            )
            main_mod.validate_all(
                output_dir, show_pairs=True,
                ignore_intra_entity=not cfg_exec.DETECT_INTRA_ENTITY,
                use_canonical_pair=CANONICAL_PAIR,
            )


if __name__ == '__main__':
    main()
