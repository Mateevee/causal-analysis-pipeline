"""
main_gpdc.py -- defines and runs the non-linear GPDC baselines, separated
from main.py because GPDC is by far the slowest part of the full suite
(same spirit as main_ablation_patch.py: a dedicated entry point that reuses
main.py's run_all/validate_all). This is the ONLY place where the GPDC
models are defined; main.py / Config.execution.ALL_MODELS do not have them.

What is run
---
  - Models (GPDC_MODELS below): PCMCI-GPDC and PCMCI+-GPDC, each at the two
    lag windows of the main run -- tau=24 (plain name, ==
    Config.execution.REFERENCE_MAX_LAG) and tau=10 ('<name>_tau10', ==
    Config.execution.TAU10_MAX_LAG). PCMCI+-GPDC also reports its lag-0
    (contemporaneous) links, under the same convention as PCMCI+ in
    main.py; PCMCI-GPDC is lagged-only, as published. The GPDC test itself
    is fixed in ModelComparison/pipeline_pcmci_gpdc.py::GPDC_CONFIG.
  - Scenarios: the 36 scenarios of GPDC_SCENARIO_FILTER (tigramite SCMs
    incl. the linear control, Lorenz-96, logistic maps, Walker climate,
    PeMS-BAY, Sachs), loaded with EXACTLY the same arguments as main.py's
    entry point -- and, for the three held-out ones (which main.py does
    not run), as main_heldout.py's -- so the results are directly comparable
    with those runs (same scenario names, same data, same GT). The run
    aborts if the scope does not have EXPECTED_N_SCENARIOS scenarios.
  - Base config: Config.execution.PIPELINE_CONFIG (the GPDC wrappers only
    read date_col/entity_col/property_cols/max_lag/alpha_discovery from it).

To compare against 'patch' and the other baselines, merge the two runs'
summaries or point the validator at both output folders.

Usage
---
No command-line arguments, by design -- edit the constants in the
"EXECUTION CONTROL" section below and run:

    python main_gpdc.py
"""
from __future__ import annotations

import sys
import time
import datetime

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import main as main_mod  # reuses run_all/validate_all/_format_duration
import main_heldout      # load arguments of the held-out scenarios
from Config import execution as cfg_exec
from ModelComparison.pipeline_pcmci_gpdc import run_pipeline_pcmci_gpdc
from ModelComparison.pipeline_pcmciplus_gpdc import run_pipeline_pcmciplus_gpdc
from Data import (
    generate_tigramite_benchmark_scenarios,
    generate_lorenz96_benchmark_scenarios,
    generate_logistic_maps_benchmark_scenarios,
    load_tigramite_climate_benchmark,
    load_sachs_benchmark,
    generate_pems_bay_benchmark_scenarios,
    load_with_fallback,
)
from Auxiliary.consolidate_summaries import consolidate
from Validation.FPChecker import export_fp
from Validation.FNChecker import export_fn


# =============================================================================
# EXECUTION CONTROL -- edit here.
# =============================================================================

# Scope of the GPDC comparison, matched as substrings of the scenario name:
# the non-linear tigramite SCMs plus the linear control family
# (baseline_linear, all seeds -- so the non-linear tests are also checked
# where the truth is linear), Lorenz-96, coupled logistic maps, and the real
# climate, PeMS-BAY and Sachs benchmarks = 15 + 6 + 12 + 1 + 1 + 1 = 36
# scenarios. GPDC is far too slow for the large scenarios (synthetic_var,
# causaltime, tep, alarm), which are left out on purpose. The last three are
# held-out datasets, outside main.py's development suite (see main_heldout.py).
GPDC_SCENARIO_FILTER = [
    "benchmark_tigramite_baseline_linear",
    "benchmark_tigramite_quadratic",
    "benchmark_tigramite_threshold",
    "benchmark_tigramite_saturation",
    "benchmark_tigramite_large_mixed",
    "benchmark_lorenz96",
    "benchmark_logistic_maps",
    "real_tigramite_climate_walker",
    "real_pems_bay_traffic",
    "real_sachs_proteins",
]
EXPECTED_N_SCENARIOS = 36   # None disables the check

# Lag windows: the same two as every model in main.py (24 and 10). The GPDC
# wrappers have no tau_max override: tau_max is always max_lag.
GPDC_WINDOWS = {
    '':       cfg_exec.REFERENCE_MAX_LAG,   # 24
    '_tau10': cfg_exec.TAU10_MAX_LAG,       # 10
}

# Same algorithms and post-processing as pcmci_tigramite / pcmciplus in
# main.py; the only differences are the conditional-independence test (GPDC,
# significance='analytic', fixed GPDC_CONFIG) and the scenario scope.
_GPDC_BASE_MODELS = [
    # Significance levels: same as pcmci_tigramite / pcmciplus (see the
    # SIGNIFICANCE LEVELS note in Config/execution.py).
    {'name': 'pcmci_gpdc', 'pipeline': run_pipeline_pcmci_gpdc,
     'config_override': {'pcmci_gpdc_pc_alpha':    0.2,
                         'pcmci_gpdc_alpha_level': 0.05,
                         'pcmci_gpdc_correction':  'fdr_bh'}},
    {'name': 'pcmciplus_gpdc', 'pipeline': run_pipeline_pcmciplus_gpdc,
     'config_override': {'pcmciplus_gpdc_include_contemporaneous': True,
                         'pcmciplus_gpdc_pc_alpha':    0.01,
                         'pcmciplus_gpdc_correction':  'none',   # native graph
                         'pcmciplus_gpdc_fdr_method':  'none'}},
]

GPDC_MODELS = [
    {**m, 'name': m['name'] + suffix,
     'dataset_filter': GPDC_SCENARIO_FILTER,
     'config_override': {**m['config_override'], 'max_lag': lag}}
    for m in _GPDC_BASE_MODELS
    for suffix, lag in GPDC_WINDOWS.items()
]

# Models run by this script (names from GPDC_MODELS).
GPDC_MODEL_NAMES = ['pcmci_gpdc', 'pcmci_gpdc_tau10',
                    'pcmciplus_gpdc', 'pcmciplus_gpdc_tau10']

# Providers loaded. The USE_* flags of Config/execution.py are still
# honored on top of the first three (a provider disabled there stays
# disabled here); the held-out ones have no flag there.
USE_TIGRAMITE    = True
USE_LORENZ96     = True
USE_LOGISTIC     = True
USE_REAL_CLIMATE = True
USE_PEMS_BAY     = True
USE_SACHS        = True

RUN_RUNNER     = True
RUN_VALIDATION = True

OUTPUT_DIR = f"./Results_GPDC/Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


# =============================================================================
# SCENARIO LOADING -- same arguments as main.py's / main_heldout.py's entry point.
# =============================================================================

def _load_scenarios() -> list:
    dataframes: list = []

    if USE_TIGRAMITE and cfg_exec.USE_TIGRAMITE_BENCHMARK:
        dataframes += load_with_fallback(
            "tigramite", generate_tigramite_benchmark_scenarios,
            seeds=[11, 22, 33], n_points=300,
        )

    if USE_LORENZ96 and cfg_exec.USE_LORENZ96_BENCHMARK:
        dataframes += load_with_fallback(
            "lorenz96", generate_lorenz96_benchmark_scenarios,
            Ks=[8, 10], n_points=300, seeds=[1, 2, 3],
        )

    if USE_LOGISTIC and cfg_exec.USE_LOGISTIC_BENCHMARK:
        dataframes += load_with_fallback(
            "logistic_maps", generate_logistic_maps_benchmark_scenarios,
            n_points=300, seeds=[1, 2, 3],
        )

    if USE_REAL_CLIMATE:
        dataframes += load_with_fallback(
            "real_climate", load_tigramite_climate_benchmark,
            n_years=main_heldout.CLIMATE_N_YEARS,
            cache_dir=main_heldout.CLIMATE_CACHE_DIR,
        )

    if USE_SACHS:
        dataframes += load_with_fallback(
            "sachs", load_sachs_benchmark, main_heldout.SACHS_CSV_PATH,
            max_rows=main_heldout.SACHS_MAX_ROWS,
        )

    if USE_PEMS_BAY:
        dataframes += load_with_fallback(
            "pems_bay", generate_pems_bay_benchmark_scenarios,
            n_sensors=main_heldout.PEMS_BAY_N_SENSORS,
            n_points=main_heldout.PEMS_BAY_N_POINTS,
        )

    # Keeps only the scenarios in the non-linear scope, so the output folder
    # has no empty "skipped (outside dataset_filter)" scenarios.
    in_scope = [d for d in dataframes
                if any(s in d['name'] for s in GPDC_SCENARIO_FILTER)]
    dropped = sorted(d['name'] for d in dataframes if d not in in_scope)
    if dropped:
        print(f"\n  [main_gpdc] {len(dropped)} scenario(s) outside "
              f"GPDC_SCENARIO_FILTER ignored: {dropped}")
    if EXPECTED_N_SCENARIOS is not None and len(in_scope) != EXPECTED_N_SCENARIOS:
        raise RuntimeError(
            f"main_gpdc: {len(in_scope)} scenario(s) in scope, expected "
            f"{EXPECTED_N_SCENARIOS}: {sorted(d['name'] for d in in_scope)}")
    return in_scope


def _gpdc_models() -> list:
    by_name = {m['name']: m for m in GPDC_MODELS}
    missing = [n for n in GPDC_MODEL_NAMES if n not in by_name]
    if missing:
        raise KeyError(f"main_gpdc: models {missing} not found in GPDC_MODELS")
    return [by_name[n] for n in GPDC_MODEL_NAMES]


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    total_start_time = time.perf_counter()

    models = _gpdc_models()
    dataframes = _load_scenarios()
    print(f"\n[main_gpdc] {len(dataframes)} scenario(s) x {len(models)} model(s) "
          f"({[m['name'] for m in models]}) -> {OUTPUT_DIR}")

    if RUN_RUNNER:
        print("\n" + "=" * 65)
        print("  STEP 1 -- RUNNER (GPDC)")
        print("=" * 65)
        main_mod.run_all(
            dataframes=dataframes,
            models=models,
            config=cfg_exec.PIPELINE_CONFIG,
            output_dir=OUTPUT_DIR,
        )

    if RUN_VALIDATION:
        print("\n" + "=" * 65)
        print("  STEP 2 -- VALIDATION (GPDC)")
        print("=" * 65)
        main_mod.validate_all(
            output_dir=OUTPUT_DIR,
            show_pairs=cfg_exec.SHOW_PAIRS,
            soft_threshold=cfg_exec.SOFT_THRESHOLD,
            ignore_intra_entity=not cfg_exec.DETECT_INTRA_ENTITY,
            use_canonical_pair=cfg_exec.USE_CANONICAL_PAIR,
        )

    print("Exporting summary")
    consolidate(OUTPUT_DIR)

    print("Checking fp")
    export_fp(OUTPUT_DIR)

    print("Checking fn")
    export_fn(OUTPUT_DIR)

    total_time_fmt = main_mod._format_duration(time.perf_counter() - total_start_time)
    print("\n" + "#" * 65)
    print(f"  TOTAL SCRIPT EXECUTION TIME: {total_time_fmt}")
    print("#" * 65)


if __name__ == '__main__':
    main()
