"""
main_pcmci_nobh.py -- runs ONLY PCMCI (tigramite, ParCorr) WITHOUT the
multiple-comparisons correction, at the two lag windows of the main run.
Same spirit as main_gpdc.py / main_heldout.py: a dedicated entry point that
reuses main.py's run_all/validate_all.

What is run
---
  - Models: 'pcmci_tigramite_nobh' (tau=24) and 'pcmci_tigramite_nobh_tau10'
    (tau=10). Each is the corresponding entry of Config.execution.ALL_MODELS
    ('pcmci_tigramite' / 'pcmci_tigramite_tau10') with a single change:
    pcmci_correction = 'none' instead of 'fdr_bh'. The decision is therefore
    the raw MCI p-value < pcmci_alpha_level (0.05), as in the experiments of
    Runge et al. (2019); pc_alpha and tau_max are untouched.
  - Scenarios: the development suite of main.py, loaded with EXACTLY the same
    arguments as main.py's entry point (the USE_* flags and scenario lists
    of Config/execution.py are honored), so the results are directly
    comparable with the BH run (same scenario names, same data, same GT).
    With INCLUDE_HELDOUT, also the held-out datasets of main_heldout.py
    (its ACTIVE_SOURCES and LOAD ARGUMENTS, so the same data as that run);
    they are listed in <OUTPUT_DIR>/evaluation_scope.txt, as main_heldout.py
    does, and must be reported apart from the development suite.
  - Base config: Config.execution.PIPELINE_CONFIG.

To compare against the BH variant, merge the two runs' summaries or point
the validator at both output folders.

Usage
---
No command-line arguments, by design -- edit the constants in the
"EXECUTION CONTROL" section below and run:

    python main_pcmci_nobh.py
"""
from __future__ import annotations

import sys
import time
import datetime

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import main as main_mod  # reuses run_all/validate_all/_format_duration
import main_heldout      # scenarios and evaluation scope of the held-out datasets
from Config import execution as cfg_exec
from Data import (
    generate_tigramite_benchmark_scenarios,
    generate_synthetic_var_benchmark_scenarios,
    generate_lorenz96_benchmark_scenarios,
    generate_logistic_maps_benchmark_scenarios,
    generate_causaltime_benchmark_scenarios,
    load_alarm_benchmark,
    generate_netsim_benchmark_scenarios,
    load_with_fallback,
)
from Auxiliary.consolidate_summaries import consolidate
from Validation.FPChecker import export_fp
from Validation.FNChecker import export_fn


# =============================================================================
# EXECUTION CONTROL -- edit here.
# =============================================================================

# Base models (names from Config.execution.ALL_MODELS): PCMCI at tau=24 and
# tau=10. Remove one to run a single window.
BASE_MODEL_NAMES = ['pcmci_tigramite', 'pcmci_tigramite_tau10']

# 'none' = no multiple-comparisons correction (raw MCI p-value < alpha_level).
PCMCI_CORRECTION = 'none'

# Inserted in the model name, before the '_tau10' suffix, so the output
# folders never collide with those of the BH run.
NAME_TAG = '_nobh'

# Also runs the held-out datasets of main_heldout.py (the sources of its
# ACTIVE_SOURCES). Expensive: PCMCI took ~5 min per scenario and lag window on
# the 29-35 series of PetShop.
INCLUDE_HELDOUT = True

# Optional scenario scope, matched as substrings of the scenario name (e.g.
# ["benchmark_tigramite", "benchmark_lorenz96"]). None = every scenario loaded.
SCENARIO_FILTER = None

RUN_RUNNER     = True
RUN_VALIDATION = True

OUTPUT_DIR = f"./Results_PCMCI_NoBH/Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


# =============================================================================
# MODELS
# =============================================================================

def _tagged_name(name: str) -> str:
    """'pcmci_tigramite_tau10' -> 'pcmci_tigramite_nobh_tau10'."""
    if name.endswith('_tau10'):
        return name[:-len('_tau10')] + NAME_TAG + '_tau10'
    return name + NAME_TAG


def _models() -> list:
    by_name = {m['name']: m for m in cfg_exec.ALL_MODELS}
    missing = [n for n in BASE_MODEL_NAMES if n not in by_name]
    if missing:
        raise KeyError(f"main_pcmci_nobh: models {missing} not found in "
                       f"Config.execution.ALL_MODELS")
    return [
        {**by_name[n], 'name': _tagged_name(n),
         'config_override': {**by_name[n]['config_override'],
                             'pcmci_correction': PCMCI_CORRECTION}}
        for n in BASE_MODEL_NAMES
    ]


# =============================================================================
# SCENARIO LOADING -- same arguments as main.py's / main_heldout.py's entry point.
# =============================================================================

def _load_scenarios() -> tuple:
    """Returns (dataframes, {held-out scenario name: source key})."""
    dataframes: list = []

    if cfg_exec.USE_SYNTHETIC_VAR_BENCHMARK:
        for (name_prefix, seed, n_points, n_entities, prob_edge,
             max_parents, max_lag, min_edges) in cfg_exec.SYNTHETIC_VAR_SCENARIOS:
            dataframes += load_with_fallback(
                name_prefix, generate_synthetic_var_benchmark_scenarios,
                seeds=[seed], n_points=n_points, name_prefix=name_prefix,
                n_entities=n_entities, prob_edge=prob_edge,
                max_parents=max_parents, max_lag=max_lag, min_edges=min_edges,
            )

    if cfg_exec.USE_TIGRAMITE_BENCHMARK:
        dataframes += load_with_fallback(
            "tigramite", generate_tigramite_benchmark_scenarios,
            seeds=[11, 22, 33], n_points=300,
        )

    if cfg_exec.USE_LORENZ96_BENCHMARK:
        dataframes += load_with_fallback(
            "lorenz96", generate_lorenz96_benchmark_scenarios,
            Ks=[8, 10], n_points=300, seeds=[1, 2, 3],
        )

    if cfg_exec.USE_LOGISTIC_BENCHMARK:
        dataframes += load_with_fallback(
            "logistic_maps", generate_logistic_maps_benchmark_scenarios,
            n_points=300, seeds=[1, 2, 3],
        )

    if cfg_exec.USE_CAUSALTIME_BENCHMARK:
        dataframes += load_with_fallback(
            "causaltime", generate_causaltime_benchmark_scenarios,
            domains=["medical", "pm25", "traffic"],
            base_dir=cfg_exec.CAUSALTIME_DIR_PATH,
            mode="separate_samples",
            max_samples=3,
        )

    if cfg_exec.USE_ALARM_BENCHMARK:
        dataframes += load_with_fallback(
            "alarm", load_alarm_benchmark,
            n_samples=500,
        )

    if cfg_exec.USE_NETSIM_BENCHMARK:
        dataframes += load_with_fallback(
            "netsim", generate_netsim_benchmark_scenarios,
            subjects=cfg_exec.NETSIM_SUBJECTS,
        )

    source_of: dict = {}
    if INCLUDE_HELDOUT:
        heldout, source_of = main_heldout._load_scenarios()
        test_names = {d['name'] for d in dataframes}
        dataframes += [d for d in heldout if d['name'] not in test_names]
        source_of = {n: s for n, s in source_of.items() if n not in test_names}

    if SCENARIO_FILTER:
        dataframes = [d for d in dataframes
                      if any(s in d['name'] for s in SCENARIO_FILTER)]
        source_of = {d['name']: source_of[d['name']] for d in dataframes
                     if d['name'] in source_of}
    return dataframes, source_of


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    total_start_time = time.perf_counter()

    models = _models()
    dataframes, source_of = _load_scenarios()
    print(f"\n[main_pcmci_nobh] {len(dataframes)} scenario(s) "
          f"(development suite: {len(dataframes) - len(source_of)}, held-out: {len(source_of)}) "
          f"x {len(models)} model(s) "
          f"({[m['name'] for m in models]}), pcmci_correction='{PCMCI_CORRECTION}' "
          f"-> {OUTPUT_DIR}")

    if not dataframes:
        raise RuntimeError("main_pcmci_nobh: no scenario could be loaded.")

    if source_of:
        main_heldout._save_evaluation_scope(
            OUTPUT_DIR, [d for d in dataframes if d['name'] in source_of], source_of)

    if RUN_RUNNER:
        print("\n" + "=" * 65)
        print("  STEP 1 -- RUNNER (PCMCI, NO BH)")
        print("=" * 65)
        main_mod.run_all(
            dataframes=dataframes,
            models=models,
            config=cfg_exec.PIPELINE_CONFIG,
            output_dir=OUTPUT_DIR,
        )

    if RUN_VALIDATION:
        print("\n" + "=" * 65)
        print("  STEP 2 -- VALIDATION (PCMCI, NO BH)")
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
