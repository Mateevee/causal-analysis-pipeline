"""
main_heldout.py -- runs the HELD-OUT datasets, separated from main.py (same
spirit as main_gpdc.py / main_ablation_patch.py: a dedicated entry point that
reuses main.py's run_all/validate_all). These providers are loaded here and
NOT by main.py, the main run.

Evaluation scope
---
These datasets are evaluated OUTSIDE the development suite. The development
suite of this project is the benchmark suite run by main.py (the scenarios
marked D in Table 2 of the paper); none of the scenarios below belongs to it
and none took part in the development of the pipeline. They are run here
with the same models and the same configuration as the development suite
(Config.execution -- nothing is re-tuned for them), and their results are
reported separately: they are never pooled into the development-suite
aggregates (W/T/L, mean F1).

Provenance of each source (SOURCES below carries the same information, and it
is written to <OUTPUT_DIR>/evaluation_scope.txt; see each Data/<source>.py
docstring for the full caveats on its ground truth)
---
  sachs           MEASURED. Multiparameter flow cytometry of human primary T
                  cells (Sachs et al. 2005, Science), 11 phospho-proteins/
                  lipids, one row per cell. GT: the consensus signalling
                  network (literature + interventions), not estimated. No
                  time axis.
  pems_bay        MEASURED. Caltrans PeMS loop-detector speeds, Bay Area,
                  5-min resolution (Li et al. 2018). GT: the road-network
                  connectivity published with the dataset -- a
                  physical-topology proxy, not a causal graph validated by
                  intervention.
  real_climate    REANALYSIS. Walker Circulation, equatorial Pacific (Runge
                  et al., tigramite climate case study): ERA5 monthly series
                  from 1950 -- the real historical atmosphere, estimated by
                  assimilating observations into a forecast model, not raw
                  measurements (the 700 hPa vertical velocity of CPAC/WPAC is
                  a model-derived quantity). GT: the chain CPAC -> WCPAC ->
                  WPAC from physical domain knowledge, mostly
                  contemporaneous.
  petshop         MEASURED. PetShop microservice application running on AWS
                  (Hardt et al. 2024): average latency of each component,
                  5-min aggregates, normal operation, one scenario per
                  traffic regime (three load profiles of the same
                  application). GT: the reversed call graph (callee ->
                  caller) -- a service-dependency proxy the authors
                  themselves do not present as the causal graph of the
                  metrics. Relations are essentially contemporaneous at this
                  sampling.
  tep             SIMULATED, not measured. Tennessee Eastman Process (Downs &
                  Vogel 1993): the output of a simulator of a real Eastman
                  Chemical plant, as re-elaborated by CIPCaD-Bench (whose
                  abstract calls it "the well-known Tennessee Eastman
                  simulator"; the measured dataset of that paper is the other
                  one, the ultra-processed food plant). GT: P&ID topology +
                  control loops. It must be reported as a simulated process,
                  not as real-world data.
  beeline_grn     SIMULATED. BEELINE curated gene networks, BoolODE (networks
                  HSC and VSC only).
  causal_chamber  MEASURED. Causal Chambers (light / wind tunnel).
  dream3          SIMULATED. DREAM3 in-silico gene networks.
  easyvista_it    MEASURED. EasyVista IT monitoring.
  rcaeval         MEASURED. RCAEval microservice fault injection; approximate
                  (fan-out) GT.

What is run
---
  - Models (MODEL_NAMES below): 'patch' and every baseline of
    Config.execution.ALL_MODELS, each at the two lag windows of the main run
    -- tau=24 (plain name) and tau=10 ('<name>_tau10'). The GPDC baselines
    are added only when INCLUDE_GPDC is True (very slow on these sizes);
    main_gpdc.py runs them on Sachs, PeMS-BAY and the Walker climate.
  - Scenarios: the sources of ACTIVE_SOURCES, loaded with the arguments of
    LOAD ARGUMENTS below (main_gpdc.py reads the same constants, so both runs
    use the same data).
  - Base config: Config.execution.PIPELINE_CONFIG.

Usage
---
No command-line arguments, by design -- edit the constants in the
"EXECUTION CONTROL" section below and run:

    python main_heldout.py
"""
from __future__ import annotations

import os
import sys
import time
import datetime

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import main as main_mod  # reuses run_all/validate_all/_format_duration
from Config import execution as cfg_exec
from Data import (
    generate_tep_benchmark_scenarios,
    load_sachs_benchmark,
    generate_pems_bay_benchmark_scenarios,
    load_tigramite_climate_benchmark,
    generate_petshop_benchmark_scenarios,
    generate_beeline_grn_benchmark_scenarios,
    generate_causal_chamber_benchmark_scenarios,
    generate_dream3_benchmark_scenarios,
    generate_easyvista_it_benchmark_scenarios,
    generate_rcaeval_benchmark_scenarios,
    load_with_fallback,
)
from Auxiliary.consolidate_summaries import consolidate
from Validation.FPChecker import export_fp
from Validation.FNChecker import export_fn


# =============================================================================
# EXECUTION CONTROL -- edit here.
# =============================================================================

# Sources loaded by this run. Comment a line out to skip that source.
ACTIVE_SOURCES = [
    "sachs",           # NO real time axis (see Data/sachs.py)
    "pems_bay",        # ~85MB download on first run
    "real_climate",    # mostly contemporaneous relations
    # Expensive for the baselines: 29-35 series (cuts_plus took 10-15 min per
    # scenario and lag window, pcmci_tigramite ~5 min); the 1500-point regime
    # is the heaviest. See PETSHOP_REGIMES below.
    "petshop",
    # Expensive: 31 variables x 1499 points (patch and pcmciplus take ~11 min
    # each, vs. seconds to ~2 min for the other scenarios).
    "tep",
    "beeline_grn",
    "causal_chamber",
    "dream3",
    "easyvista_it",
    "rcaeval",
]

# Models run by this script (names from Config.execution.ALL_MODELS): 'patch'
# and all the baselines, each at tau=24 (plain name) and tau=10.
MODEL_NAMES = [
    'patch',           'patch_tau10',
    'pcmci_tigramite', 'pcmci_tigramite_tau10',
    'pcmciplus',       'pcmciplus_tau10',
    'varlingam',       'varlingam_tau10',
    'dynotears',       'dynotears_tau10',
    'cuts_plus',       'cuts_plus_tau10',
    'granger',         'granger_tau10',
    'transfer_entropy', 'transfer_entropy_tau10',
]

# Adds the GPDC baselines of main_gpdc.py (PCMCI-GPDC and PCMCI+-GPDC, tau=24
# and tau=10), without their scenario filter. Off by default: GPDC is far too
# slow for the larger scenarios here (15-35 series).
INCLUDE_GPDC = False

# Source key -> (data origin, description). 'measured' = recorded from a
# physical system; 'reanalysis' = observations of a physical system
# assimilated into a model; 'simulated' = output of a simulator.
SOURCES = {
    "sachs": (
        "measured",
        "flow cytometry of human primary T cells (Sachs et al. 2005); GT = "
        "consensus signalling network (literature + interventions); no time "
        "axis"),
    "pems_bay": (
        "measured",
        "Caltrans PeMS loop-detector speeds, Bay Area (Li et al. 2018); GT = "
        "published road-network connectivity (physical-topology proxy)"),
    "real_climate": (
        "reanalysis",
        "Walker Circulation, equatorial Pacific, ERA5 monthly series (Runge "
        "et al., tigramite case study); GT = chain CPAC->WCPAC->WPAC "
        "(physical domain knowledge), mostly contemporaneous"),
    "petshop": (
        "measured",
        "PetShop microservice application on AWS (Hardt et al. 2024), "
        "average latency per component, normal operation; GT = reversed "
        "call graph (service dependency topology, a proxy)"),
    "tep": (
        "simulated",
        "Tennessee Eastman Process simulator (Downs & Vogel 1993, via "
        "CIPCaD-Bench); GT = P&ID topology + control loops; NOT measured "
        "data"),
    "beeline_grn": (
        "simulated",
        "BEELINE curated gene regulatory networks (Pratapa et al. 2020), "
        "expression simulated with BoolODE, cells sorted by pseudotime; GT = "
        "published boolean rules"),
    "causal_chamber": (
        "measured",
        "Causal Chambers, light tunnel and wind tunnel (Gamella et al. "
        "2025); GT = the chambers' known causal graph"),
    "dream3": (
        "simulated",
        "DREAM3 in-silico gene networks (E. coli / yeast sub-networks); GT "
        "= the network used by the simulator"),
    "easyvista_it": (
        "measured",
        "EasyVista IT monitoring data (Ait-Bachir et al. 2023); GT = "
        "summary causal graph described by the system experts (not "
        "validated by intervention)"),
    "rcaeval": (
        "measured",
        "RCAEval RE1 microservice telemetry under injected faults (Pham et "
        "al. 2025); GT = fan-out from the faulty service (root validated by "
        "injection, pairwise edges approximate)"),
}

# LOAD ARGUMENTS of each provider. Anything not listed here keeps the
# provider's own default.
SACHS_CSV_PATH          = "./external_data/sachs/sachs.csv"
SACHS_MAX_ROWS          = 300    # the file has 7466 cells; None = all
PEMS_BAY_N_SENSORS      = 15
PEMS_BAY_N_POINTS       = 400
CLIMATE_N_YEARS         = 72
CLIMATE_CACHE_DIR       = "./external_data/climate_cache"
# PetShop: traffic regimes to run, among "low_traffic", "high_traffic" and
# "temporal_traffic1" (the last one has 31 series x 1500 points).
PETSHOP_REGIMES         = ("low_traffic", "high_traffic", "temporal_traffic1")
TEP_DIR_PATH            = "./external_data/tep"
BEELINE_DIR_PATH        = "./external_data/beeline_grn/BEELINE-data"
BEELINE_NETWORKS        = ["HSC", "VSC"]   # None = mCAD, VSC, HSC, GSD
BEELINE_MAX_RUNS        = 3      # runs per network (10 available); None = all
BEELINE_MAX_CELLS       = 500
CAUSAL_CHAMBER_DIR_PATH = "./external_data/causal_chamber"
CAUSAL_CHAMBER_N_POINTS = 500
DREAM3_DIR_PATH         = "./external_data/dream3"
DREAM3_SIZES            = (10,)
EASYVISTA_DIR_PATH      = "./external_data/easyvista_it"
EASYVISTA_MAX_POINTS    = 1500
RCAEVAL_DIR_PATH        = "./external_data/rcaeval"
RCAEVAL_FOLDER          = "RE1-OB"
RCAEVAL_MAX_CASES       = 10     # ~125 available; None = all
RCAEVAL_MAX_ROWS        = 500

RUN_RUNNER     = True
RUN_VALIDATION = True

OUTPUT_DIR = f"./Results_HeldOut/Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

EVALUATION_SCOPE_NOTE = (
    "These datasets were evaluated OUTSIDE the development suite (the benchmark "
    "suite of main.py), with the same models and configuration, and are reported "
    "separately, as held-out datasets."
)


# =============================================================================
# SCENARIO LOADING
# =============================================================================

def _loaders() -> dict:
    """Source key -> (provider, positional args, kwargs)."""
    return {
        "sachs": (load_sachs_benchmark, (SACHS_CSV_PATH,), dict(
            max_rows=SACHS_MAX_ROWS)),
        "pems_bay": (generate_pems_bay_benchmark_scenarios, (), dict(
            n_sensors=PEMS_BAY_N_SENSORS, n_points=PEMS_BAY_N_POINTS)),
        "real_climate": (load_tigramite_climate_benchmark, (), dict(
            n_years=CLIMATE_N_YEARS, cache_dir=CLIMATE_CACHE_DIR)),
        "petshop": (generate_petshop_benchmark_scenarios, (), dict(
            regimes=PETSHOP_REGIMES)),
        "tep": (generate_tep_benchmark_scenarios, (), dict(
            base_dir=TEP_DIR_PATH)),
        "beeline_grn": (generate_beeline_grn_benchmark_scenarios, (), dict(
            networks=BEELINE_NETWORKS, max_runs_per_network=BEELINE_MAX_RUNS,
            max_cells=BEELINE_MAX_CELLS, base_dir=BEELINE_DIR_PATH)),
        "causal_chamber": (generate_causal_chamber_benchmark_scenarios, (), dict(
            base_dir=CAUSAL_CHAMBER_DIR_PATH, n_points=CAUSAL_CHAMBER_N_POINTS)),
        "dream3": (generate_dream3_benchmark_scenarios, (), dict(
            sizes=DREAM3_SIZES, base_dir=DREAM3_DIR_PATH)),
        "easyvista_it": (generate_easyvista_it_benchmark_scenarios, (), dict(
            max_points=EASYVISTA_MAX_POINTS, cache_dir=EASYVISTA_DIR_PATH)),
        "rcaeval": (generate_rcaeval_benchmark_scenarios, (), dict(
            benchmark_folder=RCAEVAL_FOLDER, max_cases=RCAEVAL_MAX_CASES,
            max_rows=RCAEVAL_MAX_ROWS, cache_dir=RCAEVAL_DIR_PATH)),
    }


def _load_scenarios() -> tuple:
    """Returns (dataframes, {scenario name: source key})."""
    loaders = _loaders()
    # Every source of this runner must have a declared provenance.
    unknown = [s for s in ACTIVE_SOURCES if s not in loaders or s not in SOURCES]
    if unknown:
        raise KeyError(f"main_heldout: unknown source(s) {unknown} in "
                       f"ACTIVE_SOURCES. Available: {sorted(loaders)}")

    dataframes: list = []
    source_of: dict = {}
    for source in ACTIVE_SOURCES:
        provider, args, kwargs = loaders[source]
        loaded = load_with_fallback(source, provider, *args, **kwargs)
        for d in loaded:
            source_of[d['name']] = source
        dataframes += loaded
    return dataframes, source_of


def _models() -> list:
    by_name = {m['name']: m for m in cfg_exec.ALL_MODELS}
    missing = [n for n in MODEL_NAMES if n not in by_name]
    if missing:
        raise KeyError(f"main_heldout: models {missing} not found in "
                       f"Config.execution.ALL_MODELS")
    models = [by_name[n] for n in MODEL_NAMES]

    # cuts_plus is only imported by Config.execution when it is listed in
    # ACTIVE_MODEL_NAMES there.
    not_loaded = [m['name'] for m in models if m['pipeline'] is None]
    if not_loaded:
        raise RuntimeError(
            f"main_heldout: models {not_loaded} have no pipeline loaded -- "
            f"add them to Config.execution.ACTIVE_MODEL_NAMES.")

    if INCLUDE_GPDC:
        import main_gpdc
        models += [{k: v for k, v in m.items() if k != 'dataset_filter'}
                   for m in main_gpdc.GPDC_MODELS]
    return models


def _save_evaluation_scope(output_dir: str, dataframes: list, source_of: dict):
    os.makedirs(output_dir, exist_ok=True)
    lines = [EVALUATION_SCOPE_NOTE, "", "DATA ORIGIN PER SCENARIO", "-" * 48]
    for d in dataframes:
        origin, description = SOURCES[source_of[d['name']]]
        lines.append(f"{d['name']:<44}: {origin.upper()} -- {description}")
    with open(os.path.join(output_dir, "evaluation_scope.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    total_start_time = time.perf_counter()

    models = _models()
    dataframes, source_of = _load_scenarios()
    print(f"\n[main_heldout] {len(dataframes)} scenario(s) x {len(models)} model(s) "
          f"({[m['name'] for m in models]}) -> {OUTPUT_DIR}")
    print(f"[main_heldout] {EVALUATION_SCOPE_NOTE}")
    for source in ACTIVE_SOURCES:
        origin, description = SOURCES[source]
        n = sum(1 for s in source_of.values() if s == source)
        print(f"  [{origin.upper()}] {source} ({n} scenario(s)): {description}")

    if not dataframes:
        raise RuntimeError("main_heldout: no scenario could be loaded.")

    _save_evaluation_scope(OUTPUT_DIR, dataframes, source_of)

    if RUN_RUNNER:
        print("\n" + "=" * 65)
        print("  STEP 1 -- RUNNER (HELD-OUT DATASETS)")
        print("=" * 65)
        main_mod.run_all(
            dataframes=dataframes,
            models=models,
            config=cfg_exec.PIPELINE_CONFIG,
            output_dir=OUTPUT_DIR,
        )

    if RUN_VALIDATION:
        print("\n" + "=" * 65)
        print("  STEP 2 -- VALIDATION (HELD-OUT DATASETS)")
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
