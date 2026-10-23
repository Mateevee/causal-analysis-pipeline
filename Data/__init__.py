"""
Data
===============================================================================
Package with the benchmark/validation scenario "providers" of the causal
pipeline. See `Data.base` for the common contract (provider signature,
return format, registry) and for how to add a new scenario, and
Data/README.md for how to obtain each dataset (which ones are generated,
auto-downloaded or must be downloaded by hand, and where to put them).

Modules:
    base            common contract (ScenarioEntry, @register, helpers)
    tigramite_scm   synthetic benchmark via the tigramite package's SCM
    synthetic_var   synthetic benchmark via linear VAR with lags (does not
                    depend on the tigramite package; ground truth guaranteed
                    by construction, it's the very graph used to simulate)
    lorenz96        chaotic Lorenz-96 benchmark
    logistic_maps   chaotic coupled logistic maps benchmark
    causaltime      generated from real observations — CausalTime
                    (medical/pm25/traffic domains)
    alarm           sampled — ALARM Bayesian network (ICU monitoring)
    netsim          simulated benchmark — NetSim fMRI (Smith et al. 2011)

  Held-out datasets, evaluated apart from the development suite
  (main_heldout.py):
    real_climate    reanalysis — climate (Walker Circulation, ERA5, via
                    tigramite)
    sachs           measured — Sachs et al. 2005, protein signaling
                    (flow cytometry; NO real time axis)
    pems_bay        measured — PeMS-BAY traffic sensors
    petshop         measured — PetShop microservice application on AWS
                    (latency per component, normal operation)
    tep             SIMULATED — Tennessee Eastman Process (via CIPCaD-Bench;
                    simulator of a real plant, not plant measurements;
                    ground truth derived from the process/P&ID topology)

  Further third-party datasets (not in the development suite; see each
  module's docstring for the caveats). main_heldout.py runs beeline_grn,
  causal_chamber, dream3, easyvista_it and rcaeval as held-out datasets; the
  others have a provider but are run by no entry point and are not in the
  paper:
    beeline_grn        simulated — BEELINE curated GRNs (BoolODE)
    causal_chamber     measured — Causal Chambers (light tunnel / wind tunnel)
    causal_rivers      measured — CausalRivers (Bavaria river discharge)
    cipcad_uf          measured — CIPCaD-Bench ultra-processed food plant
    dream3             simulated — DREAM3 in-silico gene networks
    dream4             simulated — DREAM4 in-silico gene networks
    easyvista_it       measured — EasyVista IT monitoring
    ess_cryo           measured — ESS cryogenic plant
    metr_la            measured — METR-LA traffic sensors
    nino_bc            measured — ENSO -> British Columbia temperature
    nino_barents_kara  measured — ENSO -> Barents-Kara sea ice (contested GT)
    rcaeval            measured — RCAEval microservice fault injection
    tuebingen_pairs    measured — Tuebingen cause-effect pairs (no time axis)

Typical usage in main.py (without changing run_all()/validate_dataset()):

    from Data import (
        generate_tigramite_benchmark_scenarios,
        generate_synthetic_var_benchmark_scenarios,
        generate_lorenz96_benchmark_scenarios,
        generate_logistic_maps_benchmark_scenarios,
        generate_causaltime_benchmark_scenarios,
    )

Or, via the registry (useful for generically iterating over all available
providers):

    from Data import list_providers, get_provider
    for key in list_providers():
        print(key, get_provider(key))
"""
from .base import (
    ScenarioEntry,
    ScenarioProvider,
    register,
    list_providers,
    get_provider,
    load_with_fallback,
    log_scenario_failure,
    SCENARIO_FAILURES_LOG,
)

from .tigramite_scm import generate_tigramite_benchmark_scenarios
from .synthetic_var import generate_synthetic_var_benchmark_scenarios
from .lorenz96 import generate_lorenz96_benchmark_scenarios
from .logistic_maps import generate_logistic_maps_benchmark_scenarios
from .real_climate import load_tigramite_climate_benchmark
from .causaltime import generate_causaltime_benchmark_scenarios
from .tep import generate_tep_benchmark_scenarios
from .sachs import load_sachs_benchmark
from .alarm import load_alarm_benchmark
from .pems_bay import generate_pems_bay_benchmark_scenarios
from .petshop import generate_petshop_benchmark_scenarios
from .netsim import generate_netsim_benchmark_scenarios
from .beeline_grn import generate_beeline_grn_benchmark_scenarios
from .causal_chamber import generate_causal_chamber_benchmark_scenarios
from .causal_rivers import generate_causal_rivers_benchmark_scenarios
from .cipcad_uf import generate_cipcad_uf_benchmark_scenarios
from .dream3 import generate_dream3_benchmark_scenarios
from .dream4 import generate_dream4_benchmark_scenarios
from .easyvista_it import generate_easyvista_it_benchmark_scenarios
from .ess_cryo import generate_ess_cryo_benchmark_scenarios
from .metr_la import generate_metr_la_benchmark_scenarios
from .nino_bc import load_nino_bc_benchmark
from .nino_barents_kara import load_nino_barents_kara_benchmark
from .rcaeval import generate_rcaeval_benchmark_scenarios
from .tuebingen_pairs import generate_tuebingen_pairs_benchmark_scenarios

__all__ = [
    "ScenarioEntry",
    "ScenarioProvider",
    "register",
    "list_providers",
    "get_provider",
    "load_with_fallback",
    "log_scenario_failure",
    "SCENARIO_FAILURES_LOG",
    "generate_tigramite_benchmark_scenarios",
    "generate_synthetic_var_benchmark_scenarios",
    "generate_lorenz96_benchmark_scenarios",
    "generate_logistic_maps_benchmark_scenarios",
    "load_tigramite_climate_benchmark",
    "generate_causaltime_benchmark_scenarios",
    "generate_tep_benchmark_scenarios",
    "load_sachs_benchmark",
    "load_alarm_benchmark",
    "generate_pems_bay_benchmark_scenarios",
    "generate_petshop_benchmark_scenarios",
    "generate_netsim_benchmark_scenarios",
    "generate_beeline_grn_benchmark_scenarios",
    "generate_causal_chamber_benchmark_scenarios",
    "generate_causal_rivers_benchmark_scenarios",
    "generate_cipcad_uf_benchmark_scenarios",
    "generate_dream3_benchmark_scenarios",
    "generate_dream4_benchmark_scenarios",
    "generate_easyvista_it_benchmark_scenarios",
    "generate_ess_cryo_benchmark_scenarios",
    "generate_metr_la_benchmark_scenarios",
    "load_nino_bc_benchmark",
    "load_nino_barents_kara_benchmark",
    "generate_rcaeval_benchmark_scenarios",
    "generate_tuebingen_pairs_benchmark_scenarios",
]
