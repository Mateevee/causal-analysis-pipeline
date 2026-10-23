"""
main_seed_replication.py -- E1a: NEW REALIZATIONS of the four synthetic
generators of the development suite (data variance), separated from main.py (same
spirit as main_heldout.py: a dedicated entry point that reuses main.py's
run_all/validate_all).

Question
---
Do the synthetic results (detection F1, lag recovery, effect estimation) hold
on new realizations of the same generators, and with what confidence interval?

Evaluation scope -- read before reporting
---
These are NEW REALIZATIONS OF DEVELOPMENT GENERATORS, NOT held-out data. The
four generators below were used while the pipeline was designed; only the
seeds are new. What this run measures is whether the pipeline was fitted to
the specific realizations of main.py (seeds 11..55 for the VAR, 11/22/33 for
tigramite, 1/2/3 for Lorenz-96 and the logistic maps) or to the TYPE of data:
in the first case its performance would drop on the new seeds. Report it in
those terms, and report these scenarios apart from the development suite and from the
held-out datasets (main_heldout.py).

What changes with respect to main.py
---
  - Seeds only: NEW_SEEDS below (10 per configuration), none of them used by
    any other run of this repository. Every other generator parameter is the
    one of main.py / Config.execution (SYNTHETIC_VAR_SCENARIOS for the VAR;
    the literals of main.py for the other three).
  - Models: the six methods at tau_max = 10 only ('patch_tau10' and the five
    baselines' '_tau10' variants). INCLUDE_TAU24 adds their tau = 24
    versions (optional, roughly doubles the run time).

    Family              Configurations                 x seeds   Scenarios
    synthetic_var       5 sizes (N = 5, 6, 12, 22, 40)   x 10       50
    logistic_maps       4 topologies                     x 10       40
    lorenz96            K = 8, 10                        x 10       20
    tigramite           5 SCM types                      x 10       50
                                                                   ---
                                                                   160

Note on the tigramite family: its seed changes the noise realization only;
the causal graph and the link functions of each SCM type are fixed by the
generator (Data/tigramite_scm.py). For the VAR, the seed also draws the graph
and the coefficients. For Lorenz-96 and the logistic maps the graph is fixed
by the equations and the seed changes the initial condition and the noise.

Written to OUTPUT_DIR:
  <scenario>/              as in main.py: source.csv (the generated series),
                           ground_truth.csv, scenario_params.txt (every
                           generator parameter, seed included; for the VAR
                           also each causal link's lag and coefficient),
                           validation_summary.*, and one folder per model
                           (results.csv, original.csv, time.txt, FP/FN)
  seeds_manifest.csv       one line per scenario: family, configuration,
                           seed and generator parameters (supplementary table)
  replication_scope.txt    the evaluation-scope note above and the seed list
  run_config.txt           git commit (and whether tracked files had
                           uncommitted changes), Python version, and the exact
                           configuration of every model and of PIPELINE_CONFIG
  seed_*.csv, var_*.csv, seed_replication_summary.txt
                           the analysis (RUN_ANALYSIS), see
                           Auxiliary/seed_replication_analysis.py
(Deliberately NOT named evaluation_scope.txt: Auxiliary/ranking_metrics.py
reads that file as the list of held-out scenarios.)

Analysis: run at the end (RUN_ANALYSIS), or on its own with
python Auxiliary/seed_replication_analysis.py <OUTPUT_DIR>.

Usage
---
No command-line arguments, by design -- edit the constants in the
"EXECUTION CONTROL" section below and run:

    python main_seed_replication.py

To resume an interrupted run, set RESUME_DIR to its folder: every
(scenario, model) whose time.txt says 'status: ok' is kept, the rest is run.
"""
from __future__ import annotations

import datetime
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

import main as main_mod  # reuses run_all/validate_all/_format_duration
from Config import execution as cfg_exec
from Data import (
    generate_synthetic_var_benchmark_scenarios,
    generate_logistic_maps_benchmark_scenarios,
    generate_lorenz96_benchmark_scenarios,
    generate_tigramite_benchmark_scenarios,
    load_with_fallback,
)
from Auxiliary.consolidate_summaries import consolidate
from Auxiliary import seed_replication_analysis
from Validation.FPChecker import export_fp
from Validation.FNChecker import export_fn


# =============================================================================
# EXECUTION CONTROL -- edit here.
# =============================================================================

# New seeds, shared by every family and configuration. None of them is used by
# main.py, main_gpdc.py, main_granger_te.py, main_pcmci_nobh.py,
# main_sensitivity.py or main_ablation_patch.py (which use 1-3, 11-55).
NEW_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]

# Seeds of the main run, written to replication_scope.txt for reference and
# checked against NEW_SEEDS at start-up.
MAIN_RUN_SEEDS = {
    "synthetic_var": sorted({s[1] for s in cfg_exec.SYNTHETIC_VAR_SCENARIOS}),
    "logistic_maps": [1, 2, 3],
    "lorenz96":      [1, 2, 3],
    "tigramite":     [11, 22, 33],
}

# Families run. Comment a line out to skip it.
ACTIVE_FAMILIES = [
    "synthetic_var",
    "logistic_maps",
    "lorenz96",
    "tigramite",
]

# Generator arguments of main.py (main.py, section "Loading with fallback"),
# except the seeds. Keep them in sync if main.py changes.
LOGISTIC_N_POINTS  = 300
LORENZ96_KS        = [8, 10]
LORENZ96_N_POINTS  = 300
TIGRAMITE_N_POINTS = 300

# The six methods at tau_max = 10.
MODEL_NAMES_TAU10 = [
    'patch_tau10',
    'pcmci_tigramite_tau10',
    'pcmciplus_tau10',
    'varlingam_tau10',
    'dynotears_tau10',
    'cuts_plus_tau10',
]
# Optional: the same six methods at tau_max = 24.
INCLUDE_TAU24 = False
MODEL_NAMES_TAU24 = ['patch', 'pcmci_tigramite', 'pcmciplus', 'varlingam',
                     'dynotears', 'cuts_plus']

RUN_RUNNER     = True
RUN_VALIDATION = True
RUN_ANALYSIS   = True   # Auxiliary/seed_replication_analysis.py at the end

# None = new run. A path = resume that run (see the module docstring).
RESUME_DIR = None

OUTPUT_DIR = RESUME_DIR or (
    f"./Results_Seeds/Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

EVALUATION_SCOPE_NOTE = (
    "New realizations (new seeds) of the four synthetic generators of the "
    "development suite -- development generators, NOT held-out data. Generator "
    "parameters identical to main.py; only the seeds differ. If the pipeline "
    "had been tuned to the specific realizations of main.py rather than to "
    "the type of data, its performance would drop on these seeds. Reported "
    "apart from the development suite and from the held-out datasets."
)


# =============================================================================
# SCENARIO LOADING
# =============================================================================

def _var_config_label(n_entities: int) -> str:
    return f"N{n_entities}"


def _load_family(family: str) -> list:
    """Scenarios of one family, each tagged with '_family', '_config' and '_seed'."""
    out = []
    if family == "synthetic_var":
        for (name_prefix, _main_seed, n_points, n_entities, prob_edge,
             max_parents, max_lag, min_edges) in cfg_exec.SYNTHETIC_VAR_SCENARIOS:
            loaded = load_with_fallback(
                name_prefix, generate_synthetic_var_benchmark_scenarios,
                seeds=NEW_SEEDS, n_points=n_points, name_prefix=name_prefix,
                n_entities=n_entities, prob_edge=prob_edge,
                max_parents=max_parents, max_lag=max_lag, min_edges=min_edges,
            )
            for d in loaded:
                d['_config'] = _var_config_label(n_entities)
            out += loaded
    elif family == "logistic_maps":
        out = load_with_fallback(
            "logistic_maps", generate_logistic_maps_benchmark_scenarios,
            n_points=LOGISTIC_N_POINTS, seeds=NEW_SEEDS)
        for d in out:
            d['_config'] = d['_params']['topology']
    elif family == "lorenz96":
        out = load_with_fallback(
            "lorenz96", generate_lorenz96_benchmark_scenarios,
            Ks=LORENZ96_KS, n_points=LORENZ96_N_POINTS, seeds=NEW_SEEDS)
        for d in out:
            d['_config'] = f"K{d['_params']['K']}"
    elif family == "tigramite":
        out = load_with_fallback(
            "tigramite", generate_tigramite_benchmark_scenarios,
            seeds=NEW_SEEDS, n_points=TIGRAMITE_N_POINTS)
        for d in out:
            d['_config'] = d['_params']['scenario']
    else:
        raise KeyError(f"main_seed_replication: unknown family '{family}'")

    for d in out:
        d['_family'] = family
        d['_seed'] = d['_params']['seed']
        # run_all writes scenario_params.txt (every generator parameter, and
        # the causal links with lag and coefficient for the VAR) only when
        # '_scenario' is set; the logistic, Lorenz-96 and tigramite providers
        # leave it as None.
        d['_scenario'] = d.get('_scenario') or f"{family}_{d['_config']}"
    return out


def _load_scenarios() -> list:
    reused = {f: sorted(set(NEW_SEEDS) & set(s)) for f, s in MAIN_RUN_SEEDS.items()}
    reused = {f: s for f, s in reused.items() if s}
    if reused:
        raise ValueError(f"main_seed_replication: NEW_SEEDS reuses seeds of the "
                         f"main run: {reused}")
    if len(set(NEW_SEEDS)) != len(NEW_SEEDS):
        raise ValueError("main_seed_replication: NEW_SEEDS has duplicates.")

    dataframes = []
    for family in ACTIVE_FAMILIES:
        loaded = _load_family(family)
        print(f"  [{family}] {len(loaded)} scenario(s)")
        dataframes += loaded
    return dataframes


def _models() -> list:
    names = MODEL_NAMES_TAU10 + (MODEL_NAMES_TAU24 if INCLUDE_TAU24 else [])
    by_name = {m['name']: m for m in cfg_exec.ALL_MODELS}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"main_seed_replication: models {missing} not found in "
                       f"Config.execution.ALL_MODELS")
    models = [by_name[n] for n in names]

    # cuts_plus is only imported by Config.execution when it is listed in
    # ACTIVE_MODEL_NAMES there.
    not_loaded = [m['name'] for m in models if m['pipeline'] is None]
    if not_loaded:
        raise RuntimeError(
            f"main_seed_replication: models {not_loaded} have no pipeline loaded "
            f"-- add them to Config.execution.ACTIVE_MODEL_NAMES.")
    return models


# =============================================================================
# MANIFEST / SCOPE
# =============================================================================

_MANIFEST_PARAMS = ["n_points", "n_entities", "prob_edge", "max_parents",
                    "max_lag", "min_edges", "n_causal_edges", "topology", "r",
                    "eps", "K", "F", "type", "n_causal_links"]


def _save_manifest(output_dir: str, dataframes: list):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for d in dataframes:
        params = d.get('_params', {})
        row = {"family": d['_family'], "config": d['_config'], "seed": d['_seed'],
               "scenario": d['name'], "generator": d.get('_scenario')}
        row.update({k: params[k] for k in _MANIFEST_PARAMS if k in params})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "seeds_manifest.csv"), index=False)

    lines = [EVALUATION_SCOPE_NOTE, "",
             f"NEW SEEDS (every family and configuration): {NEW_SEEDS}",
             "", "SEEDS OF THE MAIN RUN (not reused here)"]
    lines += [f"  {f:<15}: {s}" for f, s in MAIN_RUN_SEEDS.items()]
    lines += ["", "SCENARIOS PER FAMILY"]
    for family in ACTIVE_FAMILIES:
        fam = [d for d in dataframes if d['_family'] == family]
        configs = sorted({d['_config'] for d in fam})
        lines.append(f"  {family:<15}: {len(fam)} = {len(configs)} configuration(s) "
                     f"{configs} x {len(NEW_SEEDS)} seeds")
    lines += ["", f"Models: {[m['name'] for m in _models()]}"]
    with open(os.path.join(output_dir, "replication_scope.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:  # git missing: the run must not fail for it
        return f"unavailable ({e})"


def _save_run_config(output_dir: str, models: list):
    """Code version and the exact configuration of every model of the run."""
    changed = _git("status", "--porcelain", "--untracked-files=no")
    lines = [f"Started at   : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Git commit   : {_git('rev-parse', 'HEAD')}",
             f"Git branch   : {_git('rev-parse', '--abbrev-ref', 'HEAD')}",
             f"Uncommitted changes to tracked files: "
             f"{len(changed.splitlines()) if changed else 0} file(s)",
             f"Python       : {platform.python_version()} ({platform.platform()})",
             f"NEW_SEEDS    : {NEW_SEEDS}",
             "", "MODELS (Config.execution.ALL_MODELS entries; each runs with "
                 "PIPELINE_CONFIG updated by its config_override)"]
    for m in models:
        lines.append(f"  {m['name']}  ({m['pipeline'].__module__})")
        for k, v in sorted(m.get('config_override', {}).items()):
            lines.append(f"      {k:<40}= {v!r}")
    lines += ["", "PIPELINE_CONFIG"]
    lines += [f"  {k:<40}= {v!r}" for k, v in sorted(cfg_exec.PIPELINE_CONFIG.items())]
    with open(os.path.join(output_dir, "run_config.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# RESUME
# =============================================================================

def _is_done(model_folder: Path) -> bool:
    time_path = model_folder / "time.txt"
    if not (model_folder / "results.csv").exists() or not time_path.exists():
        return False
    for line in time_path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "status":
            return value.strip().lower() == "ok"
    return False


def _run(dataframes: list, models: list):
    if RESUME_DIR is None:
        main_mod.run_all(dataframes=dataframes, models=models,
                         config=cfg_exec.PIPELINE_CONFIG, output_dir=OUTPUT_DIR)
        return
    root = Path(OUTPUT_DIR)
    for d in dataframes:
        todo = [m for m in models if not _is_done(root / d['name'] / m['name'])]
        if not todo:
            print(f"  [RESUME] {d['name']}: all models done -- skipped.")
            continue
        print(f"  [RESUME] {d['name']}: running {[m['name'] for m in todo]}")
        main_mod.run_all(dataframes=[d], models=todo,
                         config=cfg_exec.PIPELINE_CONFIG, output_dir=OUTPUT_DIR)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    total_start_time = time.perf_counter()

    models = _models()
    dataframes = _load_scenarios()
    if not dataframes:
        raise RuntimeError("main_seed_replication: no scenario could be loaded.")
    print(f"\n[main_seed_replication] {len(dataframes)} scenario(s) x {len(models)} "
          f"model(s) ({[m['name'] for m in models]}) -> {OUTPUT_DIR}")
    print(f"[main_seed_replication] {EVALUATION_SCOPE_NOTE}")

    _save_manifest(OUTPUT_DIR, dataframes)
    if RESUME_DIR is None:  # a resumed run keeps the configuration it started with
        _save_run_config(OUTPUT_DIR, models)

    if RUN_RUNNER:
        print("\n" + "=" * 65)
        print("  STEP 1 -- RUNNER (NEW SEEDS OF THE SYNTHETIC GENERATORS)")
        print("=" * 65)
        _run(dataframes, models)

    if RUN_VALIDATION:
        print("\n" + "=" * 65)
        print("  STEP 2 -- VALIDATION (NEW SEEDS OF THE SYNTHETIC GENERATORS)")
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

    if RUN_ANALYSIS:
        print("Analysis (F1 per family, ranking stability, Tables 4a/4b)")
        seed_replication_analysis.run(OUTPUT_DIR)

    total_time_fmt = main_mod._format_duration(time.perf_counter() - total_start_time)
    print("\n" + "#" * 65)
    print(f"  TOTAL SCRIPT EXECUTION TIME: {total_time_fmt}")
    print("#" * 65)


if __name__ == '__main__':
    main()
