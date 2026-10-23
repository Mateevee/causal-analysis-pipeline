"""
main.py — Causal Discovery Pipeline: Runner + Validation (with error diagnostics)
=========================================================================================
Runs every model over every scenario (run_all) and validates the results
(validate_all). Upon catching an exception in any pipeline, writes a
detailed diagnostic file to disk at:

    <script directory>/logs/failures/<df_name>__<model_name>__<timestamp>.log

The log includes:
    - full traceback
    - shape / dtypes / head of the input dataframe
    - date_col dtypes after conversion (how many NaT, etc.)
    - the effective config used (cfg) and the pipeline's parameters

The run is controlled by Config/execution.py (execution flags, active
datasets, MODELS, PIPELINE_CONFIG) -- no command-line flags.
"""

import importlib
import os
import sys
import time
import datetime
import traceback
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Optional, List, Union
from itertools import combinations as _combinations
from scipy import stats as _scipy_stats
from scipy.optimize import nnls as _scipy_nnls, minimize_scalar as _scipy_minimize_scalar

os.environ["PYTHONWARNINGS"] = "ignore"

# Forces UTF-8 on stdout/stderr (errors="replace" so an unsupported
# character never crashes the run).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from Auxiliary.consolidate_summaries import consolidate
from Auxiliary.console_progress import PinnedProgress
from Validation.FPChecker import export_fp
from Validation.FNChecker import export_fn



# =============================================================================
# PIPELINE MODULE IMPORTS (one per file)
# =============================================================================
# Each module exposes a run_pipeline_* with the SAME call signature:
#     pipeline(df: pd.DataFrame, **cfg) -> pd.DataFrame  (with a 'significant' column)
# See each module's own file for method-specific documentation. main.py
# only orchestrates: it does not redefine any model's logic.
from ModelComparison.pipeline_logging import LOGS_DIR, _log_pipeline_error

# Run control panel (execution flags, active datasets/benchmarks, MODELS,
# PIPELINE_CONFIG) -- see Config/execution.py. Extracted out of main.py so
# that switching scenario/config does not require editing the runner.
from Config import execution as cfg_exec

# =============================================================================
# GLOBAL FLAG -- canonical (non-directional) vs. directional pairing
# =============================================================================
#
# True  (default): A->B and B->A are the SAME
#         pair for comparison with the ground truth (frozenset). Correct for
#         a dataset with fixed, disjoint entity roles, where there is never
#         real ambiguity about direction.
# False : A->B and B->A become DISTINCT pairs (ordered tuple). A model only
#         counts as a hit if it detects the correct direction of the
#         relationship -- recommended for benchmarks where the variables have
#         no fixed role (Lorenz-96, coupled logistic maps, tigramite
#         scenarios with topologies such as 'convergence', which have edges
#         in both directions between different variable pairs).
#
# Can be overridden per call (validate_dataset(..., use_canonical_pair=...))
# without needing to change this constant; the value here is used only as
# the default. The effective value lives in Config/execution.py
# (USE_CANONICAL_PAIR_DEFAULT).
# =============================================================================
USE_CANONICAL_PAIR_DEFAULT = cfg_exec.USE_CANONICAL_PAIR_DEFAULT


def _pair_key(a: str, b: str, use_canonical_pair: bool = USE_CANONICAL_PAIR_DEFAULT):
    """
    Single point of definition of "pair identity" for the whole validation.

    use_canonical_pair=True  -> frozenset({a, b})           (A->B == B->A)
    use_canonical_pair=False -> (a, b)                       (A->B != B->A)

    Every pairing function below (_causal_gt_pairs, _predicted_pairs,
    _property_pair, etc.) must call this function instead of building a
    frozenset/tuple directly, so that the flag has a single, consistent
    effect throughout the whole validation.
    """
    if use_canonical_pair:
        return frozenset([a, b])
    return (a, b)


def _unpack_pair(pair, use_canonical_pair: bool = USE_CANONICAL_PAIR_DEFAULT) -> tuple:
    """Inverse of _pair_key: returns (a, b) for display/export."""
    if use_canonical_pair:
        return _pair_to_ab(pair)
    return pair  # already an ordered (a, b) tuple


# =============================================================================
# BENCHMARK / VALIDATION SCENARIOS -- Data/ package
# =============================================================================
# The benchmark scenario generators/loaders (synthetic SCM via tigramite,
# linear synthetic VAR, Lorenz-96, coupled logistic maps, CausalTime, ALARM,
# NetSim) live in the Data/ package (same level as this file) --
# see Data/base.py for the common contract (return format {'name','df','gt',
# '_scenario','_params'}) and for how to add a new scenario without
# touching run_all()/validate_dataset().
from Data import (
    generate_tigramite_benchmark_scenarios,
    generate_synthetic_var_benchmark_scenarios,
    generate_lorenz96_benchmark_scenarios,
    generate_logistic_maps_benchmark_scenarios,
    load_with_fallback,
    generate_causaltime_benchmark_scenarios,
    load_alarm_benchmark,
    generate_netsim_benchmark_scenarios,
)
# The held-out providers (Sachs, PeMS-BAY, Walker Circulation, PetShop, TEP,
# BEELINE, Causal Chambers, DREAM3, EasyVista, RCAEval) are NOT part of this
# suite: they are evaluated outside the development suite, as held-out
# datasets, by main_heldout.py.


# =============================================================================
# RUNNER UTILITIES
# =============================================================================

def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}min")
    parts.append(f"{s:.2f}s")
    return " ".join(parts)


def _save_time(path: str, model_name: str, module: str,
                  start_dt: datetime.datetime, duration_s: float,
                  status: str = "ok", error: str = ""):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"model      : {model_name}\n")
        f.write(f"module     : {module}\n")
        f.write(f"start      : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"end        : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"duration    : {_format_duration(duration_s)}\n")
        f.write(f"duration_s  : {duration_s:.4f}\n")
        f.write(f"status     : {status}\n")
        if error:
            f.write(f"error      : {error}\n")


def _save_scenario_params(path: str, scenario, params: dict):
    with open(path, "w", encoding="utf-8") as f:
        f.write("SYNTHETIC SCENARIO PARAMETERS\n")
        f.write("=" * 48 + "\n")
        for k, v in params.items():
            f.write(f"{k:<25}: {v}\n")
        f.write("\nCAUSAL LINKS (ground truth)\n")
        f.write("-" * 48 + "\n")


import ast
import json as _json_log


def _save_cascade_log(df: pd.DataFrame, model_folder: str,
                         txt_file_name: str = "cascade_log.txt",
                         json_file_name: str = "cascade_log.json"):
    """
    If `df` (typically the UNFILTERED result -- original.csv -- so as to
    cover both significant and non-significant pairs) contains the
    'cascade_log' column (produced by dml_engine.py -- one log per pair,
    with each model's decision in the BR->RF->KRR->NadarayaWatson
    cascade), writes that log to:

        <model_folder>/cascade_log.txt   -- human-readable version, one block per pair
        <model_folder>/cascade_log.json  -- raw version (list of dicts),
                                            for programmatic reprocessing

    Does nothing (silently) if the column does not exist -- pipelines that do
    not go through this cascade (PCMCI, DYNOTEARS, CUTS+) do not have this
    column, so this function is harmless for them.
    """
    if df is None or df.empty or "cascade_log" not in df.columns:
        return

    lines = []
    json_records = []

    for _, row in df.iterrows():
        source  = row.get("source_entity", "?")
        target = row.get("target_entity", "?")
        f_o     = row.get("source_property", "")
        f_d     = row.get("target_property", "")
        lag     = row.get("lag", row.get("best_lag", "?"))

        cascade_log = row["cascade_log"]
        # If df came from a re-read CSV (string), try to deserialize it.
        if isinstance(cascade_log, str):
            try:
                cascade_log = ast.literal_eval(cascade_log)
            except Exception:
                lines.append(f"PAIR: {source} -> {target}  [raw cascade_log, "
                              f"could not be parsed]: {cascade_log}\n")
                continue

        if not isinstance(cascade_log, list):
            continue

        pair_str = f"{source}" + (f" ({f_o})" if f_o else "") + "  ->  " \
                 + f"{target}" + (f" ({f_d})" if f_d else "")

        lines.append("=" * 78)
        lines.append(f"PAIR: {pair_str}   lag={lag}")
        lines.append(f"  model_used         : {row.get('dml_model_used', '?')}")
        lines.append(f"  cascade_confidence : {row.get('cascade_confidence', '?')}")
        lines.append(f"  significant        : {row.get('significant', row.get('significant', '?'))}")
        lines.append("-" * 78)

        for entry in cascade_log:
            accepted = entry.get("accepted", False)
            lines.append(f"  [{entry.get('model','?')}]  accepted={accepted}  "
                          f"score={entry.get('score')}")
            if accepted:
                lines.append(f"      t_stat={entry.get('t_stat')}  "
                              f"pvalue={entry.get('pvalue')} "
                              f"(pval_max_used={entry.get('pval_max_used')})  "
                              f"cv_ci={entry.get('cv_ci')}  cv_theta={entry.get('cv_theta')}")
            else:
                for reason in entry.get("reasons_fail", []):
                    lines.append(f"      failed: {reason}")
            lines.append(f"      var_res_T={entry.get('var_res_T')}  "
                          f"r2_nuisance_T={entry.get('r2_nuisance_T')}  "
                          f"r2_nuisance_O={entry.get('r2_nuisance_O')}  "
                          f"n_folds_ok={entry.get('n_folds_ok')}")
        lines.append("")

        json_records.append({
            "source_entity": source, "target_entity": target,
            "source_property": f_o, "target_property": f_d, "lag": lag,
            "dml_model_used": row.get("dml_model_used"),
            "cascade_confidence": row.get("cascade_confidence"),
            "significant": row.get("significant", row.get("significant")),
            "cascade_log": cascade_log,
        })

    if not lines:
        return

    with open(os.path.join(model_folder, txt_file_name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(os.path.join(model_folder, json_file_name), "w", encoding="utf-8") as f:
        _json_log.dump(json_records, f, ensure_ascii=False, indent=2, default=str)

    print(f"  [CASCADE-LOG] {txt_file_name} + {json_file_name} saved to "
          f"{model_folder}/ ({len(json_records)} pairs with a cascade log)")


def _resolve_submodel_list(pipeline: Callable, dml_models_entry) -> Optional[List[str]]:
    if not dml_models_entry:
        return None

    if dml_models_entry is True:
        try:
            mod_name = pipeline.__module__
            module = sys.modules.get(mod_name) or importlib.import_module(mod_name)
            list_fn = getattr(module, 'list_dml_models', None)
            if list_fn is not None:
                all_models = list_fn()
                submodels = [m for m in all_models]
                return submodels if submodels else None
        except Exception as e:
            print(f"  [WARNING] Could not obtain list_dml_models(): {e}")
        return None

    if isinstance(dml_models_entry, list):
        submodels = [m for m in dml_models_entry]
        return submodels if submodels else None

    return None


def _score_submodel(results: pd.DataFrame) -> tuple:
    if results is None or results.empty:
        return (0, 0.0, 0.0)
    sig = results[results['significant'] == True]
    n_sig = len(sig)
    mean_pvalue = float(sig['pvalue'].mean()) if n_sig > 0 else 1.0
    mean_strength  = float(sig['strength'].mean())  if (n_sig > 0 and 'strength' in sig.columns) else 0.0
    return (n_sig, -mean_pvalue, mean_strength)


def _save_submodel_comparison(path: str, model_name: str, df_name: str,
                                   records: list, best_name: str):
    lines = []
    lines.append("=" * 72)
    lines.append(f"  DML SUBMODEL COMPARISON")
    lines.append(f"  Pipeline : {model_name}")
    lines.append(f"  Dataset  : {df_name}")
    lines.append(f"  Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)
    lines.append("")

    header = (
        f"  {'SUBMODEL':<20} {'STATUS':<10} {'N_SIG':>6} {'N_TOT':>6} "
        f"{'PVAL_MED':>9} {'STRENGTH_MED':>10} {'TIME':>12}  CHOSEN"
    )
    lines.append(header)
    lines.append("  " + "-" * 70)

    for r in records:
        chosen = "<-- BEST" if r['submodel'] == best_name else ""
        if r['status'] == 'ok':
            line = (
                f"  {r['submodel']:<20} {r['status']:<10} {r['n_sig']:>6} {r['n_total']:>6} "
                f"{r['mean_pvalue']:>9.4f} {r['mean_strength']:>10.4f} "
                f"{r['dur_fmt']:>12}  {chosen}"
            )
        else:
            line = (
                f"  {r['submodel']:<20} {r['status']:<10} {'---':>6} {'---':>6} "
                f"{'---':>9} {'---':>10} "
                f"{r['dur_fmt']:>12}  {chosen}"
            )
        lines.append(line)

    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  CHOSEN SUBMODEL: {best_name}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("  DETAILS PER SUBMODEL")
    lines.append("  " + "-" * 70)

    for r in records:
        lines.append(f"\n  [{r['submodel']}]")
        lines.append(f"    Status          : {r['status']}")
        if r['status'] == 'ok':
            lines.append(f"    Sig. pairs      : {r['n_sig']} of {r['n_total']}")
            lines.append(f"    Mean p-value    : {r['mean_pvalue']:.4f}")
            lines.append(f"    Mean strength   : {r['mean_strength']:.4f}")
        elif r['error']:
            lines.append(f"    Error           : {r['error']}")
        lines.append(f"    Duration        : {r['dur_fmt']}")
        if r['submodel'] == best_name:
            lines.append(f"    ** CHOSEN SUBMODEL -- this model's final result **")

    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    csv_path = path.replace(".txt", ".csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)


# =============================================================================
# SUBMODEL RUNNER
# =============================================================================

def _run_submodels(pipeline, df, cfg, submodels, model_folder,
                       model_name, df_name) -> tuple:
    sub_folder = os.path.join(model_folder, "submodels")
    os.makedirs(sub_folder, exist_ok=True)

    results_by_sub: dict = {}
    records: list = []

    for submodel in submodels:
        sm_folder = os.path.join(sub_folder, submodel)
        os.makedirs(sm_folder, exist_ok=True)

        print(f"\n  {'─'*60}")
        print(f"  [SUBMODEL] {model_name} / dml_model='{submodel}'")
        print(f"  {'─'*60}")

        cfg_sub   = {**cfg, 'dml_model': submodel}
        start    = time.perf_counter()
        start_dt = datetime.datetime.now()

        try:
            res = pipeline(df, **cfg_sub)
            duration_s = time.perf_counter() - start
            dur_fmt   = _format_duration(duration_s)

            if res is None:
                res = pd.DataFrame()

            # ── GUARD: the pipeline may return a DataFrame without the
            #    'significant' column (e.g. no candidate pair survived the
            #    pre-filters). Treated as an empty result instead of
            #    propagating a KeyError. ─────────────────────────────────
            if "significant" not in res.columns:
                print(f"  [WARNING] submodel '{submodel}' returned a DataFrame "
                      f"without a 'significant' column (columns: {list(res.columns)}). "
                      f"Treating it as an empty result.")

                _save_time(
                    path=os.path.join(sm_folder, "time.txt"),
                    model_name=f"{model_name}/{submodel}",
                    module=pipeline.__module__,
                    start_dt=start_dt,
                    duration_s=duration_s,
                    status="empty (no significant column)",
                )

                res.to_csv(os.path.join(sm_folder, "results.csv"), index=False)
                _save_cascade_log(res, sm_folder)

                results_by_sub[submodel] = (pd.DataFrame(), (0, 0.0, 0.0))
                records.append({
                    'submodel':    submodel,
                    'status':       'empty',
                    'n_sig':        0,
                    'n_total':      len(res),
                    'mean_pvalue': 1.0,
                    'mean_strength':  0.0,
                    'duration_s':    round(duration_s, 4),
                    'dur_fmt':  dur_fmt,
                    'error':         "no 'significant' column in the pipeline's return value",
                })
                continue

            _save_time(
                path=os.path.join(sm_folder, "time.txt"),
                model_name=f"{model_name}/{submodel}",
                module=pipeline.__module__,
                start_dt=start_dt,
                duration_s=duration_s,
            )

            if res is not None and not res.empty:
                res_sig = res[res['significant'] == True]
                res_sig.to_csv(os.path.join(sm_folder, "results.csv"), index=False)
                _save_cascade_log(res, sm_folder)
                n_sig   = len(res_sig)
                n_total = len(res)
                mean_pval  = float(res_sig['pvalue'].mean()) if n_sig > 0 else 1.0
                forca_m = float(res_sig['strength'].mean())  if (n_sig > 0 and 'strength' in res_sig.columns) else 0.0
                score   = _score_submodel(res_sig)
                print(f"  [OK] n_sig={n_sig}/{n_total}  mean_pval={mean_pval:.4f}  mean_strength={forca_m:.4f}  time={dur_fmt}")
            else:
                res_sig = pd.DataFrame()
                n_sig, n_total, mean_pval, forca_m = 0, 0, 1.0, 0.0
                score   = (0, 0.0, 0.0)
                print(f"  [EMPTY] no result  time={dur_fmt}")

            results_by_sub[submodel] = (res_sig, score)
            records.append({
                'submodel':    submodel,
                'status':       'ok',
                'n_sig':        n_sig,
                'n_total':      n_total,
                'mean_pvalue': round(mean_pval, 6),
                'mean_strength':  round(forca_m, 6),
                'duration_s':    round(duration_s, 4),
                'dur_fmt':  dur_fmt,
                'error':         '',
            })

        except Exception as e:
            duration_s = time.perf_counter() - start
            dur_fmt   = _format_duration(duration_s)
            _save_time(
                path=os.path.join(sm_folder, "time.txt"),
                model_name=f"{model_name}/{submodel}",
                module=pipeline.__module__,
                start_dt=start_dt,
                duration_s=duration_s,
                status="ERROR",
                error=str(e),
            )
            print(f"  [ERROR] submodel '{submodel}' failed: {e}")
            traceback.print_exc()

            _log_pipeline_error(
                df_name=df_name,
                model_name=f"{model_name}__{submodel}",
                pipeline=pipeline,
                df=df,
                cfg=cfg_sub,
                exc=e,
            )

            results_by_sub[submodel] = (pd.DataFrame(), (0, 0.0, 0.0))
            records.append({
                'submodel':    submodel,
                'status':       'error',
                'n_sig':        0,
                'n_total':      0,
                'mean_pvalue': 1.0,
                'mean_strength':  0.0,
                'duration_s':    round(duration_s, 4),
                'dur_fmt':  dur_fmt,
                'error':         str(e),
            })

    best_name = max(
        results_by_sub,
        key=lambda s: results_by_sub[s][1],
        default=submodels[0] if submodels else 'auto',
    )
    best_results = results_by_sub.get(best_name, (pd.DataFrame(),))[0]

    print(f"\n  [CHOSEN SUBMODEL] {best_name}  "
          f"(n_sig={len(best_results) if not best_results.empty else 0})")

    _save_submodel_comparison(
        path=os.path.join(model_folder, "submodel_comparison.txt"),
        model_name=model_name,
        df_name=df_name,
        records=records,
        best_name=best_name,
    )
    print(f"  [OK] submodel_comparison.txt + .csv saved")

    return best_results, best_name, records


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all(dataframes: list, models: list, config: dict, output_dir: str):
    """
    Step 1: runs the pipelines and saves the results to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    run_start = time.perf_counter()
    summary: dict  = {}
    total_scenarios = len(dataframes)
    total_models  = len(models)
    total_tasks  = total_scenarios * total_models

    # Fixed footer pinned to the bottom of the terminal ("scenario X of N |
    # model Y of M"), visible at all times while the pipeline's normal log
    # scrolls above it -- becomes a silent no-op if stdout is not a real
    # terminal (redirected to a file, CI, etc.), see
    # Auxiliary/console_progress.py.
    progress = PinnedProgress(n_lines=2)
    progress.start()

    for i_df, df_entry in enumerate(dataframes, 1):
        df_name  = df_entry['name']
        df       = df_entry['df']
        gt       = df_entry.get('gt')
        scenario  = df_entry.get('_scenario')
        params   = df_entry.get('_params', {})

        df_folder = os.path.join(output_dir, df_name)
        os.makedirs(df_folder, exist_ok=True)
        summary[df_name] = {}

        # ── Saves the raw input DataFrame used for this case ──────
        source_path = os.path.join(df_folder, "source.csv")
        try:
            df.to_csv(source_path, index=False)
            print(f"  [SOURCE] source.csv saved ({len(df)} lines x {len(df.columns)} columns)")
        except Exception as e:
            print(f"  [WARNING] Could not save source.csv for '{df_name}': {e}")

        print("\n" + "#" * 65)
        print(f"  DATAFRAME: {df_name}  ({len(df)} lines x {len(df.columns)} columns)")
        print("#" * 65)

        if gt is not None:
            gt_path = os.path.join(df_folder, "ground_truth.csv")
            gt.to_csv(gt_path, index=False)
            n_causal = int(gt['causal_gt'].sum()) if 'causal_gt' in gt.columns else '?'
            print(f"  [GT] ground_truth.csv saved  ({n_causal} causal pairs)")

        if scenario is not None:
            params_path = os.path.join(df_folder, "scenario_params.txt")
            _save_scenario_params(params_path, scenario, params)
            print(f"  [GT] scenario_params.txt saved")

        for i_model, entry in enumerate(models, 1):
            model_name      = entry['name']
            pipeline         = entry['pipeline']
            override         = entry.get('config_override', {})
            dml_models_entry = entry.get('dml_models')
            required_properties   = entry.get('required_properties')
            dataset_filter   = entry.get('dataset_filter')
            cfg              = {**config, **override}

            # ── Fixed footer: overall process progress in real time ──
            # (current scenario/model + % done + elapsed time/ETA,
            # estimated from the average duration of the tasks done so far).
            tasks_done = (i_df - 1) * total_models + (i_model - 1)
            current_task       = tasks_done + 1
            elapsed_s          = time.perf_counter() - run_start
            pct                = (current_task / total_tasks * 100) if total_tasks else 0.0
            if tasks_done > 0:
                eta_fmt = _format_duration((elapsed_s / tasks_done) * (total_tasks - tasks_done))
            else:
                eta_fmt = "?"
            progress.update([
                f"[{pct:5.1f}%] scenario {i_df} of {total_scenarios} ({df_name})  "
                f"elapsed {_format_duration(elapsed_s)}  ETA {eta_fmt}",
                f"model {i_model} of {total_models} ({model_name})",
            ])

            # ── Optional per-model scenario scope: 'dataset_filter' is a list
            #    of substrings; the model only runs on scenarios whose name
            #    contains at least one of them (used to keep the slow GPDC
            #    baselines off the large scenarios). Absent = runs everywhere.
            if dataset_filter and not any(s in df_name for s in dataset_filter):
                print(f"\n  [SKIPPED] {model_name} on {df_name}: outside this model's "
                      f"dataset_filter.")
                summary[df_name][model_name] = {
                    'n_pairs': 0, 'duration': '0.00s',
                    'status': 'skipped (outside dataset_filter)',
                    'elected_submodel': '—',
                }
                continue

            # ── Gracefully skip if the dataset lacks the columns this model
            #    needs (fixed property names that only exist in a specific
            #    generator -- other benchmark datasets use different column
            #    names) ───────
            if required_properties:
                missing = [f for f in required_properties if f not in df.columns]
                if missing:
                    print("\n" + "=" * 65)
                    print(f"  PIPELINE : {model_name}  ({pipeline.__module__})")
                    print(f"  DATASET  : {df_name}")
                    print(f"  [SKIPPED] requires columns {required_properties}, but "
                          f"{missing} does not exist in this dataset. Available "
                          f"columns: {[c for c in df.columns if c not in ('Date', 'Name')]}")
                    print("=" * 65)
                    summary[df_name][model_name] = {
                        'n_pairs': 0, 'duration': '0.00s',
                        'status': f'skipped (missing columns {missing})',
                        'elected_submodel': '—',
                    }
                    continue

            model_folder = os.path.join(df_folder, model_name)
            os.makedirs(model_folder, exist_ok=True)

            print("\n" + "=" * 65)
            print(f"  PIPELINE : {model_name}  ({pipeline.__module__})")
            print(f"  DATASET  : {df_name}")
            if override:
                print(f"  Overrides: {override}")

            submodels = _resolve_submodel_list(pipeline, dml_models_entry)
            if submodels:
                print(f"  Submodels: {submodels}")
            print("=" * 65)

            start    = time.perf_counter()
            start_dt = datetime.datetime.now()

            if submodels:
                try:
                    best_results, best_name, records = _run_submodels(
                        pipeline=pipeline, df=df, cfg=cfg, submodels=submodels,
                        model_folder=model_folder, model_name=model_name, df_name=df_name,
                    )
                    duration_s = time.perf_counter() - start
                    dur_fmt   = _format_duration(duration_s)

                    _save_time(
                        path=os.path.join(model_folder, "time.txt"),
                        model_name=model_name, module=pipeline.__module__,
                        start_dt=start_dt, duration_s=duration_s,
                        status=f"ok (chosen: {best_name})",
                    )

                    if best_results is not None and not best_results.empty:
                        best_results.to_csv(
                            os.path.join(model_folder, "results.csv"), index=False
                        )
                        n_pairs = len(best_results)
                        print(f"  [OK] results.csv -- {n_pairs} pairs (submodel: {best_name})")
                        status_summary = f"ok via {best_name}"
                    else:
                        n_pairs = 0
                        print(f"  [WARNING] No result even after the submodels.")
                        status_summary = f"empty (chosen: {best_name})"

                    summary[df_name][model_name] = {
                        'n_pairs': n_pairs, 'duration': dur_fmt,
                        'status': status_summary, 'elected_submodel': best_name,
                    }

                except Exception as e:
                    duration_s = time.perf_counter() - start
                    dur_fmt   = _format_duration(duration_s)
                    _save_time(
                        path=os.path.join(model_folder, "time.txt"),
                        model_name=model_name, module=pipeline.__module__,
                        start_dt=start_dt, duration_s=duration_s,
                        status="ERROR", error=str(e),
                    )
                    print(f"  [ERROR] '{model_name}' failed after {dur_fmt}: {e}")
                    traceback.print_exc()

                    _log_pipeline_error(
                        df_name=df_name,
                        model_name=model_name,
                        pipeline=pipeline,
                        df=df,
                        cfg=cfg,
                        exc=e,
                    )

                    summary[df_name][model_name] = {
                        'n_pairs': 0, 'duration': dur_fmt,
                        'status': f'error: {e}', 'elected_submodel': '—',
                    }

            else:
                try:
                    results = pipeline(df, **cfg)

                    if results is None:
                        results = pd.DataFrame()

                    # ── GUARD: treats a DataFrame without the 'significant'
                    #    column as an empty result instead of letting
                    #    results["significant"] blow up with a KeyError
                    #    further below (every candidate pair discarded by
                    #    the pre-filters). ─────────────────────────
                    if "significant" not in results.columns:
                        print(f"  [WARNING] pipeline '{model_name}' returned "
                              f"a DataFrame without a 'significant' column "
                              f"(columns: {list(results.columns)}). Treating "
                              f"it as an empty result instead of propagating an error.")

                        original_path = os.path.join(model_folder, "original.csv")
                        _save_cascade_log(results, model_folder)
                        results = results.drop(columns=["cascade_log"])
                        results.to_csv(original_path, index=False)


                        duration_s = time.perf_counter() - start
                        dur_fmt   = _format_duration(duration_s)

                        _save_time(
                            path=os.path.join(model_folder, "time.txt"),
                            model_name=model_name, module=pipeline.__module__,
                            start_dt=start_dt, duration_s=duration_s,
                            status="empty (no significant column)",
                        )

                        summary[df_name][model_name] = {
                            'n_pairs': 0, 'duration': dur_fmt,
                            'status': 'empty (no significant column)',
                            'elected_submodel': '—',
                        }
                        continue

                    # Saves the full (unfiltered) result to original.csv
                    # BEFORE applying the "significant" filter.
                    original_path = os.path.join(model_folder, "original.csv")
                    results.to_csv(original_path, index=False)
                    _save_cascade_log(results, model_folder)

                    results   = results[results["significant"] == True]
                    duration_s = time.perf_counter() - start

                    _save_time(
                        path=os.path.join(model_folder, "time.txt"),
                        model_name=model_name, module=pipeline.__module__,
                        start_dt=start_dt, duration_s=duration_s,
                    )
                    dur_fmt = _format_duration(duration_s)

                    if results is None or results.empty:
                        print(f"  [WARNING] No result. ({dur_fmt})")
                        summary[df_name][model_name] = {
                            'n_pairs': 0, 'duration': dur_fmt,
                            'status': 'empty', 'elected_submodel': '—',
                        }

                        csv_path = os.path.join(model_folder, "results.csv")
                        results.to_csv(csv_path, index=False)

                        continue

                    csv_path = os.path.join(model_folder, "results.csv")
                    results.to_csv(csv_path, index=False)

                    print(f"  [OK] {model_folder}/")
                    print(f"       original.csv    -- {len(pd.read_csv(original_path))} pairs (unfiltered)")
                    print(f"       results.csv  -- {len(results)} pairs")
                    print(f"       time.txt       -- {dur_fmt}")

                    summary[df_name][model_name] = {
                        'n_pairs': len(results), 'duration': dur_fmt,
                        'status': 'ok', 'elected_submodel': '—',
                    }

                except Exception as e:
                    duration_s = time.perf_counter() - start
                    dur_fmt   = _format_duration(duration_s)
                    _save_time(
                        path=os.path.join(model_folder, "time.txt"),
                        model_name=model_name, module=pipeline.__module__,
                        start_dt=start_dt, duration_s=duration_s,
                        status="ERROR", error=str(e),
                    )
                    print(f"  [ERROR] '{model_name}' failed after {dur_fmt}: {e}")
                    traceback.print_exc()

                    _log_pipeline_error(
                        df_name=df_name,
                        model_name=model_name,
                        pipeline=pipeline,
                        df=df,
                        cfg=cfg,
                        exc=e,
                    )

                    summary[df_name][model_name] = {
                        'n_pairs': 0, 'duration': dur_fmt,
                        'status': f'error: {e}', 'elected_submodel': '—',
                    }

    progress.stop()

    total_fmt = _format_duration(time.perf_counter() - run_start)
    print("\n" + "#" * 65)
    print("  RUNNER SUMMARY")
    print("#" * 65)
    for df_name, models_res in summary.items():
        print(f"\n  [{df_name}]")
        for model_name, r in models_res.items():
            chosen = f"  submodel={r['elected_submodel']}" if r['elected_submodel'] != '—' else ""
            print(f"    {model_name:<25} pairs={r['n_pairs']:>4}  "
                  f"time={r['duration']:<15}  [{r['status']}]{chosen}")
    print(f"\n  Total runner time: {total_fmt}")

    if LOGS_DIR.exists() and any(LOGS_DIR.iterdir()):
        print(f"\n  [LOGS] Error diagnostics available at: {LOGS_DIR}")


# =============================================================================
# VALIDATOR UTILITIES
# =============================================================================

def _to_bool(col: pd.Series) -> pd.Series:
    """
    Robust conversion of a 'causal_gt' column to boolean.

    `column.astype(bool)` in pandas, when the column is text (object/str
    dtype), treats ANY non-empty string as True -- including "False",
    "No", "0" as text, etc. This function normalizes the text before
    deciding the boolean instead, so it works both for native bool and for
    common textual variants (True/False, Yes/No, 1/0, t/f).
    """
    if col.dtype == bool:
        return col
    return col.astype(str).str.strip().str.lower().isin(
        ["true", "1", "1.0", "yes", "t", "y"]
    )


def _truthy(v) -> bool:
    """Scalar version of _to_bool, for use in r[col_c] inside loops/comprehensions."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ["true", "1", "1.0", "yes", "t", "y"]


def _pair_to_ab(pair) -> tuple:
    """
    Unpacks a pair (frozenset of 1 or 2 elements, OR an ordered tuple) into (a, b).

    frozenset([a, b]) collapses to 1 element when a == b -- which happens
    for INTRA-ENTITY pairs (same entity, different properties; see the
    'quadratic'/'threshold'/'saturation' scenarios of the tigramite
    benchmark). This handles that case
    explicitly rather than assuming 2 distinct elements.

    Compatible with directional pairs (use_canonical_pair=False): in that
    case `pair` is already an ordered (a, b) tuple, returned unchanged.
    """
    if isinstance(pair, tuple):
        if len(pair) == 1:
            return pair[0], pair[0]
        return pair[0], pair[1]
    elems = sorted(pair)
    if len(elems) == 1:
        return elems[0], elems[0]
    return elems[0], elems[1]


def _read_gt(dataset_folder: Path) -> Optional[pd.DataFrame]:
    gt_path = dataset_folder / "ground_truth.csv"
    if not gt_path.exists():
        return None
    return pd.read_csv(gt_path, skipinitialspace=True)


def _filter_intra_entity(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Removes lines where source_entity == target_entity (INTRA-ENTITY pairs --
    same entity, potentially different properties; e.g. F_0 -> F_1 within the
    same entity in the 'quadratic'/'threshold'/'saturation' scenarios of the
    tigramite benchmark).

    Used by the DETECT_INTRA_ENTITY / ignore_intra_entity flag in
    validate_dataset(): when False, these pairs are removed from both the
    ground truth and the predictions BEFORE any metric is computed -- i.e.,
    the pair stops existing for comparison purposes (it does not count as
    TP, FP, FN, nor TN). This is different from changing what the pipeline
    itself tries to detect (controlled by block_same_property in
    Pipeline/discovery.py and Pipeline/pipeline.py); this only controls
    what ENTERS THE VALIDATION SCORE, retroactively (it works even on
    datasets already run, without needing to re-run the runner).

    Works on any DataFrame that has the source_entity and target_entity
    columns (ground_truth.csv, results.csv, original.csv). Returns the df
    unchanged if the columns do not exist or if it is empty.
    """
    if df is None or df.empty:
        return df
    if "source_entity" not in df.columns or "target_entity" not in df.columns:
        return df
    mask = (
        df["source_entity"].astype(str).str.strip()
        != df["target_entity"].astype(str).str.strip()
    )
    return df.loc[mask].reset_index(drop=True)


def _resolve_gt_columns(gt: pd.DataFrame) -> dict:
    cols = gt.columns.tolist()
    for col in ["source_entity", "target_entity"]:
        if col not in cols:
            raise KeyError(
                f"Required column missing from the GT: '{col}'\n"
                f"Columns found: {cols}"
            )
    result = {
        "source":  "source_entity",
        "target": "target_entity",
        "causal":  "causal_gt" if "causal_gt" in cols else None,
        "group":   "group_gt"  if "group_gt"  in cols else None,
    }
    msg = "  [GT] source_entity + target_entity"
    if result["causal"]:
        msg += " | causal_gt present -- filtering by True/False"
    else:
        msg += " | no causal_gt -- all GT pairs treated as causal"
    if result["group"]:
        msg += " | group_gt present"
    print(msg)
    return result


def _causal_gt_pairs(gt: pd.DataFrame, cols: dict,
                       use_canonical_pair: bool = USE_CANONICAL_PAIR_DEFAULT) -> set:
    if cols["causal"]:
        lines = gt[_to_bool(gt[cols["causal"]])]
    else:
        lines = gt
    return {
        _pair_key(str(row[cols["source"]]).strip(), str(row[cols["target"]]).strip(),
                   use_canonical_pair)
        for _, row in lines.iterrows()
    }


def _gt_entities(gt: pd.DataFrame, cols: dict) -> set:
    return (
        set(gt[cols["source"]].astype(str).str.strip()) |
        set(gt[cols["target"]].astype(str).str.strip())
    )


def _predicted_pairs(results: pd.DataFrame,
                     use_canonical_pair: bool = USE_CANONICAL_PAIR_DEFAULT) -> set:
    return {
        _pair_key(str(row["source_entity"]).strip(), str(row["target_entity"]).strip(),
                   use_canonical_pair)
        for _, row in results.iterrows()
    }


def _with_canonical_pair(df: pd.DataFrame,
                       use_canonical_pair: bool = USE_CANONICAL_PAIR_DEFAULT) -> pd.DataFrame:
    """
    Returns a copy of df with an auxiliary '_canonical_pair' column (pair
    key -- frozenset if use_canonical_pair=True, ordered tuple otherwise),
    used to quickly locate every line belonging to a given entity pair
    (regardless of property, or of direction when use_canonical_pair=True).
    Used when exporting errors (FP/FN) per model.
    """
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        out["_canonical_pair"] = pd.Series(dtype=object)
        return out
    out = df.copy()
    out["_canonical_pair"] = [
        _pair_key(str(row["source_entity"]).strip(), str(row["target_entity"]).strip(),
                   use_canonical_pair)
        for _, row in out.iterrows()
    ]
    return out


def _rows_for_pair(df_with_pair: Optional[pd.DataFrame], pair,
                    key_col: str = "_canonical_pair") -> pd.DataFrame:
    """
    Returns the lines of df_with_pair whose pair (entity or property,
    depending on key_col) equals `pair`. Removes every internal auxiliary
    column (_par_*).
    key_col: "_canonical_pair" (entity level) or "_property_pair_key" (property level).
    """
    if df_with_pair is None or df_with_pair.empty or key_col not in df_with_pair.columns:
        return pd.DataFrame()
    mask = df_with_pair[key_col] == pair
    drop_cols = [c for c in df_with_pair.columns if c.startswith("_par_")]
    return df_with_pair.loc[mask].drop(columns=drop_cols, errors="ignore")


def _export_model_errors(
    model_folder,
    model_name: str,
    fp_pairs: set,
    fn_pairs: set,
    results_with_pair: Optional[pd.DataFrame],
    original_with_pair: Optional[pd.DataFrame],
    key_col: str = "_canonical_pair",
):
    """
    For a model (in a specific scenario/dataset), saves inside
    `model_folder` two files with the FULL LINE (every column of the
    model's output) for each error:

        fp_errors.csv -- line(s) from results.csv for every pair the model
                       predicted as significant (FP), but which is not
                       causal in the ground truth.
        fn_errors.csv -- line(s) from original.csv (the UNFILTERED output)
                       for every causal GT pair the model did NOT predict
                       as significant (FN). This allows inspecting the
                       pvalue/strength even below the significance
                       threshold. If the pair never appears in original.csv
                       (the model never tested it), a placeholder line is
                       written with source_entity/target_entity filled in
                       and a '_note' column explaining why.
    """
    # ── FP: full line(s) coming from results.csv ───────────────────
    fp_blocks = []
    for pair in fp_pairs:
        lines = _rows_for_pair(results_with_pair, pair, key_col=key_col)
        if lines.empty:
            a, b = _pair_to_ab(pair)
            lines = pd.DataFrame([{
                "source_entity":  a,
                "target_entity": b,
                "_note": "pair not found in results.csv (unexpected)",
            }])
        fp_blocks.append(lines)

    df_fp = (pd.concat(fp_blocks, ignore_index=True) if fp_blocks
              else pd.DataFrame(columns=["source_entity", "target_entity"]))
    df_fp.to_csv(os.path.join(model_folder, "fp_errors.csv"), index=False)

    # ── FN: full line(s) coming from original.csv (unfiltered) ────────
    fn_blocks = []
    for pair in fn_pairs:
        lines = _rows_for_pair(original_with_pair, pair, key_col=key_col)
        if lines.empty:
            a, b = _pair_to_ab(pair)
            lines = pd.DataFrame([{
                "source_entity":  a,
                "target_entity": b,
                "_note": "pair not found in original.csv (the model did not test this pair)",
            }])
        fn_blocks.append(lines)

    df_fn = (pd.concat(fn_blocks, ignore_index=True) if fn_blocks
              else pd.DataFrame(columns=["source_entity", "target_entity"]))
    df_fn.to_csv(os.path.join(model_folder, "fn_errors.csv"), index=False)

    print(
        f"  [ERRORS] {model_name}: fp_errors.csv ({len(df_fp)} lines, {len(fp_pairs)} pairs) "
        f"+ fn_errors.csv ({len(df_fn)} lines, {len(fn_pairs)} pairs) saved to {model_folder}/"
    )


def _universe(causal_gt: set, gt: Optional[pd.DataFrame] = None,
              cols: Optional[dict] = None,
              use_canonical_pair: bool = USE_CANONICAL_PAIR_DEFAULT) -> set:
    """
    Comparison universe.

    If the GT contains the causal_gt column: the universe is the FULL
    Cartesian product of the GT (every pair, causal and non-causal),
    enabling real TN.

    If causal_gt is absent: universe == causal_gt (TN = 0 always), since
    there is no confirmation of absence of a connection for the remaining
    pairs.
    """
    if gt is not None and cols is not None and cols.get("causal") is not None:
        col_source = cols["source"]
        col_target = cols["target"]
        return {
            _pair_key(str(r[col_source]).strip(), str(r[col_target]).strip(), use_canonical_pair)
            for _, r in gt.iterrows()
        }
    return set(causal_gt)


def _read_time(model_folder: Path) -> dict:
    time_path = model_folder / "time.txt"
    info = {"duration": "N/A", "duration_s": None, "status": "N/A"}
    if not time_path.exists():
        return info
    for line in time_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "duration":
            info["duration"] = value
        elif key == "duration_s":
            try:
                info["duration_s"] = float(value)
            except ValueError:
                pass
        elif key == "status":
            info["status"] = value
    return info


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _compute_metrics(predicted: set, causal_gt: set, universe: set) -> dict:
    TP = len(causal_gt & predicted)
    FN = len(causal_gt - predicted)
    FP = len(predicted - causal_gt)
    TN = len(universe - causal_gt - predicted)

    precision = _safe_div(TP, TP + FP)
    recall    = _safe_div(TP, TP + FN)
    f1        = _safe_div(2 * precision * recall, precision + recall)
    accuracy  = _safe_div(TP + TN, TP + FP + FN + TN)
    fpr       = _safe_div(FP, FP + TN)

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy, "fpr": fpr,
        "n_predicted": len(predicted),
        "n_causal_gt": len(causal_gt),
        "n_universe": len(universe),
    }


def _compute_soft_metrics(
    predicted: set,
    causal_gt: set,
    universe: set,
    global_tp_soft: set,
) -> dict:
    soft_predicted = predicted
    raw_fp = soft_predicted - causal_gt

    TP_hard = len(causal_gt & soft_predicted)
    TP_soft = len(raw_fp & global_tp_soft)
    FP_hard = len(raw_fp - global_tp_soft)
    FN      = len(causal_gt - soft_predicted)
    TN      = len(universe - causal_gt - soft_predicted)

    TP_total = TP_hard + TP_soft

    precision_soft = _safe_div(TP_total, TP_total + FP_hard)
    recall_soft    = _safe_div(TP_total, TP_total + FN)
    f1_soft        = _safe_div(2 * precision_soft * recall_soft,
                               precision_soft + recall_soft)

    return {
        "TP_hard":        TP_hard,
        "TP_soft":        TP_soft,
        "FP_hard":        FP_hard,
        "FN":             FN,
        "TN":             TN,
        "precision_soft": precision_soft,
        "recall_soft":    recall_soft,
        "f1_soft":        f1_soft,
    }


def _property_pair(row, use_canonical_pair: bool = USE_CANONICAL_PAIR_DEFAULT):
    """Canonical (entity, property) pair -- bidirectional if use_canonical_pair=True
    (A->B and B->A are the same physical relationship); directional (ordered
    tuple) if use_canonical_pair=False.
    """
    a = f"{str(row['source_entity']).strip()}|{str(row.get('source_property','')).strip()}"
    b = f"{str(row['target_entity']).strip()}|{str(row.get('target_property','')).strip()}"
    return _pair_key(a, b, use_canonical_pair)


def _predicted_pairs_property(results: pd.DataFrame,
                           use_canonical_pair: bool = USE_CANONICAL_PAIR_DEFAULT) -> set:
    """Predicted pairs at the (entity, property) level."""
    if "source_property" not in results.columns or "target_property" not in results.columns:
        return {_pair_key(str(r["source_entity"]).strip(),
                           str(r["target_entity"]).strip(), use_canonical_pair)
                for _, r in results.iterrows()}
    return {_property_pair(r, use_canonical_pair) for _, r in results.iterrows()}


def _compute_property_metrics(predicted_f: set, causal_gt_f: set,
                               universe_f: set) -> dict:
    """Hard metrics at the (entity, property) level."""
    TP = len(causal_gt_f & predicted_f)
    FN = len(causal_gt_f - predicted_f)
    FP = len(predicted_f - causal_gt_f)
    TN = len(universe_f - causal_gt_f - predicted_f)

    precision = _safe_div(TP, TP + FP)
    recall    = _safe_div(TP, TP + FN)
    f1        = _safe_div(2 * precision * recall, precision + recall)

    return {
        "f_TP": TP, "f_FP": FP, "f_FN": FN, "f_TN": TN,
        "f_precision": precision, "f_recall": recall, "f_f1": f1,
        "f_n_predicted": len(predicted_f),
    }


def _compute_consolidated_soft(predicted: set, causal_gt: set, universe: set,
                                global_tp_soft: set,
                                predicted_f: set, causal_gt_f: set,
                                universe_f: set) -> dict:
    """
    Table 2 -- consolidated soft (entity level):
        TP  = TP_hard + TP_soft  (summed as 1 TP)
        FP  = FP_hard
        FN, TN same as the entity-level soft table
        + n_unique_model (unique entity pairs detected by the model)
        + n_unique_gt     (unique entity pairs in the GT)
    """
    predicted_s = predicted
    raw_fp  = predicted_s - causal_gt

    TP  = len(causal_gt & predicted_s) + len(raw_fp & global_tp_soft)
    FP  = len(raw_fp - global_tp_soft)
    FN  = len(causal_gt - predicted_s)
    TN  = len(universe - causal_gt - predicted_s)

    precision = _safe_div(TP, TP + FP)
    recall    = _safe_div(TP, TP + FN)
    f1        = _safe_div(2 * precision * recall, precision + recall)

    # unique entity pairs detected by the model (ignoring property)
    n_unique_model = len(predicted)          # already a set of pair keys
    n_unique_gt     = len(causal_gt)

    return {
        "sc_TP": TP, "sc_FP": FP, "sc_FN": FN, "sc_TN": TN,
        "sc_precision": precision, "sc_recall": recall, "sc_f1": f1,
        "sc_n_unique_model": n_unique_model,
        "sc_n_unique_gt":     n_unique_gt,
    }


def _compute_soft_v2(predicted_f: set, causal_gt_f: set, universe_f: set,
                       global_tp_soft_f: set) -> dict:
    """
    Table 3 -- soft v2 (property level): the soft logic (global_tp_soft_f)
    applied to (entity, property) pairs.
    """
    raw_fp  = predicted_f - causal_gt_f

    TP_hard = len(causal_gt_f & predicted_f)
    TP_soft = len(raw_fp & global_tp_soft_f)
    FP      = len(raw_fp - global_tp_soft_f)
    FN      = len(causal_gt_f - predicted_f)
    TN      = len(universe_f - causal_gt_f - predicted_f)

    TP_total  = TP_hard + TP_soft
    precision = _safe_div(TP_total, TP_total + FP)
    recall    = _safe_div(TP_total, TP_total + FN)
    f1        = _safe_div(2 * precision * recall, precision + recall)

    return {
        "sv2_TP_hard": TP_hard, "sv2_TP_soft": TP_soft,
        "sv2_FP": FP, "sv2_FN": FN, "sv2_TN": TN,
        "sv2_precision": precision, "sv2_recall": recall, "sv2_f1": f1,
        "sv2_n_predicted_f": len(predicted_f),
    }


def _build_tp_soft_property(predicted_f_by_model: dict, causal_gt_f: set,
                               n_valid_models: int, threshold: float) -> set:
    """tp_soft set at the property level."""
    if n_valid_models == 0:
        return set()
    count: dict = {}
    for name, predicted_f in predicted_f_by_model.items():
        for pair in predicted_f:
            if pair not in causal_gt_f:
                count[pair] = count.get(pair, 0) + 1
    minimum = threshold * n_valid_models
    return {pair for pair, cnt in count.items() if cnt >= minimum}


def _build_tp_soft(
    predicted_by_model: dict,
    causal_gt: set,
    n_valid_models: int,
    threshold: float,
) -> set:
    if n_valid_models == 0:
        return set()

    count: dict = {}
    for model_name, predicted in predicted_by_model.items():
        for pair in predicted:
            if pair not in causal_gt:
                count[pair] = count.get(pair, 0) + 1

    minimum = threshold * n_valid_models
    return {pair for pair, cnt in count.items() if cnt >= minimum}


def _save_validation_summary(
    dataset_folder: Path,
    dataset_name: str,
    df_summary: pd.DataFrame,
    global_tp_soft: set,
    soft_threshold: float,
    gt: Optional[pd.DataFrame] = None,
    gt_cols: Optional[dict] = None,
    use_canonical_pair: bool = USE_CANONICAL_PAIR_DEFAULT,
):
    metric_cols = [
        "model", "status",
        "precision", "recall", "f1", "accuracy", "fpr",
        "TP", "FP", "FN", "TN",
        "TP_hard", "TP_soft", "FP_hard", "FN_soft", "TN_soft",
        "precision_soft", "recall_soft", "f1_soft",
        "soft_threshold",
        "n_predicted", "n_tested", "n_causal", "n_universe", "n_outside_universe", "duration",
    ]

    csv_path = dataset_folder / "validation_summary.csv"
    df_summary[[c for c in metric_cols if c in df_summary.columns]].to_csv(
        csv_path, index=False
    )

    txt_path = dataset_folder / "validation_summary.txt"
    SEP  = "=" * 72
    SEP2 = "-" * 72

    df_ok = df_summary[df_summary["status"] == "ok"].copy()

    # ── Extracts information from the GT (entities, unique relations, real TN) ────────────
    has_causal_gt  = (gt is not None and gt_cols is not None
                      and gt_cols.get("causal") is not None)

    n_entities_gt = 0
    n_unique_relations_gt = n_causal_relations_gt = n_noncausal_relations_gt = 0
    n_unique_entity_pairs = n_real_universe = 0

    if gt is not None and gt_cols is not None:
        col_source = gt_cols["source"]
        col_target = gt_cols["target"]

        sources_gt = set(gt[col_source].astype(str).str.strip().unique())
        targets_gt = set(gt[col_target].astype(str).str.strip().unique())
        n_entities_gt = len(sources_gt | targets_gt)

        # unique relations at the property level: (entity_a, property_x) -> (entity_b, property_y)
        # where (a,b) and (b,a) count as 1 relation IF use_canonical_pair=True;
        # otherwise, (a,b) and (b,a) count as 2 distinct relations.
        col_fsource = "source_property" if "source_property" in gt.columns else None
        col_ftarget = "target_property" if "target_property" in gt.columns else None

        if col_fsource and col_ftarget:
            unique_property_pairs = {
                _pair_key(
                    f"{str(r[col_source]).strip()}|{str(r[col_fsource]).strip()}",
                    f"{str(r[col_target]).strip()}|{str(r[col_ftarget]).strip()}",
                    use_canonical_pair,
                )
                for _, r in gt.iterrows()
            }
            n_unique_relations_gt = len(unique_property_pairs)
        else:
            n_unique_relations_gt = len({
                _pair_key(str(r[col_source]).strip(), str(r[col_target]).strip(), use_canonical_pair)
                for _, r in gt.iterrows()
            })

        # unique entity pairs, ignoring properties
        unique_entity_pairs = {
            _pair_key(str(r[col_source]).strip(), str(r[col_target]).strip(), use_canonical_pair)
            for _, r in gt.iterrows()
        }
        n_unique_entity_pairs = len(unique_entity_pairs)

        if has_causal_gt:
            # NOTE: the totals below need to be computed over the SAME
            # deduplicated property-level set (unique_property_pairs) that the
            # hard-metric tables use as the universe (universe_f in
            # validate_dataset).
            col_c = gt_cols["causal"]
            if col_fsource and col_ftarget:
                causal_by_pair = {}
                for _, r in gt.iterrows():
                    pair = _pair_key(
                        f"{str(r[col_source]).strip()}|{str(r[col_fsource]).strip()}",
                        f"{str(r[col_target]).strip()}|{str(r[col_ftarget]).strip()}",
                        use_canonical_pair,
                    )
                    is_causal = bool(_to_bool(pd.Series([r[col_c]])).iloc[0])
                    causal_by_pair[pair] = causal_by_pair.get(pair, False) or is_causal
            else:
                causal_by_pair = {}
                for _, r in gt.iterrows():
                    pair = _pair_key(str(r[col_source]).strip(), str(r[col_target]).strip(),
                                      use_canonical_pair)
                    is_causal = bool(_to_bool(pd.Series([r[col_c]])).iloc[0])
                    causal_by_pair[pair] = causal_by_pair.get(pair, False) or is_causal

            n_causal_relations_gt     = sum(1 for v in causal_by_pair.values() if v)
            n_noncausal_relations_gt = sum(1 for v in causal_by_pair.values() if not v)
            n_real_universe = len(causal_by_pair)

    lines = []
    lines.append(SEP)
    lines.append(f"  VALIDATION SUMMARY -- {dataset_name}")
    lines.append(f"  Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Pairing              : {'CANONICAL (non-directional)' if use_canonical_pair else 'DIRECTIONAL (A->B != B->A)'}")
    lines.append(SEP)

    # ── GT information block ───────────────────────────────────────────────
    if gt is not None and gt_cols is not None:
        lines.append(f"  GROUND TRUTH")
        lines.append(f"  {'─'*40}")
        lines.append(f"  Entities tested (total)        : {n_entities_gt}")
        dir_label = "(a,b) and (b,a) = 1 relation" if use_canonical_pair else "(a,b) and (b,a) = 2 distinct relations"
        lines.append(f"  Unique relations in the GT : {n_unique_relations_gt}  "
                      f"  [{dir_label}, property level]")
        lines.append(f"  Unique entity pairs        : {n_unique_entity_pairs}  "
                      f"  [{dir_label}, properties ignored]")
        if has_causal_gt:
            lines.append(f"  Causal relations (GT)      : {n_causal_relations_gt}")
            lines.append(f"  Non-causal relations (GT)  : {n_noncausal_relations_gt}"
                          f"  [true negatives available = {n_noncausal_relations_gt}]")
            lines.append(f"  Real universe (total GT)   : {n_real_universe} pairs"
                          f"  [= unique relations in the GT; TP+FP+FN+TN of the table below add up here]")
        else:
            lines.append(f"  causal_gt                  : absent -- TN not computable")
        lines.append("")

    lines.append(f"  Models evaluated  : {len(df_summary)}")
    lines.append(f"  With OK results   : {len(df_ok)}")
    if has_causal_gt:
        lines.append(f"  TN                : computed since causal_gt is present in the GT")
    else:
        lines.append(f"  TN                : always 0 (causal_gt absent; universe == causal_gt)")
    if "f_f1" in df_ok.columns:
        f_f1_numeric = pd.to_numeric(df_ok["f_f1"], errors="coerce")
        if f_f1_numeric.notna().any():
            best = df_ok.loc[f_f1_numeric.idxmax()]
            lines.append(f"  Best F1           : {best['model']}  (F1={best['f_f1']:.4f})")
    lines.append("")

    col_w = 22

    # ── TABLE 1 -- Hard, property level (entity x property) ─────────────────────
    if "f_f1" in df_summary.columns and df_summary["f_f1"].replace("", pd.NA).notna().any():
        lines.append(SEP2)
        title_t1 = ("METRICS -- PROPERTY LEVEL  (entity,property)<->(entity,property), bidirectional"
                     if use_canonical_pair else
                     "METRICS -- PROPERTY LEVEL  (entity,property)->(entity,property), DIRECTIONAL")
        lines.append(f"  {title_t1}")
        if use_canonical_pair:
            lines.append("  [(a,b)=(b,a): physical connection between entities with no preferred direction]")
        else:
            lines.append("  [(a,b) != (b,a): the direction of the relationship matters for scoring]")
        lines.append(SEP2)
        header_f = (
            f"  {'MODEL':<{col_w}} {'PREC':>7} {'REC':>7} {'F1':>7} "
            f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'PRED_F':>7}"
        )
        lines.append(header_f)
        lines.append("  " + "-" * 70)
        for _, row in df_summary.iterrows():
            if row["status"] != "ok" or row.get("f_f1") == "":
                lines.append(f"  {row['model']:<{col_w}} {'---':>7} {'---':>7} {'---':>7} "
                              f"{'---':>5} {'---':>5} {'---':>5} {'---':>5} {'---':>7}  [{row['status']}]")
            else:
                lines.append(
                    f"  {row['model']:<{col_w}} "
                    f"{row['f_precision']:>7.4f} {row['f_recall']:>7.4f} {row['f_f1']:>7.4f} "
                    f"{int(row['f_TP']):>5} {int(row['f_FP']):>5} "
                    f"{int(row['f_FN']):>5} {int(row['f_TN']):>5} "
                    f"{int(row['f_n_predicted']):>7}"
                )
        lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [SUMMARY] validation_summary.csv + .txt saved to {dataset_folder.name}/")


# =============================================================================
# SINGLE-DATASET VALIDATOR
# =============================================================================

def validate_dataset(
    dataset_name: str,
    output_dir: str,
    show_pairs: bool = False,
    soft_threshold: float = 1.0,
    ignore_intra_entity: bool = False,
    use_canonical_pair: bool = USE_CANONICAL_PAIR_DEFAULT,
):
    """
    ignore_intra_entity : bool
        If True, INTRA-ENTITY pairs (source_entity == target_entity -- same
        entity, potentially different properties) are removed from the
        ground truth AND from each model's predictions BEFORE any metric is
        computed. These pairs stop existing for TP/FP/FN/TN purposes -- they
        are neither rewarded nor penalized.

        Use True when the evaluated pipeline was not designed to detect
        relationships within the same entity (e.g. tigramite benchmark
        scenarios with more than one property per entity, such as
        'quadratic', 'threshold', 'saturation'). Use False (default) to
        count these pairs normally.

        Works retroactively on datasets already run (reads ground_truth.csv
        and results.csv/original.csv from disk and filters before
        comparing), without needing to re-run the runner.

    use_canonical_pair : bool
        True  (default) -- A->B and B->A are the SAME pair for comparison
              purposes (frozenset). Correct for a dataset with fixed,
              disjoint entity roles.
        False -- A->B and B->A are DISTINCT pairs (ordered tuple); the model
              only gets it right if it detects the correct direction of the
              relationship. Recommended for benchmarks where the variables
              have no fixed role (Lorenz-96, logistic maps, tigramite
              scenarios).
        Affects the entire metric calculation (hard, soft, property level
        and entity level) and the export of fp_errors.csv/fn_errors.csv.
        Works retroactively on datasets already run, without needing to
        re-run the runner.
    """
    dataset_folder = Path(output_dir) / dataset_name

    if not dataset_folder.exists():
        print(f"[ERROR] Dataset '{dataset_name}' not found in '{output_dir}'.")
        return None

    gt = _read_gt(dataset_folder)
    if gt is None:
        print(f"[WARNING] '{dataset_name}' has no ground_truth.csv -- skipping.")
        return None

    if ignore_intra_entity:
        n_gt_before = len(gt)
        gt = _filter_intra_entity(gt)
        n_gt_removed = n_gt_before - len(gt)
        if n_gt_removed > 0:
            print(f"  [INTRA-ENTITY] {n_gt_removed} line(s) of "
                  f"ground_truth.csv removed (source_entity == target_entity) "
                  f"-- ignored in the score.")

    try:
        cols = _resolve_gt_columns(gt)
    except KeyError as e:
        print(f"[ERROR] {e}")
        return None

    causal_gt = _causal_gt_pairs(gt, cols, use_canonical_pair)
    gt_entities   = _gt_entities(gt, cols)
    universe   = _universe(causal_gt, gt=gt, cols=cols, use_canonical_pair=use_canonical_pair)

    model_dirs = sorted([
        d for d in dataset_folder.iterdir()
        if d.is_dir() and (d / "results.csv").exists()
    ])

    if not model_dirs:
        print(f"[WARNING] '{dataset_name}' has no models with results.csv -- skipping.")
        return None

    has_causal_gt_ds = cols.get("causal") is not None
    SEP  = "=" * 72
    SEP2 = "-" * 72
    pct_label = f"{soft_threshold*100:.0f}%"

    print(f"\n{SEP}")
    print(f"  VALIDATION -- {dataset_name}")
    print(SEP)
    print(f"  Pairing: {'CANONICAL (non-directional)' if use_canonical_pair else 'DIRECTIONAL'}")
    print(f"  GT: {len(causal_gt)} causal pairs | {len(gt_entities)} entities")
    if has_causal_gt_ds:
        print(f"  Comparison universe: {len(universe)} pairs (GT Cartesian product; real TN)")
    else:
        print(f"  Comparison universe: {len(universe)} pairs (== causal pairs; TN=0 always)")
    print(f"  Soft threshold: {pct_label} of the valid models")
    if ignore_intra_entity:
        print(f"  Intra-entity pairs (source_entity == target_entity): IGNORED")
        print(f"                 (removed from the GT and the predictions before")
        print(f"                 any metric is computed -- they do not count")
        print(f"                 as TP, FP, FN, nor TN).")
    else:
        print(f"  Intra-entity pairs (source_entity == target_entity): counted normally")

    predicted_by_model: dict = {}
    predicted_f_by_model: dict = {}    # property level
    results_with_pair_by_model: dict = {}    # for exporting FP errors (full line)
    original_with_pair_by_model:   dict = {}    # for exporting FN errors (full line)
    tested_by_model: dict = {}      # unique pairs tested (original.csv, unfiltered)
    time_by_model:    dict = {}
    status_by_model:   dict = {}

    for model_folder in model_dirs:
        model_name = model_folder.name
        time_info  = _read_time(model_folder)
        time_by_model[model_name] = time_info

        if time_info["status"].upper() == "ERROR":
            status_by_model[model_name] = "error"
            continue

        results = pd.read_csv(model_folder / "results.csv", skipinitialspace=True)
        if ignore_intra_entity:
            results = _filter_intra_entity(results)
        if results.empty:
                status_by_model[model_name] = "empty"

                # empty sets
                predicted_by_model[model_name] = set()
                predicted_f_by_model[model_name] = set()

                results_with_pair_by_model[model_name] = pd.DataFrame()

                try:
                    original_df = pd.read_csv(
                        model_folder / "original.csv",
                        skipinitialspace=True,
                    )

                    if ignore_intra_entity:
                        original_df = _filter_intra_entity(original_df)

                except Exception:
                    original_df = pd.DataFrame()

                _ocp = _with_canonical_pair(original_df, use_canonical_pair)

                if (
                    not original_df.empty
                    and "source_property" in original_df.columns
                    and "target_property" in original_df.columns
                ):
                    _ocp["_property_pair_key"] = [
                        _property_pair(r, use_canonical_pair)
                        for _, r in original_df.iterrows()
                    ]

                original_with_pair_by_model[model_name] = _ocp
                tested_by_model[model_name] = set()

        status_by_model[model_name] = "ok"
        predicted_by_model[model_name]   = _predicted_pairs(results, use_canonical_pair)
        predicted_f_by_model[model_name] = _predicted_pairs_property(results, use_canonical_pair)

        _rcp = _with_canonical_pair(results, use_canonical_pair)
        if "source_property" in results.columns and "target_property" in results.columns:
            _rcp["_property_pair_key"] = [_property_pair(r, use_canonical_pair) for _, r in results.iterrows()]
        results_with_pair_by_model[model_name] = _rcp
        try:
            original_df = pd.read_csv(model_folder / "original.csv", skipinitialspace=True)
            if ignore_intra_entity:
                original_df = _filter_intra_entity(original_df)
        except Exception:
            original_df = pd.DataFrame()
        _ocp = _with_canonical_pair(original_df, use_canonical_pair)
        if not original_df.empty and "source_property" in original_df.columns \
                and "target_property" in original_df.columns:
            _ocp["_property_pair_key"] = [_property_pair(r, use_canonical_pair) for _, r in original_df.iterrows()]
        original_with_pair_by_model[model_name] = _ocp
        tested_by_model[model_name] = _predicted_pairs(original_df, use_canonical_pair)

    n_valid      = len(predicted_by_model)
    global_tp_soft = _build_tp_soft(
        predicted_by_model, causal_gt, n_valid, soft_threshold,
    )

    # ── Property-level sets for tables 1 and 3 ───────────────────────────────
    has_property_gt = (gt is not None and cols is not None
                     and "source_property" in gt.columns
                     and "target_property" in gt.columns)

    if has_property_gt:
        col_source_f = cols["source"]
        col_target_f = cols["target"]
        col_c      = cols.get("causal")

        causal_gt_f = {
            _property_pair(r, use_canonical_pair) for _, r in gt.iterrows()
            if (col_c is None or _truthy(r[col_c]))
        }
        universe_f = {_property_pair(r, use_canonical_pair) for _, r in gt.iterrows()}
    else:
        causal_gt_f = set()
        universe_f   = set()

    global_tp_soft_f = _build_tp_soft_property(
        predicted_f_by_model, causal_gt_f, n_valid, soft_threshold,
    ) if has_property_gt else set()

    print(f"  FPs reclassified as TP_soft ({pct_label} agreement): "
          f"{len(global_tp_soft)} pairs")
    if global_tp_soft:
        for p in sorted(global_tp_soft, key=lambda x: sorted(_pair_to_ab(x))):
            a, b = _unpack_pair(p, use_canonical_pair)
            print(f"    ~  {a}  {'↔' if use_canonical_pair else '→'}  {b}")

    summary_rows = []

    for model_folder in model_dirs:
        model_name = model_folder.name
        time_info  = time_by_model[model_name]
        status      = status_by_model[model_name]

        print(f"\n{SEP2}")
        print(f"  MODEL: {model_name}  |  time: {time_info['duration']}  "
              f"|  status: {time_info['status']}")
        print(SEP2)

        if status == "error":
            print("  [SKIPPED] Model finished with an error.")
            summary_rows.append({
                "model": model_name, "status": "error",
                "duration": time_info["duration"], "duration_s": time_info["duration_s"],
            })
            continue

        if status == "empty":
            print("  [SKIPPED] Empty results.csv.")
            summary_rows.append({
                "model": model_name, "status": "empty",
                "duration": time_info["duration"], "duration_s": time_info["duration_s"],
            })
            continue

        predicted        = predicted_by_model[model_name]
        predicted_f      = predicted_f_by_model[model_name]
        predicted_within = predicted & universe
        predicted_outside   = predicted - universe

        # Exports errors at the property level (aligned with TABLE 1) when
        # the GT has property columns -- ensures fp_errors.csv/fn_errors.csv
        # show the same pairs that appear as FP/FN in TABLE 1.
        # Without property columns, falls back to the entity level (TABLE 4).
        if has_property_gt:
            fp_pairs_exp = predicted_f - causal_gt_f
            fn_pairs_exp = causal_gt_f - predicted_f
            key_exp = "_property_pair_key"
        else:
            fp_pairs_exp = predicted - causal_gt
            fn_pairs_exp = causal_gt - predicted
            key_exp = "_canonical_pair"
        _export_model_errors(
            model_folder=model_folder,
            model_name=model_name,
            fp_pairs=fp_pairs_exp,
            fn_pairs=fn_pairs_exp,
            results_with_pair=results_with_pair_by_model.get(model_name),
            original_with_pair=original_with_pair_by_model.get(model_name),
            key_col=key_exp,
        )

        print(f"  Pairs tested (total unique)   : {len(tested_by_model.get(model_name, set()))}")
        print(f"  Pairs predicted (total unique): {len(predicted)}")
        print(f"  Pairs predicted inside the GT : {len(predicted_within)}")
        print(f"  Pairs predicted outside the GT: {len(predicted_outside)}")

        m   = _compute_metrics(predicted, causal_gt, universe)
        ms  = _compute_soft_metrics(predicted, causal_gt, universe, global_tp_soft)
        mf  = _compute_property_metrics(predicted_f, causal_gt_f, universe_f) \
              if has_property_gt else {}
        msc = _compute_consolidated_soft(predicted, causal_gt, universe,
                                          global_tp_soft,
                                          predicted_f, causal_gt_f, universe_f)
        msv2 = _compute_soft_v2(predicted_f, causal_gt_f, universe_f,
                                   global_tp_soft_f) \
               if has_property_gt else {}

        print(f"  Causal GT pairs               : {m['n_causal_gt']}")
        print(f"  Universe (== GT pairs)        : {m['n_universe']}")

        print()
        print(f"  {'─'*44}")
        print(f"  {'HARD METRICS (strict ground truth)':<44}")
        print(f"  {'─'*44}")
        print(f"  {'Metric':<20} {'Value':>8}")
        print(f"  {'-'*30}")
        print(f"  {'Precision':<20} {m['precision']:>8.4f}")
        print(f"  {'Recall':<20} {m['recall']:>8.4f}")
        print(f"  {'F1 Score':<20} {m['f1']:>8.4f}")
        print(f"  {'Accuracy (trivial)':<20} {m['accuracy']:>8.4f}")
        print(f"  {'FPR (trivial)':<20} {m['fpr']:>8.4f}")
        print(f"  {'-'*30}")
        print(f"  {'TP':<20} {m['TP']:>8d}")
        print(f"  {'FP':<20} {m['FP']:>8d}  (includes {len(predicted_outside)} outside the GT universe)")
        print(f"  {'FN':<20} {m['FN']:>8d}")
        print(f"  {'TN':<20} {m['TN']:>8d}  (always 0; universe == GT)")

        print()
        print(f"  {'─'*44}")
        print(f"  {f'SOFT METRICS (FP with {pct_label} agreement = TP_soft)':<44}")
        print(f"  {'─'*44}")
        print(f"  {'Metric':<20} {'Value':>8}")
        print(f"  {'-'*30}")
        print(f"  {'Precision (soft)':<20} {ms['precision_soft']:>8.4f}")
        print(f"  {'Recall    (soft)':<20} {ms['recall_soft']:>8.4f}")
        print(f"  {'F1        (soft)':<20} {ms['f1_soft']:>8.4f}")
        print(f"  {'-'*30}")
        print(f"  {'TP_hard':<20} {ms['TP_hard']:>8d}  (in the GT)")
        print(f"  {'TP_soft':<20} {ms['TP_soft']:>8d}  (outside GT, {pct_label} consensus)")
        print(f"  {'FP_hard':<20} {ms['FP_hard']:>8d}  (outside GT, no consensus)")
        print(f"  {'FN':<20} {ms['FN']:>8d}")

        if show_pairs:
            tp_pairs          = causal_gt & predicted
            soft_predicted     = predicted
            raw_fp         = soft_predicted - causal_gt
            model_fp_soft = raw_fp & global_tp_soft
            model_fp_hard = raw_fp - global_tp_soft
            soft_fn_pairs     = causal_gt - soft_predicted
            arrow = "↔" if use_canonical_pair else "→"

            if tp_pairs:
                print(f"\n  TP_hard -- in the GT and in the result ({len(tp_pairs)}):")
                for p in sorted(tp_pairs, key=lambda x: sorted(_pair_to_ab(x))):
                    a, b = _unpack_pair(p, use_canonical_pair)
                    print(f"    ✔  {a}  {arrow}  {b}")

            if model_fp_soft:
                print(f"\n  TP_soft -- not in the GT but with {pct_label} consensus ({len(model_fp_soft)}):")
                for p in sorted(model_fp_soft, key=lambda x: sorted(_pair_to_ab(x))):
                    a, b = _unpack_pair(p, use_canonical_pair)
                    print(f"    ~  {a}  {arrow}  {b}")

            if model_fp_hard:
                print(f"\n  FP_hard -- not in the GT and no consensus ({len(model_fp_hard)}):")
                for p in sorted(model_fp_hard, key=lambda x: sorted(_pair_to_ab(x))):
                    a, b = _unpack_pair(p, use_canonical_pair)
                    print(f"    ✘  {a}  {arrow}  {b}")

            if soft_fn_pairs:
                print(f"\n  FN -- in the GT but not in the result ({len(soft_fn_pairs)}):")
                for p in sorted(soft_fn_pairs, key=lambda x: sorted(_pair_to_ab(x))):
                    a, b = _unpack_pair(p, use_canonical_pair)
                    print(f"    ○  {a}  {arrow}  {b}")

        summary_rows.append({
            "model":          model_name,
            "status":          "ok",
            "duration":         time_info["duration"],
            "duration_s":       time_info["duration_s"],
            "n_predicted":      m["n_predicted"],
            "n_tested":      len(tested_by_model.get(model_name, set())),
            "n_causal":       m["n_causal_gt"],
            "n_universe":      m["n_universe"],
            "n_outside_universe": len(predicted_outside),
            # hard (entity level)
            "TP":              m["TP"],
            "FP":              m["FP"],
            "FN":              m["FN"],
            "TN":              m["TN"],
            "precision":       round(m["precision"], 4),
            "recall":          round(m["recall"],    4),
            "f1":              round(m["f1"],        4),
            "accuracy":        round(m["accuracy"],  4),
            "fpr":             round(m["fpr"],       4),
            # soft (entity level)
            "TP_hard":         ms["TP_hard"],
            "TP_soft":         ms["TP_soft"],
            "FP_hard":         ms["FP_hard"],
            "FN_soft":         ms["FN"],
            "TN_soft":         ms["TN"],
            "precision_soft":  round(ms["precision_soft"], 4),
            "recall_soft":     round(ms["recall_soft"],    4),
            "f1_soft":         round(ms["f1_soft"],        4),
            "soft_threshold":  soft_threshold,
            # table 1 -- hard, property level
            "f_TP":            mf.get("f_TP", ""),
            "f_FP":            mf.get("f_FP", ""),
            "f_FN":            mf.get("f_FN", ""),
            "f_TN":            mf.get("f_TN", ""),
            "f_precision":     round(mf["f_precision"], 4) if mf else "",
            "f_recall":        round(mf["f_recall"],    4) if mf else "",
            "f_f1":            round(mf["f_f1"],        4) if mf else "",
            "f_n_predicted":    mf.get("f_n_predicted", ""),
            # table 2 -- consolidated soft
            "sc_TP":           msc["sc_TP"],
            "sc_FP":           msc["sc_FP"],
            "sc_FN":           msc["sc_FN"],
            "sc_TN":           msc["sc_TN"],
            "sc_precision":    round(msc["sc_precision"], 4),
            "sc_recall":       round(msc["sc_recall"],    4),
            "sc_f1":           round(msc["sc_f1"],        4),
            "sc_n_unique_model": msc["sc_n_unique_model"],
            "sc_n_unique_gt":     msc["sc_n_unique_gt"],
            # table 3 -- soft v2, property level
            "sv2_TP_hard":     msv2.get("sv2_TP_hard", ""),
            "sv2_TP_soft":     msv2.get("sv2_TP_soft", ""),
            "sv2_FP":          msv2.get("sv2_FP",       ""),
            "sv2_FN":          msv2.get("sv2_FN",       ""),
            "sv2_TN":          msv2.get("sv2_TN",       ""),
            "sv2_precision":   round(msv2["sv2_precision"], 4) if msv2 else "",
            "sv2_recall":      round(msv2["sv2_recall"],    4) if msv2 else "",
            "sv2_f1":          round(msv2["sv2_f1"],        4) if msv2 else "",
            "sv2_n_predicted_f": msv2.get("sv2_n_predicted_f", ""),
        })

    df_summary = pd.DataFrame(summary_rows)

    print(f"\n{SEP}")
    print(f"  COMPARISON -- {dataset_name}")
    print(SEP)
    df_ok = df_summary[df_summary["status"] == "ok"]
    if not df_ok.empty:
        cols_hard = ["model", "precision", "recall", "f1",
                     "TP", "FP", "FN", "TN", "n_predicted", "n_tested", "n_causal",
                     "n_outside_universe", "duration"]
        cols_soft = ["model", "precision_soft", "recall_soft", "f1_soft",
                     "TP_hard", "TP_soft", "FP_hard", "FN"]
        print("  [HARD]  (Accuracy/FPR omitted: trivial with universe == GT)")
        print(df_ok[[c for c in cols_hard if c in df_ok.columns]].to_string(index=False))
        print()
        print(f"  [SOFT -- threshold={pct_label}]")
        print(df_ok[[c for c in cols_soft if c in df_ok.columns]].to_string(index=False))
    else:
        print("  No model with valid results.")
    print()

    if not df_summary.empty:
        _save_validation_summary(dataset_folder, dataset_name, df_summary,
                                 global_tp_soft, soft_threshold,
                                 gt=gt, gt_cols=cols,
                                 use_canonical_pair=use_canonical_pair)

    return df_summary


# =============================================================================
# GENERAL VALIDATOR -- discovers datasets under output_dir
# =============================================================================

def _discover_datasets(output_dir: str) -> list:
    root = Path(output_dir)
    if not root.exists():
        return []
    return sorted([
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "ground_truth.csv").exists()
    ])


def validate_all(output_dir: str, show_pairs: bool = False, soft_threshold: float = 1.0,
                   ignore_intra_entity: bool = False,
                   use_canonical_pair: bool = USE_CANONICAL_PAIR_DEFAULT):
    """
    Step 2: discovers every dataset with a GT and validates each one.

    ignore_intra_entity : see the docstring of validate_dataset().
    use_canonical_pair      : see the docstring of validate_dataset().
    """
    datasets = _discover_datasets(output_dir)

    if not datasets:
        print(f"\n[WARNING] No dataset with ground_truth.csv found in '{output_dir}'.")
        return

    print(f"\n{'#'*65}")
    print(f"  VALIDATION -- {len(datasets)} dataset(s) found")
    print(f"{'#'*65}")
    print(f"  {datasets}")
    print(f"  Pairing: {'CANONICAL (non-directional)' if use_canonical_pair else 'DIRECTIONAL'}")

    for name in datasets:
        validate_dataset(
            dataset_name=name,
            output_dir=output_dir,
            show_pairs=show_pairs,
            soft_threshold=soft_threshold,
            ignore_intra_entity=ignore_intra_entity,
            use_canonical_pair=use_canonical_pair,
        )


# =============================================================================
# ENTRY POINT  <- edit here
# =============================================================================

if __name__ == "__main__":

    total_start_time = time.perf_counter()

    # ── Execution control / active datasets / models / PIPELINE_CONFIG ────
    # The run's whole editable control panel (what to run, which benchmarks
    # are active, MODELS, PIPELINE_CONFIG) lives in Config/execution.py --
    # edit it there to switch scenario/config without touching this runner.
    # Here we only bring the names into local variables.
    RUN_RUNNER    = cfg_exec.RUN_RUNNER
    RUN_VALIDATION = cfg_exec.RUN_VALIDATION
    SHOW_PAIRS      = cfg_exec.SHOW_PAIRS

    SOFT_THRESHOLD = cfg_exec.SOFT_THRESHOLD
    DETECT_INTRA_ENTITY = cfg_exec.DETECT_INTRA_ENTITY
    USE_CANONICAL_PAIR = cfg_exec.USE_CANONICAL_PAIR

    USE_SYNTHETIC_VAR_BENCHMARK = cfg_exec.USE_SYNTHETIC_VAR_BENCHMARK
    SYNTHETIC_VAR_SCENARIOS       = cfg_exec.SYNTHETIC_VAR_SCENARIOS

    USE_TIGRAMITE_BENCHMARK  = cfg_exec.USE_TIGRAMITE_BENCHMARK
    USE_LORENZ96_BENCHMARK   = cfg_exec.USE_LORENZ96_BENCHMARK
    USE_LOGISTIC_BENCHMARK   = cfg_exec.USE_LOGISTIC_BENCHMARK
    USE_CAUSALTIME_BENCHMARK = cfg_exec.USE_CAUSALTIME_BENCHMARK
    USE_ALARM_BENCHMARK            = cfg_exec.USE_ALARM_BENCHMARK
    USE_NETSIM_BENCHMARK     = cfg_exec.USE_NETSIM_BENCHMARK

    CAUSALTIME_DIR_PATH = cfg_exec.CAUSALTIME_DIR_PATH
    NETSIM_SUBJECTS     = cfg_exec.NETSIM_SUBJECTS

    MODELS         = cfg_exec.MODELS
    PIPELINE_CONFIG = cfg_exec.PIPELINE_CONFIG

    DATAFRAMES: list = []

    # ── Loading with fallback ───────────────────────────────────────────
    # Each source below goes through load_with_fallback(): it tries to load;
    # if the source is not available IN THIS ENVIRONMENT (missing
    # dependency, file/folder not found, no internet, etc.), that
    # source is skipped, the reason is logged (console +
    # Data/base.SCENARIO_FAILURES_LOG) and loading of the remaining sources
    # proceeds normally. See Data/base.py:load_with_fallback().
    if USE_SYNTHETIC_VAR_BENCHMARK:
        # Each tuple of SYNTHETIC_VAR_SCENARIOS generates ONE scenario, via
        # an isolated call to generate_synthetic_var_benchmark_scenarios()
        # (seeds=[seed]), since each scenario has its own combination of
        # n_points/n_entities/prob_edge/max_parents/max_lag/min_edges.
        print(f"\n  Generating {len(SYNTHETIC_VAR_SCENARIOS)} synthetic scenario(s) "
              f"via VAR (Data/synthetic_var.py) ...")
        for (name_prefix, seed, n_points, n_entities, prob_edge,
             max_parents, max_lag, min_edges) in SYNTHETIC_VAR_SCENARIOS:
            DATAFRAMES += load_with_fallback(
                name_prefix, generate_synthetic_var_benchmark_scenarios,
                seeds=[seed],
                n_points=n_points,
                name_prefix=name_prefix,
                n_entities=n_entities,
                prob_edge=prob_edge,
                max_parents=max_parents,
                max_lag=max_lag,
                min_edges=min_edges,
            )

    if USE_TIGRAMITE_BENCHMARK:
        # Provider (Data/tigramite_scm.py): synthetic nonlinear scenarios
        # generated via the tigramite package's own structural causal
        # model generators (Runge et al.) -- the same library used to
        # validate PCMCI/PCMCI+ in the literature. GT known by
        # construction. T=300 (well above the short-series sparse regime)
        # -- the goal here is to isolate NONLINEAR discovery capability
        # itself, not to repeat the short-series robustness test (already
        # covered above).
        DATAFRAMES += load_with_fallback(
            "tigramite", generate_tigramite_benchmark_scenarios,
            seeds=[11, 22, 33], n_points=300,
        )

    if USE_LORENZ96_BENCHMARK:
        # Provider (Data/lorenz96.py): classic Lorenz-96 spatio-temporal
        # chaotic system -- GT is each variable's three spatial neighbors
        # (k-1, k-2, k+1 mod K), known analytically from the equations,
        # not estimated.
        DATAFRAMES += load_with_fallback(
            "lorenz96", generate_lorenz96_benchmark_scenarios,
            Ks=[8, 10], n_points=300, seeds=[1, 2, 3],
        )

    if USE_LOGISTIC_BENCHMARK:
        # Provider (Data/logistic_maps.py): classic coupled discrete
        # logistic maps benchmark -- GT is the coupling topology between
        # maps, known by construction.
        DATAFRAMES += load_with_fallback(
            "logistic_maps", generate_logistic_maps_benchmark_scenarios,
            n_points=300, seeds=[1, 2, 3],
        )

    if USE_CAUSALTIME_BENCHMARK:
        # Provider (Data/causaltime.py): reads the medical/pm25/traffic
        # domains from CausalTime (gen_data.npy + graph.npy) and returns
        # scenarios in the pipeline's standard format. The default 'mode'
        # ("separate_samples") treats each sample of the dataset as an
        # independent scenario/run -- see the module's docstring for the
        # justification (avoids concatenating samples that may not be
        # contiguous blocks in time). max_samples=3 caps the number of
        # scenarios per domain.
        DATAFRAMES += load_with_fallback(
            "causaltime", generate_causaltime_benchmark_scenarios,
            domains=["medical", "pm25", "traffic"],
            base_dir=CAUSALTIME_DIR_PATH,
            mode="separate_samples",
            max_samples=3,
        )

    if USE_ALARM_BENCHMARK:
        # Provider (Data/alarm.py): ALARM Bayesian network (Beinlich et
        # al. 1989) -- expert-curated GT (37 nodes/46 edges), via bnlearn.
        # Automatic download (with cache). NO real time axis (i.i.d.
        # sampling from the network) -- see the caveat in the module's
        # docstring.
        DATAFRAMES += load_with_fallback(
            "alarm", load_alarm_benchmark,
            n_samples=500,
        )

    if USE_NETSIM_BENCHMARK:
        # Provider (Data/netsim.py): NetSim simulated BOLD fMRI, sim3 (Smith
        # et al. 2011) -- 15 nodes x 200 points, one scenario per simulated
        # subject. GT is the network used by the simulator, known by
        # construction. The relations are essentially contemporaneous at the
        # 3 s sampling -- see the caveat in the module's docstring.
        DATAFRAMES += load_with_fallback(
            "netsim", generate_netsim_benchmark_scenarios,
            subjects=NETSIM_SUBJECTS,
        )

    # The held-out datasets are not loaded here: they are evaluated outside this
    # development suite by main_heldout.py.

    OUTPUT_DIR = f"./Results/Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ── Step 1: Runner ───────────────────────────────────────────────────────
    if RUN_RUNNER:
        print("\n" + "=" * 65)
        print("  STEP 1 -- RUNNER")
        print("=" * 65)
        run_all(
            dataframes=DATAFRAMES,
            models=MODELS,
            config=PIPELINE_CONFIG,
            output_dir=OUTPUT_DIR,
        )

    # ── Step 2: Validation ────────────────────────────────────────────────────
    if RUN_VALIDATION:
        print("\n" + "=" * 65)
        print("  STEP 2 -- VALIDATION")
        print("=" * 65)
        validate_all(
            output_dir=OUTPUT_DIR,
            show_pairs=SHOW_PAIRS,
            soft_threshold=SOFT_THRESHOLD,
            ignore_intra_entity=not DETECT_INTRA_ENTITY,
            use_canonical_pair=USE_CANONICAL_PAIR,
        )

    print("Exporting summary")
    consolidate(OUTPUT_DIR)

    print("Checking fp")
    export_fp(OUTPUT_DIR)

    print("Checking fn")
    export_fn(OUTPUT_DIR)

    total_time_fmt = _format_duration(time.perf_counter() - total_start_time)
    print("\n" + "#" * 65)
    print(f"  TOTAL SCRIPT EXECUTION TIME: {total_time_fmt}")
    print("#" * 65)