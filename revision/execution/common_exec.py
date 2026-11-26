"""
Functions shared by the revision experiments that run models.

Everything goes through the same path as the paper's runs: the scenarios are
loaded with the arguments of main.py and main_heldout.py
(main_sensitivity.load_scenarios), the models are the entries of
Config.execution.ALL_MODELS with explicit overrides, and execution and
validation are those of main.run_all and main.validate_all (directional pairs,
intra-entity pairs counted). No default of the pipeline is changed: each variant
is a new entry with its own name.
"""
from __future__ import annotations

import os
import platform
import sys
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)   # the loaders use paths relative to the repository root

import main as main_mod  # noqa: E402
from Config import execution as cfg_exec  # noqa: E402

OUTPUT_ROOT = os.path.join(ROOT, "Results_Revisao")
PAPER_MODELS_TAU10 = ["patch_tau10", "pcmci_tigramite_tau10", "pcmciplus_tau10", "dynotears_tau10",
                      "cuts_plus_tau10", "varlingam_tau10"]


def output_dir(name: str) -> str:
    d = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def model(base: str, name: Optional[str] = None, **override) -> dict:
    """Entry of Config.execution.ALL_MODELS with overrides (new entry, new name)."""
    by_name = {m["name"]: m for m in cfg_exec.ALL_MODELS}
    m = by_name[base]
    return {**m, "name": name or base, "config_override": {**m.get("config_override", {}), **override}}


def models(names: Iterable[str]) -> List[dict]:
    return [model(n) for n in names]


def paper_scenarios(name_filter: Optional[Iterable[str]] = None) -> list:
    """The 90 scenarios of the paper, loaded as in main.py and main_heldout.py."""
    import main_sensitivity
    dfs = main_sensitivity.load_scenarios()
    if name_filter:
        name_filter = list(name_filter)
        dfs = [d for d in dfs if any(f in d["name"] for f in name_filter)]
    return dfs


def generate_var(seeds, n_points, n_entities, prob_edge, max_parents, max_lag, min_edges=1,
                 name_prefix="revisao_var", **kw) -> list:
    """Scenarios of the internal VAR generator (Data/synthetic_var.py), with explicit parameters."""
    from Data import generate_synthetic_var_benchmark_scenarios
    return generate_synthetic_var_benchmark_scenarios(
        seeds=list(seeds), n_points=n_points, name_prefix=name_prefix, n_entities=n_entities,
        prob_edge=prob_edge, max_parents=max_parents, max_lag=max_lag, min_edges=min_edges, **kw)


def run(dataframes: list, models: list, folder: str, validate: bool = True):
    """main.run_all + main.validate_all, with the base configuration of the paper."""
    main_mod.run_all(dataframes=dataframes, models=models, config=cfg_exec.PIPELINE_CONFIG,
                     output_dir=folder)
    if validate:
        main_mod.validate_all(output_dir=folder, show_pairs=False, soft_threshold=cfg_exec.SOFT_THRESHOLD,
                              ignore_intra_entity=not cfg_exec.DETECT_INTRA_ENTITY,
                              use_canonical_pair=cfg_exec.USE_CANONICAL_PAIR)


def already_run(folder: str, scenario: str, model_name: str) -> bool:
    p = os.path.join(folder, scenario, model_name, "time.txt")
    return os.path.exists(p) and "status" in open(p, encoding="utf-8").read()


def pending(dataframes: list, models: list, folder: str):
    """Resumption: only the (scenario, model) pairs not yet run."""
    out = []
    for d in dataframes:
        ms = [m for m in models if not already_run(folder, d["name"], m["name"])]
        if ms:
            out.append((d, ms))
    return out


def run_resuming(dataframes: list, models: list, folder: str):
    for d, ms in pending(dataframes, models, folder):
        main_mod.run_all(dataframes=[d], models=ms, config=cfg_exec.PIPELINE_CONFIG, output_dir=folder)
    main_mod.validate_all(output_dir=folder, show_pairs=False, soft_threshold=cfg_exec.SOFT_THRESHOLD,
                          ignore_intra_entity=not cfg_exec.DETECT_INTRA_ENTITY,
                          use_canonical_pair=cfg_exec.USE_CANONICAL_PAIR)


# ----------------------------------------------------------------------------- parallel execution
# One process per scenario (each runs the models still missing for that scenario, in sequence).
# Per scenario, and not per (scenario, model), because main.run_all writes source.csv and
# ground_truth.csv in the scenario folder. The result is the same as a sequential run; only the
# order changes. The models travel as specifications (base, name, overrides), rebuilt in the child
# process, so as not to depend on serializing configuration objects.

def log_progress(path: str, msg: str) -> None:
    import datetime
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _one_scenario(d, specs, folder, logs_dir):
    import contextlib
    import time
    t0 = time.perf_counter()
    try:
        import torch  # CUTS+: one thread per process (the other baselines are deterministic)
        torch.set_num_threads(1)
    except Exception:
        pass
    missing = [model(b, n, **o) for b, n, o in specs if not already_run(folder, d["name"], n)]
    if missing:
        with open(os.path.join(logs_dir, f"{d['name']}.log"), "a", encoding="utf-8", buffering=1) as f, \
                contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            main_mod.run_all(dataframes=[d], models=missing, config=cfg_exec.PIPELINE_CONFIG, output_dir=folder)
    return d["name"], len(missing), time.perf_counter() - t0


def run_parallel(dataframes: list, specs: list, folder: str, n_proc: int, single_thread: bool = True):
    """Runs the models `specs` = [(base, name, overrides), ...] on `dataframes`, one process per
    scenario, up to n_proc at a time; resumes (skips what already has a time.txt). Progress in
    <folder>/../progresso.log and the output of each scenario in <folder>/../logs_cenarios/.
    single_thread=True limits BLAS/OpenMP to one thread per process (avoids oversubscription; does not
    change deterministic results)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if single_thread:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"   # inherited by the child processes
    base = os.path.dirname(folder)
    progress = os.path.join(base, "progresso.log")
    logs_dir = os.path.join(base, "logs_cenarios")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(folder, exist_ok=True)
    pend = [d for d in dataframes if any(not already_run(folder, d["name"], n) for _, n, _ in specs)]
    n_proc = max(1, min(n_proc, len(pend) or 1))
    log_progress(progress, f"start: {len(pend)} scenarios pending out of {len(dataframes)} x {len(specs)} "
                           f"configurations, {n_proc} processes ({environment()['plataforma']}, "
                           f"{os.cpu_count()} CPUs)")
    done = 0
    with ProcessPoolExecutor(max_workers=n_proc) as ex:
        fut = {ex.submit(_one_scenario, d, specs, folder, logs_dir): d["name"] for d in pend}
        for f in as_completed(fut):
            done += 1
            try:
                _, n_mod, dur = f.result()
                log_progress(progress, f"finished ({done}/{len(pend)}): {fut[f]} — {n_mod} run(s) "
                                       f"in {dur / 60:.1f} min")
            except Exception as e:
                log_progress(progress, f"ERROR ({done}/{len(pend)}): {fut[f]} — {type(e).__name__}: {e}")
    log_progress(progress, "validating")
    main_mod.validate_all(output_dir=folder, show_pairs=False, soft_threshold=cfg_exec.SOFT_THRESHOLD,
                          ignore_intra_entity=not cfg_exec.DETECT_INTRA_ENTITY,
                          use_canonical_pair=cfg_exec.USE_CANONICAL_PAIR)
    log_progress(progress, "end")


def summary(folder: str) -> pd.DataFrame:
    """validation_summary.csv of every scenario in a folder: one row per scenario x model."""
    frames = []
    for s in sorted(os.listdir(folder)):
        p = os.path.join(folder, s, "validation_summary.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            d.insert(0, "cenario", s)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def f1(tp, fp, fn) -> float:
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def wide_series(df: pd.DataFrame) -> pd.DataFrame:
    """Long format (Date, Name, properties) -> one column per entity.property."""
    props = [c for c in df.columns if c not in ("Date", "Name")]
    w = df.pivot_table(index="Date", columns="Name", values=props, aggfunc="first")
    w.columns = [f"{e}.{p}" for p, e in w.columns]
    return w.sort_index()


def axis_statistic(wide: pd.DataFrame, band: float = 2.0):
    """Time-axis diagnostic of the pipeline (Pipeline/discovery._detect_time_axis):
    median |ACF(1)| over the series (with at least 30 values) against band/sqrt(n)."""
    acf, n = [], []
    for c in wide.columns:
        v = wide[c].to_numpy(float)
        v = v[~np.isnan(v)]
        if len(v) < 30:
            continue
        v = v - v.mean()
        den = float(v @ v)
        if den <= 0:
            continue
        acf.append(abs(float(v[:-1] @ v[1:]) / den))
        n.append(len(v))
    if not acf:
        return None
    nm = int(np.median(n))
    med = float(np.median(acf))
    return {"mediana_acf1": med, "n": nm, "limiar": band / np.sqrt(nm), "sem_eixo": med < band / np.sqrt(nm)}


def environment() -> dict:
    """Machine the experiment ran on (for the run-time table, R1-m7a)."""
    return {"plataforma": platform.platform(), "processador": platform.processor(),
            "cpus": os.cpu_count(), "python": platform.python_version(),
            "n_jobs_pipeline": cfg_exec.PIPELINE_CONFIG.get("n_jobs")}
