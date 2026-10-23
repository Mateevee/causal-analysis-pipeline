"""
Data.rcaeval
===============================================================================
REAL BENCHMARK — RCAEval (Pham et al. 2024/2025, "RCAEval: A Benchmark for
Root Cause Analysis of Microservice Systems with Telemetry Data",
arXiv:2412.17015; ACM Web Conference 2025 Companion). REAL telemetry
collected from reference microservice systems (Online Boutique, Sock Shop,
Train Ticket) under DELIBERATELY INJECTED FAULTS (chaos engineering:
CPU/memory/disk stress, network delay/loss) — AIOps/observability domain.

GROUND TRUTH: the root SERVICE of each case is known BY CONSTRUCTION — the
authors themselves injected the fault into that specific service (it is
neither domain knowledge nor estimated). Same tier of rigour as
Data/causal_chamber.py (real controlled intervention), but here the
"intervention" is an infrastructure fault, not a continuous physical variable.

IMPORTANT — THE GT IS A GRAPH-SHAPED APPROXIMATION: the original benchmark
only validates WHICH SERVICE is the root cause (a single label per case), not
a pairwise list of edges between metrics. This loader converts that label
into a "fan-out" graph: every metric of the SERVICE WHERE THE FAULT WAS
INJECTED points to every metric of ALL the other services monitored in the
case -- an approximation of the real propagation (which follows the call
graph between services, not published per case in this loader). Report it as
a limitation: the origin of the fault IS validated by construction, only the
pairwise extent of the propagation is approximated (same spirit as the
topology GT of Data/pems_bay.py).

Each "entity" of this loader is a (service, metric) pair flattened into a
single name (e.g. 'adservice_cpu') — same pattern as Data/pems_bay.py (each
sensor is a "flat" entity) — see `array_to_pipeline_format_single_property`.

DATA FORMAT (RE1 — metric-only; RE2/RE3 also have logs/traces, out of the
scope of this loader)
-------------------------------------------------------------------------------
Each case: <base_dir>/<benchmark_folder>/<service>_<fault_type>/<instance>/
    data.csv         : 'time' column (unix, seconds) + one column per
                        service_metric pair (cpu/mem/workload/error/
                        latency-50/latency-90 -- not every service has all).
    inject_time.txt   : unix timestamp (seconds) of the start of the fault.

HOW THE DATA IS OBTAINED
-------------------------------------------------------------------------------
Downloads automatically (with disk cache) the official Zenodo mirror of
RCAEval (record 14590730). Uses RE1-OB (Online Boutique) by default -- the
smallest of the 9 files (~31MB); RE1-SS (Sock Shop, ~79MB) and RE1-TT (Train
Ticket, ~280MB) are also available via `benchmark_folder`. RE2/RE3 (with
logs/traces, up to 2.8GB) are NOT supported by this loader.
"""
from __future__ import annotations

import os
import urllib.request as _urlreq_bench
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import ScenarioEntry, register, array_to_pipeline_format_single_property

_ZENODO_BASE = "https://zenodo.org/records/14590730/files"
_BASE_DEFAULT = Path(os.environ.get("RCAEVAL_DIR", "./external_data/rcaeval"))

_DEFAULT_METRICS = ("cpu", "mem", "workload")


def _download_and_extract(cache_dir: Path, benchmark_folder: str, timeout: int = 300) -> Path:
    """Downloads (with cache) <benchmark_folder>.zip from the RCAEval Zenodo
    mirror and extracts it into cache_dir. Returns the extracted folder."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = cache_dir / benchmark_folder
    if extracted_dir.exists():
        return extracted_dir

    zip_path = cache_dir / f"{benchmark_folder}.zip"
    if not zip_path.exists():
        url = f"{_ZENODO_BASE}/{benchmark_folder}.zip"
        try:
            with _urlreq_bench.urlopen(url, timeout=timeout) as resp:
                zip_path.write_bytes(resp.read())
        except Exception as e:
            raise RuntimeError(
                f"Failed to download '{url}'. Check your internet "
                f"connection, or download it manually to '{zip_path}'. "
                f"Original error: {e}"
            ) from e

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    return extracted_dir


def _list_cases(benchmark_dir: Path) -> List[Path]:
    """Returns the <service>_<fault>/<instance>/ paths that hold a data.csv."""
    return sorted(p.parent for p in benchmark_dir.glob("*/*/data.csv"))


@register("rcaeval")
def generate_rcaeval_benchmark_scenarios(
    *,
    benchmark_folder: str = "RE1-OB",
    max_cases: Optional[int] = 10,
    metrics: Optional[List[str]] = None,
    n_services: Optional[int] = None,
    max_rows: Optional[int] = 500,
    sampling_seed: int = 0,
    cache_dir: Optional[str] = None,
    name_prefix: str = "benchmark_rcaeval",
) -> List[dict]:
    """
    Generates one ScenarioEntry per fault case of the RCAEval benchmark (RE1
    — metric-only) — see the module docstring for the source, the GT
    (fan-out from the service with the injected fault, approximate — see the
    caveat) and the automatic download (with cache).

    Parameters
    ----------
    benchmark_folder : 'RE1-OB' (Online Boutique, ~31MB, default), 'RE1-SS'
                       (Sock Shop, ~79MB) or 'RE1-TT' (Train Ticket, ~280MB).
    max_cases        : cap on the number of cases/scenarios generated (RE1-OB
                       has ~120 cases: ~24 service x fault combinations x 5
                       instances). None = all.
    metrics          : metric suffixes kept as entities (default
                       ('cpu','mem','workload') -- leaves out 'error'/
                       'latency-50'/'latency-90' to reduce the number of
                       entities per case; not every service has every
                       metric, absent columns are ignored).
    n_services       : if given, restricts the case to N services: the one
                       where the fault was injected plus the first N-1
                       others in alphabetical order. None = all the services
                       of the case.
    max_rows         : if given, keeps up to this many CONSECUTIVE rows from
                       the start (preserves the real temporal dynamics --
                       'time' is real elapsed time). None = all (~700-800
                       points per case).
    sampling_seed    : seed of the RNG used to choose which cases enter (if
                       max_cases < total available).
    cache_dir        : download/cache folder (default: env RCAEVAL_DIR or
                       "./external_data/rcaeval").
    name_prefix      : prefix of each generated scenario's name/folder.

    Returns a list of dicts (ScenarioEntry.to_dict()).
    """
    metrics = tuple(metrics) if metrics is not None else _DEFAULT_METRICS

    cache = Path(cache_dir) if cache_dir else _BASE_DEFAULT
    data_dir = _download_and_extract(cache, benchmark_folder)

    all_cases = _list_cases(data_dir)
    if not all_cases:
        raise FileNotFoundError(f"No case (*/*/data.csv) found in '{data_dir}'.")

    if max_cases is not None and max_cases < len(all_cases):
        rng = np.random.default_rng(sampling_seed)
        idx = np.sort(rng.choice(len(all_cases), size=max_cases, replace=False))
        selected_cases = [all_cases[i] for i in idx]
    else:
        selected_cases = all_cases

    scenarios = []
    for case_dir in selected_cases:
        fault_name = case_dir.parent.name
        instance = case_dir.name
        root_service, _, fault_type = fault_name.rpartition("_")
        if not root_service:
            continue

        df_raw = pd.read_csv(case_dir / "data.csv")
        if "time" not in df_raw.columns:
            continue

        metric_columns = [
            c for c in df_raw.columns
            if c != "time" and c.rsplit("_", 1)[-1] in metrics
        ]

        if n_services is not None:
            services_present = sorted({c.rsplit("_", 1)[0] for c in metric_columns})
            if len(services_present) > n_services:
                # always keeps the service where the fault was injected
                others = [s for s in services_present if s != root_service]
                services_kept = {root_service, *others[:max(0, n_services - 1)]}
                metric_columns = [
                    c for c in metric_columns
                    if c.rsplit("_", 1)[0] in services_kept
                ]

        if len(metric_columns) < 2:
            continue

        df_var = df_raw[metric_columns].apply(pd.to_numeric, errors="coerce").dropna()
        n_original_rows = len(df_var)
        if max_rows is not None and max_rows < n_original_rows:
            df_var = df_var.iloc[:max_rows]
        if df_var.empty:
            continue

        root_columns = [c for c in metric_columns if c.rsplit("_", 1)[0] == root_service]
        other_columns = [c for c in metric_columns if c.rsplit("_", 1)[0] != root_service]
        var_idx = {v: i for i, v in enumerate(metric_columns)}
        causal_pairs_idx = {
            (var_idx[a], var_idx[b]) for a in root_columns for b in other_columns
        }

        data = df_var.to_numpy(dtype=float)
        df_long, df_gt = array_to_pipeline_format_single_property(
            data, metric_columns, causal_pairs_idx,
            start_date="2015-01-01", freq="min", property_name="VAL",
        )

        scenario_name = f"{name_prefix}_{benchmark_folder}_{fault_name}_{instance}"
        params = {
            "source": "rcaeval",
            "reference": "Pham et al. 2024/2025, arXiv:2412.17015 (ACM Web "
                         "Conference 2025 Companion)",
            "benchmark_folder": benchmark_folder,
            "case": str(case_dir.relative_to(data_dir)),
            "root_service": root_service,
            "fault_type": fault_type,
            "n_original_rows": n_original_rows,
            "n_rows_used": len(df_var),
            "n_entities": len(metric_columns),
            "n_causal_links": len(causal_pairs_idx),
            "ground_truth_type": "root service validated by REAL fault "
                                  "INJECTION (not a heuristic); the pairwise "
                                  "edges are an APPROXIMATE fan-out from the "
                                  "root service (see the caveat in the "
                                  "module docstring)",
        }

        scenarios.append(ScenarioEntry(
            name=scenario_name, df=df_long, gt=df_gt,
            _scenario=f"rcaeval_{benchmark_folder}_{fault_name}", _params=params,
        ).to_dict())

    print(f"  [real-benchmark] rcaeval/{benchmark_folder} loaded — "
          f"{len(scenarios)} fault case(s) (of {len(selected_cases)} "
          f"selected) | GT: fan-out from the root service (real fault "
          f"injection, Pham et al. 2024/2025)")

    return scenarios


if __name__ == "__main__":
    scenarios = generate_rcaeval_benchmark_scenarios(max_cases=3)
    for c in scenarios:
        n_edges = int(c["gt"]["causal_gt"].sum())
        print(f"  {c['name']}: df={c['df'].shape}, {n_edges}/{len(c['gt'])} real edges")
