"""
Data.ess_cryo
===============================================================================
REAL BENCHMARK — ESS ACCELERATOR CRYOPLANT (Mogensen, Rathsman & Nilsson
2024, "Causal discovery in a complex industrial system: A time series
benchmark", CLeaR 2024, PMLR 236:1218-1236, arXiv:2310.18654).

Control-system data of the accelerator cryoplant (ACCP) of the European
Spallation Source (Lund, Sweden), an industrial helium refrigeration plant:
233 process variables (PVs: temperatures, pressures, flows, valve
positions, speeds) grouped into 35 subsystems (compressors, turbines, heat
exchangers, adsorbers, lines), recorded during three periods of steady-state
operation.

This loader uses the REDUCED dataset published by the authors for testing
(the first hour of each period; the full dataset is 5.5 GB).

GROUND TRUTH: the causal graph built by the plant's system experts, defined
ON THE LEVEL OF SUBSYSTEMS (`adjMatrix.csv`: 23 subsystems, rows are 'to'
and columns are 'from', 0 = no edge, 1 = weak edge, 2 = strong edge). It
contains feedback loops. Expert knowledge of the process — not estimated,
and not validated by intervention. 12 of the 35 subsystems are not in the
graph (see the authors' R/createAdjacencyMatrix.R) and are not loaded.

FROM PROCESS VARIABLES TO ONE SERIES PER SUBSYSTEM
-------------------------------------------------------------------------------
1. Temporal aggregation, as described by the authors (README of their
   repository, R/data.R): the PVs are sampled irregularly and at different
   rates, so each PV is averaged within each second and, where a second has
   no observation, the last observation is carried forward (seconds before
   a PV's first observation take that first value).
2. The series are cut to their first `max_points` seconds.
3. Each subsystem is summarized by the FIRST PRINCIPAL COMPONENT of its
   standardized, non-constant PVs. The ground truth has one node per
   subsystem while the data has 2-22 PVs per subsystem; the first principal
   component is a ground-truth-free one-series summary of the group.
   This step is a choice of this loader, not of the authors.

`edge_strength`: "any" (default) counts weak and strong edges as true;
"strong" only the strong ones.

HOW THE DATA IS OBTAINED
-------------------------------------------------------------------------------
Automatically downloads (with on-disk caching):
    https://zenodo.org/records/10679737  accp_dataset_reduced.zip (80 MB,
        CC-BY-4.0): <Period_k>/<subsystem>/<PV>.csv with Timestamp, Value
    https://github.com/soerenwengel/essdata  adjMatrix.csv (MIT)
The per-second aggregation of each period is cached next to the archive
(aggregated_1s_period<k>.csv), since parsing the 699 files takes a while.
"""
from __future__ import annotations

import io
import os
import urllib.request as _urlreq_bench
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .base import array_to_pipeline_format_single_property, register

_ZIP_URL = "https://zenodo.org/records/10679737/files/accp_dataset_reduced.zip"
_ADJ_URL = "https://raw.githubusercontent.com/soerenwengel/essdata/main/adjMatrix.csv"
_BASE_DEFAULT = Path(os.environ.get("ESS_CRYO_DIR", "./external_data/ess_cryo"))

# Start of each period (UTC), from the authors' R/data.R.
_PERIOD_START = {
    1: pd.Timestamp("2022-12-30 16:00:00", tz="UTC"),
    2: pd.Timestamp("2023-01-05 18:00:00", tz="UTC"),
    3: pd.Timestamp("2023-01-13 15:00:00", tz="UTC"),
}
_SECONDS = 3600  # the reduced dataset holds the first hour of each period


def _cached(path: Path, url: str, timeout: int = 900) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _urlreq_bench.urlopen(url, timeout=timeout) as resp:
            path.write_bytes(resp.read())
    except Exception as e:
        raise RuntimeError(
            f"Failed to download '{url}'. Check your internet connection, "
            f"or download it manually to '{path}'. Original error: {e}"
        ) from e
    return path


def _aggregate_period(zip_path: Path, period: int, cache_dir: Path) -> tuple:
    """Per-second means of every PV of one period (seconds 0.._SECONDS-1
    from the period start; NaN where a second has no observation) and the
    PV -> subsystem map. Cached on disk."""
    data_path = cache_dir / f"aggregated_1s_period{period}.csv"
    map_path = cache_dir / f"pv_subsystem_period{period}.csv"
    if data_path.exists() and map_path.exists():
        return (pd.read_csv(data_path, index_col=0),
                pd.read_csv(map_path, index_col=0)["subsystem"].to_dict())

    start = _PERIOD_START[period]
    columns, subsystem_of = {}, {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            parts = info.filename.split("/")
            if (info.filename.startswith("__MACOSX") or len(parts) != 4
                    or parts[1] != f"Period_{period}" or not parts[3].endswith(".csv")):
                continue
            pv = parts[3][:-4]
            raw = pd.read_csv(io.BytesIO(z.read(info)))
            subsystem_of[pv] = parts[2]
            if raw.empty:
                continue
            t = pd.to_datetime(raw["Timestamp"], format="ISO8601", utc=True)
            second = ((t - start).dt.total_seconds()).round().astype(int)
            valid = (t >= start).to_numpy() & (second < _SECONDS).to_numpy()
            columns[pv] = raw["Value"][valid].groupby(second[valid]).mean()

    aggregated = pd.DataFrame(columns).reindex(range(_SECONDS))
    aggregated.index.name = "second"
    aggregated.to_csv(data_path)
    pd.Series(subsystem_of, name="subsystem").to_csv(map_path)
    return aggregated, subsystem_of


def _first_principal_component(block: np.ndarray) -> np.ndarray:
    """First principal component scores of the standardized columns."""
    z = (block - block.mean(axis=0)) / block.std(axis=0)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    sign = 1.0 if vt[0].sum() >= 0 else -1.0  # deterministic orientation
    return sign * u[:, 0] * s[0]


@register("ess_cryo")
def generate_ess_cryo_benchmark_scenarios(
    periods: Sequence[int] = (1, 2, 3),
    max_points: Optional[int] = 1500,
    edge_strength: str = "any",
    cache_dir: Optional[str] = None,
    name_prefix: str = "real_ess_cryo",
) -> List[dict]:
    """
    Loads the ESS accelerator cryoplant benchmark, one scenario per period —
    see the module docstring for the source, the subsystem-level ground
    truth and how the process variables are turned into one series per
    subsystem.

    Parameters
    ----------
    periods       : which of the three steady-state periods to load.
    max_points    : series longer than this are cut to their first
                    `max_points` seconds (None = the whole hour).
    edge_strength : "any" (weak and strong edges are true) or "strong".
    cache_dir     : download/cache folder (default: env ESS_CRYO_DIR or
                    "./external_data/ess_cryo").
    name_prefix   : prefix for the name/folder of each generated scenario.
    """
    if edge_strength not in ("any", "strong"):
        raise ValueError("edge_strength must be 'any' or 'strong'.")
    cache = Path(cache_dir) if cache_dir else _BASE_DEFAULT
    zip_path = _cached(cache / "accp_dataset_reduced.zip", _ZIP_URL)
    adj_path = _cached(cache / "adjMatrix.csv", _ADJ_URL)

    adjacency = pd.read_csv(adj_path, index_col=0)
    graph_subsystems = list(adjacency.index)
    min_strength = 1 if edge_strength == "any" else 2

    scenarios: List[dict] = []
    for period in periods:
        aggregated, subsystem_of = _aggregate_period(zip_path, period, cache)
        window = aggregated.ffill().bfill()
        if max_points is not None:
            window = window.iloc[:max_points]

        series, names, n_pvs = [], [], {}
        for subsystem in graph_subsystems:
            pvs = [pv for pv in window.columns
                   if subsystem_of.get(pv) == subsystem
                   and window[pv].notna().all() and window[pv].nunique() > 1]
            if not pvs:
                continue
            series.append(_first_principal_component(window[pvs].to_numpy(dtype=float)))
            names.append(subsystem)
            n_pvs[subsystem] = len(pvs)
        data = np.column_stack(series)

        # adjacency[i, j] describes the edge j -> i (rows 'to', columns 'from').
        causal_pairs_idx = {
            (names.index(src), names.index(dst))
            for dst in names for src in names
            if src != dst and adjacency.loc[dst, src] >= min_strength
        }

        df_long, df_gt = array_to_pipeline_format_single_property(
            data, names, causal_pairs_idx,
            start_date=str(_PERIOD_START[period].tz_localize(None)), freq="s",
            property_name="PC1",
        )

        name = f"{name_prefix}_period{period}"
        params = {
            "source": "ess_cryo",
            "reference": "Mogensen, Rathsman & Nilsson 2024, CLeaR "
                         "(PMLR 236:1218-1236, arXiv:2310.18654)",
            "data_file": str(zip_path),
            "ground_truth_file": str(adj_path),
            "period": period,
            "period_start_utc": str(_PERIOD_START[period]),
            "n_points": len(window),
            "sampling": "1 s (per-second mean, last observation carried forward)",
            "n_entities": len(names),
            "process_variables_per_subsystem": n_pvs,
            "subsystems_without_usable_pv": [s for s in graph_subsystems if s not in names],
            "subsystem_summary": "first principal component of the "
                                 "standardized process variables",
            "edge_strength": edge_strength,
            "n_causal_links": len(causal_pairs_idx),
            "ground_truth_type": "subsystem-level causal graph built by the "
                                  "plant's system experts (expert knowledge, "
                                  "with feedback loops; not validated by "
                                  "intervention)",
        }

        print(f"  [real-benchmark] {name} loaded — {len(names)} subsystems "
              f"({sum(n_pvs.values())} process variables, first principal "
              f"component each) x {len(window)} points (1s) | GT: "
              f"{len(causal_pairs_idx)} links (expert subsystem graph, "
              f"{edge_strength} edges, Mogensen et al. 2024)")

        scenarios.append({"name": name, "df": df_long, "gt": df_gt,
                          "_scenario": None, "_params": params})

    return scenarios


if __name__ == "__main__":
    for c in generate_ess_cryo_benchmark_scenarios():
        n_edges = int(c["gt"]["causal_gt"].sum())
        print(f"  {c['name']}: df={c['df'].shape}, {n_edges}/{len(c['gt'])} real edges")
