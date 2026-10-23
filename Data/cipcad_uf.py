"""
Data.cipcad_uf
===============================================================================
REAL BENCHMARK — ULTRA-PROCESSED FOOD PLANT (UF), via CIPCaD-Bench
(Menegozzo, Dall'Alba & Fiorini 2022, "CIPCaD-Bench: Continuous Industrial
Process datasets for benchmarking Causal Discovery methods", IEEE CASE 2022,
arXiv:2208.01529) -- the second dataset of the same benchmark that already
provides the TEP (see Data/tep.py).

Unlike the TEP (a simulator), the UF data comes from a RUNNING industrial
plant: 17 anonymized variables from different production machines, sampled
every 5 minutes (23,132 samples, May/2020 - Sep/2021), with outliers,
unobserved confounders and no causal sufficiency (Table III of the paper).

VARIABLES (Section IV-B of the paper; names are anonymized by the authors)
-------------------------------------------------------------------------------
    X1        production flow
    X2..X14   parameters of 4 consecutive process phases/machines
              ({X2..X8}, {X9, X10}, {X11, X12}, {X13, X14})
    X15, X16  external variables, not controlled by the process
    X17       quality variable of the final product (affected by all others)

The full process takes about 4 hours: an intervention on X2 reaches X17
with a delay of 32-48 samples (160-240 min, Figure 3 of the paper).

GROUND TRUTH -- two versions, both BY CONSTRUCTION (production flow)
-------------------------------------------------------------------------------
Since the plant is a real production, the authors cannot trace every
relation; what IS known is the ORDER of the machines along the production
flow. Relations between parameters of the SAME machine have no certain
direction, hence two graphs:

    "UF"     (UFGroundTruth.txt, 83 edges)     only relations BETWEEN
             machines (upstream -> downstream) -- measures how well a method
             finds the relevant cross-machine relations.
    "UFIMD"  (UFIMDGroundTruth.txt, 133 edges in the published file; the
             paper's text says 108)  "Internal Machine Dependencies": also
             fully connects, in BOTH directions, the parameters of the same
             machine -- every false positive against this graph is a
             violation of the process flow.

graph[i, j] == 1 means Xi -> Xj (same convention as TEGroundTruth.txt).

PREPROCESSING ALREADY APPLIED BY THE AUTHORS
-------------------------------------------------------------------------------
Median over a 30-minute moving window, robust scaling (median/IQR), removal
of zero-variance variables. Note the consequence of the moving median at a
5-minute sampling: each series is smoothed over 6 consecutive samples.

TIME AXIS -- NOT CONTIGUOUS
-------------------------------------------------------------------------------
The 23,132 samples are NOT one continuous series: production stops leave
160 gaps in DateTime (the largest one spans ~66 days). A lagged analysis
across a gap would mix unrelated production runs, so this loader only cuts
windows INSIDE contiguous segments (consecutive samples exactly 5 minutes
apart). Each window becomes its own scenario, with the same ground truth.

HOW TO GET THE DATA
-------------------------------------------------------------------------------
Not auto-downloaded. Download the 3 files from the official CIPCaD-Bench
repository and save them into <base_dir>/ (default: env CIPCAD_UF_DIR or
"./external_data/cipcad_uf/"), with these exact names:

    https://github.com/giovanniMen/CPCaD-Bench/tree/main/UltraProcessed_Food
        DatasetUF.csv           (DateTime, X1..X17)
        UFGroundTruth.txt       (17x17 adjacency matrix, tab-separated)
        UFIMDGroundTruth.txt    (17x17 adjacency matrix, tab-separated)

Direct (raw) links -- e.g. `curl -L -o <file> <url>` from inside the folder:
    https://raw.githubusercontent.com/giovanniMen/CPCaD-Bench/main/UltraProcessed_Food/DatasetUF.csv
    https://raw.githubusercontent.com/giovanniMen/CPCaD-Bench/main/UltraProcessed_Food/UFGroundTruth.txt
    https://raw.githubusercontent.com/giovanniMen/CPCaD-Bench/main/UltraProcessed_Food/UFIMDGroundTruth.txt
The other files of that folder (DatasetUFNumpy.txt, *.svg) are not used.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .base import (
    ScenarioEntry,
    register,
    array_to_pipeline_format_single_property,
    drop_constant_columns,
)

_BASE_DEFAULT = Path(os.environ.get("CIPCAD_UF_DIR", "./external_data/cipcad_uf"))

_N_VARS = 17
_SAMPLING = pd.Timedelta("5min")
_GT_FILES = {"UF": "UFGroundTruth.txt", "UFIMD": "UFIMDGroundTruth.txt"}

# Section IV-B / Figure 3 of the CIPCaD-Bench paper. The authors anonymized
# the variables; only the role of each one in the production flow is public.
VARIABLE_ROLES = {
    "X1": "production flow",
    **{f"X{i}": "process phase 1 (machine 1)" for i in range(2, 9)},
    "X9": "process phase 2 (machine 2)", "X10": "process phase 2 (machine 2)",
    "X11": "process phase 3 (machine 3)", "X12": "process phase 3 (machine 3)",
    "X13": "process phase 4 (machine 4)", "X14": "process phase 4 (machine 4)",
    "X15": "external variable (not controlled)",
    "X16": "external variable (not controlled)",
    "X17": "final product quality",
}


def _contiguous_windows(timestamps: pd.Series, n_points: int) -> List[int]:
    """Start indices of the non-overlapping windows of `n_points` samples
    that fit entirely inside a contiguous segment (no gap in DateTime), in
    chronological order."""
    gaps = timestamps.diff() != _SAMPLING
    segment_starts = np.flatnonzero(gaps.to_numpy())  # index 0 is always a start
    segment_ends = np.r_[segment_starts[1:], len(timestamps)]
    starts = []
    for a, b in zip(segment_starts, segment_ends):
        starts.extend(range(a, b - n_points + 1, n_points))
    return starts


@register("cipcad_uf")
def generate_cipcad_uf_benchmark_scenarios(
    *,
    base_dir: Optional[Union[str, Path]] = None,
    ground_truth: str = "UF",
    n_points: int = 400,
    max_windows: Optional[int] = 3,
    name_prefix: str = "benchmark_cipcad_uf",
) -> List[dict]:
    """
    Loads the Ultra-processed Food plant benchmark (CIPCaD-Bench), in the
    format consumed by run_all()/validate_dataset() (same signature
    described in Data.base). See the module docstring for the data source,
    the two ground truths and why the series is cut into windows.

    Parameters
    ----------
    base_dir     : folder with 'DatasetUF.csv' and the ground-truth files
                   (default: env CIPCAD_UF_DIR or "./external_data/cipcad_uf").
    ground_truth : "UF" (cross-machine relations only, 83 edges) or "UFIMD"
                   (also fully connects parameters of the same machine).
    n_points     : samples per window (5 minutes apart). The full process
                   takes ~48 samples, so keep it several times larger.
    max_windows  : how many windows/scenarios to return, spread evenly over
                   the available ones (the plant may be reconfigured between
                   weeks -- spreading covers different production runs).
                   None = all of them.
    name_prefix  : prefix for the name/folder of each generated scenario.

    Returns a list of dicts (ScenarioEntry.to_dict()), one per window.
    """
    if ground_truth not in _GT_FILES:
        raise ValueError(
            f"ground_truth='{ground_truth}' is invalid. Use one of "
            f"{sorted(_GT_FILES)}."
        )

    root = Path(base_dir) if base_dir is not None else _BASE_DEFAULT
    csv_path = root / "DatasetUF.csv"
    gt_path = root / _GT_FILES[ground_truth]

    if not csv_path.exists() or not gt_path.exists():
        raise FileNotFoundError(
            f"UF (CIPCaD-Bench) files not found in '{root}'. Download "
            f"'DatasetUF.csv' and '{gt_path.name}' from "
            f"https://github.com/giovanniMen/CPCaD-Bench/tree/main/UltraProcessed_Food "
            f"and save them in that folder (or point base_dir= / the "
            f"CIPCAD_UF_DIR environment variable)."
        )

    df_raw = pd.read_csv(csv_path, parse_dates=["DateTime"])
    entity_names = [c for c in df_raw.columns if c != "DateTime"]
    expected = [f"X{i}" for i in range(1, _N_VARS + 1)]
    if entity_names != expected:
        raise ValueError(
            f"'{csv_path}' has unexpected columns {entity_names}, expected "
            f"{expected}."
        )

    full_gt = np.loadtxt(gt_path, dtype=int)
    if full_gt.shape != (_N_VARS, _N_VARS):
        raise ValueError(
            f"'{gt_path}' has unexpected shape {full_gt.shape}, expected "
            f"({_N_VARS}, {_N_VARS})."
        )
    full_pairs = {(i, j) for i in range(_N_VARS) for j in range(_N_VARS)
                  if i != j and full_gt[i, j] == 1}

    starts = _contiguous_windows(df_raw["DateTime"], n_points)
    if not starts:
        raise ValueError(
            f"No contiguous segment of '{csv_path}' holds n_points={n_points} "
            f"samples 5 minutes apart -- reduce n_points."
        )
    if max_windows is not None and len(starts) > max_windows:
        chosen = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[k] for k in chosen]

    scenarios: List[dict] = []
    for k, start in enumerate(starts):
        window = df_raw.iloc[start:start + n_points]
        data, names, pairs, removed = drop_constant_columns(
            window[entity_names].to_numpy(dtype=float), entity_names, full_pairs,
        )
        name = f"{name_prefix}_{ground_truth.lower()}_window{k}"

        df_long, df_gt = array_to_pipeline_format_single_property(
            data, names, pairs,
            start_date=str(window["DateTime"].iloc[0]), freq="5min",
            property_name="VAL",
        )

        params = {
            "source": "cipcad_uf",
            "reference": "Menegozzo, Dall'Alba & Fiorini 2022, IEEE CASE "
                         "(arXiv:2208.01529), Ultra-processed Food dataset",
            "data_file": str(csv_path),
            "ground_truth_file": str(gt_path),
            "ground_truth_variant": ground_truth,
            "window": k,
            "start_timestamp": str(window["DateTime"].iloc[0]),
            "end_timestamp": str(window["DateTime"].iloc[-1]),
            "n_points": n_points,
            "n_entities": len(names),
            "n_causal_links": len(pairs),
            "n_causal_links_full_graph": len(full_pairs),
            "variables_removed_zero_variance": removed,
            "variable_roles": VARIABLE_ROLES,
            "ground_truth_type": "production flow / order of the machines "
                                  "(by construction, not estimated); real "
                                  "plant with unobserved confounders",
        }

        print(f"  [real-benchmark] {name} loaded — {len(names)} variables x "
              f"{n_points} points (5min) | {len(pairs)} causal links "
              f"(ground truth {ground_truth}, CIPCaD-Bench) | window "
              f"{params['start_timestamp']}..{params['end_timestamp']}"
              + (f" | removed (constant): {removed}" if removed else ""))

        scenarios.append(ScenarioEntry(
            name=name, df=df_long, gt=df_gt,
            _scenario=f"cipcad_{ground_truth.lower()}", _params=params,
        ).to_dict())

    return scenarios


if __name__ == "__main__":
    # Quick manual test (needs the files in CIPCAD_UF_DIR / the default path).
    for variant in ("UF", "UFIMD"):
        for c in generate_cipcad_uf_benchmark_scenarios(ground_truth=variant):
            n_edges = int(c["gt"]["causal_gt"].sum())
            print(f"  {c['name']}: df={c['df'].shape}, "
                  f"{n_edges}/{len(c['gt'])} real edges")
