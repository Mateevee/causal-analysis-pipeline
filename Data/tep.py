"""
Data.tep
===============================================================================
INDUSTRIAL BENCHMARK (SIMULATED) — TENNESSEE EASTMAN PROCESS (TEP), via
CIPCaD-Bench (Menegozzo, Dall'Alba & Fiorini 2022, "CIPCaD-Bench: Continuous
Industrial Process datasets for benchmarking Causal Discovery methods", IEEE
CASE 2022, arXiv:2208.01529).

The TEP is a classic chemical engineering benchmark (Downs & Vogel 1993,
"A plant-wide industrial process control problem") -- a non-linear and
strongly coupled process, with 5 units (two-phase reactor, condenser, recycle
compressor, vapor-liquid separator, stripper) and multiple interconnected PID
control loops. It covers a continuous industrial domain, complementing
climate, health-traffic-PM2.5 via CausalTime, and synthetic sources.

DATA ORIGIN: SIMULATED, NOT MEASURED. The TEP is a simulator of a real
Eastman Chemical plant; the series are simulator output, not plant
measurements (the CIPCaD-Bench abstract itself: "the well-known Tennessee
Eastman simulator" -- the measured dataset of that paper is its second one,
the ultra-processed food plant). Report it as a realistic simulated process,
not as real-world data.

CIPCaD-Bench re-elaborates normal-operation data (steady-state, WITHOUT
injected faults -- original source: Manca 2020, IEEE DataPort,
dx.doi.org/10.21227/326k-qr90) and, the main differentiator of this source,
derives a GROUND-TRUTH CAUSAL GRAPH from the PROCESS TOPOLOGY (P&ID) and the
plant's control loops -- therefore BY CONSTRUCTION (engineering knowledge),
not statistically estimated. 33 variables, 32 edges.

HOW TO GET THE DATA
-------------------------------------------------------------------------------
Not auto-downloaded. Download the 2 files from the official CIPCaD-Bench
repository and save them into <base_dir>/ (default: env TEP_DIR or
"./external_data/tep/"), with these exact names:

    https://github.com/giovanniMen/CPCaD-Bench/tree/main/TenneseeEastman
        datasetTE.csv       (1499 samples x 31 variables -- subsampled by
                              median over a 3-min window across ~75h of
                              normal operation)
        TEGroundTruth.txt   (33x33 adjacency matrix, space/tab-separated
                              text; graph[i, j] == 1 means Xi -> Xj; 32 edges
                              total, null diagonal)

Direct (raw) links -- e.g. `curl -L -o <file> <url>` from inside the folder:
    https://raw.githubusercontent.com/giovanniMen/CPCaD-Bench/main/TenneseeEastman/datasetTE.csv
    https://raw.githubusercontent.com/giovanniMen/CPCaD-Bench/main/TenneseeEastman/TEGroundTruth.txt
The other files of that folder (datasetTENumpy.txt, *.svg) are not used.

FORMAT
-------------------------------------------------------------------------------
33 variables in the ground-truth graph: X1-X22 = XMEAS(1..22) (continuous
measurements -- pressures, levels, temperatures, flows, concentrations) and
X23-X33 = XMV(1..11) (manipulated variables -- valve positions/flow
setpoints). See TAG_INFO below for the original tag (Downs & Vogel 1993),
description and unit of each variable (Table II of the CIPCaD-Bench paper).

datasetTE.csv only has 31 columns: the authors removed X27 (XMV(5), Compressor
Recycle Valve) and X31 (XMV(9), Stripper Steam Valve) due to zero variance
under normal, fault-free operation (these valves only move when faults are
active, out of scope for this source). This loader restricts the ground truth
to the variables actually present in the data (it projects the 33x33 matrix
by removing the rows/columns for X27 and X31) -- the pipeline contract
requires that `gt` only reference entities that exist in `df`.

There are no fault scenarios (IDV1-20) in this source -- only normal
(fault-free) operation. The ground truth is by construction (P&ID topology +
plant control loops), not estimated; see Figure 2 / Table II of the paper.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import ScenarioEntry, register, array_to_pipeline_format_single_property

_BASE_DEFAULT = Path(os.environ.get("TEP_DIR", "./external_data/tep"))

_N_VARS_GROUND_TRUTH = 33

# Table II of the CIPCaD-Bench paper (Menegozzo et al. 2022) -- original TEP
# tag (Downs & Vogel 1993), description and unit for each of the 33 variables
# of the ground-truth graph. X1-X22 = XMEAS (continuous measurements,
# separated by a line in the original table); X23-X33 = XMV (manipulated).
TAG_INFO: Dict[str, Tuple[str, str, str]] = {
    "X1":  ("XMEAS(1)",  "A Feed (stream 1)", "kscmh"),
    "X2":  ("XMEAS(2)",  "D Feed (stream 2)", "kg/hr"),
    "X3":  ("XMEAS(3)",  "E Feed (stream 3)", "kg/hr"),
    "X4":  ("XMEAS(4)",  "A and C Feed (stream 4)", "kscmh"),
    "X5":  ("XMEAS(5)",  "Recycle Flow (stream 8)", "kscmh"),
    "X6":  ("XMEAS(6)",  "Reactor Feed Rate (stream 6)", "kscmh"),
    "X7":  ("XMEAS(7)",  "Reactor Pressure", "kPa gauge"),
    "X8":  ("XMEAS(8)",  "Reactor Level", "%"),
    "X9":  ("XMEAS(9)",  "Reactor Temperature", "Deg C"),
    "X10": ("XMEAS(10)", "Purge Rate (stream 9)", "kscmh"),
    "X11": ("XMEAS(11)", "Product Sep Temp", "Deg C"),
    "X12": ("XMEAS(12)", "Product Sep Level", "%"),
    "X13": ("XMEAS(13)", "Prod Sep Pressure", "kPa gauge"),
    "X14": ("XMEAS(14)", "Prod Sep Underflow (stream 10)", "m3/hr"),
    "X15": ("XMEAS(15)", "Stripper Level", "%"),
    "X16": ("XMEAS(16)", "Stripper Pressure", "kPa gauge"),
    "X17": ("XMEAS(17)", "Stripper Underflow (stream 11)", "m3/hr"),
    "X18": ("XMEAS(18)", "Stripper Temperature", "Deg C"),
    "X19": ("XMEAS(19)", "Stripper Steam Flow", "kg/hr"),
    "X20": ("XMEAS(20)", "Compressor Work", "kW"),
    "X21": ("XMEAS(21)", "Reactor Cooling Water Outlet Temp", "Deg C"),
    "X22": ("XMEAS(22)", "Separator Cooling Water Outlet Temp", "Deg C"),
    "X23": ("XMV(1)",  "D Feed Flow (stream 2)", "%"),
    "X24": ("XMV(2)",  "E Feed Flow (stream 3)", "%"),
    "X25": ("XMV(3)",  "A Feed Flow (stream 1)", "%"),
    "X26": ("XMV(4)",  "A and C Feed Flow (stream 4)", "%"),
    "X27": ("XMV(5)",  "Compressor Recycle Valve", "%"),
    "X28": ("XMV(6)",  "Purge Valve (stream 9)", "%"),
    "X29": ("XMV(7)",  "Separator Pot Liquid Flow (stream 10)", "%"),
    "X30": ("XMV(8)",  "Stripper Liquid Product Flow (stream 11)", "%"),
    "X31": ("XMV(9)",  "Stripper Steam Valve", "%"),
    "X32": ("XMV(10)", "Reactor Cooling Water Flow", "%"),
    "X33": ("XMV(11)", "Condenser Cooling Water Flow", "%"),
}

VARIABLES_REMOVED_ZERO_VARIANCE = ["X27", "X31"]


def _load_ground_truth(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=int)
    if matrix.shape != (_N_VARS_GROUND_TRUTH, _N_VARS_GROUND_TRUTH):
        raise ValueError(
            f"'{path}' has unexpected shape {matrix.shape}, expected "
            f"({_N_VARS_GROUND_TRUTH}, {_N_VARS_GROUND_TRUTH})."
        )
    return matrix


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    invalid_columns = [c for c in df.columns if c not in TAG_INFO]
    if invalid_columns:
        raise ValueError(
            f"'{path}' has unrecognized columns (outside X1..X33): "
            f"{invalid_columns}"
        )
    return df


@register("tep")
def generate_tep_benchmark_scenarios(
    *,
    base_dir: Optional[Union[str, Path]] = None,
    start_date: str = "2015-01-01",
    freq: str = "3min",
    name: str = "benchmark_tep_normal",
) -> List[dict]:
    """
    Loads the TEP (Tennessee Eastman Process) benchmark via CIPCaD-Bench, in
    the format consumed by run_all()/validate_dataset() (same signature
    described in Data.base). See the module docstring for the data source,
    the format and the provenance of the ground truth.

    Parameters
    ----------
    base_dir : root folder where 'datasetTE.csv' and 'TEGroundTruth.txt'
               live (default: env TEP_DIR or "./external_data/tep").
    start_date, freq : synthetic dates for the 'Date' column (the TEP does
               not carry real timestamps); default freq "3min" reflects the
               median subsampling over a 3-min window already applied by
               CIPCaD-Bench.
    name     : name/folder of the generated scenario (single source, no
               seeds -- already-fixed simulator output).

    Returns a list with ONE dict (ScenarioEntry.to_dict()).
    """
    root = Path(base_dir) if base_dir is not None else _BASE_DEFAULT
    csv_path = root / "datasetTE.csv"
    gt_path = root / "TEGroundTruth.txt"

    if not csv_path.exists() or not gt_path.exists():
        raise FileNotFoundError(
            f"TEP (CIPCaD-Bench) files not found in '{root}'. "
            f"Download 'datasetTE.csv' and 'TEGroundTruth.txt' from "
            f"https://github.com/giovanniMen/CPCaD-Bench/tree/main/TenneseeEastman "
            f"and save them in that folder (or point base_dir= / the TEP_DIR "
            f"environment variable)."
        )

    df_raw = _load_data(csv_path)
    full_gt = _load_ground_truth(gt_path)

    entity_names = list(df_raw.columns)  # 31 of the 33 (X27/X31 removed -- zero variance)
    # 'X7' -> index 6 in the 33x33 matrix (order X1..X33)
    matrix_indices = [int(var_name[1:]) - 1 for var_name in entity_names]

    causal_pairs_idx = set()
    for a, i in enumerate(matrix_indices):
        for b, j in enumerate(matrix_indices):
            if i != j and full_gt[i, j] == 1:
                causal_pairs_idx.add((a, b))

    data = df_raw.to_numpy(dtype=float)
    df_long, df_gt = array_to_pipeline_format_single_property(
        data, entity_names, causal_pairs_idx,
        start_date=start_date, freq=freq, property_name="VAL",
    )

    n_edges_removed = int(full_gt.sum()) - len(causal_pairs_idx)
    params = {
        "source": "tep_cipcadbench",
        "reference": "Menegozzo, Dall'Alba & Fiorini 2022, IEEE CASE "
                     "(arXiv:2208.01529); original process: Downs & Vogel "
                     "1993, Computers & Chemical Engineering 17(3):245-255",
        "data_file": str(csv_path),
        "ground_truth_file": str(gt_path),
        "data_origin": "simulated (Tennessee Eastman simulator output, "
                       "not plant measurements)",
        "n_points": data.shape[0],
        "n_entities": len(entity_names),
        "n_causal_links": len(causal_pairs_idx),
        "n_causal_links_full_graph_33var": int(full_gt.sum()),
        "variables_removed_zero_variance": VARIABLES_REMOVED_ZERO_VARIANCE,
        "ground_truth_type": "P&ID topology + plant control loops "
                              "(by construction, not estimated)",
        "fault_scenarios_included": False,
    }

    if n_edges_removed:
        print(f"  [WARNING] {name}: {n_edges_removed} edge(s) from the full "
              f"ground truth (33 var) discarded for involving "
              f"{VARIABLES_REMOVED_ZERO_VARIANCE} (absent from the data).")

    print(f"  [simulated-benchmark] {name} loaded — {len(entity_names)} "
          f"variables (XMEAS/XMV) x {data.shape[0]} points | "
          f"{len(causal_pairs_idx)} causal links (ground truth via P&ID, "
          f"CIPCaD-Bench)")

    return [ScenarioEntry(
        name=name, df=df_long, gt=df_gt, _scenario="tep_normal",
        _params=params,
    ).to_dict()]


if __name__ == "__main__":
    # Quick manual test (needs the files in TEP_DIR / the default path).
    scenarios = generate_tep_benchmark_scenarios()
    for c in scenarios:
        n_edges = int(c["gt"]["causal_gt"].sum())
        print(f"  {c['name']}: df={c['df'].shape}, {n_edges}/{len(c['gt'])} real edges")
