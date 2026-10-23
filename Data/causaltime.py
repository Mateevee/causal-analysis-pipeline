"""
Data.causaltime
===============================================================================
Provider of scenarios from CausalTime (domains "medical", "pm25",
"traffic"), in the SAME format used by the other providers in the Data
package (see Data.base for the contract: ScenarioEntry, run_all()/validate_dataset()).

Reuses the generic converter already used by lorenz96 and logistic_maps
(`array_to_pipeline_format_single_property`), since each CausalTime domain
is also a (T, N) array with a single "property" per node/entity.

HOW TO GET THE DATA
-------------------------------------------------------------------------------
CausalTime (Cheng et al., 2024, "CausalTime: Realistically Generated Time-series
for Benchmarking Causal Discovery") is NOT bundled with this repository and is
NOT auto-downloaded -- it must be obtained manually:

    1. Go to https://www.causaltime.cc/dataset/ and download the generated
       dataset (the "medical", "pm25", and "traffic" domains). This is the
       "Dataset Download" link of the CausalTime section of the authors'
       repository, https://github.com/jarrycyx/UNN (the generation code is
       in its CausalTime/ folder). The site is sometimes unreachable
       (connection timeout); if so, try again later or from another
       network -- and keep a backup copy of the six .npy files once you
       have them (they total ~12 MB).
    2. Place the downloaded content so that each domain ends up in its own
       subfolder under the base directory (default:
       "Data/CausalTimeData/Dataset/", relative to the repository root --
       see Data/CausalTimeData/Readme.txt), one level per domain:

           Data/CausalTimeData/Dataset/medical/gen_data.npy
           Data/CausalTimeData/Dataset/medical/graph.npy
           Data/CausalTimeData/Dataset/pm25/gen_data.npy
           Data/CausalTimeData/Dataset/pm25/graph.npy
           Data/CausalTimeData/Dataset/traffic/gen_data.npy
           Data/CausalTimeData/Dataset/traffic/graph.npy

    3. To use a different location instead, either pass `base_dir=` to
       generate_causaltime_benchmark_scenarios()/set CAUSALTIME_DIR_PATH in
       Config/execution.py, or set the CAUSALTIME_DIR environment variable
       (see BASE PATH below for the exact priority order). The entry points
       (main.py, main_sensitivity.py, ...) always pass
       base_dir=CAUSALTIME_DIR_PATH, so for them the environment variable
       has no effect -- edit Config/execution.py.

If a domain's files are missing, this loader raises FileNotFoundError with
the exact path it looked for and repeats these instructions (see
_load_domain_raw() below) -- it does not fail with a bare numpy traceback.

SOURCE DATA FORMAT
-------------------------------------------------------------------------------
Each <base_dir>/<domain>/ folder contains:

    gen_data.npy : (Sample_num, Time_step, n_columns)
    graph.npy    : (Node_num, Node_num)  -- adjacency matrix

If n_columns == Node_num, all columns are used. If n_columns == 2*Node_num,
only the first Node_num are kept (corresponding to the graph -- see the note
in CausalTime's original load_data.py: "graph.npy is only the upper-left
corner of the full adjacency matrix").

WARNING -- the samples axis (Sample_num) is the trickiest part of this
source:
- If the Sample_num samples are CONTIGUOUS blocks in time (one long series
  cut into pieces), concatenating them makes sense (mode="concatenated").
- If each sample is an INDEPENDENT REALIZATION (bootstrap/replicas),
  concatenating introduces an artificial break in the series -- treat each
  sample as its own "run", in the same spirit as the lorenz96 /
  tigramite_scm scenarios (several independent seeds/runs). That is why the
  default here is mode="separate_samples": each sample becomes its own
  ScenarioEntry, with the same ground truth (the domain's graph).

There is also NO confirmation here that graph[i, j] means "i causes j" (as
opposed to "j causes i") -- the convention used is source_entity=i,
target_entity=j, causal_gt=graph[i, j]; confirm this orientation before
using the results for validation.

BASE PATH
-------------------------------------------------------------------------------
This source only exists on machines that have the CausalTime .npy files
downloaded (see HOW TO GET THE DATA above). The path is resolved in order of
priority: the `base_dir` parameter, the CAUSALTIME_DIR environment variable,
or the default "Data/CausalTimeData/Dataset" (relative to the repository
root).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .base import ScenarioEntry, register, array_to_pipeline_format_single_property

_BASE_DEFAULT = Path(os.environ.get(
    "CAUSALTIME_DIR",
    "Data/CausalTimeData/Dataset",
))

_DOWNLOAD_URL = "https://www.causaltime.cc/dataset/"

DEFAULT_DOMAINS = ["medical", "pm25", "traffic"]


def _load_domain_raw(folder: Path) -> tuple:
    """Reads gen_data.npy/graph.npy from a domain folder and returns
    (useful_data, graph) with any extra columns (if present) already dropped."""
    data_path = folder / "gen_data.npy"
    graph_path = folder / "graph.npy"
    if not data_path.exists() or not graph_path.exists():
        missing = [p.name for p in (data_path, graph_path) if not p.exists()]
        raise FileNotFoundError(
            f"CausalTime file(s) not found in '{folder}': {missing}. "
            f"Download the dataset from {_DOWNLOAD_URL} and place "
            f"gen_data.npy/graph.npy for this domain in that folder "
            f"(see the 'HOW TO GET THE DATA' section in this module's "
            f"docstring, Data/causaltime.py)."
        )
    data = np.load(data_path)   # (Sample_num, Time_step, n_columns)
    graph = np.load(graph_path) # (Node_num, Node_num)

    n_samples, n_steps, n_columns = data.shape
    n_nodes = graph.shape[0]

    if n_columns == n_nodes:
        useful_data = data
    elif n_columns == 2 * n_nodes:
        useful_data = data[:, :, :n_nodes]
    else:
        raise ValueError(
            f"Unexpected shape in '{folder}': n_columns={n_columns} does not "
            f"match n_nodes={n_nodes} nor 2*n_nodes={2 * n_nodes}."
        )

    return useful_data, graph


def _causal_pair_idx_from_graph(graph: np.ndarray) -> set:
    n_nodes = graph.shape[0]
    return {
        (i, j)
        for i in range(n_nodes)
        for j in range(n_nodes)
        if i != j and bool(graph[i, j])
    }


@register("causaltime")
def generate_causaltime_benchmark_scenarios(
    *,
    domains: Optional[List[str]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    mode: str = "separate_samples",
    max_samples: Optional[int] = None,
    start_date: str = "2015-01-01",
    freq: str = "H",
    property_name: str = "VAL",
    name_prefix: str = "benchmark_causaltime",
) -> List[dict]:
    """
    Generates scenarios from the CausalTime domains, in the format consumed
    by run_all()/validate_dataset() (the same signature described in Data.base).

    Parameters
    ----------
    domains       : which subfolders to load (default: medical/pm25/traffic).
    base_dir      : root where the <domain>/gen_data.npy and graph.npy folders
                     live (default: the CAUSALTIME_DIR env var or
                     Data/CausalTimeData/Dataset -- see the module docstring).
    mode          : "separate_samples" (default, each sample becomes its own
                     ScenarioEntry with the same ground truth for the domain)
                     or "concatenated" (all samples glued into a single long
                     series -- only makes sense if they are contiguous blocks
                     in time, see the docstring).
    max_samples   : caps how many samples to process per domain (useful to
                     avoid generating hundreds of scenarios if Sample_num is
                     high). None = all of them.
    start_date, freq, property_name : passed through to
                     array_to_pipeline_format_single_property() (synthetic
                     dates, since CausalTime does not carry real timestamps).
    name_prefix   : prefix for the name/folder of each generated scenario.

    Returns a list of dicts (ScenarioEntry.to_dict()).
    """
    if mode not in ("separate_samples", "concatenated"):
        raise ValueError(
            f"mode='{mode}' is invalid. Use 'separate_samples' or 'concatenated'."
        )

    root = Path(base_dir) if base_dir is not None else _BASE_DEFAULT
    domains = domains if domains is not None else DEFAULT_DOMAINS

    scenarios: List[dict] = []

    for domain in domains:
        folder = root / domain
        useful_data, graph = _load_domain_raw(folder)
        n_samples, n_steps, n_nodes = useful_data.shape
        entity_names = [f"var_{i}" for i in range(n_nodes)]
        causal_pair_idx = _causal_pair_idx_from_graph(graph)

        base_params = {
            "source": "causaltime",
            "domain": domain,
            "mode": mode,
            "n_nodes": n_nodes,
            "n_available_samples": n_samples,
            "n_steps_per_sample": n_steps,
        }

        if mode == "concatenated":
            # (Sample_num, Time_step, N) -> (Sample_num * Time_step, N),
            # gluing the samples in order -- only valid if they are
            # contiguous blocks in time (see WARNING in the module docstring).
            long_series = useful_data.reshape(-1, n_nodes)
            df_long, df_gt = array_to_pipeline_format_single_property(
                long_series, entity_names, causal_pair_idx,
                start_date=start_date, freq=freq, property_name=property_name,
            )
            scenarios.append(ScenarioEntry(
                name=f"{name_prefix}_{domain}_concatenated",
                df=df_long,
                gt=df_gt,
                _scenario=domain,
                _params={**base_params, "sample": None},
            ).to_dict())

        else:  # "separate_samples"
            limit = n_samples if max_samples is None else min(max_samples, n_samples)
            for k in range(limit):
                df_long, df_gt = array_to_pipeline_format_single_property(
                    useful_data[k], entity_names, causal_pair_idx,
                    start_date=start_date, freq=freq, property_name=property_name,
                )
                scenarios.append(ScenarioEntry(
                    name=f"{name_prefix}_{domain}_sample{k}",
                    df=df_long,
                    gt=df_gt,
                    _scenario=domain,
                    _params={**base_params, "sample": k},
                ).to_dict())

    return scenarios


if __name__ == "__main__":
    # Quick manual test (needs the .npy files in CAUSALTIME_DIR / the default path).
    for domain in DEFAULT_DOMAINS:
        try:
            scenarios = generate_causaltime_benchmark_scenarios(
                domains=[domain], max_samples=2,
            )
        except Exception as e:
            print(f"[{domain}] failed: {e}")
            continue
        for c in scenarios:
            n_edges = int(c["gt"]["causal_gt"].sum())
            print(f"  {c['name']}: df={c['df'].shape}, "
                  f"{n_edges}/{len(c['gt'])} real edges")
