"""
Data.dream3
===============================================================================
QUASI-REAL BENCHMARK (SIMULATED) — DREAM3 IN SILICO NETWORK CHALLENGE
(Prill, Marbach, Saez-Rodriguez, Sorger, Alexopoulos, Xue, Clarke, Altan-
Bonnet & Stolovitzky 2010, "Towards a rigorous assessment of systems biology
models: the DREAM3 challenges", PLoS ONE 5(2):e9202; networks and simulator
by Marbach, Schaffter, Mattiussi & Floreano 2009, J. Comput. Biol. 16(2)).

Simulated gene-expression time series: network topologies extracted from
REAL transcriptional regulatory networks (E. coli and S. cerevisiae) with
nonlinear kinetic models of transcription and translation plus noise. With
NetSim, one of the most used benchmarks of time-series causal discovery
(Neural Granger causality, eSRU, CUTS, CUTS+, ...). It is the predecessor of
DREAM4, with the same kind of data.

DATA ORIGIN: SIMULATED, not measured.

Five networks (Ecoli1, Ecoli2, Yeast1, Yeast2, Yeast3) in three sizes. Each
trajectory has 21 time points; the number of trajectories grows with the
size, and they are CONCATENATED into one series, as done by the works above:

    size 10     4 trajectories   ->   84 points x  10 genes
    size 50    23 trajectories   ->  483 points x  50 genes
    size 100   46 trajectories   ->  966 points x 100 genes  (the size used
                                       by most of that literature)

GROUND TRUTH: the regulatory network used to simulate each dataset
(regulator -> target), known BY CONSTRUCTION. The file lists only the true
edges, one "regulator <TAB> target <TAB> sign" per line. It contains
feedback loops.

CAVEATS
-------------------------------------------------------------------------------
  - Short trajectories glued together: the series jumps every 21 points,
    where one trajectory ends and the next (a different perturbation)
    begins. A lag taken across a junction mixes unrelated trajectories.
  - COST. The pipeline was designed for tens of series. Size 100 means
    9,900 directed pairs and 966 points: every model becomes far more
    expensive (and some baselines may not finish). Size 10 runs in minutes.

HOW THE DATA IS OBTAINED
-------------------------------------------------------------------------------
Automatically downloads (with on-disk caching) the copy distributed with
eSRU (Khanna & Tan 2020, MIT licence), the one those works share:
    https://github.com/sakhanna/SRU_for_GCI/tree/master/data/dream3
        Dream3TensorData/Size<S><Network>.pt     torch tensor 'TsData'
        TrueGeneNetworks/InSilicoSize<S>-<Network>.tsv
The .pt tensor is converted once to a CSV next to it (this conversion is
the only step that needs torch); later loads read the CSV.
"""
from __future__ import annotations

import os
import urllib.request as _urlreq_bench
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base import array_to_pipeline_format_single_property, register

_BASE_URL = "https://raw.githubusercontent.com/sakhanna/SRU_for_GCI/master/data/dream3"
_BASE_DEFAULT = Path(os.environ.get("DREAM3_DIR", "./external_data/dream3"))
_NETWORKS = ("Ecoli1", "Ecoli2", "Yeast1", "Yeast2", "Yeast3")
_SIZES = (10, 50, 100)
_POINTS_PER_TRAJECTORY = 21


def _cached(path: Path, url: str, timeout: int = 120) -> Path:
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


def _load_timeseries(root: Path, size: int, network: str) -> np.ndarray:
    """(points x genes) array; the .pt tensor is converted to CSV once."""
    stem = f"Size{size}{network}"
    csv_path = root / "Dream3TensorData" / f"{stem}.csv"
    if csv_path.exists():
        return np.loadtxt(csv_path, delimiter=",")
    pt_path = _cached(root / "Dream3TensorData" / f"{stem}.pt",
                      f"{_BASE_URL}/Dream3TensorData/{stem}.pt")
    import torch  # only needed for this one-time conversion
    data = torch.load(pt_path, weights_only=False)["TsData"].numpy().astype(float)
    np.savetxt(csv_path, data, delimiter=",")
    return data


def _load_gold_standard(root: Path, size: int, network: str, genes: List[str]) -> set:
    name = f"InSilicoSize{size}-{network}.tsv"
    path = _cached(root / "TrueGeneNetworks" / name, f"{_BASE_URL}/TrueGeneNetworks/{name}")
    table = pd.read_csv(path, sep="\t", header=None,
                        names=["regulator", "target", "sign"])
    unknown = (set(table["regulator"]) | set(table["target"])) - set(genes)
    if unknown:
        raise ValueError(f"'{path}' references unknown genes: {sorted(unknown)}.")
    return {(genes.index(a), genes.index(b))
            for a, b in zip(table["regulator"], table["target"]) if a != b}


@register("dream3")
def generate_dream3_benchmark_scenarios(
    sizes: Sequence[int] = (10,),
    networks: Sequence[str] = _NETWORKS,
    base_dir: Optional[Union[str, Path]] = None,
    start_date: str = "2015-01-01",
    freq: str = "D",
    name_prefix: str = "benchmark_dream3",
) -> List[dict]:
    """
    Loads the DREAM3 in silico time series, one scenario per (size, network)
    — see the module docstring for the source, the ground truth and the cost
    of the larger sizes.

    Parameters
    ----------
    sizes       : network sizes to load, among 10, 50 and 100.
    networks    : among "Ecoli1", "Ecoli2", "Yeast1", "Yeast2", "Yeast3".
    base_dir    : download/cache folder (default: env DREAM3_DIR or
                  "./external_data/dream3").
    start_date, freq : synthetic dates for the 'Date' column (the simulated
                  time unit is arbitrary).
    name_prefix : prefix for the name/folder of each generated scenario.
    """
    root = Path(base_dir) if base_dir is not None else _BASE_DEFAULT

    scenarios: List[dict] = []
    for size in sizes:
        if size not in _SIZES:
            raise ValueError(f"size={size} is invalid. Use one of {_SIZES}.")
        for network in networks:
            if network not in _NETWORKS:
                raise ValueError(
                    f"network='{network}' is invalid. Use one of {_NETWORKS}.")
            data = _load_timeseries(root, size, network)
            if data.shape[1] != size or data.shape[0] % _POINTS_PER_TRAJECTORY:
                raise ValueError(
                    f"DREAM3 size {size} {network}: unexpected shape {data.shape}.")
            genes = [f"G{i}" for i in range(1, size + 1)]
            causal_pair_idx = _load_gold_standard(root, size, network, genes)

            df_long, df_gt = array_to_pipeline_format_single_property(
                data, genes, causal_pair_idx,
                start_date=start_date, freq=freq, property_name="EXPR",
            )

            name = f"{name_prefix}_size{size}_{network}"
            params = {
                "source": "dream3",
                "reference": "Prill et al. 2010, PLoS ONE 5(2):e9202 (DREAM3 "
                             "In Silico Network Challenge); Marbach et al. "
                             "2009, J. Comput. Biol. 16(2)",
                "data_origin": "https://github.com/sakhanna/SRU_for_GCI "
                               "(data/dream3)",
                "size": size,
                "network": network,
                "mode": "concatenated",
                "n_trajectories": data.shape[0] // _POINTS_PER_TRAJECTORY,
                "points_per_trajectory": _POINTS_PER_TRAJECTORY,
                "n_points": data.shape[0],
                "n_entities": size,
                "n_causal_links": len(causal_pair_idx),
                "simulated": True,
                "ground_truth_type": "regulatory network used by the "
                                      "simulator (by construction); topology "
                                      "extracted from real E. coli / yeast "
                                      "networks, contains feedback loops",
            }

            print(f"  [benchmark] {name} loaded — {size} genes x "
                  f"{data.shape[0]} points ({params['n_trajectories']} "
                  f"trajectories of {_POINTS_PER_TRAJECTORY}, concatenated) | "
                  f"{len(causal_pair_idx)} causal links (simulator's "
                  f"network, DREAM3)")

            scenarios.append({"name": name, "df": df_long, "gt": df_gt,
                              "_scenario": None, "_params": params})

    return scenarios


if __name__ == "__main__":
    for c in generate_dream3_benchmark_scenarios(sizes=_SIZES):
        n_edges = int(c["gt"]["causal_gt"].sum())
        print(f"  {c['name']}: df={c['df'].shape}, {n_edges}/{len(c['gt'])} real edges")
