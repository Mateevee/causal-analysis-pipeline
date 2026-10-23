"""
Data.causal_rivers
===============================================================================
REAL BENCHMARK — CAUSALRIVERS (Stein, Shadaydeh, Blunk, Penzel & Denzler
2025, "CausalRivers - Scaling up benchmarking of causal discovery for
real-world time-series", ICLR 2025).

River discharge measured at gauging stations in Germany, 15-minute
resolution, 2019-2023. This loader uses the BAVARIA split (494 stations,
490 edges, 175,296 timestamps); the benchmark also publishes "east_germany"
and "flood" splits in the same format (see `region` below).

GROUND TRUTH: the directed graph of the RIVER NETWORK itself -- an edge
u -> v means station u is the next gauged point UPSTREAM of v (water
measured at u physically flows past v). It is a graph BY CONSTRUCTION from
hydrography, built by the benchmark's authors from station coordinates,
elevation and river topology, not estimated from the series; it is a DAG
(water only flows downhill). The same caution of any real-world source
applies: rainfall is an unobserved common driver of every station in a
catchment, so neighbouring stations that are NOT connected by flow are still
strongly correlated -- which is exactly the difficulty this benchmark is
meant to expose.

SCALE / SUBSAMPLING
-------------------------------------------------------------------------------
The full split (494 stations x 175k points) is far beyond what this pipeline
was designed for (tens of entities x hundreds of points), so, in the same
spirit as Data/pems_bay.py, this loader cuts:

  - in SPACE: a CONNECTED subgraph of `n_nodes` stations per seed (random
    breadth-first growth from a random station), so the ground truth is
    dense enough to be informative. The ground truth of a scenario is the
    subgraph INDUCED on the chosen stations (direct edges only, as in the
    benchmark's own subgraph sampling).
  - in TIME: a window of `n_points` samples starting at `start`, after
    resampling the 15-minute series to `resample` (mean).

Default `resample="1h"`: on this split the cross-correlation between
upstream and downstream stations peaks at a median delay of ~4.5 h (10%-90%
range ~0.75 h to ~23 h), so 1 h keeps most edges as genuinely LAGGED
relations (1 to ~12 samples); at 6 h, ~60% of them would collapse into the
same sample.

Stations with more than `max_nan_frac` missing readings inside the window,
or constant inside it, are not eligible; the remaining gaps are linearly
interpolated per station.

HOW TO GET THE DATA
-------------------------------------------------------------------------------
Not auto-downloaded. Download `product.zip` (~570 MB) from the benchmark's
release page:

    https://github.com/CausalRivers/benchmark/releases/download/First_release/product.zip
    (release "First_release" of https://github.com/CausalRivers/benchmark)

The archive extracts into a `product/` subfolder holding the files of every
region. Move the two files of the chosen region OUT of `product/` and
directly into <base_dir>/ (default: env CAUSAL_RIVERS_DIR or
"./external_data/causal_rivers/"):

        rivers_ts_<region>.csv   ('datetime' + one column per station id)
        rivers_<region>.p        (pickled networkx.DiGraph; node = station
                                   id, node attribute 'R' = river name)

so that, for the default region, the layout is:

    ./external_data/causal_rivers/rivers_ts_bavaria.csv
    ./external_data/causal_rivers/rivers_bavaria.p

(rivers_meta_<region>.csv and the files of the other regions are not
needed; alternatively, point base_dir= / CAUSAL_RIVERS_DIR at the
extracted `product/` folder itself.)

Requires the `networkx` package (to unpickle the graph).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .base import ScenarioEntry, register, array_to_pipeline_format_single_property

_BASE_DEFAULT = Path(os.environ.get("CAUSAL_RIVERS_DIR", "./external_data/causal_rivers"))

_RAW_SAMPLING = pd.Timedelta("15min")
_RELEASE_URL = ("https://github.com/CausalRivers/benchmark/releases/download/"
                "First_release/product.zip")


def _load_window(csv_path: Path, start: pd.Timestamp, n_raw: int) -> pd.DataFrame:
    """Reads only the rows [start, start + n_raw) of the (large, ~470MB)
    time-series CSV. The file is a regular 15-minute grid, so the row offset
    of `start` is computed from the first timestamp instead of parsing the
    whole datetime column; the result is checked against `start`."""
    first = pd.Timestamp(pd.read_csv(csv_path, usecols=[0], nrows=1).iloc[0, 0])
    offset = (start - first) / _RAW_SAMPLING
    if offset < 0 or offset != int(offset):
        raise ValueError(
            f"start={start} is before the first timestamp ({first}) of "
            f"'{csv_path}' or not aligned with its 15-minute grid."
        )
    window = pd.read_csv(csv_path, index_col=0, parse_dates=True,
                         skiprows=range(1, int(offset) + 1), nrows=n_raw)
    if len(window) < n_raw:
        raise ValueError(
            f"start={start} + {n_raw} samples (15min) exceeds the end of "
            f"'{csv_path}' ({len(window)} samples available from there)."
        )
    if window.index[0] != start or window.index[-1] - start != (n_raw - 1) * _RAW_SAMPLING:
        raise ValueError(
            f"'{csv_path}' is not a regular 15-minute grid around {start} "
            f"(read {window.index[0]}..{window.index[-1]})."
        )
    window.columns = window.columns.astype(int)
    return window


def _sample_connected_subgraph(graph, eligible: set, n_nodes: int,
                                rng: np.random.Generator) -> List[int]:
    """Random breadth-first growth, ignoring edge direction, restricted to
    the eligible stations: starts at a random station whose (eligible)
    component holds at least n_nodes stations and repeatedly adds a random
    neighbour of the current set."""
    import networkx as nx

    undirected = graph.to_undirected().subgraph(eligible)
    candidates = sorted(
        node for component in nx.connected_components(undirected)
        if len(component) >= n_nodes for node in component
    )
    if not candidates:
        raise ValueError(
            f"No connected group of {n_nodes} eligible stations in this "
            f"window -- reduce n_nodes or relax max_nan_frac."
        )

    selected = [int(rng.choice(candidates))]
    while len(selected) < n_nodes:
        frontier = sorted({nb for node in selected for nb in undirected[node]}
                          - set(selected))
        selected.append(int(rng.choice(frontier)))
    return sorted(selected)


@register("causal_rivers")
def generate_causal_rivers_benchmark_scenarios(
    *,
    seeds: Optional[List[int]] = None,
    n_nodes: int = 8,
    n_points: int = 500,
    start: str = "2021-01-01",
    resample: str = "1h",
    max_nan_frac: float = 0.05,
    region: str = "bavaria",
    base_dir: Optional[Union[str, Path]] = None,
    name_prefix: str = "real_causal_rivers",
) -> List[dict]:
    """
    Generates scenarios from the CausalRivers benchmark (real river
    discharge, ground truth = river network), in the format consumed by
    run_all()/validate_dataset() (same signature described in Data.base).
    See the module docstring for the source, the ground truth and the
    subsampling in space/time.

    Parameters
    ----------
    seeds        : one scenario per seed -- each seed draws a different
                   connected subgraph of `n_nodes` stations (default
                   [1, 2, 3]). All seeds share the same time window.
    n_nodes      : stations per scenario.
    n_points     : samples per scenario AFTER resampling.
    start        : first timestamp of the window (must lie on the 15-minute
                   grid of the source).
    resample     : target resolution (pandas offset, a multiple of 15min;
                   "15min" keeps the source resolution).
    max_nan_frac : maximum fraction of missing readings a station may have
                   inside the window to be eligible.
    region       : which split's files to read ("bavaria", "east_germany",
                   "flood").
    base_dir     : folder with rivers_ts_<region>.csv and rivers_<region>.p
                   (default: env CAUSAL_RIVERS_DIR or
                   "./external_data/causal_rivers").
    name_prefix  : prefix for the name/folder of each generated scenario.

    Returns a list of dicts (ScenarioEntry.to_dict()), one per seed.
    """
    seeds = seeds if seeds is not None else [1, 2, 3]
    root = Path(base_dir) if base_dir is not None else _BASE_DEFAULT
    csv_path = root / f"rivers_ts_{region}.csv"
    graph_path = root / f"rivers_{region}.p"

    if not csv_path.exists() or not graph_path.exists():
        raise FileNotFoundError(
            f"CausalRivers files not found in '{root}'. Download "
            f"{_RELEASE_URL}, extract it and move '{csv_path.name}' and "
            f"'{graph_path.name}' out of its 'product/' subfolder into that "
            f"folder (or point base_dir= / the CAUSAL_RIVERS_DIR environment "
            f"variable at 'product/')."
        )

    step = pd.Timedelta(resample) / _RAW_SAMPLING
    if step < 1 or step != int(step):
        raise ValueError(
            f"resample='{resample}' must be a multiple of the source's "
            f"15-minute resolution."
        )
    step = int(step)

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    window = _load_window(csv_path, pd.Timestamp(start), n_points * step)

    nan_frac = window.isna().mean()
    eligible = {
        node for node in graph.nodes
        if node in window.columns
        and nan_frac[node] <= max_nan_frac
        and window[node].nunique() > 1
    }

    resampled = (
        window[sorted(eligible)]
        .interpolate(method="linear", limit_direction="both")
        .resample(resample).mean()
    )

    scenarios: List[dict] = []
    for seed in seeds:
        stations = _sample_connected_subgraph(
            graph, eligible, n_nodes, np.random.default_rng(seed),
        )
        position = {node: k for k, node in enumerate(stations)}
        causal_pairs_idx = {
            (position[u], position[v])
            for u, v in graph.subgraph(stations).edges
        }
        entity_names = [f"station_{node}" for node in stations]
        name = f"{name_prefix}_{region}_n{n_nodes}_seed{seed}"

        df_long, df_gt = array_to_pipeline_format_single_property(
            resampled[stations].to_numpy(dtype=float), entity_names,
            causal_pairs_idx,
            start_date=str(resampled.index[0]), freq=resample,
            property_name="DISCHARGE",
        )

        params = {
            "source": "causal_rivers",
            "reference": "Stein, Shadaydeh, Blunk, Penzel & Denzler 2025, "
                         "ICLR (CausalRivers)",
            "region": region,
            "data_file": str(csv_path),
            "graph_file": str(graph_path),
            "seed": seed,
            "stations": stations,
            "rivers": {f"station_{node}": str(graph.nodes[node].get("R"))
                       for node in stations},
            "n_entities": len(stations),
            "n_points": n_points,
            "resample": resample,
            "start_timestamp": str(resampled.index[0]),
            "end_timestamp": str(resampled.index[-1]),
            "max_nan_frac": max_nan_frac,
            "n_eligible_stations": len(eligible),
            "n_causal_links": len(causal_pairs_idx),
            "ground_truth_type": "river network topology (upstream station "
                                  "-> next downstream station), by "
                                  "construction; subgraph induced on the "
                                  "sampled stations. Rainfall is an "
                                  "unobserved common driver (see the module "
                                  "docstring)",
        }

        print(f"  [real-benchmark] {name} loaded — {len(stations)} stations "
              f"x {n_points} points ({resample}) | GT: {len(causal_pairs_idx)} "
              f"links (river network, CausalRivers) | window "
              f"{params['start_timestamp']}..{params['end_timestamp']}")

        scenarios.append(ScenarioEntry(
            name=name, df=df_long, gt=df_gt, _scenario=region, _params=params,
        ).to_dict())

    return scenarios


if __name__ == "__main__":
    # Quick manual test (needs the files in CAUSAL_RIVERS_DIR / the default path).
    for c in generate_causal_rivers_benchmark_scenarios(n_points=200):
        n_edges = int(c["gt"]["causal_gt"].sum())
        print(f"  {c['name']}: df={c['df'].shape}, "
              f"{n_edges}/{len(c['gt'])} real edges | {c['_params']['rivers']}")
