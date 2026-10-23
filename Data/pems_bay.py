"""
Data.pems_bay
===============================================================================
REAL BENCHMARK — PEMS-BAY (California Department of Transportation,
Performance Measurement System — Bay Area), pre-processed and made available
by Li, Yi, Shahabi & Liu 2018, "Diffusion Convolutional Recurrent Neural
Network: Data-Driven Traffic Forecasting" (ICLR 2018, "DCRNN") and widely
reused since then (Graph WaveNet, STGCN, ...) as the standard benchmark for
spatio-temporal time series on real traffic sensors.

325 average traffic speed sensors in the Bay Area (California), 5-minute
resolution, ~6 months (01/01/2017-06/30/2017, 52,116 points per sensor).

GROUND TRUTH: connectivity graph `adj_mx` published together with the
dataset by Li et al. 2018 — a Gaussian kernel over the REAL ROAD-NETWORK
DISTANCE (not straight-line distance) between each pair of sensors,
thresholded (>= adjacency_threshold becomes an edge). It is a graph BY
CONSTRUCTION from the physical topology of the road network (same spirit as
the TEP ground truth via P&ID — Data/tep.py), not estimated. Note that it is
a PHYSICAL PROXIMITY/CONNECTIVITY proxy, not a causal graph validated by
intervention — the reasonable reading is "sensors close together along the
same flow direction tend to have congestion propagated between them with a
delay" (upstream congestion -> downstream), which IS the regime this
pipeline was designed to detect (X_(t-tau) -> Y_t), but it should be
reported as a limitation (topology proxy, not causal ground truth validated
by intervention) if used in the paper — the same caution that applies to
any proximity-based ground-truth heuristic, except here the proxy is
PUBLISHED by the dataset's original authors, not invented by this project.

IMPORTANT: `adj_mx` is NOT symmetric (dist(i,j) on the road network can
differ from dist(j,i) because of one-way roads/flow direction) — that is why
the ground truth here is naturally DIRECTED, unlike a plain Euclidean
proximity heuristic.

SCALE: the full dataset (325 sensors x 52k points) is too large to run in
the pipeline as-is (designed for tens of entities x hundreds of points) —
this loader SUBSAMPLES by default: `n_sensors` sensors (the ones closest to
each other according to `adj_mx`, to keep a neighborhood with a ground truth
dense enough to be informative) x a window of `n_points` consecutive
timestamps starting at `offset`.

HOW THE DATA IS OBTAINED
-------------------------------------------------------------------------------
Automatically downloads (with on-disk caching) the public mirror on Zenodo
(Zenodo record 5724362, "PEMS-BAY and METR-LA in csv"):
    https://zenodo.org/records/5724362/files/PEMS-BAY.csv
    https://zenodo.org/records/5724362/files/adj_mx_PEMS-BAY.pkl
Requires internet on the first run (uses cache_dir afterwards; the CSV is
~85MB).
"""
from __future__ import annotations

import os
import pickle
import urllib.request as _urlreq_bench
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import array_to_pipeline_format_single_property, register

_CSV_URL = "https://zenodo.org/records/5724362/files/PEMS-BAY.csv"
_ADJ_URL = "https://zenodo.org/records/5724362/files/adj_mx_PEMS-BAY.pkl"
_BASE_DEFAULT = Path(os.environ.get("PEMS_BAY_DIR", "./external_data/pems_bay"))


def _download(cache_dir: Path, timeout: int = 300) -> tuple:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "PEMS-BAY.csv"
    adj_path = cache_dir / "adj_mx_PEMS-BAY.pkl"

    for path, url in [(csv_path, _CSV_URL), (adj_path, _ADJ_URL)]:
        if path.exists():
            continue
        try:
            with _urlreq_bench.urlopen(url, timeout=timeout) as resp:
                path.write_bytes(resp.read())
        except Exception as e:
            raise RuntimeError(
                f"Failed to download '{url}'. Check your internet "
                f"connection, or download it manually to '{path}'. "
                f"Original error: {e}"
            ) from e
    return csv_path, adj_path


def _select_neighborhood(adj_mx: np.ndarray, sensor_ids: list,
                          n_sensors: int, adjacency_threshold: float,
                          seed: int) -> list:
    """Picks a 'seed' sensor (deterministic seed) and expands through the
    highest-connectivity neighborhood (BFS guided by adj_mx weight) until
    reaching n_sensors -- this produces a sub-region with a ground truth
    dense enough to be informative, instead of n_sensors random sensors
    scattered across the whole Bay Area (which would have an almost entirely
    empty ground truth)."""
    n_total = len(sensor_ids)
    n_sensors = min(n_sensors, n_total)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n_total))

    selected = {start}
    frontier = {start}
    while len(selected) < n_sensors and frontier:
        candidates = []
        for i in frontier:
            weights = adj_mx[i]
            for j in np.argsort(-weights):
                if j not in selected and weights[j] >= adjacency_threshold:
                    candidates.append((weights[j], j))
        if not candidates:
            # no neighbors above the threshold -- fill in with the globally
            # closest sensors to the already-selected set, ignoring the
            # threshold
            remaining = [i for i in range(n_total) if i not in selected]
            if not remaining:
                break
            chosen = max(remaining, key=lambda i: max(adj_mx[i, s] for s in selected))
            selected.add(chosen)
            frontier = {chosen}
            continue
        candidates.sort(reverse=True)
        new_node = candidates[0][1]
        selected.add(new_node)
        frontier = {new_node}

    return sorted(selected)


@register("pems_bay")
def generate_pems_bay_benchmark_scenarios(
    n_sensors: int = 15,
    n_points: int = 400,
    offset: int = 0,
    adjacency_threshold: float = 0.1,
    seed: int = 0,
    cache_dir: Optional[str] = None,
    name: str = "real_pems_bay_traffic",
) -> dict:
    """
    Loads a sub-region of the PEMS-BAY benchmark (real traffic sensors, Bay
    Area) — see the module docstring for the source, the ground truth
    (road-network connectivity published by Li et al. 2018, DIRECTED) and
    the automatic download (with caching).

    Parameters
    ----------
    n_sensors            : how many sensors to include (subsampled by
                            connected neighborhood — see
                            _select_neighborhood).
    n_points             : size of the time window (consecutive points 5
                            minutes apart) starting at `offset`.
    offset               : index of the window's first timestamp.
    adjacency_threshold  : minimum adj_mx weight to count as a ground-truth
                            edge (same threshold used when selecting the
                            neighborhood).
    seed                 : seed for the neighborhood's starting sensor.
    cache_dir            : download/cache folder (default: env PEMS_BAY_DIR
                            or "./external_data/pems_bay").
    name                 : name of the generated scenario.
    """
    cache = Path(cache_dir) if cache_dir else _BASE_DEFAULT
    csv_path, adj_path = _download(cache)

    with open(adj_path, "rb") as f:
        sensor_ids, _sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

    neighborhood_idx = _select_neighborhood(
        adj_mx, sensor_ids, n_sensors, adjacency_threshold, seed,
    )
    sensors_used = [str(sensor_ids[i]) for i in neighborhood_idx]

    df_raw = pd.read_csv(csv_path, index_col=0, usecols=[0] + [
        1 + list(map(str, sensor_ids)).index(s) for s in sensors_used
    ])
    df_raw = df_raw[sensors_used]  # ensures the SAME order as neighborhood_idx

    if offset + n_points > len(df_raw):
        raise ValueError(
            f"offset={offset} + n_points={n_points} exceeds the total of "
            f"{len(df_raw)} timestamps available in '{csv_path}'."
        )
    window = df_raw.iloc[offset:offset + n_points]
    data = window.to_numpy(dtype=float)

    # NaN (missing readings, common in real traffic sensors) -- linearly
    # interpolated per sensor; endpoints with no valid neighbor use
    # fill-forward/fill-backward from the nearest valid value.
    if np.any(np.isnan(data)):
        window_interp = window.interpolate(method="linear", limit_direction="both")
        data = window_interp.to_numpy(dtype=float)

    sub_adj = adj_mx[np.ix_(neighborhood_idx, neighborhood_idx)]
    causal_pairs_idx = set()
    for a in range(len(neighborhood_idx)):
        for b in range(len(neighborhood_idx)):
            if a != b and sub_adj[a, b] >= adjacency_threshold:
                causal_pairs_idx.add((a, b))

    df_long, df_gt = array_to_pipeline_format_single_property(
        data, sensors_used, causal_pairs_idx,
        start_date=str(window.index[0]), freq="5min", property_name="SPEED",
    )

    params = {
        "source": "pems_bay",
        "reference": "Li, Yi, Shahabi & Liu 2018, ICLR (DCRNN); original "
                     "data from Caltrans PEMS (Bay Area)",
        "csv_file": str(csv_path),
        "adjacency_file": str(adj_path),
        "sensors_used": sensors_used,
        "n_sensors": len(sensors_used),
        "n_points": n_points,
        "offset": offset,
        "start_timestamp": str(window.index[0]),
        "end_timestamp": str(window.index[-1]),
        "adjacency_threshold": adjacency_threshold,
        "n_causal_links": len(causal_pairs_idx),
        "ground_truth_type": "road-network distance connectivity (Gaussian "
                              "kernel, published by Li et al. 2018) -- "
                              "DIRECTED physical-topology proxy, not causal "
                              "ground truth validated by intervention (see "
                              "the caveat in the module docstring)",
    }

    print(f"  [real-benchmark] {name} loaded — {len(sensors_used)} "
          f"sensors x {n_points} points (5min) | GT: {len(causal_pairs_idx)} "
          f"links (road-network connectivity, Li et al. 2018) | window "
          f"{window.index[0]}..{window.index[-1]}")

    return {"name": name, "df": df_long, "gt": df_gt, "_scenario": None,
            "_params": params}


if __name__ == "__main__":
    c = generate_pems_bay_benchmark_scenarios(n_sensors=10, n_points=200)
    n_edges = int(c["gt"]["causal_gt"].sum())
    print(f"  {c['name']}: df={c['df'].shape}, {n_edges}/{len(c['gt'])} real edges")
