"""
Data.metr_la
===============================================================================
REAL BENCHMARK — METR-LA (Los Angeles County highway loop detectors),
pre-processed and made available by Li, Yi, Shahabi & Liu 2018, "Diffusion
Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"
(ICLR 2018, "DCRNN") together with PEMS-BAY (see Data/pems_bay.py, of which
this loader is the counterpart: same source, same ground-truth construction,
same subsampling).

207 average traffic speed sensors on the highways of Los Angeles County,
5-minute resolution, 4 months (03/01/2012-06/27/2012, 34,272 points per
sensor).

GROUND TRUTH: connectivity graph `adj_mx` published together with the
dataset by Li et al. 2018 — a Gaussian kernel over the REAL ROAD-NETWORK
DISTANCE between each pair of sensors, thresholded (>= adjacency_threshold
becomes an edge). DIRECTED (adj_mx is not symmetric) and built from the
physical topology, not estimated; it is a PHYSICAL PROXIMITY/CONNECTIVITY
proxy, not a causal graph validated by intervention — the same caveat as
Data/pems_bay.py applies.

MISSING READINGS: unlike PEMS-BAY, METR-LA encodes a missing reading as a
speed of 0 (about 8% of the full dataset). Zeros are treated as missing and
linearly interpolated per sensor, like the NaN of PEMS-BAY.

SCALE: subsampled exactly like PEMS-BAY — `n_sensors` sensors from a
connected neighborhood of `adj_mx` x a window of `n_points` consecutive
timestamps starting at `offset`.

HOW THE DATA IS OBTAINED
-------------------------------------------------------------------------------
Automatically downloads (with on-disk caching) the public mirror on Zenodo
(Zenodo record 5724362, "PEMS-BAY and METR-LA in csv"):
    https://zenodo.org/records/5724362/files/METR-LA.csv
    https://zenodo.org/records/5724362/files/adj_mx_METR-LA.pkl
Requires internet on the first run (uses cache_dir afterwards; the CSV is
~72MB).
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
from .pems_bay import _select_neighborhood

_CSV_URL = "https://zenodo.org/records/5724362/files/METR-LA.csv"
_ADJ_URL = "https://zenodo.org/records/5724362/files/adj_mx_METR-LA.pkl"
_BASE_DEFAULT = Path(os.environ.get("METR_LA_DIR", "./external_data/metr_la"))


def _download(cache_dir: Path, timeout: int = 300) -> tuple:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "METR-LA.csv"
    adj_path = cache_dir / "adj_mx_METR-LA.pkl"

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


@register("metr_la")
def generate_metr_la_benchmark_scenarios(
    n_sensors: int = 15,
    n_points: int = 400,
    offset: int = 0,
    adjacency_threshold: float = 0.1,
    seed: int = 0,
    cache_dir: Optional[str] = None,
    name: str = "real_metr_la_traffic",
) -> dict:
    """
    Loads a sub-region of the METR-LA benchmark (real traffic sensors, Los
    Angeles County) — see the module docstring for the source, the ground
    truth (road-network connectivity published by Li et al. 2018, DIRECTED),
    the handling of missing readings and the automatic download.

    The parameters and their defaults are those of
    Data.pems_bay.generate_pems_bay_benchmark_scenarios.
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

    # Missing readings are stored as 0 (see the module docstring) -- linearly
    # interpolated per sensor; endpoints with no valid neighbor use
    # fill-forward/fill-backward from the nearest valid value.
    window = window.where(window != 0)
    missing_frac = float(window.isna().to_numpy().mean())
    all_missing = [s for s in sensors_used if window[s].isna().all()]
    if all_missing:
        raise ValueError(
            f"sensor(s) {all_missing} have no valid reading in the window "
            f"{window.index[0]}..{window.index[-1]}."
        )
    if missing_frac > 0:
        window = window.interpolate(method="linear", limit_direction="both")
    data = window.to_numpy(dtype=float)

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
        "source": "metr_la",
        "reference": "Li, Yi, Shahabi & Liu 2018, ICLR (DCRNN); original "
                     "data from LA Metro highway loop detectors "
                     "(Jagadish et al. 2014)",
        "csv_file": str(csv_path),
        "adjacency_file": str(adj_path),
        "sensors_used": sensors_used,
        "n_sensors": len(sensors_used),
        "n_points": n_points,
        "offset": offset,
        "start_timestamp": str(window.index[0]),
        "end_timestamp": str(window.index[-1]),
        "adjacency_threshold": adjacency_threshold,
        "missing_fraction_interpolated": round(missing_frac, 4),
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
          f"{window.index[0]}..{window.index[-1]} | "
          f"{missing_frac:.1%} missing readings interpolated")

    return {"name": name, "df": df_long, "gt": df_gt, "_scenario": None,
            "_params": params}


if __name__ == "__main__":
    c = generate_metr_la_benchmark_scenarios(n_sensors=10, n_points=200)
    n_edges = int(c["gt"]["causal_gt"].sum())
    print(f"  {c['name']}: df={c['df'].shape}, {n_edges}/{len(c['gt'])} real edges")
