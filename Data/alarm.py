"""
Data.alarm
===============================================================================
REAL BENCHMARK — ALARM NETWORK (Beinlich et al. 1989, "The ALARM Monitoring
System: A Case Study with Two Probabilistic Inference Techniques for Belief
Networks", 2nd European Conference on AI in Medicine). Classic causal
Bayesian network for ICU anesthesia monitoring — one of the most cited
benchmarks in causal structure learning (constraint-based and score-based:
PC, GES, NOTEARS, etc.), distributed via the official bnlearn repository
(Scutari, maintained since 2007).

GROUND TRUTH: DAG with 37 nodes / 46 edges, curated by domain experts
(medicine) from the causal knowledge of the modeled physiological process —
NOT estimated automatically. Same graph used as reference in dozens of
causal structure learning papers.

WARNING — SAME CAVEAT AS Data/sachs.py: the data here is SAMPLED i.i.d. from
the Bayesian network (forward sampling), NOT a real time series — there is
no genuine lag dynamics to be recovered. The order of the sampled rows is
used as pseudo-time only to fit the pipeline's common contract. Use with a
low max_lag (or 0); treat lag findings with caution — this benchmark is
better suited to validating the CONTEMPORANEOUS stage (structure/DML) of
the pipeline than lag detection.

Furthermore, ALARM's 37 nodes are all DISCRETE (2-3 states each, e.g.
TRUE/FALSE or LOW/NORMAL/HIGH) — this loader converts each state to an
integer code (0, 1, 2, ...) in the ORDER declared in the .bif file, and
treats that code as a numeric variable for the pipeline (the same
limitation any continuous causal discovery method has with ordinal/nominal
categorical variables — report as a limitation if used in the paper).

HOW THE DATA IS OBTAINED
-------------------------------------------------------------------------------
Automatically downloads (with on-disk cache) the alarm.bif.gz file from the
official bnlearn repository:
    https://www.bnlearn.com/bnrepository/alarm/alarm.bif.gz
Requires internet on first run (uses cache_dir afterward) and the `pgmpy`
package (already used in this project) to parse the .bif file and CPDs.

ABOUT THE SAMPLING (manual forward sampling, not via pgmpy)
-------------------------------------------------------------------------------
pgmpy 1.0.0 has a bug in BayesianModelSampling.forward_sample /
DiscreteBayesianNetwork.simulate (IndexError in sample_discrete_maps for
CPDs with >=2 parents). This loader implements its own direct topological
sampling instead — see _sample_bayesian_network() below.
"""
from __future__ import annotations

import gzip
import os
import urllib.request as _urlreq_bench
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from .base import array_to_pipeline_format_single_property, register

_ALARM_BIF_URL = "https://www.bnlearn.com/bnrepository/alarm/alarm.bif.gz"
_BASE_DEFAULT = Path(os.environ.get("ALARM_DIR", "./external_data/alarm"))


def _download_bif(cache_dir: Path, timeout: int = 30) -> Path:
    """Downloads (with cache) and decompresses alarm.bif.gz from the bnlearn repository."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    bif_path = cache_dir / "alarm.bif"
    if bif_path.exists():
        return bif_path

    gz_path = cache_dir / "alarm.bif.gz"
    try:
        with _urlreq_bench.urlopen(_ALARM_BIF_URL, timeout=timeout) as resp:
            gz_content = resp.read()
    except Exception as e:
        raise RuntimeError(
            f"Failed to download the ALARM network from {_ALARM_BIF_URL}. Check "
            f"your internet connection, or download it manually and save it to "
            f"'{bif_path}'. Original error: {e}"
        ) from e
    gz_path.write_bytes(gz_content)
    with gzip.open(gz_path, "rb") as f_in:
        bif_path.write_bytes(f_in.read())
    return bif_path


def _sample_bayesian_network(model, n: int, seed: int) -> pd.DataFrame:
    """Manual topological forward sampling (see the NOTE in the module docstring
    about the pgmpy 1.0.0 bug this works around). Returns a DataFrame with one
    column per node, values = the sampled state's INDEX (0, 1, 2, ...), in the
    same order as cpd.state_names[node]."""
    order = list(nx.topological_sort(model))
    cpds = {cpd.variable: cpd for cpd in model.get_cpds()}
    rng = np.random.default_rng(seed)

    samples: dict = {}
    for node in order:
        cpd = cpds[node]
        parents = cpd.variables[1:]
        vals = cpd.values  # shape (n_node_states, parent1_card, parent2_card, ...)
        n_states = vals.shape[0]
        output = np.empty(n, dtype=np.int64)
        if not parents:
            probs = vals / vals.sum()
            output[:] = rng.choice(n_states, size=n, p=probs)
        else:
            parent_idx = np.stack([samples[p] for p in parents], axis=1)
            for i in range(n):
                probs = vals[(slice(None),) + tuple(parent_idx[i])]
                probs = probs / probs.sum()
                output[i] = rng.choice(n_states, p=probs)
        samples[node] = output

    return pd.DataFrame({node: samples[node] for node in model.nodes()})


@register("alarm")
def load_alarm_benchmark(
    n_samples: int = 500,
    seed: int = 0,
    cache_dir: Optional[str] = None,
    name: str = "real_alarm_icu",
) -> dict:
    """
    Loads the ALARM benchmark (Beinlich et al. 1989) — a causal Bayesian
    network for ICU anesthesia monitoring, 37 nodes / 46 edges (ground truth
    curated by experts, via the bnlearn repository). Downloads the .bif file
    automatically (with cache) and generates `n_samples` i.i.d. rows via its
    own forward sampling (see the module docstring).

    IMPORTANT: NO real time axis (same caveat as Data/sachs.py) — the order
    of the sampled rows is used as pseudo-time only to fit the pipeline
    format. Variables are categorical, converted to integer code
    (0, 1, 2, ...).

    Parameters
    ----------
    n_samples : number of i.i.d. rows to sample from the network (default 500).
    seed      : RNG seed for the sampling.
    cache_dir : folder where the .bif file is downloaded/cached (default: env
                ALARM_DIR or "./external_data/alarm").
    name      : name of the generated scenario.
    """
    try:
        from pgmpy.readwrite import BIFReader
    except ImportError as e:
        raise ImportError("pgmpy not found. Install it with: pip install pgmpy") from e

    cache = Path(cache_dir) if cache_dir else _BASE_DEFAULT
    bif_path = _download_bif(cache)

    model = BIFReader(str(bif_path)).get_model()
    nodes = list(model.nodes())
    gt_edges = {(nodes.index(a), nodes.index(b)) for a, b in model.edges()}

    df_states = _sample_bayesian_network(model, n=n_samples, seed=seed)
    df_states = df_states[nodes]  # stable column order == order of `nodes`
    data = df_states.to_numpy(dtype=float)

    df_long, df_gt = array_to_pipeline_format_single_property(
        data, nodes, gt_edges,
        start_date="2015-01-01", freq="min", property_name="STATE",
    )

    cpds = {cpd.variable: cpd for cpd in model.get_cpds()}
    params = {
        "source": "alarm_bnlearn",
        "reference": "Beinlich et al. 1989, 2nd European Conference on AI "
                     "in Medicine; network distributed via bnlearn (Scutari)",
        "bif_url": _ALARM_BIF_URL,
        "n_nodes": len(nodes),
        "n_gt_edges": len(gt_edges),
        "n_samples": n_samples,
        "sampling_seed": seed,
        "ground_truth_type": "DAG curated by domain experts "
                              "(medicine), not estimated",
        "states_per_node": {node: cpds[node].state_names[node] for node in nodes},
        "caveat": "i.i.d. data via forward sampling of the Bayesian network, NO "
                  "real time axis; categorical variables converted "
                  "to integer code (0,1,2,...)",
    }

    print(f"  [real-benchmark] {name} loaded — {len(nodes)} nodes (ALARM) x "
          f"{n_samples} samples | GT: {len(gt_edges)} edges "
          f"(expert-curated, bnlearn) | WARNING: no real time axis, see "
          f"docstring of Data/alarm.py")

    return {"name": name, "df": df_long, "gt": df_gt, "_scenario": None,
            "_params": params}


if __name__ == "__main__":
    c = load_alarm_benchmark(n_samples=200, seed=0)
    n_edges = int(c["gt"]["causal_gt"].sum())
    print(f"  {c['name']}: df={c['df'].shape}, {n_edges}/{len(c['gt'])} real edges")
